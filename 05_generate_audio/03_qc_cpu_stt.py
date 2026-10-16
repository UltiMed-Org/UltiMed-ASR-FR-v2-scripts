#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnx-asr[cpu]>=0.6",
#   "soundfile>=0.12",
#   "jiwer>=3.0",
#   "requests>=2.31",
#   "numpy",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""Tail a running stage-05 generation and STT-check a SAMPLE of its clips on the CPU.

Why this exists
---------------
Stage 06 (`06_hotfixes/driver.sh`) already scores every clip against its label with a
reference STT model, but it runs AFTER the whole corpus is on disk and it needs the GPU,
so a systematic defect introduced on day 1 of an 8-day run is only discovered on day 9.
Meanwhile the generator leaves 4-5 of this host's 12 threads idle (measured under the
live v2 run: 35-46% CPU idle, the GPU being the bottleneck), and the UltiMed Parakeet
ONNX export runs on CPU. This script spends those idle threads on an early warning.

Why a SAMPLE and not every clip, which is the whole design constraint
---------------------------------------------------------------------
Measured on this host on 2026-10-14, on 12 real v2 dictionary clips (226.6 s of audio),
WHILE the generator was running, with the int8 UltiMed encoder:

    threads=4 workers=1   7.7x realtime   2.44 s per clip
    threads=6 workers=1   6.7x realtime           (worse: memory-bandwidth bound)
    threads=3 workers=2   9.4x realtime   2.01 s per clip   (best)

Stage 05 writes audio at ~28.9x realtime (4,900 clips/h x ~21 s each), so the best CPU
configuration covers about a THIRD of the stream: checking every clip would fall ~1,900
hours of audio behind over the remaining 485k clips, and the benchmark runs themselves
cost 7-10% of generation throughput while they ran (the 500-clip interval that contained
them did 4,380 clips/h against ~4,860 for the three before it). So this script checks:

  1. every clip that needed a RETRY, i.e. every clip with a row in
     `cfg_alpha_sidecar.jsonl` (~0.3% of the stream). Highest value per second of CPU:
     those takes were rejected at least once for their duration and then re-drawn one
     rung down the cfg_alpha ladder, so they are the only clips in the corpus that
     nobody, and nothing, has looked at twice.
  2. a deterministic ~5% SAMPLE of everything else, which is enough to see per-voice CER
     drift (a voice that starts dropping text, a text class that makes it ramble) within
     hours instead of at stage 06.

Together that is ~5.3% of the stream, i.e. ~1.5x realtime of STT, which one worker at 3
threads delivers with ~80% of its duty cycle to spare, and costs the generator ~1-2%.
Hence the defaults: `--workers 1 --threads 3 --nice 10`. Raising them buys nothing (the
queue is already empty most of the time) and is taken straight out of the GPU's feed.

What this is NOT
----------------
The UltiMed checkpoint was fine-tuned on **v1 of this very dataset**, so it is circular
as a quality gate: it has been taught this corpus's vocabulary, including whatever v1 got
wrong, and it will happily agree with audio a neutral listener would reject. Treat a CER
here as a GROSS-error detector (text dropped, syllables looped, the wrong text spoken,
the clip truncated) and not as evidence of quality. For an impartial read, point
`--model-dir` at the base export (`parakeet-tdt-0.6b-v3-redux-onnx`) or score with
Whisper via stage 06, which talks to an STT endpoint over HTTP.

Output
------
One JSON line per checked clip in `qc_stt_sidecar.jsonl`, beside the audio and beside
`cfg_alpha_sidecar.jsonl` (`--out-name` to rename, the file is append-only and a restart
resumes from it). A clip over either gate is also logged at WARNING with its transcript,
which is the line to grep for. The row carries `reason` ("retry" or "sample"), `voice`,
`cfg_alpha`, `cer`, `wer`, `tail_cer`, `window_cer`, `bad_run_chars`, `duration` and
`stt_s`, so the same file answers both "which clips are bad" and "is one voice
drifting". The last two are the length-independent pair: they see a defect in the middle
of a long clip, which the two averages cannot (see the gate constants), and they are
stored on clean rows too so the gates can be re-swept without transcribing again.

Usage
-----
    # tail the live run (all five stages; a stage is picked up when it starts producing)
    ./03_qc_cpu_stt.py

    # also work through the clips already on disk, then keep tailing
    ./03_qc_cpu_stt.py --backfill

    # one-shot, manual: check what is on disk now for one stage and exit
    ./03_qc_cpu_stt.py --stage 01_dictionnary --backfill --once

    # impartial second opinion on the retried clips only, no sample
    ./03_qc_cpu_stt.py --sample-percent 0 \
        --model-dir ../../parakeet-tdt-0.6b-v3-redux-onnx

Stopping with Ctrl-C is safe at any moment: the clip in flight finishes, the queue is
dropped, and the next run re-reads the output file and picks up where this one stopped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import signal
import statistics
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import click
import soundfile as sf
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent


def _load_module(path: Path, name: str):
    """Import a sibling script whose filename starts with a digit.

    `02_audit_durations.py` and `06_hotfixes/01_compute_stt.py` are not importable by
    name, and every other script in this repo reaches them with exactly this
    `spec_from_file_location` trick, so this follows the convention rather than adding
    a shared helper module no caller would find. Importing runs the module body only;
    their `main()` is click-decorated and guarded by `if __name__ == "__main__"`.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The scoring half is stage 06's, not a second implementation: `normalize_for_scoring`
# encodes years of decisions about what counts as a difference in this corpus (numbers
# read aloud, roman numerals, plural endings, accents), and a private copy here would
# drift and then disagree with the stage that actually gates the dataset.
_stt = _load_module(ROOT / "06_hotfixes" / "01_compute_stt.py", "compute_stt_mod")
# ...and the stage table is the duration auditor's, which already justifies its copy of
# the runner's stage -> voice-mode pairing. Getting the mode wrong would judge a clip
# against another voice's reader, so there must be exactly one copy to fix.
_audit = _load_module(_HERE / "02_audit_durations.py", "stage05_audit")

sys.path.insert(0, str(ROOT / "utils"))
from cfg_sidecar import SIDECAR_FILENAME  # noqa: E402
from nemo_manifest import parse_clip_name  # noqa: E402
from voice_assignment import group_field, voice_for_row  # noqa: E402

#: Where the per-clip QC rows land, beside the audio and the cfg_alpha sidecar.
QC_FILENAME = "qc_stt_sidecar.jsonl"

#: Salt for the sample draw. It exists so the sample is independent of the VOICE draw:
#: `voice_assignment.voice_for_key` also hashes with blake2b, and a sample correlated
#: with the voice assignment is exactly the one thing that would make the per-voice CER
#: table in the summary lie. Changing this string re-draws the whole sample, so don't.
QC_SAMPLE_SALT = "ultimed-v2-qc-sample"

#: Resolution of the sample draw (a percentage is resolved to 1/10,000 of a percent).
SAMPLE_RESOLUTION = 1_000_000

#: The int8 ONNX export used by default. A sibling of this repo, so the path carries no
#: username. It is the fine-tuned checkpoint: read the "What this is NOT" note above.
DEFAULT_MODEL_DIR = ROOT.parent / "parakeet-tdt-0.6b-v3-UltiMed-onnx"

#: Gates, same numbers stage 06 uses (`--cer-threshold` 0.08, `--tail-cer-threshold`
#: 0.12, the latter looser because the tail window cuts mid-sentence). Shared values so
#: "over the gate" means the same thing in both stages, and the early warning predicts
#: what the real QC loop will flag rather than inventing a second standard.
DEFAULT_CER_GATE = 0.08
DEFAULT_TAIL_CER_GATE = 0.12

#: The WINDOWED gates (`window_metrics` in stage 06), which exist because both gates
#: above average over the whole clip and so go blind on a long one: five words lost in
#: the middle of a 48 s PARHAF chunk is CER 0.044, half the gate, and the tail window
#: only covers the clip's last 30 s. That is the v1.3 failure mode (14 clips retired for
#: a skipped phrase, none caught by the v1 QC), and 9.1% of the v2 dictionary is long
#: enough to hide it.
#:
#: Both numbers are measured, on the 19,958 v1 clips of `06_hotfixes/improved/full.stt.jsonl`
#: scored against their labels (2026-10-16). That set is the right calibration: v1 shipped,
#: so nearly every clip is good, and its transcripts come from whisper-1, a model that is
#: WORSE on this vocabulary than the one here, which makes the false-positive rates below
#: an upper bound.
#:   * `bad_run_chars` p50 1, p90 2, p99 5, p99.9 9, max 26. Five French words are ~30
#:     normalized characters, so the gate sits above every clean clip in 20k and still
#:     catches the defect asked about: 0 false positives at 30, and the worst offenders
#:     at 25-26 are not defects at all but whisper writing "mUI/L" where the label spells
#:     "milliunites internationales".
#:   * `window_cer` p99 0.061, p99.9 0.103, max 0.211 -> 1 clip in 20k at 0.15 (0.005%).
#:     Looser than the tail gate on purpose: the window is 213 characters against the
#:     tail's 498, so it cuts mid-sentence more often and is noisier.
#: A drop SHORTER than ~4 words stays under both, which is the honest limit of the
#: method. Both values are stored on every row, so the thresholds can be re-swept
#: offline over the sidecar without transcribing anything again.
DEFAULT_WINDOW_CER_GATE = 0.15
DEFAULT_BAD_RUN_GATE = 30


#: Defect shapes a flagged clip can have, cheapest-to-act-on first. See
#: `classify_defect` for what separates them and why the split is worth recording.
DEFECT_SHAPES = ("loop", "overrun", "truncated", "skipped", "wrong_words")

#: Longest contiguous run of missing reference characters that makes a clip a `skipped`
#: rather than a `wrong_words`. Same number as `DEFAULT_BAD_RUN_GATE` and for the same
#: reason, but kept separate: that one decides whether to FLAG a clip, this one only
#: names the shape of a clip already flagged, so a stricter flag gate must not silently
#: empty the shape.
_SKIP_RUN_CHARS = 30

#: Shortest missing run worth printing a line about when a clip is flagged. ~0.7 s of
#: speech: below it the "missing" span is a word ending or a unit abbreviation, which
#: says nothing, while the window excerpt above it already shows the damage.
_MISSING_SHOW_CHARS = 10


def log_window_evidence(score, *, log=logger.warning, indent: str = "  ") -> None:
    """Print the TEXT behind a window score: what the label says over the worst stretch
    of the clip, what the recognizer heard over that same stretch, and the longest span
    that went missing.

    This is the line to read when deciding whether a flag is a real defect or a
    recognizer artefact, which is why it prints both sides rather than a number. Both
    texts are the NORMALIZED forms, because those are what the CER scored: showing the
    raw label beside a normalized hypothesis would invite chasing differences the metric
    never saw (digits spelled out, punctuation dropped).

    Nothing is stored in the sidecar row for this: the row already carries the full
    `reference` and `transcript`, so `window_score` reproduces every line below from
    disk at any time (that is what `--defect-mix` does).

    Parameters
    ----------
    score : stage 06's WindowScore
        From `_stt.window_score(reference, transcript)`.
    log : callable
        Logger method to print with, so the live loop can warn and a report can info.
    indent : str
        Prefix for each line, to nest it under its clip's own line.
    """
    if score.window_cer is None:
        return
    log("{}expected (worst {:.0f}s window, normalized): {}", indent,
        _stt.WINDOW_SECONDS, score.ref_window)
    log("{}heard    (same stretch, {} chars): {}", indent, len(score.hyp_window),
        score.hyp_window)
    if (score.bad_run_chars or 0) >= _MISSING_SHOW_CHARS:
        log("{}missing {} chars ({:.1f}s), heard {!r} in its place: {}", indent,
            score.bad_run_chars,
            score.bad_run_chars / _stt.NORM_CHARS_PER_SECOND,
            score.missing_heard, score.missing_context)

#: Immediately-repeating n-gram sizes the loop test looks for. A derailed take
#: repeats whole words, so word n-grams find it; 2 to 5 covers "de la" through a
#: short clause, and anything longer is already caught by the shorter windows.
_LOOP_NGRAMS = (2, 3, 4, 5)

#: How many times an n-gram must occur before it is a loop rather than ordinary
#: French repetition, and how far it must beat the LABEL's own repetition. Both
#: are needed: clinical passages legitimately repeat "de la" or a drug name, so an
#: absolute count alone flags healthy clips.
_LOOP_MIN_COUNT = 3
_LOOP_MARGIN = 1

#: Long-clause arm of the same test, and the much looser count is the point: TWO
#: occurrences are enough once the repeated span is 6 or more words, because correct
#: French prose does not say a whole clause over verbatim unless the label does, and
#: the label's own repetition is the baseline this is measured against (see
#: `_added_clause_repeat`; the short arm needs 3 because its spans are as generic as
#: "de la"). Measured on the 2026-10-16 dictionary sidecar: this arm fires on 151 of
#: 250 flagged clips (60%) and on 0 of 2126 CER-passing ones, so a repeated clause is
#: a more specific signal than the CER gate it explains. Sizes are sparse
#: because a 12-word repeat contains a 6-word one; 8 and 10 only add the case where a
#: clause's 6-word windows all straddle an edit.
_LOOP_LONG_NGRAMS = (6, 8, 10)
_LOOP_LONG_MIN_COUNT = 2

#: Transcript-to-label word-count ratios that mean the take did not end where the
#: text did. Wide on purpose: STT word counts drift a few percent on their own.
_SHORT_RATIO = 0.75
_LONG_RATIO = 1.30


def classify_defect(reference: str, transcript: str, *,
                    bad_run_chars: int | None = None) -> str:
    """Name the SHAPE of a flagged clip's failure, from the two texts alone.

    The CER gate answers "is this clip wrong"; this answers "wrong how", which is a
    different and cheaper question, and the two disagree about what is worth building
    next. Five shapes, and only the first two leave a trace in the waveform:

    ``loop``
        A span repeats more often than the label says it: the model derailed into
        saying the same thing over again. Audible as the repetitive mumbling, and the
        one shape a self-similarity detector can find without transcribing. Two arms,
        because loops come in two lengths: a short n-gram (2 to 5 words) has to occur
        `_LOOP_MIN_COUNT` times, while a clause of 6 or more words counts from the
        second occurrence, the label's own count of that same clause subtracted.
    ``overrun``
        Materially more words than the label and no repeated span to explain them: the
        model kept talking, inventing new material rather than repeating. Rare once
        the long-clause arm above has taken its share, since a take that will not stop
        usually stops making up new text and starts repeating.
    ``truncated``
        Materially fewer words: the take stopped early.
    ``skipped``
        The right AMOUNT of speech, but a long contiguous span of the label is missing
        from it: the voice jumped over a phrase and carried on. Invisible to the
        whole-clip CER on a long clip, which is why `window_metrics` exists; separated
        from `wrong_words` because the repair differs (the text is fine, the take is
        not) and because it is the defect v1.3 had to retire clips for.

        `06_hotfixes/04_flag_asr_defects.py` detects the same failure for the
        exhaustive offline pass, and the two are deliberately NOT one function: that
        one works word-level and only fires when a SECOND recognizer agrees the phrase
        is absent, because its output drops clips from the release, while this one is a
        single-recognizer character metric that only asks for a listen. See that file's
        header for what would have to be measured before merging them.
    ``wrong_words``
        The same amount of speech, at a normal rate, saying something else. This has
        no acoustic signature by construction, so STT is the only thing that can see
        it. Read the measured share of this shape as a ceiling on what a cheap
        acoustic pre-filter must leave to STT, and note that the ceiling moved a long
        way once doubled clauses stopped landing here (below).

    The mix is the whole reason this field exists: it bounds how much of the QC catch a
    cheap acoustic pre-filter could ever cover. It has to be watched over time rather
    than measured once, and the history is the argument for that. On the first 51
    flagged dictionary clips (2026-10-15) it read 73% wrong_words / 16% loop / 8%
    overrun / 4% truncated, and the conclusion drawn from it was that most defects have
    no acoustic signature. That conclusion was an artefact of this function: printing
    the window texts (`log_window_evidence`) showed clause-length loops filed as
    wrong_words, because `_LOOP_MIN_COUNT` wanted three occurrences and a derailed take
    most often says a clause exactly twice. With the long-clause arm added, 250 flagged
    clips read **63% loop, 36% wrong_words, 1% overrun** overall and **90% loop, 7%
    wrong_words** among the severe ones (CER >= 0.25), against 10% loop / 79%
    wrong_words as the old rule labelled the same rows. So a self-similarity pre-filter
    could in principle see most of the catch, and nearly all of the severe catch,
    rather than a sixth of it. The mix still points at single voices rather than at the
    corpus, which is the other reason to keep watching it.

    Parameters
    ----------
    reference
        The label the clip was generated from.
    transcript
        What the STT model heard.
    bad_run_chars
        The clip's longest missing run, when the caller has already computed it (the
        live loop has, from the same alignment it gated on). Recomputed here when not
        given, so a re-read of old rows still classifies.

    Returns
    -------
    str
        One of `DEFECT_SHAPES`.

    Examples
    --------
    >>> classify_defect("le patient tousse", "le patient tousse tousse tousse tousse")
    'loop'
    >>> classify_defect("le patient tousse", "le patient")
    'truncated'
    >>> classify_defect("le patient tousse", "la patiente touche")
    'wrong_words'
    >>> ref = "le patient tousse " * 8 + "il presente une dyspnee aux efforts moderes"
    >>> classify_defect(ref, ref.replace("il presente une dyspnee aux efforts ", ""))
    'skipped'
    """
    # The same normaliser the CER gate scores with, so a shape is judged on exactly
    # the token stream that produced the number that flagged the clip.
    ref = _stt.normalize_for_scoring(reference or "").split()
    hyp = _stt.normalize_for_scoring(transcript or "").split()
    if not hyp:
        return "truncated"
    ratio = len(hyp) / max(len(ref), 1)
    hyp_rep, ref_rep = _max_ngram_count(hyp), _max_ngram_count(ref)
    if hyp_rep >= _LOOP_MIN_COUNT and hyp_rep > ref_rep + _LOOP_MARGIN:
        return "loop"
    # Before the length ratios on purpose: a take that loops usually also runs long, and
    # of the two names the loop is the cause and `overrun` the symptom. This is what
    # moved 26 of the 28 stored `overrun` rows to `loop` on 2026-10-16.
    if _added_clause_repeat(ref, hyp) is not None:
        return "loop"
    if ratio < _SHORT_RATIO:
        return "truncated"
    if ratio > _LONG_RATIO:
        return "overrun"
    # Last, because a loop or a truncation also leaves missing spans behind, and those
    # names are the more actionable ones. Reached only by a clip of the right length,
    # where a long missing run can mean nothing else.
    bad_run = (bad_run_chars if bad_run_chars is not None
               else _stt.window_metrics(reference, transcript)[1])
    if bad_run is not None and bad_run >= _SKIP_RUN_CHARS:
        return "skipped"
    return "wrong_words"


def _max_ngram_count(tokens: list[str]) -> int:
    """How often the most repeated word n-gram occurs, over `_LOOP_NGRAMS` sizes.

    Returns the plain count rather than a rate: a loop is three or more occurrences
    whether the clip is 30 s or 70 s, while a rate would shrink as the clip grows and
    hide exactly the long derailments that matter most.
    """
    best = 0
    for n in _LOOP_NGRAMS:
        if len(tokens) < n:
            break
        counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        best = max(best, counts.most_common(1)[0][1])
    return best


def _added_clause_repeat(reference: list[str],
                         hypothesis: list[str]) -> tuple[int, str] | None:
    """The long clause the hypothesis repeats beyond the label's own repetitiveness.

    A clause is only a loop relative to its label, so the test is per n-gram size: the
    hypothesis's most repeated span of that size must occur at least
    `_LOOP_LONG_MIN_COUNT` times AND strictly more often than ANY span of that size
    occurs in the label. The label's own maximum is the baseline rather than its count
    of that same clause, and the difference is not academic: when a label repeats a
    clause and the take misreads a word the same way in both copies, the misread clause
    is a span the label never contains, so a same-clause comparison would call that a
    loop (it did; see `tests/test_qc_cpu_stt.py`). The baseline form cannot, since the
    take says it no more often than the label says its own.

    What that costs: a take that loops clause B twice inside a label that legitimately
    repeats clause A twice is missed. Accepted knowingly, because labels that repeat a
    6+ word clause at all are 6 of 2295 rows measured (0.26%), and a wrong `loop` on a
    dosage list would be the more misleading error of the two.

    Parameters
    ----------
    reference
        Normalized label tokens, as `normalize_for_scoring().split()` returns them.
    hypothesis
        Normalized transcript tokens.

    Returns
    -------
    tuple of (int, str) or None
        Occurrences in the hypothesis and the clause itself, for the clause with the
        most occurrences beyond the label's own count; None when nothing qualifies.

    Examples
    --------
    >>> label = "la patiente recoit 2 grammes par jours pendant 10 jours".split()
    >>> _added_clause_repeat(label, label + label[2:])[0]
    2
    >>> _added_clause_repeat(label, label) is None     # said once, heard once
    True
    >>> _added_clause_repeat(label + label, label + label) is None   # label repeats too
    True
    """
    best: tuple[int, str] | None = None
    for n in _LOOP_LONG_NGRAMS:
        if len(hypothesis) < n:
            break
        ref_counts = Counter(tuple(reference[i:i + n])
                             for i in range(len(reference) - n + 1))
        floor = max(_LOOP_LONG_MIN_COUNT - 1, max(ref_counts.values(), default=0))
        for gram, count in Counter(tuple(hypothesis[i:i + n])
                                   for i in range(len(hypothesis) - n + 1)).items():
            if count > floor and (best is None or count > best[0]):
                best = (count, " ".join(gram))
    return best


def is_sampled(stem: str, *, percent: float, salt: str = QC_SAMPLE_SALT) -> bool:
    """Whether a clip belongs to the deterministic sample.

    Parameters
    ----------
    stem
        Clip stem, e.g. ``004534_0006_apophyse_unciforme_des_vertebres_cervicales``.
    percent
        Share of clips to draw, in percent. ``0`` draws none, ``100`` draws all.
    salt
        See `QC_SAMPLE_SALT`.

    Returns
    -------
    bool
        Membership, which is a pure function of the stem: a restart, a re-run or a
        second process all pick the SAME clips, so the sample stays a sample instead of
        becoming a biased re-draw of whatever was on disk at restart time.

    Examples
    --------
    >>> stems = [f"{i:06d}_0001_terme" for i in range(20000)]
    >>> drawn = [s for s in stems if is_sampled(s, percent=5.0)]
    >>> 850 <= len(drawn) <= 1150          # ~5% of 20,000, binomial noise aside
    True
    >>> is_sampled(stems[0], percent=5.0) == is_sampled(stems[0], percent=5.0)
    True
    """
    if percent <= 0.0:
        return False
    if percent >= 100.0:
        return True
    digest = hashlib.blake2b(f"{salt}:{stem}".encode(), digest_size=8).digest()
    cut = round(percent / 100.0 * SAMPLE_RESOLUTION)
    return int.from_bytes(digest, "big") % SAMPLE_RESOLUTION < cut


def load_cpu_model(model_dir: Path, *, quantization: str, threads: int):
    """Load an onnx-asr Parakeet model from a LOCAL per-precision export directory.

    Parameters
    ----------
    model_dir
        A `parakeet-tdt-0.6b-v3-*-onnx` checkout.
    quantization
        Subfolder/precision to load, e.g. ``"int8"``.
    threads
        onnxruntime intra-op threads. Pinned deliberately: left alone, onnxruntime
        takes every core and starves the process feeding the GPU, which is the one
        thing this script must not do.

    Notes
    -----
    The `relocate` shim is what the model repo's own README prescribes: onnx-asr globs
    the repository ROOT for its model files, while these exports keep them one folder
    deeper (`int8/encoder-model.int8.onnx`), so the loader finds nothing without it.
    """
    import onnx_asr
    import onnxruntime as ort
    from onnx_asr.loader import create_asr_resolver

    def relocate(pattern: str) -> str:
        basename = pattern.replace("?", ".")
        for folder in ("int8/", "w4a8/", "fp16/", "fp32/", ""):
            if (model_dir / folder / basename).is_file():
                return folder + pattern
        return pattern

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1

    name = "nemo-parakeet-tdt-0.6b-v3"
    model_type = create_asr_resolver(name, str(model_dir)).model_type
    original = model_type._get_model_files
    model_type._get_model_files = staticmethod(
        lambda quantization=None: {k: relocate(v)
                                   for k, v in original(quantization).items()})
    try:
        return onnx_asr.load_model(name, path=str(model_dir),
                                   quantization=quantization, sess_options=opts)
    finally:
        model_type._get_model_files = staticmethod(original)


@dataclass
class Job:
    """One clip to check."""

    stage: str
    stem: str
    path: Path
    reason: str                     # "retry" or "sample"
    key: str                        # dedup key, see StageWatch.handled
    cfg_alpha: float | None = None
    src_ts: str | None = None       # the cfg_alpha sidecar row's timestamp, if any


@dataclass
class StageWatch:
    """Everything needed to follow one stage directory.

    The manifest is NOT held in memory as text: 602,885 rows of
    `asr_training_source` would be a few hundred MB for the ~5% of them this script
    ever reads. Instead `offsets` maps the clip's identity to a byte offset and the row
    is seeked and parsed on demand, which is one disk read per checked clip.
    """

    name: str
    mode: str
    jsonl: Path
    audio: Path
    offsets: dict[int, int] | None = None   # term*10_000 + variant -> byte offset
    voice_field: str | None = None          # the grouping field, decided once
    handle: object | None = None            # open binary handle on the manifest
    lock: threading.Lock = field(default_factory=threading.Lock)
    seen: set[str] = field(default_factory=set)      # every clip name listed so far
    handled: set[str] = field(default_factory=set)   # dedup keys already queued/done
    stems: set[str] = field(default_factory=set)     # stems queued under ANY reason
    sidecar_offset: int = 0
    ready: bool = False


def row_key(term_index: int, variant_index: int) -> int:
    """Pack a clip's identity into one int key.

    The variant index is zero-padded to 4 digits in every clip name, so 10,000 is a
    safe radix; a row above it could not be named by stage 05 in the first place.
    """
    return term_index * 10_000 + variant_index


def ensure_index(watch: StageWatch) -> None:
    """Build `watch.offsets` and decide `watch.voice_field`, then keep the manifest open.

    Costs one pass over the stage's `generated_dataset.jsonl` (~8 s for the 534k-row
    dictionary stage) and is done LAZILY, on the first clip of that stage that is
    actually checked, so following all five stages costs nothing until they produce.
    Called from the worker threads, hence the lock and the second `ready` test.
    """
    if watch.ready:
        return
    with watch.lock:
        if watch.ready:
            return
        _index_manifest(watch)


def _index_manifest(watch: StageWatch) -> None:
    """The index pass itself. Call `ensure_index`, which guards it."""
    offsets: dict[int, int] = {}
    field_name: str | None = None
    t0 = time.monotonic()
    with watch.jsonl.open("rb") as fh:
        offset = 0
        for raw in fh:
            length = len(raw)
            if raw.strip():
                try:
                    row = json.loads(raw)
                    key = row_key(int(row["term_index"]), int(row["variant_index"]))
                except (ValueError, KeyError, TypeError):
                    offset += length
                    continue
                offsets[key] = offset
                # The grouping field is decided ONCE from the first valid row, exactly
                # as the generator does it, because a later row missing `source_id`
                # would otherwise regroup the rest of the file onto `term_index` and
                # hand back the wrong voice for every clip after it.
                if field_name is None:
                    field_name = group_field(row)
            offset += length
    watch.offsets = offsets
    watch.voice_field = field_name
    watch.handle = watch.jsonl.open("rb")
    watch.ready = True
    logger.info("{}: indexed {} manifest rows in {:.1f}s (voice mode {}, field {})",
                watch.name, len(offsets), time.monotonic() - t0, watch.mode, field_name)


def manifest_row(watch: StageWatch, stem: str) -> dict | None:
    """The manifest row behind a clip stem, or None when there is no such row.

    `ensure_index` must have returned first: it holds the same lock while indexing, and
    the two are called in sequence from the worker, never nested.
    """
    parsed = parse_clip_name(f"{stem}.flac")
    if parsed is None or watch.offsets is None or watch.handle is None:
        return None
    offset = watch.offsets.get(row_key(parsed[0], parsed[1]))
    if offset is None:
        return None
    with watch.lock:                     # one shared handle, so seek+read is critical
        watch.handle.seek(offset)
        raw = watch.handle.readline()
    try:
        return json.loads(raw)
    except ValueError:
        return None


@dataclass
class Stats:
    """Running totals, for the periodic summary."""

    checked: int = 0
    by_reason: Counter = field(default_factory=Counter)
    over_gate: int = 0
    dropped: int = 0
    failed: int = 0
    audio_s: float = 0.0
    stt_s: float = 0.0
    started: float = field(default_factory=time.monotonic)
    cer_by_voice: dict[str, list[float]] = field(default_factory=dict)
    over_by_voice: Counter = field(default_factory=Counter)
    defects: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, *, reason: str, voice: str | None, cer: float | None,
            over: bool, duration: float, stt_s: float,
            defect: str | None = None) -> int:
        """Fold one finished check in, and return the new total.

        Locked because every worker writes here and ``+=`` on an attribute is a
        read-modify-write, not an atomic operation: without it the totals quietly
        drift low as soon as --workers is above 1. The returned count is what the
        caller uses to decide whether this check lands on a summary boundary, so that
        decision is made inside the lock too and no boundary is missed or doubled.
        """
        with self.lock:
            self.checked += 1
            self.by_reason[reason] += 1
            self.audio_s += duration
            self.stt_s += stt_s
            if over:
                self.over_gate += 1
                self.over_by_voice[voice or "?"] += 1
            if defect:
                self.defects[defect] += 1
            if cer is not None:
                self.cer_by_voice.setdefault(voice or "?", []).append(cer)
            return self.checked

    def bump(self, attr: str) -> None:
        """Increment one of the error counters under the same lock."""
        with self.lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def lines(self, *, workers: int) -> list[str]:
        """The summary, as log lines: the cost of the QC first, then per-voice CER."""
        with self.lock:
            return self._lines(workers=workers)

    def _lines(self, *, workers: int) -> list[str]:
        """Render the summary. Call `lines`, which holds the lock."""
        wall = max(time.monotonic() - self.started, 1e-9)
        pct = 100.0 * self.over_gate / self.checked if self.checked else 0.0
        out = [
            f"qc: {self.checked} clip(s) checked "
            f"({', '.join(f'{n} {r}' for r, n in sorted(self.by_reason.items()))}), "
            f"{self.over_gate} over gate ({pct:.1f}%)"
            + (f", {self.dropped} sample(s) dropped to keep up" if self.dropped else "")
            + (f", {self.failed} failed" if self.failed else ""),
            f"qc cost: {self.audio_s / 3600:.2f} h of audio read in "
            f"{self.stt_s / 60:.1f} min of CPU "
            f"({self.audio_s / self.stt_s if self.stt_s else 0:.1f}x realtime), "
            f"{100 * self.stt_s / (wall * max(workers, 1)):.0f}% duty cycle over "
            f"{wall / 3600:.2f} h",
        ]
        if self.defects:
            total = sum(self.defects.values())
            shapes = ", ".join(
                f"{self.defects[s]} {s} ({100 * self.defects[s] / total:.0f}%)"
                for s in DEFECT_SHAPES if self.defects.get(s))
            # Reported because only loop/overrun are visible to a cheap acoustic
            # detector; see `classify_defect`.
            out.append(f"qc defect mix ({total} flagged): {shapes}")
        for voice, cers in sorted(self.cer_by_voice.items()):
            ordered = sorted(cers)
            p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
            out.append(f"  {voice:<12} n={len(ordered):<6} "
                       f"CER median {statistics.median(ordered):.4f} p90 {p90:.4f} "
                       f"over {self.over_by_voice.get(voice, 0)}")
        return out


class Context:
    """Queues, output file and stats, shared by the poller and the worker threads."""

    def __init__(self, cfg, watches: dict[str, StageWatch]) -> None:
        self.cfg = cfg
        self.watches = watches
        self.stop = threading.Event()
        self.lock = threading.Lock()
        # Two queues, not one, because the two populations are not worth the same: a
        # retried clip is the whole point of this script, a sampled one is one of
        # 25,000 interchangeable draws. So retries are served first and are never the
        # ones dropped when the CPU falls behind.
        self.retries: deque[Job] = deque()
        self.samples: deque[Job] = deque()
        self.in_flight = 0
        self.stats = Stats()
        self.write_lock = threading.Lock()

    def enqueue(self, job: Job) -> None:
        watch = self.watches[job.stage]
        with self.lock:
            # A retried clip is dedup'd on its TAKE (stem + the sidecar row's
            # timestamp), a sampled one on its stem alone. The extra stem test is what
            # keeps a retried clip out of the sample as well: it is already going to be
            # checked, under the label that says why, and transcribing it twice would
            # spend the scarce CPU on a clip already covered and count it twice in the
            # per-voice table.
            if job.key in watch.handled:
                return
            if job.reason == "sample" and job.stem in watch.stems:
                return
            watch.handled.add(job.key)
            watch.stems.add(job.stem)
            if job.reason == "retry":
                self.retries.append(job)
                return
            self.samples.append(job)
            # The backlog is capped so a stall (a slow disk, a paused process, a
            # --backfill of a finished stage) cannot grow the queue without bound.
            # The OLDEST samples go, since a fresh clip says more about the run's
            # current state than one from an hour ago.
            while len(self.samples) > self.cfg.max_queue:
                self.samples.popleft()
                self.stats.bump("dropped")

    def next_job(self) -> Job | None:
        with self.lock:
            for queue in (self.retries, self.samples):
                if queue:
                    self.in_flight += 1
                    return queue.popleft()
            return None

    def done(self) -> None:
        with self.lock:
            self.in_flight -= 1

    def idle(self) -> bool:
        with self.lock:
            return not self.retries and not self.samples and self.in_flight == 0

    def depth(self) -> tuple[int, int]:
        with self.lock:
            return len(self.retries), len(self.samples)

    def write(self, path: Path, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self.write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)


def poll_sidecar(watch: StageWatch, ctx: Context) -> None:
    """Queue every clip that appeared in `cfg_alpha_sidecar.jsonl` since the last poll.

    The sidecar is append-only, so it is read from a byte offset rather than re-parsed
    whole: that is cheap, and it also makes a RE-DRAWN clip a new job. Stage 05 appends
    a fresh row whenever it re-draws a stem that already had one (`02_audit_durations.py`
    repairs a clip by deleting it, and the next resume regenerates it), and the new take
    deserves its own check, which a stem-keyed dedup would swallow.
    """
    path = watch.audio / SIDECAR_FILENAME
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= watch.sidecar_offset:
        if size < watch.sidecar_offset:      # truncated/replaced: start over
            watch.sidecar_offset = 0
        else:
            return
    with path.open("rb") as fh:
        fh.seek(watch.sidecar_offset)
        data = fh.read()
    # Only advance past COMPLETE lines: stage 05 may be mid-append, and half a row now
    # must be re-read in full on the next poll rather than parsed and lost.
    tail = data.rfind(b"\n")
    if tail < 0:
        return
    watch.sidecar_offset += tail + 1
    for raw in data[:tail].splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        stem = row.get("stem")
        if not stem:
            continue
        clip = watch.audio / f"{stem}.flac"
        if not clip.is_file():               # deleted by the duration auditor since
            continue
        ts = row.get("ts") or ""
        ctx.enqueue(Job(stage=watch.name, stem=stem, path=clip, reason="retry",
                        key=f"{stem}@{ts}", cfg_alpha=row.get("cfg_alpha"),
                        src_ts=ts or None))


def poll_audio(watch: StageWatch, ctx: Context) -> int:
    """Queue the sampled clips that appeared in the audio directory since the last poll.

    Returns the number of NEW clip files seen, sampled or not, which is what the caller
    uses to notice that a stage has started producing.
    """
    try:
        names = os.listdir(watch.audio)
    except OSError as exc:
        logger.debug("{}: cannot list {} ({})", watch.name, watch.audio, exc)
        return 0
    fresh = 0
    for name in names:
        if name in watch.seen:
            continue
        watch.seen.add(name)
        # parse_clip_name rejects everything that is not a finished clip: the two
        # sidecars, voice_assignment.json, and the `*.flac.tmp` a clip is written to
        # before its atomic rename (so a half-written file is never read).
        if parse_clip_name(name) is None:
            continue
        fresh += 1
        stem = name[: -len(".flac")]
        if is_sampled(stem, percent=ctx.cfg.sample_percent):
            ctx.enqueue(Job(stage=watch.name, stem=stem, path=watch.audio / name,
                            reason="sample", key=stem))
    return fresh


def seed_handled(watch: StageWatch, out_path: Path) -> int:
    """Re-read this stage's own QC output so a restart does not redo its work.

    Keys are rebuilt exactly as `poll_sidecar` / `poll_audio` build them, which is why
    the retry rows carry `src_ts`: it is the identity of the TAKE, not of the clip.
    """
    if not out_path.is_file():
        return 0
    count = 0
    with out_path.open(encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue                     # torn final line of a killed run
            stem = row.get("stem")
            if not stem:
                continue
            if row.get("reason") == "retry":
                watch.handled.add(f"{stem}@{row.get('src_ts') or ''}")
            else:
                watch.handled.add(stem)
            watch.stems.add(stem)
            count += 1
    return count


def check_clip(job: Job, model, watch: StageWatch, ctx: Context) -> None:
    """Transcribe one clip, score it against its label, and append its QC row."""
    cfg = ctx.cfg
    ensure_index(watch)
    row = manifest_row(watch, job.stem)
    if row is None:
        logger.warning("{}: no manifest row, not checked", job.stem)
        ctx.stats.bump("failed")
        return
    reference = _stt.get_reference(row, cfg.reference_key)
    if not reference:
        logger.warning("{}: manifest row has no {}, not checked",
                       job.stem, cfg.reference_key)
        ctx.stats.bump("failed")
        return
    voice = voice_for_row(row, field=watch.voice_field, mode=watch.mode)

    t0 = time.monotonic()
    try:
        wav, sr = sf.read(str(job.path), dtype="float32", always_2d=False)
    except Exception as exc:                 # noqa: BLE001 - a vanished/torn clip is
        logger.warning("{}: unreadable ({}), not checked", job.stem, exc)
        ctx.stats.bump("failed")                # reported, never fatal
        return
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    duration = len(wav) / sr if sr else 0.0
    # onnx-asr resamples to the model's 16 kHz itself when handed a waveform plus its
    # rate; its path API cannot be used here because it reads WAV through the stdlib
    # `wave` module only, and this corpus is 24 kHz FLAC.
    transcript = model.recognize(wav, sample_rate=sr)
    stt_s = time.monotonic() - t0

    wer, cer = _stt.compute_metrics(reference, transcript)
    tail = _stt.tail_cer(reference, transcript, duration)
    # Length-independent companions to the two averages above: see the gate constants.
    # `window_score` rather than `window_metrics` because the same alignment also yields
    # the text excerpts the WARNING below prints, at no extra cost.
    score = _stt.window_score(reference, transcript)
    window_cer, bad_run = score.window_cer, score.bad_run_chars
    over = ((cer is not None and cer >= cfg.cer_threshold)
            or (tail is not None and tail >= cfg.tail_cer_threshold)
            or (window_cer is not None and window_cer >= cfg.window_cer_threshold)
            or (bad_run is not None and bad_run >= cfg.bad_run_threshold))

    out_row = {
        "stem": job.stem,
        "stage": job.stage,
        "reason": job.reason,
        "voice": voice,
        "duration": round(duration, 2),
        "cer": None if cer is None else round(cer, 4),
        "wer": None if wer is None else round(wer, 4),
        "tail_cer": None if tail is None else round(tail, 4),
        # Stored on EVERY row, clean ones included: they are two numbers, and keeping
        # them is what lets the gates be re-swept over the sidecar later without
        # transcribing 602,885 clips a second time.
        "window_cer": None if window_cer is None else round(window_cer, 4),
        "bad_run_chars": bad_run,
        "over_gate": over,
        "stt_s": round(stt_s, 2),
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if job.cfg_alpha is not None:
        out_row["cfg_alpha"] = job.cfg_alpha
    if job.src_ts:
        out_row["src_ts"] = job.src_ts
    # The transcript is kept only where it is evidence. On a clean clip it repeats the
    # label and would add ~8 MB per 25,000 rows for nothing; on a flagged one it is the
    # entire reason to look, and the audio may well be deleted and re-drawn before
    # anyone does.
    if over or cfg.keep_transcripts:
        out_row["transcript"] = transcript
        out_row["reference"] = reference
    # Free: pure string work on text already in hand, and it is what turns a pile of
    # CER numbers into "how many of these are the mumbling, how many are fluent and
    # wrong". Only stored where it means something, i.e. on a flagged clip.
    defect = (classify_defect(reference, transcript, bad_run_chars=bad_run)
              if over else None)
    if defect:
        out_row["defect"] = defect
    ctx.write(watch.audio / cfg.out_name, out_row)

    total = ctx.stats.add(reason=job.reason, voice=voice, cer=cer, over=over,
                          duration=duration, stt_s=stt_s, defect=defect)
    if cfg.summary_every and total % cfg.summary_every == 0:
        for line in ctx.stats.lines(workers=cfg.workers):
            logger.info(line)
    if over:
        logger.warning("{} [{}, {}, {}]: CER {} tail {} window {} run {} ({:.1f}s) -> {}",
                       job.stem, job.reason, voice, defect,
                       "n/a" if cer is None else f"{cer:.3f}",
                       "n/a" if tail is None else f"{tail:.3f}",
                       "n/a" if window_cer is None else f"{window_cer:.3f}",
                       "n/a" if bad_run is None else bad_run,
                       duration, transcript[:160])
        # The two numbers above say a clip is bad; these lines say WHAT about it is bad,
        # which is what makes a flag actionable without opening the audio.
        log_window_evidence(score)


def report_defect_mix(paths: dict[str, Path], *, severe_cer: float = 0.25,
                      examples: int = 1) -> int:
    """Print the defect mix of sidecar rows ALREADY on disk, and return the row count.

    A pure re-read: it classifies from the `transcript`/`reference` the rows already
    carry, so it needs neither the ONNX model nor the audio, and it costs milliseconds
    over tens of thousands of rows. That is what makes it the right tool for rows
    written before `classify_defect` existed, for asking the mix question of a finished
    stage without re-transcribing anything, and for re-measuring the mix after a change
    to the classifier itself: it recomputes every shape rather than reading the row's
    stored `defect`, so what it prints is always the current rule's answer, not the one
    the running QC happened to write.

    The split at `severe_cer` is the point of the report rather than a detail: the two
    halves answer the pre-filter question differently, since the mild end is mostly
    wording and the severe end mostly derailment, and one global percentage would be
    wrong in both directions.

    Parameters
    ----------
    paths
        Stage name -> sidecar file. Missing files are reported and skipped.
    severe_cer
        CER at or above which a flagged clip counts as a derailment rather than a
        wording miss.
    examples
        How many worked examples to print per defect shape, worst window first. 0 for
        the counts alone. Each one prints the two texts behind its score, which is what
        makes a flag judgeable without opening the audio.

    Returns
    -------
    int
        How many flagged rows carried enough text to classify.
    """
    total = 0
    for name, path in sorted(paths.items()):
        if not path.is_file():
            logger.warning("{}: no {}", name, path.name)
            continue
        mix_all: Counter = Counter()
        mix_severe: Counter = Counter()
        by_voice: dict[str, Counter] = {}
        # Worst stored row per shape, to print the evidence for. Keyed by shape because
        # the shapes live at different CERs: `skipped` is low by construction (that is
        # the point of the window gates), so a global "worst N" would only ever show
        # loops and never the defect hardest to believe without seeing it.
        worst: dict[str, list[tuple[float, dict, object]]] = {}
        rows = flagged = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue              # a torn final line, same as everywhere else here
            rows += 1
            if not row.get("over_gate"):
                continue
            flagged += 1
            if not row.get("transcript") or not row.get("reference"):
                continue
            score = _stt.window_score(row["reference"], row["transcript"])
            # Recomputed, not read from the row: the stored `defect` is a cache written
            # by whatever rule version was running when the clip was checked, so
            # trusting it would freeze the mix at the old rule and hide exactly the
            # change one wants to measure (that is how the long-clause loop arm was
            # measured over rows already on disk, with no re-transcription). The row's
            # own field is still what the dataset carries; this report is the live read.
            shape = classify_defect(row["reference"], row["transcript"],
                                    bad_run_chars=score.bad_run_chars)
            if examples:
                rank = score.window_cer or row.get("cer") or 0.0
                keep = worst.setdefault(shape, [])
                keep.append((rank, row, score))
                keep.sort(key=lambda item: -item[0])
                del keep[examples:]      # bounded, so a whole stage costs no memory
            mix_all[shape] += 1
            if (row.get("cer") or 0.0) >= severe_cer:
                mix_severe[shape] += 1
            by_voice.setdefault(row.get("voice") or "?", Counter())[shape] += 1
            total += 1

        def fmt(mix: Counter) -> str:
            n = sum(mix.values())
            if not n:
                return "none"
            return ", ".join(f"{mix[s]} {s} ({100 * mix[s] / n:.0f}%)"
                             for s in DEFECT_SHAPES if mix.get(s))

        logger.info("{}: {} row(s), {} flagged, {} classified", name, rows, flagged,
                    sum(mix_all.values()))
        logger.info("  all flagged     : {}", fmt(mix_all))
        logger.info("  CER >= {:.2f} only: {}", severe_cer, fmt(mix_severe))
        for voice, mix in sorted(by_voice.items()):
            logger.info("  {:<12} {}", voice, fmt(mix))
        for shape in DEFECT_SHAPES:
            for rank, row, score in worst.get(shape, ()):
                logger.info("  worst {} ({}, {}, {:.1f}s): CER {} window {} run {}",
                            shape, row.get("stem", "?"), row.get("voice", "?"),
                            row.get("duration") or 0.0, row.get("cer"),
                            "n/a" if score.window_cer is None
                            else f"{score.window_cer:.3f}", score.bad_run_chars)
                log_window_evidence(score, log=logger.info, indent="    ")
    return total


def worker_loop(model, ctx: Context) -> None:
    """Drain the queues until asked to stop, retries before samples."""
    while not ctx.stop.is_set():
        job = ctx.next_job()
        if job is None:
            ctx.stop.wait(1.0)
            continue
        try:
            check_clip(job, model, ctx.watches[job.stage], ctx)
        except Exception as exc:             # noqa: BLE001 - one bad clip must not
            logger.exception("{}: check failed ({})", job.stem, exc)   # end the run
            ctx.stats.bump("failed")
        finally:
            ctx.done()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stage", "stages", multiple=True, type=str,
              help="Stage directory to follow, repeatable (default: all five, "
                   f"{', '.join(_audit.STAGES)}). A stage that is not producing yet "
                   "costs nothing: its manifest is only indexed once a clip of its "
                   "own shows up.")
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path),
              default=str(DEFAULT_MODEL_DIR), show_default=True,
              help="A parakeet-tdt-0.6b-v3-*-onnx export. The default is fine-tuned "
                   "on v1 of THIS dataset, so it is a gross-error detector, not an "
                   "impartial judge: see the module docstring.")
@click.option("--quantization", default="int8", show_default=True,
              help="Precision subfolder of the export to load. int8 is the one "
                   "measured on this host; fp16 is ~2x slower on CPU.")
@click.option("--sample-percent", default=5.0, show_default=True,
              help="Percent of clips to check beyond the retried ones. The draw is a "
                   "hash of the clip stem, so it is identical across restarts. 0 "
                   "checks only clips that needed a retry.")
@click.option("--retries/--no-retries", "do_retries", default=True, show_default=True,
              help="Check every clip that has a cfg_alpha_sidecar.jsonl row, i.e. "
                   "every clip whose first take was rejected for its duration.")
@click.option("--threads", default=3, show_default=True,
              help="onnxruntime intra-op threads PER worker. Measured on this host: "
                   "3 threads x 2 workers 9.4x realtime, 4x1 7.7x, 6x1 6.7x (past ~4 "
                   "it is memory-bandwidth bound and fights the generator).")
@click.option("--workers", default=1, show_default=True,
              help="Parallel checks. 1 is deliberate: a 5% sample needs ~1.5x "
                   "realtime and one worker delivers ~7x, so a second one would only "
                   "take threads from the process feeding the GPU.")
@click.option("--nice", "nice_level", default=10, show_default=True,
              help="Niceness applied to this process, so the generator wins every "
                   "contested timeslice. 0 to leave the priority alone.")
@click.option("--poll-seconds", default=30.0, show_default=True,
              help="How often to look for new clips and new sidecar rows.")
@click.option("--max-queue", default=2000, show_default=True,
              help="Cap on QUEUED samples; the oldest are dropped past it (counted in "
                   "the summary). Retries are never dropped.")
@click.option("--cer-threshold", default=DEFAULT_CER_GATE, show_default=True,
              help="Whole-clip CER at or above which a clip is logged and keeps its "
                   "transcript. Stage 06's own gate.")
@click.option("--tail-cer-threshold", default=DEFAULT_TAIL_CER_GATE, show_default=True,
              help="Same, for the CER of the clip's last ~30 s, where truncations and "
                   "derailed endings live. Looser, like stage 06's.")
@click.option("--window-cer-threshold", default=DEFAULT_WINDOW_CER_GATE,
              show_default=True,
              help="CER of the worst ~15 s window of the clip at or above which it is "
                   "flagged. Catches a defect in the MIDDLE of a long clip, which the "
                   "two gates above average away.")
@click.option("--bad-run-threshold", default=DEFAULT_BAD_RUN_GATE, show_default=True,
              help="Longest contiguous run of missing label characters (~14.2 per "
                   "second of speech) at or above which a clip is flagged. 30 is five "
                   "French words, and above every clean clip in a 20k calibration.")
@click.option("--reference-key", default="asr_training_source", show_default=True,
              help="Manifest field holding the text the TTS was asked to say.")
@click.option("--backfill", is_flag=True,
              help="Also check the clips (and sidecar rows) ALREADY on disk. Without "
                   "it only clips that appear from now on are checked.")
@click.option("--once", is_flag=True,
              help="Drain the current queue and exit instead of tailing. With "
                   "--backfill this is the manual 'check this folder' mode.")
@click.option("--summary-every", default=200, show_default=True,
              help="Log the per-voice CER table every N checks (0 to only log it at "
                   "exit).")
@click.option("--keep-transcripts", is_flag=True,
              help="Store the transcript of every checked clip, not just the flagged "
                   "ones (~8 MB per 25k clips).")
@click.option("--examples", default=1, show_default=True,
              help="With --defect-mix, how many worked examples to print per defect "
                   "shape (worst window first), each with the label text of its worst "
                   "window and what the recognizer heard over the same stretch. 0 for "
                   "the counts alone.")
@click.option("--defect-mix", is_flag=True,
              help="Do not transcribe anything: re-read the sidecar rows already on "
                   "disk and report what SHAPE their failures have (loop, overrun, "
                   "truncated, wrong_words), overall and among the severe ones. Needs "
                   "neither the model nor the audio.")
@click.option("--out-name", default=QC_FILENAME, show_default=True,
              help="Output filename, written in each stage's audio directory.")
@click.option("--log-file", default="qc_cpu_stt.log", show_default=True,
              help="Append the log here as well as to stderr; empty string disables. "
                   "A bare filename lands beside this script.")
def main(stages: tuple[str, ...], model_dir: Path, quantization: str,
         sample_percent: float, do_retries: bool, threads: int, workers: int,
         nice_level: int, poll_seconds: float, max_queue: int, cer_threshold: float,
         tail_cer_threshold: float, window_cer_threshold: float,
         bad_run_threshold: int, reference_key: str, backfill: bool, once: bool,
         summary_every: int, keep_transcripts: bool, defect_mix: bool,
         examples: int, out_name: str,
         log_file: str) -> None:
    """STT-check a sample of a running stage-05 generation on the spare CPU cores."""
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | <level>{message}</level>",
               level="INFO")
    if log_file:
        log_path = Path(log_file)
        logger.add(log_path if log_path.is_absolute() else _HERE / log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   level="INFO", enqueue=True)

    names = stages or tuple(_audit.STAGES)
    unknown = [n for n in names if n not in _audit.STAGES]
    if unknown:
        raise click.BadParameter(
            f"unknown stage(s) {unknown}, known: {list(_audit.STAGES)}")
    if nice_level:
        try:
            os.nice(nice_level)
        except OSError as exc:
            logger.warning("could not renice to +{} ({})", nice_level, exc)

    cfg = SimpleNamespace(
        sample_percent=sample_percent, max_queue=max_queue,
        cer_threshold=cer_threshold, tail_cer_threshold=tail_cer_threshold,
        window_cer_threshold=window_cer_threshold, bad_run_threshold=bad_run_threshold,
        reference_key=reference_key, out_name=out_name, workers=workers,
        keep_transcripts=keep_transcripts, summary_every=summary_every)

    watches: dict[str, StageWatch] = {}
    for name in names:
        stage_dir = ROOT / name
        jsonl = stage_dir / "generated_dataset.jsonl"
        audio = stage_dir / "generated_audios"
        if not jsonl.is_file() or not audio.exists():
            logger.warning("{}: no generated_dataset.jsonl or generated_audios, skipped",
                           name)
            continue
        # Resolve the symlink: these directories point at an external SSD, and
        # `os.listdir` on the link works but every logged path would be the link.
        watches[name] = StageWatch(name=name, mode=_audit.STAGES[name], jsonl=jsonl,
                                   audio=audio.resolve())
    if not watches:
        raise click.ClickException("no stage to follow")

    if defect_mix:
        n = report_defect_mix({w.name: w.audio / out_name for w in watches.values()},
                              examples=examples)
        if not n:
            logger.warning("no flagged row carried a transcript to classify")
        return

    ctx = Context(cfg, watches)

    for watch in watches.values():
        resumed = seed_handled(watch, watch.audio / out_name)
        if not backfill:
            # Mark what is already on disk as seen WITHOUT queueing it, which is what
            # makes the default mode a tail: only clips generated from now on are
            # checked. The sidecar is skipped the same way, by starting at its EOF.
            try:
                watch.seen.update(os.listdir(watch.audio))
            except OSError:
                pass
            try:
                watch.sidecar_offset = (watch.audio / SIDECAR_FILENAME).stat().st_size
            except OSError:
                watch.sidecar_offset = 0
        logger.info("{}: following {} ({} clip(s) on disk, {} already checked{})",
                    watch.name, watch.audio, len(watch.seen), resumed,
                    ", backfilling" if backfill else "")

    logger.info("loading {} ({}) with {} thread(s) x {} worker(s)...",
                model_dir.name, quantization, threads, workers)
    t0 = time.monotonic()
    model = load_cpu_model(model_dir.resolve(), quantization=quantization,
                           threads=threads)
    logger.info("model loaded in {:.1f}s; sample {}%, retries {}, gates {} / {} (tail) "
                "/ {} (window) / {} chars (missing run)",
                time.monotonic() - t0, sample_percent,
                "on" if do_retries else "off", cer_threshold, tail_cer_threshold,
                window_cer_threshold, bad_run_threshold)

    def _stop(signum, _frame):
        logger.warning("signal {}, finishing the clip in flight and stopping", signum)
        ctx.stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # The model is shared by every worker on purpose: an onnxruntime session is
    # thread-safe for Run(), and a second copy of the encoder would cost another
    # 619 MB of RAM for nothing.
    pool = [threading.Thread(target=worker_loop, args=(model, ctx), daemon=True,
                             name=f"qc{i}") for i in range(max(workers, 1))]
    for thread in pool:
        thread.start()

    try:
        polled = False
        while not ctx.stop.is_set():
            # --once means "drain what is on disk NOW", so it polls exactly once: the
            # generator keeps adding clips, so re-listing a 100k-entry directory every
            # few seconds would both cost real CPU and keep moving the finish line.
            if once and polled:
                if ctx.idle():
                    break
                ctx.stop.wait(5.0)
                continue
            for watch in watches.values():
                # Classifying a clip needs only its stem (the sample is a hash) and the
                # cfg_alpha sidecar, so the poller never touches the manifest: the index
                # is built by the first worker that actually has a clip to check, which
                # is what keeps following all five stages free until they produce.
                poll_audio(watch, ctx)
                if do_retries:
                    poll_sidecar(watch, ctx)
            polled = True
            retry_depth, sample_depth = ctx.depth()
            if retry_depth or sample_depth:
                logger.info("queue: {} retry, {} sample", retry_depth, sample_depth)
            if once:
                continue
            ctx.stop.wait(poll_seconds)
    finally:
        ctx.stop.set()
        for thread in pool:
            thread.join(timeout=120)
        for line in ctx.stats.lines(workers=workers):
            logger.info(line)


if __name__ == "__main__":
    main()
