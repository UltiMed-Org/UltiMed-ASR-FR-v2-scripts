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

    # Transcribe ONLY the clips the acoustic pre-filter flagged, to confirm or clear
    # them (see 04_qc_acoustic_loops.py; --sample-percent 0 turns the sample off):
    ./03_qc_cpu_stt.py --stage 01_dictionnary --backfill --once --sample-percent 0 \
        --stems-from /path/to/01_dictionnary/generated_audios/qc_acoustic_sidecar.jsonl

    # impartial second opinion on the retried clips only, no sample
    ./03_qc_cpu_stt.py --sample-percent 0 \
        --model-dir ../../parakeet-tdt-0.6b-v3-redux-onnx

Stopping with Ctrl-C is safe at any moment: the clip in flight finishes, the queue is
dropped, and the next run re-reads the output file and picks up where this one stopped.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib.util
import json
import os
import signal
import statistics
import sys
import threading
import types
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
import stem_requests as _stems  # noqa: E402
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
DEFAULT_MODEL_DIR = ROOT.parent / "parakeet-tdt-0.6b-v3-ultra-UltiMed-onnx"

#: The four gates, re-exported from the module that owns the four scores
#: (`06_hotfixes/01_compute_stt.py`, which carries the measured calibration behind each
#: number). Read from there rather than restated here so "over the gate" cannot come to
#: mean two different things in two stages: this early warning exists to predict what the
#: real QC loop will flag, which it only does while both read the same numbers.
DEFAULT_CER_GATE = _stt.DEFAULT_CER_GATE
DEFAULT_TAIL_CER_GATE = _stt.DEFAULT_TAIL_CER_GATE
DEFAULT_WINDOW_CER_GATE = _stt.DEFAULT_WINDOW_CER_GATE
DEFAULT_BAD_RUN_GATE = _stt.DEFAULT_BAD_RUN_GATE
DEFAULT_REPEAT_GATE = _stt.DEFAULT_REPEAT_GATE


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
        one shape a self-similarity detector can find without transcribing. Counted by `repeat_excess`, which asks how many word windows the take reads
      more often than the label writes them, forgiving a window the label writes
      that often within one word (an honestly repeated clause misheard the same way
      twice is not a loop; see that function for the measurement).
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
    wrong_words, because the rule of the day wanted three occurrences of a short n-gram
    and a derailed take most often says a clause exactly twice. Once a clause said twice
    counted (first as a second arm here, now as `repeat_excess`), 250 flagged
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
    # One rule, shared with the offline scorer (`06_hotfixes/01_compute_stt.py`), because
    # "did this take repeat itself" is one question and two spellings of it drift apart
    # the first time either is tuned. It replaced a pair of arms here (a 2-to-5-word
    # n-gram seen 3+ times, plus a 6+-word clause seen twice) that it measurably
    # subsumes: over the 19,120 clips carrying both a Parakeet transcript and an
    # independent acoustic verdict, the old pair fired on 308 (87.0% of them
    # acoustically confirmed, 17.1% of the acoustic flags recovered) and this fires on
    # 457 (81.2%, 23.7%), catching 307 of the old 308 plus 150 more of which 68.7% are
    # acoustically confirmed against an 8.2% base rate. The one clip only the old pair
    # flagged is acoustically clean. Asked before the length ratios on purpose: a take
    # that loops usually also runs long, and of the two names the loop is the cause and
    # `overrun` the symptom, which is what moved 26 of the 28 stored `overrun` rows to
    # `loop` on 2026-10-16.
    if _stt.repeat_excess(reference or "", transcript or "") >= _stt.DEFAULT_REPEAT_GATE:
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


# Batch defaults for the GPU sweep, measured on an RTX 3090 Ti over 300 dictionary
# clips (p50 20.3 s): see docs/PARAKEET_SWEEP_THROUGHPUT.md. The clip count is a ceiling
# the padded-seconds cap almost always reaches first, which is why it can sit this high.
# The cap is 600 and not the fastest value measured (900 s, 39.9 clips/s against 35.7)
# because that bench was 300 clips long and a sweep is half a million: onnxruntime's BFC
# arena never returns GPU memory, so thousands of distinct padded shapes walk its
# high-water mark up for hours. At 900 it reached 23.4 GB of a 24.6 GB card within 2500
# clips and the largest batches then failed to allocate. 600 trades 10% of the rate for
# a footprint measured at 8.6 GB, and `run_group` catches what still does not fit.
DEFAULT_BATCH_SEC = 600.0
# How many batches' worth of clips to pop and length-sort together. 3 keeps the audio
# held in RAM near 400 MB at batch 64 while capturing the sort's full 20%.
BATCH_SORT_WINDOW = 3
# onnxruntime CUDA defaults this workload does not want. The conv algorithm search
# re-benchmarks every unseen input shape, and nearly every clip has a unique length;
# HEURISTIC picks instantly and was measured BIT-IDENTICAL over 60 clips (it costs only
# 2% here, but there is nothing to lose). The arena doubles on a miss, so the peak
# depended on allocation order: one configuration touched 24.0 GB of a 24.6 GB card
# where the same work fits in 13 GB. Growing by exactly what was asked keeps a long
# run's footprint where the batch cap puts it, on a card that also drives a display.
CUDA_PROVIDER_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC",
                         "arena_extend_strategy": "kSameAsRequested"}


def batched_tdt_decoding(self, encoder_out, encoder_out_lens, /, **kwargs):
    """Decode a whole batch of clips in lockstep, replacing onnx-asr's per-clip loop.

    COPIED, deliberately and with the user's agreement, from the model repo's
    `parakeet_web/scripts/wer-quants.py::batched_tdt_decoding`. It is the same function
    in two repos: a fix to one belongs in the other, and neither copy can be changed
    without the other. The alternative was importing it across repos by relative path,
    which breaks as soon as `--model-dir` points at a bare HF download.

    This is the single biggest throughput lever in the sweep, and it exists because
    onnx-asr's batching is encoder-only: `NemoConformerTdt._decode` runs `decoder_joint`
    with `encoder_outputs[None, :, None]` and `targets [[tok]]`, i.e. batch 1, one
    encoder frame at a time. A 64-clip batch of 20 s clips (250 frames each at 12.5 fps)
    therefore costs ~16,000 sequential ONNX calls, each a few hundred microseconds of
    launch latency on a GPU that then sits 74% idle: measured 10 clips/s, against 38.5
    for this function on the same clips and the same card.

    Here each step feeds every clip still running as one row of ONE call, e.g. 8 clips ->
    encoder_outputs (8, 1024, 1), targets (8, 1), states (2, 8, 640), so the batch costs
    ~250 calls rather than 16,000. The per-clip rules are the stock loop's (argmax token,
    keep the state only on a non-blank, advance by the predicted duration, else by 1 on a
    blank or after `max_tokens_per_step`), so at batch 1 it IS the stock loop.

    One correctness condition: batching reproduces the per-clip result only if the
    decoder's rows cannot interact. That holds for the fp32 decoder and NOT for the
    dynamic-int8 one, which derives a single activation scale per call, so a clip could
    flip a near-tie word depending on its batch mates. That is why `--decoder-quantization
    fp32` is the default and why `load_cpu_model` resolves the decoder separately.
    """
    import numpy as np
    from onnx_asr.utils import log_softmax

    need_logprobs = kwargs.get("need_logprobs")
    lens = np.minimum(encoder_out_lens, encoder_out.shape[1])
    n, blank, vocab, max_tokens = (len(lens), self._blank_idx, self._vocab_size,
                                   self._max_tokens_per_step)
    s1, s2 = (np.repeat(s, n, axis=1) for s in self._create_state())
    last = np.full(n, blank, np.int32)   # previously emitted token, blank before the first
    t = np.zeros(n, np.int64)
    emitted = np.zeros(n, np.int64)      # tokens emitted at the current frame
    tokens, stamps, logprobs = ([[] for _ in range(n)] for _ in range(3))
    act = np.flatnonzero(lens > 0)       # the clips still running, shrinking every step
    while len(act):
        out, n1, n2 = self._decoder_joint.run(
            ["outputs", "output_states_1", "output_states_2"],
            {"encoder_outputs": encoder_out[act, t[act]][:, :, None],
             "targets": last[act, None],
             "target_length": np.ones(len(act), np.int32),
             "input_states_1": s1[:, act], "input_states_2": s2[:, act]})
        out = out.reshape(len(act), -1)
        tok, step = out[:, :vocab].argmax(1), out[:, vocab:].argmax(1)
        emit = tok != blank
        for k in np.flatnonzero(emit):
            b = act[k]
            tokens[b].append(int(tok[k]))
            stamps[b].append(int(t[b]))
            if need_logprobs:
                logprobs[b].append(log_softmax(out[k, :vocab])[tok[k]])
        e = act[emit]
        last[e], s1[:, e], s2[:, e] = tok[emit], n1[:, emit], n2[:, emit]
        emitted[e] += 1
        adv = np.where(step > 0, step,
                       (~emit | (emitted[act] == max_tokens)).astype(np.int64))
        t[act] += adv
        emitted[act[adv > 0]] = 0
        act = act[t[act] < lens[act]]
    for b in range(n):
        yield tokens[b], stamps[b], logprobs[b] if need_logprobs else None


def load_cpu_model(model_dir: Path, *, quantization: str, threads: int,
                   providers: tuple[str, ...] = (),
                   decoder_quantization: str | None = None):
    """Load an onnx-asr Parakeet model from a LOCAL per-precision export directory.

    Parameters
    ----------
    model_dir
        A `parakeet-tdt-0.6b-v3-*-onnx` checkout.
    quantization
        Subfolder/precision to load, e.g. ``"int8"``. Empty for the export's
        unsuffixed files, which is what the fp32 pair is named.
    threads
        onnxruntime intra-op threads. Pinned deliberately: left alone, onnxruntime
        takes every core and starves the process feeding the GPU, which is the one
        thing this script must not do.
    providers
        onnxruntime execution providers, empty for onnxruntime's own default (CPU on
        the wheel this script pins). See the `--providers` option for when the GPU is
        worth it and what else has to be installed for it to exist.

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

    def model_files(quantization=None):
        """Resolve the encoder at `quantization`, the decoder_joint independently.

        onnx-asr's single `quantization` argument picks BOTH files, so asking for the
        fp16 encoder makes it look for `decoder_joint-model.fp16.onnx`, which this
        export does not ship (its fp16/ folder holds an encoder only), and the load
        fails before a session is built. Pairing a quantized encoder with the fp32
        decoder is also what keeps a BATCHED transcript independent of its batch mates:
        the int8 decoder derives one activation scale per call, so a clip's near-tie
        word could flip with its neighbours.
        """
        files = dict(original(quantization))
        if decoder_quantization is not None and "decoder_joint" in files:
            want = None if decoder_quantization in ("", "fp32") else decoder_quantization
            files["decoder_joint"] = original(want)["decoder_joint"]
        return {key: relocate(value) for key, value in files.items()}

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1

    name = "nemo-parakeet-tdt-0.6b-v3"
    model_type = create_asr_resolver(name, str(model_dir)).model_type
    original = model_type._get_model_files
    model_type._get_model_files = staticmethod(model_files)
    extra = {"providers": [(p, CUDA_PROVIDER_OPTIONS) if p == "CUDAExecutionProvider"
                           else p for p in providers]} if providers else {}
    try:
        # Empty string means "the files carry no precision infix at all"
        # (`encoder-model.onnx`, not `encoder-model.int8.onnx`), which onnx-asr spells
        # as None. The ultra export needs it: its fp16/ folder holds only an encoder,
        # so the only pair that resolves is the unsuffixed one under fp32/.
        model = onnx_asr.load_model(name, path=str(model_dir),
                                    quantization=quantization or None,
                                    sess_options=opts, **extra)
        # Swap in the batched TDT decode loop. Bound per instance rather than on the
        # class so nothing else importing onnx-asr in this process is affected, and
        # only on a TDT model, since the loop reads `_vocab_size` to split the token
        # head from the duration head.
        from onnx_asr.models.nemo import NemoConformerTdt
        if isinstance(model.asr, NemoConformerTdt):
            model.asr._decoding = types.MethodType(batched_tdt_decoding, model.asr)
        else:
            logger.warning("{} is not a TDT model, keeping onnx-asr's per-clip decode",
                           name)
        return model
    finally:
        model_type._get_model_files = staticmethod(original)


@dataclass
class Job:
    """One clip to check."""

    stage: str
    stem: str
    path: Path
    reason: str                     # "sample", "retry", or "acoustic" (--stems-from)
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
        self.warned_drop = False

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
            # The backlog is capped so a stall (a slow disk, a paused process) cannot
            # grow the queue without bound. The OLDEST samples go, since a fresh clip
            # says more about the run's current state than one from an hour ago.
            #
            # A cap of 0 means no cap, which is what --backfill defaults to. There the
            # queue is not a backlog but the KNOWN population on disk, and trimming it
            # returns incomplete coverage while still exiting 0: a --backfill --once
            # sweep of 602k clips under the old default of 2000 checked 2000 of them,
            # logged "63795 sample(s) dropped to keep up", and looked finished.
            if not self.cfg.max_queue:
                return
            while len(self.samples) > self.cfg.max_queue:
                if self.cfg.backfill and not self.warned_drop:
                    self.warned_drop = True
                    logger.warning(
                        "--max-queue {} is smaller than the backfill population: the "
                        "oldest queued clips are being DROPPED, so this pass will not "
                        "cover the directory. Raise it, or pass 0 for no cap.",
                        self.cfg.max_queue)
                self.samples.popleft()
                self.stats.bump("dropped")

    def next_job(self) -> Job | None:
        with self.lock:
            for queue in (self.retries, self.samples):
                if queue:
                    self.in_flight += 1
                    return queue.popleft()
            return None

    def next_batch(self, limit: int) -> list[Job]:
        """Pop up to `limit` jobs, retries first, WITHOUT waiting for a full batch.

        Returning a short list is deliberate: at the tail of a stage, or while following
        a live generation, holding clips back until `limit` of them exist would park the
        sweep behind the generator instead of behind the GPU.
        """
        with self.lock:
            jobs: list[Job] = []
            for queue in (self.retries, self.samples):
                while queue and len(jobs) < limit:
                    jobs.append(queue.popleft())
            self.in_flight += len(jobs)
            return jobs

    def done(self, count: int = 1) -> None:
        with self.lock:
            self.in_flight -= count

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


def load_stem_requests(paths: tuple[Path, ...]) -> dict[str, set[str]]:
    """Read "check exactly these clips" lists. Returns stage name -> stems, "" = any.

    Thin wrapper over `utils/stem_requests.py`, which owns the parsing because stage 06's
    repair pass reads the same two file shapes: this one only adds the click error type
    and the loguru line. See that module for what the shapes are and why a missing file
    is fatal.

    This is the second half of the acoustic pre-filter
    (`04_qc_acoustic_loops.py`): that script scores every clip in the corpus from the
    waveform alone and flags ~1% of them, and this is how those flags come back here to
    be confirmed by an actual transcription. Confirming 1% of 602,885 clips costs ~2 h on
    one core, where transcribing all of them would cost ~230 h, which is the entire reason
    the pre-filter exists.

    Two input shapes are accepted because both are what a caller actually has: an
    acoustic sidecar (JSONL, one object per clip) contributes the stems of its `looped`
    rows and keys them by the row's own `stage`, so a file cannot leak stems into a
    sibling stage; anything else is read as a plain list of stems, one per line, `#`
    comments allowed, and applies to every followed stage. A JSONL row with no `looped`
    key at all still counts, so a hand-filtered file works.

    Parameters
    ----------
    paths
        Files to read. A missing file is a hard error: silently checking nothing would
        look exactly like a clean corpus.

    Returns
    -------
    dict
        Stage name -> set of stems. The key `""` holds stems that apply to any stage.
    """
    def log_one(path: Path, rows: int, plain: int) -> None:
        logger.info("{}: {} flagged row(s), {} bare stem(s) to check", path.name, rows,
                    plain)

    try:
        return _stems.load_stem_requests(paths, on_file=log_one)
    except FileNotFoundError as exc:
        # The shared loader is stdlib-only and cannot raise click's error type, but a
        # bad path given on this command line should still read as a usage error.
        raise click.BadParameter(f"--stems-from: {exc}") from exc


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
    # Union of the lists aimed at this stage and the stage-agnostic ones, resolved once
    # per poll rather than per clip: this loop runs over a 100k-entry directory.
    requested = (ctx.cfg.stem_requests.get(watch.name, frozenset())
                 | ctx.cfg.stem_requests.get("", frozenset()))
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
        # An explicitly requested clip is checked whatever the sample says, which is what
        # lets --sample-percent 0 mean "only what I asked for". The key stays the stem, so
        # a clip the sample already transcribed is not transcribed a second time: the
        # confirmation pass only pays for clips that have never been looked at.
        asked = stem in requested
        if asked or is_sampled(stem, percent=ctx.cfg.sample_percent):
            ctx.enqueue(Job(stage=watch.name, stem=stem, path=watch.audio / name,
                            reason="acoustic" if asked else "sample", key=stem))
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


@dataclass
class ClipInput:
    """One clip read and ready to transcribe, plus what scoring it will need later.

    Split out of `check_clip` so the batched path can read many clips on its loader
    threads, hand every waveform to ONE `recognize()` call, and then score each with
    the very same code the one-clip-at-a-time path uses.
    """

    row: dict
    reference: str
    voice: str
    wav: object
    sr: int
    duration: float
    t0: float


def prepare_clip(job: Job, watch: StageWatch, ctx: Context) -> ClipInput | None:
    """Find a clip's label and read its audio, or count it failed and return None."""
    cfg = ctx.cfg
    ensure_index(watch)
    row = manifest_row(watch, job.stem)
    if row is None:
        logger.warning("{}: no manifest row, not checked", job.stem)
        ctx.stats.bump("failed")
        return None
    reference = _stt.get_reference(row, cfg.reference_key)
    if not reference:
        logger.warning("{}: manifest row has no {}, not checked",
                       job.stem, cfg.reference_key)
        ctx.stats.bump("failed")
        return None
    voice = voice_for_row(row, field=watch.voice_field, mode=watch.mode)

    t0 = time.monotonic()
    try:
        wav, sr = sf.read(str(job.path), dtype="float32", always_2d=False)
    except Exception as exc:                 # noqa: BLE001 - a vanished/torn clip is
        logger.warning("{}: unreadable ({}), not checked", job.stem, exc)
        ctx.stats.bump("failed")                # reported, never fatal
        return None
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return ClipInput(row=row, reference=reference, voice=voice, wav=wav, sr=sr,
                     duration=len(wav) / sr if sr else 0.0, t0=t0)


def check_clip(job: Job, model, watch: StageWatch, ctx: Context) -> None:
    """Transcribe one clip, score it against its label, and append its QC row."""
    prep = prepare_clip(job, watch, ctx)
    if prep is None:
        return
    # onnx-asr resamples to the model's 16 kHz itself when handed a waveform plus its
    # rate; its path API cannot be used here because it reads WAV through the stdlib
    # `wave` module only, and this corpus is 24 kHz FLAC.
    transcript = model.recognize(prep.wav, sample_rate=prep.sr)
    score_clip(job, prep, transcript, time.monotonic() - prep.t0, watch, ctx)


def score_clip(job: Job, prep: ClipInput, transcript: str, stt_s: float,
               watch: StageWatch, ctx: Context) -> None:
    """Score one transcript against its clip's label and append the clip's QC row."""
    cfg = ctx.cfg
    reference, voice, duration = prep.reference, prep.voice, prep.duration
    wer, cer = _stt.compute_metrics(reference, transcript)
    tail = _stt.tail_cer(reference, transcript, duration)
    # Length-independent companions to the two averages above: see the gate constants.
    # `window_score` rather than `window_metrics` because the same alignment also yields
    # the text excerpts the WARNING below prints, at no extra cost.
    score = _stt.window_score(reference, transcript)
    window_cer, bad_run = score.window_cer, score.bad_run_chars
    # The fifth gate is not a distance at all, and that is the point: the four above all
    # measure how far the transcript is from the label, and a loop can sit at any
    # distance. The confirmed loop in `022432_0001_syndrome_de_frankl_hochwart_pellizzi`
    # scores CER 0.338 while an innocent "monsieur renaud" read "monsieur renault"
    # scores 0.321, so no threshold on these four separates them; raising the CER gate
    # to 0.15 to catch loops drops recall from 98% to 68% and to 46% at 0.20. Counting
    # repeats instead asks a different question of the same transcript, for free.
    repeat = _stt.repeat_excess(reference, transcript)
    over = ((cer is not None and cer >= cfg.cer_threshold)
            or (tail is not None and tail >= cfg.tail_cer_threshold)
            or (window_cer is not None and window_cer >= cfg.window_cer_threshold)
            or (bad_run is not None and bad_run >= cfg.bad_run_threshold)
            or (cfg.repeat_threshold > 0 and repeat >= cfg.repeat_threshold))

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
        "repeat_excess": repeat,
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
        logger.warning("{} [{}, {}, {}]: CER {} tail {} window {} run {} repeat {}"
                       " ({:.1f}s) -> {}",
                       job.stem, job.reason, voice, defect,
                       "n/a" if cer is None else f"{cer:.3f}",
                       "n/a" if tail is None else f"{tail:.3f}",
                       "n/a" if window_cer is None else f"{window_cer:.3f}",
                       "n/a" if bad_run is None else bad_run, repeat,
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


def batch_groups(prepared: list[tuple[Job, ClipInput]], batch: int, max_sec: float):
    """Cut length-sorted clips into `recognize()` calls.

    A call is capped BOTH by clip count and by PADDED seconds (clips x the longest clip
    in the call), and the second cap is the one that binds: that product is what the
    encoder allocates for. Measured on a 24 GB card over 300 clips, a 600 s call peaks at
    8.6 GB and a 900 s one at 13.0 GB, while a 1300 s one asks for a single 4.24 GB
    buffer and dies. Those peaks are a floor rather than a budget over a long run (see
    `DEFAULT_BATCH_SEC`), which is why `run_group` can still be handed a batch that does
    not fit. One rate per call, since `recognize()` takes one `sample_rate` for the whole
    batch.
    """
    group: list[tuple[Job, ClipInput]] = []
    longest = 0
    for item in prepared:
        prep = item[1]
        if group and (len(group) == batch
                      or prep.sr != group[0][1].sr
                      or (len(group) + 1) * max(longest, len(prep.wav)) > max_sec * prep.sr):
            yield group
            group, longest = [], 0
        group.append(item)
        longest = max(longest, len(prep.wav))
    if group:
        yield group


def run_group(model, group: list[tuple[Job, ClipInput]], ctx: Context) -> None:
    """Transcribe ONE batch and score its clips, halving the batch if it cannot allocate.

    The halving is what keeps a batch that does not fit from costing clips, and it is
    needed because the cap `batch_groups` enforces is sized from a peak that MOVES:
    onnxruntime's BFC arena never returns GPU memory to the driver, and a sweep hands it
    thousands of distinct padded shapes, so its high-water mark climbs for hours. At a
    900 s cap it reached 23.4 GB of a 24.6 GB card within 2500 clips and the widest
    batches then died on a single 1.47 GB pre-encode conv buffer. Before this, that error
    was caught one level up and charged every clip of the 192-clip window to `failed`.
    Retrying in halves instead costs a repeated encode of one batch's clips, and bottoms
    out at a single clip, where the blame really is the clip's.
    """
    t0 = time.monotonic()
    try:
        hyps = model.recognize([prep.wav for _, prep in group],
                               sample_rate=group[0][1].sr)
    except Exception as exc:                 # noqa: BLE001 - at >1 clip this is a batch
        if len(group) > 1:                   # size problem, so say so and shrink it
            logger.warning("batch of {} clip(s) failed ({}), retrying in halves",
                           len(group), exc)
            half = len(group) // 2
            run_group(model, group[:half], ctx)
            run_group(model, group[half:], ctx)
            return
        job = group[0][0]
        logger.exception("{}: transcription failed ({}), not checked", job.stem, exc)
        ctx.stats.bump("failed")
        return
    # Per clip, the batch's cost divided by the clips that shared it: the only reading of
    # `stt_s` that stays comparable to the per-clip path's, and the one that makes the
    # summary's rate line honest.
    stt_s = (time.monotonic() - t0) / len(group)
    for (job, prep), transcript in zip(group, hyps):
        try:
            score_clip(job, prep, transcript, stt_s, ctx.watches[job.stage], ctx)
        except Exception as exc:             # noqa: BLE001 - one bad clip must not end
            logger.exception("{}: scoring failed ({})", job.stem, exc)
            ctx.stats.bump("failed")         # the run


def batch_worker_loop(model, ctx: Context) -> None:
    """Drain the queues through BATCHED recognize() calls instead of one per clip.

    This is the whole reason a full-corpus sweep is hours rather than a day: the TDT
    decode runs in lockstep over a batch, one decoder_joint call per step for every clip
    at once, instead of per clip. Measured on an RTX 3090 Ti over 300 dictionary clips
    (fp16 encoder, fp32 decoder), 35.7 clips/s at this cap against 4.3 one at a time. The two
    details that carry most of it are in `batch_groups` (the padded-seconds cap) and in
    the sort below; `docs/PARAKEET_SWEEP_THROUGHPUT.md` has the measurements.

    Only ONE of these threads is started. A second would not help: the GPU sits around
    56% busy here, idling inside the Python-side decode loop, but two concurrent streams
    measured only 8% more throughput for 4 GB more VRAM, because without CUDA MPS the
    kernels of two contexts time-slice rather than overlap. `--workers` becomes the
    number of threads READING audio ahead of the GPU, and 4 already saturates it.
    """
    cfg = ctx.cfg
    # Sorting by length only pays over a window bigger than one batch, so clips are
    # popped several batches at a time: a batch then pads to a length close to its own.
    # Unsorted costs 20% (31.5 clips/s against 39.5) and 42% padding waste against 23%.
    window = max(1, cfg.batch) * BATCH_SORT_WINDOW
    with cf.ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        while not ctx.stop.is_set():
            jobs = ctx.next_batch(window)
            if not jobs:
                ctx.stop.wait(1.0)
                continue
            try:
                prepared = [(job, prep) for job, prep in
                            zip(jobs, pool.map(lambda j: prepare_clip(
                                j, ctx.watches[j.stage], ctx), jobs))
                            if prep is not None]
                prepared.sort(key=lambda item: len(item[1].wav))
                for group in batch_groups(prepared, cfg.batch, cfg.batch_sec):
                    run_group(model, group, ctx)
            except Exception as exc:               # noqa: BLE001 - nor one bad window
                logger.exception("window of {} clip(s) failed ({})", len(jobs), exc)
                for _ in jobs:
                    ctx.stats.bump("failed")
            finally:
                ctx.done(len(jobs))


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
                   "measured on this host; fp16 is ~2x slower on CPU. Pass an EMPTY "
                   "value for an export's unsuffixed files. On the GPU the measured "
                   "pick is `--quantization fp16 --decoder-quantization fp32`: the ultra "
                   "export's fp16/ folder ships an encoder only, so fp16 needs the "
                   "decoder named separately, and batched it is the fastest of the four.")
@click.option("--decoder-quantization", default=None,
              help="Precision for the fused decoder_joint, when it must differ from "
                   "--quantization. Omitted, the decoder follows the encoder (onnx-asr's "
                   "own behaviour). `fp32` means the unsuffixed files. Two reasons to "
                   "split them: a quantized encoder may ship without a matching decoder "
                   "(fp16, w4a8), and the int8 decoder derives one activation scale per "
                   "CALL, so with --batch a clip's transcript could depend on its batch "
                   "mates. fp32 is batch-invariant and the decoder is 18 MB.")
@click.option("--batch", default=1, show_default=True,
              help="Clips per recognize() call. 1 keeps the one-at-a-time path. Above "
                   "that, the TDT decode runs in lockstep over the batch and a single "
                   "thread does the transcribing: measured 38.5 clips/s at 64 against "
                   "4.3 at 1 (fp16 encoder, RTX 3090 Ti, 300 dictionary clips). The "
                   "padded-seconds cap below almost always binds first, so 64 is a "
                   "ceiling rather than the real batch size.")
@click.option("--batch-sec", default=DEFAULT_BATCH_SEC, show_default=True,
              help="Cap on a batch's PADDED seconds (clips x the longest clip in it), "
                   "which is what the encoder allocates for and the real VRAM knob: 900 "
                   "peaks at 13.0 GB of a 24 GB card, 1300 asks for a single 4.24 GB "
                   "buffer and dies. Lower it for a smaller card or a busier display.")
@click.option("--providers", "providers_csv", default="", show_default=False,
              help="Comma-separated onnxruntime execution providers, e.g. "
                   "CUDAExecutionProvider. Empty (the default) leaves onnxruntime on "
                   "the CPU, which is what this script is named for and what lets it "
                   "run beside a GPU job. The GPU is worth it for a whole-corpus pass: "
                   "one needs ~230 h of CPU. NOTE the wheel pinned above is CPU-only, "
                   "so a CUDA run needs `uv run --with onnxruntime-gpu ./03_qc_cpu_stt.py "
                   "--providers CUDAExecutionProvider --quantization fp16 ...`. A "
                   "provider onnxruntime does not have is a hard error here rather than "
                   "a silent fall back to the CPU, since the difference is 100x and a "
                   "pass that quietly took the slow path would look like a hung run.")
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
                   "take threads from the process feeding the GPU. With --batch above 1 "
                   "this is instead the number of threads READING audio ahead of the "
                   "GPU, where 4 already saturates it (4/6/8/12 measured 39.5/41.8/39.9/"
                   "40.1 clips/s): the transcribing itself stays one thread.")
@click.option("--nice", "nice_level", default=10, show_default=True,
              help="Niceness applied to this process, so the generator wins every "
                   "contested timeslice. 0 to leave the priority alone.")
@click.option("--poll-seconds", default=30.0, show_default=True,
              help="How often to look for new clips and new sidecar rows.")
@click.option("--max-queue", default=None, type=int,
              help="Cap on QUEUED samples; the oldest are dropped past it (counted in "
                   "the summary). Retries are never dropped. 0 means no cap. Default: "
                   "2000 when following a live run, no cap with --backfill, where the "
                   "queue is the known population on disk rather than a backlog.")
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
@click.option("--repeat-threshold", default=DEFAULT_REPEAT_GATE, show_default=True,
              help="Number of word windows the transcript may read more often than the "
                   "label writes them before the clip is flagged. This is the only gate "
                   "that can see a LOOP: Whisper deduplicates repeated speech, so the "
                   "four CER gates above are structurally blind to it. 0 disables.")
@click.option("--reference-key", default="asr_training_source", show_default=True,
              help="Manifest field holding the text the TTS was asked to say.")
@click.option("--stems-from", "stems_from", multiple=True,
              type=click.Path(path_type=Path),
              help="Check exactly the clips listed in this file, whatever --sample-percent "
                   "says: an acoustic sidecar from 04_qc_acoustic_loops.py (its `looped` "
                   "rows) or a plain list of stems. Repeatable, one per stage. Needs "
                   "--backfill to reach clips already on disk.")
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
         decoder_quantization: str | None, batch: int, batch_sec: float,
         providers_csv: str,
         sample_percent: float, do_retries: bool, threads: int, workers: int,
         nice_level: int, poll_seconds: float, max_queue: int | None,
         cer_threshold: float,
         tail_cer_threshold: float, window_cer_threshold: float,
         bad_run_threshold: int, repeat_threshold: int, reference_key: str,
         stems_from: tuple[Path, ...],
         backfill: bool, once: bool,
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

    if max_queue is None:
        # --backfill asks for a whole directory; --follow asks for whatever is arriving.
        # Only the second one has a backlog that a cap protects.
        max_queue = 0 if backfill else 2000
    cfg = SimpleNamespace(
        sample_percent=sample_percent, max_queue=max_queue, backfill=backfill,
        cer_threshold=cer_threshold, tail_cer_threshold=tail_cer_threshold,
        window_cer_threshold=window_cer_threshold, bad_run_threshold=bad_run_threshold,
        repeat_threshold=repeat_threshold,
        reference_key=reference_key, out_name=out_name, workers=workers,
        batch=batch, batch_sec=batch_sec,
        keep_transcripts=keep_transcripts, summary_every=summary_every,
        stem_requests=load_stem_requests(stems_from))

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

    requested = sum(len(v) for v in cfg.stem_requests.values())
    if requested and not backfill:
        # Without --backfill the run is a tail: everything on disk is marked seen and
        # never polled, so a list of already-generated clips would quietly check nothing.
        logger.warning("--stems-from given {} stem(s) but not --backfill: only clips "
                       "generated from now on can match", requested)
    providers = tuple(x.strip() for x in providers_csv.split(",") if x.strip())
    if providers:
        import onnxruntime as ort
        missing = [x for x in providers if x not in ort.get_available_providers()]
        if missing:
            raise click.BadParameter(
                f"--providers: onnxruntime {ort.__version__} has no {', '.join(missing)} "
                f"(it offers {', '.join(ort.get_available_providers())}). Install the "
                f"GPU wheel: uv run --with onnxruntime-gpu ...")
    logger.info("loading {} (encoder {}, decoder {}) with {} thread(s) x {} worker(s)"
                " on {}{}...",
                model_dir.name, quantization or "fp32",
                decoder_quantization or quantization or "fp32", threads, workers,
                ", ".join(providers) if providers else "the onnxruntime default (CPU)",
                f", batches of <={batch} clips / {batch_sec:g} padded s"
                if batch > 1 else "")
    t0 = time.monotonic()
    model = load_cpu_model(model_dir.resolve(), quantization=quantization,
                           threads=threads, providers=providers,
                           decoder_quantization=decoder_quantization)
    logger.info("model loaded in {:.1f}s; sample {}%, retries {}, gates {} / {} (tail) "
                "/ {} (window) / {} chars (missing run) / {} repeated windows",
                time.monotonic() - t0, sample_percent,
                "on" if do_retries else "off", cer_threshold, tail_cer_threshold,
                window_cer_threshold, bad_run_threshold,
                repeat_threshold if repeat_threshold > 0 else "off")

    def _stop(signum, _frame):
        logger.warning("signal {}, finishing the clip in flight and stopping", signum)
        ctx.stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # The model is shared by every worker on purpose: an onnxruntime session is
    # thread-safe for Run(), and a second copy of the encoder would cost another
    # 619 MB of RAM for nothing.
    #
    # Batching replaces the workers with ONE consumer that transcribes whole batches
    # (--workers becomes its loader threads). A second batch thread was measured to add
    # only 8% for 4 GB more VRAM, because the GPU's idle 44% sits inside the Python
    # decode loop and two CUDA contexts time-slice rather than overlap without MPS.
    if batch > 1:
        pool = [threading.Thread(target=batch_worker_loop, args=(model, ctx),
                                 daemon=True, name="qcbatch")]
    else:
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
