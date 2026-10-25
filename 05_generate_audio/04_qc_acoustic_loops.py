#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "soundfile>=0.12",
#   "click>=8.1",
#   "loguru>=0.7",
#   # Only because `03_qc_cpu_stt.py` is imported for its labels and stage table, and it
#   # in turn imports stage 06's scorer. Deliberately WITHOUT onnx-asr: nothing here
#   # transcribes, which is the whole point, and the QC's model loader is lazy.
#   "jiwer>=3.0",
#   "requests>=2.31",
#   "tqdm>=4.66",
# ]
# ///
"""Catch a take that says the same thing twice, from the WAVEFORM alone, no STT.

Why this exists: the STT QC (`03_qc_cpu_stt.py`) can only afford a sample of the
corpus, because transcribing one clip costs ~1.4 s of CPU and the corpus is 602,885
clips. Its `--defect-mix` says what that sample is catching, and after the loop
detector was fixed (2026-10-16) the answer is **63% `loop` overall and 90% of the
severe end**: takes that derail into repeating a clause. A repeated clause is the one
defect shape with a signature in the audio itself, so it does not need a recognizer,
which means it can be run over **every** clip instead of a quarter of them.

What it measures: a loop makes the same stretch of speech occur twice in one clip, so
the clip's self-similarity matrix grows a stripe parallel to its diagonal. This scores
that stripe directly in the lag domain (see `loop_score`): the best windowed cosine
similarity between frame t and frame t+lag, over every lag that could hold a repeat.
Nothing here understands words, so it is blind on purpose to `wrong_words`, the 36% of
the catch that is fluent speech saying the wrong thing, and near-blind to `truncated`
and `skipped` (no repeat, no stripe). Read it as a pre-filter that makes the
transcribing QC a second opinion rather than the first, NOT as a replacement.

Usage
-----
    # Score every clip of a stage that has no row yet, writing a sidecar beside the
    # audio (append-only, so it resumes by itself):
    ./04_qc_acoustic_loops.py --stage 01_dictionnary

    # Measure the gate instead of using it: join against the STT sidecar's shapes and
    # print the score distribution per shape plus a threshold sweep.
    ./04_qc_acoustic_loops.py --stage 01_dictionnary --calibrate

Calibration
-----------
Measured 2026-10-16 on 01_dictionnary, 1279 clips: every flagged STT row carrying both
texts (176 `loop`, 100 `wrong_words`, 2 `overrun`, 1 `skipped`, shapes recomputed by
`03_qc_cpu_stt.py::classify_defect`) plus 1000 clips the STT QC passed. Score
distributions barely overlap::

    clean        median 0.356   p90 0.464   max 0.883
    loop         p10 0.708   median 0.848   max 0.965

and the sweep (`--calibrate` prints it) reads::

    gate   recall on loop   fires on clean   fires on wrong_words
    0.65        94.9%            1.4%               34%
    0.70        92.0%            0.9%               31%
    0.75        86.4%            0.4%               26%
    0.80        72.2%            0.1%               19%

`DEFAULT_LOOP_GATE` is 0.70, which trades a 0.9% fire rate on good clips for keeping 92%
of the loops. That is the right side to err on here because a flag is not a verdict: the
flagged subset is small enough to confirm with the transcribing QC (0.9% of 602,885 clips
is ~5400 clips, ~2 h of one CPU core), whereas a loop missed on an unsampled clip is
missed for good.

Only the `clean` column is a false-positive rate. The `wrong_words` column is agreement,
not error: those clips are already flagged by CER, and a third of them repeat audio too
(a clause shorter than the 6 words the text-side detector counts from). And the handful
of high-scoring `clean` clips are the interesting ones: a clip this detector flags and
the STT QC never sampled is exactly what an exhaustive pre-filter is for, so listen
before calling it a false positive.

First flags, verified
---------------------
A 300-clip smoke scan of 01_dictionnary fired 5 times, and **none of those 5 clips had
ever been STT-sampled**, which is the whole argument for this script in one sentence.
Transcribing them (int8 model) confirmed one real loop, `000002_0002_a_beta_40`
(similarity 0.900, CER 0.169, window CER 0.282: "phosphotao a 80 picogrammes par
millilitres" said twice), on a clip the 25% sample was never going to look at. The other
4 (similarity 0.737-0.849, CER 0.005-0.029) are the detector working correctly on a text
that repeats itself: the LABEL says "picogrammes par millilitres" twice, so the audio
does too. Lab-value dictation is full of that, which is both where the loops concentrate
and where the false positives come from.

Rejected: gating on the label's own repeat
------------------------------------------
`label_repeat_seconds` is recorded per row (`label_rep_s`) but deliberately NOT part of
the gate. The idea looked strong: labels repeating a 3+ word phrase are 24% of looped
clips but only 4% of clean ones, and 9 of the 14 clean clips firing at 0.70 have such a
label. Measured, it does not pay. Requiring `stripe_s > label_rep_s` drops recall 92.0%
-> 83.6% to move the clean fire rate 0.90% -> 1.10% (the rate moves the *wrong* way
because dropping flags also drops the loop clips whose labels repeat), and a +0.3 s
margin gives 79.1% / 0.80%. Just raising the gate to 0.75 beats every variant on both
axes (86.4% / 0.40%), because a clip whose only repeat is in its label scores lower to
begin with. The clearest counter-example is the one confirmed defect above: its label
repeats 1.90 s against a 2.05 s stripe, so a +0.3 s margin would have suppressed the
single real loop this script has found. Hence: gate on similarity, keep `label_rep_s` as
the first thing to read when triaging a flag.

Re-gating costs nothing
-----------------------
Rows carry `similarity`, and `looped` is just `similarity >= gate` frozen at scan time.
Choosing a different operating point later is a re-read of the sidecar, not a re-scan, so
the gate is a view and the score is the measurement.

Written with Claude Code.
"""

from __future__ import annotations

import os

# Before numpy: every worker is a process, and numpy's FFT and BLAS would each open a
# thread pool per worker. With `--workers 4` that oversubscribes the machine against the
# GPU generator it is meant to run beside, and the OpenMP pools then busy-wait. Measured
# on one 20 s clip: 122 ms unpinned against 42 ms pinned, and 1.5 s against 216 ms on a
# 68 s one. Set only when the caller has not, so the choice stays theirs.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import click
import numpy as np
import soundfile as sf
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent


def _load_module(path: Path, name: str):
    """Import a sibling script whose filename starts with a digit.

    Same `spec_from_file_location` trick every script in this repo uses; see the copy
    in `03_qc_cpu_stt.py` for why there is no shared helper module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The STT QC is imported, not copied: `--calibrate` needs its `classify_defect` to know
# which clips are loops (the labels this detector is measured against), its stage table
# and its manifest index to name the voice behind a clip. A private copy of any of those
# would drift from the gate that actually judges the dataset.
_qc = _load_module(_HERE / "03_qc_cpu_stt.py", "qc_cpu_stt_mod")
# Its copy of the stage -> voice-mode table, reached through it rather than loaded a
# second time: loading `02_audit_durations.py` again under the same module name would
# replace the one the imported QC is already holding.
_audit = _qc._audit

sys.path.insert(0, str(ROOT / "utils"))
from nemo_manifest import parse_clip_name  # noqa: E402
from progress import bar_safe_sink, resumed_bar  # noqa: E402
from voice_assignment import voice_for_row  # noqa: E402

#: Where the per-clip acoustic rows land: beside the audio and the STT sidecar, one
#: Seconds a clip may be newer than its row before the row is called stale. Covers the
#: sub-second gap between statting a clip and writing its row, and filesystem mtime
#: granularity on a copy; a real redraw is minutes or days newer, never one second.
_MTIME_SLACK = 2.0

#: JSON line per clip, append-only so a killed run resumes from what is already there.
ACOUSTIC_FILENAME = "qc_acoustic_sidecar.jsonl"

#: Frame geometry, in seconds rather than samples because the generated audio is 24 kHz
#: but nothing here depends on that. The hop is deliberately coarse (32 ms, ~31 fps):
#: the two renditions of a looped clause are RE-GENERATED, not copied, so they differ in
#: duration by a few percent, and the stripe drifts off its diagonal by that much. A
#: coarse hop halves the drift measured in frames, which is what the stripe score sees.
#: The 64 ms window (2x the hop) keeps the usual overlap.
FRAME_SECONDS = 0.064
HOP_SECONDS = 0.032

#: Mel/MFCC front end. 40 bands over the speech band, then 12 cepstral coefficients
#: with **c0 dropped**: c0 is loudness, and two renditions of the same clause can differ
#: in level without being any less of a repeat. Per-clip mean/variance normalisation
#: then removes the voice's own timbre, so the score answers "does this clip repeat
#: itself" rather than "which voice is this".
MEL_BANDS = 40
MFCC_COEFFS = 12
MEL_FMIN = 80.0
MEL_FMAX = 8000.0

#: A repeat shorter than this is ordinary French (a drug name, "de la", a hesitation),
#: so the stripe has to be at least this long to count. 1.6 s is ~23 normalized chars at
#: the corpus's measured 14.2 chars/s, i.e. a short clause, and comfortably under the
#: ~2.5 s a 6-word clause takes (the length the text-side detector counts from).
MIN_STRIPE_SECONDS = 1.6

#: Lags below this are the diagonal's own neighbourhood: a sustained vowel or a pause
#: is self-similar at 200 ms and says nothing about looping.
MIN_LAG_SECONDS = 1.0

#: Lag resolution, in frames. Every lag is probed (1), not every other one: probing at
#: 64 ms misaligns a stripe by one frame whenever its true lag is skipped, and a
#: one-frame offset costs real similarity. The saving was not worth it either, because
#: the scan is dominated by per-lag Python overhead rather than by arithmetic, which is
#: why `loop_score` hoists its buffers out of the loop instead.
LAG_STEP = 1

#: Silence is self-similar, and a clip with two long pauses would score like a loop. A
#: frame counts as voiced when it is within this many dB of the clip's loudest frame,
#: and a candidate stripe must be this voiced in BOTH copies to count.
VOICED_FLOOR_DB = 35.0
MIN_VOICED_FRACTION = 0.8

#: Per-frame cosine a stripe must keep to still count as part of the same repeat. Used
#: only to measure how long the repeated span is, never to decide whether it is one.
STRIPE_FLOOR = 0.5

#: Score at or above which a clip is called a loop: 92% of known loops, 0.9% of clips
#: the STT QC passed. See the module docstring for the sweep this comes from, and
#: `--calibrate` to re-measure it on another stage or another voice roster.
DEFAULT_LOOP_GATE = 0.70


class LoopScore(NamedTuple):
    """One clip's best self-similar stripe.

    Attributes
    ----------
    similarity
        Mean cosine similarity over the best `MIN_STRIPE_SECONDS` window, between the
        clip and itself offset by `lag_seconds`. This is the gated quantity: 1.0 is the
        same audio twice, and unrelated speech from one voice sits far below it.
    stripe_seconds
        How much audio keeps repeating around that window (per-frame cosine above
        `STRIPE_FLOOR`). The interpretable number: "it said 2.8 s of speech twice".
    lag_seconds
        Distance between the two occurrences.
    start_seconds
        Where the first occurrence starts, so a listener can jump straight to it.
    frames
        Frames analysed, 0 when the clip was too short to score.
    """

    similarity: float
    stripe_seconds: float
    lag_seconds: float
    start_seconds: float
    frames: int


@lru_cache(maxsize=8)
def _mel_filterbank(sample_rate: int, n_fft: int) -> np.ndarray:
    """Triangular mel filterbank, `(MEL_BANDS, n_fft // 2 + 1)`.

    Cached because it only depends on the geometry, and every clip of a stage shares it.
    """
    def to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)

    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fmax = min(MEL_FMAX, sample_rate / 2.0)
    edges = to_hz(np.linspace(to_mel(MEL_FMIN), to_mel(fmax), MEL_BANDS + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    bank = np.zeros((MEL_BANDS, freqs.size), dtype=np.float32)
    for band in range(MEL_BANDS):
        low, mid, high = edges[band], edges[band + 1], edges[band + 2]
        rising = (freqs >= low) & (freqs <= mid)
        falling = (freqs > mid) & (freqs <= high)
        bank[band, rising] = (freqs[rising] - low) / max(mid - low, 1e-9)
        bank[band, falling] = (high - freqs[falling]) / max(high - mid, 1e-9)
    return bank


@lru_cache(maxsize=2)
def _dct_matrix() -> np.ndarray:
    """DCT-II basis for coefficients 1..`MFCC_COEFFS`, `(MFCC_COEFFS, MEL_BANDS)`.

    Coefficient 0 is not in the basis at all, which is how c0 (loudness) is dropped.
    """
    coeff = np.arange(1, MFCC_COEFFS + 1, dtype=np.float64)[:, None]
    band = np.arange(MEL_BANDS, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * coeff * (2.0 * band + 1.0) / (2.0 * MEL_BANDS))
    return (basis * np.sqrt(2.0 / MEL_BANDS)).astype(np.float32)


def frame_features(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm MFCC frames and per-frame level, ready for a cosine comparison.

    Parameters
    ----------
    samples
        Mono or multi-channel float samples; channels are averaged.
    sample_rate
        Sample rate of `samples`.

    Returns
    -------
    features : numpy.ndarray
        `(frames, MFCC_COEFFS)` float32, mean/variance normalized over the clip and then
        L2-normalized per frame, so a dot product between two rows IS their cosine.
    level_db : numpy.ndarray
        `(frames,)` float32 mel-band energy in dB, for the voiced test.
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    window_len = int(round(FRAME_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    if window_len <= 0 or hop <= 0 or samples.size < window_len:
        return np.zeros((0, MFCC_COEFFS), dtype=np.float32), np.zeros(0, dtype=np.float32)
    count = 1 + (samples.size - window_len) // hop
    # One strided gather instead of a Python loop over frames: (count, window_len).
    index = np.arange(window_len)[None, :] + hop * np.arange(count)[:, None]
    frames = samples[index] * np.hanning(window_len).astype(np.float32)
    n_fft = 1 << (window_len - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float32) ** 2
    mel = power @ _mel_filterbank(int(sample_rate), n_fft).T
    level_db = (10.0 * np.log10(mel.sum(axis=1) + 1e-10)).astype(np.float32)
    mfcc = np.log(mel + 1e-10) @ _dct_matrix().T
    mfcc -= mfcc.mean(axis=0, keepdims=True)
    mfcc /= np.maximum(mfcc.std(axis=0, keepdims=True), 1e-6)
    mfcc /= np.maximum(np.linalg.norm(mfcc, axis=1, keepdims=True), 1e-6)
    return mfcc.astype(np.float32), level_db


def _stripe_extent(similarity: np.ndarray, start: int, width: int) -> int:
    """Length in frames of the repeat around `[start, start + width)`.

    Grows the window while the per-frame cosine stays above `STRIPE_FLOOR`, which turns
    the fixed-width score into the honest "this much audio repeats".
    """
    left = start
    while left > 0 and similarity[left - 1] >= STRIPE_FLOOR:
        left -= 1
    right = min(start + width, similarity.size)
    while right < similarity.size and similarity[right] >= STRIPE_FLOOR:
        right += 1
    return right - left


def loop_score(samples: np.ndarray, sample_rate: int, *,
               min_stripe_seconds: float = MIN_STRIPE_SECONDS,
               min_lag_seconds: float = MIN_LAG_SECONDS) -> LoopScore:
    """Score how much a clip repeats itself, with no reference text.

    Scans the self-similarity in the LAG domain rather than building the full matrix: for
    each candidate lag it correlates the frame sequence with itself offset by that lag,
    box-averages over `min_stripe_seconds`, and keeps the best position. Cost is
    `O(frames^2 * MFCC_COEFFS / LAG_STEP)` arithmetic but only `O(frames)` memory, which
    matters because a 68 s clip (the corpus maximum) would otherwise hold a 2100x2100
    matrix per worker.

    Parameters
    ----------
    samples
        Audio samples, as `soundfile.read` returns them.
    sample_rate
        Sample rate of `samples`.
    min_stripe_seconds
        Width of the averaging window: the shortest repeat that should count.
    min_lag_seconds
        Smallest offset considered, to stay out of the diagonal's neighbourhood.

    Returns
    -------
    LoopScore
        All-zero (with `frames` set) when the clip is too short to hold a repeat at all.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> clip = rng.normal(size=4 * 24000).astype(np.float32) * 0.1
    >>> looped = np.concatenate([clip, clip])          # says itself twice
    >>> loop_score(looped, 24000).similarity > 0.9
    True
    >>> loop_score(clip, 24000).similarity < 0.9
    True
    """
    features, level_db = frame_features(samples, sample_rate)
    total = features.shape[0]
    width = max(int(round(min_stripe_seconds / HOP_SECONDS)), 1)
    min_lag = max(int(round(min_lag_seconds / HOP_SECONDS)), 1)
    if total < width + min_lag:
        return LoopScore(0.0, 0.0, 0.0, 0.0, total)

    # Which window starts are voiced enough to judge, computed ONCE: the test is about a
    # window of the clip, so the answer for the second copy is the same array read
    # `lag` later. Doing it per lag (two cumsums) cost more than the similarity itself.
    voiced = (level_db > level_db.max() - VOICED_FLOOR_DB).astype(np.float32)
    voiced_sum = np.concatenate(([0.0], np.cumsum(voiced)))
    starts = total - width + 1
    voiced_ok = ((voiced_sum[width: width + starts] - voiced_sum[:starts]) / width
                 >= MIN_VOICED_FRACTION)

    # Buffers hoisted out of the lag loop. The scan is thousands of iterations over a few
    # thousand elements each, so numpy's per-call allocation dominates: reusing one
    # cumulative-sum buffer and one means buffer is what makes this ~4x faster than the
    # obvious `concatenate`/`cumsum`/`where` per lag, with no change to the result.
    running = np.zeros(total + 1, dtype=np.float64)
    means = np.empty(total, dtype=np.float64)
    best_mean, best_lag, best_start = 0.0, 0, 0
    for lag in range(min_lag, total - width + 1, LAG_STEP):
        count = total - lag - width + 1
        if count <= 0:
            break
        # Per-frame cosine between the clip and itself offset by `lag`, then its box mean
        # over `width` frames, both written into the hoisted buffers.
        pairwise = np.einsum("ij,ij->i", features[: total - lag], features[lag:])
        np.cumsum(pairwise, out=running[1: pairwise.size + 1])
        view = means[:count]
        np.subtract(running[width: width + count], running[:count], out=view)
        view /= width
        # A candidate needs both copies voiced; -1 removes the rest from the argmax.
        view[~(voiced_ok[:count] & voiced_ok[lag: lag + count])] = -1.0
        start = int(view.argmax())
        if view[start] > best_mean:
            best_mean, best_lag, best_start = float(view[start]), lag, start

    if best_lag == 0:
        return LoopScore(0.0, 0.0, 0.0, 0.0, total)
    pairwise = np.einsum("ij,ij->i", features[: total - best_lag], features[best_lag:])
    extent = _stripe_extent(pairwise, best_start, width)
    return LoopScore(similarity=best_mean,
                     stripe_seconds=extent * HOP_SECONDS,
                     lag_seconds=best_lag * HOP_SECONDS,
                     start_seconds=best_start * HOP_SECONDS,
                     frames=total)


def score_path(path: Path) -> tuple[str, LoopScore | None, float, str | None]:
    """Score one clip. Returns `(stem, score, duration_seconds, clip_mtime, error)`.

    Errors are returned rather than raised because this runs in a process pool over a
    directory that the generator is still writing to: a clip can vanish or be half
    written between the listing and the read, and that must cost one skipped row, not
    the run.
    """
    stem = path.name[: -len(path.suffix)]
    # Stat BEFORE the read, not after: a redraw round can replace this clip while the
    # pool is scoring it, and an mtime taken afterwards would describe audio this row
    # never heard, which is exactly the staleness the row is meant to let a later run
    # detect. 0.0 on failure makes the row look infinitely old, so it is rescanned.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:                                        # noqa: BLE001
        return stem, None, 0.0, mtime, str(exc)
    duration = len(samples) / float(sample_rate or 1)
    return stem, loop_score(samples, int(sample_rate)), duration, mtime, None


def _rows_already_scored(path: Path, audio_dir: Path) -> set[str]:
    """Stems whose sidecar row still describes the clip on disk, so a restart is a resume.

    A torn final line (the run was killed mid-write) is skipped, same as everywhere else
    in this repo's sidecar readers, so that clip is simply scored again.

    A row is also treated as absent when its clip has been REWRITTEN since it was
    scored. That case is not hypothetical: a repair round regenerates a clip while
    keeping its stem, so without this the clip keeps its pre-redraw verdict forever and
    the scan reports on audio that no longer exists. It went unnoticed once already
    (2026-10-21's scan against the T3 redraws: 9,829 of 22,710 `looped` flags described
    replaced audio), and it is silent by nature, because a stale row looks exactly like
    a fresh one. Rows carry the clip's mtime for the comparison; rows written before
    that field existed fall back to the sidecar's own mtime, the latest moment they
    could have been written, which is why that is read BEFORE this run appends to it.
    """
    seen: set[str] = set()
    if not path.is_file():
        return seen
    legacy_scored_at = path.stat().st_mtime
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        stem = row.get("stem")
        if not stem:
            continue
        try:
            clip_mtime = (audio_dir / f"{stem}.flac").stat().st_mtime
        except OSError:
            seen.add(stem)          # clip gone: nothing left to rescore
            continue
        scored_at = row.get("mtime")
        if not isinstance(scored_at, (int, float)):
            scored_at = legacy_scored_at
        if clip_mtime > scored_at + _MTIME_SLACK:
            continue                # audio replaced since: score it again
        seen.add(stem)
    return seen


def label_repeat_seconds(text: str | None, *, min_words: int = 3) -> float:
    """Seconds of speech the label's OWN longest repeated phrase accounts for.

    A loop is a stripe the model added. A label that says "picogrammes par millilitres"
    twice puts the same stripe in a perfectly good take, and that is what every false
    positive of the first smoke scan turned out to be. This number is the stripe such a
    label can explain on its own, so a reader looking at a flag whose `stripe_s` is no
    longer than its `label_rep_s` knows what to suspect first. It is recorded, not gated
    on: the module docstring has the measurement that rejected gating on it.

    The phrase is measured in normalized characters and divided by the corpus' speaking
    rate, reusing stage 06's `NORM_CHARS_PER_SECOND` (14.2) so that this agrees with the
    chars-to-seconds conversion the STT QC's window gate already uses.

    Parameters
    ----------
    text
        The label, raw. None or empty gives 0.0.
    min_words
        Shortest phrase that counts as a repeat. Pairs recur constantly in French ("par
        jour", "de la") without ever accounting for a 1.6 s stripe.

    Returns
    -------
    float
        Seconds, or 0.0 when no phrase of `min_words` words or more occurs twice.

    Examples
    --------
    >>> label_repeat_seconds("phosphotao a 80 picogrammes par millilitres et "
    ...                      "calcium a 2 picogrammes par millilitres") > 1.0
    True
    >>> label_repeat_seconds("la patiente recoit 2 grammes par jour")
    0.0
    """
    words = _qc._stt.normalize_for_scoring(text or "").split()
    # Longest first, so the answer is the longest repeat rather than any repeat.
    for size in range(len(words) // 2, min_words - 1, -1):
        seen: set[tuple[str, ...]] = set()
        for start in range(len(words) - size + 1):
            gram = tuple(words[start: start + size])
            if gram in seen:
                return len(" ".join(gram)) / _qc._stt.NORM_CHARS_PER_SECOND
            seen.add(gram)
    return 0.0


def _voice_and_label(watch, stem: str) -> tuple[str | None, float]:
    """Voice behind a clip and its label's own repeat length, via the STT QC's index.

    Both come out of the same manifest row, so they are fetched in one lookup: this runs
    once per clip over the whole corpus.
    """
    row = _qc.manifest_row(watch, stem)
    if row is None:
        return None, 0.0
    voice = voice_for_row(row, field=watch.voice_field, mode=watch.mode)
    # Empty key = stage 06's fallback chain, so this follows whatever field the manifest
    # actually carries instead of hardcoding `asr_training_source` a second time.
    return voice, label_repeat_seconds(_qc._stt.get_reference(row, ""))


def scan_stage(watch, *, gate: float, workers: int, limit: int,
               out_name: str, log_every: int) -> tuple[int, int]:
    """Score every unscored clip of one stage. Returns `(scored, flagged)`."""
    out_path = watch.audio / out_name
    done = _rows_already_scored(out_path, watch.audio)
    try:
        names = sorted(n for n in os.listdir(watch.audio) if parse_clip_name(n))
    except OSError as exc:
        logger.warning("{}: cannot list {} ({})", watch.name, watch.audio, exc)
        return 0, 0
    todo = [watch.audio / n for n in names if n[: -len(".flac")] not in done]
    if limit:
        todo = todo[:limit]
    logger.info("{}: {} clip(s) on disk, {} already scored, {} to score",
                watch.name, len(names), len(done), len(todo))
    if not todo:
        return 0, 0

    _qc.ensure_index(watch)
    scored = flagged = 0
    started = time.monotonic()
    # The bar counts the WHOLE stage, not just this run's share, so a resumed scan reads
    # 44% rather than 0%. `len(names)` is every clip on disk; `todo` is what is left,
    # after both the sidecar and --limit.
    with out_path.open("a") as sink, ProcessPoolExecutor(max_workers=workers) as pool, \
         resumed_bar(total=len(done) + len(todo), remaining=len(todo),
                     desc=f"{watch.name} loops", smoothing=0.02) as bar:
        for stem, score, duration, mtime, error in pool.map(score_path, todo,
                                                           chunksize=8):
            bar.update(1)
            if error is not None or score is None:
                logger.warning("{}: unreadable ({}), not scored", stem, error)
                continue
            scored += 1
            voice, label_rep = _voice_and_label(watch, stem)
            looped = score.similarity >= gate
            flagged += looped
            # refresh=False: the bar's own tick draws it, so this costs nothing per clip.
            bar.set_postfix(flagged=f"{flagged} ({100.0 * flagged / scored:.2f}%)",
                            refresh=False)
            sink.write(json.dumps({
                "stem": stem, "stage": watch.name, "voice": voice,
                "duration": round(duration, 2),
                "similarity": round(score.similarity, 4),
                "stripe_s": round(score.stripe_seconds, 2),
                "lag_s": round(score.lag_seconds, 2),
                "start_s": round(score.start_seconds, 2),
                "label_rep_s": round(label_rep, 2),
                "looped": bool(looped),
                # The clip this verdict actually describes, so a later run can tell that
                # a redraw has replaced it. See `_rows_already_scored`.
                "mtime": round(mtime, 3),
            }, ensure_ascii=False) + "\n")
            if looped:
                sink.flush()                 # a flag is worth having on disk right away
                logger.warning("{} [{}]: similarity {:.3f}, {:.1f}s of audio repeated "
                               "{:.1f}s apart, first at {:.1f}s (clip {:.1f}s){}",
                               stem, voice or "?", score.similarity,
                               score.stripe_seconds, score.lag_seconds,
                               score.start_seconds, duration,
                               # Say it on the flag line, not only in the row: it is the
                               # difference between "listen to this" and "read the label".
                               f", but the label repeats {label_rep:.1f}s itself"
                               if label_rep else "")
            if log_every and scored % log_every == 0:
                rate = scored / max(time.monotonic() - started, 1e-9)
                logger.info("{}: {}/{} scored, {} flagged ({:.2f}%), {:.1f} clips/s",
                            watch.name, scored, len(todo), flagged,
                            100.0 * flagged / scored, rate)
    return scored, flagged


def _percentiles(values: list[float]) -> str:
    """Compact p10/median/p90/max line for a score distribution."""
    if not values:
        return "n=0"
    array = np.asarray(values, dtype=np.float64)
    return (f"n={array.size:5d}  p10 {np.percentile(array, 10):.3f}  "
            f"median {np.median(array):.3f}  p90 {np.percentile(array, 90):.3f}  "
            f"max {array.max():.3f}")


def calibrate_stage(watch, *, workers: int, clean_limit: int, out_name: str) -> int:
    """Print the score distribution per STT-defined shape, and a threshold sweep.

    The labels are the text side's: every flagged row of the STT sidecar that carries
    both texts is classified with `03_qc_cpu_stt.py::classify_defect` (recomputed, not
    read from the row, for the reason that function's own docstring gives), and the clean
    rows are a slice of the CER-passing ones. So this answers the only question that
    matters about a pre-filter: at a threshold that almost never fires on a clip the STT
    QC was happy with, how much of the loop catch does it keep?

    Returns the number of clips scored.
    """
    stt_path = watch.audio / _qc.QC_FILENAME
    if not stt_path.is_file():
        logger.warning("{}: no {} to calibrate against", watch.name, stt_path.name)
        return 0
    labels: dict[str, str] = {}
    clean: list[str] = []
    for line in stt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        stem = row.get("stem")
        if not stem:
            continue
        if not row.get("over_gate"):
            clean.append(stem)
            continue
        if row.get("reference") and row.get("transcript"):
            labels[stem] = _qc.classify_defect(row["reference"], row["transcript"],
                                               bad_run_chars=row.get("bad_run_chars"))
    # Deterministic, spread-out slice of the clean clips rather than the first N, which
    # would all come from the start of the alphabet and so from the same few terms.
    clean = sorted(set(clean) - set(labels))
    if clean_limit and len(clean) > clean_limit:
        stride = len(clean) / clean_limit
        clean = [clean[int(i * stride)] for i in range(clean_limit)]
    for stem in clean:
        labels[stem] = "clean"
    paths = [watch.audio / f"{stem}.flac" for stem in labels]
    paths = [p for p in paths if p.is_file()]
    logger.info("{}: calibrating on {} clip(s) ({})", watch.name, len(paths),
                ", ".join(f"{k} {v}" for k, v in
                          Counter(labels[p.name[: -len('.flac')]] for p in paths).items()))

    by_shape: dict[str, list[float]] = {}
    rows: list[tuple[str, str, LoopScore]] = []
    started = time.monotonic()
    # Nothing is resumable here (a calibration always scores its whole sample), so the
    # bar starts at zero: same helper, `remaining` equal to `total`.
    with ProcessPoolExecutor(max_workers=workers) as pool, \
         resumed_bar(total=len(paths), remaining=len(paths),
                     desc=f"{watch.name} calibrate") as bar:
        for stem, score, _duration, _mtime, error in pool.map(score_path, paths,
                                                               chunksize=8):
            bar.update(1)
            if error is not None or score is None:
                continue
            shape = labels.get(stem, "?")
            by_shape.setdefault(shape, []).append(score.similarity)
            rows.append((stem, shape, score))
    elapsed = time.monotonic() - started
    logger.info("scored {} clip(s) in {:.1f}s = {:.1f} clips/s on {} worker(s)",
                len(rows), elapsed, len(rows) / max(elapsed, 1e-9), workers)

    for shape in sorted(by_shape):
        logger.info("  {:12} {}", shape, _percentiles(by_shape[shape]))

    loops = by_shape.get("loop", [])
    negatives = {k: v for k, v in by_shape.items() if k != "loop"}
    # Only the `clean` column is a false-positive rate; the others are agreement on
    # clips the CER gates already flagged, which is why they are printed separately.
    logger.info("  threshold sweep (recall on loop | fire rate on each other shape, "
                "clean = false positives)")
    for gate in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        recall = sum(s >= gate for s in loops) / max(len(loops), 1)
        others = "  ".join(
            f"{shape} {100 * sum(s >= gate for s in scores) / max(len(scores), 1):.1f}%"
            for shape, scores in sorted(negatives.items()))
        logger.info("    {:.2f}: recall {:.1f}%  |  {}", gate, 100 * recall, others)

    # The clips a reader should listen to first: the loudest stripes among the clips the
    # STT QC never flagged, because those are either this detector's false positives or
    # defects the 25% sample missed, and they cannot be told apart from the numbers.
    unseen = sorted((r for r in rows if r[1] == "clean"),
                    key=lambda r: -r[2].similarity)[:5]
    _qc.ensure_index(watch)
    for stem, _shape, score in unseen:
        _voice, label_rep = _voice_and_label(watch, stem)
        logger.info("  clean clip, top stripe: {} similarity {:.3f}, {:.1f}s repeated "
                    "{:.1f}s apart, first at {:.1f}s, label repeats {:.1f}s",
                    stem, score.similarity, score.stripe_seconds, score.lag_seconds,
                    score.start_seconds, label_rep)
    return len(rows)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stage", "stages", multiple=True, type=str,
              help="Stage directory name, repeatable (default: all of "
                   f"{', '.join(_audit.STAGES)}).")
@click.option("--workers", default=4, show_default=True,
              help="Worker processes. The GPU generator needs CPU too, so leave some.")
@click.option("--gate", default=DEFAULT_LOOP_GATE, show_default=True,
              help="Similarity at or above which a clip is called a loop.")
@click.option("--limit", default=0,
              help="Score at most N clips per stage (0 = all). For a quick look.")
@click.option("--calibrate", is_flag=True,
              help="Measure the gate against the STT sidecar's shapes instead of "
                   "scoring new clips.")
@click.option("--clean-limit", default=1000, show_default=True,
              help="With --calibrate, how many CER-passing clips to score as negatives.")
@click.option("--nice", "nice_level", default=10, show_default=True,
              help="Renice, since this is meant to run beside the generator.")
@click.option("--log-every", default=500, show_default=True,
              help="Progress line every N clips (0 to silence).")
@click.option("--out-name", default=ACOUSTIC_FILENAME, show_default=True,
              help="Sidecar filename, written beside the audio.")
@click.option("--log-file", default="qc_acoustic_loops.log",
              help="Log file beside this script ('' for stderr only).")
def main(stages: tuple[str, ...], workers: int, gate: float, limit: int, calibrate: bool,
         clean_limit: int, nice_level: int, log_every: int, out_name: str,
         log_file: str) -> None:
    """Score clips for self-similarity and flag the ones that repeat themselves."""
    logger.remove()
    # Through tqdm.write, on stderr, so a flag line does not tear the bar it prints over.
    bar_safe_sink(logger, fmt="{time:HH:mm:ss} | {level} | {message}", stream=sys.stderr)
    if log_file:
        path = Path(log_file)
        logger.add(path if path.is_absolute() else _HERE / path,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
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

    total_scored = total_flagged = 0
    for name in names:
        stage_dir = ROOT / name
        jsonl = stage_dir / "generated_dataset.jsonl"
        audio = stage_dir / "generated_audios"
        if not jsonl.is_file() or not audio.exists():
            logger.warning("{}: no generated_dataset.jsonl or generated_audios, skipped",
                           name)
            continue
        watch = _qc.StageWatch(name=name, mode=_audit.STAGES[name], jsonl=jsonl,
                               audio=audio.resolve())
        if calibrate:
            total_scored += calibrate_stage(watch, workers=workers,
                                            clean_limit=clean_limit, out_name=out_name)
            continue
        scored, flagged = scan_stage(watch, gate=gate, workers=workers, limit=limit,
                                     out_name=out_name, log_every=log_every)
        total_scored += scored
        total_flagged += flagged
    if not calibrate:
        logger.info("done: {} clip(s) scored, {} flagged ({:.2f}%)", total_scored,
                    total_flagged, 100.0 * total_flagged / max(total_scored, 1))


if __name__ == "__main__":
    main()
