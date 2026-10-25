#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "soundfile>=0.12",
#   "jiwer>=3.0",
#   "requests>=2.31",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""The acoustic loop pre-filter (`05_generate_audio/04_qc_acoustic_loops.py`).

What is worth testing here is not "does it find loops in this corpus" (that question is
answered by `--calibrate` against the STT QC's own labels, and the answer is in the
script's docstring), but that the detector measures what it claims to measure on signals
whose answer is known by construction:

- a clip that literally contains the same stretch of audio twice scores near 1, and the
  reported lag and stripe length are the ones that were planted;
- a clip with no repeat scores far below the gate, which is what keeps the false-positive
  rate payable over 602,885 clips;
- two long silences are not a repeat, since a corpus of read sentences is full of pauses
  and a detector that fired on them would be useless;
- a repeat whose second rendition runs a few percent faster is still found. This one
  guards a design decision rather than a behaviour: the two renditions of a looped clause
  are re-generated, not copied, so the stripe drifts off its diagonal, and the coarse
  32 ms hop exists to absorb that drift.

The synthetic signal is band-limited noise bursts at changing centre frequencies rather
than white noise, because white noise has a flat envelope in every frame and would make
the "no repeat" case trivially easy, hiding a detector that only works on toys.

uv run tests/test_acoustic_loops.py

This file was written by Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 24000


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ac = _load(ROOT / "05_generate_audio" / "04_qc_acoustic_loops.py", "acoustic_loops")


CHUNK = SAMPLE_RATE // 10          # 100 ms, the unit every clip below is built from


def _babble(chunks_count: int, rng: np.random.Generator) -> np.ndarray:
    """Speech-like filler: `chunks_count` bursts of 100 ms noise around a changing centre.

    Not speech, but it shares the property that matters: consecutive frames differ in
    spectral shape, so frame-to-frame similarity is low unless the audio really repeats.

    The length is given in whole 100 ms chunks rather than in seconds on purpose. A
    seconds argument would go through `int(seconds / 0.1)`, and `int(2.4 / 0.1)` is 23,
    not 24: a clip meant to plant a 6.0 s lag would really plant 5.9 s and the test would
    accuse the detector of being 3 frames off when it was exactly right.
    """
    hop = CHUNK
    chunks = []
    for _ in range(max(chunks_count, 1)):
        centre = rng.uniform(200.0, 3000.0)
        time = np.arange(hop) / SAMPLE_RATE
        carrier = np.sin(2 * np.pi * centre * time)
        envelope = 0.5 + 0.5 * rng.random()
        noise = rng.normal(scale=0.2, size=hop)
        chunks.append(((carrier + noise) * envelope).astype(np.float32))
    return np.concatenate(chunks)


def test_a_planted_repeat_is_found_with_its_lag_and_length() -> None:
    """The same 2.4 s of audio twice, 6.0 s apart, read back as exactly that."""
    rng = np.random.default_rng(7)
    clause = _babble(24, rng)                       # 2.4 s
    head = _babble(30, rng)                         # 3.0 s before the first copy
    gap = _babble(36, rng)                          # 3.6 s between the two copies
    clip = np.concatenate([head, clause, gap, clause, _babble(30, rng)])
    score = ac.loop_score(clip, SAMPLE_RATE)
    # Not 1.0: the frames straddling each copy's edges mix clause with filler, and the
    # per-clip CMVN means a frame's features depend slightly on the rest of the clip.
    assert score.similarity > 0.9, score
    # The lag is exact to the frame grid (32 ms), the only resolution it claims.
    true_lag = (clause.size + gap.size) / SAMPLE_RATE
    assert abs(score.lag_seconds - true_lag) <= 2 * ac.HOP_SECONDS, (score, true_lag)
    # The stripe is the clause, give or take the frames straddling its edges.
    assert abs(score.stripe_seconds - clause.size / SAMPLE_RATE) < 0.2, score
    # `start` is the best 1.6 s box inside the clause, so anywhere in that slack is right.
    first_copy = head.size / SAMPLE_RATE
    latest = first_copy + clause.size / SAMPLE_RATE - ac.MIN_STRIPE_SECONDS
    assert first_copy - 0.1 <= score.start_seconds <= latest + 0.1, score


def test_a_clip_without_a_repeat_stays_far_below_the_gate() -> None:
    """Nothing repeats, so nothing is flagged. The false-positive side of the gate."""
    rng = np.random.default_rng(11)
    score = ac.loop_score(_babble(200, rng), SAMPLE_RATE)
    assert score.similarity < ac.DEFAULT_LOOP_GATE, score


def test_two_silences_are_not_a_repeat() -> None:
    """Pauses repeat in every read sentence; only voiced spans count as a loop.

    Built so the ONLY repeat is the silence: two different babble spans separated by
    identical 3 s silences. Without the voiced test this scores near 1.
    """
    rng = np.random.default_rng(13)
    quiet = np.zeros(30 * CHUNK, dtype=np.float32)          # 3 s of nothing, twice
    clip = np.concatenate([_babble(20, rng), quiet, _babble(20, rng), quiet,
                           _babble(20, rng)])
    score = ac.loop_score(clip, SAMPLE_RATE)
    assert score.similarity < ac.DEFAULT_LOOP_GATE, score


def test_a_repeat_said_slightly_faster_is_still_found() -> None:
    """The second rendition runs 4% faster, as a re-generated clause would.

    This is why the hop is 32 ms rather than the usual 10: at 4% drift over a 2.4 s
    clause the two copies end ~96 ms apart, which is 3 frames here and 10 at a 10 ms hop.
    """
    rng = np.random.default_rng(17)
    clause = _babble(24, rng)
    faster = np.interp(np.arange(0, clause.size, 1.04), np.arange(clause.size),
                       clause).astype(np.float32)
    clip = np.concatenate([_babble(30, rng), clause, _babble(30, rng), faster,
                           _babble(20, rng)])
    score = ac.loop_score(clip, SAMPLE_RATE)
    assert score.similarity >= ac.DEFAULT_LOOP_GATE, score


def test_a_clip_too_short_to_hold_a_repeat_scores_zero() -> None:
    """Shorter than one stripe plus the minimum lag: no verdict, not a crash."""
    rng = np.random.default_rng(19)
    score = ac.loop_score(_babble(15, rng), SAMPLE_RATE)
    assert score.similarity == 0.0 and score.lag_seconds == 0.0, score
    assert score.frames > 0, score          # it DID analyse the clip
    # An empty clip is the degenerate case of the same thing.
    empty = ac.loop_score(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    assert empty.frames == 0 and empty.similarity == 0.0, empty


def test_features_are_unit_norm_and_level_is_reported() -> None:
    """A frame is a unit vector, so a dot product between two of them IS their cosine."""
    rng = np.random.default_rng(23)
    features, level_db = ac.frame_features(_babble(20, rng), SAMPLE_RATE)
    assert features.shape[0] == level_db.shape[0] > 10
    assert features.shape[1] == ac.MFCC_COEFFS
    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), norms[:5]
    # Silence is quieter than babble by far more than the voiced floor.
    _, quiet_db = ac.frame_features(np.zeros(20 * CHUNK, dtype=np.float32), SAMPLE_RATE)
    assert quiet_db.max() < level_db.max() - ac.VOICED_FLOOR_DB


def test_a_replaced_clip_is_scored_again() -> None:
    """A row whose clip has been rewritten since must not count as already scored.

    This is the bug that made 9,829 of the 2026-10-21 scan's 22,710 `looped` flags
    describe audio that no longer existed: a repair round regenerates a clip while
    keeping its stem, the resume set matched on the stem alone, and the clip kept its
    pre-redraw verdict forever. It is silent by nature, since a stale row looks exactly
    like a fresh one, so it earns a test rather than a comment.

    Four clips, one per case: scored after its last write (keep), rewritten since
    (rescan), a legacy row carrying no `mtime` at all and so judged against the
    sidecar's own mtime (rescan, the clip being newer than the file holding the row),
    and a stem whose clip is gone (keep, there is nothing left to score).
    """
    mod = _load(ROOT / "05_generate_audio/04_qc_acoustic_loops.py", "qc_acoustic_loops")
    with tempfile.TemporaryDirectory() as d:
        audio = Path(d)
        now = 1_700_000_000.0
        rows = []
        for stem, clip_mtime, recorded in (
            ("fresh",    now - 600.0, now - 300.0),  # scored after its last write
            ("redrawn",  now + 600.0, now - 300.0),  # rewritten after it was scored
            ("legacy",   now + 600.0, None),         # no mtime field: fall back
            ("vanished", None,        now - 300.0),  # clip deleted since
        ):
            row = {"stem": stem, "stage": "01_dictionnary", "looped": True}
            if recorded is not None:
                row["mtime"] = recorded
            rows.append(json.dumps(row))
            if clip_mtime is not None:
                clip = audio / f"{stem}.flac"
                clip.write_bytes(b"not really a flac, never read by this function")
                os.utime(clip, (clip_mtime, clip_mtime))
        sidecar = audio / "qc_acoustic_sidecar.jsonl"
        sidecar.write_text("\n".join(rows) + "\n", encoding="utf-8")
        # The fallback for a legacy row is the sidecar's OWN mtime, so it has to be set
        # explicitly here rather than left as "whenever this test happened to run".
        os.utime(sidecar, (now, now))
        done = mod._rows_already_scored(sidecar, audio)

    assert "fresh" in done, "a clip older than its row must not be scored twice"
    assert "redrawn" not in done, "a clip rewritten since its row must be scored again"
    assert "legacy" not in done, \
        "a pre-mtime row must be judged against the sidecar's own mtime"
    assert "vanished" in done, "a deleted clip has nothing left to score"


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall acoustic loop tests passed")


if __name__ == "__main__":
    main()
