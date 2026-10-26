#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "soundfile>=0.12",
# ]
# ///
"""What happens to the OTHER judges' readings when a draw is promoted?

A row carries one reading per judge (`whisper.cpp/ggml-large-v3-turbo`,
`onnx-asr/parakeet-tdt-0.6b-v3-ultra`, the non-turbo second opinion, the v1 `whisper-1`),
and a promotion OVERWRITES the clip. Every reading but the promoting run's then describes
bytes that no longer exist, and nothing else would ever notice: a sweep keyed on another
model skips a row that already has its key (`needs_work`), `03_sync_hotfix_results.py`
copies the turbo reading into the release columns as if it were current, and
`06_hotfixes/04_flag_asr_defects.py` reads the parakeet one to decide which clips ship.

That is not hypothetical twice over. The committed `asr_flagged_clips.jsonl` held 63
verdicts measured on audio v2 had already replaced (60 of them would have dropped a good
clip), and the acoustic scan of 2026-10-21 resumed on stem alone and carried 9,829 flags
about replaced takes (see `utils/clip_freshness.py`). A reading cannot defend itself with
that module's mtime rule either: a state-file reading carries no `ts`, so `is_stale` calls
all of them stale. So `_apply_improvement` drops them, and both halves are pinned here:
dropped when the audio really was replaced, kept when it was not.

The harness (module loader, fakes, cfg, one-row pass runner) is imported from
tests/test_recursive_improvement_modes.py rather than copied, the way test_voice_swap.py
does it.

    uv run tests/test_stale_judge_readings.py

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_recursive_improvement_modes as harness  # noqa: E402

TARGET = harness.TARGET
#: The two readings that must not survive a promotion. Spelled with the real keys so the
#: test says which consumers are at stake: parakeet feeds 04_flag_asr_defects.py, and
#: `whisper-1` is a v1 reading that can never be remade (the remote judge answers 502),
#: which is exactly the kind of row one is tempted to keep.
PARAKEET = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
LEGACY = "whisper-1"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def make_row() -> dict:
    """A bad clip that two other judges have already read."""
    return {
        "audio_filepath": "clip.flac",
        "asr_training_source": TARGET,
        "asr_training_target": TARGET,
        "index": 7,
        "term": "paracetamol",
        "transcriptions": {
            PARAKEET: {"text": TARGET, "cer": 0.0049, "cer_window": 0.01,
                       "bad_run_chars": 1, "repeat_excess": 0,
                       "stt_seconds": 0.4, "stt_rtf": 0.02},
            LEGACY: {"text": TARGET, "cer": 0.012, "wer": 0.03,
                     "stt_seconds": 1.1, "stt_rtf": 0.05},
        },
    }


def judges(row: dict) -> list[str]:
    return sorted(row.get("transcriptions") or {})


def drive(mod, root: Path, row: dict, modes, *, min_improvement: float) -> None:
    """Run the real stt -> tts -> stt alternation over one row."""
    input_file = root / "manifest.jsonl"
    improved_file = root / "manifest.improved.jsonl"
    improved: dict = {}
    for mode in modes:
        cfg = harness.make_cfg(mode, root, min_improvement=min_improvement)
        harness.run_pass(mod, cfg, row, improved, input_file, improved_file)


def test_a_promotion_drops_the_other_judges() -> None:
    print("\na promoted draw drops the readings of the audio it replaced")
    mod = harness.load_module()
    harness.install_fakes(mod)          # harness.GOOD_SEED reads as the target
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clip = root / "clip.flac"
        clip.write_bytes(b"ORIG")
        row = make_row()

        # Pass 1 (stt): the clip is FLAGGED, not touched. A flag is a verdict about the
        # audio on disk, so the other judges still describe it and must survive.
        drive(mod, root, row, ("stt",), min_improvement=0.0)
        check("the flag leaves the other judges alone",
              judges(row) == sorted([PARAKEET, LEGACY, "default"]), str(judges(row)))
        check("and the clip is queued", row["improvement"]["status"] == "pending_tts",
              json.dumps(row["improvement"]))

        # Pass 2 (tts): candidates are written BESIDE the clip, which is still the
        # original, so the readings are still current here too.
        drive(mod, root, row, ("tts",), min_improvement=0.0)
        check("drawing candidates leaves them alone too",
              judges(row) == sorted([PARAKEET, LEGACY, "default"]), str(judges(row)))
        check("and the original audio is still on disk", clip.read_bytes() == b"ORIG")

        # Pass 3 (stt): a draw wins and the clip is overwritten. Now they are lies.
        drive(mod, root, row, ("stt",), min_improvement=0.0)
        check("the win really replaced the audio",
              clip.read_bytes() == f"AUD{harness.GOOD_SEED}".encode(), str(clip.read_bytes()))
        check("and the promoting judge's reading is the new transcript",
              row["transcriptions"].get("default", {}).get("text") == TARGET,
              json.dumps(row["transcriptions"].get("default")))
        check("while every other judge's reading is gone", judges(row) == ["default"],
              str(judges(row)))


def test_a_clip_that_is_not_replaced_keeps_them() -> None:
    """The other half: dropping is tied to the OVERWRITE, not to being judged bad.

    An exhausted clip ships its original audio, so its parakeet and whisper-1 readings
    are still true and dropping them would throw away work no GPU can cheaply redo (and
    in `whisper-1`'s case, cannot redo at all).
    """
    print("\na clip whose audio was never replaced keeps its other judges")
    mod = harness.load_module()
    harness.install_fakes(mod, transcripts={})   # no draw reads well
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        clip = root / "clip.flac"
        clip.write_bytes(b"ORIG")
        row = make_row()

        # min_improvement 1.0 so no draw can be promoted on a gain either: the promotion
        # test is `best.cer <= orig.cer - min_improvement`, and at 0.0 two equally bad
        # readings satisfy it, which would overwrite the clip and prove nothing here.
        drive(mod, root, row, ("stt", "tts", "stt"), min_improvement=1.0)
        check("the clip exhausted rather than winning",
              row["improvement"]["status"] == "exhausted", json.dumps(row["improvement"]))
        check("its original audio is untouched", clip.read_bytes() == b"ORIG")
        check("so the other judges' readings are kept",
              judges(row) == sorted([PARAKEET, LEGACY, "default"]), str(judges(row)))


def main() -> int:
    test_a_promotion_drops_the_other_judges()
    test_a_clip_that_is_not_replaced_keeps_them()
    if failures:
        print(f"\nFAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
