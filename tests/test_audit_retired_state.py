#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "soundfile", "numpy", "tqdm"]
# ///
"""`05_generate_audio/02_audit_durations.py --state` must judge a clip by its READER.

Stage 05 draws a clip's voice by hash, but stage 06 rewrites clips in place with another
voice and records that reader in the `voice` field of its state file
(`06_hotfixes/improved/full.stt.jsonl`). The audit used to attribute every clip to its
hash voice, so after the 2026-11-03 retirement it would have MISSED 3,015 clips a retired
voice re-read and DELETED 3,834 that stage 06 had already moved to a survivor. Two clips
pin both directions:

- (a) stored voice retired, hash voice not: flagged `retired_voice` only with `--state`;
- (b) hash voice retired, stored voice a survivor: flagged without `--state`, not with it.

The retired set is passed explicitly, so this does not depend on the live roster. Real
short wav files are written with `wave`, so the real `probe_duration` runs.

Run: `uv run tests/test_audit_retired_state.py`

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "utils"))
from voice_assignment import VOICES_FR_V2, voice_for_row  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "audit_under_test", ROOT / "05_generate_audio" / "02_audit_durations.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
normalize_term = audit._load_generator().normalize_term

_failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if not cond else ""))
    if not cond:
        _failures.append(label)


def write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


TEXT = "Le patient prend du paracétamol."   # 32 chars, read in ~2.5 s by any voice
ROWS = [{"term_index": i, "variant_index": 0, "term": term, "asr_training_source": TEXT}
        for i, term in ((1, "paracetamol"), (2, "ibuprofene"))]
HASH = [voice_for_row(r, mode="row", retired=()) for r in ROWS]
# (a) row 0 was re-voiced by stage 06 to a voice that is now retired; (b) row 1's hash
# voice is retired but stage 06 had already moved it to a survivor.
STORED_A = next(v for v in VOICES_FR_V2 if v not in (HASH[0], HASH[1]))
STORED_B = next(v for v in VOICES_FR_V2 if v not in (HASH[0], HASH[1], STORED_A))
RETIRED = (STORED_A, HASH[1])
assert HASH[0] not in RETIRED and STORED_B not in RETIRED, (HASH, RETIRED)


def stem(row: dict) -> str:
    return f"{row['term_index']:06d}_{row['variant_index']:04d}_{normalize_term(row['term'])}"


def flagged(stage: Path, stored) -> dict[str, str]:
    bad = audit.audit_stage(stage, "row", normalize_term, fixed=audit.BUDGET_FIXED_S,
                            slow=None, fast=None, workers=2, stored=stored,
                            retired=RETIRED)
    return {Path(b["path"]).stem: b["verdict"] for b in bad}


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    # The real layout: the stage's generated_audios is a symlink into the audio tree,
    # whose folder name ("dictionary") is what stage 06's relative paths carry.
    data = tmp / "data" / "dictionary"
    data.mkdir(parents=True)
    stage = tmp / "01_dictionnary"
    stage.mkdir()
    (stage / "generated_audios").symlink_to(data)
    (stage / "generated_dataset.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ROWS))
    for r in ROWS:
        write_wav(data / f"{stem(r)}.wav", 2.5)
    state = tmp / "full.stt.jsonl"
    state.write_text("".join(json.dumps({"audio_filepath": f"../dictionary/{stem(r)}.flac",
                                         "voice": v}) + "\n"
                             for r, v in zip(ROWS, (STORED_A, STORED_B))))
    a, b = stem(ROWS[0]), stem(ROWS[1])

    print("stage-06 state loading")
    stored = audit.load_state_voices([state])
    check("state rows key on (audio folder, stem)",
          stored == {("dictionary", a): STORED_A, ("dictionary", b): STORED_B}, str(stored))

    print("without --state: the hash voice")
    without = flagged(stage, None)
    check("(a) a retired reader recorded only in the state is missed", a not in without,
          str(without))
    check("(b) a clip whose hash voice is retired is flagged",
          without.get(b) == "retired_voice", str(without))

    print("with --state: the recorded reader")
    with_state = flagged(stage, stored)
    check("(a) the stored retired reader is flagged",
          with_state.get(a) == "retired_voice", str(with_state))
    check("(b) a clip re-voiced to a survivor is NOT flagged", b not in with_state,
          str(with_state))

    print("the command line")
    from click.testing import CliRunner
    res = CliRunner().invoke(audit.main, ["--state", str(state), "--help"])
    check("--state is a known option", res.exit_code == 0 and "--state" in res.output,
          res.output[-300:])

if _failures:
    print(f"\n{len(_failures)} FAILED")
    sys.exit(1)
print("\nall ok")
