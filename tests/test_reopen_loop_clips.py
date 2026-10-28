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
"""`06_hotfixes/14_reopen_loop_clips.py`: which clips it queues, and what it writes.

The script exists because a looping take is invisible to the judge the pass runs (see
tests/test_loop_gate.py), so its rule is a CONJUNCTION of two signals that are each
wrong on their own, and the marker it writes has to carry the flagging judge's numbers
or the repair cannot be promoted. Both are easy to break by a one-line "simplification",
so both are pinned here:

  * a clip whose parakeet reading is bad AND repeats is queued;
  * a clip that repeats but whose parakeet CER is clean is NOT (the metric tolerates a
    label that legitimately repeats a phrase, and 160 corpus clips sit in that bucket
    with chars/second indistinguishable from the controls);
  * a clip parakeet calls bad for any OTHER reason is NOT (its CER gates alone flag
    8,096 clips, mostly its known medical-vocabulary misreads);
  * the marker gets PARAKEET's cer, not whisper's, because promotion needs a draw to
    beat it by `--min-improvement` and whisper's view of a looping clip is clean;
  * `seeds_spent` survives, since the TTS is deterministic per (voice, seed) and a reset
    ladder would redraw byte-identical audio;
  * a clip already queued is left exactly as it is, and a row no judge read is untouched.

    uv run tests/test_reopen_loop_clips.py

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REOPEN_PATH = HERE.parent / "06_hotfixes" / "14_reopen_loop_clips.py"
PASS_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"
JUDGE = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
TURBO = "whisper.cpp/ggml-large-v3-turbo"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def reading(cer: float, repeat: int | None, text: str = "ce que le juge a entendu"):
    out = {"text": text, "cer": cer, "cer_tail": 0.01, "cer_window": 0.02,
           "bad_run_chars": 1}
    if repeat is not None:
        out["repeat_excess"] = repeat
    return out


#: One row per case the rule has to separate. `clean_turbo` is on every row, because in
#: the corpus it always is, and it must never be what the decision is made on.
ROWS = [
    # the defect: parakeet rejects it and what it rejects is a repeat
    {"audio_filepath": "../PARHAF/loop_improved_t0_c1.flac", "voice": "h_frf_ptm",
     "text": "le compte rendu", "improvement": {"status": "improved", "orig_cer": 0.21,
                                                "seeds_spent": 5, "attempts": 5},
     "transcriptions": {TURBO: reading(0.019, None), JUDGE: reading(0.276, 8)}},
    # repeats, but parakeet's CER is clean: the label repeats a phrase by itself
    {"audio_filepath": "../dictionary/repeat_but_clean.flac", "voice": "fr_male",
     "text": "un comprime matin midi et soir",
     "transcriptions": {TURBO: reading(0.012, None), JUDGE: reading(0.013, 9)}},
    # parakeet calls it bad, but nothing repeated: its own misreading, not our defect
    {"audio_filepath": "../dictionary/bad_but_no_loop.flac", "voice": "fr_female",
     "text": "myocardite des dermatomyosites",
     "transcriptions": {TURBO: reading(0.02, None), JUDGE: reading(0.31, 0)}},
    # a loop that is already queued: nothing to do, and re-marking would reshuffle it
    {"audio_filepath": "../PARHAF/loop_already_queued.flac", "voice": "h_frm_ptf",
     "text": "le compte rendu", "improvement": {"status": "pending_tts",
                                                "orig_cer": 0.30, "seeds_spent": 2},
     "transcriptions": {TURBO: reading(0.02, None), JUDGE: reading(0.29, 11)}},
    # no parakeet reading at all: the judge never answered, so there is no verdict
    {"audio_filepath": "../drugs/never_read_by_parakeet.flac", "voice": "fr_male",
     "text": "thyrofix", "transcriptions": {TURBO: reading(0.02, None)}},
]


def run(tmp: Path, mod, pass_mod, repeat_gate: int = 4):
    state = tmp / "full.stt.jsonl"
    state.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS),
                     encoding="utf-8")
    counts, by_voice, by_cat = Counter(), Counter(), Counter()
    gates = mod.gates_of(pass_mod._stt, repeat_gate)
    stems = mod.process(state, pass_mod, lambda row: mod.loops(pass_mod, row, JUDGE, gates),
                        True, counts, by_voice, by_cat)
    rows = {Path(json.loads(line)["audio_filepath"]).stem: json.loads(line)
            for line in state.read_text(encoding="utf-8").splitlines() if line.strip()}
    return stems, counts, rows, by_voice, by_cat


def test_only_the_conjunction_is_queued(mod, pass_mod, tmp: Path) -> None:
    stems, counts, rows, by_voice, by_cat = run(tmp, mod, pass_mod)
    assert stems == ["loop_improved_t0_c1"], \
        f"exactly one row qualifies, got {stems}"
    assert counts["reopened"] == 1 and counts["already_queued"] == 1, dict(counts)
    assert counts["rows"] == len(ROWS), dict(counts)
    assert rows["repeat_but_clean"].get("improvement") is None, \
        "a repeat with clean parakeet CER is the metric's tolerance, not a defect"
    assert rows["bad_but_no_loop"].get("improvement") is None, \
        "parakeet calling a clip bad is not enough: 8,096 clips do, mostly wrongly"
    assert rows["never_read_by_parakeet"].get("improvement") is None, \
        "no reading from the flagging judge means no verdict"
    assert by_voice == Counter({"h_frf_ptm": 1}) and by_cat == Counter({"PARHAF": 1}), \
        f"the report's breakdown must count the queued clip only: {by_voice} {by_cat}"


def test_the_marker_carries_the_flagging_judges_numbers(mod, pass_mod, tmp: Path) -> None:
    _stems, _counts, rows, _v, _c = run(tmp, mod, pass_mod)
    imp = rows["loop_improved_t0_c1"]["improvement"]
    assert imp["status"] == "pending_tts", imp
    assert imp["orig_cer"] == 0.276, \
        ("the marker must hold PARAKEET's cer: with whisper's 0.019 no draw can clear "
         f"--min-improvement and every clip would exhaust, got {imp}")
    assert imp["seeds_spent"] == 5, \
        f"spent seeds must survive or the redraw repeats itself byte for byte: {imp}"
    assert "best_cer" not in imp, \
        "the previous round's win is resolved; carrying it would compare against it"


def test_an_untouched_row_is_rewritten_verbatim(mod, pass_mod, tmp: Path) -> None:
    """Every row is rewritten, so a row the rule skips must come back byte-identical."""
    _stems, _counts, rows, _v, _c = run(tmp, mod, pass_mod)
    for row in ROWS:
        stem = Path(row["audio_filepath"]).stem
        if stem == "loop_improved_t0_c1":
            continue
        assert rows[stem] == row, f"{stem} was modified: {rows[stem]} != {row}"


def test_the_gate_is_a_knob_not_a_constant(mod, pass_mod, tmp: Path) -> None:
    """At the live QC's gate of 1 the clean-CER repeat is still spared (the conjunction),
    and at a gate above the clip's count nothing is queued at all."""
    stems, _c, _r, _v, _cat = run(tmp, mod, pass_mod, repeat_gate=1)
    assert stems == ["loop_improved_t0_c1"], stems
    stems, _c, _r, _v, _cat = run(tmp, mod, pass_mod, repeat_gate=9)
    assert stems == [], f"nothing reaches a gate of 9 but the spared row: {stems}"


if __name__ == "__main__":
    reopen = load_module(REOPEN_PATH, "reopen_under_test")
    recimp = load_module(PASS_PATH, "recimp_for_reopen_test")
    with tempfile.TemporaryDirectory() as d:
        for fn in (test_only_the_conjunction_is_queued,
                   test_the_marker_carries_the_flagging_judges_numbers,
                   test_an_untouched_row_is_rewritten_verbatim,
                   test_the_gate_is_a_knob_not_a_constant):
            fn(reopen, recimp, Path(d))
            print(f"  ok  {fn.__name__}")
    print("OK: the conjunction holds and the marker carries parakeet's numbers")
