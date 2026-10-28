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
"""`06_hotfixes/16_reopen_from_ledger.py`: the T6 redraw queue built from the clip ledger.

The rules themselves are doctested in `reasons_of`. This pins the wiring, where a
one-line slip would queue the wrong clips or redraw them with the wrong voice:

  * the ledger's `clip` key matches the state row through the state's own base
    directory, so `../PARHAF/x.flac` in the state is `PARHAF/x.flac` in the ledger;
  * a NUMBER clip read by an `h_itf_*` voice is swapped to the fr rung at once, and one
    read by another voice keeps it but has the itf rung marked spent, so its next
    escalation lands on fr; the rule also sees an itf voice the LADDER handed it
    (`swap_voice`), not just the row's original one;
  * a clip queued for any other reason keeps the ladder exactly as it was;
  * the marker carries parakeet's numbers and `seeds_spent` (same reasons as 14);
  * a clip already queued, or absent from the queue, is left byte for byte.

    uv run tests/test_reopen_from_ledger.py

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "utils"))
from script_module import load_script  # noqa: E402

JUDGE = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
TURBO = "whisper.cpp/ggml-large-v3-turbo"
ROSTER = ["fr_female", "fr_male", "h_itf_arm", "h_frf_esm", "h_frm_nlf", "h_itf_def",
          "h_frf_itm", "h_frm_ptf", "h_frf_ptm"]


def reading(cer: float) -> dict:
    return {"text": "x", "cer": cer, "cer_tail": cer, "cer_window": cer, "bad_run_chars": 1}


def row(stem: str, voice: str, improvement: dict | None = None) -> dict:
    out = {"audio_filepath": f"../PARHAF/{stem}.flac", "voice": voice, "text": "x",
           "group_id": f"doc-{stem}",
           "transcriptions": {TURBO: reading(0.01), JUDGE: reading(0.08)}}
    if improvement is not None:
        out["improvement"] = improvement
    return out


ROWS = [
    row("number_itf", "h_itf_def", {"status": "improved", "orig_cer": 0.2, "seeds_spent": 5}),
    row("number_fr", "fr_female"),
    row("number_laddered_itf", "fr_male",
        {"status": "improved", "swap_voice": "h_itf_arm", "voice_tier": 1, "orig_cer": 0.1}),
    row("loop_itf", "h_itf_arm"),
    row("number_already_queued", "h_itf_def", {"status": "pending_stt", "orig_cer": 0.3}),
    row("not_queued", "h_itf_def"),
]
QUEUE = {"PARHAF/number_itf.flac": ["number"], "PARHAF/number_fr.flac": ["number"],
         "PARHAF/number_laddered_itf.flac": ["loop", "number"],
         "PARHAF/loop_itf.flac": ["loop"], "PARHAF/number_already_queued.flac": ["number"]}


def main() -> int:
    t16 = load_script(ROOT / "06_hotfixes" / "16_reopen_from_ledger.py", "t16_under_test")
    reopen = load_script(ROOT / "06_hotfixes" / "14_reopen_loop_clips.py", "t14_for_t16_test")
    mod = reopen._load_pass()
    failures = []

    def check(label, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if not ok else ""))
        if not ok:
            failures.append(label)

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        base = data / "NeMO_files"
        base.mkdir(parents=True)
        (data / "PARHAF").mkdir()
        (data / "PARHAF" / "voice_assignment.json").write_text(json.dumps({"03_PARHAF/x.jsonl": {
            "voices": ROSTER, "mode": "group", "group_field": "source_id"}}))
        state = base / "full.stt.jsonl"
        state.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
        before = {json.loads(line)["audio_filepath"]: line for line in state.read_text().splitlines()}

        tally = t16.new_tally()
        picked = t16.reopen_state(state, base, QUEUE, reopen, mod, True, tally)
        after = {Path(r["audio_filepath"]).stem: r for r in map(json.loads, state.read_text().splitlines())}
        imp = {k: v.get("improvement") or {} for k, v in after.items()}

        check("four clips marked, matched through the state's base dir",
              sorted(p["clip"] for p in picked) == sorted(k for k in QUEUE if "already" not in k),
              [p["clip"] for p in picked])
        check("every marked clip is pending_tts",
              all(imp[Path(p["clip"]).stem]["status"] == "pending_tts" for p in picked))
        check("itf number clip swapped to the fr rung now",
              imp["number_itf"].get("swap_voice") in ("fr_male", "fr_female")
              and imp["number_itf"].get("voice_tier") == 2, imp["number_itf"])
        check("its marker carries parakeet's cer and its seeds",
              imp["number_itf"].get("orig_cer") == 0.08 and imp["number_itf"].get("seeds_spent") == 5,
              imp["number_itf"])
        check("fr number clip keeps its voice, itf rung marked spent",
              "swap_voice" not in imp["number_fr"] and imp["number_fr"].get("voice_tier") == 1,
              imp["number_fr"])
        check("an itf voice the ladder handed out is swapped too, never back to a tried one",
              imp["number_laddered_itf"].get("swap_voice") == "fr_female"
              and imp["number_laddered_itf"].get("voice_tier") == 2, imp["number_laddered_itf"])
        check("a loop-only clip keeps the ladder untouched",
              "swap_voice" not in imp["loop_itf"] and not imp["loop_itf"].get("voice_tier"),
              imp["loop_itf"])
        rows_now = {json.loads(line)["audio_filepath"]: line for line in state.read_text().splitlines()}
        for stem in ("number_already_queued", "not_queued"):
            key = f"../PARHAF/{stem}.flac"
            check(f"{stem} left byte for byte", rows_now[key] == before[key])
        check("the census counts the voice moves",
              sum(tally.voice_moves.values()) == 3 and tally.by_reason["number"] == 3,
              dict(tally.voice_moves))

    print("FAILED" if failures else "all ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
