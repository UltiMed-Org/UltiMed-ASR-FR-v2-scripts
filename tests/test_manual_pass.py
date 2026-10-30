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
"""`06_hotfixes/18_manual_pass.py`: a listened-to clip ships and stays shipped.

  * an exhausted clip in the list becomes ``improved`` with ``manual_pass`` set, its
    scores kept; a clip not in the list is left byte for byte;
  * its drop-ledger entry goes, the others stay;
  * a later reopen (`14.process`, which `16` goes through) leaves it alone;
  * `17_reopen_hgvs` still queues it when its text changes, lifting the pass.

    uv run tests/test_manual_pass.py

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "utils"))
from script_module import load_script  # noqa: E402

failures = []


def check(name: str, ok: bool, detail=None) -> None:
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        failures.append(name)


def read(path: Path) -> dict:
    return {Path(r["audio_filepath"]).stem: r for r in map(json.loads, path.read_text().splitlines())}


def main() -> int:
    t18 = load_script(ROOT / "06_hotfixes" / "18_manual_pass.py", "t18_test")
    t17 = load_script(ROOT / "06_hotfixes" / "17_reopen_hgvs.py", "t17_mp_test")
    reopen = load_script(ROOT / "06_hotfixes" / "14_reopen_loop_clips.py", "reopen_t18_test")
    mod = reopen._load_pass()
    rows = [{"audio_filepath": f"../PARHAF/{s}.flac", "text": t, "asr_training_source": t,
             "improvement": {"status": "exhausted", "cer": 0.09, "seeds_spent": 12}}
            for s, t in (("a", "Boucle entendue."), ("b", "Autre clip."), ("c", "c.35G en T."))]
    with tempfile.TemporaryDirectory() as tmp:
        state, ledger = Path(tmp) / "full.stt.jsonl", Path(tmp) / "dropped_clips.jsonl"
        state.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        ledger.write_text("".join(json.dumps({"clip": f"PARHAF/{s}.flac", "reason": "exhausted",
                                              "audio_mtime_ns": 1}) + "\n" for s in "ab"))
        b_before = state.read_text().splitlines()[1]
        found = t18.pass_state(state, {"a", "c"}, "2026-10-30 listened", reopen, mod, True, Counter())
        got = read(state)
        check("listed clips are marked", sorted(found) == ["a", "c"], found)
        check("marker is improved + manual_pass, scores and seeds kept",
              got["a"]["improvement"] == {"status": "improved", "cer": 0.09, "seeds_spent": 12,
                                          "manual_pass": "2026-10-30 listened"}, got["a"]["improvement"])
        check("an unlisted clip is untouched", state.read_text().splitlines()[1] == b_before)
        n = t18.unledger(ledger, {"a", "c"}, True)
        check("only the listed ledger entry goes",
              n == 1 and [json.loads(x)["clip"] for x in ledger.read_text().splitlines()] == ["PARHAF/b.flac"],
              ledger.read_text())

        counts = Counter()
        again = reopen.process(state, mod, lambda row: mod.Scores(), True, counts, Counter(), Counter())
        check("a later reopen skips manually passed clips",
              again == ["b"] and counts["left::manual_pass"] == 2, (again, counts))

        hg = t17.reopen_hgvs(state, reopen, mod, True, Counter())
        c = read(state)["c"]
        check("a text change still queues it and lifts the pass",
              hg == ["c"] and c["improvement"]["status"] == "pending_tts"
              and "manual_pass" not in c["improvement"] and c["text"] == "c.35G vers T.", (hg, c))

    print("FAILED" if failures else "all ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
