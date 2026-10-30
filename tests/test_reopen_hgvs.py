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
"""`06_hotfixes/17_reopen_hgvs.py`: the HGVS ``vers`` rewrite and its redraw queue.

  * a state row whose source spells the operator another way gets every text field
    rewritten and is marked `pending_tts` with no reference score, so only a draw
    clearing the gates can replace its audio;
  * a row already reading ``vers``, or with no substitution, is left byte for byte;
  * a row already queued keeps its marker but still gets the new text, since that is
    what the queued draw will read;
  * a stage file gets the label and the source rewritten, nothing else.

    uv run tests/test_reopen_hgvs.py

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


def row(stem: str, text: str, improvement: dict | None = None) -> dict:
    out = {"audio_filepath": f"../PARHAF/{stem}.flac", "voice": "fr_male", "text": text,
           "asr_training_source": text, "transcriptions": {}}
    if improvement is not None:
        out["improvement"] = improvement
    return out


def main() -> int:
    t17 = load_script(ROOT / "06_hotfixes" / "17_reopen_hgvs.py", "t17_test")
    reopen = load_script(ROOT / "06_hotfixes" / "14_reopen_loop_clips.py", "reopen_t17_test")
    mod = reopen._load_pass()
    rows = [
        row("a", "mutation c.1163C supérieur à T.", {"status": "improved", "cer": 0.01}),
        row("b", "mutation c.1163C vers T."),
        row("c", "Pas de mutation."),
        row("d", "c.35G supérieur à T.", {"status": "pending_tts", "cer": 0.2}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "full.stt.jsonl"
        state.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        before = state.read_text().splitlines()
        stems = t17.reopen_hgvs(state, reopen, mod, True, Counter())
        after = state.read_text().splitlines()
        got = {Path(r["audio_filepath"]).stem: r for r in map(json.loads, after)}
        a = got["a"]
        check("the changed row is queued", stems == ["a"], stems)
        check("its label and source both read vers",
              a["text"] == a["asr_training_source"] == "mutation c.1163C vers T.", a)
        check("it is marked pending_tts with no reference score",
              a["improvement"]["status"] == "pending_tts"
              and a["improvement"].get("cer") is None, a["improvement"])
        check("rows already in vers or without a substitution are untouched",
              after[1:3] == before[1:3], after[1:3])
        d = got["d"]
        check("an already queued row gets the new text and keeps its marker",
              d["asr_training_source"] == "c.35G vers T."
              and d["improvement"] == {"status": "pending_tts", "cer": 0.2}, d)

        stage = Path(tmp) / "generated_dataset.jsonl"
        srow = {"term": "BRAF", "asr_training_target": "c.1799T supérieur à A",
                "asr_training_source": "c.1799T supérieur à A", "other": "c.1799T supérieur à A"}
        stage.write_text(json.dumps(srow, ensure_ascii=False) + "\n" + json.dumps({"term": "x"}) + "\n")
        n = t17.rewrite_stage(stage, True)
        out = [json.loads(line) for line in stage.read_text().splitlines()]
        check("a stage file rewrites label and source only",
              n == 1 and out[0]["asr_training_target"] == out[0]["asr_training_source"] == "c.1799T vers A"
              and out[0]["other"] == "c.1799T supérieur à A" and out[1] == {"term": "x"}, (n, out))

    print("FAILED" if failures else "all ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
