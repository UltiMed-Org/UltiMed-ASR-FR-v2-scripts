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
"""Keep clips a human listened to and judged fine, whatever the judges said.

The judges' verdicts are proxies; a listen is the ground truth. On 2026-10-30 the 60
clips left exhausted after T9d (loops, numbers, windows the judges kept flagging over
three voices and cfg 1.0) were listened to and all judged near-perfect to perfect, so
they ship. For each clip in ``--stems``:

* its stage-06 marker becomes ``improved`` with ``manual_pass`` set to ``--note``
  (scores and seeds kept), so the release's sync stops reporting it exhausted;
* ``14_reopen_loop_clips.process`` skips a ``manual_pass`` row, so no later ledger or
  loop reopen queues it again (``17_reopen_hgvs.py`` lifts the pass when the text
  itself changes, since the audio then no longer matches);
* its entry in the release's drop ledger (``99_hf_release/dropped_clips.jsonl``) is
  removed, or ``04_drop_bad_rows.py`` would re-apply the old drop on the same audio.

    uv run 18_manual_pass.py --stems STEMS.txt --note "2026-10-30 listened"   # census
    uv run 18_manual_pass.py --stems STEMS.txt --note "..." --apply

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
RELEASE = ROOT / "99_hf_release"
sys.path.insert(0, str(ROOT / "utils"))
from script_module import load_script  # noqa: E402


def pass_state(state: Path, stems: set[str], note: str, reopen, mod, apply: bool,
               counts: Counter) -> list[str]:
    """Mark the rows of `stems` in one state file as manually passed."""
    def select(row):
        return mod.Scores() if Path(row.get("audio_filepath") or "").stem in stems else None

    def mark(row, _scores):
        row["improvement"] = {**(row.get("improvement") or {}), "status": "improved",
                              reopen.MANUAL_PASS: note}

    return reopen.process(state, mod, select, apply, counts, Counter(), Counter(), mark)


def unledger(ledger: Path, stems: set[str], apply: bool) -> int:
    """Drop the ledger entries of `stems`; returns how many there were."""
    if not ledger.is_file():
        return 0
    lines = ledger.read_text(encoding="utf-8").splitlines(keepends=True)
    keep = [line for line in lines if Path(json.loads(line)["clip"]).stem not in stems]
    if apply and len(keep) != len(lines):
        ledger.write_text("".join(keep), encoding="utf-8")
    return len(lines) - len(keep)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stems", "stems_file", required=True, type=click.Path(exists=True, path_type=Path),
              help="One clip stem per line.")
@click.option("--note", required=True, help="Stored as the marker's manual_pass value (who, when).")
@click.option("--apply", is_flag=True, help="Write. Without it: census only.")
def main(stems_file, note, apply) -> None:
    stems = {s.strip() for s in stems_file.read_text().splitlines() if s.strip()}
    reopen = load_script(_HERE / "14_reopen_loop_clips.py", "reopen_for_manual")
    sync = load_script(RELEASE / "03_sync_hotfix_results.py", "sync_for_manual")
    drop = load_script(RELEASE / "04_drop_bad_rows.py", "drop_for_manual")
    mod = reopen._load_pass()
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")
    counts, found = Counter(), set()
    for src, _ in sync.HOTFIX_SOURCES:
        state = (RELEASE / src).resolve()
        if state.exists():
            found.update(pass_state(state, stems, note, reopen, mod, apply, counts))
    for key in sorted(k for k in counts if k.startswith(("from::", "left::"))):
        logger.info(f"    {key}: {counts[key]}")
    logger.info(f"{len(found)} of {len(stems)} clip(s) marked manually passed")
    if stems - found:
        logger.warning(f"not marked (absent, or already passed): {sorted(stems - found)}")
    logger.info(f"drop ledger: {unledger(drop.DROP_LEDGER, stems, apply)} entr(y/ies) removed")


if __name__ == "__main__":
    main()
