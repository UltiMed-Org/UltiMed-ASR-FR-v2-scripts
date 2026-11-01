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
"""Queue named clips for a redraw, whatever the judges say about them.

``driver.sh STEMS_FROM=...`` restricts a pass to the listed stems but does NOT force
work on them: a clip whose stored judge reading passes every gate is "0 need work" and
its audio is never touched. Found on 2026-11-01 with the PARHAF clip
``010290_0000_chir_ortho_et_traumato_00339_t0_c1``: whisper.cpp heard "De Devae-Donneau
2024" for "Le 22 août 2024" (CER 0.032, under the gate, the lost number is a deletion
rather than a swap so the number gate misses it too), every fine-tuned model misheard
it the same way, and a STEMS_FROM run left it as it was.

This script sets the stage's own ``pending_tts`` marker on each listed clip, the same
way ``14_reopen_loop_clips.py`` / ``16_reopen_from_ledger.py`` do (it reuses 14's
streamed ``process`` and the parakeet judge's scores as the "when flagged" view), so the
driver's next tts pass redraws them. Rows already queued or kept by a human
(``18_manual_pass.py``) are left alone. Then run the driver as usual.

    uv run 19_queue_stems.py STEMS.txt            # census only
    uv run 19_queue_stems.py STEMS.txt --apply    # write the markers

Written with Claude Code.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from script_module import load_script  # noqa: E402

STATE = _HERE / "improved" / "full.stt.jsonl"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("stems_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--state", default=STATE, show_default=True, type=click.Path(path_type=Path),
              help="Stage-06 state file to mark (improved_parrot/full.stt.jsonl for PARROT).")
@click.option("--apply", is_flag=True, help="Write the markers. Without it: census only.")
def main(stems_file: Path, state: Path, apply: bool) -> None:
    stems = {ln.strip() for ln in stems_file.read_text().splitlines() if ln.strip()}
    reopen = load_script(_HERE / "14_reopen_loop_clips.py", "reopen_for_queue")
    mod = reopen._load_pass()

    def select(row):
        if Path(row.get("audio_filepath") or "").stem not in stems:
            return None
        reading = (row.get("transcriptions") or {}).get(reopen.DEFAULT_JUDGE_KEY) or {}
        return mod.scores_from_reading(reading) if reading.get("text") else mod.Scores()

    counts, by_voice, by_cat = Counter(), Counter(), Counter()
    found = reopen.process(state, mod, select, apply, counts, by_voice, by_cat)
    logger.info(f"{'APPLIED' if apply else 'DRY RUN'}: {dict(counts)} voices={dict(by_voice)} "
                f"cats={dict(by_cat)}")
    missing = sorted(stems - set(found))
    if missing:
        logger.warning(f"{len(missing)} stem(s) not queued (absent, already queued or manual_pass): "
                       f"{missing[:10]}")
    if not apply:
        logger.info("census only, rerun with --apply to write the markers")


if __name__ == "__main__":
    main()
