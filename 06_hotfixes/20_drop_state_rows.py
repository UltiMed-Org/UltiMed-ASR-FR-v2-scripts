#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""Drop named clips' rows from stage 06's state, so the next driver run judges them fresh.

Stage 06 merges the rebuilt release manifest with its state file keyed on
`audio_filepath`, and an existing state row always wins (`load_existing` and the merge in
`01_recursive_improvement.py`). That is right for a clip whose audio stage 06 itself owns,
and wrong for one whose audio was replaced UNDER it: after a voice retirement,
`05_generate_audio/02_audit_durations.py --apply` deletes every clip a retired voice read
(~266k on 2026-11-03) and stage 05 regenerates them in place with survivor voices, but
the state still holds each stem's OLD reading, flags, `improvement` record and the
retired voice name. Nothing would ever notice the new audio, and the release would ship
the old take's verdicts and voice for a clip that no longer sounds like that.

Dropping the row is the whole fix. The rebuilt manifest then brings the clip in as a new
row (new voice, new duration), and the driver's stt pass, which selects rows lacking a
reading from the current judge, re-judges exactly those clips and nothing else.

When: after stage 05 has regenerated the audit's deleted clips, before
`RESET_STATE=1 ./driver.sh`. Which rows: the stems of the audit's report
(`--stems-from`, its JSONL rows carry `path`; a plain one-stem-per-line list works too),
and/or every row whose stored `voice` is a retired one (`--voice`). Dropped rows are
appended verbatim to `<state>.dropped-<YYYYmmdd-HHMMSS>.jsonl` beside the state, so
nothing is lost. Dry run by default:

    cd 06_hotfixes
    uv run 20_drop_state_rows.py --stems-from ../05_generate_audio/retired.jsonl
    uv run 20_drop_state_rows.py --stems-from ../05_generate_audio/retired.jsonl --apply

The stems are matched on `Path(audio_filepath).stem` alone, across every stage: stems
carry their source's slug, so two stages minting the same one is not expected, and the
report counts the requested stems found in no state file.

Written with Claude Code.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from script_module import load_script  # noqa: E402

reopen = load_script(_HERE / "14_reopen_loop_clips.py", "reopen_for_drop")


def read_stems(path: Path) -> set[str]:
    """Stems listed in `path`: a JSONL whose rows carry `path` or `audio_filepath` (the
    audit report), or one stem per line. The format is read off the first non-blank
    line, so a list cannot be half one and half the other."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    if not lines:
        raise click.UsageError(f"{path} lists no stem: an empty list would drop nothing "
                               "and look exactly like a clean run")
    try:
        is_jsonl = isinstance(json.loads(lines[0]), dict)
    except json.JSONDecodeError:
        is_jsonl = False
    if not is_jsonl:
        return {Path(ln).stem for ln in lines}
    stems, keyless = set(), 0
    for ln in lines:
        row = json.loads(ln)
        value = row.get("path") or row.get("audio_filepath")
        if value:
            stems.add(Path(value).stem)
        else:
            keyless += 1
    if keyless:
        logger.warning(f"{path}: {keyless} row(s) carry neither `path` nor "
                       "`audio_filepath`, ignored")
    return stems


def process(state: Path, stems: set[str], voices: set[str], apply: bool,
            tally: dict[str, Counter], found: set[str]) -> int:
    """One state file: drop every row named by `stems` or read by one of `voices`, keep
    the dropped rows verbatim beside it under `apply`. Returns how many were dropped."""
    dropped_path = state.with_name(
        f"{state.name}.dropped-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
    dropped = dropped_path.open("a", encoding="utf-8") if apply else None
    n = 0

    def edit(row: dict, line: str) -> dict | None:
        nonlocal n
        tally["counts"]["rows"] += 1
        audio = row.get("audio_filepath") or ""
        stem = Path(audio).stem
        if stem not in stems and row.get("voice") not in voices:
            return row
        n += 1
        found.add(stem)
        tally["voice"][row.get("voice") or "?"] += 1
        tally["category"][reopen.category_of(audio)] += 1
        tally["status"][(row.get("improvement") or {}).get("status") or "no marker"] += 1
        if dropped is not None:
            dropped.write(line if line.endswith("\n") else line + "\n")
        return None

    try:
        reopen.stream_rewrite(state, apply, edit, tmp_suffix=".drop.tmp")
    finally:
        if dropped is not None:
            dropped.close()
    if apply:
        if n:
            logger.info(f"{state.name}: kept the {n} dropped row(s) in {dropped_path}")
        else:
            dropped_path.unlink()
    return n


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--state", "states", multiple=True, type=click.Path(path_type=Path),
              help=f"State file(s). Default: {', '.join(reopen.DEFAULT_STATES)}")
@click.option("--stems-from", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Stems to drop: one per line, or a JSONL with `path` / "
                   "`audio_filepath` (the audit report). Required unless --voice.")
@click.option("--voice", "voices", multiple=True,
              help="Also drop every row whose stored `voice` is this one (repeatable).")
@click.option("--apply", is_flag=True,
              help="Actually rewrite the state files (default: report only).")
def main(states, stems_from, voices, apply) -> None:
    """Drop the named clips' rows from stage 06's state so they are judged afresh."""
    if stems_from is None and not voices:
        raise click.UsageError("give --stems-from, --voice, or both")
    stems = read_stems(stems_from) if stems_from else set()
    voices = set(voices)
    logger.info(f"{len(stems)} stem(s) requested, voice(s) {sorted(voices) or 'none'}")
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")
    tally = {k: Counter() for k in ("counts", "voice", "category", "status")}
    found: set[str] = set()
    for state in ([Path(s) for s in states] or [_HERE / s for s in reopen.DEFAULT_STATES]):
        if not state.exists():
            logger.warning(f"{state} does not exist, skipping it")
            continue
        n = process(state, stems, voices, apply, tally, found)
        logger.info(f"{state.name}: {n} row(s) {'dropped' if apply else 'to drop'}")
    logger.info(f"{tally['counts']['rows']} row(s) read, "
                f"{sum(tally['voice'].values())} {'dropped' if apply else 'to drop'}")
    logger.info(f"by voice: {dict(tally['voice'].most_common())}")
    logger.info(f"by category: {dict(tally['category'].most_common())}")
    logger.info(f"by improvement.status: {dict(tally['status'].most_common())}")
    missing = sorted(stems - found)
    if missing:
        logger.warning(f"{len(missing)} requested stem(s) found in no state file: "
                       f"{missing[:10]}")
    if not apply:
        logger.info("dry run only, rerun with --apply to drop them")


if __name__ == "__main__":
    main()
