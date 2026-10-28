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
"""Build `clip_ledger.jsonl` in every category directory: one line per clip, every QC
verdict that still describes its audio, and the flags they raise.

The format and every merge rule are in `utils/clip_ledger.py`; this is only the wiring
(which state files, which release files, which gate thresholds). Rebuild it whenever a
QC stage, a repair round or the release chain has run, and BEFORE anything decides
about clips from it (the T6 redraw queue, the drop step):

    uv run 15_build_clip_ledger.py                     # all five categories
    uv run 15_build_clip_ledger.py --category PARHAF   # just one
    uv run 15_build_clip_ledger.py --dry-run           # census only, writes nothing

The census it prints is the flag Counter per category, which is also the quickest way to
see what a QC stage changed: build, run the stage, build again, diff the two censuses.

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "utils"))
import clip_ledger as L  # noqa: E402

RELEASE = ROOT / "99_hf_release"
CATEGORIES = ("PARHAF", "PARROT", "acronyms", "dictionary", "drugs")


def _load(path: Path, name: str):
    """Import a numbered uv script by path (its filename is not a module name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@click.command()
@click.option("--category", "categories", multiple=True, type=click.Choice(CATEGORIES),
              help="Only these categories (repeatable). Default: all of them.")
@click.option("--dry-run", is_flag=True, help="Print the census, write nothing.")
def main(categories: tuple[str, ...], dry_run: bool) -> None:
    data = RELEASE / "data"
    cats = categories or CATEGORIES
    unknown = {c: L.check_known(data / c) for c in cats}
    if any(unknown.values()):
        raise click.ClickException(
            "sidecars the ledger does not know how to read (teach utils/clip_ledger.py "
            f"SOURCES about them, never skip them): {dict((c, u) for c, u in unknown.items() if u)}")

    compare = _load(_HERE / "10_compare_judges.py", "compare_for_ledger")
    recimp = compare._load_pass()
    stt = recimp._stt
    reopen = _load(_HERE / "14_reopen_loop_clips.py", "reopen_for_ledger")
    gates = compare.gates_from(stt.DEFAULT_CER_GATE, stt.DEFAULT_TAIL_CER_GATE,
                               stt.DEFAULT_WINDOW_CER_GATE, stt.DEFAULT_BAD_RUN_GATE,
                               reopen.DEFAULT_REPEAT_GATE)
    judge_fn = partial(L.judge_block, scores_from_reading=recimp.scores_from_reading,
                       which_gates=compare.which_gates, gates=gates)
    sync = _load(RELEASE / "03_sync_hotfix_results.py", "sync_for_ledger")
    drop = _load(RELEASE / "04_drop_bad_rows.py", "drop_for_ledger")

    states = [((RELEASE / src).resolve(), RELEASE / base) for src, base in sync.HOTFIX_SOURCES]
    logger.info(f"reading stage-06 state: {[str(p.relative_to(ROOT)) for p, _ in states]}")
    state = L.load_state(states)
    repairs = L.load_repairs([p.parent / "full.improved.jsonl" for p, _ in states])
    release = L.load_release(data / "NeMO_files", drop.DROP_LEDGER, drop.ASR_FLAGGED_LIST,
                             drop.still_dropped)
    logger.info(f"{len(state)} clips in state, {len(release)} in the release files")

    for cat in cats:
        rows = L.build_category(data / cat, state, repairs, release, judge_fn,
                                normalize=stt.normalize_for_scoring)
        census = Counter(f for r in rows for f in r["flags"])
        shipped = sum(bool(r["release"]["splits"]) for r in rows)
        logger.info(f"{cat}: {len(rows)} clips, {shipped} shipped")
        for flag, n in sorted(census.items(), key=lambda kv: (-kv[1], kv[0])):
            logger.info(f"    {n:7d}  {flag}")
        if not dry_run:
            logger.success(f"wrote {L.write_ledger(data / cat, rows)}")


if __name__ == "__main__":
    main()
