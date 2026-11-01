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
"""Rewrite HGVS substitutions to ``vers`` and queue their clips for redraw (2026-10-30).

The LLM could not write ``c.1163C>T`` (``>`` is forbidden in a label, and the parakeet
tokenizer has no such piece), so it spelt the operator in words. Most are correct French
(``en``, ``vers``, ``substitué en``, ``remplacé par``), three are not: ``supérieur à`` and
``plus grand que`` (a comparison) and ``majuscule``. ``label_conventions.normalize_hgvs``
rewrites those three to ``vers``; this script applies it to what already exists:

* each stage's ``generated_dataset.jsonl`` (``asr_training_target`` and
  ``asr_training_source``), which the release rebuilds its manifests from;
* the release manifests the rounds read (the ``HOTFIX_SOURCES`` targets'
  ``full.jsonl``). The stt pass takes a clip's text from there and drops a stored
  state row whose text differs as stale, marker included: rewriting the state alone
  was silently undone by the next round on 2026-10-30 (425 of 430 clips never redrawn);
* the stage-06 state files (``text``, ``asr_training_source``, ``asr_training_target``),
  which the tts pass reads its text from. A row whose source changed is marked
  ``pending_tts`` with no reference score, so a draw replaces the audio only once it
  clears every gate against the new text; a clip no draw fixes ends ``exhausted`` and the
  release drops it, since its audio still says the old operator.

Rows already reading ``vers`` change nothing and are not queued. The release's
``05_normalize_text.py`` applies the same rule to the manifests, so a stage file this
script missed cannot ship the old form either.

    uv run 17_reopen_hgvs.py                       # census
    uv run 17_reopen_hgvs.py --apply --out-stems hgvs_stems.txt

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
from label_conventions import normalize_hgvs  # noqa: E402
from nemo_manifest import rewrite_jsonl  # noqa: E402
from script_module import load_script  # noqa: E402

STAGES = ("01_dictionnary", "02_drugs", "03_PARHAF", "04_PARROT", "07_acronyms")
STAGE_FIELDS = ("asr_training_target", "asr_training_source")
STATE_FIELDS = ("text",) + STAGE_FIELDS


def rewrite_fields(row: dict, fields: tuple[str, ...]) -> bool:
    """Apply `normalize_hgvs` to `fields` of `row` in place; whether the TTS source changed."""
    spoken = row.get("asr_training_source")
    for field in fields:
        if isinstance(row.get(field), str):
            row[field] = normalize_hgvs(row[field])
    return row.get("asr_training_source") != spoken


def rewrite_stage(path: Path, apply: bool, fields: tuple[str, ...] = STAGE_FIELDS) -> int:
    """Rewrite `fields` of one jsonl file, streamed; returns how many rows changed."""
    return rewrite_jsonl(path, lambda row: rewrite_fields(row, fields), apply)


def reopen_hgvs(state: Path, reopen, mod, apply: bool, counts: Counter) -> list[str]:
    """Rewrite one state file's texts and mark the rows whose source changed.

    A manual pass is lifted from a changed row: the take that was listened to says the
    old operator, so it no longer matches the label."""
    def select(row):
        if not rewrite_fields(row, STATE_FIELDS):
            return None
        (row.get("improvement") or {}).pop(reopen.MANUAL_PASS, None)
        return mod.Scores()
    return reopen.process(state, mod, select, apply, counts, Counter(), Counter())


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--out-stems", default=None, type=click.Path(path_type=Path),
              help="Write the queued stems here, for the round's STEMS_FROM.")
@click.option("--apply", is_flag=True, help="Write. Without it: census only.")
def main(out_stems, apply) -> None:
    reopen = load_script(_HERE / "14_reopen_loop_clips.py", "reopen_for_hgvs")
    sync = load_script(RELEASE / "03_sync_hotfix_results.py", "sync_for_hgvs")
    mod = reopen._load_pass()
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")
    for stage in STAGES:
        path = ROOT / stage / "generated_dataset.jsonl"
        if path.exists():
            logger.info(f"{stage}: {rewrite_stage(path, apply)} row(s) rewritten")
    for _, target in sync.HOTFIX_SOURCES:
        path = RELEASE / target / "full.jsonl"
        if path.exists():
            logger.info(f"{target}/full.jsonl: {rewrite_stage(path, apply, STATE_FIELDS)} row(s) rewritten")
    counts, stems = Counter(), set()
    for src, _ in sync.HOTFIX_SOURCES:
        state = (RELEASE / src).resolve()
        if state.exists():
            found = reopen_hgvs(state, reopen, mod, apply, counts)
            logger.info(f"{state.parent.name}/{state.name}: {len(found)} clip(s) marked pending_tts")
            stems.update(found)
    logger.info(f"{counts['rows']} state row(s) read, {counts['reopened']} marked, "
                f"{len(stems)} distinct clip(s)")
    for key in sorted(k for k in counts if k.startswith(("from::", "left::"))):
        logger.info(f"    {key}: {counts[key]}")
    if out_stems:
        path = out_stems if out_stems.is_absolute() else _HERE / out_stems
        path.write_text("".join(f"{s}\n" for s in sorted(stems)), encoding="utf-8")
        logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
