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
* every release manifest (``nemo_manifest.MANIFEST_GLOBS``: the ``full.jsonl`` files
  AND the splits, down-samples and per-source files the fixtures and training read).
  The stt pass takes a clip's text from the ``HOTFIX_SOURCES`` targets' ``full.jsonl``
  and drops a stored state row whose text differs as stale, marker included: rewriting
  the state alone was silently undone by the next round on 2026-10-30 (425 of 430
  clips never redrawn). The splits matter as much: until 2026-11-17 only the two
  ``full.jsonl`` were rewritten, and a round run with ``REFRESH_MANIFESTS=0`` left the
  329 re-synthesised clips' old text in ``test.jsonl``, ``train.jsonl`` and the rest
  (``rewrite_manifests`` alone repairs that, see 21's ``--manifests-only``);
* the stage-06 state files (``text``, ``asr_training_source``, ``asr_training_target``),
  which the tts pass reads its text from. A row whose source changed is marked
  ``pending_tts`` with no reference score, so a draw replaces the audio only once it
  clears every gate against the new text; a clip no draw fixes ends ``exhausted`` and the
  release drops it, since its audio still says the old operator.

Rows already reading ``vers`` change nothing and are not queued. The release's
``05_normalize_text.py`` applies the same rule to the manifests, so a stage file this
script missed cannot ship the old form either.

The chain itself (which files, which fields, the marker) does not depend on the rewrite:
``reopen_chain`` takes the text rewrite as an argument, and every function below it does
too, with ``normalize_hgvs`` as the default so this script's own behaviour is unchanged.
``21_reopen_resynth_typos.py`` reuses it with the held-back spelling table
(``RESYNTH_TYPOS``), which is why the chain lives here once rather than in a copy.

    uv run 17_reopen_hgvs.py                       # census
    uv run 17_reopen_hgvs.py --apply --out-stems hgvs_stems.txt

This file was written by Claude Code.
"""
from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
RELEASE = ROOT / "99_hf_release"
sys.path.insert(0, str(ROOT / "utils"))
from label_conventions import normalize_hgvs  # noqa: E402
from nemo_manifest import MANIFEST_GLOBS, rewrite_jsonl  # noqa: E402
from script_module import load_script  # noqa: E402

STAGES = ("01_dictionnary", "02_drugs", "03_PARHAF", "04_PARROT", "07_acronyms")
STAGE_FIELDS = ("asr_training_target", "asr_training_source")
STATE_FIELDS = ("text",) + STAGE_FIELDS

#: A text -> text rewrite, idempotent (a second run of the chain must change nothing).
Rewrite = Callable[[str], str]
#: Called once per state row the rewrite changed, with the row's ``STATE_FIELDS`` as they
#: were BEFORE the rewrite and the row after it, so a caller can tell which rule fired.
Observer = Callable[[dict, dict], None]


def rewrite_fields(row: dict, fields: tuple[str, ...], rewrite: Rewrite = normalize_hgvs) -> bool:
    """Apply `rewrite` to `fields` of `row` in place; whether the TTS source changed.

    The return value is about ``asr_training_source`` alone because that is what the TTS
    reads: a label-only change needs no new audio, so it must not queue a redraw.
    """
    spoken = row.get("asr_training_source")
    for field in fields:
        if isinstance(row.get(field), str):
            row[field] = rewrite(row[field])
    return row.get("asr_training_source") != spoken


def rewrite_stage(path: Path, apply: bool, fields: tuple[str, ...] = STAGE_FIELDS,
                  rewrite: Rewrite = normalize_hgvs) -> int:
    """Rewrite `fields` of one jsonl file, streamed; returns how many rows changed."""
    return rewrite_jsonl(path, lambda row: rewrite_fields(row, fields, rewrite), apply)


def reopen_rows(state: Path, reopen, mod, apply: bool, counts: Counter,
                rewrite: Rewrite = normalize_hgvs, observe: Observer | None = None) -> list[str]:
    """Rewrite one state file's texts and mark the rows whose source changed.

    A manual pass is lifted from a changed row: the take a human listened to says the old
    text, so it no longer matches the label and must not be protected from the redraw.
    `observe` sees each changed row (old fields, new row) whether or not it gets marked.
    """
    def select(row: dict):
        before = {f: row.get(f) for f in STATE_FIELDS}
        if not rewrite_fields(row, STATE_FIELDS, rewrite):
            return None
        if observe is not None:
            observe(before, row)
        (row.get("improvement") or {}).pop(reopen.MANUAL_PASS, None)
        return mod.Scores()
    return reopen.process(state, mod, select, apply, counts, Counter(), Counter())


def rewrite_manifests(rewrite: Rewrite, apply: bool, root: Path = ROOT) -> dict[Path, int]:
    """Apply `rewrite` to ``STATE_FIELDS`` of every release manifest; changed rows per file.

    Every file of ``MANIFEST_GLOBS`` (the list the release scripts 03 to 06 iterate), so a
    split never keeps a text its ``full.jsonl`` lost. Touches nothing else: no stage file,
    no state file, nothing queued, which is what makes it safe to run on its own.
    """
    release = root / "99_hf_release"
    paths = sorted(p for pattern in MANIFEST_GLOBS for p in release.glob(pattern))
    counts = {p: rewrite_stage(p, apply, STATE_FIELDS, rewrite) for p in paths}
    for path, n in counts.items():
        if n:
            logger.info(f"{path.relative_to(release)}: {n} row(s) rewritten")
    logger.info(f"{sum(counts.values())} manifest row(s) rewritten in {sum(map(bool, counts.values()))} "
                f"of {len(counts)} file(s)")
    return counts


def reopen_chain(rewrite: Rewrite, apply: bool, out_stems: Path | None, tag: str,
                 root: Path = ROOT, observe: Observer | None = None) -> set[str]:
    """Run `rewrite` over the stage files, every release manifest and the state files,
    marking every state row whose TTS source changed ``pending_tts``; returns the stems.

    Parameters
    ----------
    rewrite : Rewrite
        The idempotent text rewrite (``normalize_hgvs`` here).
    apply : bool
        Write the files; without it everything is read and counted, nothing written.
    out_stems : Path or None
        Where to write the marked stems (one per line), the driver's ``STEMS_FROM``.
        Relative paths are taken from ``06_hotfixes/``, where the driver runs. Written
        in a dry run too, so the list can be inspected before ``--apply``.
    tag : str
        Suffix for the module names the helpers are loaded under (``load_script`` needs
        one unique per caller).
    root : Path
        Repo root; a test points it at a temp tree laid out like the repo.
    observe : Observer or None
        Forwarded to `reopen_rows`, for a per-rule census.
    """
    here, release = root / "06_hotfixes", root / "99_hf_release"
    reopen = load_script(_HERE / "14_reopen_loop_clips.py", f"reopen_for_{tag}")
    sync = load_script(RELEASE / "03_sync_hotfix_results.py", f"sync_for_{tag}")
    mod = reopen._load_pass()
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")
    for stage in STAGES:
        path = root / stage / "generated_dataset.jsonl"
        if path.exists():
            logger.info(f"{stage}: {rewrite_stage(path, apply, rewrite=rewrite)} row(s) rewritten")
    rewrite_manifests(rewrite, apply, root)
    counts, stems = Counter(), set()
    for src, _ in sync.HOTFIX_SOURCES:
        state = (release / src).resolve()
        if state.exists():
            found = reopen_rows(state, reopen, mod, apply, counts, rewrite, observe)
            logger.info(f"{state.parent.name}/{state.name}: {len(found)} clip(s) marked pending_tts")
            stems.update(found)
    logger.info(f"{counts['rows']} state row(s) read, {counts['reopened']} marked, "
                f"{len(stems)} distinct clip(s)")
    for key in sorted(k for k in counts if k.startswith(("from::", "left::"))):
        logger.info(f"  {key}: {counts[key]}")
    if out_stems:
        path = out_stems if out_stems.is_absolute() else here / out_stems
        path.write_text("".join(f"{s}\n" for s in sorted(stems)), encoding="utf-8")
        logger.info(f"wrote {path}")
    return stems


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--out-stems", default=None, type=click.Path(path_type=Path),
              help="Write the queued stems here, for the round's STEMS_FROM.")
@click.option("--apply", is_flag=True, help="Write. Without it: census only.")
def main(out_stems, apply) -> None:
    reopen_chain(normalize_hgvs, apply, out_stems, "hgvs")


if __name__ == "__main__":
    main()
