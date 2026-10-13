#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Copy the manifests' final texts back into every stage's ``generated_dataset.jsonl``.

Steps 04 to 06 of this folder rewrite the texts IN THE MANIFESTS only (label
conventions, TTS respellings, names and ages spread, bad rows dropped), but stage 05
synthesizes from each stage's ``generated_dataset.jsonl``. Before the v2 audio is
generated, this script makes those files say what the manifests say: for each row it
takes ``text`` -> ``asr_training_target`` and ``asr_training_source`` from the manifest
row of the same clip, matched on (audio folder, ``term_index``, ``variant_index``) as
``nemo_manifest.parse_clip_name`` reads them from the clip file name. Every other field
(``term``, indices, ``source_id``...) is kept, so the stage-05 file names and the
voice assignment's group keys do not move.

A generated row with no manifest row is DROPPED and listed: the manifests are the
reviewed set. To keep a row that 04 dropped for its audio rather than its text, put it
in an extra manifest, run 06 and 05 over that file, and pass it here too (that is how
the 72 audio-flagged v1 rows were kept for v2, 2026-10-13). A manifest row matching no
generated row is an error, as is the same clip in two manifests with different texts.

Writes a backup of each changed file into ``--backup-dir`` first (never overwritten).

    uv run 07_write_back_texts.py data/NeMO_files/full.jsonl data/NeMO_files/PARROT/full.jsonl EXTRA.jsonl
    uv run 07_write_back_texts.py ... --apply --backup-dir DIR

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import parse_clip_name, read_jsonl, write_jsonl  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_manifest", _HERE / "01_build_nemo_manifest.py")
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)
STAGES = _build.STAGES


def manifest_texts(manifests: list[Path]) -> dict[tuple[str, int, int], tuple[str, str]]:
    """``(audio folder, term_index, variant_index) -> (text, asr_training_source)``."""
    texts: dict[tuple[str, int, int], tuple[str, str]] = {}
    for path in manifests:
        for row in read_jsonl(path):
            folder, name = Path(row["audio_filepath"]).parts[-2:]
            parsed = parse_clip_name(name)
            if parsed is None:
                raise click.ClickException(f"{path}: not a stage-05 clip name: {row['audio_filepath']}")
            key, value = (folder, parsed[0], parsed[1]), (row["text"], row["asr_training_source"])
            if texts.setdefault(key, value) != value:
                raise click.ClickException(f"{path}: {key} has two different texts across manifests")
    return texts


def write_back(rows: list[dict], folder: str, texts: dict) -> tuple[list[dict], list[dict], int, set]:
    """Returns (new rows, dropped rows, rows whose texts changed, keys used)."""
    out, dropped, changed, used = [], [], 0, set()
    for row in rows:
        key = (folder, int(row["term_index"]), int(row["variant_index"]))
        if key not in texts:
            dropped.append(row)
            continue
        used.add(key)
        target, source = texts[key]
        changed += (target, source) != (row["asr_training_target"], row["asr_training_source"])
        out.append({**row, "asr_training_target": target, "asr_training_source": source})
    return out, dropped, changed, used


@click.command()
@click.argument("manifests", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--apply", is_flag=True, help="Write. Without it, a dry run.")
@click.option("--backup-dir", type=click.Path(path_type=Path),
              help="Where the original files are copied before --apply (required with it).")
def main(manifests: tuple[Path, ...], apply: bool, backup_dir: Path | None) -> None:
    if apply and backup_dir is None:
        raise click.ClickException("--apply needs --backup-dir")
    texts = manifest_texts(list(manifests))
    used_all: set = set()
    for stage in STAGES:
        path = _HERE.parent / stage["input"]
        rows = read_jsonl(path)
        out, dropped, changed, used = write_back(rows, stage["audio"], texts)
        used_all |= used
        logger.info(f"{stage['input']}: {len(rows)} rows, {changed} texts changed, {len(dropped)} dropped")
        for row in dropped:
            logger.info(f"    dropped {row['term_index']}_{row['variant_index']}: {row['asr_training_target'][:100]}")
        if apply and (changed or dropped):
            backup = backup_dir / stage["input"]
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            write_jsonl(path, out)
    orphans = set(texts) - used_all
    if orphans:
        raise click.ClickException(f"{len(orphans)} manifest rows match no generated row, e.g. {sorted(orphans)[:3]}")
    if not apply:
        logger.warning("dry run, nothing written (pass --apply --backup-dir DIR)")


if __name__ == "__main__":
    main()
