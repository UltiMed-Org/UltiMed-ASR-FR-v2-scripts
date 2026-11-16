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
"""Put the correct spelling into the held-back clips and queue them for re-synthesis.

Some label typos change the SOUND of the word (``PERINDIPRIL``, ``Cetriaxone``,
``KAREGIC``, ``empagliflozin`` read with a nasal ``-in``): voxtral read the typo, so the
clip's audio says it, and fixing only the label would make the label lie. The owner's
rule (2026-11-16) is that every such fix is held back from the label tables until its
clips are re-synthesized; the held table is ``RESYNTH_TYPOS`` in
``07_build_accent_spellings.py`` and the clip list is in ``docs/BUILD_LOG.md`` ("Label
spelling fixes after the v2 model"). This script is step 1 of that re-synthesis: it
rewrites each affected clip's text, label AND the TTS source, to the corrected spelling
and marks it ``pending_tts``, so the next driver round redraws it from the right text
and the normal QC gates judge the new take against the new label.

It is ``17_reopen_hgvs.py``'s chain with another rewrite, reused through
``reopen_chain`` (same files, same fields, same marker, same stale-row pitfall, read
17's docstring for why each file matters). The rewrite is
``label_conventions.normalize_accents(text, table)``, which keeps a capital and elides
the article when the fix makes the word vowel-initial (``le cestradiol`` ->
``l'estradiol``). On top of it, ``--generic-labs`` (default on) strips the generic lab
names ``strip_generic_labs`` drops at export: one clip,
``PARHAF/011560_0000_chirurgie_viscerale_00018_t0_c1``, says ``Simvastatine Téva`` while
its exported label drops ``Téva`` (BUILD_LOG, same section), and it rides along so one
round fixes both. The 2026-11-16 check found no other clip it changes.

The census logs, per table key, how many state rows the rewrite changed and their stems,
then every key that matched nothing. One key is expected to match nothing, as BUILD_LOG
says: ``actisenan`` (its clip spells it ``Actisénan``, also a key). The 2026-11-16 dry run
found 60 clips (59 typo clips + the Téva one) and no other zero-hit key; a key with more
clips than the BUILD_LOG table lists means new clips carry it.

    uv run 21_reopen_resynth_typos.py --out-stems resynth_stems.txt    # census, writes the list
    uv run 21_reopen_resynth_typos.py --apply --out-stems resynth_stems.txt

``--table`` takes another JSON table (``{"typo": "fix"}``, or the ``{"rewrites": ...}``
shape of ``utils/accent_spellings.json``), for a later batch held the same way.

Never ``--apply`` while something else rewrites the release manifests
(``99_hf_release/05_normalize_text.py --apply``, a driver round): both stream the same
``full.jsonl`` through a temp file and the last ``os.replace`` wins.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from label_conventions import normalize_accents, strip_generic_labs  # noqa: E402
from script_module import load_script  # noqa: E402

chain = load_script(_HERE / "17_reopen_hgvs.py", "t17_for_resynth")

#: Census key for the clips the generic-lab strip changed (not a table key).
GENERIC_LAB = "<generic lab name>"


def default_table() -> dict[str, str]:
    """``RESYNTH_TYPOS`` from ``07_build_accent_spellings.py``, the one place it is kept."""
    return load_script(_HERE / "07_build_accent_spellings.py", "accent_for_resynth").RESYNTH_TYPOS


def load_table(path: Path | None) -> dict[str, str]:
    """The ``typo -> fix`` table, keys lowercased as ``normalize_accents`` looks them up.

    Parameters
    ----------
    path : Path or None
        A JSON file, either a flat ``{"typo": "fix"}`` or ``{"rewrites": {...}}``;
        None means ``default_table()``.
    """
    if path is None:
        table = default_table()
    else:
        table = json.loads(path.read_text(encoding="utf-8"))
        table = table.get("rewrites", table)
    return {k.lower(): v for k, v in table.items()}


def make_rewrite(table: dict[str, str], generic_labs: bool):
    """The text rewrite handed to the chain: optional lab strip, then the spelling table.

    The lab strip runs first so a lab name never shields or splits a table word. Both
    halves are idempotent, so a second run (or a resumed one) changes nothing.

    >>> make_rewrite({"cestradiol": "estradiol"}, True)("le cestradiol et Simvastatine Téva")
    "l'estradiol et Simvastatine"
    """
    def rewrite(text: str) -> str:
        return normalize_accents(strip_generic_labs(text) if generic_labs else text, table)
    return rewrite


def make_census(table: dict[str, str], generic_labs: bool):
    """An observer for ``reopen_chain`` plus the dict it fills, ``key -> set of stems``.

    A changed row is credited to every table key found in its OLD label or source (whole
    word, case-insensitive, the way ``normalize_accents`` looks words up), and to
    ``GENERIC_LAB`` when the lab strip alone changes it. A row the rewrite changed with
    no key found would be a rule this census does not know; it lands under ``"?"`` so it
    cannot go unseen.
    """
    keys = sorted(table, key=len, reverse=True)
    key_re = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, keys)) + r")(?![\w-])", re.IGNORECASE)
    hits: dict[str, set[str]] = defaultdict(set)

    def observe(before: dict, row: dict) -> None:
        stem = Path(row.get("audio_filepath") or "").stem
        olds = [v for v in before.values() if isinstance(v, str)]
        found = {m.group(1).lower() for text in olds for m in key_re.finditer(text)}
        if generic_labs and any(strip_generic_labs(t) != t for t in olds):
            found.add(GENERIC_LAB)
        for key in found or {"?"}:
            hits[key].add(stem)
    return observe, hits


def report(table: dict[str, str], hits: dict[str, set[str]]) -> None:
    """Log the per-key census and the keys that matched no clip."""
    logger.info(f"{'key':<34} {'fix':<32} clips")
    for key in sorted(hits, key=lambda k: (-len(hits[k]), k)):
        logger.info(f"{key:<34} {table.get(key, ''):<32} {len(hits[key]):>4}  {' '.join(sorted(hits[key]))}")
    unmatched = sorted(set(table) - set(hits))
    logger.info(f"{len(unmatched)} key(s) matched no clip: {', '.join(unmatched) or '-'}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--table", "table_path", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="JSON typo -> fix table. Default: RESYNTH_TYPOS of 07_build_accent_spellings.py.")
@click.option("--generic-labs/--no-generic-labs", default=True, show_default=True,
              help="Also strip the generic lab names the export drops (the Téva clip).")
@click.option("--out-stems", default=None, type=click.Path(path_type=Path),
              help="Write the queued stems here (relative to 06_hotfixes/), for the round's STEMS_FROM.")
@click.option("--apply", is_flag=True, help="Write. Without it: census only.")
def main(table_path: Path | None, generic_labs: bool, out_stems: Path | None, apply: bool) -> None:
    table = load_table(table_path)
    logger.info(f"{len(table)} held spelling(s){' + generic lab names' if generic_labs else ''}")
    observe, hits = make_census(table, generic_labs)
    stems = chain.reopen_chain(make_rewrite(table, generic_labs), apply, out_stems, "resynth",
                               observe=observe)
    report(table, hits)
    logger.info(f"{len(stems)} clip(s) {'queued' if apply else 'would be queued'}")


if __name__ == "__main__":
    main()
