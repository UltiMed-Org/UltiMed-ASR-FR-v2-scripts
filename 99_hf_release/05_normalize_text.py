#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Normalize the transcripts of every NeMo manifest of the release.

Run AFTER ``04_drop_bad_rows.py --apply`` and BEFORE ``scripts/build_parquet.py``:

    uv run 05_normalize_text.py            # dry run, reports counts + examples
    uv run 05_normalize_text.py --apply
    uv run 05_normalize_text.py --stages [--apply]   # the stage generated_dataset.jsonl labels

Each row carries two texts, and they are normalized differently because they serve
different readers:

- ``text`` is the ASR label, what the model learns to WRITE. A percentage is written
  ``%`` there: ``95 pour cent`` and ``95 pourcent`` both become ``95 %``. UltiMed v1
  mixed the three spellings (16.6k ``pour cent``, 0.4k ``pourcent``, 11.6k ``%``), so
  the model learnt no consistent form.
- ``asr_training_source`` is what the TTS READ. There ``pour cent`` and ``%`` both
  become the single-word ``pourcent``, the form ``voxtral_normalize`` gives a fresh
  source. The audio itself is unchanged: both spellings are spoken
  identically, so this only makes the shipped source text consistent with its audio.

Both rewrites live in ``utils/percent_normalize.py``, which documents the guards that
keep a "per hundred X" (``106 garçons pour cent filles``, ``pour cent millilitres``) and
the vol% / g% units. The text generators apply the label rewrite themselves when they
parse the LLM output, so a fresh generation already writes ``%``; 05 exists for the v1
manifests, generated before that. Rows left with a "pour cent" are reported so they can
be eyeballed (26 in UltiMed v1, all "per hundred X" or units).

``text`` also goes through ``ParakeetTokenizer.clean_label`` (``utils/parakeet_tokenizer.py``),
which collapses whitespace runs (6 v1 labels held a raw newline, an artefact of the source
document) and replaces every character the Parakeet vocab does not cover AS WRITTEN by
its NFKC form: ``CO₂`` -> ``CO2``, ``Brª`` -> ``Bra``, a decomposed ``c`` + combining
cedilla -> ``ç``, with trademark-like signs dropped (``CUBE™`` -> ``CUBE``), and repairs
cp1252 mojibake (U+009C + ``il`` -> ``œil``, seen in a hand-made manifest). None of these
ever produced ``<unk>`` (SentencePiece NFKC-folds them at training time), but the label
then showed a character the model is never taught. The generators apply it at parse
time too.
``asr_training_source`` is left alone there: the TTS read it, and Parakeet never does.

Both texts then go through ``utils/label_conventions.apply_label_conventions``: one
written form for titles (``M. Dupont`` -> ``Monsieur Dupont``, ``Dr`` -> ``docteur``),
dates (``le quinze mars deux mille vingt`` -> ``le 15 mars 2020``, ``premier mai`` ->
``1er mai``), clock times (``14h30`` -> ``14 heures 30``), quantities (``quatre
milligrammes`` -> ``4 milligrammes``, ``un comprimé`` -> ``1 comprimé``, ``Un an plus
tard`` kept), compounds
(``petit déjeuner`` -> ``petit-déjeuner``) and drug names (``PRIMPERAN`` / ``Primperan``
-> ``Primpéran``, ``UVEDOSE`` -> ``Uvedose``, from
``utils/drug_casing.json``). The source gets them too because each rewrite is spoken
exactly like the original (``M.`` was read ``monsieur``, casing is silent), so the
shipped source keeps matching both the audio and what ``voxtral_normalize`` derives
from the new label. The generators apply the same rewrites at parse time.
The spelling fixes are the exception, applied to ``text`` ONLY: typos (``PERINDIPRIL`` ->
``périndopril``), INN accents (``bétaméthasone`` -> ``betamethasone``) and spelling winners
(``ciclosporine`` -> ``cyclosporine``) change what a word looks like, and for a typo what it
sounds like, so the source keeps what voxtral actually read (``tts_source=True`` skips them).

The hand-made NeMo-repo sets (``perso/drug_sentence_dataset/*.json``, the private
``perso/oli_spoken_dataset`` manifests) are normalized by passing them explicitly:
``uv run 05_normalize_text.py --apply <manifest> ...``.

Each distinct (text, source) pair is normalized once over all manifests, on ``--jobs``
processes, and every manifest is rewritten independently, like 04, and the run is idempotent.

This file was written by Claude Code.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import MANIFEST_GLOBS, read_jsonl, rewrite_jsonl, write_jsonl  # noqa: E402
from script_module import load_script  # noqa: E402
from label_conventions import apply_label_conventions  # noqa: E402
from parakeet_tokenizer import ParakeetTokenizer  # noqa: E402
from percent_normalize import (  # noqa: E402
    LEFTOVER_RE, percent_sign_to_one_word, percent_to_one_word, percent_to_symbol,
)

def normalize_label(text: str, tok: ParakeetTokenizer) -> str:
    """The ASR label form: character fold, ``%``, then every label convention."""
    return apply_label_conventions(percent_to_symbol(tok.clean_label(text)))


def normalize_row(row: dict, tok: ParakeetTokenizer) -> dict:
    """Return a copy of ``row`` with ``text`` and ``asr_training_source`` normalized.

    ``tok`` is a ``ParakeetTokenizer``; its ``clean_label`` checks coverage raw, so
    its ``nfkc`` setting does not matter here.
    """
    out = dict(row)
    out["text"] = normalize_label(row["text"], tok)
    if "asr_training_source" in row:
        # One space between words here too, without clean_label's character fold.
        out["asr_training_source"] = apply_label_conventions(" ".join(percent_sign_to_one_word(
            percent_to_one_word(row["asr_training_source"])).split()), tts_source=True)
    return out


_TOK: ParakeetTokenizer | None = None


def _pair_key(row: dict) -> tuple[str, str | None]:
    """``normalize_row`` reads only these two fields, so equal keys give equal results."""
    return row["text"], row.get("asr_training_source")


def _normalize_key(key: tuple[str, str | None]) -> tuple[tuple, tuple[str, str | None]]:
    global _TOK
    if _TOK is None:
        _TOK = ParakeetTokenizer(nfkc=False)
    row = {"text": key[0]} if key[1] is None else {"text": key[0], "asr_training_source": key[1]}
    return key, _pair_key(normalize_row(row, _TOK))


def normalize_keys(keys: set, jobs: int) -> dict:
    """``{(text, source): (new text, new source)}``, each distinct pair normalized once.

    The release manifests repeat every row three times (``full``, the split files, the
    per-subset files), and the conventions cost about 2 ms a text, so this plus ``jobs``
    processes takes the whole corpus from hours to about 4 minutes.
    """
    if jobs <= 1:
        return dict(map(_normalize_key, keys))
    with Pool(jobs) as pool:
        return dict(pool.imap_unordered(_normalize_key, list(keys), chunksize=500))


def normalize_stage_targets(paths: list[Path], apply: bool, jobs: int) -> dict[Path, int]:
    """``--stages``: the label normalization applied to ``asr_training_target`` of each
    stage ``generated_dataset.jsonl``; returns the changed-row count per file.

    The release manifests are rebuilt from those files (``01_build_nemo_manifest.py``
    copies ``asr_training_target`` into ``text`` as is, the hotfix driver reruns it every
    round), so a rewrite of the manifests alone is undone by the next rebuild. Only that
    field changes: ``asr_training_source`` (what the TTS read) and every other field and
    row stay, unchanged lines byte for byte (``rewrite_jsonl``). No row is ever dropped,
    unlike ``07_write_back_texts.py``, which drops rows missing from the manifests. The
    audio says the same thing before and after, so nothing is queued for a redraw.
    """
    keys = {(r["asr_training_target"], None) for p in paths for r in read_jsonl(p)
            if isinstance(r.get("asr_training_target"), str)}
    done = normalize_keys(keys, jobs)

    def rewrite(row: dict) -> None:
        if isinstance(row.get("asr_training_target"), str):
            row["asr_training_target"] = done[(row["asr_training_target"], None)][0]

    return {p: rewrite_jsonl(p, rewrite, apply) for p in paths}


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
@click.option("--jobs", default=os.cpu_count() or 1, show_default="all cores")
@click.option("--examples", default=10, show_default=True, help="Changed labels to print.")
@click.option("--stages", is_flag=True,
              help="Normalize asr_training_target of the stage generated_dataset.jsonl files "
                   "(the ones 01_build_nemo_manifest.py reads) instead of the manifests.")
def main(manifests: tuple[Path, ...], root: str, apply: bool, jobs: int, examples: int,
         stages: bool = False) -> None:
    """Normalize percent spellings, whitespace and uncovered characters in the manifests."""
    if stages:
        build = load_script(_HERE / "01_build_nemo_manifest.py", "build_manifest_for_normalize")
        paths = [_HERE.parent / stage["input"] for stage in build.STAGES]
        for path, n in normalize_stage_targets(paths, apply, jobs).items():
            logger.info(f"{path.relative_to(_HERE.parent)}: asr_training_target changed in {n}")
        if not apply:
            logger.warning("dry run, nothing written (pass --apply)")
        return
    root_path = Path(root).resolve()
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    if not targets:
        raise SystemExit(f"no manifest found under {root_path} (is the `data` symlink mounted?)")

    tok = ParakeetTokenizer(nfkc=False)
    targets = [p for p in targets if p.is_file() and not p.name.startswith(".")]
    keys = {_pair_key(r) for path in targets for r in read_jsonl(path)}
    logger.info(f"{len(targets)} manifests, {len(keys)} distinct (text, source) pairs, {jobs} jobs")
    done = normalize_keys(keys, jobs)
    for (text, _), (new_text, _) in [kv for kv in done.items() if kv[0][0] != kv[1][0]][:examples]:
        logger.info(f"  - {text[:160]}\n    + {new_text[:160]}")
    for path in targets:
        rows = read_jsonl(path)
        label = path.relative_to(root_path) if path.is_relative_to(root_path) else path
        new_rows = []
        for r in rows:
            text, source = done[_pair_key(r)]
            new = dict(r, text=text)
            if source is not None:
                new["asr_training_source"] = source
            new_rows.append(new)
        changed = Counter()
        leftovers, still_bad = [], Counter()
        for old, new in zip(rows, new_rows):
            for key in ("text", "asr_training_source"):
                if old.get(key) != new.get(key):
                    changed[key] += 1
            if LEFTOVER_RE.search(new["text"]):
                leftovers.append(new["text"])
            for ch in tok.offending_chars(new["text"]):
                still_bad[ch] += 1
        if not changed:
            continue
        logger.info(f"{label}: {len(rows)} rows, text changed in {changed['text']}, "
                    f"asr_training_source changed in {changed['asr_training_source']}")
        if leftovers:
            logger.info(f"    {len(leftovers)} labels keep a non-percent 'pour cent', e.g. "
                        f"{leftovers[0][:120]!r}")
        if still_bad:
            logger.warning(f"    still uncovered after the fix: {dict(still_bad)}")
        if apply:
            write_jsonl(path, new_rows)
    if not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    else:
        logger.info("now rebuild the parquet: uv run scripts/build_parquet.py")


if __name__ == "__main__":
    main()
