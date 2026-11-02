#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click>=8.1",
# ]
# ///
"""Audit the spelled numbers the label rules leave behind.

The release writes quantities in digits (``utils/label_conventions.py``), and each rule
there covers one shape. This report lists the shapes that still carry a spelled number
in the shipped labels, so a new rule is decided from counts rather than from memory:

  a. spelled numbers right before a unit of ``_QTY_UNITS`` (dosage forms included),
     split into a lone ``un`` / ``une`` (often left on purpose: ``Un an plus tard``,
     ``une fois``) and every other value;
  b. elided or contracted ones before a unit (``d'un``, ``qu'un``, ``l'une``,
     ``jusqu'à un``);
  c. spelled numbers before ``pour cent`` / ``pour mille`` / ``%`` / ``sur dix|vingt|cent``;
  d. one label writing the same unit with digits AND with a spelled number
     (``2 comprimés ... un comprimé``);
  e. the words that most often follow a spelled number of two or more (``deux
     patients``), a pointer to recurring shapes none of the above names.

Each with its count and a few examples. Reads only. ``--conventions`` first runs every
label through ``percent_to_symbol`` and ``apply_label_conventions`` (what 05 would ship
with the current rules).

    uv run audit_number_words.py
    uv run audit_number_words.py --conventions --manifest data/NeMO_files/full.jsonl

Written with Claude Code.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from label_conventions import (  # noqa: E402
    _QTY_SEQ, _QTY_UNITS, _words, apply_label_conventions, parse_french_number,
)
from percent_normalize import percent_to_symbol  # noqa: E402

DEFAULT_MANIFESTS = (
    _HERE / "data" / "NeMO_files" / "full.jsonl",
    _HERE.parent.parent / "UltiMed-ASR-FR-v1-NeMo_training_scripts" / "perso"
    / "downloaded_datasets_ignore-backups" / "commonvoice" / "cv_fr_train.converted.jsonl",
)

_UNIT = rf"(?i:{_QTY_UNITS})\b"
SPELLED_UNIT_RE = re.compile(rf"({_QTY_SEQ})\s+({_UNIT})")
ELIDED_UNIT_RE = re.compile(rf"(?i:\b(?:d|qu|l|jusqu)['’]\s*(?:à\s+)?une?)\s+({_UNIT})")
PERCENT_RE = re.compile(rf"({_QTY_SEQ})\s+(pour\s+cent\b|pourcents?\b|pour\s+mille\b|%|sur\s+(?:dix|vingt|cent)\b)")
DIGIT_UNIT_RE = re.compile(rf"(?<![\w,])\d+(?:,\d+)?\s+({_UNIT})")
FOLLOWER_RE = re.compile(rf"({_QTY_SEQ})\s+([^\W\d_]+)")
_UNIT_ONLY_RE = re.compile(rf"^{_UNIT}$")


def number_of(run: str) -> int | None:
    """The value a run of number words spells, a leading sentence ``et`` dropped."""
    words = _words(run)
    while words and words[0].lower() == "et":
        words.pop(0)
    return parse_french_number(words) if words else None


def unit_stem(unit: str) -> str:
    return unit.lower().rstrip("s") if unit.lower() != "mois" else "mois"


def audit_text(text: str, report: dict[str, Counter], examples: dict[str, list]) -> None:
    def hit(bucket: str, key: str, snippet: str) -> None:
        report[bucket][key] += 1
        if len(examples[bucket, key]) < 5:
            examples[bucket, key].append(snippet)

    spelled_units: set[str] = set()
    for m in SPELLED_UNIT_RE.finditer(text):
        value = number_of(m.group(1))
        if value is None:
            continue
        spelled_units.add(unit_stem(m.group(2)))
        bucket = "a. lone un/une before a unit" if value == 1 else "a. value >= 2 before a unit"
        hit(bucket, unit_stem(m.group(2)), text[max(0, m.start() - 30):m.end() + 20])
    for m in ELIDED_UNIT_RE.finditer(text):
        form = text[m.start():m.start(1)].strip().lower().replace("’", "'")
        hit("b. elided one before a unit", form, text[max(0, m.start() - 30):m.end() + 20])
    for m in PERCENT_RE.finditer(text):
        if number_of(m.group(1)) is not None:
            hit("c. spelled number before a percent or score", " ".join(m.group(2).lower().split()),
                text[max(0, m.start() - 30):m.end() + 20])
    digit_units = {unit_stem(m.group(1)) for m in DIGIT_UNIT_RE.finditer(text)}
    for stem in sorted(spelled_units & digit_units):
        hit("d. same unit in digits and spelled", stem, text[:160])
    for m in FOLLOWER_RE.finditer(text):
        value = number_of(m.group(1))
        if value is None or value < 2 or _UNIT_ONLY_RE.match(m.group(2)):
            continue
        hit("e. word after a spelled number >= 2", m.group(2).lower(), text[max(0, m.start() - 30):m.end() + 20])


def print_report(name: str, rows: int, report: dict[str, Counter], examples: dict, top: int) -> None:
    click.echo(f"\n=== {name}: {rows} labels")
    for bucket in sorted(report):
        counter = report[bucket]
        click.echo(f"\n{bucket}: {sum(counter.values())} occurrence(s), {len(counter)} key(s)")
        for key, count in counter.most_common(top):
            click.echo(f"  {count:6d}  {key}")
            for snippet in examples[bucket, key]:
                click.echo(f"            | {snippet}")


@click.command()
@click.option("--manifest", "manifests", multiple=True, type=click.Path(path_type=Path),
              help="NeMo manifest(s) to read, repeatable. Default: the release full.jsonl "
                   "and the training repo's converted Common Voice fr manifest.")
@click.option("--conventions", is_flag=True,
              help="Run every label through the label rules 05 applies first.")
@click.option("--top", default=12, show_default=True, help="Keys shown per bucket.")
def main(manifests: tuple[Path, ...], conventions: bool, top: int) -> None:
    """Count the spelled numbers the label rules leave behind."""
    for path in manifests or DEFAULT_MANIFESTS:
        if not path.exists():
            click.echo(f"skip {path}: not found", err=True)
            continue
        report: dict[str, Counter] = defaultdict(Counter)
        examples: dict = defaultdict(list)
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                if conventions:
                    text = apply_label_conventions(percent_to_symbol(text))
                audit_text(text, report, examples)
                rows += 1
        print_report(path.name, rows, report, examples, top)


if __name__ == "__main__":
    main()
