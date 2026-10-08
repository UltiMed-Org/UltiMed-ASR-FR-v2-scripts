#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Write ``convention_fixes.tsv``: the label spelling conventions as output fixes.

``utils/label_conventions.py`` rewrites the labels to one spelling per word
(``post-opératoire`` -> ``postopératoire``, ``artéfact`` -> ``artefact``, ``périndopril``
-> ``perindopril``). A model trained on older labels, or one that still slips, writes the
other forms; these rules give its output the same spelling (author's request
2026-10-08: "all the terms we fixed, add them to the regex list, for the future").

Input: the audit TSV of ``06_hotfixes/05_label_variant_audit.py``. Every form of a
``hyphen`` or ``accent`` family not marked ``homograph`` goes through
``apply_label_conventions``; a form it rewrites to the same letters (only hyphens, spaces
or accents differ) becomes ``form<TAB>canonical``. The conventions decide, so a family
they leave alone (``bassin et`` / ``bassinet``, two words) yields nothing. Notations that
need context (``à un mois`` -> ``à 1 mois``, ``phase II`` -> ``phase 2``) are not word
rules and stay in the conventions only.

The audit forms are lowercased, so a rule can lose case the labels have (``metavir`` ->
``Metavir`` would undo ``METAVIR``). With ``--labels``, every label is converted and a
rule that still changes more than ``--max-fire`` of its family's occurrences is dropped:
a converted label is canonical by definition. Not every change: a few labels shout a
drug name (``injection BEVACIZUMAB``), and the rule, which keeps only the initial's case,
would drop ``bévacizumab`` for them.

``05_build_manual_rules.py`` reads this file after ``manual_fixes.tsv`` (a hand fix
wins), so the rules ship in ``manual_fix_rules.jsonl``, applied before the drug and term
rules.

    uv run 08_drug_asr_rules/06_build_convention_fixes.py path/to/variants.tsv \
        --labels 99_hf_release/data/NeMO_files/full.jsonl --labels 99_hf_release/data/NeMO_files/PARROT/full.jsonl

This file was written by Claude Code.
"""
import importlib.util
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from label_conventions import apply_label_conventions  # noqa: E402

_spec = importlib.util.spec_from_file_location("b", _HERE / "02_build_fix_rules.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def _letters(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if c.isalpha() and unicodedata.category(c) != "Mn")


def convention_fixes(forms: list[str]) -> dict[str, str]:
    """``form -> canonical`` for one family's forms the conventions respell (same letters only).

    All or nothing: a rule pattern ignores accents and treats a hyphen like a space, so if
    the conventions spare a form (a word of its own: ``débordé`` against ``déborde``), a
    rule for its sibling would rewrite it too, and the family yields no rule.

    >>> convention_fixes(["post-opératoire", "postopératoire", "post opératoire"])
    {'post-opératoire': 'postopératoire', 'post opératoire': 'postopératoire'}
    >>> convention_fixes(["bassin et", "bassinet"]), convention_fixes(["déborde", "deborde", "débordé"])
    ({}, {})
    """
    out, kept = {}, set()
    for f in forms:
        canon = apply_label_conventions(f)
        if canon.lower() == f.lower():
            kept.add(f.lower())
        elif _letters(canon) == _letters(f):
            out[f] = canon[0].lower() + canon[1:] if f[0].islower() else canon
    canons = {v.lower() for v in out.values()}
    return out if out and kept <= canons else {}


def firing(fixes: dict[str, str], labels) -> Counter:
    """How many of ``labels``, once converted, each variant's rule changes.

    >>> firing({"metavir": "metavir", "post-opératoire": "postopératoire"}, ["Score METAVIR, post-opératoire."])
    Counter({'metavir': 1})
    """
    rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": r} for v, r in fixes.items()]
    fix_all = b.compile_rules(rules)
    out: Counter = Counter()
    for label in labels:
        text = apply_label_conventions(label)
        if fix_all(text) != text:  # rare: find the culprits one rule at a time
            out.update(r["variant"] for r in rules if b.apply_rules(text, [r]) != text)
    return out


@click.command()
@click.argument("variants_tsv", type=click.Path(exists=True))
@click.argument("out", type=click.Path(path_type=Path), default=_HERE / "convention_fixes.tsv")
@click.option("--labels", multiple=True, type=click.Path(exists=True), help="NeMo manifests: drop rules that change converted labels.")
@click.option("--max-fire", default=0.1, show_default=True, help="Drop a rule changing more than this share of its family's label occurrences.")
def main(variants_tsv, out: Path, labels, max_fire) -> None:
    fixes, family_total = {}, {}
    for line in list(open(variants_tsv, encoding="utf-8"))[1:]:
        kind, _, homo, shown, _ = line.rstrip("\n").split("\t")
        if homo == "1":
            continue
        forms = [f.rsplit(" (", 1) for f in shown.split(" | ")]
        total = sum(int(n.rstrip(")")) for _, n in forms)
        family_total.update({f: total for f, _ in forms})
        fixes.update(convention_fixes([f for f, _ in forms]))
    if labels:
        texts = (json.loads(line)["text"] for mf in labels for line in open(mf, encoding="utf-8"))
        fired = firing(fixes, texts)
        bad = {v for v, n in fired.items() if n > max_fire * family_total[v]}
        logger.info(f"dropped {len(bad)} rules that change converted labels: {sorted(bad)}")
        logger.info(f"kept despite a few changes: {sorted((v, n) for v, n in fired.items() if v not in bad)}")
        fixes = {k: v for k, v in fixes.items() if k not in bad}
    head = ("# variant<TAB>replacement, GENERATED by 06_build_convention_fixes.py from the label audit: the label\n"
            "# spelling conventions (utils/label_conventions.py) as output fixes. Do not edit, rerun the script.\n")
    out.write_text(head + "".join(f"{k}\t{v}\n" for k, v in sorted(fixes.items())), encoding="utf-8")
    logger.info(f"{len(fixes)} convention fixes -> {out}")


if __name__ == "__main__":
    main()
