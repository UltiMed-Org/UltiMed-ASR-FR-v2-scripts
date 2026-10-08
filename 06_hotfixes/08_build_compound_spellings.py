#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""Build ``utils/compound_spellings.json``: one written form per compound the labels
glue, hyphenate or space several ways.

``utils/label_conventions.normalize_compounds`` applies the general glue rule, but it
cannot split a word that was already glued (``transsphénoïdale``) nor choose between two
spellings it cannot tell apart (``gastro-entérologie`` 1587 / ``gastroentérologie`` 987,
``tep scanner`` / ``tep-scanner``). ``05_label_variant_audit.py`` run on the labels AFTER
that rule lists those ``hyphen`` families; this script picks a canonical form per family
and writes ``observed form -> canonical``, which
``utils/label_conventions.normalize_compound_spellings`` applies. Made 2026-10-08 for
UltiMed v2.

Canonical form, author's choices (2026-10-08), in order:

1. ``OVERRIDES``: imaging (``TEP-scanner``, ``DaTscan``, ``SUVmax``, ``PICC line``),
   English terms in their dictionary form (``check-list``, ``bypass``, ``Mini-Mental``),
   Latin locutions hyphenated (``post-partum``), drug combinations hyphenated and salts
   spaced (``sacubitril-valsartan``, ``fosfomycine trométamol``), and a few dictionary
   forms (``rémittente-récurrente``, ``diamniotique``, ``kala-azar``, ``moyamoya``).
2. Rules, applied at the boundary between two parts:
   - a doubled letter keeps a hyphen (``trans-sphénoïdale``, ``hyper-réflectifs``);
   - ``sous`` / ``sus`` before an adjective take a hyphen (``sous-costale``,
     ``sus-pubienne``); before a noun the family is left alone (``sous traitement``);
   - ``sur`` is glued (``surinfection``, ``surexposition``, ``sursoit``);
   - ``post`` / ``pré`` / ``per`` follow ``normalize_compounds``: a space before a noun
     (``pré admission``), glued before an adjective (``préhospitalier``);
   - ``alpha`` / ``bêta`` take a hyphen (``bêta-lactamines``, ``alpha-bloquant``);
   - ``écho`` takes a hyphen before a vowel or a noun (``écho-endoscopie``,
     ``écho-doppler``) and is glued before an adjective (``échoguidée``).
3. Otherwise the most frequent form (``thromboembolique``, ``neuro-imagerie``,
   ``ponction biopsie``).

Elided forms count with their word (``l'écho-doppler`` adds to ``écho-doppler``); the
applier strips the elision before its lookup. Families are skipped when two real words
can be mistaken for one (the audit's ``homograph`` mark, ``NOISE``: ``bassin et`` /
``bassinet``, ``la tente`` / ``latente``), when a part is a function word or 1 to 2
letters (``t au``, ``résiduel le``, ``pd-l``: the lab names with digits are
``label_conventions._OFFICIAL_RULES``), or when the first part is ``non`` (the
``non`` rule depends on the preceding article, ``label_conventions.normalize_non``).

Prints the families whose two top forms are within 2:1 and that no rule or override
decided, to check by hand.

Usage::

    uv run 06_hotfixes/08_build_compound_spellings.py path/to/variants-v2.tsv

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
import label_conventions as lc  # noqa: E402

OVERRIDES = {
    # imaging
    "tepscanner": "TEP-scanner", "teptdm": "TEP-TDM", "datscan": "DaTscan", "suvmax": "SUVmax",
    "piccline": "PICC line",
    # English, dictionary forms
    "checklist": "check-list", "bypass": "bypass", "coldcream": "cold-cream", "peakflow": "peak-flow",
    "opencoelioscopie": "open coelioscopie", "minimental": "Mini-Mental",
    # Latin locutions
    "postpartum": "post-partum", "antepartum": "ante-partum", "perpartum": "per-partum",
    "postmortem": "post-mortem", "postabortum": "post-abortum",
    # lab: the names with digits are label_conventions._OFFICIAL_RULES
    "probnp": "proBNP",
    # drug combinations hyphenated, salts spaced
    "amoxicillineacide": "amoxicilline-acide", "sacubitrilvalsartan": "sacubitril-valsartan",
    "fosfomycinetrométamol": "fosfomycine trométamol",
    # dictionary forms
    "rémittenterécurrente": "rémittente-récurrente", "diamniotique": "diamniotique",
    "monoamniotique": "monoamniotique", "bichoriale": "bichoriale", "kalaazar": "kala-azar",
    "moyamoya": "moyamoya", "stromaréaction": "stroma-réaction", "surexposition": "surexposition",
    "surinfection": "surinfection", "nitroimidazolé": "nitro-imidazolé",
    "mésoappendiculaire": "mésoappendiculaire", "alphacalcidol": "alphacalcidol",
}
NOISE = frozenset("""bassinet latente lata duret sangle labile encéphalique vitréen lapeyronie lefort
    tau tnm ptnm pde pdeb sla cil bbb gd etv pta ret pik diffuk zymad zoneneutre nashtest""".split())
FUNCTION = frozenset("""le la les l ne en et de du des un une au aux ce cet se sa son ses il elle on y à ou
    où que qui mais pas par pour tout dans lui leur nous vous je tu ni si""".split())
ELISION = re.compile(r"^[^\W\d_]+['’]")
GREEK = frozenset({"alpha", "bêta", "béta"})


def parse_forms(field: str) -> list[tuple[str, int]]:
    """
    >>> parse_forms("check-list (341) | checklist (75)")
    [('check-list', 341), ('checklist', 75)]
    """
    out = []
    for f in field.split(" | "):
        form, n = f.rsplit(" (", 1)
        out.append((form, int(n[:-1])))
    return out


def boundaries(form: str) -> dict[int, str]:
    """``{letter offset: separator}`` of a form: ``trans-sphénoïdale`` -> ``{5: '-'}``."""
    out, pos = {}, 0
    for part, sep in re.findall(r"([^ -]+)([ -]?)", form):
        pos += len(part)
        if sep:
            out[pos] = sep
    return out


def rebuild(key: str, seps: dict[int, str]) -> str:
    return "".join(key[i - 1] + seps.get(i, "") for i in range(1, len(key) + 1))


def canonical(forms: list[tuple[str, int]]) -> tuple[str | None, str]:
    """``(canonical form or None, reason)`` for one family, most frequent form first.

    >>> canonical([("transsphénoïdale", 113), ("trans-sphénoïdale", 75)])
    ('trans-sphénoïdale', 'rule')
    >>> canonical([("sous-costale", 71), ("souscostale", 22), ("sous costale", 5)])
    ('sous-costale', 'rule')
    >>> canonical([("échoguidée", 546), ("écho-guidée", 75)]), canonical([("échodoppler", 9), ("écho-doppler", 5)])
    (('échoguidée', 'rule'), ('écho-doppler', 'rule'))
    >>> canonical([("thromboembolique", 567), ("thrombo-embolique", 58)])
    ('thromboembolique', 'majority')
    >>> canonical([("tep scanner", 548), ("tep-scanner", 424)]), canonical([("t au", 9), ("tau", 5)])
    (('TEP-scanner', 'override'), (None, 'skip'))
    """
    key = forms[0][0].replace("-", "").replace(" ", "")
    if key in OVERRIDES:
        return OVERRIDES[key], "override"
    parts = [p for f, _ in forms for p in re.split(r"[ -]", f)]
    if key in NOISE or parts[0] == "non" or any(len(p) <= 2 or p in FUNCTION for p in parts):
        return None, "skip"
    seps = dict(boundaries(forms[0][0]))
    ruled = False
    for f, _ in forms:
        for pos in boundaries(f):
            head, tail = key[:pos], key[pos:]
            if head[-1] == tail[0] and head[-1] != "é":
                seps[pos], ruled = "-", True
            elif head in ("sous", "sus"):
                if lc._is_noun(tail):
                    return None, "skip"
                seps[pos], ruled = "-", True
            elif head in GREEK:
                seps[pos], ruled = "-", True
            elif head == "sur":
                seps.pop(pos, None)
                ruled = True
            elif head in ("post", "pré", "per"):
                if lc._spaced_noun(head, tail):
                    seps[pos] = " "
                else:
                    seps.pop(pos, None)
                ruled = True
            elif head == "écho":
                vowel_or_noun = tail[0] in "aeiouéèêy" or lc._is_noun(tail)
                if vowel_or_noun:
                    seps[pos] = "-"
                else:
                    seps.pop(pos, None)
                ruled = True
    return rebuild(key, seps), "rule" if ruled else "majority"


def merge_elisions(families: list[list[tuple[str, int]]]) -> list[list[tuple[str, int]]]:
    """Add each elided form to its bare form's family: ``l'écho-doppler`` -> ``écho-doppler``."""
    by_key: dict[str, Counter] = {}
    for forms in families:
        for f, n in forms:
            bare = ELISION.sub("", f)
            by_key.setdefault(bare.replace("-", "").replace(" ", ""), Counter())[bare] += n
    return [c.most_common() for c in by_key.values() if len(c) > 1]


@click.command()
@click.argument("variants_tsv", type=click.Path(exists=True))
@click.option("--out", default=str(ROOT / "utils" / "compound_spellings.json"), show_default=True)
def main(variants_tsv, out):
    families = []
    for line in open(variants_tsv, encoding="utf-8").readlines()[1:]:
        kind, _cost, homograph, shown, _ex = line.rstrip("\n").split("\t")
        if kind == "hyphen" and homograph != "1":
            families.append(parse_forms(shown))
    table, close, reasons = {}, [], Counter()
    for forms in merge_elisions(families):
        canon, reason = canonical(forms)
        reasons[reason] += 1
        if canon is None:
            continue
        for f, _ in forms:
            if f != canon:
                table[f] = canon
        if reason == "majority" and len(forms) > 1 and forms[1][1] >= 20 and 2 * forms[1][1] >= forms[0][1]:
            close.append(forms)
    # The table runs after normalize_compounds: a canonical form it would change again
    # would make the conventions flip-flop between runs.
    # The audit lowercases, so these are mostly acronyms (``anti-ALK``, ``micro-ARN``) the
    # rule never touches in their real case; dropped to be safe.
    unstable = {c for c in table.values() if lc.normalize_compounds(c) != c}
    table = {f: c for f, c in table.items() if c not in unstable}
    click.echo(f"dropped {len(unstable)} canonicals normalize_compounds would change: {sorted(unstable)}")
    doc = ("observed form -> canonical compound spelling, built by "
           "06_hotfixes/08_build_compound_spellings.py; applied by "
           "label_conventions.normalize_compound_spellings")
    Path(out).write_text(json.dumps({"_doc": doc, "rewrites": dict(sorted(table.items()))},
                                    ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    click.echo(f"{dict(reasons)}; {len(table)} rewrites -> {out}")
    for forms in sorted(close, key=lambda fs: -fs[1][1]):
        click.echo("close: " + " | ".join(f"{f} ({n})" for f, n in forms))


if __name__ == "__main__":
    main()
