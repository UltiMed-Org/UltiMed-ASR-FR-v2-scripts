#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "wordfreq"]
# ///
"""Build ``utils/accent_spellings.json``: one written form per word the labels accent several ways.

``05_label_variant_audit.py`` lists the ``accent`` families (``défect`` 414 / ``defect``
317, ``périndopril`` 579 / ``perindopril`` 150, ``wirsung`` 148 / ``wirsüng`` 47): one
spoken word, several spellings, so the model is taught two answers for one sound. This
picks the canonical form of each family and writes ``minority spelling -> canonical``;
``utils/label_conventions.normalize_accents`` applies it. Made 2026-10-08 for UltiMed v2.

Canonical form, in order:

1. ``OVERRIDES``: the author's choices (2026-10-08): ``défect``, ``artefact``, ``olécrane``
   (and ``olécranien``), and drug INNs WITHOUT accents (``perindopril``, ``bevacizumab``,
   ``enoxaparine``: the international spelling), for every drug word whose family the
   audit found (``is_drug``). A drug the labels always accent the same way
   (``paracétamol``) has no family, so it is not touched.
2. The form wordfreq knows best, when it is a real French word and clearly ahead
   (``thérèse`` over the majority ``therese``, ``françoise`` over ``francoise``).
3. Otherwise the most frequent form.

Every ``-génèse`` ends ``-genèse`` (``dentinogenèse``, ``mutagenèse``: the majority
varied per word).

A minority form is never rewritten when it is a word in its own right: a verb form
against its past participle (``débordé`` / ``déborde``), a word wordfreq knows
(``isolément`` / ``isolement``), or a family in ``KEEP`` (``môle`` / ``mole``, ``gènes`` /
``gênes``, Latin binomials ``pyogenes`` / ``Strongyloides`` / ``Anopheles``).

Run from the repo root on the audit TSV:

    uv run 06_hotfixes/07_build_accent_spellings.py path/to/variants.tsv

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("variant_audit", ROOT / "06_hotfixes" / "05_label_variant_audit.py")
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)
sys.path.insert(0, str(ROOT / "utils"))
from label_conventions import INN_STEM  # noqa: E402

# Drug substances (accent-free, lowercase), written without accents: the substances of
# the drug stage (``drugs_freq_dosages.jsonl`` rows of type ``substance``), the substances
# of the drug lexicon (canonical form lowercase; a capitalized one is a brand, whose
# accents stay: ``Flécaïne``), and any word with an INN stem (``-mab``, ``-tinib``,
# ``-pril``, ``-sartan``: the dictionary stage holds many drugs the drug stage does not).
def _drug_words() -> frozenset[str]:
    words = {"pemetrexed", "abeparvovec", "bulevirtide", "flumazenil", "imipenem", "metreleptine", "naropeine",
             "mexiletine", "mianserine", "mofetil", "mycophenolate", "droperidol", "pimecrolimus", "cefixime",
             "donepezil", "cetirizine", "nefopam"}
    for line in open(ROOT / "02_drugs" / "drugs_freq_dosages.jsonl", encoding="utf-8"):
        row = json.loads(line)
        names = [row["term"]] if row.get("type") == "substance" else []
        names += row.get("substances") or []
        words.update(_va.strip_accents(w).lower() for n in names for w in re.findall(r"[^\W\d_]{4,}", n))
    caps = json.loads((ROOT / "utils" / "drug_casing.json").read_text(encoding="utf-8"))["caps"]
    words.update(k.lower() for k, v in caps.items() if v.islower())
    return frozenset(words)


def is_drug(key: str) -> bool:
    """
    >>> is_drug("bevacizumab"), is_drug("perindopril"), is_drug("paracetamol"), is_drug("hexokinase")
    (True, True, True, False)
    """
    return key in DRUG_WORDS or bool(INN_STEM.search(key))


DRUG_WORDS = _drug_words()
OVERRIDES = {"defect": "défect", "artefact": "artefact", "artefacts": "artefacts", "olecrane": "olécrane",
             "olecranien": "olécranien", "olecranienne": "olécranienne", "retro-olecranienne": "rétro-olécranienne"}
# Two words, not two spellings: never rewritten.
KEEP = frozenset("""mole gene genes herpes pyogenes strongyloides anopheles area lutea cinereum reverse
    senior isolement interferent jeunes dependent coccidioides paracoccidioides""".split())
# Families whose majority, and wordfreq, are wrong (a typo or a lost umlaut won the count).
FIXED = {"hexokinase": "hexokinase", "chondroitinase": "chondroïtinase", "subarachnoidien": "subarachnoïdien",
         "helix": "hélix", "seton": "séton", "eicosatetraenoique": "eicosatétraénoïque", "sterno-cleido-mastoidiens": "sterno-cléido-mastoïdiens", "frankenhauser": "frankenhäuser", "immunonephelemetrie": "immunonéphélémétrie",
         "gerstmann-straussler-scheinker": "gerstmann-sträussler-scheinker", "giraldes": "giraldès",
         # Added with --min-cost 3 (2026-10-08): eponyms and brands as their owners write them,
         # French words whose majority lost or gained an accent, and INNs (unaccented).
         "bitot": "bitot", "nuck": "nuck", "kossa": "kossa", "quantiferon": "quantiferon", "ethilon": "ethilon",
         "hemoloc": "hemoloc", "hemolock": "hemolock", "hemolok": "hemolok", "bevitine": "bévitine",
         "flecaine": "flécaïne", "nephelemetrie": "néphélémétrie", "colpocephalie": "colpocéphalie",
         "myeloproliferative": "myéloproliférative", "extraconique": "extraconique",
         "secretome": "sécrétome", "fadenoperation": "fadenopération",
         "cefotaxime": "cefotaxime", "piperacilline-tazobactam": "piperacilline-tazobactam",
         "eflornithine": "eflornithine", "lynestrenol": "lynestrenol", "vericiguat": "vericiguat",
         "irinotecan": "irinotecan", "sebelipase": "sebelipase"}


def canonical(forms: list[tuple[str, int]]) -> str | None:
    """The form a family is written in, or None to leave the family alone.

    >>> canonical([("périndopril", 579), ("perindopril", 150)]), canonical([("therese", 34), ("thérèse", 9)])
    ('perindopril', 'thérèse')
    >>> canonical([("défect", 414), ("defect", 317)]), canonical([("môle", 276), ("mole", 127)])
    ('défect', None)
    """
    from wordfreq import zipf_frequency

    key = _va.strip_accents(forms[0][0]).split("'")[-1]
    elision = forms[0][0][: len(forms[0][0]) - len(key)] if "'" in forms[0][0] else ""
    if key in KEEP:
        return None
    if key in OVERRIDES:
        return elision + OVERRIDES[key]
    if key in FIXED:
        return elision + FIXED[key]
    if is_drug(key):
        return elision + key
    zipf = sorted(((zipf_frequency(f, "fr"), f) for f, _ in forms), reverse=True)
    canon = zipf[0][1] if zipf[0][0] >= 1.5 and zipf[0][0] - zipf[1][0] >= 0.5 else forms[0][0]
    # One ending for every "-genèse" (ostéogenèse, dentinogenèse), the dictionary one.
    return re.sub(r"génèse$", "genèse", canon)


def rewrites(forms: list[tuple[str, int]]) -> dict[str, str]:
    """``minority form -> canonical`` for one family, sparing forms that are words of their own.

    >>> rewrites([("déborde", 10), ("deborde", 9), ("débordé", 6)])
    {'deborde': 'déborde'}
    """
    from wordfreq import zipf_frequency

    canon = canonical(forms)
    if canon is None:
        return {}
    out = {}
    for f, _ in forms:
        if f == canon or _va.is_participle_pair(f, canon) or zipf_frequency(f, "fr") >= 2.5:
            continue
        out[f] = canon
    return out


@click.command()
@click.argument("variants_tsv", type=click.Path(exists=True))
@click.option("--min-cost", default=3, show_default=True, help="Skip families whose minority forms total fewer occurrences.")
@click.option("--out", type=click.Path(), default=str(ROOT / "utils" / "accent_spellings.json"), show_default=True)
def main(variants_tsv, min_cost, out):
    table = {}
    for line in list(open(variants_tsv, encoding="utf-8"))[1:]:
        kind, cost, homo, shown, _ = line.rstrip("\n").split("\t")
        if kind != "accent" or homo == "1" or int(cost) < min_cost:
            continue
        forms = [(f.rsplit(" (", 1)[0], int(f.rsplit(" (", 1)[1][:-1])) for f in shown.split(" | ")]
        table.update(rewrites(forms))
    doc = "Built by 06_hotfixes/07_build_accent_spellings.py: lowercase minority spelling -> canonical form."
    Path(out).write_text(json.dumps({"_doc": doc, "rewrites": dict(sorted(table.items()))}, ensure_ascii=False, indent=0) + "\n",
                         encoding="utf-8")
    click.echo(f"{len(table)} rewrites -> {out}")


if __name__ == "__main__":
    main()
