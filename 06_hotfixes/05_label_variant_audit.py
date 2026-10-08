#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "wordfreq"]
# ///
"""List the words the labels spell several ways, to fix before a new TTS round.

The LLM generators wrote one word several ways across the corpus (``pré-remplie`` /
``préremplie``, ``petit déjeuner`` / ``petit-déjeuner``, ``aigüe`` / ``aiguë``), so the
model is taught two answers for one sound and loses WER on whichever it did not pick.
``utils/label_conventions.py`` already fixes the families found so far; this lists every
family still left, ranked by how much each costs, so new conventions can be chosen
before the dataset is re-synthesized.

Two kinds of family, each a group of written forms with the same key:

- ``hyphen``: a word written glued, hyphenated or as separate words. Key: the lowercased
  form without hyphens and spaces, accents KEPT (so ``de croissance`` never joins
  ``décroissance``). At least one form must be a single token, which keeps out runs like
  ``a sa`` / ``as a``. Forms are words and runs of 2 or 3 words.

      {"post-opératoire": 12062, "postopératoire": 7764, "post opératoire": 174}

  A spaced form can also be two real words (``mais on`` / ``maison``, ``en fin`` /
  ``enfin``): the ``homograph`` mark below applies here too.

- ``accent``: one word written with different accents. Key: the lowercased word without
  accents, single words only. Most of these are two real words (``confirme`` /
  ``confirmé``, ``iode`` / ``iodé``): when the two most frequent forms are both common
  French words, or a verb form and its past participle, the family is marked
  ``homograph`` and listed after the others.

      {"aiguë": 900, "aigüe": 40}       {"défect": 414, "defect": 317}

A family's cost is the count of every form but the most frequent one (7938 for the
``post-opératoire`` family): those clips teach a minority spelling.

Input: NeMo manifests (``text`` is the label). Output: a TSV, one family per row,
``kind  cost  homograph  forms (count)  example label of the top minority form``.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict

import click

# An elision stays on its word (``s'est``, ``l'étoposide``), else ``elle s'est`` would yield
# the run ``elle s`` and join the ``elles`` family.
WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")


def strip_accents(s: str) -> str:
    """
    >>> strip_accents("aiguë préremplie")
    'aigue preremplie'
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def hyphen_key(form: str) -> str | None:
    """Key of a ``hyphen`` family: drop hyphens and spaces, keep accents.

    >>> hyphen_key("post-opératoire"), hyphen_key("post opératoire"), hyphen_key("de croissance")
    ('postopératoire', 'postopératoire', 'decroissance')
    """
    return form.replace("-", "").replace(" ", "")


def accent_key(form: str) -> str | None:
    """Key of an ``accent`` family: single words only, accents removed.

    >>> accent_key("aigüe"), accent_key("petit déjeuner")
    ('aigue', None)
    """
    return None if " " in form else strip_accents(form)


def count_forms(labels, max_n: int = 3) -> tuple[Counter, dict]:
    """Count every lowercased word and run of 2..max_n words, keeping one example label each.

    A run never holds a hyphenated word, so ``petit-déjeuner`` is one form, never
    part of ``seringue petit-déjeuner``.

    >>> c, _ = count_forms(["Une seringue pré-remplie.", "Seringue préremplie, petit déjeuner", "Elle s'est levée."])
    >>> c["pré-remplie"], c["préremplie"], c["petit déjeuner"], c["seringue pré-remplie"], c["elle s"]
    (1, 1, 1, 0, 0)
    """
    counts: Counter = Counter()
    example: dict = {}
    for label in labels:
        words = [w.lower() for w in WORD.findall(label)]
        for i in range(len(words)):
            for n in range(1, max_n + 1):
                run = words[i:i + n]
                if len(run) < n or (n > 1 and any("-" in x for x in run)):
                    break
                form = " ".join(run)
                counts[form] += 1
                example.setdefault(form, label)
    return counts, example


def families(counts: Counter, key, min_cost: int = 1, need_single_token: bool = False):
    """Group forms by ``key`` and keep groups whose minority forms total >= min_cost.

    >>> c = Counter({"pré-remplie": 5, "préremplie": 2, "seringue": 9, "petit déjeuner": 1,
    ...              "petit-déjeuner": 4, "a sa": 3, "as a": 2})
    >>> families(c, hyphen_key, need_single_token=True)
    [(2, [('pré-remplie', 5), ('préremplie', 2)]), (1, [('petit-déjeuner', 4), ('petit déjeuner', 1)])]
    """
    groups = defaultdict(list)
    for form, n in counts.items():
        k = key(form)
        if k is not None:
            groups[k].append((form, n))
    out = []
    for forms in groups.values():
        if len(forms) < 2 or (need_single_token and all(" " in f for f, _ in forms)):
            continue
        forms.sort(key=lambda x: (-x[1], x[0]))
        cost = sum(n for _, n in forms[1:])
        if cost >= min_cost:
            out.append((cost, forms))
    out.sort(key=lambda x: (-x[0], x[1][0][0]))
    return out


def is_homograph(forms: list[tuple[str, int]], zipf: float = 3.0) -> bool:
    """True when the two most frequent forms are both common French words or phrases (two words, not two spellings).

    >>> is_homograph([("confirme", 9), ("confirmé", 3), ("confi rmé", 1)]), is_homograph([("aiguë", 5), ("aigüe", 2)])
    (True, False)
    >>> is_homograph([("mais on", 9), ("maison", 3)]), is_homograph([("post-opératoire", 9), ("postopératoire", 7)])
    (True, False)
    """
    from wordfreq import zipf_frequency

    a, b = forms[0][0], forms[1][0]
    return is_participle_pair(a, b) or all(zipf_frequency(f, "fr") >= zipf for f in (a, b))


PARTICIPLE_END = re.compile(r"é(?:e?s?)$")


def is_participle_pair(a: str, b: str) -> bool:
    """True for a verb form against its past participle (``visualise`` / ``visualisé``,
    ``altère`` / ``altéré``): they differ only in e / é / è and exactly one ends in a
    participle ending. Too rare for wordfreq to call both common, yet two real words.

    >>> is_participle_pair("visualise", "visualisé"), is_participle_pair("altéré", "altère")
    (True, True)
    >>> is_participle_pair("ménière", "menière"), is_participle_pair("cône", "cone")
    (False, False)
    """
    if len(a) != len(b) or any(x != y and not {x, y} <= set("eéè") for x, y in zip(a, b)):
        return False
    return bool(PARTICIPLE_END.search(a)) != bool(PARTICIPLE_END.search(b))


@click.command()
@click.argument("manifests", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--out", type=click.Path(), required=True, help="TSV to write.")
@click.option("--min-cost", default=3, show_default=True, help="Drop families whose minority forms total fewer clips.")
def main(manifests, out, min_cost):
    labels = (json.loads(line)["text"] for mf in manifests for line in open(mf, encoding="utf-8"))
    counts, example = count_forms(labels)
    rows = [("hyphen", cost, forms, is_homograph(forms)) for cost, forms in families(counts, hyphen_key, min_cost, True)]
    rows += [("accent", cost, forms, is_homograph(forms)) for cost, forms in families(counts, accent_key, min_cost)]
    rows.sort(key=lambda r: (r[3], -r[1]))
    with open(out, "w", encoding="utf-8") as f:
        f.write("kind\tcost\thomograph\tforms\texample_of_minority\n")
        for kind, cost, forms, homo in rows:
            shown = " | ".join(f"{form} ({n})" for form, n in forms)
            f.write(f"{kind}\t{cost}\t{int(homo)}\t{shown}\t{example[forms[1][0]]}\n")
    for kind in ("hyphen", "accent"):
        sel = [r for r in rows if r[0] == kind and not r[3]]
        click.echo(f"{kind}: {len(sel)} families, {sum(r[1] for r in sel)} minority occurrences")
    click.echo(f"-> {out}")


if __name__ == "__main__":
    main()
