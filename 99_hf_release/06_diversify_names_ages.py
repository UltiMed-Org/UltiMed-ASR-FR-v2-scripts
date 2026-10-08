#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "wordfreq"]
# ///
"""Spread the patient surnames and ages the text LLM kept reusing, before the audio is regenerated.

The v1 generator wrote the same few patients over and over: 173.6k title + surname
mentions, 91% of them seven names (``Girard`` 41.0k, ``Lefèvre`` 29.6k, ``Bernard``
21.4k, ``Lambert`` 17.7k, ``Petit`` 17.7k, ``Dubois`` 17.4k, ``Moreau`` 12.1k), and ages
piled on a few values (``45 ans`` 22.9k against 125 for ``46 ans``, ``34 ans`` 20.3k,
``58 ans`` 19.2k). A model trained on that learns that ``Madame`` is followed by
``Girard`` and that a patient is 45. This rewrites both texts of a row (``text`` and
``asr_training_source``) the same way:

- a surname from ``OVERUSED`` after a title (``Madame``, ``docteur``, ``Patient``), a
  first name (``Léa Bernard``) or ``petit`` / ``petite`` / ``jeune`` becomes a surname drawn
  from ``surname_pool.tsv`` (the 1,000 most frequent INSEE surnames of people born
  1931-2000, frequency-weighted, accents taken from the corpus or the INSEE first-name
  list, French words and the overused names left out). ``Claude Bernard`` (the
  physiologist, ``Claude Bernard-Horner``) stays.
- an adult age in ``OVERUSED_AGES`` after a person (``patiente de 45 ans``, ``Girard, 45
  ans``, ``âgé de 45 ans``) keeps its value with probability ``AGE_KEEP / count``, so
  about 3.4k labels (the flat share of an adult age) keep it, and otherwise moves by up
  to ``AGE_SPREAD`` years to an age outside the set. Durations (``depuis 45 ans``) are not
  after a person and stay.

Only the categories made of independent sentences (``dictionary``, ``drugs``,
``acronyms``): a PARHAF / PARROT row is a chunk of a document whose other chunks would
keep the old name. Each draw is seeded by the clip file name plus the original name or
age (``hashlib``, stable across runs and machines), so the ``full`` / split / per-subset
copies of a row get the same rewrite. That also makes the run idempotent: a new surname
is never an overused one, a moved age is never an overused one, and a kept age is drawn
again with the same seed.

    uv run 06_diversify_names_ages.py                 # dry run, reports counts + examples
    uv run 06_diversify_names_ages.py --apply
    uv run 06_diversify_names_ages.py --build-pool noms2008nat_txt.txt --first-names nat2021.csv

Run before ``05_normalize_text.py``'s final check and ``scripts/build_parquet.py``. The
pool is rebuilt only with ``--build-pool`` (the INSEE files come from
``parakeet-tdt-0.6b-v3-ultra-onnx/scripts/drug-rules/insee-names.py``).

This file was written by Claude Code.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import MANIFEST_GLOBS, read_jsonl, write_jsonl  # noqa: E402

POOL = _HERE / "surname_pool.tsv"
CATEGORIES = ("dictionary", "drugs", "acronyms")
OVERUSED = ("Girard", "Lefèvre", "Bernard", "Lambert", "Petit", "Dubois", "Moreau")
# Adult ages said 6k times or more in the v2 corpus (a flat adult age is about 3.4k).
OVERUSED_AGES = {22: 6053, 28: 10287, 29: 7328, 34: 20299, 38: 6372, 42: 7773, 45: 22880,
                 52: 12660, 55: 11616, 58: 19203, 62: 6284, 68: 10807, 72: 12772}
AGE_KEEP = 3440
AGE_SPREAD = 7
POOL_SIZE = 1000

_TITLES = ("Monsieur", "Madame", "Mademoiselle", "docteur", "Docteur", "professeur", "Professeur",
           "Patient", "patient", "Patiente", "patiente", "petit", "petite", "jeune")
# not first names: a capitalized word before the surname must be a title or a first name
_NOT_FIRST = {"Le", "La", "Les", "De", "Du", "Des", "Et", "En", "Au", "Aux", "Un", "Une", "Claude"}
_NAME_RE = re.compile(rf"\b([A-ZÉ][\wéèêëïîôç]+|{'|'.join(_TITLES)}) ({'|'.join(OVERUSED)})\b(?!-)")
_PERSON = (r"(?:[Pp]atiente?|[Hh]omme|[Ff]emme|âgée?|[Ss]ujet|adulte|primipare|primigeste|multipare"
           r"|sportive?|fils|fille|soeur|frère|mère|père|épouse|mari|retraitée?|diabétique"
           r"|hypertendue?|fumeur|fumeuse|obèse|asthmatique|agricultrice|agriculteur|étudiante?"
           r"|enseignante?|salariée?|ouvrière?|ouvrier|infirmière?|infirmier|coureur|coureuse)")
_AGE_RE = re.compile(rf"(\b{_PERSON} de |\b[A-ZÉ][\wéèêëïîôç]+, )(\d{{2}}) ans\b")


def _draw(seed: str) -> float:
    """A number in [0, 1) that only depends on ``seed``."""
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big") / 2 ** 64


def load_pool(path: Path = POOL) -> tuple[list[str], list[float]]:
    """``(names, cumulative weights)`` of ``surname_pool.tsv``."""
    names, cum, total = [], [], 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        name, weight = line.split("\t")
        total += float(weight)
        names.append(name)
        cum.append(total)
    return names, [c / total for c in cum]


def new_surname(seed: str, pool: tuple[list[str], list[float]]) -> str:
    names, cum = pool
    x = _draw(seed)
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return names[lo]


def new_age(seed: str, age: int) -> int:
    """``age`` itself with probability ``AGE_KEEP / count``, else a draw among the ages
    within ``AGE_SPREAD`` that are not overused."""
    if age not in OVERUSED_AGES or _draw(seed + "|keep") < AGE_KEEP / OVERUSED_AGES[age]:
        return age
    near = [a for a in range(age - AGE_SPREAD, age + AGE_SPREAD + 1) if a not in OVERUSED_AGES]
    return near[int(_draw(seed + "|move") * len(near))]


def rewrite(text: str, clip: str, pool: tuple[list[str], list[float]]) -> str:
    """``Madame Girard, 45 ans`` -> ``Madame Rousseau, 47 ans``, same output for the same
    clip, idempotent."""
    def name(m: re.Match) -> str:
        if m.group(1)[0].isupper() and m.group(1) not in _TITLES and m.group(1) in _NOT_FIRST:
            return m.group(0)
        seed = f"{clip}|{m.group(2)}"
        surname = new_surname(seed, pool)
        while surname == m.group(1):  # ``Lucas Lucas``
            seed += "+"
            surname = new_surname(seed, pool)
        return f"{m.group(1)} {surname}"

    text = _NAME_RE.sub(name, text)
    return _AGE_RE.sub(lambda m: f"{m.group(1)}{new_age(f'{clip}|{m.group(2)}', int(m.group(2)))} ans", text)


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _e_variants(name: str) -> list[str]:
    """``Guerin`` -> ``Guerin``, ``Guérin``, ``Guèrin``, ``Guêrin``: every way to accent its
    ``e`` (at most four of them)."""
    out = [""]
    for ch in name:
        out = [o + c for o in out for c in ("eéèê" if ch == "e" and name.count("e") <= 4 else ch)]
    return out


def build_pool(insee: Path, first_names: Path | None, manifests: list[Path]) -> list[tuple[str, int]]:
    """The ``POOL_SIZE`` most frequent INSEE surnames (births 1931-2000), title-cased.

    INSEE writes them without accents. The spelling kept is, in order: the accented one
    the corpus wrote after a title (``Lefèvre``), the accented ``e`` that wordfreq knows
    better than the bare one (``Guérin``, ``Rivière``), the accented first name when it is
    the usual one (``André``, not ``Mathïs``), else the bare name. A name the corpus also
    writes in lowercase 20 times or more is a French word (``Blanc``, ``Fontaine``) and
    is left out, as are the overused names.
    """
    from wordfreq import zipf_frequency

    counts = Counter()
    for line in insee.read_text(encoding="latin-1").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) == 12 and re.fullmatch(r"[A-Z]{3,}", cols[0]):
            counts[cols[0]] = sum(int(c) for c in cols[5:12])
    after_title: dict[str, Counter] = {}
    lower = Counter()
    title_re = re.compile(r"\b(?:Monsieur|Madame|docteur) ([A-ZÉ][a-zéèêëïîôç]+)\b")
    for path in manifests:
        for row in read_jsonl(path):
            for n in title_re.findall(row["text"]):
                after_title.setdefault(_fold(n), Counter())[n] += 1
            lower.update(re.findall(r"(?<=\w )[a-zéèêëïîôç]+\b", row["text"]))
    first: dict[str, Counter] = {}
    if first_names:
        with first_names.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=";"):
                n = r["preusuel"].capitalize()
                first.setdefault(_fold(n), Counter())[n] += int(r["nombre"])
    lower_folded = Counter()
    for w, c in lower.items():
        lower_folded[_fold(w)] += c
    overused = {_fold(n) for n in OVERUSED}
    pool = []
    for upper, count in counts.most_common():
        key = upper.lower()
        if key in overused or lower_folded[key] >= 20:
            continue
        bare = upper.capitalize()
        corpus = [n for n, _ in after_title.get(key, Counter()).most_common() if n != bare]
        best_e = max(_e_variants(bare), key=lambda v: (zipf_frequency(v, "fr"), v == bare))
        usual_first = first.get(key, Counter()).most_common(1)
        if corpus:
            spelling = corpus[0]
        elif best_e != bare and zipf_frequency(best_e, "fr") > zipf_frequency(bare, "fr"):
            spelling = best_e
        elif usual_first:
            spelling = usual_first[0][0]
        else:
            spelling = bare
        pool.append((spelling, count))
        if len(pool) == POOL_SIZE:
            break
    return pool


def _targets(manifests: tuple[Path, ...], root_path: Path) -> list[Path]:
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    targets = [p for p in targets if p.is_file() and not p.name.startswith(".")]
    if not targets:
        raise SystemExit(f"no manifest under {root_path}")
    return targets


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it, a dry run.")
@click.option("--examples", default=10, show_default=True, help="Changed labels to print.")
@click.option("--build-pool", "insee", type=click.Path(exists=True, path_type=Path),
              help="INSEE noms2008nat_txt.txt: rebuild surname_pool.tsv from it and exit.")
@click.option("--first-names", type=click.Path(exists=True, path_type=Path),
              help="INSEE nat2021.csv, for the accents of surnames that are also first names.")
def main(manifests: tuple[Path, ...], root: str, apply: bool, examples: int, insee: Path | None,
         first_names: Path | None) -> None:
    """Spread the overused patient surnames and ages of the independent-sentence rows."""
    targets = _targets(manifests, Path(root).resolve())
    if insee:
        main_manifests = [p for p in targets if p.name == "full.jsonl" and p.parent.name == "NeMO_files"]
        pool = build_pool(insee, first_names, main_manifests or targets)
        POOL.write_text("".join(f"{n}\t{c}\n" for n, c in pool), encoding="utf-8")
        logger.info(f"{len(pool)} surnames -> {POOL}, e.g. {[n for n, _ in pool[:12]]}")
        return
    pool = load_pool()
    shown = 0
    for path in targets:
        rows = read_jsonl(path)
        changed, names_after, ages_after = 0, Counter(), Counter()
        new_rows = []
        for r in rows:
            if r.get("category") not in CATEGORIES:
                new_rows.append(r)
                continue
            clip = Path(r["audio_filepath"]).name
            new = dict(r, text=rewrite(r["text"], clip, pool))
            if r.get("asr_training_source") is not None:
                new["asr_training_source"] = rewrite(r["asr_training_source"], clip, pool)
            if new != r:
                changed += 1
                if shown < examples:
                    logger.info(f" - {r['text'][:160]}\n + {new['text'][:160]}")
                    shown += 1
            names_after.update(m.group(1) for m in re.finditer(rf"\b(?:{'|'.join(_TITLES)}) ([A-ZÉ]\w+)", new["text"]))
            ages_after.update(int(m.group(2)) for m in _AGE_RE.finditer(new["text"]))
            new_rows.append(new)
        if not changed:
            continue
        logger.info(f"{path}: {len(rows)} rows, {changed} rewritten; top surnames after: "
                    f"{names_after.most_common(5)}; top ages after: {ages_after.most_common(5)}")
        if apply:
            write_jsonl(path, new_rows)
    if not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    else:
        logger.info("now rerun 05_normalize_text.py (expect 0 changes) and rebuild parquet")


if __name__ == "__main__":
    main()
