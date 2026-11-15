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

``HAND_REWRITES`` (2026-11-15) then adds what the audit cannot see: typos, source-sheet
misspellings, two right spellings settled by the owner, and accented INNs below
``--min-cost``. The TSV is not committed, so after editing those dicts without it, merge
them into the committed JSON by hand (``tests/test_label_variant_audit.py`` checks they match).

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
             "olecranien": "olécranien", "olecranienne": "olécranienne", "retro-olecranienne": "rétro-olécranienne",
             # an enzyme, not a drug (the drug lexicon holds "inhibiteur de protéase")
             "protease": "protéase"}
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

# Hand-checked rewrites the audit cannot find, merged over the computed table (2026-11-15).
# Each key was looked up in the labels and the canonical form checked against the ANSM
# drug database, the Vidal or the INN list; each value is final (never a key itself).
# Unlike the families above they are not all accent variants, so ``label_conventions``
# keeps them out of the TTS respellings (``_tts_accent_table`` requires the same letters).
# Misspellings inherited from a source sheet (``02_drugs`` CNAM L_ATC5 labels, one LLM typo).
SOURCE_SHEET_FIXES = {"nalbufine": "nalbuphine", "polyomielite": "poliomyélite",
                      "diphtérie-polyomielite-tétanos": "diphtérie-poliomyélite-tétanos",
                      "glycopirronium": "glycopyrronium"}
# Typos in the labels, the owner's list then a systematic search; brands keep their capital.
TYPOS = {"perindoprill": "perindopril", "azythromycine": "azithromycine",
         "gentamycine": "gentamicine", "cloxaciline": "cloxacilline", "entreesto": "Entresto",
         "luspathercept": "luspatercept", "affinitor": "Afinitor", "ellebrato": "Elebrato",
         "clotafact": "Clottafact", "hydrochlorotiazide": "hydrochlorothiazide",
         "hydrochlorothéiazide": "hydrochlorothiazide", "tazociline": "Tazocilline",
         "aktiskenan": "Actiskenan", "ackiskenan": "Actiskenan", "actikskenan": "Actiskenan",
         "actiksenan": "Actiskenan", "actikenan": "Actiskenan",
         "doxycyline": "doxycycline", "cefriaxone": "ceftriaxone",
         "amplodipine": "amlodipine", "pantopazole": "pantoprazole",
         "desmoprésine": "desmopressine", "candersartan": "candesartan", "aprazolam": "alprazolam",
         "irbersartan": "irbésartan", "adalmumab": "adalimumab", "zolendronique": "zolédronique",
         "formeterol": "formotérol", "formétérol": "formotérol", "imbrutinib": "ibrutinib",
         "glicazide": "gliclazide", "lanceotide": "lanréotide", "selpercatnib": "selpercatinib",
         "tezepilumab": "tezepelumab", "rilmenildine": "rilmenidine", "teimocilline": "temocilline",
         "amitryptiline": "amitriptyline", "amitryptyline": "amitriptyline", "oxybutinine": "oxybutynine",
         "néphopam": "nefopam", "bénzatine": "benzathine", "béclométhasone": "beclometasone",
         "métrondiazole": "métronidazole", "lévifloxacine": "levofloxacine", "acétrétine": "acitrétine",
         "bétaméthazone": "betamethasone", "eczétimibe": "ézétimibe", "clomiphène": "clomifène",
         "arthéméter": "artéméther", "fénothérol": "fénotérol", "tocoferol": "tocophérol",
         "isétionate": "iséthionate", "syncitial": "syncytial",
         "natyspray": "Natispray", "flamazine": "Flammazine", "immovane": "Imovane",
         "zythromax": "Zithromax", "théralite": "Téralithe", "theralithe": "Téralithe",
         "bacrim": "Bactrim", "ferrinject": "Ferinject", "ferinjec": "Ferinject",
         "vycril": "Vicryl", "bêtadine": "bétadine", "betadine": "bétadine",
         # English INN forms in a French label
         "empagliflozin": "empagliflozine", "midostaurin": "midostaurine", "darbepoetin": "darbepoetine"}
# Two right spellings, one taught (owner's choice). ``cefalexine`` follows the INN
# convention, no accent (2026-11-15: it used to keep it as an exception).
SPELLING_WINNERS = {"acyclovir": "aciclovir", "ciclosporine": "cyclosporine",
                    "ciclosporinémie": "cyclosporinémie", "tioguanine": "thioguanine",
                    "céphalexine": "cefalexine", "cephalexine": "cefalexine", "céfalexine": "cefalexine",
                    "alphacalcidol": "alfacalcidol", "alfacalcidiol": "alfacalcidol"}
# INNs still accented in the labels, below ``--min-cost`` or missed by ``is_drug``, where
# the unaccented form already had the majority (``ertapénème`` is the French INN, kept to
# one spelling with the others). Compounds are keys of their own: the lookup is per word.
INN_LEFTOVERS = {"ertapénem": "ertapenem", "ertapénème": "ertapenem", "lanadélumab": "lanadelumab",
                 "olodatérol": "olodaterol", "dexaméthasone-néomycine": "dexamethasone-néomycine",
                 "céftazidime": "ceftazidime", "émicizumab": "emicizumab", "énfortumab": "enfortumab",
                 "eptinézumab": "eptinezumab", "eskétamine": "esketamine", "fédratinib": "fedratinib",
                 "fénfluramine": "fenfluramine", "ganirélix": "ganirelix", "nirsévimab": "nirsevimab",
                 "ocrélizumab": "ocrelizumab", "pégfilgrastim": "pegfilgrastim", "pégol": "pegol",
                 "prédnisolone": "prednisolone", "sonidégib": "sonidegib", "télmisartan": "telmisartan",
                 "volanésorsen": "volanesorsen", "grazoprévir": "grazoprevir",
                 "atézolizumab-bévacizumab": "atezolizumab-bevacizumab", "dégarélix": "degarelix",
                 "dégarelix": "degarelix", "setmélanotide": "setmelanotide", "sétmélanotide": "setmelanotide",
                 "alitrétinoïne": "alitretinoine", "alitretinoïne": "alitretinoine",
                 "bempédoïque": "bempedoique", "bempédoique": "bempedoique", "époétine": "epoetine",
                 "epoétine": "epoetine", "vélaglucérase": "velaglucerase", "velaglucérase": "velaglucerase"}
# Same convention where the ACCENTED form still had the majority: a larger rewrite (``lévétiracétam``
# 1068 labels), kept apart so it can be dropped on its own. The TTS text keeps the accents.
INN_LEFTOVERS_MAJORITY = {
    "alafénamide": "alafenamide", "béclométasone": "beclometasone", "bélatacept": "belatacept",
    "bétahistine": "betahistine", "bétaméthasone": "betamethasone", "betaméthasone": "betamethasone",
    "budésonide": "budesonide", "céfoxitine": "cefoxitine", "céfuroxime": "cefuroxime",
    "duloxétine": "duloxetine", "érythromycine": "erythromycine", "étanercept": "etanercept",
    "éthosuximide": "ethosuximide", "fluoxétine": "fluoxetine", "indacatérol": "indacaterol",
    "irinotécan": "irinotecan", "lévétiracétam": "levetiracetam", "lévothyroxine": "levothyroxine",
    "lopéramide": "loperamide", "mépolizumab": "mepolizumab", "méropénem": "meropenem",
    "méthylprednisolone": "methylprednisolone", "méthylprédnisolone": "methylprednisolone",
    "méticilline": "meticilline", "nébivolol": "nebivolol", "nomégestrol": "nomegestrol",
    "pinavérium": "pinaverium", "pipéracilline": "piperacilline", "saproptérine": "sapropterine",
    "ténofovir": "tenofovir", "topotécan": "topotecan", "tréprostinil": "treprostinil",
    "varénicline": "varenicline", "vérapamil": "verapamil",
    "méticilline-sensible": "meticilline-sensible", "méticilline-résistant": "meticilline-résistant",
    "budésonide-formotérol": "budesonide-formotérol", "céfuroxime-axétil": "cefuroxime-axétil",
    "ténofovir-emtricitabine": "tenofovir-emtricitabine"}
# Typos whose fix changes the SOUND and that each sit in exactly one existing clip (all
# PARHAF), whose audio says the typo: fixing the label alone would make it disagree with
# its audio. Held out of HAND_REWRITES (and the JSON) until those clips are re-synthesized
# from the fixed text; docs/BUILD_LOG.md ("Label spelling fixes after the v2 model") has
# the clip ids and the steps. ``actisénan`` is how that clip actually spells it.
RESYNTH_TYPOS = {"perindipril": "perindopril", "cetriaxone": "ceftriaxone",
                 "bisorolol": "bisoprolol", "actisenan": "Actiskenan", "actisénan": "Actiskenan",
                 "karegic": "Kardégic"}

# Owner's hand review of the leftover variants (2026-11-15). Brands take the drug caser's
# capitals (``utils/drug_casing.json``), else a second pass would recase them. ``tymoferol``
# maps to itself so an all-caps ``TYMOFEROL`` gets the table casing. ``mesalamine`` GAINS
# its accent (French spelling). Reviewed and kept as they are: gonadotropine, homatropine.
OWNER_REVIEW = {"patidexaméthasone": "patidexamethasone", "tymoferol": "tymoferol",
                "laxilix": "Lasilix", "cestradiol": "estradiol", "thiorphan": "Tiorfan",
                "cromoglycique": "cromoglicique", "oxyconazole": "oxiconazole",
                "mesalamine": "mésalamine"}
HAND_REWRITES = (SOURCE_SHEET_FIXES | TYPOS | SPELLING_WINNERS | INN_LEFTOVERS
                 | INN_LEFTOVERS_MAJORITY | OWNER_REVIEW)


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
        # keyed after normalize_spelling, which writes ``æ`` as ``ae`` first
        forms = [(f.rsplit(" (", 1)[0].replace("æ", "ae"), int(f.rsplit(" (", 1)[1][:-1])) for f in shown.split(" | ")]
        table.update(rewrites(forms))
    table.update(HAND_REWRITES)
    doc = "Built by 06_hotfixes/07_build_accent_spellings.py: lowercase minority spelling -> canonical form."
    Path(out).write_text(json.dumps({"_doc": doc, "rewrites": dict(sorted(table.items()))}, ensure_ascii=False, indent=0) + "\n",
                         encoding="utf-8")
    click.echo(f"{len(table)} rewrites -> {out}")


if __name__ == "__main__":
    main()
