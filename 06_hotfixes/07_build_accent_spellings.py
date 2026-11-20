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
``--min-cost``. The TSV is not committed, so after editing those dicts without it, run
the script with no TSV: it merges ``HAND_REWRITES`` over the committed JSON and keeps the
rest of it (``tests/test_label_variant_audit.py`` checks they match).

Run from the repo root on the audit TSV, or with none to sync ``HAND_REWRITES`` only:

    uv run 06_hotfixes/07_build_accent_spellings.py path/to/variants.tsv
    uv run 06_hotfixes/07_build_accent_spellings.py

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
             "protease": "protéase",
             # Owner's choices on colliding spellings (2026-11-16, docs/BUILD_LOG.md); the
             # separator is the compound table's (08_build_compound_spellings.py), which runs first.
             "echoendoscopie": "écho-endoscopie", "cranio-encephalique": "cranio-encéphalique",
             "cranio-encephaliques": "cranio-encéphaliques", "antepartum": "antépartum",
             "ante-partum": "antépartum"}
# Two words, not two spellings: never rewritten. Also the families a
# ``label_conventions._SPELLING_RULES`` rule owns, which runs before the table: the clip
# ``Hem-o-lok`` (2026-11-18, every spelling and its plural).
KEEP = frozenset("""mole gene genes herpes pyogenes strongyloides anopheles area lutea cinereum reverse
    senior isolement interferent jeunes dependent coccidioides paracoccidioides
    hemoloc hemolock hemolok hemolocs hemolocks hemoloks hem-o-lock hem-o-lok""".split())
# Families whose majority, and wordfreq, are wrong (a typo or a lost umlaut won the count).
FIXED = {"hexokinase": "hexokinase", "chondroitinase": "chondroïtinase", "subarachnoidien": "subarachnoïdien",
         "helix": "hélix", "seton": "séton", "eicosatetraenoique": "eicosatétraénoïque", "sterno-cleido-mastoidiens": "sterno-cléido-mastoïdiens", "frankenhauser": "frankenhäuser", "immunonephelemetrie": "immunonéphélémétrie",
         "gerstmann-straussler-scheinker": "gerstmann-sträussler-scheinker", "giraldes": "giraldès",
         # Added with --min-cost 3 (2026-10-08): eponyms and brands as their owners write them,
         # French words whose majority lost or gained an accent, and INNs (unaccented).
         "bitot": "bitot", "nuck": "nuck", "kossa": "kossa", "quantiferon": "quantiferon", "ethilon": "ethilon",
         "bevitine": "bévitine",
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
                      "glycopirronium": "glycopyrronium"}
# Typos in the labels, the owner's list then a systematic search; brands keep their capital.
# Only fixes that keep the sound (the sound-changing ones are in RESYNTHESIZED_TYPOS below).
TYPOS = {"azythromycine": "azithromycine", "gentamycine": "gentamicine", "cloxaciline": "cloxacilline",
         "luspathercept": "luspatercept", "affinitor": "Afinitor", "ellebrato": "Elebrato",
         "clotafact": "Clottafact", "hydrochlorotiazide": "hydrochlorothiazide",
         "tazociline": "Tazocilline", "aktiskenan": "Actiskenan",
         "amitryptiline": "amitriptyline", "amitryptyline": "amitriptyline", "oxybutinine": "oxybutynine",
         "néphopam": "nefopam", "béclométhasone": "beclometasone",
         "bétaméthazone": "betamethasone", "clomiphène": "clomifène",
         "arthéméter": "artéméther", "fénothérol": "fénotérol",
         "isétionate": "iséthionate", "syncitial": "syncytial",
         "natyspray": "Natispray", "flamazine": "Flammazine", "immovane": "Imovane",
         "zythromax": "Zithromax", "théralite": "Teralithe", "ferrinject": "Ferinject",
         "vycril": "Vicryl", "bêtadine": "bétadine", "betadine": "bétadine",
         # Sound-changing, but no clip in the release carries it, so nothing to re-synthesize.
         "actiksenan": "Actiskenan"}
# Two right spellings, one taught (owner's choice). ``cefalexine`` follows the INN
# convention, no accent (2026-11-15: it used to keep it as an exception).
SPELLING_WINNERS = {"acyclovir": "aciclovir", "ciclosporine": "cyclosporine",
                    "ciclosporinémie": "cyclosporinémie", "tioguanine": "thioguanine",
                    "céphalexine": "cefalexine", "cephalexine": "cefalexine", "céfalexine": "cefalexine",
                    "alphacalcidol": "alfacalcidol"}
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
# Typos whose fix changes the SOUND, so the clips that carried one had audio saying the
# typo. Owner's rule (2026-11-16): every sound-changing fix is held back in RESYNTH_TYPOS
# (below) until its clips are re-synthesized, and when in doubt it is held (pure accent,
# case, hyphen or silent-letter fixes are not). These were that table: their clips were
# redrawn from the fixed text, and a census on 2026-11-19 (``21_reopen_resynth_typos.py``,
# every stage file and all 42 release manifests) found no clip still carrying a key, so
# they are now ordinary HAND_REWRITES, kept as a guard against the typo coming back.
# docs/BUILD_LOG.md ("Label spelling fixes after the v2 model") has the clip ids.
# ``actisénan`` is how its clip spelled it.
RESYNTHESIZED_TYPOS = {
    # First five (2026-11-16), one PARHAF clip each.
    "perindipril": "perindopril", "cetriaxone": "ceftriaxone", "bisorolol": "bisoprolol",
    "actisenan": "Actiskenan", "actisénan": "Actiskenan", "karegic": "Kardégic",
    # Owner's typo list and the systematic search.
    "perindoprill": "perindopril", "entreesto": "Entresto",
    "hydrochlorothéiazide": "hydrochlorothiazide", "ackiskenan": "Actiskenan",
    "actikskenan": "Actiskenan", "actikenan": "Actiskenan", "doxycyline": "doxycycline",
    "cefriaxone": "ceftriaxone", "amplodipine": "amlodipine", "pantopazole": "pantoprazole",
    "desmoprésine": "desmopressine", "candersartan": "candesartan", "aprazolam": "alprazolam",
    "irbersartan": "irbésartan", "adalmumab": "adalimumab", "zolendronique": "zolédronique",
    "formeterol": "formotérol", "formétérol": "formotérol", "imbrutinib": "ibrutinib",
    "glicazide": "gliclazide", "lanceotide": "lanréotide", "selpercatnib": "selpercatinib",
    "tezepilumab": "tezepelumab", "rilmenildine": "rilmenidine", "teimocilline": "temocilline",
    "bénzatine": "benzathine", "métrondiazole": "métronidazole", "lévifloxacine": "levofloxacine",
    "acétrétine": "acitrétine", "eczétimibe": "ézétimibe", "tocoferol": "tocophérol",
    "theralithe": "Teralithe", "bacrim": "Bactrim", "ferinjec": "Ferinject",
    # English INN forms in a French label (voxtral may read the final ``-in`` as a nasal).
    "empagliflozin": "empagliflozine", "midostaurin": "midostaurine", "darbepoetin": "darbepoetine",
    # Source sheet (``02_drugs`` CNAM label), spoken ``polyo-mi-lite``.
    "diphtérie-polyomielite-tétanos": "diphtérie-poliomyélite-tétanos",
    # Spelling winner and owner's hand review.
    "alfacalcidiol": "alfacalcidol", "laxilix": "Lasilix", "cestradiol": "estradiol"}

# The hold itself: sound-changing typo fixes whose clips still say the typo, kept out of
# HAND_REWRITES (and the JSON) until ``21_reopen_resynth_typos.py`` has queued those clips
# and the driver has redrawn them; then they move to a dict like RESYNTHESIZED_TYPOS above.
# The re-synthesis rewrite is ``label_conventions.normalize_accents(text, RESYNTH_TYPOS)``,
# which also elides the article (``le cestradiol`` -> ``l'estradiol``). Emptied on
# 2026-11-19, refilled on 2026-11-20; ``21_reopen_resynth_typos.py`` loads it by name, and
# ``tests/test_label_variant_audit.py::test_resynth_typos_held_back`` keeps it out of the JSON.
RESYNTH_TYPOS: dict[str, str] = {
    # Owner's unification of 2026-11-20 (see OWNER_SPELLING_20261120), the sound-changing
    # or doubtful half, about 180 clips.
    "bégnine": "bénigne", "bégnines": "bénignes",
    "pléiomorphe": "pléomorphe", "pléiomorphes": "pléomorphes",
    "lipocalicine": "lipocaline", "lipocalycine": "lipocaline",
    "damètre": "diamètre", "dystropnie": "dystrophie",
    "déferrioxamine": "desferrioxamine",
    "castelman": "Castleman", "jiroveci": "jirovecii",
    "soemmering": "Sömmerring", "soemmerring": "Sömmerring",
    "jadassohn-levandovsky": "Jadassohn-Lewandowsky",
    "gangraenosum": "gangrenosum",
}

# Same hold, for fixes that span several words (``RESYNTH_TYPOS`` is looked up word by word),
# read by ``21_reopen_resynth_typos.py``. Meant for scrambled dictionary terms, e.g. the
# extraction moved a headword's trailing parenthesis to the front (``moniteur (d'un appareil
# de radiothérapie)`` became ``d'un appareil de radiothérapie moniteur``). Each key must be a
# term WITH the words around it that make it unambiguous, so no key matches a clip of another
# term, and no value may contain a key, so a second run changes nothing
# (``tests/test_reopen_resynth_typos.py``). Never merged into HAND_REWRITES: once the clips are
# re-synthesized no label carries the old phrase. Empty since the 2026-11-17 batch (78
# phrases) was redrawn and a census on 2026-11-19 found no clip still carrying one;
# those entries are in git history.
RESYNTH_PHRASES: dict[str, str] = {}

# Owner's hand review of the leftover variants (2026-11-15). Brands take the drug caser's
# capitals (``utils/drug_casing.json``), else a second pass would recase them. ``tymoferol``
# maps to itself so an all-caps ``TYMOFEROL`` gets the table casing. ``mesalamine`` GAINS
# its accent (French spelling). Reviewed and kept as they are: gonadotropine, homatropine.
OWNER_REVIEW = {"patidexaméthasone": "patidexamethasone", "tymoferol": "tymoferol",
                "thiorphan": "Tiorfan",
                "cromoglycique": "cromoglicique", "oxyconazole": "oxiconazole",
                "mesalamine": "mésalamine"}
# Eponyms whose labels lost their inner capital (2026-11-17, owner: "fix Mounier-Kuhn in
# the labels too"). The case transfer kept only the first letter, so ``Mounier-Kühn``
# became ``Mounier-kuhn`` (fixed in ``label_conventions._case_like``). A canonical with
# capitals is written as is, and the key without the umlaut repairs the labels already
# built when ``99_hf_release/05_normalize_text.py`` runs again. Casing only: silent, the
# audio stays.
# The 10 after Mounier-Kuhn are the other victims of the same case transfer found in
# the labels on 2026-11-17 (132 occurrences in all); each has its minority spelling and
# its own lowercase form as keys.
EPONYM_CASING = {
    "mounier-kühn": "Mounier-Kuhn",
    "mounier-kuhn": "Mounier-Kuhn",
    "gömöri-grocott": "Gomori-Grocott",
    "gomori-grocott": "Gomori-Grocott",
    "malpighi-gärtner": "Malpighi-Gartner",
    "malpighi-gartner": "Malpighi-Gartner",
    "lich-grégoir": "Lich-Gregoir",
    "lich-gregoir": "Lich-Gregoir",
    "gerstmann-straüssler-scheinker": "Gerstmann-Sträussler-Scheinker",
    "gerstmann-sträussler-scheinker": "Gerstmann-Sträussler-Scheinker",
    "heřmansky-pudlak": "Hermansky-Pudlak",
    "hermansky-pudlak": "Hermansky-Pudlak",
    "chediak-higashi": "Chédiak-Higashi",
    "chédiak-higashi": "Chédiak-Higashi",
    "boucher-neuhaüser": "Boucher-Neuhäuser",
    "boucher-neuhäuser": "Boucher-Neuhäuser",
    "rokitansky-kuster-hauser": "Rokitansky-Küster-Hauser",
    "rokitansky-küster-hauser": "Rokitansky-Küster-Hauser",
    "klippel-trenaunay": "Klippel-Trénaunay",
    "klippel-trénaunay": "Klippel-Trénaunay",
    "roch-leri": "Roch-Léri",
    "roch-léri": "Roch-Léri",
}
# Owner's spellings (2026-11-18), casing and accents only, silent: ``Sjögren`` (2
# ``Sjogren`` in labels), the brand ``Teralithe`` without its accent (19 ``Téralithe``;
# the drug caser still writes ``Téralithe``, which the TTS text keeps, and this row
# wins for labels). The clip ``Hem-o-lok`` is a rule (see ``KEEP``). The ``hCG`` and ``méti-S``
# rows only keep the table from undoing ``label_conventions._SPELLING_RULES``, which
# does those (and ``SARS-CoV-2``): the generated rows wrote ``bêta-hcg`` and ``meti-s``.
OWNER_SPELLING_20261118 = {
    "sjogren": "Sjögren",
    "téralithe": "Teralithe",
    "beta-hcg": "bêta-hCG", "béta-hcg": "bêta-hCG", "bêta-hcg": "bêta-hCG",
    "méti-s": "méti-S",
}
# Owner's unification of 2026-11-20: dictionary headwords that were misspelled, or spelled
# differently from the rest of the labels, so one term had two written forms. This is the
# silent half (double letters, a mute ``h``, ``k``/``c``, ``ß``/``ss``, a hyphen): the
# audio already says the fixed word. The sound-changing half waits in RESYNTH_TYPOS.
# ``déferrioxamine`` only occurs in French text; ``peroxisome`` only inside the English
# ``peroxisome proliferator-activated receptor`` and is left alone.
# ``méthyle aminolévulinate`` -> ``méthyl`` is two words, so it is a
# ``label_conventions._SPELLING_RULES`` regex.
OWNER_SPELLING_20261120 = {
    "cokett": "Cockett", "abrikossof": "Abrikossoff", "rosenmüler": "Rosenmüller",
    "grynfelt": "Grynfeltt", "lachapèle": "Lachapelle",
    "schwartzman": "Shwartzman", "sanarelli-schwartzman": "Sanarelli-Shwartzman",
    "kußmaul": "Kussmaul", "kußmaul-maier": "Kussmaul-Maier",
    "thornwaldt": "Tornwaldt",
    "idradénome": "hidradénome", "hydrocystome": "hidrocystome",
    "médulaire": "médullaire", "souscutanée": "sous-cutanée",
    "falsiforme": "falciforme", "correctopie": "corectopie",
    "callidine": "kallidine",
    "karyophérine": "caryophérine", "karyophérines": "caryophérines",
    "cholestyramine": "colestyramine",
}
HAND_REWRITES = (SOURCE_SHEET_FIXES | TYPOS | SPELLING_WINNERS | INN_LEFTOVERS
                 | INN_LEFTOVERS_MAJORITY | OWNER_REVIEW | EPONYM_CASING | OWNER_SPELLING_20261118
                 | OWNER_SPELLING_20261120
                 | RESYNTHESIZED_TYPOS)


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
@click.argument("variants_tsv", type=click.Path(exists=True), required=False)
@click.option("--min-cost", default=3, show_default=True, help="Skip families whose minority forms total fewer occurrences.")
@click.option("--out", type=click.Path(), default=str(ROOT / "utils" / "accent_spellings.json"), show_default=True)
def main(variants_tsv, min_cost, out):
    # No audit TSV: start from the committed table and only sync HAND_REWRITES into it.
    table = {} if variants_tsv else json.loads(Path(out).read_text(encoding="utf-8"))["rewrites"]
    for line in list(open(variants_tsv, encoding="utf-8"))[1:] if variants_tsv else []:
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
