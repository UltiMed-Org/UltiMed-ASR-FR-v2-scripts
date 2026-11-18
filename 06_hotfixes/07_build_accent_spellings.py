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
# Only fixes that keep the sound (the sound-changing ones are in RESYNTH_TYPOS below).
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
# Typos whose fix changes the SOUND, so every existing clip that carries one has audio
# saying the typo: fixing the label alone would make it disagree with its audio. Held out
# of HAND_REWRITES (and the JSON) until those clips are re-synthesized from the fixed text;
# docs/BUILD_LOG.md ("Label spelling fixes after the v2 model") has the clip ids and the
# steps. Owner's rule (2026-11-16): every sound-changing fix is held back, and when in doubt
# it is held (pure accent, case, hyphen or silent-letter fixes are not). The re-synthesis
# rewrite is ``label_conventions.normalize_accents(text, RESYNTH_TYPOS)``, which also elides
# the article (``le cestradiol`` -> ``l'estradiol``). ``actisénan`` is how its clip spells it.
RESYNTH_TYPOS = {
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

# Same hold, for fixes that span several words (``RESYNTH_TYPOS`` is looked up word by word).
# These are scrambled dictionary terms: the extraction moved a headword's trailing
# parenthesis to the front (``moniteur (d'un appareil de radiothérapie)`` became
# ``d'un appareil de radiothérapie moniteur``) and the sentence generator used the result
# verbatim. Each key is a term WITH the words around it that make it unambiguous, so no key
# matches a clip of another term (checked on every stage file, 2026-11-17), and no value
# contains a key, so a second run changes nothing (``tests/test_reopen_resynth_typos.py``).
# The comment above each group is the dictionary index and its corrected term. Never merged
# into HAND_REWRITES: once the clips are re-synthesized no label carries the old phrase.
RESYNTH_PHRASES = {
    # 023937, the first one found (owner, 2026-11-17).
    "capsule de réfléchie Glisson": "capsule de Glisson réfléchie",
    # 000728 acide peptidonucléique
    "acide peptidique nucléique": "acide peptidonucléique",
    # 006233 aspergillose pulmonaire chronique nécrosante
    "ou semi-invasive aspergillose pulmonaire chronique nécrosante": "ou aspergillose semi-invasive",
    # 012668 triangle de Codman
    "un Codman triangle de": "un triangle de Codman au niveau de",
    # 020198 paralysie faciale du nouveau-né
    "paralysie du nouveau-né faciale droite": "paralysie faciale droite du nouveau-né",
    "paralysie du nouveau-né faciale gauche": "paralysie faciale gauche du nouveau-né",
    "paralysie du nouveau-né faciale gauche complète": "paralysie faciale gauche complète du nouveau-né",
    "paralysie du nouveau-né faciale droite isolée": "paralysie faciale droite isolée du nouveau-né",
    # 020322 facteurs temps d'une irradiation
    "d'une irradiation facteurs temps": "des facteurs temps d'une irradiation",
    # 020868 FFT
    "en IRM FFT": "en IRM avec reconstruction par FFT",
    # 022899 gamma du film
    "du film gamma": "du gamma du film",
    "le développement du film gamma": "le développement du film et son gamma",
    # 022900 gamma
    "en IRM gamma, le rapport gyromagnétique": "en IRM, le rapport gyromagnétique gamma",
    # 024271 Goméga
    "en IRM Goméga": "par le gradient Goméga",
    "avec séquences en IRM Goméga": "avec séquences utilisant le gradient Goméga",
    # 024420 Gphi
    "en IRM Gphi": "à gradient Gphi",
    # 024974 Gz ou Gs
    "en IRM Gz ou Gs": "avec gradient Gz ou Gs",
    "de diffusion et en IRM Gz ou Gs": "de diffusion et à gradient Gz ou Gs",
    # 026359 High Intensity Transient Signals
    "en IRM High Intensity Transient Signals": "en IRM, à la recherche de High Intensity Transient Signals",
    # 027449 hypersignal
    "retrouve en IRM hypersignal": "retrouve un hypersignal",
    "montre en IRM hypersignal": "montre un hypersignal",
    "En IRM hypersignal en séquence": "En IRM, hypersignal en séquence",
    "recherchera en IRM hypersignal": "recherchera en IRM un hypersignal",
    "objective en IRM hypersignal": "objective un hypersignal",
    "évidence en IRM hypersignal": "évidence un hypersignal",
    # 027769 hyposignal
    "en IRM hyposignal": "en hyposignal",
    "avec en IRM hyposignal": "avec un hyposignal",
    # 028117 image en
    "image de en": "image en",
    # 028163 imagerie de l'eau
    "en IRM imagerie de l'eau": "en IRM par imagerie de l'eau",
    # 028166 imagerie en tenseur de diffusion
    "en IRM imagerie en tenseur de diffusion": "en imagerie en tenseur de diffusion",
    # 028182 imageur de radiothérapie
    "de radiothérapie imageur": "imageur de radiothérapie",
    # 028589 incrément
    "un en informatique incrément": "un incrément",
    # 028590 incrément
    "en scanographie incrément": "avec un incrément",
    "protocole en scanographie incrément": "protocole à incrément",
    # 028684 indice de qualité d'un rayonnement
    "d'un rayonnement indice de qualité": "de l'indice de qualité du rayonnement",
    "paramètre d'un rayonnement indice de qualité": "paramètre indice de qualité du rayonnement",
    # 030988 laser
    "en dermatologie laser": "au laser",
    "traitement en dermatologie laser par laser": "traitement par laser",
    "séance en dermatologie laser": "séance de laser",
    # 031324 lésion directement létale
    "d'une cellule lésion directement létale": "d'une lésion directement létale",
    "la mort d'une cellule lésion directement létale": "la mort d'une cellule par lésion directement létale",
    "une atteinte d'une cellule lésion directement létale": "une lésion directement létale",
    "événements d'une cellule lésion directement létale": "événements de lésion directement létale",
    "capable d'une cellule lésion directement létale": "capable d'induire une lésion directement létale",
    # 031330 lésion sublétale
    "d'une cellule lésion sublétale": "d'une lésion sublétale",
    "les dommages d'une cellule lésion sublétale": "les lésions sublétales",
    "réparation d'une cellule lésion sublétale": "réparation des lésions sublétales",
    "accumulation d'une cellule lésion sublétale": "accumulation de lésions sublétales",
    "accumulation d'une cellule lésion sublétale non réparée": "accumulation de lésions sublétales non réparées",
    # 034660 Hutchinson melanotic freckle
    "un de Hutchinson melanotic freckle": "un Hutchinson melanotic freckle",
    "du de Hutchinson melanotic freckle": "du Hutchinson melanotic freckle",
    "type de Hutchinson melanotic freckle": "type Hutchinson melanotic freckle",
    "de de Hutchinson melanotic freckle": "de Hutchinson melanotic freckle",
    # 035432 microphtalmie et déficience mentale de Sjögren et Larsson
    "de Sjögren et Larsson microphtalmie et déficience mentale": "de microphtalmie et déficience mentale de Sjögren et Larsson",
    # 035858 moniteur d'un appareil de radiothérapie
    "d'un appareil de radiothérapie moniteur": "du moniteur de l'appareil de radiothérapie",
    # 036343 multiple
    "soit des unités multiples de la valeur cible": "soit au-dessus de la valeur cible",
    "à des unités multiples de la normale": "à plusieurs fois la normale",
    "soit des unités multiples du seuil": "soit plusieurs fois le seuil",
    "en deçà des unités multiples de la normale basse": "en deçà de la normale basse",
    # 039728 ombre compagne
    "en radiologie ombre compagne": "une ombre compagne",
    "linéaire en radiologie ombre compagne": "linéaire d'ombre compagne",
    # 040923 maladie de Paget du mamelon
    "maladie du mamelon de Paget": "maladie de Paget du mamelon",
    "maladie du mamelon de Paget droite": "maladie de Paget du mamelon droit",
    # 042168 pénombre d'un faisceau
    "d'un faisceau pénombre": "de la pénombre du faisceau",
    # 042407 période
    "avec d'un signal périodique période": "avec une période",
    # 042841 phase d'une maladie
    "d'une maladie phase": "d'une maladie en phase",
    "d'une maladie phase par phase": "de sa maladie phase par phase",
    "d'une maladie phase aiguë puis convalescence": "d'une maladie en phase aiguë puis en convalescence",
    # 048186 réduction septale par alcoolisation
    "par alcoolisation réduction septale": "par réduction septale par alcoolisation",
    # 048782 réseau coordonné de soins en médecine nucléaire
    "réseau en médecine nucléaire réseau coordonné de soins": "réseau coordonné de soins en médecine nucléaire",
    # 051878 syndrome de skin peeling
    "syndrome skin peeling": "syndrome de skin peeling",
    # 053414 surélévation congénitale de la scapula
    "de la scapula surélévation congénitale de l'omoplate": "surélévation congénitale de la scapula",
    "malformation de la scapula surélévation congénitale de l'omoplate": "surélévation congénitale de la scapula",
    "suspicion de la scapula surélévation congénitale de l'omoplate": "suspicion de surélévation congénitale de la scapula",
    "cure de la scapula surélévation congénitale de l'omoplate": "cure de surélévation congénitale de la scapula",
    "antécédent de la scapula surélévation congénitale de l'omoplate": "antécédent de surélévation congénitale de la scapula",
    # 054944 technique 3DFT
    "en IRM technique 3DFT": "en technique 3DFT",
    # 057059 troncature
    "d'une lentille troncature": "d'une lentille à troncature",
}

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
HAND_REWRITES = (SOURCE_SHEET_FIXES | TYPOS | SPELLING_WINNERS | INN_LEFTOVERS
                 | INN_LEFTOVERS_MAJORITY | OWNER_REVIEW | EPONYM_CASING | OWNER_SPELLING_20261118)


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
