"""One written form for titles, dates, clock times, quantities, sutures, compounds, spelling variants and drug names in the ASR label.

Like ``percent_normalize.py``, this canonicalizes what the LLM wrote AFTER the LLM:
UltiMed v1 labels spell the same spoken thing several ways, so the model learns no
consistent form. Each rule below picks the form the corpus already uses most and
rewrites the minority spellings to it:

- **Titles** (``expand_titles``). ``M. Dupont`` (1.6k) -> ``Monsieur Dupont`` (92k),
  ``Mme`` -> ``Madame``, ``Mlle`` -> ``Mademoiselle``, ``Dr`` / ``Pr`` -> ``docteur`` /
  ``professeur``. Casing follows the corpus majority mid-sentence: ``Monsieur`` /
  ``Madame`` / ``Mademoiselle`` are capitalized before a name (``Monsieur Dupont``,
  65k against 1.3k ``monsieur Dupont``), ``docteur`` / ``professeur`` are not (``le
  docteur Martin``, 5k against 0.5k ``le Docteur Martin``); every title is capitalized
  at a sentence start. ``M.`` is only a title before a capitalized name that is not
  itself a sentence opener, and not after another title (``Madame M. présente``,
  ``PCR M. tuberculosis``, ``stade M. Le patient`` are untouched), and a lowercase
  ``monsieur`` / ``madame`` without a name (``le monsieur du lit 4``) stays lowercase.
- **Dates** (``normalize_dates``). Days and years in a date are digits (22.7k ``le 15
  mars`` against 1.2k ``le quinze mars``), and the first of the month is ``1er`` (0.7k
  against 0.5k ``premier mars``). A spelled-out year (1900 to 2099, every spelling:
  ``mille neuf cent``, ``mil neuf cent``, ``dix-neuf cent``, ``deux mille``, hyphenated
  or not, ``vingt et un`` or ``vingt-et-un``) becomes digits after a month, and after
  ``en`` / ``depuis`` when nothing but punctuation, the end, ``au``, ``et`` or ``puis``
  follows (``en deux mille dix-neuf.`` -> ``en 2019.``, but ``depuis deux mille ans``
  and ``mille neuf cent soixante-quatorze grammes`` are left alone). A spelled day
  (2 to 31) becomes digits before a month, or before ``au`` and a converted day
  (``du six au treize mai`` -> ``du 6 au 13 mai``); ``un`` is not converted (``un
  mars`` is not a date anyone dictates).
- **Clock times** (``normalize_clock``). ``14h30`` / ``14 h 30`` / ``09h45`` / ``20 h 00``
  (a few dozen) -> ``14 heures 30`` / ``9 heures 45`` / ``20 heures`` (47k labels write
  ``N heures``), ``1 heure`` in the singular. Durations (``24h``) get the same form,
  which is also how they are written elsewhere.
- **Quantities** (``normalize_quantities``). A spelled number before a unit of measure
  or a duration becomes digits (``quatre milligrammes`` -> ``4 milligrammes``, 62k
  against 1.9k; ``zéro virgule vingt-cinq microgrammes`` -> ``0,25 microgrammes``;
  ``pendant dix-huit mois`` -> ``pendant 18 mois``), and so do ``fois`` / ``séances``
  counts, a range head and a score (``deux fois`` -> ``2 fois``, ``un à deux jours`` ->
  ``1 à 2 jours``, ``deux sur dix`` -> ``2 sur 10``, 2026-09-28), and a single unit or
  duration (``à un mois`` -> ``à 1 mois``, ``un gramme`` -> ``1 gramme``, 2026-10-08).
  Other counts of things stay spelled, as the corpus spells them (``deux comprimés``),
  and so does a single ``une fois`` / ``une séance``. Added 2026-09-28 because the hand-made
  ``drug_sentence`` sets spelled every dose while UltiMed writes digits: the model
  wrote ``4 milligrammes`` on the synthetic voice and 63% of that set's WER was
  number formatting, not recognition. The radiology units (``grays``, ``hertz``,
  ``teslas``) count as units, and a decimal written half in digits becomes a decimal
  (``3 virgule 5 mégahertz`` -> ``3,5 mégahertz``).
- **Suture gauges** (``normalize_sutures``, label only). ``Vicryl trois zéro`` /
  ``Vicryl 3 zéros`` / ``Vicryl 3/0`` -> ``Vicryl 3-0``, only after a suture material.
  Skipped for a TTS source (``tts_source=True``) because voxtral does not read ``3-0``
  reliably as what the source said.
- **Staging numbers** (``normalize_staging``). After ``stade``, ``grade``, ``type``,
  ``classe``, ``palier``, ``niveau``, ``phase`` or ``NYHA``, a Roman or spelled number becomes digits
  (``stade IIIb`` -> ``stade 3b``, ``palier deux`` -> ``palier 2``), the author's choice
  (2026-09-28) over a corpus that mixed all three.
- **Roman numerals** (``normalize_roman``). Elsewhere too, digits (author's choice
  2026-10-08): ``nerf III`` -> ``nerf 3``, ``angiotensine II`` -> ``angiotensine 2``,
  ``Garden IV`` -> ``Garden 4``. Kings, intravenous ``IV`` and letters (``rayons X``,
  ``pénicilline V``) stay.
- **Compounds** (``normalize_compounds``). ``petit déjeuner`` -> ``petit-déjeuner``
  (213 against 35, and the dictionary spelling of the noun), and prefix and
  combining-form compounds are glued unless a digraph would misread (``post-opératoire``
  -> ``postopératoire``, ``extra hépatiques`` -> ``extrahépatiques``, ``abdomino-pelvien``
  -> ``abdominopelvien``, but ``gastro-entérologie`` / ``anti-inflammatoire`` stay;
  2026-10-08, was "glue a split prefix, hyphen before a vowel" since 2026-09-28).
- **Compound spellings** (``normalize_compound_spellings``, 2026-10-08). One form per
  compound the rule above cannot decide, from ``compound_spellings.json`` built by
  ``06_hotfixes/08_build_compound_spellings.py``: ``gastroentérologie`` ->
  ``gastro-entérologie``, ``transsphénoïdale`` -> ``trans-sphénoïdale``, ``sous costale``
  -> ``sous-costale``, ``tep scanner`` -> ``TEP-scanner``. Then the official lab names
  (``normalize_official_names``: ``NT-proBNP``, ``PD-L1``, ``HLA-B27``), ``non`` hyphenated
  before a noun only (``normalize_non``: ``un non-fumeur``, ``patiente non fumeuse``),
  ordinals in words up to ``dixième`` (``normalize_ordinals``: ``3e`` -> ``troisième``,
  ``le 1er mai`` stays), ``doppler`` / ``holter`` lowercase (``lowercase_eponym_nouns``)
  and Latin names Latin in a binomial only (``normalize_latin_names``: ``Herpes
  simplex``, ``un herpès``).
- **Generic labs** (``strip_generic_labs``, 2026-10-09). The lab after a generic drug is
  dropped (``cholecalciferol Viatris 2,5`` -> ``cholecalciferol 2,5``). Unlike the other
  rewrites this one is not spoken identically: the v1 audio is regenerated from the
  rewritten TTS source.
- **Spelling variants** (``normalize_spelling``). ``œ`` -> ``oe`` everywhere (``cœur`` ->
  ``coeur``), ``aigüe`` -> ``aiguë``, ``compte-rendu`` -> ``compte rendu``,
  ``bêta-bloquant`` / ``bêtabloquant`` -> ``bétabloquant``, ``anévrysme`` -> ``anévrisme``,
  ``urèthre`` / ``uréthral`` -> ``urètre`` / ``urétral``, ``aigue`` -> ``aiguë``,
  ``sénior`` -> ``senior``, and a sentence-initial ``A``
  before an infinitive -> ``À`` (``A surveiller`` -> ``À surveiller``). Added 2026-09-28 from an
  inference sweep of val/test, where these pairs were among the most frequent diffs.
- **Accent variants** (``normalize_accents``, 2026-10-08). One accent spelling per word
  from ``accent_spellings.json``, built from the audit by
  ``06_hotfixes/07_build_accent_spellings.py``: ``defect`` -> ``défect``, ``artéfact`` ->
  ``artefact``, ``l'olécrâne`` -> ``l'olécrane``, ``wirsüng`` -> ``wirsung``, and drug
  INNs without accents (``périndopril`` -> ``perindopril``, ``énoxaparine`` ->
  ``enoxaparine``). Runs after the drug caser, which would put the ANSM accents back.
- **Drug names** (``DrugCaser``). One spelling per drug word, from the lexicon
  ``drug_casing.json`` built by ``02_drugs/02_build_drug_casing.py`` (see its
  docstring for how the canonical form is chosen): ``PRIMPERAN`` / ``Primperan`` ->
  ``Primpéran``, ``PARACETAMOL`` -> ``paracétamol``, ``UVEDOSE`` -> ``Uvedose``, with
  ALLCAPS acronyms (``LP``, ``BCG``) protected by the lexicon.

``apply_label_conventions`` runs them all. The text generators apply it in
``_pipeline_shared.parse_asr_training_target``, so every freshly generated label is
canonical (and so is the TTS source ``voxtral_normalize`` derives from it);
``99_hf_release/05_normalize_text.py`` applies it to the v1 manifests, to both the
label and the TTS source: every rewrite here is spoken identically before and after
(``M.`` was read ``monsieur``, ``2019`` and ``deux mille dix-neuf`` are the same
words, ``4 milligrammes`` is read ``quatre milligrammes``, a hyphen and casing are
silent), so the audio still says the rewritten text.

Stdlib only, so both the LLM stack and the light release scripts can import it.
Tested by ``tests/test_label_conventions.py``.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

# voxtral_normalize is a sibling module (stdlib only too), whatever the caller's sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voxtral_normalize import roman_to_int  # noqa: E402

# One alphabetic word (Unicode letters only, so digits, "×" and apostrophes split).
WORD_RE = re.compile(r"[^\W\d_]+")


# Ending of a drug INN (lowercase): ``-mab``, ``-tinib``, ``-pril``, ``-sartan``, ``-taxel``...
INN_STEM = re.compile(r"(?:mab|nib|pril|sartan|olol|alol|prazole|parine|taxel|platine|lukast|flurane|plase|"
                      r"xetine|apine|azépam|azepam|cilline|mycine|cycline|statine|vir|setron|tidine|"
                      r"floxacine|conazole|semide|dronate|glinide|gliptine|gliflozine|rubicine|"
                      r"poétine|poetine|lizumab|ximab|zumab|mumab)s?$")


def fold_key(word: str) -> str:
    """Accent- and case-insensitive key: ``Primpéran`` -> ``PRIMPERAN``."""
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if unicodedata.category(c) != "Mn").upper()


def at_sentence_start(text: str, pos: int) -> bool:
    """True when ``pos`` starts a sentence: the start of the text, or after ".", "!" or "?"
    (spaces allowed). A colon does not end a sentence."""
    prefix = text[:pos].rstrip()
    return not prefix or prefix[-1] in ".!?"


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

# A capitalized word: a name, when it is not one of the words that open a sentence
# (then the "M." before it is a sentence-final initial or TNM stage, not "Monsieur").
_NAME = r"[A-ZÀ-Ý][\w'-]*"
_OPENERS = frozenset("""
    Le La Les L Un Une Des De Du D Il Elle Ils Elles On Nous Vous Je Ce Cet Cette Ces
    Son Sa Ses Leur Leurs Au Aux À En Et Ou Mais Donc Pas Par Pour Sur Sous Dans Avec
    Sans Après Avant Depuis Pendant Lors Il Y Absence Présence Pas Aucun Aucune Tout
    Toute Tous Toutes Cela Ceci Selon Suite Chez Quant Puis Enfin Ainsi Alors
""".split())
# A title word right before "M." makes that "M." an initial (Madame M. Dupont).
_TITLE_BEFORE = re.compile(r"(?:Monsieur|Madame|Mademoiselle|Mme|Mlle|docteur|Docteur|Dr|"
                           r"professeur|Professeur|Pr)\s+$")
_M_AND_MME_RE = re.compile(r"(?<![\w.'-])M\.?\s+et\s+Mme\b\.?")
_M_DOT_RE = re.compile(rf"(?<![\w.'-])M\.(?=\s+{_NAME})")
_ABBREV_RE = re.compile(r"(?<![\w.'-])(Mmes|Mme|Mlles|Mlle)\b\.?")
_ABBREV_FULL = {"Mme": "Madame", "Mmes": "Mesdames", "Mlle": "Mademoiselle", "Mlles": "Mesdemoiselles"}
_DR_PR_RE = re.compile(rf"(?<![\w.'-])(Drs|Dr|Pr)\b\.?(?=\s+{_NAME})")
_DR_PR_FULL = {"Dr": "docteur", "Drs": "docteurs", "Pr": "professeur"}
# Casing of the spelled-out titles before a name.
_UPPER_TITLE_RE = re.compile(rf"\b(monsieur|madame|mademoiselle)(?=\s+{_NAME})")
_LOWER_TITLE_RE = re.compile(rf"\b(Docteur|Professeur)(?=\s+{_NAME})")


def _is_name(following: str) -> bool:
    """Whether the text after a title starts with a name. A sentence opener counts
    only when a capitalized word follows it (``M. Le Corre``, not ``M. Le patient``)."""
    words = following.split(maxsplit=2)
    if not words:
        return False
    if words[0].strip(".,;:") not in _OPENERS:
        return True
    return len(words) > 1 and words[1][:1].isupper()


def expand_titles(text: str) -> str:
    """``M. Dupont`` -> ``Monsieur Dupont``, ``Dr Martin`` -> ``docteur Martin``, etc. Idempotent."""
    text = _M_AND_MME_RE.sub("Monsieur et Madame", text)

    def m_dot(m: re.Match) -> str:
        if not _is_name(text[m.end():]) or _TITLE_BEFORE.search(text[:m.start()]):
            return m.group(0)
        return "Monsieur"
    text = _M_DOT_RE.sub(m_dot, text)
    text = _ABBREV_RE.sub(lambda m: _ABBREV_FULL[m.group(1)], text)

    def dr_pr(m: re.Match) -> str:
        full = _DR_PR_FULL[m.group(1)]
        return _cap(full) if at_sentence_start(text, m.start()) else full
    text = _DR_PR_RE.sub(dr_pr, text)

    def upper(m: re.Match) -> str:
        return _cap(m.group(1)) if _is_name(text[m.end():]) else m.group(1)
    text = _UPPER_TITLE_RE.sub(upper, text)
    return _LOWER_TITLE_RE.sub(
        lambda m: m.group(1) if at_sentence_start(text, m.start()) else m.group(1).lower(), text)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
          "onze", "douze", "treize", "quatorze", "quinze", "seize"]
_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _spell(n: int) -> set[str]:
    """Every spelling of 0..99, words joined by single spaces (hyphens folded to spaces)."""
    if n <= 16:
        return {_UNITS[n]}
    if n < 20:
        return {f"dix {_UNITS[n - 10]}"}
    if n < 70:
        tens, unit = n // 10 * 10, n % 10
        if unit == 0:
            return {_TENS[tens]}
        if unit == 1:
            return {f"{_TENS[tens]} et un"}
        return {f"{_TENS[tens]} {u}" for u in _spell(unit)}
    if n < 80:
        rest = n - 60
        if rest == 11:
            return {"soixante et onze", "soixante onze"}
        return {f"soixante {s}" for s in _spell(rest)}
    if n == 80:
        return {"quatre vingts", "quatre vingt"}
    return {f"quatre vingt {s}" for s in _spell(n - 80)}


def _build_years() -> dict[str, int]:
    years: dict[str, int] = {}
    for rest in range(100):
        tails = [""] if rest == 0 else [" " + s for s in _spell(rest)]
        for tail in tails:
            for head in ("mille neuf cent", "mil neuf cent", "dix neuf cent"):
                years[head + tail] = 1900 + rest
            years["deux mille" + tail] = 2000 + rest
    for head in ("mille neuf cents", "dix neuf cents"):
        years[head] = 1900
    return years


_YEARS = _build_years()
_DAYS = {s: n for n in range(2, 32) for s in _spell(n)}
_NUM_WORDS = sorted({w for s in list(_YEARS) + list(_DAYS) for w in s.split()} | {"cents"},
                    key=len, reverse=True)
# A run of number words, hyphens or spaces between them.
_SEQ = rf"(?i:\b(?:{'|'.join(_NUM_WORDS)})(?:[\s-]+(?:{'|'.join(_NUM_WORDS)}))*\b)"
_MONTHS = (r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre"
           r"|novembre|d[ée]cembre)")
_YEAR_AFTER_MONTH_RE = re.compile(rf"(\b{_MONTHS}\s+)({_SEQ})")
_YEAR_AFTER_EN_RE = re.compile(rf"(\b(?:en|depuis)\s+)({_SEQ})")
_DAY_BEFORE_MONTH_RE = re.compile(rf"({_SEQ})(\s+{_MONTHS}\b)")
# The first day of a range written before a converted one: "du six au 13 mai".
_DAY_BEFORE_RANGE_RE = re.compile(rf"({_SEQ})(\s+au\s+(?:\d{{1,2}}|1er)\s+{_MONTHS}\b)")
_PREMIER_RE = re.compile(rf"\b[Pp]remier(\s+{_MONTHS}\b|\s+au\s+\d{{1,2}}\s+{_MONTHS}\b)")
# What may follow a year after "en" / "depuis" for it to be a year and not a count.
_AFTER_EN_YEAR_RE = re.compile(r"^(?:\s*(?:[.,;:!?)]|$)|\s+(?:au|et|puis)\b)")


def _words(seq: str) -> list[str]:
    return re.split(r"[\s-]+", seq)


def _year_prefix(seq: str) -> tuple[int, int] | None:
    """Longest leading run of ``seq``'s words that spells a year: (year, n_words)."""
    words = _words(seq)
    for n in range(len(words), 1, -1):
        year = _YEARS.get(" ".join(words[:n]).lower())
        if year:
            return year, n
    return None


def _replace_year(m: re.Match, text: str, guard: bool) -> str:
    seq = m.group(2)
    found = _year_prefix(seq)
    if not found:
        return m.group(0)
    year, n = found
    # Keep the separators of the words after the year exactly as written.
    parts = re.split(r"([\s-]+)", seq)
    rest = "".join(parts[2 * n - 1:])
    if guard and not _AFTER_EN_YEAR_RE.match(rest + text[m.end():]):
        return m.group(0)
    return f"{m.group(1)}{year}{rest}"


def _replace_day(m: re.Match) -> str:
    seq = m.group(1)
    words = _words(seq)
    for n in range(len(words), 0, -1):
        day = _DAYS.get(" ".join(words[-n:]).lower())
        if day:
            parts = re.split(r"([\s-]+)", seq)
            head = "".join(parts[:len(parts) - (2 * n - 1)])
            return f"{head}{day}{m.group(2)}"
    return m.group(0)


def normalize_dates(text: str) -> str:
    """``le quinze mars deux mille vingt`` -> ``le 15 mars 2020``, ``premier mai`` -> ``1er mai``. Idempotent."""
    text = _YEAR_AFTER_MONTH_RE.sub(lambda m: _replace_year(m, text, guard=False), text)
    text = _YEAR_AFTER_EN_RE.sub(lambda m: _replace_year(m, text, guard=True), text)
    text = _DAY_BEFORE_MONTH_RE.sub(_replace_day, text)
    text = _DAY_BEFORE_RANGE_RE.sub(_replace_day, text)
    return _PREMIER_RE.sub(r"1er\1", text)


# ---------------------------------------------------------------------------
# Clock times
# ---------------------------------------------------------------------------

# "14h30", "14 h 30", "20h", "20 h 00"; an uppercase H only attached and with
# minutes ("14H30"), since "les 4H et 4T" is the cardiac-arrest mnemonic. A number
# followed by a comma and another number is an enumeration of labels ("blocs 1a à
# 1h, 2, 3a"), not a time.
_CLOCK_RE = re.compile(r"(?<![\w,.])(\d{1,2})(?:\s?h(?:\s?(\d{2}))?|H(\d{2}))(?![\w])(?!,\s*\d)")


def normalize_clock(text: str) -> str:
    """``14h30`` -> ``14 heures 30``, ``09h00`` -> ``9 heures``, ``1h`` -> ``1 heure``. Idempotent."""
    def repl(m: re.Match) -> str:
        hours, minutes = int(m.group(1)), m.group(2) or m.group(3)
        if minutes is not None and int(minutes) >= 60:
            return m.group(0)
        out = f"{hours} {'heure' if hours <= 1 else 'heures'}"
        return out if minutes in (None, "00") else f"{out} {minutes}"
    return _CLOCK_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------

# Units of measure and durations: the corpus writes the number before them in digits
# (``milligrammes`` 62k digits against 1.9k words, ``ans`` 241k / 2.9k, ``mois``
# 64k / 11k, ``heures`` 38k / 3.3k). ``fois`` and ``séances`` were split (8.9k digits
# against 31.7k words, 1.8k against 2.2k) and the author chose digits (2026-09-28,
# "2 fois par jour"), although words were the majority. Other counts of things are NOT
# here, the corpus spells them (``comprimés`` 263 digits / 406 words, ``bouffées``,
# ``gélules``, ``prises``, ``doses``): "deux comprimés de 4 milligrammes". The singular
# forms follow a decimal (``zéro virgule cinq milligramme``) or a single ``un``/``une``,
# which becomes ``1`` before a unit or a duration (``à un mois`` -> ``à 1 mois``, author's
# choice 2026-10-08: the corpus mixed both in the same phrases, 16.6k ``un mois`` against
# 4.6k ``1 mois``, and the fine-tuned model wrote whichever it liked, about 2.6k flips in
# a third of the corpus) but stays spelled before ``fois`` / ``séances`` (``une fois par
# jour``: 4.8k against 64 ``1 fois``, and ``une fois que`` is a conjunction).
_QTY_UNITS = (r"(?:(?:milli|micro|nano|kilo)?grammes?|kilos?|(?:milli|micro|centi)?litres?"
              r"|(?:milli|centi|kilo)?mètres?|(?:milli|micro)moles?|unités?|degrés?"
              r"|heures?|minutes?|secondes?|jours?|semaines?|mois|ans?|fois|séances?"
              # Radiology units: 588 labels write "N grays", 11 spelled them; hertz
              # 356 / 3, teslas 114 / 3.
              r"|(?:milli|centi)?grays?|(?:kilo|méga)?hertz|teslas?|dioptries?)")
# Every word a French number up to 999,999 is spelled with ("et" only inside one).
_NUMBER_VALUES = {w: i for i, w in enumerate(_UNITS)} | {w: n for n, w in _TENS.items()}
_NUMBER_VALUES |= {"une": 1, "vingts": 20}
_QTY_WORD = rf"(?:{'|'.join(sorted(_NUMBER_VALUES, key=len, reverse=True))}|cents?|mille|mil|et)"
# Not after an elision: "d'un mois" stays, "d'1 mois" is not written.
_QTY_SEQ = rf"(?i:(?<![\w'’-]){_QTY_WORD}(?:[\s-]+{_QTY_WORD})*)"
# A ratio's second number ("cinq jours sur sept") follows the first, else the label
# mixes forms ("5 jours sur sept") the corpus never uses (362 all-words, 280 all-digits).
# Not right after "pour": "0,10 gramme pour cent une fois par jour" is a percentage
# followed by a count, not "101 fois" (the match then starts at "une", which stays).
_QUANTITY_RE = re.compile(rf"(?<!\bpour )({_QTY_SEQ})(?:(\s+virgule\s+)({_QTY_SEQ}))?(\s+{_QTY_UNITS})\b"
                          rf"(?:(\s+sur\s+)({_QTY_SEQ})\b)?")
# A decimal the LLM wrote half in digits (94 labels, "3 virgule 5 mégahertz"): the
# corpus writes "3,5" everywhere else.
_DIGIT_VIRGULE_RE = re.compile(r"(?<![\w,])(\d+) virgule (\d+)\b")
# The first number of a range whose second one is already digits ("deux à 3 jours",
# ~200 labels, and what ``_QUANTITY_RE`` alone makes of "deux à trois jours", since only
# the number next to the unit matches it). "un à 2 jours" -> "1 à 2 jours" too.
_RANGE_HEAD_RE = re.compile(rf"({_QTY_SEQ})(\s+(?:à|ou)\s+\d+(?:,\d+)?\s+{_QTY_UNITS})\b")
# A score out of ten, twenty or a hundred (pain scale, "deux sur dix"): 6k labels write
# "2 sur 10", 146 spell it.
# Where a single one stays spelled, unlike a single unit of measure or duration ("1 mois",
# author's choice 2026-10-08): the counts (``une fois par jour``, ``une séance``) and the
# words that are rarely a measure after ``un``/``une`` (``un degré de gravité``, ``une
# unité de soins``, ``en seconde intention``).
_COUNT_UNIT_RE = re.compile(r"\s+(?:fois|séances?|degrés?|unités?|secondes?)$")
_SCORE_BASE = {"dix": 10, "vingt": 20, "cent": 100}
_SCORE_RE = re.compile(rf"({_QTY_SEQ})(\s+sur\s+)(dix|vingt|cent)\b")
# Suture gauges: "Vicryl trois zéro" (~250), "Vicryl 3 zéro" (~255), "Vicryl 3-0" (~300,
# also what the model writes). Only right after a suture material, optionally followed
# by "rapide" / "résorbable" / "fast", because "N zéro" elsewhere is not a gauge
# ("schéma zéro un zéro", a score). "Vicryl zéro" (a gauge of plain 0) stays.
_SUTURE_MATERIAL = (r"vicryl|monocryl|prol[èe]ne|pds|surgil[èe]ne|ticron|monofil(?:ament)?|fils?"
                    r"|[dl]'[eé]th[iy]lon|[eé]th[iy]lon|polysorb|fil ?[àa] ?peau|polydioxanone"
                    r"|monosyn|v-?lo[c]?k|velock|flexocrin|nylon|soie|caprosyn|dafilon"
                    r"|polypropyl[èe]ne|r[ée]sorbables?")
_GAUGE_NUM = r"un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|\d{1,2}"
_SUTURE_RE = re.compile(
    rf"(?i:\b((?:{_SUTURE_MATERIAL})(?:\s+(?:rapide|r[ée]sorbable|fast))?\s+)({_GAUGE_NUM})"
    rf"(?:[\s-]+zéros?\b|/0\b))")


def parse_french_number(words: list[str]) -> int | None:
    """Value of spelled-out French number words, or None when they do not spell one.

    ``["deux", "cent", "cinquante"]`` -> 250, ``["quatre", "vingt", "dix", "sept"]``
    -> 97, ``["trois", "mille", "cinq", "cents"]`` -> 3500. Hyphens must already be
    split off. Lenient on spelling variants (``cent``/``cents``, ``vingt``/``vingts``),
    strict on structure (``_can_follow``, one ``mille``, one ``cent`` per thousand).
    """
    total, current = 0, 0
    for i, raw in enumerate(words):
        w = raw.lower()
        if w == "et":
            if i == 0 or i == len(words) - 1:
                return None
            continue
        if w in ("cent", "cents"):
            if current >= 100:
                return None
            current = (current or 1) * 100
        elif w in ("mille", "mil"):
            if total:
                return None
            total, current = (current or 1) * 1000, 0
        elif w in ("vingt", "vingts") and current % 100 == 4:
            current += 76  # "quatre vingt" is 80
        elif w in _NUMBER_VALUES:
            if not _can_follow(current % 100, _NUMBER_VALUES[w]):
                return None
            current += _NUMBER_VALUES[w]
        else:
            return None
    return total + current


def _can_follow(r: int, v: int) -> bool:
    """Whether a unit/tens word worth ``v`` may follow a number ending in ``r`` (its last
    two digits). Rejects juxtapositions that are not one number: ``deux trois jours``
    (two or three days) is not 5, ``vingt trente`` is not 50."""
    if r == 0:
        return True
    if r % 10 == 0 and r >= 20:
        return v < 20 if r in (60, 80) else v < 10
    return r in (10, 70, 90) and v in (7, 8, 9)


def _decimal_digits(words: list[str]) -> str | None:
    """Digits after "virgule" as said: ``zéro cinq`` -> ``05``, ``cinquante`` -> ``50``."""
    zeros = 0
    while zeros < len(words) and words[zeros].lower() == "zéro":
        zeros += 1
    rest = parse_french_number(words[zeros:]) if words[zeros:] else None
    if words[zeros:] and rest is None:
        return None
    return "0" * zeros + ("" if rest is None else str(rest))


def _replace_quantity(m: re.Match) -> str:
    whole = _words(m.group(1))
    # A leading "et" belongs to the sentence ("et deux milligrammes"), not the number.
    head = ""
    while whole and whole[0].lower() == "et":
        head += whole.pop(0) + " "
    value = parse_french_number(whole) if whole else None
    if value is None:
        return m.group(0)
    ratio = ""
    if m.group(6) is not None:
        per = parse_french_number(_words(m.group(6)))
        # Not a ratio number ("sur deux trois"): leave the tail as written.
        ratio = f"{m.group(5)}{per}" if per is not None else f"{m.group(5)}{m.group(6)}"
    if m.group(3) is not None:
        decimals = _decimal_digits(_words(m.group(3)))
        if decimals is None:
            return m.group(0)
        return f"{head}{value},{decimals}{m.group(4)}{ratio}"
    if value == 1 and _COUNT_UNIT_RE.match(m.group(4)):
        # "une fois par jour", "une unité de soins": see _COUNT_UNIT_RE.
        return m.group(0)
    return f"{head}{value}{m.group(4)}{ratio}"


def normalize_quantities(text: str) -> str:
    """``quatre milligrammes`` -> ``4 milligrammes``, ``zéro virgule vingt-cinq microgrammes``
    -> ``0,25 microgrammes``, ``toutes les huit heures`` -> ``toutes les 8 heures``,
    ``dans un mois`` -> ``dans 1 mois``.

    Only before a unit of measure, a duration, ``fois`` or ``séances`` (``_QTY_UNITS``);
    other counts (``deux comprimés``) and a single ``une fois`` / ``une séance`` stay
    spelled, except as the head of a range (``un à deux fois`` -> ``1 à 2 fois``). A score becomes digits
    too (``deux sur dix`` -> ``2 sur 10``). Minutes after
    a spelled hour (``huit heures trente``) are left alone: telling them from a count
    that follows (``deux heures deux fois par jour``) needs more than a regex. Idempotent.
    """
    text = _QUANTITY_RE.sub(_replace_quantity, text)
    text = _RANGE_HEAD_RE.sub(lambda m: _digits_or_keep(m, 1) + m.group(2), text)
    text = _SCORE_RE.sub(lambda m: f"{_digits_or_keep(m, 1)}{m.group(2)}{_SCORE_BASE[m.group(3)]}"
                         if parse_french_number(_words(m.group(1))) is not None else m.group(0), text)
    return _DIGIT_VIRGULE_RE.sub(r"\1,\2", text)


def _digits_or_keep(m: re.Match, group: int) -> str:
    """Group ``group`` of ``m`` (spelled number words, maybe led by a sentence "et") as
    digits, or unchanged when the words do not spell one number."""
    whole = _words(m.group(group))
    head = ""
    while whole and whole[0].lower() == "et":
        head += whole.pop(0) + " "
    value = parse_french_number(whole) if whole else None
    return m.group(group) if value is None else f"{head}{value}"


def normalize_sutures(text: str) -> str:
    """Suture gauges to ``N-0``: ``Vicryl trois zéro`` -> ``Vicryl 3-0``.

    Label only (``apply_label_conventions(..., tts_source=True)`` skips it): voxtral
    does not read ``3-0`` reliably as "trois zéro" (Whisper heard ``3-0``, ``3.0`` and
    ``2 à 0`` on such clips), so a TTS source that said ``trois zéro`` keeps saying it.
    """
    return _SUTURE_RE.sub(_replace_suture, text)


# Thousands glued (author's choice 2026-10-08, the majority: 26k labels wrote ``3200``,
# 4.6k ``3 200``), so ``18 000`` and ``18000`` are not two transcripts of one sound.
# Groups after the first are exactly three digits, so ``12 18`` (two numbers) stays.
_THOUSANDS_RE = re.compile(r"(?<![\w,.])(\d{1,3})((?: \d{3})+)(?![\d])")


def normalize_thousands(text: str) -> str:
    """``18 000`` -> ``18000``, ``1 sur 10 000`` -> ``1 sur 10000``. Label only: the TTS
    source keeps the spaced form voxtral reads as one number. Idempotent."""
    return _THOUSANDS_RE.sub(lambda m: m.group(1) + m.group(2).replace(" ", ""), text)


def _replace_suture(m: re.Match) -> str:
    """``Vicryl trois zéro`` / ``Vicryl 3 zéros`` / ``Vicryl 3/0`` -> ``Vicryl 3-0``."""
    num = m.group(2)
    value = num if num.isdigit() else parse_french_number([num.lower()])
    return f"{m.group(1)}{value}-0"


# ---------------------------------------------------------------------------
# Staging numbers
# ---------------------------------------------------------------------------

# After a staging / classification word the corpus mixed Roman numerals, digits and
# words (stade 5.4k Roman / 0.4k digits, grade 2.9k / 2.4k, type 6.2k / 3.4k, classe
# 0.6k / 0.7k, palier 0.8k / 5.0k / 0.5k "palier deux"), the same sound written three
# ways. The author chose digits for all of them (2026-09-28): "stade 3", "grade 2b",
# "type 1 et 2". ``phase`` joined on 2026-10-08 ("phase II" 222 times heard as "phase 2"
# in a third of the corpus, the author chose digits). Roman numerals elsewhere are ``normalize_roman``'s. Voxtral read every form the
# same, so the TTS source gets the rewrite too.
_STAGING_WORD = r"\b(?:[Ss]tades?|[Gg]rades?|[Tt]ypes?|[Cc]lasses?|[Pp]aliers?|[Nn]iveaux?|[Pp]hases?|NYHA)"
# A Roman numeral (uppercase only: "civil" is valid Roman letters) with an optional
# sub-stage letter ("IIIb", "IVB"), a spelled number, or digits. "un" not before "peu"
# ("un type un peu particulier").
_STAGE_NUM = (r"(?:[IVX]+[A-Da-d]?|(?:un(?!\s+peu\b)|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)"
              r"|\d+[A-Da-d]?)")
# Not before ``-word``: the TTS source respells ``de type IgM`` as ``de type I-gé-M``.
_STAGING_RE = re.compile(
    rf"({_STAGING_WORD}\s+)({_STAGE_NUM}(?:\s*(?:-|/|à|et|ou|,)\s*{_STAGE_NUM})*)\b(?![-'’]\w)")
_STAGE_TOKEN_RE = re.compile(r"\b([IVX]+)([A-Da-d]?)\b|\b(un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\b")


def _stage_token(m: re.Match) -> str:
    if m.group(3):
        return str(_NUMBER_VALUES[m.group(3)])
    n = roman_to_int(m.group(1))
    # A letter that happens to be Roman ("type C" is 100, "classe D" is 500) is not a
    # stage, nor is a karyotype ("type XX" is not 20): stages stop at 12.
    return m.group(0) if n is None or not 1 <= n <= 12 else f"{n}{m.group(2)}"


def normalize_staging(text: str) -> str:
    """``stade IIIb`` -> ``stade 3b``, ``palier deux`` -> ``palier 2``, ``grade I à II``
    -> ``grade 1 à 2``, ``classe III NYHA`` -> ``classe 3 NYHA``. Idempotent."""
    return _STAGING_RE.sub(lambda m: m.group(1) + _STAGE_TOKEN_RE.sub(_stage_token, m.group(2)), text)


# Roman numerals outside the staging words, digits too (author's choice 2026-10-08: the
# 2,999 labels with ``nerf III``, ``segment IV``, ``facteur VIII``, ``Garden IV``,
# ``angiotensine II`` were heard as digits). A multi-letter numeral is always a number
# (``II`` to ``XII``, sub-stage letter kept: ``IIb`` -> ``2b``) but for ``IV``, which is
# also intravenous (``voriconazole IV``, ``Rifadine IV pendant``), and kings (``Henri
# IV``). ``IV`` and the one-letter ``I`` / ``V`` / ``X`` are numbers only after a numbered
# structure (``nerf``, ``facteur``) or a named classification (``Lefort I``, ``Garden
# IV``), or inside a chain that a number starts (``III et IV``); elsewhere they are
# letters (``rayons X``, ``pénicilline V``, ``troponine I``, ``chromosome X``).
_ROMAN_HEADS = (r"(?:[Nn]erfs?|[Ss]egments?|[Ff]acteurs?|[Cc]omplexes?|[Zz]ones?|[Ss]ecteurs?|[Hh]orizons?"
                r"|[Cc]ouches?|[Pp]hototypes?|[Dd]egrés?|[Cc]ollagènes?|[Aa]ngiotensine|[Tt]opoisomérases?"
                r"|[Aa]ntithrombine|Garden|Lefort|Fort|Clark|Forrest|Bosniak|Mobitz|Tokyo|Chaussé|Billroth"
                r"|Bethesda|Fuhrman|Carnegie|Richardson|Lang|Simpson|Gold|Chiari|Siewert|Piver|Hinchey"
                r"|Mallampati|Bismuth|Broders|Cauchoix|Tyradz|Salter|Schatzker|Neer|Hunt|Fisher|Hurley"
                r"|Ficat|Kellgren|Outerbridge|Gustilo|BI-RADS|ACR)")
_ROMAN_KINGS = (r"(?:Henri|Louis|Charles|Napoléon|François|Philippe|Pie|Benoît|Jean-Paul|Élisabeth"
                r"|Elizabeth|Édouard|Guillaume)")
_ROMAN_MULTI = ("XII", "XI", "IX", "VIII", "VII", "VI", "III", "II")
_ROMAN_ANY = r"(?:XII|XI|IX|VIII|VII|VI|IV|V|III|II|I|X)[a-d]?"
_ROMAN_SEP = r"(?:\s*(?:-|/|à|et|ou|,)\s*)"
# ``nerfs III, IV et VI``, ``Lefort I``, ``III et IV``, ``Henri IV``
_ROMAN_RE = re.compile(
    rf"(\b{_ROMAN_HEADS}\s+|\b{_ROMAN_KINGS}\s+)?(?<![-\w])({_ROMAN_ANY}(?:{_ROMAN_SEP}{_ROMAN_ANY})*)\b(?![-'’]\w)")
_ROMAN_TOKEN_RE = re.compile(r"\b([IVX]+)([a-d]?)\b")


def _roman(m: re.Match) -> str:
    head, chain = m.group(1) or "", m.group(2)
    tokens = _ROMAN_TOKEN_RE.findall(chain)
    if re.match(_ROMAN_KINGS, head) or not head and not any(t in _ROMAN_MULTI for t, _ in tokens):
        return m.group(0)
    # ``facteur Xa`` (activated factor, as in ``anti-Xa``) is a name, not ``10a``
    return head + _ROMAN_TOKEN_RE.sub(
        lambda t: t.group(0) if t.group(0) == "Xa" else f"{roman_to_int(t.group(1))}{t.group(2)}", chain)


def normalize_roman(text: str) -> str:
    """``nerf III`` -> ``nerf 3``, ``facteur V Leiden`` -> ``facteur 5 Leiden``, ``Garden
    IV`` -> ``Garden 4``, ``angiotensine II`` -> ``angiotensine 2``; ``rayons X``,
    ``pénicilline V``, ``voriconazole IV``, ``Henri IV`` untouched. Idempotent.

    >>> normalize_roman("Atteinte des nerfs III, IV et VI, fracture Garden IV, Lefort I, III et IV.")
    'Atteinte des nerfs 3, 4 et 6, fracture Garden 4, Lefort 1, 3 et 4.'
    >>> normalize_roman("Rayons X, pénicilline V, troponine I, voriconazole IV, Henri IV, GP IIb IIIa.")
    'Rayons X, pénicilline V, troponine I, voriconazole IV, Henri IV, GP 2b 3a.'
    """
    return _ROMAN_RE.sub(_roman, text)


# ---------------------------------------------------------------------------
# Hyphenated compounds
# ---------------------------------------------------------------------------

# ``petit-déjeuner`` (213 labels) against ``petit déjeuner`` (35); also the
# dictionary spelling of the noun. The WER normaliser deletes hyphens, so the two
# spellings are two different tokens (``petitdéjeuner`` / ``petit déjeuner``) and
# cost a word every time they disagree.
_PETIT_DEJEUNER_RE = re.compile(r"\b([Pp]etits?) (déjeuners?)\b")


# Prefix and combining-form compounds are glued (the 1990 spelling reform, what most
# medical dictionaries now print), author's choice 2026-10-08: ``post-opératoire`` /
# ``post opératoire`` -> ``postopératoire``, ``abdomino-pelvien`` -> ``abdominopelvien``,
# ``oeso-gastro-duodénale`` -> ``oesogastroduodénale``. The corpus mixed all three forms
# (``post-opératoire`` 12.1k, ``postopératoire`` 7.8k, ``post opératoire`` 174), the
# fine-tuned model wrote the glued one, and the WER normaliser keeps a hyphen (one token)
# and a space (two) apart. 05_label_variant_audit.py counted ~550 such families.
#
# The hyphen stays where gluing would misread: two vowels that form a French digraph
# (``gastro-entérologie``, ``intra-utérin``, ``auto-immun``) or a doubled letter
# (``anti-inflammatoire``, ``intra-abdominal``, ``post-traumatique``, ``trans-sphénoïdal``),
# but ``réévaluation`` / ``prééclampsie`` / ``préexistant`` glue, as in the dictionary.
# A part that is itself a word kept hyphenated stops the chain: ``cortico-sous-corticale``. Latin locutions keep
# their form (``post-partum``, ``per os``).
#
# Hyphenated: the first part is one of ``_PREFIXES`` or any lowercase combining form in
# ``-o`` (``cervico-``, ``fémoro-``, ``staturo-``: hundreds, so a pattern rather than a
# list), minus the nouns in ``_NOT_COMBINING``. Not glued: ``sous-`` / ``sus-`` /
# ``demi-`` / ``semi-`` / ``non-`` / ``contre-`` (prepositions and words the reform keeps
# hyphenated), ``écho-`` (``écho-doppler`` 1.8k against 116, left to the author), Greek
# letters (``bêta-lactamine``), names (a capitalized second part: ``Marco-Polo``).
#
# Split by a space: only a bound prefix, or a combining form from ``_SPACED_COMBINING``
# (the ones the audit found split), never before a conjunction or a preposition
# (``intra et extra-hépatiques`` keeps its ``intra``). A bare ``-o`` pattern would glue
# nouns: ``une radio thoracique``, ``chimio adjuvante``, ``placebo contrôlé``.
#
# ``post`` / ``pré`` / ``per`` before a NOUN mean "after / before / during X" and are
# written with a space, author's choice 2026-10-08: ``post-chimiothérapie`` /
# ``post chimiothérapie`` -> ``post chimiothérapie``, ``pré-greffe`` -> ``pré greffe``,
# ``post opération`` stays (gluing gave non-words like ``postopération``). Before an
# adjective they still glue (``postopératoire``, ``prémorbide``). A noun is told from an
# adjective by its ending (``_is_noun``); the established glued nouns in ``_GLUED_NOUNS``
# (``prééclampsie``, ``postcure``) glue as before.
_PREFIXES = frozenset("""anti auto bi co endo extra hémi hyper hypo infra inter intra macro méta micro
    mini mono multi néo oligo para per péri poly post pré primo pro pseudo ré rétro sub supra
    trans tri ultra uni allo""".split())
_NOT_COMBINING = frozenset("""écho echo vidéo video photo zéro numéro stylo euro duo trio solo ratio placebo
    libido lumbago tempo logo mémo métro kilo""".split())
_SPACED_PREFIXES = "anti|auto|extra|hyper|hypo|infra|inter|intra|multi|per|péri|post|pré|pseudo|rétro|supra|trans|ultra"
_SPACED_COMBINING = ("abdomino|anatomo|antéro|brachio|broncho|cérébro|clinico|cutanéo|hépato|insulino"
                     "|latéro|lombo|médiastino|naso|oedémato|postéro|pyélo|sensitivo|supéro|temporo|thoraco|ulcéro")
_LATIN = frozenset({"post-partum", "per-partum", "ante-partum", "post-mortem", "post-abortum"})
_DIGRAPHS = frozenset({"ai", "au", "ei", "eu", "oi", "ou", "oe"})
_NEVER_GLUED = frozenset("sous sus sur contre non demi semi mi".split())
_NOUN_PREFIXES = frozenset({"post", "pré", "per"})
_GLUED_NOUNS = frozenset("""prééclampsie préexcitation précharge postcharge postcure prémédication
    préimplantation prémutation présaturation posthypophyse préalbumine préalbuminémie""".split())
_NOUN_END = re.compile(r"(?:ment|tion|sion|ie|age|ure|ence|ance|isme)s?$")
_ADJ_END = re.compile(r"(?:ique|oire|al|aux|ale|ien|ienne|if|ive|é|ée|ant|ante|aire|eux|euse|el|elle"
                      r"|ile|in|ine|ide|u|ue|i|is)s?$")
_HYPHENATED_RE = re.compile(r"(?<![\w-])[^\W\d_]+(?:-[^\W\d_]+)+(?![\w-])")
_SPACED_RE = re.compile(rf"\b((?i:{_SPACED_PREFIXES}|{_SPACED_COMBINING})) "
                        r"(?!(?:et|ou|ni|sur|de|du|des|à|en)\b)(?=([a-zà-ÿ][^\W\d_]{3,}))")


def _sep(a: str, b: str) -> str:
    """``-`` where gluing ``a`` and ``b`` would misread, else nothing."""
    pair = (a[-1] + b[0]).lower()
    return "-" if pair in _DIGRAPHS or (pair[0] == pair[1] and pair[0] != "é") else ""


def _join(a: str, b: str) -> str:
    return a + _sep(a, b) + b


def _is_combining(part: str) -> bool:
    low = part.lower()
    if part != low and part != low.capitalize():
        return False
    return low in _PREFIXES or (low.endswith("o") and len(low) >= 3 and low not in _NOT_COMBINING)


def _is_noun(word: str) -> bool:
    """Guess from the ending: ``chimiothérapie``, ``traitement``, ``greffe`` are nouns,
    ``opératoire``, ``morbide``, ``remplie`` are not (unknown endings count as nouns).

    >>> [_is_noun(w) for w in ("chimiothérapie", "traitement", "greffe", "opératoire", "remplie")]
    [True, True, True, False, False]
    """
    low = word.lower()
    if low in ("remplie", "remplies"):
        return False
    return bool(_NOUN_END.search(low)) or not _ADJ_END.search(low)


def _spaced_noun(prefix: str, word: str) -> bool:
    """True when ``prefix word`` keeps a space: ``post`` / ``pré`` / ``per`` before a noun."""
    return (prefix.lower() in _NOUN_PREFIXES and _is_noun(word)
            and (prefix + word).lower() not in _GLUED_NOUNS)


def _glue_hyphenated(m: re.Match) -> str:
    word = m.group(0)
    if word.lower() in _LATIN:
        return word
    parts = word.split("-")
    out = parts[0]
    for prev, part in zip(parts, parts[1:]):
        # The second part must be a lowercase word of 3+ letters: "pro-BNP", "apo-E",
        # "co-a", "Marco-Polo" stay.
        if _is_combining(prev) and len(part) >= 3 and part.islower() and part not in _NEVER_GLUED:
            out = out + " " + part if _spaced_noun(prev, part) else _join(out, part)
        else:
            out += "-" + part
    return out


def normalize_compounds(text: str) -> str:
    """``petit déjeuner`` -> ``petit-déjeuner``; prefix and combining-form compounds glued:
    ``post-opératoire`` / ``post opératoire`` -> ``postopératoire``, ``extra hépatiques`` ->
    ``extrahépatiques``, ``gastro-entérologie`` and ``anti-inflammatoire`` keep their
    hyphen, ``post-chimiothérapie`` -> ``post chimiothérapie`` (a noun). Idempotent."""
    text = _PETIT_DEJEUNER_RE.sub(r"\1-\2", text)
    # Only the space is consumed (the second word is a lookahead), so a chain like
    # "thoraco abdomino pelvienne" glues every pair in one pass.
    text = _SPACED_RE.sub(lambda m: m.group(1) + (" " if _spaced_noun(m.group(1), m.group(2))
                                                   else _sep(m.group(1), m.group(2))), text)
    return _HYPHENATED_RE.sub(_glue_hyphenated, text)


# ---------------------------------------------------------------------------
# Spelling variants
# ---------------------------------------------------------------------------

# Pairs spelled two ways with no difference in sound. The WER normaliser keeps both
# spellings apart, so every disagreement costs a word even though the audio cannot
# tell them apart. The choices below are the author's (2026-09-28), mostly the majority:
# - ``oe`` for the ``œ`` ligature, everywhere (``oeil`` 10.7k against ``œil`` 1.5k,
#   ``oedème`` 4.4k against 2.7k; ``cœur`` and ``œsophage`` were the majority but one
#   rule for every word beats a per-word choice nobody can remember).
# - ``ae`` for the ``æ`` ligature too (author's choice 2026-10-08: the model should never
#   write ``æ``; 2.7k labels, ``nævus`` 1.4k, ``cæcum``, ``prævia``, ``melæna``).
# - ``aiguë`` (28.4k) over ``aigüe`` (1.4k), same for ``subaiguë``, ``contiguë``,
#   ``ambiguïté``... (the pre-1990 spelling, the corpus majority). ``Argüelles`` (a name)
#   is left alone because ``güe`` must end the word.
# - ``compte rendu`` (81k) over ``compte-rendu`` (2k).
# - ``bétabloquant`` (1.2k) over ``bêtabloquant`` (1.1k) / ``bêta-bloquant`` (0.1k).
# - ``anévrisme`` (2.7k) over ``anévrysme`` (1.0k).
# - ``urètre`` / ``urétral`` without the h, everywhere (2.8k against 2.3k over the
#   noun and its derivatives together; the noun alone preferred ``urèthre``, 741 / 274,
#   but one rule for the family beats a noun and adjective spelled differently).
_SPELLING_RULES = (
    (re.compile("œ"), "oe"),
    (re.compile("Œ"), "Oe"),
    (re.compile("æ"), "ae"),
    (re.compile("Æ"), "Ae"),
    (re.compile(r"güe(s?)\b"), r"guë\1"),
    (re.compile(r"güité"), "guïté"),
    (re.compile(r"\b([Cc]omptes?)-(rendus?)\b"), r"\1 \2"),
    (re.compile(r"\b([Bb])[êée]ta[- ]?(bloqu(?:ant|eur)\w*)"), r"\1éta\2"),
    (re.compile(r"([Aa])névrysm"), r"\1névrism"),
    (re.compile(r"([Uu])réthr"), r"\1rétr"),
    (re.compile(r"([Uu])rèthr"), r"\1rètr"),
    # 2026-10-08, typos the v2 accent audit found (``tète`` is the verb téter, kept).
    (re.compile(r"\b(\w*(?:[Aa]ig|[Cc]ontig|[Aa]mbig|[Ee]xig))ue(s?)\b"), r"\1uë\2"),
    (re.compile(r"\b([Ss])énior(s?)\b"), r"\1enior\2"),
    (re.compile(r"cléïdo"), "cléido"),
    (re.compile(r"\b([Tt])ete(s?)\b"), r"\1ête\2"),
    (re.compile(r"\b([Gg])enès\b"), r"\1ènes"),
    # 2026-10-08, the majority: ``gène`` (15.7k against 309 ``gene``), ``côlon`` (1.7k /
    # 101), ``scanographique`` (651 / 146), ``shampooing`` (149 / 30), ``paquets-année``
    # (740 / 498), ``grays`` after a number from 2 (756 / 679 ``gray``).
    (re.compile(r"\bgene(s?)\b"), r"gène\1"),  # ``Docteur Genes`` is a name
    (re.compile(r"(?<!-)\b([Cc])olon(s?)\b"), r"\1ôlon\2"),  # not ``semi-colon``
    (re.compile(r"\b([Ss])cannographi"), r"\1canographi"),
    (re.compile(r"\b([Ss])hampoing"), r"\1hampooing"),
    (re.compile(r"\b(paquets?)[- ]années?\b"), r"\1-année"),
    (re.compile(r"(?<![\d,])((?:[2-9]|\d{2,})(?:,\d+)? )gray\b"), r"\1grays"),
)


# "A surveiller", "A jeun", "A l'examen" at a sentence start (~720 labels, against 8.9k
# "À"): the preposition lost its accent. Only before an infinitive, "jeun", "distance"
# or an article: before a participle it is the verb ("A présenté", "A bien toléré",
# telegraphic notes drop the subject), and "A encore" is ambiguous.
_A_GRAVE_RE = re.compile(r"\bA (?=(?:l'|la\b|jeun\b|distance\b|(?!encore\b|\w+oire\b)[^\W\d_]+(?:er|ir|oir|re)\b))")


def _a_grave(text: str) -> str:
    return _A_GRAVE_RE.sub(lambda m: "À " if at_sentence_start(text, m.start()) else m.group(0), text)


def normalize_spelling(text: str) -> str:
    """``cœur`` -> ``coeur``, ``aigüe`` -> ``aiguë``, ``compte-rendu`` -> ``compte rendu``,
    ``bêta-bloquant`` -> ``bétabloquant``, ``anévrysme`` -> ``anévrisme``, ``urèthre`` ->
    ``urètre``, ``A surveiller.`` -> ``À surveiller.`` Idempotent."""
    for pattern, repl in _SPELLING_RULES:
        text = pattern.sub(repl, text)
    text = _a_grave(text)
    return text


# ---------------------------------------------------------------------------
# Compound spellings, official names, ``non``, ordinals, eponyms, Latin
# (author's choices 2026-10-08, from the v2 variant audit)
# ---------------------------------------------------------------------------

COMPOUND_SPELLINGS_PATH = Path(__file__).resolve().parent / "compound_spellings.json"


@lru_cache(maxsize=1)
def _compound_table() -> dict[str, str]:
    return json.loads(COMPOUND_SPELLINGS_PATH.read_text(encoding="utf-8"))["rewrites"]


def _transfer_case(form: str, canon: str) -> str:
    """``canon`` with the letter case of ``form`` (same letters, other separators):
    ``Gastroentérologie`` + ``gastro-entérologie`` -> ``Gastro-entérologie``. A canonical
    with capitals of its own (``TEP-scanner``, ``proBNP``) is kept as is."""
    if canon != canon.lower():
        return _cap(canon) if form[0].isupper() else canon
    # (letter, follows a separator) pairs; a capital that only started a part the
    # canonical glues is dropped: ``Moya-Moya`` -> ``Moyamoya``, not ``MoyaMoya``.
    letters = [(c, i > 0 and form[i - 1] in " -") for i, c in enumerate(form) if c not in " -"]
    out = []
    for c in canon:
        if c in " -":
            out.append(c)
            continue
        k = len(out) - out.count(" ") - out.count("-")
        letter, after_sep = letters[k]
        nxt = letters[k + 1][0] if k + 1 < len(letters) else ""
        if after_sep and out and out[-1] not in " -" and letter.isupper() and not nxt.isupper():
            letter = letter.lower()
        out.append(letter)
    return "".join(out)


def normalize_compound_spellings(text: str, table: dict[str, str] | None = None) -> str:
    """One spelling per compound the labels wrote several ways, from ``compound_spellings.json``
    (``06_hotfixes/08_build_compound_spellings.py``): ``gastroentérologie`` ->
    ``gastro-entérologie``, ``transsphénoïdale`` -> ``trans-sphénoïdale``, ``sous costale``
    -> ``sous-costale``, ``tep scanner`` -> ``TEP-scanner``, ``l'échodoppler`` ->
    ``l'écho-doppler``. Only the forms the audit saw are rewritten, the longest first (up to
    four words). Idempotent."""
    table = _compound_table() if table is None else table
    words = list(WORD_RE.finditer(text))
    out, last, i = [], 0, 0
    while i < len(words):
        start = words[i].start()
        if start and (text[start - 1] == "-" or text[start - 1].isdigit()):
            i += 1
            continue
        for n in range(min(4, len(words) - i), 0, -1):
            j = i + n - 1
            if any(text[words[k].end():words[k + 1].start()] not in (" ", "-") for k in range(i, j)):
                continue
            end = words[j].end()
            # Not a piece of a longer hyphen chain or of a name with digits (``PD-L1``).
            if text[end:end + 1].isdigit() or (text[end:end + 1] == "-" and text[end + 1:end + 2].isalnum()):
                continue
            form = text[start:end]
            canon = table.get(form.lower())
            if canon is not None:
                out.append(text[last:start] + _transfer_case(form, canon))
                last, i = end, j
                break
        i += 1
    return "".join(out) + text[last:]


# Lab names with digits or capitals in the middle, the official forms (the variant
# audit cannot key them: its words drop digits).
_OFFICIAL_RULES = (
    (re.compile(r"\b[Nn][Tt][- ]?[Pp]ro[- ]?(?:BNP|bnp)\b"), "NT-proBNP"),
    (re.compile(r"\bLH[- ]?RH\b"), "LH-RH"),
    (re.compile(r"\b([Aa]nti)[- ]?SS-?([AB])\b"), r"\1-SS\2"),
    (re.compile(r"\bHLA[- ]?(B|DR|DQ)[- ]?(\d{1,2})\b"), r"HLA-\1\2"),
    (re.compile(r"\bPD[- ]?L[- ]?([12])\b"), r"PD-L\1"),
    (re.compile(r"\bPD[- ]?1\b"), "PD-1"),
    (re.compile(r"\bPI[- ]?3[- ]?K\b"), "PI3K"),
    # 2026-10-08, the majority unless noted: ``L4-L5`` (682 against 165 ``L4 L5``, lumbar
    # and sacral levels only, ``T1 T2`` are MRI sequences and ``C3 C4`` complement),
    # ``D-dimères`` (329 / 26), ``18-FDG`` / ``18F-FDG`` (492 / 17), ``Ki-67`` (the
    # official name, 160 against 324 ``Ki67``), ``ASA 2`` (2.2k / 127), ``T CD4`` (174 /
    # 1), ``VIH-1`` (155 / 9), ``46 XX`` (180 / 2), ``G-CSF`` (124 / 17), ``GLP-1`` (89 /
    # 6), ``CA 125`` / ``CA 19-9`` / ``CA 15-3`` (223 / 61).
    (re.compile(r"\b(L\d) (?=[LS]\d\b)"), r"\1-"),
    (re.compile(r"\b[Dd][- ]([Dd])imères\b"), r"D-\1imères"),
    (re.compile(r"\b18 ?(F?)[- ]?FDG\b"), r"18\1-FDG"),
    (re.compile(r"\bKi[- ]?67\b"), "Ki-67"),
    (re.compile(r"\bASA-([1-6])\b"), r"ASA \1"),
    (re.compile(r"\bT(CD[48])\b"), r"T \1"),
    (re.compile(r"\bVIH ([12])\b"), r"VIH-\1"),
    (re.compile(r"\b46(X[XY])\b"), r"46 \1"),
    (re.compile(r"\bGCSF\b"), "G-CSF"),
    (re.compile(r"\bGLP[- ]?1\b"), "GLP-1"),
    (re.compile(r"\bCA[- ]?(125|19[- ]9|15[- ]3)\b"), lambda m: "CA " + re.sub("[- ]", "-", m.group(1))),
)


# Generic labs written after a drug (``cholecalciferol Viatris 2,5``, ``perindopril Arrow
# Génériques``): dropped (user decision 2026-10-09), so the sentence names the drug only.
# Unambiguous lab names only (not ``Accord``, ``EG``, ``Sun``), not after a title
# or a word that introduces a person (``Monsieur Teva``, ``l'enfant Mylan Martin``,
# ``pour Mylan``), and not the products named after their lab (``Phosphore Sandoz``,
# ``Aspirine Upsa``).
_LAB_RE = re.compile(
    r"(?<!Monsieur)(?<!Madame)(?<!docteur)(?<!Docteur)(?<!enfant)(?<!patient)(?<!patiente)(?<!petit)(?<!jeune)"
    r"(?<!petite)(?<!pour)(?<!de)(?<!et)(?<!Phosphate)(?<!Phosphore)(?<=[^\W\d_]) "
    r"(?:Arrow|Biogaran|Cristers|Evolugen|Mylan|Sandoz|Substipharm|Teva|Viatris|Zentiva|Zydus|"
    r"Almus|Ratiopharm|Qualimed|Krka|Ranbaxy)(?: (?:G[ée]n[ée]riques|France|Santé|Pharma))?\b(?! [A-ZÀ-Ý][a-zà-ÿ])")


def strip_generic_labs(text: str) -> str:
    """``cholecalciferol Viatris 2,5`` -> ``cholecalciferol 2,5``. Idempotent."""
    return _LAB_RE.sub("", text)


def normalize_official_names(text: str) -> str:
    """``NT pro BNP`` -> ``NT-proBNP``, ``LHRH`` -> ``LH-RH``, ``anti-SS-A`` -> ``anti-SSA``,
    ``HLA B27`` -> ``HLA-B27``, ``PDL1`` -> ``PD-L1``, ``PD1`` -> ``PD-1``. Idempotent."""
    for pattern, repl in _OFFICIAL_RULES:
        text = pattern.sub(repl, text)
    return text


# ``non`` before a noun is hyphenated, before an adjective it is spaced (the dictionary
# rule): ``un non-fumeur``, ``le non-respect``, ``un taux de non-réponse``, but ``une
# patiente non fumeuse``, ``un lymphome non hodgkinien``. A noun is told by the word
# before ``non``: a determiner, or ``de`` before a noun-looking word.
_DETERMINERS = frozenset("""le la les un une des du ce cet cette ces son sa ses leur leurs mon ma mes
    notre nos votre vos aux au""".split())
_NOT_AFTER_NON = frozenset("""seulement plus pas loin sans si que pour de du des la le les et ou ni à au aux
    en avec par sur dans une un ce cette ces il elle ils elles on est sont moins mais""".split())
# Nouns whatever precedes them: the fixed compounds (``une mutation non-sens``) and the
# noun endings (``non-conformité``, ``en non-appui`` aside). ``-ment`` is left out, it
# would catch adverbs (``non totalement``).
_NON_NOUNS = frozenset("sens lieu dit dits soi être droit appui".split())
_NON_NOUN_END = re.compile(r"(?:tion|sion|ité|age|ure|ence|ance|isme)s?$")
_NON_RE = re.compile(r"\b([Nn]on)[- ](?=([^\W\d_]{3,}))")
_WORD_BEFORE_RE = re.compile(r"([^\W\d_]+)\s+$")


def normalize_non(text: str) -> str:
    """``un non fumeur`` -> ``un non-fumeur``, ``patiente non-fumeuse`` -> ``patiente non
    fumeuse``. Idempotent."""
    def repl(m: re.Match) -> str:
        nxt = m.group(2)
        if nxt.lower() in _NOT_AFTER_NON:
            return m.group(0)
        before = _WORD_BEFORE_RE.search(text[max(0, m.start() - 30):m.start()])
        prev = before.group(1).lower() if before else ""
        low = nxt.lower()
        noun = (prev in _DETERMINERS or low in _NON_NOUNS or bool(_NON_NOUN_END.search(low))
                or (prev == "de" and _is_noun(nxt)))
        return m.group(1) + ("-" if noun else " ")
    return _NON_RE.sub(repl, text)


# Ordinals in words up to ``dixième``, digits with ``e`` from 11 up (``3e percentile`` ->
# ``troisième percentile``, ``15ème`` -> ``15e``). A first-of-month date keeps ``1er``
# (``le 1er mai``, ``du 1er au 5 mai``): ``normalize_dates`` writes it.
_ORDINALS = {2: "deuxième", 3: "troisième", 4: "quatrième", 5: "cinquième", 6: "sixième",
             7: "septième", 8: "huitième", 9: "neuvième", 10: "dixième"}
_ORDINAL_RE = re.compile(r"(?<![\w,.])(\d+)(er|re|ère|e|è|ème|eme)(s?)\b")
_DATE_AFTER_RE = re.compile(rf"\s+(?:(?:au|et|ou)\s+\d{{1,2}}\s+)?{_MONTHS}\b")


def normalize_ordinals(text: str) -> str:
    """``3e`` -> ``troisième``, ``1re`` -> ``première``, ``1er trimestre`` -> ``premier
    trimestre``, ``12ème`` -> ``12e``; ``le 1er mai`` stays. Idempotent."""
    def repl(m: re.Match) -> str:
        n, suffix, plural = int(m.group(1)), m.group(2), m.group(3)
        if n == 1:
            if suffix not in ("er", "re", "ère") or _DATE_AFTER_RE.match(text, m.end()):
                return m.group(0)
            word = "premier" if suffix == "er" else "première"
        elif n in _ORDINALS and suffix not in ("er", "re", "ère"):
            word = _ORDINALS[n]
        elif n > 10 and suffix in ("e", "è", "ème", "eme"):
            return f"{n}e{plural}"
        else:
            return m.group(0)
        word += plural
        return _cap(word) if at_sentence_start(text, m.start()) else word
    return _ORDINAL_RE.sub(repl, text)


# ``doppler`` / ``holter`` are common nouns, lowercase (``Doppler`` 12.7k against 13.1k,
# ``Holter`` 2.3k against 1.1k); the physicist keeps his capital in ``effet Doppler``,
# and a sentence start keeps its capital.
_EPONYM_NOUN_RE = re.compile(r"(?<!effet )\b(Doppler|Holter)s?\b")


def lowercase_eponym_nouns(text: str) -> str:
    """``un écho-Doppler`` -> ``un écho-doppler``, ``un Holter ECG`` -> ``un holter ECG``;
    ``l'effet Doppler`` and ``Doppler artériel.`` at a sentence start stay. Idempotent."""
    return _EPONYM_NOUN_RE.sub(lambda m: m.group(0) if at_sentence_start(text, m.start())
                               else m.group(0).lower(), text)


# Latin in a binomial or when capitalized (a genus), French otherwise: ``Herpes
# simplex``, ``Streptococcus pyogenes``, ``Strongyloides stercoralis``, ``Anopheles``,
# but ``un herpès``, ``des germes pyogènes``, ``des larves strongyloïdes``, ``des
# anophèles``. The genus words take a capital in a binomial; ``pyogenes`` is an epithet.
_LATIN_FORMS = {"herpes": ("Herpes", "herpès"), "pyogenes": ("pyogenes", "pyogènes"),
                "strongyloides": ("Strongyloides", "strongyloïdes"),
                "anopheles": ("Anopheles", "anophèles")}
_LATIN_RE = re.compile(r"\b(?i:herp[eè]s|pyog[eè]nes|strongylo[iï]des|anoph[eè]les)\b")
# The epithets the corpus writes after them; not a pattern, ``herpès iris`` and ``herpès
# virus`` are French.
_EPITHET_RE = re.compile(r"\s+(?:simplex|zoster|gestationn?is|genitalis|labialis|e?s?terc?h?oralis"
                         r"|fulleborni|gambiae|funestus|arabiensis|maculipennis|barbirostris|viridae|virinae)\b")
_GENUS_BEFORE_RE = re.compile(r"\b[A-Za-z]+(?:us|um)\s+$")


def normalize_latin_names(text: str) -> str:
    """``herpes simplex`` -> ``Herpes simplex``, ``un herpes`` -> ``un herpès``,
    ``Streptococcus pyogènes`` -> ``Streptococcus pyogenes``. Idempotent."""
    def repl(m: re.Match) -> str:
        w = m.group(0)
        latin, french = _LATIN_FORMS[fold_key(w).lower()]
        start = at_sentence_start(text, m.start())
        binomial = bool(_EPITHET_RE.match(text, m.end()) or _GENUS_BEFORE_RE.search(text[:m.start()]))
        if binomial or (w[0].isupper() and not start):
            return _cap(latin) if start else latin
        return _cap(french) if start else french
    return _LATIN_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Drug names
# ---------------------------------------------------------------------------

DRUG_CASING_PATH = Path(__file__).resolve().parent / "drug_casing.json"


class DrugCaser:
    """Rewrite each drug word to its canonical spelling (see ``02_drugs/02_build_drug_casing.py``).

    Parameters
    ----------
    caps : dict[str, str]
        Accent-free ALLCAPS key -> canonical form, for ALLCAPS spellings.
    variants : dict[str, str]
        Lowercased non-ALLCAPS spelling -> canonical form.
    """

    def __init__(self, caps: dict[str, str], variants: dict[str, str]) -> None:
        self.caps = caps
        self.variants = variants

    @classmethod
    def from_json(cls, path: Path = DRUG_CASING_PATH) -> "DrugCaser":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(caps=data["caps"], variants=data["variants"])

    def __call__(self, text: str) -> str:
        """Idempotent: every canonical form maps to itself."""
        def repl(m: re.Match) -> str:
            w = m.group(0)
            if w.isupper() and len(w) > 1:
                canon = self.caps.get(fold_key(w))
            else:
                canon = self.variants.get(w.lower())
                # A lowercase canonical keeps the capital the spelling had (a
                # list item, a heading): only the accents are harmonized.
                if canon and w[0].isupper():
                    canon = _cap(canon)
            if canon is None:
                return w
            return _cap(canon) if at_sentence_start(text, m.start()) else canon
        return WORD_RE.sub(repl, text)


@lru_cache(maxsize=1)
def default_drug_caser() -> DrugCaser:
    return DrugCaser.from_json()


# ---------------------------------------------------------------------------
# Accent variants
# ---------------------------------------------------------------------------

ACCENT_SPELLINGS_PATH = Path(__file__).resolve().parent / "accent_spellings.json"
# A word with its elision and inner hyphens (``l'olécrâne``, ``sterno-cléïdo-mastoïdien``).
_ACCENT_WORD_RE = re.compile(r"(?<![\w'’-])(?:[^\W\d_]+['’])?[^\W\d_]+(?:-[^\W\d_]+)*")


@lru_cache(maxsize=1)
def _accent_table() -> dict[str, str]:
    return json.loads(ACCENT_SPELLINGS_PATH.read_text(encoding="utf-8"))["rewrites"]


def normalize_accents(text: str, table: dict[str, str] | None = None) -> str:
    """One accent spelling per word, from ``accent_spellings.json``
    (``06_hotfixes/07_build_accent_spellings.py``, author's choices 2026-10-08):
    ``defect`` -> ``défect``, ``artéfact`` -> ``artefact``, ``l'olécrâne`` -> ``l'olécrane``,
    ``wirsüng`` -> ``wirsung``, drug INNs without accents (``périndopril`` ->
    ``perindopril``). A capital first letter is kept. Idempotent."""
    table = _accent_table() if table is None else table

    def repl(m: re.Match) -> str:
        w = m.group(0)
        canon = table.get(w.lower())
        if canon is not None:
            return _cap(canon) if w[0].isupper() else canon
        # Elided word missing as a whole (``l'énoxaparine``): look up the bare word.
        elision = re.match(r"[^\W\d_]+['’]", w)
        if elision is None or (canon := table.get(w[elision.end():].lower())) is None:
            return w
        word = w[elision.end():]
        return elision.group(0) + (_cap(canon) if word[0].isupper() else canon)
    return _ACCENT_WORD_RE.sub(repl, text)


def _n_accents(word: str) -> int:
    return sum(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", word))


@lru_cache(maxsize=1)
def _tts_accent_table() -> dict[str, str]:
    """``canonical -> accented spelling`` for the accent rewrites that REMOVE accents."""
    out: dict[str, str] = {}
    for form, canon in _accent_table().items():
        if _n_accents(form) > _n_accents(canon) and _n_accents(form) > _n_accents(out.get(canon, canon)):
            out[canon] = form
    return out


def restore_tts_accents(text: str, table: dict[str, str] | None = None) -> str:
    """Put back, for the TTS only, the accents ``normalize_accents`` removes from a label.

    The labels write drug INNs without accents (``perindopril``), but voxtral reads an
    unaccented ``e`` as a mute e ("peu-rindopril"), so the TTS text keeps the French
    spelling it pronounces right. The written label is not touched.
    """
    return normalize_accents(text, _tts_accent_table() if table is None else table)


# TTS-only respellings for words voxtral (voice fr_female) misreads, each one checked by
# ear (2026-10-09): ``naevus`` was read "naévouze", ``praevia`` "praévia", ``99m`` "99
# mètres", ``foetus`` "feutusse", ``caecum`` "ka-é-cum", ``macroadénome`` "macrodénome",
# ``chordome`` "corps-d'homme", ``foramens`` "fora-ment", ``eutrophe`` "utrophe",
# ``aphtes`` "aptes", ``canthus`` "kintus", ``bullage`` "boulage", ``HELLP`` spelled,
# ``IgA`` / ``IgE`` / ``IgM`` in English. Latin ``oe`` / ``ae`` read ``é`` (``œ`` did not help), ``-ome`` gets the closed ``ô`` where it
# was misread (not ``syndrome``, ``carcinome``: those were fine). A respelling is kept only
# when it sounds strictly better: on a tie the real spelling wins, as the TTS was trained on
# it (user rule 2026-10-13). The label keeps the written form.
_TTS_RESPELLINGS = (
    (re.compile(r"\b([Nn])aev"), r"\1év"),
    (re.compile(r"\b([Pp])raevia\b"), r"\1révia"),
    (re.compile(r"\b([Tt])aeni"), r"\1éni"),
    (re.compile(r"([Cc])aec"), r"\1éc"),
    (re.compile(r"\b([Aa])eruginosa\b"), r"\1éruginosa"),
    (re.compile(r"([Ff])oet"), r"\1ét"),
    (re.compile(r"([Cc])oeli"), r"\1éli"),
    # not the o of a prefix: ``phytoestrogènes`` is phyto + estrogènes
    (re.compile(r"(?<![Pp]hyt)(?<![Xx]én)(?<![Hh]yp)(?<![Mm]yc)(?<!o-)oestr"), "éstr"),
    (re.compile(r"\bOestr"), "Éstr"),
    (re.compile(r"(?<=[Ll]ymph)oed"), "éd"),
    (re.compile(r"\b([Mm]acro|[Ff]ibro)adénome"), r"\1-adénome"),
    (re.compile(r"\b([Cc]hord|[Cc]olob|[Ss]cot|[Ll]ymph|[Nn]eurin)ome"), r"\1ôme"),
    (re.compile(r"\b([Ff])oramens\b"), r"\1oramènes"),
    (re.compile(r"\b([Ee])utroph"), r"\1utrof"),
    (re.compile(r"\b([Aa])phte(s?)\b"), r"\1fte\2"),
    (re.compile(r"\b([Cc])anthus\b"), r"\1antus"),
    (re.compile(r"\b([Bb])ullage\b"), r"\1ulage"),
    (re.compile(r"\bNaegele\b"), "Négélé"),
    # the adjectives only: the eponym ``Descemet`` itself was fine
    (re.compile(r"([Dd])escemét"), r"\1éscémét"),
    (re.compile(r"\b([Ii])nfluenzae\b"), r"\1nfluenza-é"),
    (re.compile(r"\bE-C-B-C\b"), "Eu-C-B-C"),
    (re.compile(r"fungine\b"), "fongine"),
    (re.compile(r"\b([Ff])undique"), r"\1ondique"),
    (re.compile(r"\b([Aa])pneustique"), r"\1pneu-stique"),
    (re.compile(r"\bSever\b"), "Séver"),
    (re.compile(r"\b([Mm])icrotie"), r"\1icrossie"),
    (re.compile(r"\bHELLP\b"), "Help"),
    # read as English letters; ``IgG`` was fine
    (re.compile(r"\bIg([AEM])\b"), r"I-gé-\1"),
    (re.compile(r"\b99m\b"), "99-M"),
)


def respell_for_tts(text: str) -> str:
    """``naevus`` -> ``névus``, ``foetale`` -> ``fétale``, ``caecum`` -> ``cécum``, ``99m`` -> ``99-M``. Idempotent."""
    for pattern, repl in _TTS_RESPELLINGS:
        text = pattern.sub(repl, text)
    return text


def apply_label_conventions(text: str, tts_source: bool = False) -> str:
    """Every rewrite, titles first so ``M.`` is gone before sentence starts are judged,
    and dates before quantities so a spelled year is read as a year, not as a count.

    ``tts_source=True`` is for a text the TTS reads (05 on the v1 sources): it skips
    ``normalize_sutures`` and ``normalize_thousands``, which voxtral may not speak
    identically, keeps the accents of drug INNs (``restore_tts_accents``) and respells
    the words voxtral misreads (``respell_for_tts``).
    """
    text = normalize_roman(normalize_staging(normalize_quantities(normalize_clock(normalize_dates(expand_titles(text))))))
    if not tts_source:
        text = normalize_thousands(normalize_sutures(text))
    # Accents last: the drug caser restores the ANSM accents an INN choice removes.
    # Eponyms lowercased before the compound table, which keys on lowercase forms but
    # keeps their capitals (``Echo-Doppler`` is looked up as ``echo-doppler``).
    text = normalize_compound_spellings(lowercase_eponym_nouns(normalize_compounds(text)))
    text = normalize_latin_names(normalize_ordinals(normalize_non(normalize_official_names(strip_generic_labs(text)))))
    text = normalize_accents(default_drug_caser()(normalize_spelling(text)))
    return respell_for_tts(restore_tts_accents(text)) if tts_source else text
