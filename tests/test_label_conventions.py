"""Stdlib test for utils/label_conventions.py and its use by the text generators.

The cases are real UltiMed v1 label fragments: every rewrite must reach the corpus
majority form, leave the look-alikes alone (``PCR M. tuberculosis``, ``le monsieur du
lit 4``, ``les 4H et 4T``, ``mille neuf cent soixante-quatorze grammes``, ``Un an
plus tard``, ``deux trois jours``), and be idempotent. ``test_generator_parse`` checks that ``parse_asr_training_target`` (every
generator's parse step) applies them; it needs the LLM stack and is skipped when that
is not installed.

Run: python tests/test_label_conventions.py

This file was written by Claude Code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from percent_normalize import percent_to_symbol  # noqa: E402
from label_conventions import (  # noqa: E402
    DrugCaser, apply_label_conventions, default_drug_caser, expand_titles, normalize_clock,
    normalize_accents, normalize_compounds, normalize_dates, normalize_quantities, normalize_spelling, normalize_staging, normalize_roman, normalize_sutures, normalize_thousands, respell_for_tts, restore_tts_accents, strip_generic_labs, _tts_accent_table,
    parse_french_number, normalize_compound_spellings, normalize_official_names, normalize_non, normalize_ordinals,
    lowercase_eponym_nouns, normalize_latin_names, normalize_hgvs, normalize_numbers,
    normalize_elided_quantities, normalize_label_numbers,
)


def check(fn, cases) -> None:
    for raw, expected in cases:
        got = fn(raw)
        assert got == expected, (raw, got, expected)
        assert fn(got) == got, ("not idempotent", got, fn(got))


def test_titles() -> None:
    check(expand_titles, [
        ("Compte rendu de M. Lefèvre, 45 ans.", "Compte rendu de Monsieur Lefèvre, 45 ans."),
        ("Ordonnance pour M. et Mme Petit.", "Ordonnance pour Monsieur et Madame Petit."),
        ("La patiente, Mme Dubois, 24 ans, et Mlle Girard.",
         "La patiente, Madame Dubois, 24 ans, et Mademoiselle Girard."),
        ("Transmission de soins pour M. Le Corre, 34 ans.",
         "Transmission de soins pour Monsieur Le Corre, 34 ans."),
        ("Adressée par le Dr Moreau. Pr Girard a validé.",
         "Adressée par le docteur Moreau. Professeur Girard a validé."),
        ("Dr Martin a vu la patiente.", "Docteur Martin a vu la patiente."),
        # Casing of the spelled-out titles before a name, mid-sentence.
        ("Le dossier de monsieur Girard et de madame Colin.",
         "Le dossier de Monsieur Girard et de Madame Colin."),
        ("Suivi pour le Docteur Lefèvre, médecin traitant : Professeur Dubois.",
         "Suivi pour le docteur Lefèvre, médecin traitant : professeur Dubois."),
        ("Docteur Lefèvre est informé.", "Docteur Lefèvre est informé."),
    ])
    for untouched in [
        "La PCR M. tuberculosis est négative.",
        "Madame M. présente une odynophagie.",
        "Plaque de taille M. La plaque n'est pas déplacée.",
        "Cellules en phase M. On propose une chimiothérapie.",
        "Le monsieur du lit 4 est sorti.",
        "Stade T2 N0 M0, madame, vous allez bien.",
        "Prescription pour M. le docteur Martin.",
    ]:
        assert expand_titles(untouched) == untouched, (untouched, expand_titles(untouched))
    print("test_titles: OK")


def test_dates() -> None:
    check(normalize_dates, [
        ("Opérée le quinze mars deux mille vingt.", "Opérée le 15 mars 2020."),
        ("Née le premier janvier mille neuf cent quatre-vingt-deux.", "Née le 1er janvier 1982."),
        ("Le trente et un octobre deux-mille-vingt-et-un, puis",
         "Le 31 octobre 2021, puis"),
        ("Né le 3 mai mil neuf cent soixante et onze.", "Né le 3 mai 1971."),
        ("Le 3 mai dix-neuf cent quatre-vingts.", "Le 3 mai 1980."),
        ("Hospitalisé du six au treize mai deux mille vingt-cinq.",
         "Hospitalisé du 6 au 13 mai 2025."),
        ("Du premier au trois juin.", "Du 1er au 3 juin."),
        ("Fracture en deux mille onze. Opéré depuis deux mille dix-neuf,",
         "Fracture en 2011. Opéré depuis 2019,"),
        ("Suivi en deux mille dix et deux mille onze.", "Suivi en 2010 et deux mille onze."),
        ("Bilan du premier mai 2025.", "Bilan du 1er mai 2025."),
    ])
    for untouched in [
        "Une dose de mille neuf cent soixante-quatorze grammes.",
        "Il vit depuis deux mille ans dans la légende.",
        "Tiré en deux mille exemplaires.",
        "Il reviendra dans deux mois.",
        "Un mars pluvieux.",
    ]:
        assert normalize_dates(untouched) == untouched, (untouched, normalize_dates(untouched))
    print("test_dates: OK")


def test_clock() -> None:
    check(normalize_clock, [
        ("Arrivée aux urgences à 14h30 pour brûlure.", "Arrivée aux urgences à 14 heures 30 pour brûlure."),
        ("Injecté à 10 h 15, la douleur cède.", "Injecté à 10 heures 15, la douleur cède."),
        ("Plan rouge à 09h45, décès à 02h15.", "Plan rouge à 9 heures 45, décès à 2 heures 15."),
        ("Reprendre à 8 h 00 avec le reste, alimentation à 17h00.",
         "Reprendre à 8 heures avec le reste, alimentation à 17 heures."),
        ("Antalgiques à 20h, constantes à 14 h : stables.",
         "Antalgiques à 20 heures, constantes à 14 heures : stables."),
        ("Pause de 1h puis 14H30.", "Pause de 1 heure puis 14 heures 30."),
    ])
    for untouched in [
        "Selon l'algorithme des 4H et 4T.",
        "Blocs 1a à 1h, 2, 3a.",
        "Une perfusion de 0,5 h par jour.",
        "Glycémie à J6h de la greffe.",
    ]:
        assert normalize_clock(untouched) == untouched, (untouched, normalize_clock(untouched))
    print("test_clock: OK")


def test_parse_french_number() -> None:
    for words, want in [
        ("deux cent cinquante", 250), ("quatre vingt dix sept", 97), ("soixante et onze", 71),
        ("trois mille cinq cents", 3500), ("cent quatre vingts", 180), ("vingt et une", 21),
        ("dix sept", 17), ("mille", 1000), ("zéro", 0), ("septante cinq", 75),
        ("un million cinq cent mille", 1500000), ("deux millions trois mille", 2003000),
        ("trois milliards", 3000000000),
        # A multiplier smaller than the one before it, and a bare "million".
        ("mille million", None), ("million", None), ("deux mille mille", None),
        # Juxtaposed numbers are not one number: "deux trois jours" is 2 or 3 days.
        ("deux trois", None), ("vingt trente", None), ("cent cent", None),
        ("et deux", None), ("deux et", None), ("comprimé", None),
    ]:
        assert parse_french_number(words.split()) == want, (words, parse_french_number(words.split()))
    print("test_parse_french_number: OK")


def test_quantities() -> None:
    # Real drug_sentence references: the dose is a measure (digits), and so is the count
    # of tablets since 2026-11-01 (author's choice), "un" included.
    check(normalize_quantities, [
        ("On réduit la dose de silodosine à une gélule de quatre milligrammes par jour.",
         "On réduit la dose de silodosine à 1 gélule de 4 milligrammes par jour."),
        ("deux comprimés de deux cent cinquante microgrammes par jour",
         "2 comprimés de 250 microgrammes par jour"),
        ("une capsule de zéro virgule vingt-cinq microgrammes chaque matin",
         "une capsule de 0,25 microgrammes chaque matin"),
        ("un flacon de deux cent cinquante-sept virgule cinquante milligrammes",
         "1 flacon de 257,50 milligrammes"),
        ("zéro virgule zéro cinq milligramme", "0,05 milligramme"),
        ("une goutte toutes les deux heures pendant dix-huit mois",
         "1 goutte toutes les 2 heures pendant 18 mois"),
        ("Soixante-quinze milligrammes et quatre-vingt-dix-sept ans, trois mille cinq cents unités.",
         "75 milligrammes et 97 ans, 3500 unités."),
        ("vingt et une heures", "21 heures"),
        # A ratio converts both halves, never "5 jours sur sept".
        ("cinq jours sur sept, vingt-quatre heures sur vingt-quatre, trois semaines sur quatre",
         "5 jours sur 7, 24 heures sur 24, 3 semaines sur 4"),
        # Counts of times / sessions, range heads and scores: digits (author's choice).
        ("deux comprimés trois fois par jour, dix séances", "2 comprimés 3 fois par jour, 10 séances"),
        ("pendant deux à trois jours, un à 2 fois, et deux à 4 semaines",
         "pendant 2 à 3 jours, 1 à 2 fois, et 2 à 4 semaines"),
        ("douleur à deux sur dix, EVA zéro sur dix", "douleur à 2 sur 10, EVA 0 sur 10"),
        # Eponym units written with their capital (2026-11-02, 8 release labels).
        ("IRM trois Tesla, en 5 fois cinq Gray, deux Hertz", "IRM 3 Tesla, en 5 fois 5 Gray, 2 Hertz"),
        # A score after "et" (2026-11-02): the "et" belongs to the sentence.
        ("5 sur 10 au repos et huit sur dix à la marche", "5 sur 10 au repos et 8 sur 10 à la marche"),
        # A percentage before a count is not one number.
        ("0,10 gramme pour cent une fois par jour", "0,10 gramme pour cent une fois par jour"),
        # A single unit or duration: digits (author's choice 2026-10-08).
        ("Contrôle à un mois, puis dans une semaine. Un gramme de paracétamol, un an après.",
         "Contrôle à 1 mois, puis dans 1 semaine. 1 gramme de paracétamol, un an après."),
        ("correction de deux dioptries", "correction de 2 dioptries"),
    ])
    for untouched in [
        "une fois par jour",
        # Counts, and words rarely a measure after "un"/"une".
        "une séance, une unité de soins, en seconde intention, un degré de gravité",
        # An elided "un" stays a word.
        "au bout d'un mois, l'effet d'une semaine",
        "pendant deux trois jours",
        "Une dose de 4 milligrammes.",
    ]:
        assert normalize_quantities(untouched) == untouched, (untouched, normalize_quantities(untouched))
    print("test_quantities: OK")


def test_dosage_forms() -> None:
    # 2026-11-01 (author's choice): a dosage form takes digits for every value, 1 included.
    check(normalize_quantities, [
        ("Prendre un comprimé le soir.", "Prendre 1 comprimé le soir."),
        ("deux comprimés matin et soir", "2 comprimés matin et soir"),
        ("un comprimé et demi au coucher", "1 comprimé et demi au coucher"),
        ("une gélule, un sachet, deux bouffées, une cuillère-mesure",
         "1 gélule, 1 sachet, 2 bouffées, 1 cuillère-mesure"),
        ("trois gouttes dans chaque oeil, un suppositoire", "3 gouttes dans chaque oeil, 1 suppositoire"),
        # From two up, the article-heavy forms convert too.
        ("deux capsules, trois ampoules, deux doses, deux injections, deux patchs",
         "2 capsules, 3 ampoules, 2 doses, 2 injections, 2 patchs"),
    ])
    for untouched in [
        # A "demi" stays spelled: 310 "un demi" against 1 "1 demi" in the labels.
        "un demi-comprimé le soir",
        "un demi comprimé le soir",
        # A single "un"/"une" before these is mostly an article.
        "une capsule de Glisson, une ampoule galactophore, une dose d'irradiation",
        "admis pour une injection sous-conjonctivale, un patch veineux",
    ]:
        assert normalize_quantities(untouched) == untouched, (untouched, normalize_quantities(untouched))
    print("test_dosage_forms: OK")


def test_single_duration_context() -> None:
    # 2026-11-01: a lone "un"/"une" before a duration converts only in a quantity context.
    check(normalize_quantities, [
        ("Revoir dans un mois, contrôle à une semaine.", "Revoir dans 1 mois, contrôle à 1 semaine."),
        ("Traité pendant un an, depuis une semaine, il y a un an.",
         "Traité pendant 1 an, depuis 1 semaine, il y a 1 an."),
        ("Bilan a un mois, en une heure, après une minute, avant un jour.",
         "Bilan a 1 mois, en 1 heure, après 1 minute, avant 1 jour."),
        ("une durée de un an, sur un mois, pour une semaine, environ une heure",
         "une durée de 1 an, sur 1 mois, pour 1 semaine, environ 1 heure"),
        ("durant un mois, sous une semaine, par une heure, toutes les une heure, tous les un an",
         "durant 1 mois, sous 1 semaine, par 1 heure, toutes les 1 heure, tous les 1 an"),
        ("plus de un an, moins de une heure", "plus de 1 an, moins de 1 heure"),
        # "et" after a number, whether that number was already digits or not.
        ("2 ans et un mois", "2 ans et 1 mois"),
        ("deux ans et un mois", "2 ans et 1 mois"),
        ("à un an et un mois", "à 1 an et 1 mois"),
        # A ratio is a quantity wherever it stands.
        ("Un jour sur deux.", "1 jour sur 2."),
        # Values from two up and other units keep converting anywhere.
        ("Deux heures plus tard, un gramme.", "2 heures plus tard, 1 gramme."),
        ("deux années", "2 années"),
        # "pour" before any number but "cent".
        ("pour deux mois", "pour 2 mois"),
    ])
    for untouched in [
        "Un an plus tard, il revient.",
        "Et un jour, le fiancé est parti.",
        "un jour ou l'autre",
        "Une semaine après, la fièvre cède.",
        "C'est un an de suivi.",
        "un an et un mois",
        "0,10 gramme pour cent une fois par jour",
        # 2026-11-02 (author's choice): a lone "un"/"une" before "année(s)" never converts,
        # whatever precedes it ("une année olympique" is no duration).
        "pendant une année olympique, dans une année, 2 ans et une année",
        "une année sur deux",
    ]:
        assert normalize_quantities(untouched) == untouched, (untouched, normalize_quantities(untouched))
    check(normalize_quantities, [("cinquante années, pendant deux années", "50 années, pendant 2 années")])
    print("test_single_duration_context: OK")


def test_compounds() -> None:
    check(normalize_compounds, [
        ("À prendre au petit déjeuner.", "À prendre au petit-déjeuner."),
        ("Les petits déjeuners. Petit déjeuner léger.", "Les petits-déjeuners. Petit-déjeuner léger."),
        ("Voies biliaires extra hépatiques, rein multi lithiasique, Intra abdominal.",
         "Voies biliaires extrahépatiques, rein multilithiasique, Intra-abdominal."),
        ("Lésions intra et extra-hépatiques, supra ou infra centimétriques.",
         "Lésions intra et extrahépatiques, supra ou infracentimétriques."),
        # Glued (author's choice 2026-10-08), every part of a chain, case kept.
        ("J2 post-opératoire. Post opératoire, pré-opératoire, oeso-gastro-duodénale.",
         "J2 postopératoire. Postopératoire, préopératoire, oesogastroduodénale."),
        ("Douleur abdomino-pelvienne, déficit sensitivo moteur, l'abdomino-pelvien, ré-évaluation.",
         "Douleur abdominopelvienne, déficit sensitivomoteur, l'abdominopelvien, réévaluation."),
        # A misread digraph or a doubled a / i / o / u keeps the hyphen.
        ("gastro-entérologie, intra oculaire, intra-utérin, auto-immune, anti-inflammatoire, radio-opaque",
         "gastro-entérologie, intraoculaire, intra-utérin, auto-immune, anti-inflammatoire, radio-opaque"),
        ("thoraco abdomino pelvienne, post traumatique, pré-éclampsie, cortico-sous-corticale", "thoracoabdominopelvienne, post-traumatique, prééclampsie, cortico-sous-corticale"),
        # post / pré / per before a noun keep a space (author's choice 2026-10-08), not "postopération".
        ("J5 post opération, post-chimiothérapie, pré-greffe, Post-traitement, per procédure, post-cure, pré-remplie, pré morbide",
         "J5 post opération, post chimiothérapie, pré greffe, Post traitement, per procédure, postcure, préremplie, prémorbide"),
    ])
    assert normalize_compounds("un petit déjeune") == "un petit déjeune"
    # Latin, excluded prefixes, nouns, names, uppercase or short second parts.
    for kept in ("post-partum, per os, sous-cutané, sus-pubien, semi-circulaire, contre-indication",
                 "l'écho-doppler, bêta-lactamines, vidéo-assistée",
                 "une radio thoracique, chimio adjuvante, une gastro sévère",
                 "Marco-Polo, Jean-Pierre, pro-BNP, anti-Xa, apo-E, L4-L5, check-list, c'est-à-dire, peut-être"):
        assert normalize_compounds(kept) == kept, (kept, normalize_compounds(kept))
    print("test_compounds: OK")


def test_sutures_and_decimals() -> None:
    check(normalize_sutures, [
        ("Fermeture au Vicryl trois zéro.", "Fermeture au Vicryl 3-0."),
        ("Surjet au Monocryl 4 zéros, PDS rapide 3/0.", "Surjet au Monocryl 4-0, PDS rapide 3-0."),
        ("Points à l'Ethilon neuf zéro.", "Points à l'Ethilon 9-0."),
    ])
    # Not a gauge: no suture material before it, or a plain "Vicryl zéro".
    for kept in ("schéma zéro un zéro", "surjet de Vicryl zéro", "Monocryl 4-0."):
        assert normalize_sutures(kept) == kept, kept
    check(normalize_quantities, [
        ("un transducteur de 3 virgule 5 mégahertz", "un transducteur de 3,5 mégahertz"),
        ("cent vingt milligrays centimètres", "120 milligrays centimètres"),
        ("une IRM trois teslas", "une IRM 3 teslas"),
    ])
    # The TTS source keeps what voxtral said.
    assert apply_label_conventions("Vicryl trois zéro", tts_source=True) == "Vicryl trois zéro"
    # The misspelt brand gets its gauge in the same pass as its spelling: was "Vicryl
    # trois zéro" after one pass, "Vicryl 3-0" after two (2026-11-15).
    assert apply_label_conventions("Fermeture au vycril trois zéro.") == "Fermeture au Vicryl 3-0."
    print("test_sutures_and_decimals: OK")


def test_generic_labs() -> None:
    check(strip_generic_labs, [
        ("Sous cholecalciferol Viatris 2,5 milligrammes et perindopril Arrow Génériques 4.", "Sous cholecalciferol 2,5 milligrammes et perindopril 4."),
        ("Ténofovir Mylan : charge virale indétectable.", "Ténofovir : charge virale indétectable."),
        # people named like a lab, and products named after one
        ("Monsieur Teva, l'enfant Mylan Martin, le jeune Mylan, pour Mylan Millet.",) * 2,
        ("Phosphore Sandoz 500, Aspirine Upsa, Accord pour la chirurgie.",) * 2,
    ])



def test_tts_source_pipeline_idempotent() -> None:
    """A respelled TTS source must survive a second full pass: 05 is re-run over sources it
    already rewrote. ``de type I-gé-M`` became ``de type 1-gé-M`` (staging numeral) and
    ``xéno-éstrogènes`` lost its hyphen (2026-10-13 SSD run)."""
    tts = lambda t: apply_label_conventions(t, tts_source=True)
    for text, want in [
        ("Anticorps de type IgM, classe IgA, grade IgE.", "Anticorps de type I-gé-M, classe I-gé-A, grade I-gé-E."),
        # the label's oe-less spelling (2026-11-15) is label-only: the TTS source keeps it
        ("Phyto-oestrogènes et xéno-oestrogènes.",) * 2,
        ("Stade I-II, type II.", "Stade 1-2, type 2."),
    ]:
        once = tts(text)
        assert once == want, (text, once)
        assert tts(once) == once, ("not idempotent", once, tts(once))
    print("test_tts_source_pipeline_idempotent: OK")

def test_thousands() -> None:
    check(normalize_thousands, [
        ("Plaquettes à 1 200 000 par millimètre cube, 1 sur 10 000.", "Plaquettes à 1200000 par millimètre cube, 1 sur 10000."),
        ("Garçon de 3 450 grammes, 1 234,5 euros.", "Garçon de 3450 grammes, 1234,5 euros."),
        # two numbers, a name with digits, a decimal part
        ("Tension 12 18, vitamine D3 100 000 unités, coenzyme Q10 300, 0,5 000.",
         "Tension 12 18, vitamine D3 100000 unités, coenzyme Q10 300, 0,5 000."),
        # a code ending in a digit, then a number
        ("Oméga-3 500 milligrammes, CA 15-3 272 unités.",) * 2,
    ])
    # idempotent once the spelled number is converted (it ran before normalize_numbers)
    once = apply_label_conventions("CA 19-9 cent trente-quatre versus 90.")
    assert once == "CA 19-9 134 versus 90." == apply_label_conventions(once), once
    assert apply_label_conventions("Soit cent trente-quatre 500 unités.") == "Soit 134500 unités."
    # the TTS source keeps the spaced form
    assert apply_label_conventions("18 000 plaquettes.", tts_source=True) == "18 000 plaquettes."
    tts = "Naevus et naevomatose, placenta praevia, technétium-99m."
    assert apply_label_conventions(tts, tts_source=True) == "Névus et névomatose, placenta prévia, technétium-99-M."
    assert apply_label_conventions(tts) == tts
    check(respell_for_tts, [
        ("Souffrance foetale, Oestradiol et oestrogènes, maladie coeliaque, lymphoedème.", "Souffrance fétale, Éstradiol et éstrogènes, maladie céliaque, lymphédème."),
        ("Caecum, iléocaecale, Taenia, aeruginosa.", "Cécum, iléocécale, Ténia, aéruginosa."),
        ("Macroadénome, fibroadénomes, chordome, colobomes, scotome, foramens, eutrophe.", "Macro-adénome, fibro-adénomes, chordôme, colobômes, scotôme, foramènes, eutrofe."),
        # what the ear kept as-is
        ("Oedème, oesophage, oeil, hypoesthésie, foramen, microadénome, athérome.",) * 2,
        ("Phytoestrogènes, xénoestrogènes, hypoestrogénie.",) * 2,
        ("Aphtes, un aphte, canthus, bullage, règle de Naegele, prédescemétique.", "Aftes, un afte, cantus, bulage, règle de Négélé, prédéscémétique."),
        ("Aphtose, épicanthus, Naegeli, membrane de Descemet, Henle, Dix-Hallpike.",) * 2,
        ("Eutrophique, lymphome, neurinomes, descemétite, Haemophilus influenzae, un E-C-B-C.", "Eutrofique, lymphôme, neurinômes, déscémétite, Haemophilus influenzae".replace("influenzae", "influenza-é") + ", un Eu-C-B-C."),
        ("Micafungine, muqueuse fundique, centre apneustique, maladie de Sever, microtie, HELLP syndrome.", "Micafongine, muqueuse fondique, centre apneu-stique, maladie de Séver, microssie, Help syndrome."),
        ("IgA, IgE, IgM, IgG, IgA1, IgG4.", "I-gé-A, I-gé-E, I-gé-M, IgG, IgA1, IgG4."),
        ("Phyto-oestrogènes, xéno-oestrogènes.",) * 2,
        # ties go to the real spelling
        ("Syndrome, carcinome, hématome, glaucome, eutocique, aphteuse, fundus, sacrum, sever.",) * 2,
    ])
    print("test_thousands: OK")


def test_spelling() -> None:
    check(normalize_spelling, [
        ("Le cœur, l'Œdème, une manœuvre.", "Le coeur, l'Oedème, une manoeuvre."),
        ("Pancréatite aigüe, lésions subaigües, ambigüité.", "Pancréatite aiguë, lésions subaiguës, ambiguïté."),
        ("Compte-rendu opératoire, les comptes-rendus.", "Compte rendu opératoire, les comptes rendus."),
        ("Bêta-bloquants, bêtabloquant, bêta bloqueur.", "Bétabloquants, bétabloquant, bétabloqueur."),
        ("Anévrysme, urèthre, sténose uréthrale, périuréthral.", "Anévrisme, urètre, sténose urétrale, périurétral."),
        ("A surveiller. A jeun. A l'examen, A renouveler.", "À surveiller. À jeun. À l'examen, A renouveler."),
        ("Douleur aigue, pancréatite suraigues, un sénior, sterno-cléïdo, la tete, les genès.",
         "Douleur aiguë, pancréatite suraiguës, un senior, sterno-cléido, la tête, les gènes."),
        # téter: the bébé tète, kept.
        ("Le bébé tète bien.", "Le bébé tète bien."),
        # The verb avoir, mid-sentence, ambiguous: unchanged.
        ("A présenté une fièvre. A bien toléré. A encore mal. Il A voir.",
         "A présenté une fièvre. A bien toléré. A encore mal. Il A voir."),
    ])
    assert normalize_spelling("Docteur Argüelles") == "Docteur Argüelles"
    check(normalize_spelling, [
        ("Mutation du gene JAK2, colon droit, semi-colon. Docteur Genes.", "Mutation du gène JAK2, côlon droit, semi-colon. Docteur Genes."),
        ("Scannographique, shampoing, 30 paquets années, 1 paquet-années.", "Scanographique, shampooing, 30 paquets-année, 1 paquet-année."),
        ("50 gray, 2,5 gray, 1,8 gray, 1 gray.", "50 grays, 2,5 grays, 1,8 gray, 1 gray."),
        ("Elodie, Élodie et Mélodie.", "Hélodie, Élodie et Mélodie."),
        ("Du Ketoprofen, sous cisplatin, par Céfépim, le CEFEPIM, aspirin, ASPIRIN Protect, la Fucidin, BisoC, myopathie statin-induite, une penicillin binding protein.",
         "Du Kétoprofène, sous cisplatine, par Céfépime, le céfépime, aspirine, Aspirine Protect, la Fucidine, Bisoce, myopathie statino-induite, une penicillin binding protein."),
        ("Nævus du cæcum, placenta prævia, Æ.", "Naevus du caecum, placenta praevia, Ae."),
        # oestr- lost to estr- in the labels (2026-11-15), prefix hyphen and all; the
        # parasite, the cycle phase and the Oestrogel brand keep theirs.
        ("Oestradiol, oestrogènes, phyto-oestrogènes, xéno-oestrogènes, hypo-oestrogénie, éthinyl-oestradiol.",
         "Estradiol, estrogènes, phytoestrogènes, xénoestrogènes, hypoestrogénie, éthinylestradiol."),
        ("Diéthyl-stilboestrol, hexoestrol, hyperoestrogénie, bêta-oestradiol, traitement oestro progestatif.",
         "Diéthylstilbestrol, hexestrol, hyperestrogénie, bêta-estradiol, traitement estro progestatif."),
        ("Oestrus ovis, anoestrus, métaoestrus, cycle oestral, gel d'Oestrogel.",) * 2,
    ])
    once = normalize_spelling("Phyto-oestrogènes et Oestradiol.")
    assert normalize_spelling(once) == once, once
    print("test_spelling: OK")


def test_accents() -> None:
    table = {"defect": "défect", "l'olécrâne": "l'olécrane", "périndopril": "perindopril",
             "sterno-cléïdo-mastoïdien": "sterno-cléido-mastoïdien"}
    check(lambda t: normalize_accents(t, table), [
        ("Sans defect, sommet de l'olécrâne. Defect.", "Sans défect, sommet de l'olécrane. Défect."),
        ("Périndopril 5 mg, muscle sterno-cléïdo-mastoïdien.", "Perindopril 5 mg, muscle sterno-cléido-mastoïdien."),
        # Elided: the table holds the bare word only.
        ("Dose d'périndopril, l'Périndopril.", "Dose d'perindopril, l'Perindopril."),
    ])
    # The TTS text keeps the accents a label loses, never the reverse.
    tts = {"perindopril": "périndopril"}
    check(lambda t: restore_tts_accents(t, tts), [
        ("Perindopril puis l'perindopril, défect.", "Périndopril puis l'périndopril, défect."),
    ])
    # A typo entry is no accent variant: it must not be inverted into the TTS text, or
    # ``nefopam`` would be respelled ``néphopam`` (2026-11-15).
    assert _tts_accent_table()["nefopam"] == "néfopam", "a typo leaked into the TTS respellings"
    assert restore_tts_accents("Nefopam et perindopril.") == "Néfopam et périndopril."
    # Part of a longer word or a hyphenated compound: untouched.
    assert normalize_accents("antidefect, defect-like", table) == "antidefect, defect-like"
    # All caps (PARHAF drug lists) takes the table's casing, capitalised at a sentence start
    # only: was ``Il prend aussi Perindopril`` (2026-11-15).
    check(lambda t: normalize_accents(t, {"perindipril": "perindopril", "affinitor": "Afinitor"}), [
        ("Il prend aussi PERINDIPRIL 4 mg. PERINDIPRIL le soir, AFFINITOR.",
         "Il prend aussi perindopril 4 mg. Perindopril le soir, Afinitor."),
    ])
    # The committed table, after the drug caser that would put the ANSM accents back.
    for raw, want in [("PERINDOPRIL et énoxaparine, un artéfact, le canal de Wirsüng.",
                       "Perindopril et enoxaparine, un artefact, le canal de Wirsung."),
                      ("paracétamol", "paracétamol"),
                      # An enzyme, not an INN: keeps its French accent.
                      ("inhibiteur de protease", "inhibiteur de protéase"),
                      # Hand rewrites (2026-11-15): typos, source-sheet misspellings, the
                      # spelling winners, INN leftovers; longer words and other words untouched.
                      ("Sous AZYTHROMYCINE et TAZOCILINE, puis acyclovir et valaciclovir, kyste à Cyclospora.",
                       "Sous azithromycine et Tazocilline, puis aciclovir et valaciclovir, kyste à Cyclospora."),
                      ("La ciclosporine, l'ertapénème, du lévétiracétam, un staphylocoque méticilline-sensible.",
                       "La cyclosporine, l'ertapenem, du levetiracetam, un staphylocoque meticilline-sensible."),
                      ("Vaccin polyomielite, nalbufine, glycopirronium, amitryptiline.",
                       "Vaccin poliomyélite, nalbuphine, glycopyrronium, amitriptyline."),
                      ("Aktiskenan 10 mg, Gentamycine, céphalexine, paracétamol, méthotrexate.",
                       "Actiskenan 10 mg, Gentamicine, cefalexine, paracétamol, méthotrexate.")]:
        assert apply_label_conventions(raw) == want, (raw, apply_label_conventions(raw))
    print("test_accents: OK")


def test_cefalexine_follows_inn_convention() -> None:
    """No accent exception any more (was ``céfalexine``, 2026-11-15); every spelling
    in the labels folds to it, and the TTS still hears the accent on a fresh generation."""
    for text, want in [("Céfalexine monohydratée 500 mg.", "Cefalexine monohydratée 500 mg."),
                       ("La céfalexine et la céphalexine.", "La cefalexine et la cefalexine."),
                       ("CÉFALEXINE ce jour, Cephalexine hier.", "Cefalexine ce jour, Cefalexine hier."),
                       ("de la cefalexine.", "de la cefalexine.")]:
        assert apply_label_conventions(text) == want, (text, apply_label_conventions(text))
    assert restore_tts_accents("la cefalexine") == "la céfalexine"
    print("test_cefalexine_follows_inn_convention: OK")


def test_accent_rewrite_elides() -> None:
    """A rewrite that makes a word vowel-initial elides the particle before it: ``Le
    cestradiol`` -> ``L'estradiol``, ``du`` -> ``de l'``, ``ce`` -> ``cet`` (2026-11-16). A
    word that was already vowel-initial, or unchanged, keeps its particle as written."""
    check(lambda t: normalize_accents(t, {"cestradiol": "estradiol", "perindipril": "perindopril"}), [
        ("Le cestradiol, le CESTRADIOL et LE CESTRADIOL.", "L'estradiol, l'estradiol et L'estradiol."),
        ("De cestradiol, de la cestradiol, que le cestradiol.", "D'estradiol, de l'estradiol, que l'estradiol."),
        ("Du cestradiol, au cestradiol, ce cestradiol, qu'on dit cestradiol.",
         "De l'estradiol, à l'estradiol, cet estradiol, qu'on dit estradiol."),
        ("Le perindipril, le estradiol, l'cestradiol.", "Le perindopril, le estradiol, l'estradiol."),
    ])
    print("test_accent_rewrite_elides: OK")


def test_owner_review_rewrites() -> None:
    """The owner's hand review of leftover variants (2026-11-15): brands keep the drug
    caser's capitals, all-caps follows the table casing, ``mésalamine`` keeps the accent
    it gains (the INN accent stripping must not undo it), two words stay as they are."""
    for text, want in [("Sous Patidexaméthasone.", "Sous Patidexamethasone."),
                       ("Il prend du TYMOFEROL.", "Il prend du tymoferol."),
                       ("Le thiorphan.", "Le Tiorfan."),
                       ("L'acide cromoglycique.", "L'acide cromoglicique."),
                       ("De l'oxyconazole, d'oxyconazole, Oxyconazole.",
                        "De l'oxiconazole, d'oxiconazole, Oxiconazole."),
                       ("La mesalamine et la mésalamine.", "La mésalamine et la mésalamine."),
                       ("La gonadotropine et l'homatropine.", "La gonadotropine et l'homatropine.")]:
        got = apply_label_conventions(text)
        assert got == want, (text, got)
        assert apply_label_conventions(got) == got, got
    print("test_owner_review_rewrites: OK")


def test_spelling_fixes_never_reach_tts_source() -> None:
    """Spelling fixes are LABEL-only: the TTS source is the text the audio was read from,
    so a typo, accent or spelling-winner fix there would make it lie about the audio.
    Was ``La cyclosporine, du lévétiracétam, périndopril.`` and ``Phytoestrogènes``
    (2026-11-15, the owner's rule: only the label changes)."""
    tts = lambda t: apply_label_conventions(t, tts_source=True)
    for text in ["La ciclosporine, du lévétiracétam, betaméthasone, céphalexine, acyclovir.",
                 "Phyto-oestrogènes et xéno-oestrogènes.",
                 "Sous Patidexaméthasone, cestradiol, thiorphan et cromoglycique."]:
        assert tts(text) == text, (text, tts(text))
    # Every word of the accent table, in a neutral sentence: the TTS source keeps its
    # letters, up to what ran on the source before these fixes existed and is spoken
    # identically (drug casing, TTS respellings, compound hyphen joins).
    table = __import__("label_conventions")._accent_table()
    letters = lambda t: t.replace("-", "").replace(" ", "").lower()
    changed = {}
    for form in table:
        text = f"Il dit {form} ce matin."
        if letters(tts(text)) != letters(respell_for_tts(default_drug_caser()(text))):
            changed[form] = tts(text)
    assert not changed, ("a label spelling fix reached the TTS source", changed)
    print("test_spelling_fixes_never_reach_tts_source: OK")


def test_drug_caser() -> None:
    caser = DrugCaser(
        caps={"PRIMPERAN": "Primpéran", "PARACETAMOL": "paracétamol", "UVEDOSE": "Uvedose",
              "FORTE": "forte"},
        variants={"primperan": "Primpéran", "skénan": "Skenan", "paracetamol": "paracétamol"},
    )
    check(caser, [
        ("PRIMPERAN et Primperan puis primperan.", "Primpéran et Primpéran puis Primpéran."),
        ("Prendre PARACETAMOL 1 g, puis paracetamol. PARACETAMOL encore.",
         "Prendre paracétamol 1 g, puis paracétamol. Paracétamol encore."),
        ("UVEDOSE une ampoule, l'UVEDOSE FORTE aussi.", "Uvedose une ampoule, l'Uvedose forte aussi."),
        ("Du Skénan.", "Du Skenan."),
    ])
    # Canonical forms of the committed lexicon, and a protected acronym.
    default = default_drug_caser()
    assert default("PARACETAMOL LP et KARDEGIC.") == "Paracétamol LP et Kardégic.", default("PARACETAMOL LP et KARDEGIC.")
    assert default("un comprimé SANS SUCRE") == "un comprimé SANS sucre", default("un comprimé SANS SUCRE")
    assert default("paracétamol codéiné") == "paracétamol codéiné"
    assert default("une note de menthe") == "une note de menthe"
    print("test_drug_caser: OK")


def test_staging() -> None:
    check(normalize_staging, [
        ("Cancer de stade IIIb, grade I à II, type I et II.", "Cancer de stade 3b, grade 1 à 2, type 1 et 2."),
        ("Antalgique de palier deux, classe III NYHA, NYHA II.", "Antalgique de palier 2, classe 3 NYHA, NYHA 2."),
        ("Diabète de type 2, stades IVB.", "Diabète de type 2, stades 4B."),
        ("Essai de phase III, phases I et II.", "Essai de phase 3, phases 1 et 2."),
    ])
    # Roman numerals that are names, letters that happen to be Roman, "un peu".
    for kept in ("angiotensine II", "APACHE II", "métaphase II", "type C", "type XX", "un type un peu particulier"):
        assert normalize_staging(kept) == kept, kept
    print("test_staging: OK")


def test_roman() -> None:
    check(normalize_roman, [
        ("Atteinte des nerfs III, IV et VI.", "Atteinte des nerfs 3, 4 et 6."),
        ("Mutation du facteur V Leiden, facteur VIIIc.", "Mutation du facteur 5 Leiden, facteur 8c."),
        ("Fracture Garden IV, Le Fort I, Mobitz I.", "Fracture Garden 4, Le Fort 1, Mobitz 1."),
        ("De l'angiotensine II, APACHE II, GP IIb IIIa.", "De l'angiotensine 2, APACHE 2, GP 2b 3a."),
        ("Stades I, II ou III.", "Stades 1, 2 ou 3."),
    ])
    # letters, intravenous, kings, activated factor, ECG leads
    for kept in ("rayons X", "chromosome X", "pénicilline V", "troponine I", "voriconazole IV",
                 "Rifadine IV pendant", "Henri IV", "Louis XIII", "facteur Xa", "anti-Xa", "D-II"):
        assert normalize_roman(kept) == kept, kept
    print("test_roman: OK")


def test_compound_spellings() -> None:
    table = {"gastroentérologie": "gastro-entérologie", "tep scanner": "TEP-scanner",
             "transsphénoïdale": "trans-sphénoïdale", "sous costale": "sous-costale",
             "écho doppler": "écho-doppler", "pro bnp": "proBNP", "adn-polymérase": "adn polymérase",
             "moya-moya": "moyamoya", "anti-rhésus": "antirhésus", "angioirm": "angio-irm"}
    check(lambda t: normalize_compound_spellings(t, table), [
        ("Gastroentérologie. Un TEP scanner, un Tep scanner, voie transsphénoïdale.",
         "Gastro-entérologie. Un TEP-scanner, un TEP-scanner, voie trans-sphénoïdale."),
        ("Voie sous costale, l'écho doppler, d'écho doppler, l'ADN-polymérase.",
         "Voie sous-costale, l'écho-doppler, d'écho-doppler, l'ADN polymérase."),
        # Part of a longer hyphen chain or a name with digits: untouched.
        ("un super-tep scanner, le pro bnp2", "un super-tep scanner, le pro bnp2"),
        ("NT pro BNP, Pro BNP", "NT proBNP, ProBNP"),
        # A capital that only started a glued part is dropped, an acronym's is kept.
        ("Moya-Moya, anti-Rhésus, AngioIRM", "Moyamoya, antirhésus, Angio-IRM"),
    ])
    # The shipped table: idempotent through the whole pipeline.
    for canon in ("gastro-entérologie", "trans-sphénoïdale", "sous-costale", "check-list", "post-partum"):
        assert apply_label_conventions(canon) == canon, (canon, apply_label_conventions(canon))
    # Unaccented "Echo" is not a combining form either; overlapping families settle in one run.
    for once in ("Echo-Doppler artériel.", "Le test de Cochran-Mantel-Haenszel montre"):
        twice = apply_label_conventions(apply_label_conventions(once))
        assert twice == apply_label_conventions(once), (once, apply_label_conventions(once), twice)
    assert apply_label_conventions("Echo-Doppler artériel.") == "Echo-doppler artériel."
    assert apply_label_conventions("Une gastroentérologie, un postpartum, une checklist.") == \
        "Une gastro-entérologie, un post-partum, une check-list."
    print("test_compound_spellings: OK")


def test_official_names() -> None:
    check(normalize_official_names, [
        ("NT pro BNP, NTproBNP, NT-ProBNP, LHRH, anti-SS-A, anti SSB", "NT-proBNP, NT-proBNP, NT-proBNP, LH-RH, anti-SSA, anti-SSB"),
        ("HLA B27, PDL1, PD-L 1, PD1, PI3K, PIK3CA", "HLA-B27, PD-L1, PD-L1, PD-1, PI3K, PIK3CA"),
    ])
    check(normalize_official_names, [
        ("Hernie L4 L5 et L5 S1, L3 L4 L5.", "Hernie L4-L5 et L5-S1, L3-L4-L5."),
        ("D dimères, d-dimères, TEP 18 FDG, 18F FDG, Ki67, Ki 67.", "D-dimères, D-dimères, TEP 18-FDG, 18F-FDG, Ki-67, Ki-67."),
        ("ASA-2, TCD4, VIH 1, 46XY, GCSF, GLP1, CA-125, CA19-9, CA 15 3.",
         "ASA 2, T CD4, VIH-1, 46 XY, G-CSF, GLP-1, CA 125, CA 19-9, CA 15-3."),
        ("Séquences T1 T2, C3 C4 bas.", "Séquences T1 T2, C3 C4 bas."),
    ])
    print("test_official_names: OK")


def test_non() -> None:
    check(normalize_non, [
        ("Un non fumeur, le non respect, des non-répondeurs, un taux de non réponse.",
         "Un non-fumeur, le non-respect, des non-répondeurs, un taux de non-réponse."),
        ("Patiente non-fumeuse, lymphome non-hodgkinien, non significatif, non seulement.",
         "Patiente non fumeuse, lymphome non hodgkinien, non significatif, non seulement."),
        ("Mutation non sens, en non appui, critère non progression, non totalement résolu.",
         "Mutation non-sens, en non-appui, critère non-progression, non totalement résolu."),
    ])
    print("test_non: OK")


def test_ordinals() -> None:
    check(normalize_ordinals, [
        ("Au 3e percentile, du 2e au 5e jour, la 1re côte, 1ère fois.",
         "Au troisième percentile, du deuxième au cinquième jour, la première côte, première fois."),
        ("1er trimestre. Le 1er mai, du 1er au 5 juin, 10e jour, 12ème semaine, 15e.",
         "Premier trimestre. Le 1er mai, du 1er au 5 juin, dixième jour, 12e semaine, 15e."),
        ("2,5e, C2e, 4ème et 8èmes", "2,5e, C2e, quatrième et huitièmes"),
    ])
    print("test_ordinals: OK")


def test_eponym_nouns() -> None:
    check(lowercase_eponym_nouns, [
        ("Un écho-Doppler, un Holter ECG, l'effet Doppler. Doppler normal.",
         "Un écho-doppler, un holter ECG, l'effet Doppler. Doppler normal."),
    ])
    print("test_eponym_nouns: OK")


def test_latin_names() -> None:
    check(normalize_latin_names, [
        ("PCR herpes simplex, un herpes labial, virus Herpès simplex, Herpes zoster.",
         "PCR Herpes simplex, un herpès labial, virus Herpes simplex, Herpes zoster."),
        ("Streptococcus pyogènes, germes pyogenes, larves strongyloides, Strongyloïdes stercoralis.",
         "Streptococcus pyogenes, germes pyogènes, larves strongyloïdes, Strongyloides stercoralis."),
        ("Des anopheles femelles, Anophèles gambiae. Herpès récidivant.",
         "Des anophèles femelles, Anopheles gambiae. Herpès récidivant."),
        ("un herpes iris, un herpes plus étendu", "un herpès iris, un herpès plus étendu"),
    ])
    print("test_latin_names: OK")


def test_hgvs() -> None:
    # The comparison spellings of ">" become "vers"; the correct French ones stay, and so
    # do look-alikes that are not a substitution (a protein change, an HbA1c, a date).
    check(normalize_hgvs, [
        ("mutation homozygote c.1163C supérieur à T", "mutation homozygote c.1163C vers T"),
        ("c.877-2A plus grand que G, pathogène", "c.877-2A vers G, pathogène"),
        ("c.1146A majuscule G et c.1277A majuscule vers G", "c.1146A vers G et c.1277A vers G"),
        ("c.1A majuscule à G", "c.1A vers G"),
        ("c.1331 plus 1 G supérieur à A", "c.1331 plus 1 G vers A"),
        ("c.-14C>T", "c.-14C vers T"),
        ("c.524G en A, c.646G vers A", "c.524G en A, c.646G vers A"),
        ("c.1234C substitué en T, c.893T remplacée par C", "c.1234C substitué en T, c.893T remplacée par C"),
        ("p.Arg388 vers His, HbA1c en A, c.12 en mai", "p.Arg388 vers His, HbA1c en A, c.12 en mai"),
    ])
    assert apply_label_conventions("c.35G supérieur à T") == "c.35G vers T"
    assert apply_label_conventions("c.35G supérieur à T", tts_source=True) == "c.35G vers T"
    print("test_hgvs: OK")


def test_numbers() -> None:
    # Compound number words become digits wherever their role is unknown; real v2 label
    # fragments, plus the user's examples (2026-11-01).
    check(normalize_numbers, [
        ("La maladie débute fin deux mille vingt-quatre.", "La maladie débute fin 2024."),
        ("Saturation à quatre-vingt-dix % en air ambiant.", "Saturation à 90 % en air ambiant."),
        ("Plaquettes à deux cents giga par litre.", "Plaquettes à 200 giga par litre."),
        ("On compte vingt-cinq mitoses, curage en onze à dix-huit.", "On compte 25 mitoses, curage en onze à 18."),
        ("Escherichia coli à cent mille par millilitre.", "Escherichia coli à 100000 par millilitre."),
        ("Grossissement fois quatre cents.", "Grossissement fois 400."),
        ("Sur vingt et un ganglions, Vingt-et-un lobe droit.", "Sur 21 ganglions, 21 lobe droit."),
        ("Tension à cent quarante et un sur 80, quatre-vingt-un.", "Tension à 141 sur 80, 81."),
        ("Macrogol quatre mille, mille neuf cent quatre-vingt-six.", "Macrogol 4000, 1986."),
        # Decimals, comma like the 43k "N,N" of v2.
        ("Mitoses à zéro virgule quatre-vingt-seize par millimètre carré.", "Mitoses à 0,96 par millimètre carré."),
        ("Ballon de trois virgule cinq, pH à sept virgule trente et un.", "Ballon de 3,5, pH à 7,31."),
        ("TSH à un virgule deux, éosinophiles à zéro virgule zéro deux.", "TSH à 1,2, éosinophiles à 0,02."),
        # A final million / milliard keeps its word (v2: "4 millions d'unités").
        ("Colimycine deux millions d'unités, un milliard d'UFC.", "Colimycine 2 millions d'unités, 1 milliard d'UFC."),
        ("Leucocyturie supérieure à un million par millilitre.", "Leucocyturie supérieure à 1 million par millilitre."),
        ("Rovamycine un virgule cinq millions d'unités.", "Rovamycine 1,5 millions d'unités."),
        ("Dose de un million cinq cent mille unités.", "Dose de 1500000 unités."),
        # Half converted by the LLM: the multiplier run and its digits are one number.
        ("Spiramycine un million 500000 unités internationales.", "Spiramycine 1500000 unités internationales."),
        ("À la dose d'un million 500000 unités.", "À la dose de 1500000 unités."),
        ("D'un million 500000 unités.", "De 1500000 unités."),
        ("Thrombocytose à six cent mille, deux mille 24.", "Thrombocytose à 600000, 2024."),
    ])
    for untouched in [
        "Cytokératine cinq six négative.",
        "Valeur cinq 3, facteur cinq 67 %, nerf cinq 3.",
        "Mille mercis, cent pas, grade trois, cotée trois croix.",
        "Le vingt-cinquième jour, au un sur deux cent cinquantième, au mille six centième.",
        "Hôpital des Quinze-Vingts.",
        "Schéma zéro un zéro, mobilité zéro cinq.",
        "Un deux trois, une vingtaine, d'un million de cellules.",
        "Mille parties par million 2 fois par jour.",
        # Two numbers side by side, not one.
        "Janumet mille sur cinquante un comprimé, Symbicort deux cents deux bouffées.",
        "Deux trois jours, vingt trente.",
        "Collyre 0,10 grammes pour cent une fois par jour.",
    ]:
        assert normalize_numbers(untouched) == untouched, (untouched, normalize_numbers(untouched))
    # Label only: the TTS source keeps the words voxtral read.
    assert apply_label_conventions("Saturation à quatre-vingt-dix %.", tts_source=True) == "Saturation à quatre-vingt-dix %."
    assert apply_label_conventions("Saturation à quatre-vingt-dix %.") == "Saturation à 90 %."
    print("test_numbers: OK")


def test_single_word_percent() -> None:
    # 2026-11-02 (author's choice): one number word before "%" converts too, cent and
    # mille included, so "cent pour cent" is "100 %" rather than "cent %".
    check(normalize_numbers, [
        ("Réduction de dix %, cent %, mille %.", "Réduction de 10 %, 100 %, 1000 %."),
        ("Un % des cas, zéro % de rechute.", "1 % des cas, 0 % de rechute."),
    ])
    # Through the label path 05 uses: percent_to_symbol, then the conventions.
    for raw, want in [("dix pour cent", "10 %"), ("cent pour cent", "100 %"),
                      ("zéro virgule cinq pour cent", "0,5 %"), ("cent pourcent", "100 %")]:
        got = apply_label_conventions(percent_to_symbol(raw))
        assert got == want, (raw, got)
    for untouched in ["Collyre 0,10 grammes pour cent une fois par jour.",
                      "dix patients, cent fois, quelques %"]:
        assert normalize_numbers(untouched) == untouched, (untouched, normalize_numbers(untouched))
    # Label only: the TTS source keeps the word.
    assert apply_label_conventions("cent %", tts_source=True) == "cent %"
    print("test_single_word_percent: OK")


def test_elided_one() -> None:
    # 2026-11-02 (author's choice): an elided "d'un"/"d'une" before a unit that takes
    # "1" becomes "d'1"; the elision is the quantity context a duration needs.
    check(normalize_elided_quantities, [
        ("La prise d'un comprimé et d'une gélule.", "La prise d'1 comprimé et d'1 gélule."),
        ("Un délai d'une heure, d'un mois, D'un an.", "Un délai d'1 heure, d'1 mois, D'1 an."),
        ("Ajout d'un gramme, d'un jour sur deux.", "Ajout d'1 gramme, d'1 jour sur 2."),
        ("Le contenu d'une cuillère-mesure.", "Le contenu d'1 cuillère-mesure."),
        # the restrictive "ne ... qu'un" likewise
        ("Ne prendre qu'un comprimé le matin.", "Ne prendre qu'1 comprimé le matin."),
        ("Il n'en faut qu'une goutte, il n'a pris qu'un sachet.", "Il n'en faut qu'1 goutte, il n'a pris qu'1 sachet."),
        ("Le vent ne dure qu'un jour, il n'avait qu’une heure.", "Le vent ne dure qu'1 jour, il n'avait qu’1 heure."),
    ])
    for untouched in [
        "d'une part, d'un côté, d'une manière, d'un patient",
        # what keeps a lone un spelled keeps it after the elision too
        "d'une fois par jour, d'une année, d'une séance, d'une ampoule",
        "d'un jour à l'autre, d'une semaine sur l'autre",
        # Common Voice 2026-11-02: an adjective after the unit is the idiom, not a span
        "est éclairé d'un jour nouveau, d'un jour meilleur",
        "d'un demi-comprimé",
        "d'un million cinq cent mille unités",
        # qu'un: same exceptions, a non-unit noun, and no "ne" before (conjunction, relative)
        "ne prendre qu'une fois par jour, il n'en reste qu'un seul, il n'a qu'un patient",
        "il ne voit qu'un jour nouveau, ne dure qu'un jour à l'autre",
        "Espérons qu'un jour il guérira. Si bien qu'un jour ils partent, prévenir qu'une goutte suffit",
        "Il est venu. Lorsqu'un comprimé manque, jusqu'un mois, puisqu'une heure",
    ]:
        assert normalize_elided_quantities(untouched) == untouched, (untouched, normalize_elided_quantities(untouched))
    # The large-number merge (commit 4128ac2) is unchanged.
    assert apply_label_conventions("Dose d'un million 500000 unités.") == "Dose de 1500000 unités."
    assert apply_label_conventions("La prise d'un comprimé.") == "La prise d'1 comprimé."
    # Label only: the TTS source keeps what voxtral reads.
    assert apply_label_conventions("La prise d'un comprimé.", tts_source=True) == "La prise d'un comprimé."
    # The label-only number steps as one function (what Common Voice rehearsal runs).
    check(normalize_label_numbers, [("Prise d'un comprimé, dix %, un million 500 000 unités.",
                                     "Prise d'1 comprimé, 10 %, 1500000 unités.")])
    print("test_elided_one: OK")


def test_all() -> None:
    raw = ("M. Dupont prend du DOLIPRANE à 14h30 depuis le quinze mars deux mille vingt, "
           "deux comprimés de cinq cents milligrammes au petit déjeuner.")
    want = ("Monsieur Dupont prend du Doliprane à 14 heures 30 depuis le 15 mars 2020, "
            "2 comprimés de 500 milligrammes au petit-déjeuner.")
    assert apply_label_conventions(raw) == want, apply_label_conventions(raw)
    assert apply_label_conventions(want) == want
    print("test_all: OK")


def test_generator_parse() -> None:
    try:
        from _pipeline_shared import parse_asr_training_target
    except ImportError as exc:
        print(f"test_generator_parse: SKIPPED ({exc})")
        return
    raw = "<t>Mme Petit prend du KARDEGIC.</t><t>Vu le premier mai à 9h.</t>"
    assert parse_asr_training_target(raw, expected=2) == [
        "Madame Petit prend du Kardégic.", "Vu le 1er mai à 9 heures."], parse_asr_training_target(raw, expected=2)
    print("test_generator_parse: OK")


if __name__ == "__main__":
    test_titles()
    test_dates()
    test_clock()
    test_parse_french_number()
    test_quantities()
    test_dosage_forms()
    test_single_duration_context()
    test_compounds()
    test_sutures_and_decimals()
    test_tts_source_pipeline_idempotent()
    test_thousands()
    test_spelling()
    test_staging()
    test_roman()
    test_drug_caser()
    test_accents()
    test_cefalexine_follows_inn_convention()
    test_accent_rewrite_elides()
    test_owner_review_rewrites()
    test_spelling_fixes_never_reach_tts_source()
    test_compound_spellings()
    test_official_names()
    test_generic_labs()
    test_non()
    test_ordinals()
    test_eponym_nouns()
    test_latin_names()
    test_hgvs()
    test_numbers()
    test_single_word_percent()
    test_elided_one()
    test_all()
    test_generator_parse()
