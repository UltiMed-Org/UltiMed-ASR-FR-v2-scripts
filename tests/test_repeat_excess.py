#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""`repeat_excess` in `06_hotfixes/01_compute_stt.py`: the loop detector.

The defect it guards against is the only one all four CER gates are structurally blind
to. Whisper's decoder suppresses repeated n-grams, so a clip that stutters a phrase
comes back as a clean transcript: over 211 acoustically-flagged clips its transcript ran
0.968x the reference length at CER 0.017, where Parakeet transcribed the repeats
verbatim and scored 0.180. No threshold can recover information the decoder removed, so
the metric has to read a non-deduplicating transcript and look at structure instead of
distance.

The properties worth pinning are not "the metric has value X" but:
  * a stutter in a transcript is counted, and counted more the longer it ran;
  * a label that legitimately repeats a phrase (this corpus is full of them: "1 comprime
    matin midi et soir", once per drug in a discharge summary) is NOT a defect, however
    many times it repeats, which is what makes the metric usable here at all;
  * one extra reading of such a phrase still counts, so a loop inside repetitive text is
    not forgiven along with the text;
  * the two error classes that pollute Parakeet's CER (French medical mishearings and
    mid-sentence English language-id flips) score zero, since each says its words once;
  * normalization is applied, so a repeat spelled with different digits/units still
    counts;
  * a transcript shorter than the window, or empty, is not a defect.

    uv run tests/test_repeat_excess.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STT_PATH = HERE.parent / "06_hotfixes" / "01_compute_stt.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stt = load_module(STT_PATH, "compute_stt_under_test")

# Lifted from a real flagged clip (022432_0001_syndrome_de_frankl_hochwart_pellizzi,
# Parakeet CER 0.338): the voice re-read a stretch it had already spoken.
LABEL = ("la puberte precoce est bien controlee sous agoniste de la gnrh l irm de "
         "controle montre une stabilite lesionnelle et le bilan cognitif met en "
         "evidence des difficultes attentionnelles")
LOOPED = ("la puberte precoce est bien controlee sous agoniste de la gnrh l irm de "
          "controle montre une stabilite lesionnelle et le bilan cognitif met en "
          "evidence des deficits precoce est bien controlee sous agoniste de la gnrh "
          "l irm de controle montre une stabilite lesionnelle et le bilan cognitif met "
          "en evidence des difficultes attentionnelles")


def test_a_clean_take_counts_nothing() -> None:
    assert stt.repeat_excess(LABEL, LABEL) == 0, "a perfect transcript is not a loop"


def test_a_stutter_is_counted_and_grows_with_its_length() -> None:
    short = stt.repeat_excess(LABEL, LOOPED)
    assert short > 0, f"the real looped clip scored {short}, expected > 0"
    # Re-reading twice as much text should cost more, since the metric is a count of
    # windows rather than a yes/no: that is what lets the repair round rank its queue.
    longer = stt.repeat_excess(LABEL, LOOPED + " " + LABEL)
    assert longer > short, f"a longer loop scored {longer}, not more than {short}"


def test_a_label_that_repeats_itself_is_not_a_defect() -> None:
    """The reason this is `excess` and not a plain repeat count.

    A discharge summary says the same posology once per drug. Read correctly, the
    transcript repeats it exactly as often as the label does, and must score zero.
    """
    phrase = "1 comprime matin midi et soir pendant 5 jours"
    label = (f"le traitement comprend du paracetamol {phrase} il comprend aussi du "
             f"spasfon {phrase} il comprend enfin de l ibuprofene {phrase}")
    assert stt.repeat_excess(label, label) == 0, (
        "a label repeating its own phrasing three times was called a loop")


def test_one_extra_reading_inside_repetitive_text_still_counts() -> None:
    """The forgiveness is per occurrence, not a blanket pass for repetitive labels.

    This is the case that decides whether a loop hiding inside a drug list is caught, and
    PARHAF (22.9% acoustically flagged) is mostly drug lists.
    """
    phrase = "1 comprime matin midi et soir pendant 5 jours"
    label = f"paracetamol {phrase} et spasfon {phrase}"
    heard = f"paracetamol {phrase} {phrase} et spasfon {phrase}"
    assert stt.repeat_excess(label, heard) > 0, (
        "an extra reading of an already-repeated phrase was forgiven")


def test_the_recognizer_s_own_error_classes_score_zero() -> None:
    """What makes this metric gradeable where Parakeet's CER is not.

    Both of these are real transcripts from the sweep. Neither is the TTS's fault, and
    both score high CER: a mishearing of a rare medical name, and a mid-sentence flip
    into English. A loop detector must ignore them, or it inherits the CER's noise.
    """
    label = "les suites sont simples le signataire est le docteur mayssem cassorla"
    lid_flip = "les tweets ensemble le signateur and le doctor mysem casola"
    assert stt.repeat_excess(label, lid_flip) == 0, "an English language-id flip counted"

    label2 = ("1 anticoagulation par aod est introduite avec eliquis 10 milligrammes "
              "deux fois par jours")
    misheard = ("1 anticoagulation par aod est intraduite avec helichis 10 milligrammes "
                "deux fois par jours")
    assert stt.repeat_excess(label2, misheard) == 0, "a medical mishearing counted"


def test_the_comparison_is_normalized_like_every_other_metric() -> None:
    """A repeat spelled differently each time is still a repeat.

    `normalize_for_scoring` folds "deux" to "2" and "mg" to "milligrammes", so a loop
    whose two readings were transcribed with different spellings of the same words has to
    survive into the count. Without normalization the windows would not match and the
    loop would be invisible.
    """
    label = "la dose est de deux milligrammes par jours"
    heard = "la dose est de 2 mg par jours deux milligrammes par jours"
    assert stt.repeat_excess(label, heard) > 0, (
        "a loop whose readings were spelled differently was missed")


def test_short_and_empty_transcripts_are_not_defects() -> None:
    assert stt.repeat_excess("bisoprolol", "bisoprolol") == 0
    assert stt.repeat_excess("bisoprolol dix milligrammes", "") == 0, (
        "an empty transcript is a different defect (the CER gates own it), not a loop")
    assert stt.repeat_excess("", "") == 0


def test_an_honestly_repeated_clause_misheard_twice_is_not_a_loop() -> None:
    """The one-word tolerance, which is the difference between a useful rule and a noisy
    one on repetitive clinical text.

    A label that writes a clause twice, read correctly, but with the recognizer
    mishearing the same word in both copies, yields a hypothesis window the label never
    contains, said twice. Comparing windows verbatim calls that a loop. Measured on the
    corpus: that mistake was 50 clips of 19,120 whose median CER is 0.025, i.e. clean
    takes (a vital-signs list at CER 0.0097, a label writing "1 comprime le matin / 1
    comprime le soir" whose "1" both times came back "on"). The tolerance also costs
    nothing the pipeline does not recover: of the 185 verbatim fires it suppresses, the
    independent acoustic scan flags 135 anyway.
    """
    clause = "la patiente recoit deux grammes par jour pendant dix jours"
    label = f"{clause} puis {clause} puis une surveillance"
    # Read as written, with one word misheard identically in both copies.
    heard = label.replace("recoit", "recevait")
    assert stt.repeat_excess(label, heard) == 0, (
        "a clause the LABEL repeats is not a loop just because one word in it was "
        "misheard the same way each time")
    # The tolerance is one word wide, not a free pass: three words off in both copies is
    # no longer recognisable as the label's own clause.
    assert stt.repeat_excess(label, label.replace("recoit deux grammes",
                                                  "recevait trois litres")) > 0
    # And a take that says the clause MORE often than the label still counts, tolerance
    # or not: the forgiveness is capped at the label's own count.
    assert stt.repeat_excess(label, f"{heard} puis {clause}") > 0


def test_the_window_width_is_the_measured_one() -> None:
    """Guards the constant, because the whole metric's precision is a function of it.

    Agreement with the independent acoustic scan over 19,120 clips was 54.4% precision at
    n=2, 73.1% at n=3, 81.2% at n=4, 86.0% at n=6 and 92.7% at n=8, against recalls of
    29.1%, 27.2%, 23.7%, 18.8% and 14.5%. If this value moves, that trade moved with it
    and the sweep's flag counts are not comparable to the ones in the plan.
    """
    assert stt.REPEAT_NGRAM_WORDS == 4, (
        f"window width is {stt.REPEAT_NGRAM_WORDS}, the measured choice was 4")
    assert stt.DEFAULT_REPEAT_GATE == 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        else:
            print("  ok")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
