"""Stdlib test for `utils/number_swaps.py`, the wrong-number detector.

Pins the shapes found reading the 2026-10-28 hits one by one: the real TTS errors must
count (teen/seventy swaps, a truncated "cinquante", "une" heard as "mille"), while the
recognizers' own formatting choices and an alignment that drifted onto another number
must not.

Run: python tests/test_number_swaps.py

This file was written by Claude Code.
"""
from __future__ import annotations

import doctest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import number_swaps as N  # noqa: E402


def test_doctests() -> None:
    failed, attempted = doctest.testmod(N)
    assert attempted and not failed, (failed, attempted)
    print(f"test_doctests: OK ({attempted} cases)")


def test_real_errors_count() -> None:
    cases = [  # (label, what both recognizers wrote, expected swap), all from v2 clips
        ("Les réticulocytes sont à 50 giga par litre.",
         "Les réticulocytes sont à 5 gigas par litre.", ("50", "5")),
        ("évoluant depuis 1 semaine. Il n'y a pas",
         "évoluant depuis 1000 semaines. Il n'y a pas", ("1", "1000")),
        ("Madame X, 74 ans, est hospitalisée", "Madame X, 64 ans, est hospitalisée",
         ("74", "64")),
        ("le NT-proBNP est à 2951 picogrammes", "le NT-proBNP est à 2 551 picogrammes",
         ("2951", "2551")),
    ]
    for label, heard, want in cases:
        assert N.agreed_swaps(label, heard, heard) == [want], (label, heard)
    print("test_real_errors_count: OK")


def test_formatting_is_not_a_swap() -> None:
    same = [
        ("à 1,5 litre par 24 heures", "à 1.5 litre par 24 heures"),
        ("100 000 plaquettes", "100000 plaquettes"),
        ("2 milligrammes et demi par jour", "2,5 milligrammes par jour"),
        ("à 007 jours", "à 7 jours"),
    ]
    for label, heard in same:
        assert N.swaps(label, heard) == {}, (label, heard)
    print("test_formatting_is_not_a_swap: OK")


def test_drift_does_not_pair_unrelated_numbers() -> None:
    # 007775: the recognizer lost the thread for a few words; the label's "3" and the
    # hypothesis's "1" sit among different words, so they are not one swapped number.
    label = "un contrôle est prévu dans 3 semaines avec la bilirubine"
    heard = "un contrôle est prévu zone Lailé 1 avec la bilirubine"
    assert N.swaps(label, heard) == {}, N.swaps(label, heard)
    print("test_drift_does_not_pair_unrelated_numbers: OK")


def test_one_recognizer_is_not_enough() -> None:
    label = "patient de 75 ans"
    assert N.agreed_swaps(label, "patient de 65 ans", "patient de 75 ans") == []
    # Both wrong but disagreeing on HOW: a recognition problem, not a TTS one.
    assert N.agreed_swaps(label, "patient de 65 ans", "patient de 76 ans") == []
    print("test_one_recognizer_is_not_enough: OK")


if __name__ == "__main__":
    test_doctests()
    test_real_errors_count()
    test_formatting_is_not_a_swap()
    test_drift_does_not_pair_unrelated_numbers()
    test_one_recognizer_is_not_enough()
