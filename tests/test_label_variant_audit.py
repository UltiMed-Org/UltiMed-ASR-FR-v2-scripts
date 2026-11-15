#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "wordfreq"]
# ///
"""Tests for the 06_hotfixes label audit scripts (05_label_variant_audit.py,
06_asr_residuals.py, 07_build_accent_spellings.py): their doctests hold the cases (the ``pré-remplie`` / ``préremplie`` and ``petit déjeuner`` /
``petit-déjeuner`` families, runs not crossing a hyphenated word, the homograph mark; the
tokenizing with elisions kept, the case / ortho / other residual kinds and the WER count;
the canonical accent form, drug INNs without accents, words of their own spared).

    uv run tests/test_label_variant_audit.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
import json
from pathlib import Path

HOTFIXES = Path(__file__).resolve().parent.parent / "06_hotfixes"


def _load(script: str):
    spec = importlib.util.spec_from_file_location(Path(script).stem, HOTFIXES / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _doctest(script: str) -> None:
    mod = _load(script)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (script, failed, attempted)
    print(f"{script}: OK ({attempted} cases)")


def test_label_variant_audit() -> None:
    _doctest("05_label_variant_audit.py")


def test_asr_residuals() -> None:
    _doctest("06_asr_residuals.py")


def test_build_accent_spellings() -> None:
    _doctest("07_build_accent_spellings.py")


def test_hand_rewrites_committed() -> None:
    """The committed table carries every ``HAND_REWRITES`` entry (the TSV is not in the
    repo, so the JSON is synced by hand), and every canonical form is final: a value that
    is also a key would make ``normalize_accents`` need two passes (2026-11-15)."""
    hand = _load("07_build_accent_spellings.py").HAND_REWRITES
    table = json.loads((HOTFIXES.parent / "utils" / "accent_spellings.json").read_text(encoding="utf-8"))["rewrites"]
    missing = {k: v for k, v in hand.items() if table.get(k) != v}
    assert not missing, ("HAND_REWRITES not synced into accent_spellings.json", missing)
    assert all(k == k.lower() for k in hand), "keys are looked up lowercased"
    # An identity entry (``tymoferol``) only recases all-caps words, it is not a chain.
    chained = {k: v for k, v in table.items() if v.lower() in table and v.lower() != k}
    assert not chained, ("canonical form is itself rewritten", chained)
    print(f"test_hand_rewrites_committed: OK ({len(hand)} entries)")



def test_resynth_typos_held_back() -> None:
    """The typos whose clips still say them stay out of the committed table until those
    clips are re-synthesized, else the next export puts a label on audio that never said it."""
    held = _load("07_build_accent_spellings.py").RESYNTH_TYPOS
    table = json.loads((HOTFIXES.parent / "utils" / "accent_spellings.json").read_text(encoding="utf-8"))["rewrites"]
    assert not set(held) & set(table), set(held) & set(table)
    print("test_resynth_typos_held_back: OK")


if __name__ == "__main__":
    test_label_variant_audit()
    test_asr_residuals()
    test_build_accent_spellings()
    test_hand_rewrites_committed()
    test_resynth_typos_held_back()
