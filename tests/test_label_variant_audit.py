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
from pathlib import Path

HOTFIXES = Path(__file__).resolve().parent.parent / "06_hotfixes"


def _doctest(script: str) -> None:
    spec = importlib.util.spec_from_file_location(Path(script).stem, HOTFIXES / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (script, failed, attempted)
    print(f"{script}: OK ({attempted} cases)")


def test_label_variant_audit() -> None:
    _doctest("05_label_variant_audit.py")


def test_asr_residuals() -> None:
    _doctest("06_asr_residuals.py")


def test_build_accent_spellings() -> None:
    _doctest("07_build_accent_spellings.py")


if __name__ == "__main__":
    test_label_variant_audit()
    test_asr_residuals()
    test_build_accent_spellings()
