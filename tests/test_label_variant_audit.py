"""Tests for 06_hotfixes/05_label_variant_audit.py: its doctests hold the cases (the
``pré-remplie`` / ``préremplie`` and ``petit déjeuner`` / ``petit-déjeuner`` families,
runs not crossing a hyphenated word, the homograph mark).

    uv run --with click --with wordfreq tests/test_label_variant_audit.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "06_hotfixes" / "05_label_variant_audit.py"


def test_label_variant_audit() -> None:
    spec = importlib.util.spec_from_file_location("label_variant_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (failed, attempted)
    print(f"test_label_variant_audit: OK ({attempted} cases)")


if __name__ == "__main__":
    test_label_variant_audit()
