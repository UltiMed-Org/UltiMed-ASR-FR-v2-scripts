"""Tests for 06_hotfixes/06_asr_residuals.py: its doctests hold the cases (tokenizing with
elisions and hyphens kept, the case / ortho / other kinds, the aligned label -> hyp pairs,
the WER count).

    uv run --with click --with wordfreq tests/test_asr_residuals.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "06_hotfixes" / "06_asr_residuals.py"


def test_asr_residuals() -> None:
    spec = importlib.util.spec_from_file_location("asr_residuals", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (failed, attempted)
    print(f"test_asr_residuals: OK ({attempted} cases)")


if __name__ == "__main__":
    test_asr_residuals()
