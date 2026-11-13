#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Tests for 08_drug_asr_rules: the doctests of 01_extract_drug_errors.py (tokenizing,
drug spotting, the character-level mapping inside a multi-word replace, glued elisions)
and of 02_build_fix_rules.py (pattern bounds, rejection reasons, ordering, apply_rules),
03_merge_rules.py, 04_drop_unk_variants.py, 05_build_manual_rules.py and
06_build_convention_fixes.py.

    uv run tests/test_drug_asr_rules.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

STAGE = Path(__file__).resolve().parent.parent / "08_drug_asr_rules"


def _run_doctests(name: str) -> None:
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), STAGE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (name, failed, attempted)
    print(f"{name}: OK ({attempted} cases)")


def test_extract_drug_errors() -> None:
    _run_doctests("01_extract_drug_errors.py")


def test_extract_hyps_out() -> None:
    """``01 --hyps-out`` keeps only the hypotheses of the manifests' clips, in manifest
    order: how the v2 build restricts the whole-corpus transcription to one subset's
    splits before 02 reads it."""
    import json
    import tempfile

    from click.testing import CliRunner

    spec = importlib.util.spec_from_file_location("extract", STAGE / "01_extract_drug_errors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        hyps = [{"audio": str(d / "drugs" / f"{n}.flac"), "hyp": f"hyp {n}"} for n in ("a", "b", "c")]
        (d / "hyps.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hyps))
        (d / "drugs").mkdir()
        rows = [{"audio_filepath": f"{n}.flac", "text": f"label {n}"} for n in ("c", "a", "z")]
        (d / "drugs" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        res = CliRunner().invoke(mod.main, [str(d / "hyps.jsonl"), str(d / "drugs" / "train.jsonl"),
                                            "--out", str(d / "err.json"), "--hyps-out", str(d / "sub.jsonl")])
        assert res.exit_code == 0, res.output
        got = [json.loads(line)["hyp"] for line in (d / "sub.jsonl").open()]
        assert got == ["hyp c", "hyp a"], got
    print("01 --hyps-out: OK")


def test_build_fix_rules() -> None:
    _run_doctests("02_build_fix_rules.py")


def test_merge_rules() -> None:
    _run_doctests("03_merge_rules.py")


def test_drop_unk_variants() -> None:
    _run_doctests("04_drop_unk_variants.py")


def test_build_manual_rules() -> None:
    _run_doctests("05_build_manual_rules.py")


def test_build_convention_fixes() -> None:
    _run_doctests("06_build_convention_fixes.py")


if __name__ == "__main__":
    test_extract_drug_errors()
    test_extract_hyps_out()
    test_build_fix_rules()
    test_merge_rules()
    test_drop_unk_variants()
    test_build_manual_rules()
    test_build_convention_fixes()
