"""Tests for 99_hf_release/audit_number_words.py: each bucket catches its shape and
nothing a rule already converts.

Run: uv run --with click python tests/test_audit_number_words.py

Written with Claude Code.
"""
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("audit_number_words", ROOT / "99_hf_release" / "audit_number_words.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def run(text: str) -> dict[str, Counter]:
    report: dict[str, Counter] = defaultdict(Counter)
    audit.audit_text(text, report, defaultdict(list))
    return report


def test_buckets() -> None:
    r = run("Un an plus tard, deux comprimés puis 2 comprimés.")
    assert r["a. lone un/une before a unit"] == Counter({"an": 1}), r
    assert r["a. value >= 2 before a unit"] == Counter({"comprimé": 1}), r
    assert r["d. same unit in digits and spelled"] == Counter({"comprimé": 1}), r
    r = run("Ne prendre qu'un comprimé, jusqu'à une heure, l'une gélule.")
    assert r["b. elided one before a unit"] == Counter({"qu'un": 1, "jusqu'à une": 1, "l'une": 1}), r
    r = run("dix pour cent, trois sur dix, cinq pour mille")
    assert r["c. spelled number before a percent or score"] == Counter(
        {"pour cent": 1, "sur dix": 1, "pour mille": 1}), r
    r = run("deux patients et trois patients, deux jours")
    assert r["e. word after a spelled number >= 2"] == Counter({"patients": 2}), r


def test_digits_only_is_silent() -> None:
    assert not any(run("2 comprimés de 500 milligrammes, 10 %, 3 sur 10.").values())
    # "pourcentage" is no "pourcent"
    assert not run("avec un pourcentage de cellules")["c. spelled number before a percent or score"]


if __name__ == "__main__":
    test_buckets()
    test_digits_only_is_silent()
    print("all checks passed")
