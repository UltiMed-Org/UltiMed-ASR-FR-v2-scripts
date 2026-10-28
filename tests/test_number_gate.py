#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "soundfile>=0.12",
# ]
# ///
"""The number gate: a draw that says "65" for "75" must not be promoted.

Found 2026-10-28: 5,291 released clips where turbo and parakeet both hear a label number
as the same other number, every one of them under every CER gate (two characters out of
hundreds). A repair round promotes the first draw `is_bad` calls clean, so without this
gate a redraw that repeats the error would be promoted and the clip closed as fixed.

What this pins:

  * `score_reading` counts the swap, against whichever reference explains it best;
  * `is_bad` ignores it while the gate is off (the default, so no existing pass is
    re-judged) and fails the reading once it is on;
  * `badness` puts the draw that says the right number first, which is what makes the
    round promote it.

    uv run tests/test_number_gate.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

MOD_PATH = Path(__file__).resolve().parent.parent / "06_hotfixes" / "01_recursive_improvement.py"

LABEL = "Patiente de 75 ans, hospitalisée pour une décompensation cardiaque."


def load_module():
    spec = importlib.util.spec_from_file_location("recimp_number_gate", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recimp_number_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def gates(numbers: int) -> SimpleNamespace:
    return SimpleNamespace(cer_threshold=0.08, tail_cer_threshold=0.12,
                           window_cer_threshold=0.15, bad_run_threshold=30,
                           repeat_threshold=0, number_threshold=numbers)


def test_the_swap_is_counted(mod) -> None:
    wrong = mod.score_reading(LABEL.replace("75", "65"), LABEL, None, 5.0)
    assert wrong.numbers == 1, wrong
    assert wrong.cer < 0.08, "under the CER gate, which is the whole defect"
    right = mod.score_reading(LABEL, LABEL, None, 5.0)
    assert right.numbers == 0, right
    # A target that spells the number the way the judge heard it explains the reading.
    explained = mod.score_reading(LABEL.replace("75", "65"), LABEL, LABEL.replace("75", "65"), 5.0)
    assert explained.numbers == 0, explained
    assert mod.score_reading(None, LABEL, None, 5.0).numbers is None


def test_the_gate_is_opt_in(mod) -> None:
    wrong = mod.score_reading(LABEL.replace("75", "65"), LABEL, None, 5.0)
    assert not mod.is_bad(wrong, gates(0)), "off by default: no existing pass re-judged"
    assert not mod.is_bad(wrong, SimpleNamespace(**{k: v for k, v in vars(gates(0)).items()
                                                    if k != "number_threshold"})), \
        "a cfg that predates the gate has no number_threshold at all"
    assert mod.is_bad(wrong, gates(1)), "on: the wrong number fails the draw"
    stored = mod.scores_from_reading({"text": "x", "cer": 0.01})
    assert stored.numbers is None and not mod.is_bad(stored, gates(1)), \
        "a stored reading carries no count, and a missing count never trips the gate"


def test_badness_prefers_the_right_number(mod) -> None:
    cfg = gates(1)
    wrong = mod.score_reading(LABEL.replace("75", "65"), LABEL, None, 5.0)
    right = mod.score_reading(LABEL.replace("pour une", "pour la"), LABEL, None, 5.0)
    assert right.cer > wrong.cer, "the right draw even has the higher CER here"
    assert mod.badness(right, cfg) < mod.badness(wrong, cfg)


if __name__ == "__main__":
    mod = load_module()
    test_the_swap_is_counted(mod)
    test_the_gate_is_opt_in(mod)
    test_badness_prefers_the_right_number(mod)
    print("OK: the number gate counts swaps, is opt-in, and ranks the right draw first")
