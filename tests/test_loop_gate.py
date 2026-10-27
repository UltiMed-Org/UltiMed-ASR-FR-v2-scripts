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
"""The loop gate: `repeat_excess` reaching `is_bad`, which it did not until 2026-10-27.

`tests/test_repeat_excess.py` already pins the detector itself (a stutter is counted, a
label that legitimately repeats a phrase is not). What was missing was the wiring: the
count was computed by `12_merge_judge_sidecars.py`, stored on every parakeet reading, and
then read by nothing that judges. So a take that says a clause twice was promoted and
shipped, because the four CER gates are structurally blind to it and the one judge that
is not blind had no vote.

The hole was not theoretical. Of the 1,624 clips a T3 round promoted on 2026-10-27, a
three-judge poll found 141 that BOTH turbo and parakeet still reject and 230 more that
only parakeet rejects, against a non-turbo arbiter that cleared every one of them; the
disputed clips carry 12-17% more audio per character than length-matched controls, which
is the excess speech the repeated clause is made of. Corpus-wide, 1,391 clips sit at
`repeat_excess >= 4` and 1,272 of those read clean to turbo.

What this test pins, i.e. the four things that had to be true for the defect to ship:

  * a reading with clean CER everywhere and a high `repeat_excess` IS bad once the gate
    is on, and is NOT bad while it is off (the defect, and the opt-in);
  * `scores_from_reading` carries the count off a stored reading at all (the wiring that
    was absent: the field did not exist, so the number was dropped on the floor);
  * a whisper reading, which carries no count, is unaffected either way, so turning the
    gate on cannot re-judge a whisper-judged pass;
  * `badness` still prefers a clean draw over a looping one, since that ordering is what
    makes a repair round promote the right candidate.

    uv run tests/test_loop_gate.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MOD_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"
CMP_PATH = HERE.parent / "06_hotfixes" / "10_compare_judges.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gates(repeat: int) -> SimpleNamespace:
    """The stage's four gates at their real values, plus the loop gate under test."""
    return SimpleNamespace(cer_threshold=0.08, tail_cer_threshold=0.12,
                           window_cer_threshold=0.15, bad_run_threshold=30,
                           repeat_threshold=repeat)


#: A real looping clip's parakeet reading, rounded: 000766_0000_anatomopathologie_00193
#: (52.24 s, voice h_frm_ptf, cfg 1.5). It says a whole block twice ("L'examen
#: macroscopique retrouve trois fragments biopsiques de 9 mm ... inclus en totalite dans
#: un bloc."), and every CER it has is comfortably clean, because the repetition adds
#: text rather than removing or changing any: that is exactly why no CER gate can see it.
LOOPING = {"text": "l examen macroscopique retrouve trois fragments ... un bloc "
                   "l examen macroscopique retrouve trois fragments ... un bloc",
           "cer": 0.021, "cer_tail": 0.03, "cer_window": 0.05, "bad_run_chars": 2,
           "repeat_excess": 12}
#: The same clip as the turbo judge reads it: the decoder collapsed the repeat, so there
#: is nothing to count and no field to count it in.
TURBO_BLIND = {"text": "l examen macroscopique retrouve trois fragments ... un bloc",
               "cer": 0.019, "cer_tail": 0.02, "cer_window": 0.04, "bad_run_chars": 2}


def test_a_looping_reading_is_bad_only_once_the_gate_is_on(mod) -> None:
    scores = mod.scores_from_reading(LOOPING)
    assert scores.repeat == 12, f"the count must survive the mapping: {scores}"
    assert not mod.is_bad(scores, gates(0)), \
        "gate off: the four CER gates are blind to a repeat, which is the whole defect"
    assert mod.is_bad(scores, gates(4)), \
        "gate on at the measured cut: the clip that shipped must now read bad"
    assert mod.is_bad(scores, gates(12)), "the gate is >=, not >"
    assert not mod.is_bad(scores, gates(13)), "and it does not fire below its threshold"


def test_an_absent_count_never_trips_the_gate(mod) -> None:
    """Turning the gate on must not re-judge whisper-judged work: `score_texts` does not
    compute the count, so every whisper reading in the corpus lacks the field, and a
    missing number has to mean "this judge cannot answer", not "zero" and not "bad"."""
    scores = mod.scores_from_reading(TURBO_BLIND)
    assert scores.repeat is None, f"no field, no count: {scores}"
    for threshold in (0, 1, 4, 30):
        assert not mod.is_bad(scores, gates(threshold)), \
            f"a reading with no loop count is not bad at threshold {threshold}"
    live = mod.score_reading("un texte dit une fois", "un texte dit une fois", None, 3.0)
    assert live.repeat is None, \
        "the live scorer does not count repeats either (the merge script owns it)"


def test_badness_prefers_the_candidate_that_does_not_loop(mod) -> None:
    """A repair round promotes by `badness`, so the ordering is what fixes the clip."""
    cfg = gates(4)
    looping = mod.scores_from_reading(LOOPING)
    clean = mod.Scores(cer=0.03, tail=0.03, window=0.05, bad_run=2, repeat=0)
    assert mod.badness(clean, cfg) < mod.badness(looping, cfg), \
        "a clean draw must beat a looping one even though its CER is HIGHER"


def test_the_comparator_names_the_gate_that_fired(cmp_mod) -> None:
    """`10_compare_judges.py` is how the defect is read out of the state files, so its
    report has to say `repeat` rather than reporting a clip with no gate at all."""
    recimp = cmp_mod._load_pass()
    scores = recimp.scores_from_reading(LOOPING)
    assert cmp_mod.which_gates(scores, cmp_mod.gates_from(0.08, 0.12, 0.15, 30, 4)) \
        == ["repeat"], "the loop is the only thing wrong with this reading"
    assert cmp_mod.which_gates(scores, cmp_mod.gates_from(0.08, 0.12, 0.15, 30)) == [], \
        "and with the gate off (its default argument) the clip looks clean"


if __name__ == "__main__":
    pass_mod = load_module(MOD_PATH, "recimp_loop_gate")
    test_a_looping_reading_is_bad_only_once_the_gate_is_on(pass_mod)
    test_an_absent_count_never_trips_the_gate(pass_mod)
    test_badness_prefers_the_candidate_that_does_not_loop(pass_mod)
    test_the_comparator_names_the_gate_that_fired(load_module(CMP_PATH, "cmp_loop_gate"))
    print("OK: the loop gate is wired, opt-in, and inert on a judge that cannot count")
