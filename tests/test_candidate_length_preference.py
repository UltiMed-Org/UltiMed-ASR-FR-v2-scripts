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
"""Among candidate draws of one sentence, the shorter clean one wins: the loop preference.

The defect this closes. A repair round judged by whisper can promote a looping take over
another looping take, because whisper's decoder collapses a repeated clause and so reads
both as clean (tests/test_loop_gate.py pins that blindness, and the parakeet gate that
sees it). On the T4 round of 2026-10-27 that is exactly what happened: of 807 promotions
with a measurable before/after, 199 came back LONGER on identical text, and against
voice- AND length-matched controls those sit at -11.2 pct chars per second with 47 pct
carrying a loop-sized deficit, while the 608 that came back shorter sit at +1.3 pct, i.e.
repaired. Duration is the one signal available without a second judge: all five draws
speak the same sentence, so only speaking-rate jitter (small, symmetric) or a repeat
(large, one-sided) can move it.

Why a preference and not a veto, which is the part worth pinning. The 1,703 same-text
draw pairs that round left on disk gave no gap to hang a threshold on: the ratios to each
clip's shortest draw form one smooth mode (p50 1.122, p75 1.223, p90 1.324, p99 1.571).
So the rule may only choose between takes that are already acceptable, and it must never
be able to prefer a BROKEN short take: `is_bad` sorts ahead of it, the yardstick is the
low median of the draws that PASS every gate rather than of all draws, and in a class where
nothing passes it stays out altogether, because there the promotion test is pure CER and
a shorter-but-worse draw could only block a real gain.

    uv run tests/test_candidate_length_preference.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MOD_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cfg_of(mod, length_tol: float = 0.15) -> SimpleNamespace:
    """The stage's real gates, plus the knob under test."""
    return SimpleNamespace(
        cer_threshold=0.08, tail_cer_threshold=0.12, window_cer_threshold=0.15,
        bad_run_threshold=30, repeat_threshold=0,
        min_improvement=0.02, length_tol=length_tol, n_improv=5,
    )


def pick(mod, tmp: Path, candidates: list[dict], length_tol: float = 0.15,
         orig_cer: float = 0.30) -> dict:
    """Run `score_and_pick` over fabricated draws, with the STT call stubbed out.

    Each candidate is `{"seed", "dur", "scores"}`. A real (empty) file has to exist per
    draw because the picker skips a missing draft and reads the winner's bytes back.
    """
    drafts = []
    by_path: dict[str, dict] = {}
    for c in candidates:
        path = tmp / f"clip.flac.draft{c['seed']:02d}.flac"
        path.write_bytes(b"")
        drafts.append({"path": str(path), "seed": c["seed"], "cfg_alpha": 2.0,
                       "dur": c["dur"]})
        by_path[str(path)] = c

    def fake_score(audio_file, cfg, source, target, duration=None):
        c = by_path[str(audio_file)]
        return {"transcript": f"seed {c['seed']}", "scores": c["scores"],
                "stt_seconds": 0.1}

    mod._score_audio_file = fake_score
    return mod.score_and_pick(
        0, "clip.flac", drafts, "le texte source", None,
        mod.Scores(cer=orig_cer, tail=orig_cer, window=orig_cer, bad_run=0),
        cfg_of(mod, length_tol),
    )


def clean(cer: float) -> "object":
    """A draw that passes every gate, at the given CER."""
    return SimpleNamespace(cer=cer, tail=0.02, window=0.04, bad_run=0, repeat=None)


def test_the_shorter_clean_draw_beats_the_lower_cer_one(mod, tmp: Path) -> None:
    """The whole point: both pass every gate, so CER is no longer the thing to trust.
    13.0 / 10.0 = 1.30, past the 15 pct tolerance, so the long one is ranked down even
    though its CER is five times better."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 10.0, "scores": clean(0.05)},
                           {"seed": 2, "dur": 13.0, "scores": clean(0.01)}])
    assert got["seed"] == 1, f"the shorter clean draw must win, got seed {got['seed']}"
    assert got["payload"] is not None, "and it must still be promoted"


def test_within_the_tolerance_cer_still_decides(mod, tmp: Path) -> None:
    """11.0 / 10.0 = 1.10 is ordinary speaking-rate jitter (the measured p50 is 1.12), so
    the rule must stay out of the way and let the better reading win."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 10.0, "scores": clean(0.05)},
                           {"seed": 2, "dur": 11.0, "scores": clean(0.01)}])
    assert got["seed"] == 2, \
        f"inside the tolerance the lower CER must win, got seed {got['seed']}"


def test_a_truncated_draw_cannot_win_by_being_short(mod, tmp: Path) -> None:
    """The hazard the ordering exists to prevent. Seed 1 is short because it stops before
    the end of the sentence, which is what `cer_tail` catches; it must neither win nor
    become the yardstick that ranks the honest take too long."""
    truncated = SimpleNamespace(cer=0.07, tail=0.90, window=0.95, bad_run=0, repeat=None)
    got = pick(mod, tmp, [{"seed": 1, "dur": 6.0, "scores": truncated},
                           {"seed": 2, "dur": 13.0, "scores": clean(0.01)}])
    assert got["seed"] == 2, \
        f"a gate-failing draw must lose however short it is, got seed {got['seed']}"


def test_the_yardstick_ignores_failing_draws(mod, tmp: Path) -> None:
    """With a 6.0 s truncated draw present, the two clean draws (12.0 and 13.0) are 1.08
    apart and must both rank 0, so CER decides between them. Measured against the 6.0 s
    wreck instead, both would rank 1 and the preference would silently do nothing."""
    truncated = SimpleNamespace(cer=0.07, tail=0.90, window=0.95, bad_run=0, repeat=None)
    got = pick(mod, tmp, [{"seed": 1, "dur": 6.0, "scores": truncated},
                           {"seed": 2, "dur": 12.0, "scores": clean(0.05)},
                           {"seed": 3, "dur": 13.0, "scores": clean(0.01)}])
    assert got["seed"] == 3, \
        f"the reference must be 12.0 (passing draws only), not 6.0; got seed {got['seed']}"


def test_a_fast_outlier_does_not_set_the_standard(mod, tmp: Path) -> None:
    """Why the anchor is a median. Anchored on the shortest passing draw, the 8.0 s take
    would make 10.5 s look 31 pct long and win on speed alone, which is what round 3
    showed (kept draws +17.5 pct faster than their voice, rejected ones at the norm).
    Against the median, 10.0 s, all three are ordinary jitter and CER decides."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 8.0, "scores": clean(0.05)},
                           {"seed": 2, "dur": 10.0, "scores": clean(0.03)},
                           {"seed": 3, "dur": 10.5, "scores": clean(0.01)}])
    assert got["seed"] == 3, \
        f"a fast take must not make typical ones look long, got seed {got['seed']}"


def test_a_long_take_is_still_demoted_against_the_median(mod, tmp: Path) -> None:
    """The case the rule exists for survives the change: 15.0 s against a 10.0 s median
    is a repeat-sized 1.5, so its better CER does not save it."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 9.0, "scores": clean(0.05)},
                           {"seed": 2, "dur": 10.0, "scores": clean(0.03)},
                           {"seed": 3, "dur": 15.0, "scores": clean(0.01)}])
    assert got["seed"] == 2, f"the long take must lose, got seed {got['seed']}"


def test_two_passing_draws_anchor_on_the_shorter(mod, tmp: Path) -> None:
    """Why median_low and not median. With two draws the plain median is their midpoint,
    41.7 s, and a 46.5 s loop of a 36.9 s sentence would slip under 41.7 * 1.15 = 48.0."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 36.9, "scores": clean(0.04)},
                           {"seed": 2, "dur": 46.5, "scores": clean(0.01)}])
    assert got["seed"] == 1, \
        f"a 1.26 loop between two draws must still lose, got seed {got['seed']}"


def test_when_every_draw_fails_a_gate_cer_alone_decides(mod, tmp: Path) -> None:
    """The interaction that would otherwise stall convergence. When no draw passes, the
    only question left is whether the best of them beats the original by
    `--min-improvement`, and that test is pure CER. Original at 0.11, draws at 10.0 s /
    0.10 and 13.0 s / 0.085: on CER the second promotes (0.085 <= 0.11 - 0.02), on length
    the first wins and promotes nothing, so the clip would keep its worse audio and go
    round again for no reason. The length rank must stay out of a class where nothing is
    acceptable."""
    over_short = SimpleNamespace(cer=0.100, tail=0.02, window=0.04, bad_run=0, repeat=None)
    over_long = SimpleNamespace(cer=0.085, tail=0.02, window=0.04, bad_run=0, repeat=None)
    got = pick(mod, tmp, [{"seed": 1, "dur": 10.0, "scores": over_short},
                           {"seed": 2, "dur": 13.0, "scores": over_long}],
               orig_cer=0.11)
    assert got["seed"] == 2, \
        f"among failing draws the lowest CER must win, got seed {got['seed']}"
    assert got["payload"] is not None, \
        "and it must still be promoted on --min-improvement, which is the point"


def test_a_passing_draw_still_beats_a_shorter_failing_one(mod, tmp: Path) -> None:
    """The restriction must not reopen the hazard: ordering by `is_bad` first is what
    keeps a short wreck from winning, and that is unchanged."""
    broken = SimpleNamespace(cer=0.02, tail=0.95, window=0.99, bad_run=0, repeat=None)
    got = pick(mod, tmp, [{"seed": 1, "dur": 7.0, "scores": broken},
                           {"seed": 2, "dur": 14.0, "scores": clean(0.03)}])
    assert got["seed"] == 2, f"a passing draw must win outright, got seed {got['seed']}"


def test_the_knob_turns_the_preference_off(mod, tmp: Path) -> None:
    """A negative tolerance restores the CER-only ordering, which is what makes the change
    auditable: the same draws, the other winner."""
    draws = [{"seed": 1, "dur": 10.0, "scores": clean(0.05)},
             {"seed": 2, "dur": 13.0, "scores": clean(0.01)}]
    assert pick(mod, tmp, draws, length_tol=-1.0)["seed"] == 2, \
        "with the knob off the lowest CER must win again"


def test_the_winner_does_not_carry_the_duration_key(mod, tmp: Path) -> None:
    """`dur` is bookkeeping for the new log line. The returned dict is spread into the
    state row's improvement marker, so an extra key would be persisted on every promoted
    clip in the corpus."""
    got = pick(mod, tmp, [{"seed": 1, "dur": 10.0, "scores": clean(0.01)}])
    assert "dur" not in got, f"the marker must not gain a dur field: {sorted(got)}"


def test_length_rank_of_is_inert_without_numbers(mod) -> None:
    """Three ways the rank has nothing to say, all of which must mean "decide on CER"
    rather than "penalise": the knob is off, the draw's length could not be probed, or
    no draw at all had a length."""
    cfg = cfg_of(mod)
    assert mod.length_rank_of(13.0, 10.0, cfg) == 1, "the baseline case still fires"
    assert mod.length_rank_of(11.5, 10.0, cfg) == 0, "1.15 is the boundary, not past it"
    assert mod.length_rank_of(None, 10.0, cfg) == 0, "unknown duration: not penalised"
    assert mod.length_rank_of(13.0, None, cfg) == 0, "no reference: nothing to compare"
    assert mod.length_rank_of(13.0, 10.0, cfg_of(mod, -1.0)) == 0, "knob off"
    assert mod.length_rank_of(13.0, 10.0, SimpleNamespace()) == 0, \
        "a cfg built before the knob existed must not crash or penalise"


def test_badness_keeps_its_old_ordering_for_readings(mod) -> None:
    """`badness` is also the sort key for several JUDGES' readings of one clip, where
    there is no length to compare. The new middle term defaults to 0, so that ordering
    must be untouched: a clean reading still beats one with a derailed tail."""
    cfg = cfg_of(mod)
    assert (mod.badness(mod.Scores(cer=0.02, tail=0.9), cfg)
            > mod.badness(mod.Scores(cer=0.05, tail=0.01), cfg)), \
        "a clean tail must still beat a lower CER with a derailed one"
    assert mod.badness(mod.Scores(cer=0.02, tail=0.01), cfg)[1] == 0, \
        "and a reading comparison must never carry a length rank"


if __name__ == "__main__":
    pass_mod = load_module(MOD_PATH, "recimp_length_pref")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        test_the_shorter_clean_draw_beats_the_lower_cer_one(pass_mod, tmp)
        test_within_the_tolerance_cer_still_decides(pass_mod, tmp)
        test_a_truncated_draw_cannot_win_by_being_short(pass_mod, tmp)
        test_the_yardstick_ignores_failing_draws(pass_mod, tmp)
        test_a_fast_outlier_does_not_set_the_standard(pass_mod, tmp)
        test_a_long_take_is_still_demoted_against_the_median(pass_mod, tmp)
        test_two_passing_draws_anchor_on_the_shorter(pass_mod, tmp)
        test_when_every_draw_fails_a_gate_cer_alone_decides(pass_mod, tmp)
        test_a_passing_draw_still_beats_a_shorter_failing_one(pass_mod, tmp)
        test_the_knob_turns_the_preference_off(pass_mod, tmp)
        test_the_winner_does_not_carry_the_duration_key(pass_mod, tmp)
    test_length_rank_of_is_inert_without_numbers(pass_mod)
    test_badness_keeps_its_old_ordering_for_readings(pass_mod)
    print("OK: the shorter clean draw wins, a truncated one never does, and the knob is off-able")
