"""Tests for 06_hotfixes/04_flag_asr_defects.py: its ``find_defect`` doctests hold the
cases the rules were tuned on (a TTS preamble, a skip both recognizers confirm, and the
false positives a glued code / compound / reworded stretch produced), plus the two input
shapes ``load_hyps`` has to read the same way, which is what lets the flag list be
rebuilt from the sweep the shipped audio already went through instead of from a
transcription pass whose audio may since have been replaced.

    uv run --with click --with loguru tests/test_flag_asr_defects.py

This file was written by Claude Code.
"""
import doctest
import importlib.util
import json
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "06_hotfixes" / "04_flag_asr_defects.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("flag_asr_defects", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_find_defect() -> None:
    mod = _load()
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (failed, attempted)
    print(f"test_find_defect: OK ({attempted} cases)")


def test_load_hyps_accepts_both_shapes() -> None:
    """A hypotheses file and a state file must come out as the same mapping.

    The state branch is the one that matters in practice: it is how the list is rebuilt
    on regenerated audio, and a silent empty result there would hand `04_drop_bad_rows`
    an unchanged, stale list.
    """
    mod = _load()
    judge = mod.DEFAULT_JUDGE_KEY
    key = "PARHAF/000001_0000_x.flac"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)

        hyps = tmp / "hyps.jsonl"
        hyps.write_text(json.dumps({"audio": f"/anywhere/else/{key}", "hyp": "bonjour"}) + "\n",
                        encoding="utf-8")
        assert mod.load_hyps(hyps) == {key: "bonjour"}, mod.load_hyps(hyps)

        state = tmp / "full.stt.jsonl"
        state.write_text("".join(json.dumps(r) + "\n" for r in [
            {"audio_filepath": f"../{key.split('/')[0]}/{key.split('/')[1]}",
             "transcriptions": {judge: {"text": "bonjour"},
                                "whisper.cpp/ggml-large-v3-turbo": {"text": "au revoir"}}},
            # No reading from that judge yet: a resumable sweep's normal mid-state, and
            # it must be skipped rather than stored as an empty hypothesis (which would
            # read as "the recognizer heard nothing" and flag the clip as a total skip).
            {"audio_filepath": "../PARHAF/000002_0000_y.flac",
             "transcriptions": {"whisper.cpp/ggml-large-v3-turbo": {"text": "au revoir"}}},
        ]), encoding="utf-8")
        got = mod.load_hyps(state)
        assert got == {key: "bonjour"}, got
        # The wrong key must come back empty, not come back with Whisper's reading: that
        # is the one recognizer whose blindness this whole script works around.
        assert mod.load_hyps(state, "nope/not-a-judge") == {}
    print("test_load_hyps_accepts_both_shapes: OK")


def test_default_reasons_exclude_skipped() -> None:
    """The default rule set must be `preamble` alone, and `skipped` must still work.

    Pinned because the default is a release decision, not a style choice: the module
    docstring's 2026-10-26 measurement found `skipped` firing on dosage enumerations
    where the clip's own CER (0.0049) and bad_run_chars (1) rule a real omission out, so
    leaving it on would drop ~520 good long PARHAF clips. A future change that flips it
    back on by accident has to fail here.
    """
    mod = _load()
    assert mod.DEFAULT_REASONS == (mod.PREAMBLE,), mod.DEFAULT_REASONS

    label, hyp = "le patient est sorti hier soir", "le patient"
    # The rule itself is unchanged: asked for, it still fires.
    assert mod.find_defect(label, hyp, hyp, (mod.SKIPPED,))[0] == mod.SKIPPED
    # Not asked for, it does not, and the preamble rule is unaffected either way.
    assert mod.find_defect(label, hyp, hyp, mod.DEFAULT_REASONS) is None
    assert mod.find_defect("Courrier de suivi.", "Jackie Orbad dit. Courrier de suivi.",
                           None, mod.DEFAULT_REASONS)[0] == mod.PREAMBLE
    # And a clip with no Whisper transcript must not crash the skipped branch.
    assert mod.find_defect(label, hyp, None, (mod.SKIPPED,)) is None
    print("test_default_reasons_exclude_skipped: OK")


if __name__ == "__main__":
    test_find_defect()
    test_load_hyps_accepts_both_shapes()
    test_default_reasons_exclude_skipped()
