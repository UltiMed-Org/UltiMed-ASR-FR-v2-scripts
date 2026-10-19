#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""The windowed scoring in `06_hotfixes/01_compute_stt.py`, i.e. the blind spot the
whole-clip CER has on long clips.

The defect this guards against is a voice that skips a phrase in the MIDDLE of a long
clip, which is the v1.3 failure mode (14 clips retired for a skipped phrase, mostly
inside long lab-value lists, none of them caught by the v1 QC). A whole-clip CER cannot
see it: five lost words are ~30 normalized characters, which is 0.106 of a 20 s clip
(over the 0.08 gate) but 0.044 of a 48 s PARHAF chunk (under it). The existing tail CER
does not help either, since it only isolates the LAST 30 s of a clip.

So the properties worth pinning are not "the metric has value X" but:
  * a mid-clip dropped phrase that the whole-clip gate misses IS flagged by the window;
  * the window score of one defect barely moves when the clip around it gets longer,
    while the whole-clip score collapses (that length-independence is the whole point);
  * `bad_run_chars` measures missing REFERENCE text, so a loop (inserted text) leaves it
    at zero while still showing up in the window CER;
  * the window is a restriction of the CER, never a different number: it is >= the
    whole-clip CER, and exactly equal to it when the reference fits in one window.

    uv run tests/test_window_cer.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STT_PATH = HERE.parent / "06_hotfixes" / "01_compute_stt.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stt = load_module(STT_PATH, "compute_stt_under_test")

# A plausible French medical sentence, repeated to build a reference of any length. It is
# deliberately NOT random text: `normalize_for_scoring` runs first, and the metric has to
# behave on text that survives its number/unit rules.
SENTENCE = (
    "le patient presente une dyspnee aux efforts moderes avec une toux seche nocturne "
    "et une asthenie marquee depuis trois semaines sans fievre ni amaigrissement "
)

CER_GATE = 0.08           # 03_qc_cpu_stt.py DEFAULT_CER_GATE
WINDOW_GATE = 0.15        # 03_qc_cpu_stt.py DEFAULT_WINDOW_CER_GATE
RUN_GATE = 30             # 03_qc_cpu_stt.py DEFAULT_BAD_RUN_GATE


def reference(seconds: float) -> str:
    """A reference whose normalized length is about `seconds` of speech."""
    want = int(seconds * stt.NORM_CHARS_PER_SECOND)
    text = SENTENCE
    while len(stt.normalize_for_scoring(text)) < want:
        text += SENTENCE
    return text.strip()


def drop_words(text: str, at: int, count: int) -> str:
    """The hypothesis a voice produces when it skips `count` words mid-sentence."""
    words = text.split()
    return " ".join(words[:at] + words[at + count:])


def test_mid_clip_drop_escapes_the_whole_clip_gate_but_not_the_window() -> None:
    ref = reference(48)  # a PARHAF-sized chunk
    hyp = drop_words(ref, at=len(ref.split()) // 2, count=5)
    _, cer = stt.compute_metrics(ref, hyp)
    window_cer, run = stt.window_metrics(ref, hyp)
    # The premise: this is exactly the defect the live gate cannot see.
    assert cer < CER_GATE, f"premise broken, the whole-clip gate already flags it ({cer})"
    assert stt.tail_cer(ref, hyp, 48.0) is not None
    # The fix: both new metrics catch it.
    # The run gate is the one that catches a clean phrase drop: five French words are
    # ~30 characters, and no clean clip in the 20k calibration reached 27. The window
    # CER lands just under its own (looser) gate here, which is why both exist.
    assert run >= RUN_GATE, run
    # ...and the window CER is 1.7x the whole-clip gate on the very clip whose own
    # CER sits at half of it, which is the dilution this metric undoes.
    assert window_cer > 1.5 * CER_GATE, window_cer
    print(f"  48s clip, 5 words dropped mid-clip: cer {cer:.4f} (silent), "
          f"window_cer {window_cer:.4f}, run {run}")


def test_window_score_is_length_independent() -> None:
    """The same defect in a longer clip: the whole-clip CER dilutes, the window does not."""
    scores = {}
    for seconds in (15, 30, 48, 68, 120):
        ref = reference(seconds)
        hyp = drop_words(ref, at=len(ref.split()) // 2, count=5)
        _, cer = stt.compute_metrics(ref, hyp)
        window_cer, run = stt.window_metrics(ref, hyp)
        scores[seconds] = (cer, window_cer, run)
        print(f"  {seconds:3d}s: cer {cer:.4f}  window_cer {window_cer:.4f}  run {run}")
    cers = [s[0] for s in scores.values()]
    windows = [s[1] for s in scores.values()]
    runs = [s[2] for s in scores.values()]
    # The whole-clip CER falls by more than 5x across that range: the dilution, measured.
    assert cers[0] / cers[-1] > 5, cers
    # The window CER stays inside a +/-25% band across a 8x change in clip length,
    # which is the length-independence this metric exists for.
    assert max(windows) / min(windows) < 1.25, windows
    # And it never dilutes under the whole-clip gate the way the CER does.
    assert min(windows) > CER_GATE, windows
    # The missing run is the length of the dropped phrase, whatever the clip length.
    assert max(runs) - min(runs) <= 4, runs


def test_the_excerpt_shows_both_sides_of_the_worst_window() -> None:
    """The numbers say a clip is bad; these fields say what about it is bad, which is
    what the QC log prints so a flag can be judged without opening the audio."""
    ref = reference(48)
    words = ref.split()
    cut = len(words) // 2
    dropped = " ".join(words[cut:cut + 5])
    hyp = drop_words(ref, at=cut, count=5)
    s = stt.window_score(ref, hyp)
    # The missing span IS the dropped phrase (normalized, so "trois" is "3" here), and
    # nothing was heard in its place: that pair is what separates a skip from a misread.
    assert s.missing.strip() == stt.normalize_for_scoring(dropped).strip(), s.missing
    assert s.missing_heard == "", s.missing_heard
    assert f">>>{s.missing}<<<" in s.missing_context, s.missing_context
    # The window is shown at its full width and the expected text contains the span.
    assert len(s.ref_window) == stt.WINDOW_CHARS, len(s.ref_window)
    assert s.missing in s.ref_window
    # The heard text covers the SAME stretch of audio and is shorter by exactly what
    # went missing. (Not asserting the span is absent from `hyp_window`: this fixture
    # repeats one sentence, so the phrase legitimately occurs again elsewhere in the
    # window. The length relation is the invariant that holds on any text.)
    assert len(s.ref_window) - len(s.hyp_window) == s.bad_run_chars
    print(f"  missing {s.bad_run_chars} chars: {s.missing!r}")


def test_the_excerpt_shows_a_loop_as_text_it_did_not_ask_for() -> None:
    ref = reference(40)
    words = ref.split()
    cut = len(words) // 2
    looped = " ".join(words[:cut] + ["seche nocturne"] * 12 + words[cut:])
    s = stt.window_score(ref, looped)
    # Nothing missing, but the recognizer produced far more text over the same stretch,
    # which is the shape of a loop in these two fields.
    assert s.missing == "" and s.bad_run_chars == 0
    assert len(s.hyp_window) > 1.5 * len(s.ref_window), (len(s.hyp_window),
                                                         len(s.ref_window))


def test_the_two_entry_points_cannot_disagree() -> None:
    """`window_metrics` is a reduction of `window_score`, not a second implementation."""
    ref = reference(30)
    hyp = drop_words(ref, at=12, count=4)
    s = stt.window_score(ref, hyp)
    assert stt.window_metrics(ref, hyp) == (s.window_cer, s.bad_run_chars)
    assert stt.window_metrics("", "x") == (None, None)


def test_clean_hypothesis_scores_zero() -> None:
    ref = reference(40)
    assert stt.window_metrics(ref, ref) == (0.0, 0)
    # And normalization still applies: a different spelling of the same thing is clean.
    assert stt.window_metrics("2 mg par jour", "deux milligrammes par jour")[0] == 0.0


def test_a_loop_shows_in_the_window_but_not_in_the_missing_run() -> None:
    """`bad_run_chars` is about reference text that went missing, so inserted text must
    leave it at zero. The window CER still has to see the loop, which is what keeps the
    two metrics complementary rather than redundant."""
    ref = reference(40)
    words = ref.split()
    cut = len(words) // 2
    looped = " ".join(words[:cut] + ["seche nocturne"] * 12 + words[cut:])
    window_cer, run = stt.window_metrics(ref, looped)
    assert run == 0, run
    assert window_cer > 0.5, window_cer


def test_window_is_a_restriction_of_the_cer() -> None:
    """Never a different number: >= the whole-clip CER on anything, and exactly equal
    when the reference fits inside one window (nothing to undo)."""
    short = "le patient tousse depuis trois jours"
    assert len(stt.normalize_for_scoring(short)) < stt.WINDOW_CHARS
    _, cer = stt.compute_metrics(short, "le patient touche depuis trois jours")
    window_cer, _ = stt.window_metrics(short, "le patient touche depuis trois jours")
    assert abs(window_cer - cer) < 1e-9, (window_cer, cer)

    rng = random.Random(11)
    ref = reference(60)
    words = ref.split()
    for _ in range(40):
        hyp = list(words)
        for _ in range(rng.randint(1, 6)):
            at = rng.randrange(len(hyp))
            kind = rng.choice(("drop", "swap", "insert"))
            if kind == "drop":
                del hyp[at:at + rng.randint(1, 4)]
            elif kind == "swap":
                hyp[at] = "quelquechose"
            else:
                hyp.insert(at, "tousse")
        _, cer = stt.compute_metrics(ref, " ".join(hyp))
        window_cer, _ = stt.window_metrics(ref, " ".join(hyp))
        assert window_cer >= cer - 1e-9, (window_cer, cer)


def test_empty_reference_is_not_scored() -> None:
    assert stt.window_metrics("", "le patient tousse") == (None, None)
    assert stt.window_metrics("   ", "") == (None, None)


def _entry_for(rec: dict, transcript: str) -> dict:
    """`compute_entry` on one record, with the HTTP call and the file probe stubbed.

    Both are module globals in `01_compute_stt.py`, so replacing them is enough; the
    scoring under test is the real thing.
    """
    original = (stt.resolve_audio, stt.transcribe_timed)
    stt.resolve_audio = lambda *a, **k: Path("clip.flac")
    stt.transcribe_timed = lambda *a, **k: (transcript, 1.0)
    try:
        return stt.compute_entry(
            rec, Path("manifest.jsonl"),
            endpoint="http://127.0.0.1:8002/v1/audio/transcriptions", token="",
            model=None, temperature=0.0, language="fr", response_format="json",
            extra_params={}, timeout=1.0, max_retries=0,
            audio_key="audio_filepath", reference_key="asr_training_source",
            audio_root=None, wer_threshold=1.0, cer_threshold=1.0,
        )
    finally:
        stt.resolve_audio, stt.transcribe_timed = original


def test_compute_entry_stores_the_window_scores() -> None:
    """The sweep has to WRITE the two window scores, not merely be able to compute them.

    Stage 05's live QC (`03_qc_cpu_stt.py`) has stored `window_cer` and `bad_run_chars`
    on every row it scores from the start. Stage 06's sweep stored `cer` and `cer_tail`
    only, so the repair pass had nothing to gate on and every other consumer had to
    re-derive them from the transcript, at a measured ~11 ms per clip each time.
    """
    ref = reference(48)  # a PARHAF-sized chunk
    hyp = drop_words(ref, at=len(ref.split()) // 2, count=5)
    entry = _entry_for(
        {"audio_filepath": "clip.flac", "duration": 48.0, "asr_training_source": ref}, hyp
    )
    expected_window, expected_run = stt.window_metrics(ref, hyp)
    assert entry["cer_window"] == expected_window, entry
    assert entry["bad_run_chars"] == expected_run, entry
    # Why storing them is what matters: this clip passes the whole-clip gate it is
    # scored against, so with only `cer` and `cer_tail` on the row nothing downstream
    # could tell it apart from a clean one.
    assert entry["cer"] < CER_GATE, entry["cer"]
    assert entry["bad_run_chars"] >= RUN_GATE, entry


def test_compute_entry_omits_the_window_scores_when_there_is_no_reference() -> None:
    """An unscorable row must not grow the keys with empty values: `02_statistics.py`
    treats "key present and numeric" as the definition of a scored row, so a None here
    would be counted as coverage the sweep never had."""
    entry = _entry_for({"audio_filepath": "clip.flac", "duration": 12.0}, "quelque chose")
    assert "cer_window" not in entry, entry
    assert "bad_run_chars" not in entry, entry
    assert entry["error"], "a row with no reference must say why it was not scored"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n{fn.__name__}")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        else:
            print("  ok")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
