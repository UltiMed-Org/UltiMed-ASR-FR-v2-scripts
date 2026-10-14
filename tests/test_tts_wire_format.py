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
"""The seed and cfg_alpha must travel NESTED in ``extra_params``, on both TTS stages.

This is a regression test for a bug that produced no error at all. vllm-omni's
``OpenAICreateSpeechRequest`` declares no ``model_config``, so pydantic v2's default
``extra='ignore'`` DROPS unknown top-level keys and still answers 200. A flat
``cfg_alpha`` was therefore discarded silently, and a flat ``seed``, while a real
field, only reaches the vLLM sampler, which Voxtral's stage 0 feeds a vocab-wide -inf
row with one finite entry: arithmetically inert. The visible symptom was nil, because
unseeded takes drift anyway (the flow-matching noise came from a global RNG), so five
"different" regeneration draws were five identical requests and nothing ever flagged it.

What the server reads is stage 0's ``SamplingParams.extra_args``, populated from
``extra_params``. Hence the assertions below: the nested keys are what matter, the flat
``seed`` stays for the backends that do read it (TADA, qwen3-tts), and both are omitted
entirely when unset so the server keeps its startup defaults.

No server and no network: only the request bodies are built.

Run: uv run tests/test_tts_wire_format.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TTS_PATH = HERE.parent / "05_generate_audio" / "01_generate_audio.py"
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage05_args(**overrides) -> SimpleNamespace:
    """Only the fields build_body reads."""
    base = dict(
        format="flac", model=None, instruction=None, voice=None, seed=None, speed=None,
        cfg_alpha=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None, do_sample=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_stage05_nests_the_seed() -> None:
    tts = load_module(TTS_PATH, "tts_wire_under_test")

    body = tts.build_body("bonjour", _stage05_args(seed=43))
    assert body["seed"] == 43, "flat seed must stay for TADA / qwen3-tts"
    assert body["extra_params"] == {"voxtral_noise_seed": 43}, (
        "the seed must ALSO be nested, or it never reaches Voxtral's noise draw"
    )

    # Unset knobs stay off the wire so the server keeps its startup defaults.
    plain = tts.build_body("bonjour", _stage05_args())
    assert "seed" not in plain
    assert "extra_params" not in plain, (
        "an empty extra_params would override nothing but still change the request"
    )

    tuned = tts.build_body("bonjour", _stage05_args(seed=43, cfg_alpha=2.0))
    assert tuned["extra_params"] == {"cfg_alpha": 2.0, "voxtral_noise_seed": 43}
    assert "cfg_alpha" not in tuned, "a flat cfg_alpha is dropped by a stock server"
    print("OK stage 05 build_body nests voxtral_noise_seed and cfg_alpha")


def test_stage06_nests_seed_and_cfg_alpha() -> None:
    improve = load_module(IMPROVE_PATH, "improve_wire_under_test")

    sent: dict = {}

    def fake_post_json(url, body, timeout):
        sent["url"] = url
        sent["body"] = body
        return 200, b"audio-bytes", 0.01

    original = improve._tts.post_json
    improve._tts.post_json = fake_post_json
    try:
        payload = improve.tts_synthesize(
            "http://127.0.0.1:8003", "bonjour",
            response_format="flac", seed=44, voice="fr_male", speed=None,
            model=None, timeout=5.0, cfg_alpha=1.6,
        )
    finally:
        improve._tts.post_json = original

    assert payload == b"audio-bytes"
    body = sent["body"]
    assert body["seed"] == 44, "flat seed must stay for TADA / qwen3-tts"
    assert body["extra_params"] == {"cfg_alpha": 1.6, "voxtral_noise_seed": 44}, (
        f"cfg_alpha and the seed must both be nested, got {body.get('extra_params')!r}"
    )
    assert "cfg_alpha" not in body, (
        "a flat cfg_alpha is silently dropped by the server; sending it hides the bug"
    )
    print("OK stage 06 tts_synthesize nests cfg_alpha and voxtral_noise_seed")


def test_stage06_omits_extra_params_when_nothing_is_set() -> None:
    improve = load_module(IMPROVE_PATH, "improve_wire_under_test_2")

    sent: dict = {}

    def fake_post_json(url, body, timeout):
        sent["body"] = body
        return 200, b"x", 0.01

    original = improve._tts.post_json
    improve._tts.post_json = fake_post_json
    try:
        improve.tts_synthesize(
            "http://127.0.0.1:8003", "bonjour",
            response_format="flac", seed=None, voice=None, speed=None,
            model=None, timeout=5.0, cfg_alpha=None,
        )
    finally:
        improve._tts.post_json = original

    assert "extra_params" not in sent["body"]
    print("OK stage 06 omits extra_params when no knob is set")


def test_stage05_token_budget_is_on_by_default_and_sized_from_the_text() -> None:
    """The per-clip frame budget: ON unless disabled, and proportional to the text.

    The default is what this asserts, through the real parser rather than a hand-made
    namespace, because the default IS the decision (2026-10-14): 0.52% of the clips
    generated at cfg_alpha 2.0 run past the budget, so a run that forgets the flag
    writes rambling audio. The budget itself is the one measured in budget_seconds():
    3 s plus one second per 8 characters, at 12.5 frames per second, which for a
    60-character term is 10.5 s (132 frames) instead of the server's 4096.
    """
    tts = load_module(TTS_PATH, "tts_budget_under_test")
    text = "a" * 60

    parsed = tts.build_parser().parse_args(["--input", "x.jsonl", "--output", "out"])
    assert parsed.auto_token_budget is True, "the budget must be on by default"
    assert tts.build_parser().parse_args(
        ["--input", "x.jsonl", "--output", "out", "--no-auto-token-budget"]
    ).auto_token_budget is False, "--no-auto-token-budget must turn it off"

    off = tts.build_body(text, _stage05_args(auto_token_budget=False))
    assert "max_new_tokens" not in off, "disabled means the field is not sent at all"

    args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                         budget_chars_per_second=8.0)
    body = tts.build_body(text, args)
    assert body["max_new_tokens"] == 132, body["max_new_tokens"]
    assert abs(tts.budget_seconds(text, 3.0, 8.0) - 10.5) < 1e-9

    # Clamped at both ends: never above the server's ceiling, never so low that a
    # short text plus a lead-in silence could be cut legitimately.
    assert tts.budget_tokens("x" * 100_000, 3.0, 8.0) == tts.VOXTRAL_DEFAULT_MAX_NEW_TOKENS
    assert tts.budget_tokens("", 0.1, 8.0) == 64
    print("OK stage 05 token budget is on by default and proportional to the text")


def test_stage05_judges_truncation_against_the_budget() -> None:
    """A clip that runs long FOR ITS TEXT must be rejected, not written.

    This is the half that keeps rambling audio out of the dataset: a 60-character
    term that comes back as 30 s of audio stopped for length somewhere, and under
    the global 327.68 s cap alone nothing would notice (observed on the live run at
    cfg_alpha 2.0: 1.68% of dictionary clips over 41.2 s, v1's own maximum, up to
    202 s).
    """
    import wave, io, tempfile

    tts = load_module(TTS_PATH, "tts_budget_truncation")

    def wav_of(seconds: float) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * int(24000 * seconds))
        return buf.getvalue()

    tts.post_json = lambda url, body, timeout: (200, wav_of(30.0), 1.0)
    text = "a" * 60          # budget 10.5 s, so 30 s of audio is a runaway

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.flac"
        args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                             budget_chars_per_second=8.0, timeout=600.0,
                             max_audio_seconds=327.68)
        try:
            tts.synthesize("http://stub", text, out, args)
            raise AssertionError("a 30 s clip for a 10.5 s budget must be rejected")
        except tts.TruncatedAudioError:
            pass
        assert not out.exists(), "nothing may be written, so a resume retries the clip"

        # Same clip, budget off: the global cap is 327.68 s, so 30 s is accepted.
        args.auto_token_budget = False
        tts.synthesize("http://stub", text, out, args)
        assert out.exists()
    print("OK stage 05 rejects a clip that runs long for its own text")


def test_stage05_rejects_an_implausibly_short_clip() -> None:
    """The mirror image: a clip far SHORTER than its text can be read in.

    This failure is as silent as the runaway (stop token emitted early, HTTP 200, a
    clip that sounds complete in isolation), so only the duration-vs-length comparison
    catches it. Checked whether or not the budget is on, since the floor needs nothing
    from the server.
    """
    import io
    import tempfile
    import wave

    tts = load_module(TTS_PATH, "tts_budget_short")

    def wav_of(seconds: float) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * int(24000 * seconds))
        return buf.getvalue()

    # 300 characters in 2 s is 150 chars/s, five times the fastest voice ever measured
    # (21.3 chars/s), so all but a few words went missing.
    text = "a" * 300
    tts.post_json = lambda url, body, timeout: (200, wav_of(2.0), 1.0)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.flac"
        args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                             budget_chars_per_second=5.0, floor_chars_per_second=30.0,
                             timeout=600.0, max_audio_seconds=327.68, retries=0)
        try:
            tts.synthesize("http://stub", text, out, args)
            raise AssertionError("2 s of audio for 300 chars must be rejected")
        except tts.ImplausiblyShortError:
            assert not out.exists(), "nothing may be written, so a resume retries it"

        # The floor is independent of the budget: turning the budget off must not
        # turn it off too, or an unbudgeted run silently keeps truncated clips.
        args.auto_token_budget = False
        try:
            tts.synthesize("http://stub", text, out, args)
            raise AssertionError("the floor must apply with the budget off too")
        except tts.ImplausiblyShortError:
            pass

        # 25 s for 300 chars is 12 chars/s, inside every voice's measured range.
        tts.post_json = lambda url, body, timeout: (200, wav_of(25.0), 1.0)
        tts.synthesize("http://stub", text, out, args)
        assert out.exists()
    print("OK stage 05 rejects a clip too short to hold its own text")


def test_stage05_retries_a_duration_rejection_at_the_same_cfg() -> None:
    """A rejected take is retried in place, and the retry is a genuinely new take.

    The retry is worth GPU time only because the flow-matching noise is fresh per
    request, so re-sending the same text is a new sample rather than a replay. The
    cfg_alpha must NOT change between attempts: the dataset's cfg_alpha column is
    written as one flat value, and nothing records a per-clip one.
    """
    import io
    import tempfile
    import wave

    tts = load_module(TTS_PATH, "tts_budget_retry")

    def wav_of(seconds: float) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * int(24000 * seconds))
        return buf.getvalue()

    text = "a" * 60  # budget 3 + 60/5 = 15 s, floor 60/30 = 2 s
    seen_cfg = []
    calls = {"n": 0}

    def flaky(url, body, timeout):
        calls["n"] += 1
        seen_cfg.append(body.get("extra_params", {}).get("cfg_alpha"))
        # First two takes ramble, the third reads the text in a plausible 4 s.
        return (200, wav_of(40.0 if calls["n"] <= 2 else 4.0), 1.0)

    tts.post_json = flaky
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.flac"
        args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                             budget_chars_per_second=5.0, floor_chars_per_second=30.0,
                             timeout=600.0, max_audio_seconds=327.68, retries=2,
                             cfg_alpha=2.0)
        elapsed, dur, attempts = tts.synthesize_with_retries("http://stub", text, out, args)
        assert attempts == 3, f"expected 3 attempts, got {attempts}"
        assert out.exists() and abs(dur - 4.0) < 0.1
        assert seen_cfg == [2.0, 2.0, 2.0], f"cfg_alpha must not move: {seen_cfg}"

        # Out of retries: the exception reaches the caller, which counts the clip as
        # failed so the next resume picks it up.
        calls["n"] = -10  # every take rambles from here on
        out.unlink()
        args.retries = 1
        try:
            tts.synthesize_with_retries("http://stub", text, out, args)
            raise AssertionError("a clip that fails every attempt must raise")
        except tts.TruncatedAudioError:
            assert not out.exists()
    print("OK stage 05 retries a duration rejection at an unchanged cfg_alpha")


def test_the_bounds_are_per_voice_and_overridable() -> None:
    """Each voice is judged by ITS measured pace, and an explicit flag still wins.

    The point of per-voice bounds is that the same clip gets a different verdict
    depending on who read it: 60 characters in 30 s is 2 chars/s, a third of
    fr_female's natural pace (median 16.4) and therefore a runaway, while h_frm_chf
    (median 10.7) is given more room by its own slower bound. Without this, a bound
    loose enough for the slow voice lets a fast voice ramble for twice as long before
    anything notices.
    """
    import io
    import tempfile
    import wave

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    import frame_budget as fb

    tts = load_module(TTS_PATH, "tts_per_voice")

    # The table itself: every voice's upper bound must sit below its own slow tail and
    # its floor above its own fastest clip, and the fast voices must be bounded more
    # tightly than the slow ones (that is the whole gain over one global band).
    assert fb.rates_for_voice("fr_female")[0] > fb.rates_for_voice("h_frm_chf")[0], (
        "the fastest voice must get the tighter upper bound"
    )
    assert fb.rates_for_voice("not_a_voice") == (fb.SLOW_CHARS_PER_S,
                                                fb.FAST_CHARS_PER_S), (
        "an unlisted voice must fall back to the widest band, never be rejected for "
        "being unlisted"
    )
    assert fb.rates_for_voice("h_frm_chf", slow=8.0)[0] == 8.0, (
        "an explicit --budget-chars-per-second must override the table"
    )

    def wav_of(seconds: float) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * int(24000 * seconds))
        return buf.getvalue()

    text = "a" * 60  # fr_female: 3 + 60/9.9 = 9.1 s; h_frm_chf: 3 + 60/5.3 = 14.3 s
    tts.post_json = lambda url, body, timeout: (200, wav_of(11.0), 1.0)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip.flac"
        # The rate flags are None by default precisely so the table applies; a
        # namespace that sets them to numbers would pin every voice to one band.
        args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                             budget_chars_per_second=None, floor_chars_per_second=None,
                             timeout=600.0, max_audio_seconds=327.68, retries=0)
        try:
            tts.synthesize("http://stub", text, out, args, "fr_female")
            raise AssertionError("11 s for 60 chars is a runaway for fr_female")
        except tts.TruncatedAudioError as e:
            assert "fr_female at 9.9 chars/s" in str(e), f"message must name it: {e}"

        # Same text, same audio, slower voice: inside ITS bound, so it is written.
        tts.synthesize("http://stub", text, out, args, "h_frm_chf")
        assert out.exists()

        # The budget actually asked for also follows the voice, or the server would
        # keep generating long after the client would reject the result.
        fast = tts.build_body(text, args, "fr_female")["max_new_tokens"]
        slow = tts.build_body(text, args, "h_frm_chf")["max_new_tokens"]
        assert fast < slow, f"max_new_tokens must follow the voice: {fast} vs {slow}"
    print("OK the bounds are per voice, and a flag still overrides them")


def test_the_band_covers_the_slowest_and_the_fastest_voice() -> None:
    """The bounds must sit outside every VOICE, not outside the corpus average.

    Measured 2026-10-14 over 28,749 v2 dictionary clips: the ten voices read at
    medians 10.74 (h_frm_chf) to 16.36 (fr_female) chars per second of audio, and
    h_frm_chf's own slow tail reaches 5.58 chars/s at p0.1 once its real runaways are
    excluded. The fastest clip ever seen reads at 21.3 chars/s here and 28.5 over the
    602,813 single-voice v1 clips. A bound inside those numbers rejects a voice for
    its natural pace: at 3 s + chars/8 that cost h_frm_chf 3.05% of its clips against
    0.00-0.31% for the other nine.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
    import frame_budget as fb

    assert fb.SLOW_CHARS_PER_S <= 5.58, (
        f"{fb.SLOW_CHARS_PER_S} chars/s is inside h_frm_chf's slow tail (5.58 at p0.1)"
    )
    assert fb.FAST_CHARS_PER_S >= 28.5, (
        f"{fb.FAST_CHARS_PER_S} chars/s is below the fastest clip measured (28.5)"
    )
    # A 20-character dictionary term and a 300-character document, the two shapes the
    # corpus actually has: both bands must contain every voice's natural duration.
    for chars, slowest, fastest in ((20, 10.74, 16.36), (300, 10.74, 16.36)):
        text = "a" * chars
        assert fb.budget_seconds(text) > chars / slowest, (
            f"{chars} chars read by the slowest voice would be rejected")
        assert fb.floor_seconds(text) < chars / fastest, (
            f"{chars} chars read by the fastest voice would be rejected")
    # And the measured disaster is still caught by a wide margin: 202 s for 317 chars.
    assert fb.classify_duration("a" * 317, 202.2) == "over"
    print("OK the band sits outside the slowest and the fastest voice")


if __name__ == "__main__":
    test_stage05_nests_the_seed()
    test_stage06_nests_seed_and_cfg_alpha()
    test_stage06_omits_extra_params_when_nothing_is_set()
    test_stage05_token_budget_is_on_by_default_and_sized_from_the_text()
    test_stage05_judges_truncation_against_the_budget()
    test_stage05_rejects_an_implausibly_short_clip()
    test_stage05_retries_a_duration_rejection_at_the_same_cfg()
    test_the_band_covers_the_slowest_and_the_fastest_voice()
    test_the_bounds_are_per_voice_and_overridable()
    print("OK all")
