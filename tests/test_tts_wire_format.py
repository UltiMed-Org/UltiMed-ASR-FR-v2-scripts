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


def test_stage05_token_budget_is_opt_in_and_sized_from_the_text() -> None:
    """The per-clip frame budget: off by default, and proportional when on.

    It must stay off unless asked for, because it changes what the server generates
    for every clip of a 600k-row run. When on, the budget is the one measured in
    budget_seconds(): 3 s plus one second per 8 characters, at 12.5 frames per
    second, which for a 60-character term is 10.5 s (132 frames) instead of the
    server's global 4096 frames.
    """
    tts = load_module(TTS_PATH, "tts_budget_under_test")
    text = "a" * 60

    plain = tts.build_body(text, _stage05_args())
    assert "max_new_tokens" not in plain, "the budget must be opt-in"

    args = _stage05_args(auto_token_budget=True, budget_fixed_seconds=3.0,
                         budget_chars_per_second=8.0)
    body = tts.build_body(text, args)
    assert body["max_new_tokens"] == 132, body["max_new_tokens"]
    assert abs(tts.budget_seconds(text, 3.0, 8.0) - 10.5) < 1e-9

    # Clamped at both ends: never above the server's ceiling, never so low that a
    # short text plus a lead-in silence could be cut legitimately.
    assert tts.budget_tokens("x" * 100_000, 3.0, 8.0) == tts.VOXTRAL_DEFAULT_MAX_NEW_TOKENS
    assert tts.budget_tokens("", 0.1, 8.0) == 64
    print("OK stage 05 token budget is opt-in and proportional to the text")


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


if __name__ == "__main__":
    test_stage05_nests_the_seed()
    test_stage06_nests_seed_and_cfg_alpha()
    test_stage06_omits_extra_params_when_nothing_is_set()
    test_stage05_token_budget_is_opt_in_and_sized_from_the_text()
    test_stage05_judges_truncation_against_the_budget()
    print("OK all")
