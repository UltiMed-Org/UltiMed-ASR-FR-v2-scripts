#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31","jiwer>=3.0","tqdm>=4.66","loguru>=0.7","click>=8.1","soundfile>=0.12"]
# ///
"""What each regeneration draw actually asks the TTS server for.

Run: `uv run tests/test_tts_request_knobs.py`

This exists because the knobs failed SILENTLY. A TTS server takes a request that sets
`cfg_alpha`, answers 200, logs nothing, and synthesizes at its own startup value anyway
if the field is spelled the way it is not read; the flat `seed` was likewise trusted to
vary the audio when on Voxtral it cannot. Five "different" draws were then five
identical requests whose audio differed only through global-RNG drift, and the cfg ramp
in `_synthesize_draft` had no effect at all. Nothing about the resulting wav says so. So
the request BODY is pinned here, not just the arithmetic that feeds it.

Where the knobs ride is the whole subject, so read this before changing an assertion.
Both travel NESTED, in `extra_params`: `cfg_alpha` and `voxtral_noise_seed`. That is the
route a STOCK vllm-omni reads too, because it merges a request's `extra_params` dict into
stage 0's `SamplingParams.extra_args`, which is exactly where the model looks
(`_extract_cfg_alpha`). The flat spellings are not an equivalent choice: a flat
`cfg_alpha` only works against the forked vllm-omni the CrispASR voxtral-tts image is
built from (`CrispASR/voxtral/vllm-omni`, which promotes it to a real field), and
upstream's request model declares no `model_config`, so pydantic's `extra='ignore'` drops
it and still answers 200. Nesting therefore works on both servers and flat does not,
which is why stage 05's `build_body` and this stage's `tts_synthesize` agree on it
(`tests/test_tts_wire_format.py` pins the same shape from the other side).

The flat `seed` is still sent, and for a different reason: TADA and qwen3-tts read it,
while on Voxtral it is arithmetically inert (stage 0 hands vLLM's sampler a vocab-wide
-inf row with one finite entry), so `voxtral_noise_seed` in `extra_params` is the copy
that actually pins Voxtral's per-frame noise. Both spellings are asserted below;
dropping either silently de-seeds one backend.

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import re
import threading
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"
DRIVER_PATH = HERE.parent / "06_hotfixes" / "driver.sh"

_spec = importlib.util.spec_from_file_location("improve_under_test", IMPROVE_PATH)
_imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_imp)

_failures: list[str] = []


def eq(label: str, got, want) -> None:
    if got != want:
        _failures.append(f"{label}: got {got!r} want {want!r}")


def _capture_bodies(n_improv: int, **overrides) -> list[dict]:
    """Run n_improv drafts with the network stubbed, returning the request bodies."""
    bodies: list[dict] = []

    def fake_post(url, body, timeout):  # signature of the real post_json
        bodies.append(dict(body))
        return 200, b"FAKEAUDIO" + bytes([len(bodies)]), 0.5

    cfg = SimpleNamespace(
        start_seed=43, tts_speed=None, speed_jitter=0.0,
        cfg_alpha_start=1.0, cfg_alpha_step=1.0, cfg_alpha_floor=None, n_improv=n_improv,
        tts_url="http://localhost:8003", tts_format="flac", tts_voice="fr_female",
        tts_model=None, tts_timeout=30.0, tts_max_retries=1,
        max_audio_seconds=327.68, tts_sem=threading.Semaphore(1),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)

    # Both live on the stage-05 module this stage imports as `_tts` (the HTTP call and
    # the frame-cap duration probe exist once, shared by the two stages).
    orig_post, orig_dur = _imp._tts.post_json, _imp._tts.payload_duration
    _imp._tts.post_json = fake_post
    _imp._tts.payload_duration = lambda payload: 12.0    # never near the frame cap
    try:
        for attempt in range(n_improv):
            _imp._synthesize_draft(attempt, cfg, "clip", "bonjour", 999, "nohash", set())
    finally:
        _imp._tts.post_json, _imp._tts.payload_duration = orig_post, orig_dur
    return bodies


# --- the guidance ladder ------------------------------------------------------------
# Driven by the DRIVER's defaults, not by a copy of them: the ladder is a tuning choice
# that lives in driver.sh, and a test that restated it here would keep passing after the
# driver changed. Parsed, fed to the client, and the rendered ladder is what is pinned.
def _driver_default(name: str) -> float:
    text = DRIVER_PATH.read_text(encoding="utf-8")
    # The leading `-?` is load-bearing: CFG_ALPHA_STEP is negative now (the ladder
    # descends), and a pattern that only matched digits would silently find nothing.
    m = re.search(rf'^{name}="\$\{{{name}:-(-?[0-9.]+)\}}"', text, re.MULTILINE)
    assert m, f"{name} default not found in driver.sh"
    return float(m.group(1))


bodies = _capture_bodies(5, cfg_alpha_start=_driver_default("CFG_ALPHA_START"),
                         cfg_alpha_step=_driver_default("CFG_ALPHA_STEP"),
                         cfg_alpha_floor=_driver_default("CFG_ALPHA_FLOOR"))
eq("one request per draw", len(bodies), 5)


def _alpha(body: dict):
    """The cfg_alpha the server will read: nested, see the module docstring."""
    return (body.get("extra_params") or {}).get("cfg_alpha")


# DESCENDING, from the value the corpus was generated at (the CrispASR voxtral-tts
# service boots at cfg_alpha 2.0) down to the measured floor of the usable band. Not a
# wide sweep: the draws have to land where the model is usable, and the winner is
# picked on CER anyway. The direction is the assertion: a clip that derailed at 2.0 is
# over-guided, and above the generation value takes distort and stop terminating.
eq("cfg_alpha ladder descends to the floor",
   [_alpha(b) for b in bodies], [2.0, 1.8, 1.6, 1.4, 1.2])
eq("seed ladder", [b.get("seed") for b in bodies], [43, 44, 45, 46, 47])

# --- where the server actually reads them -------------------------------------------
# NESTED in extra_params, both of them, which is the only route a stock vllm-omni acts
# on: keep these assertions, they are the whole point of the file. A flat cfg_alpha is
# the failure that made the ramp inert, and it is invisible from the client side (200,
# no log, audio at the startup default).
first = bodies[0]
eq("cfg_alpha rides inside extra_params", _alpha(first), 2.0)
eq("cfg_alpha is not sent flat", "cfg_alpha" in first, False)
eq("the noise seed rides inside extra_params too",
   (first.get("extra_params") or {}).get("voxtral_noise_seed"), 43)
# Kept flat as well, for the backends that do read it (TADA, qwen3-tts).
eq("seed is also sent flat", first.get("seed"), 43)
eq("text is sent as input", first.get("input"), "bonjour")
eq("voice is forwarded", first.get("voice"), "fr_female")

# --- unset knobs stay off -----------------------------------------------------------
# Same contract as stage 05: a knob nobody set is omitted so the server keeps its
# startup default, rather than being pinned to a client-side guess.
eq("speed omitted when jitter is off", "speed" in first, False)
eq("model omitted when unset", "model" in first, False)
# A negative start is the documented way to leave guidance entirely to the server.
off = _capture_bodies(2, cfg_alpha_start=None)
eq("cfg_alpha omitted when disabled", [_alpha(b) for b in off], [None, None])
# The seed still has to reach the server when guidance is off, so extra_params must not
# vanish with cfg_alpha: it is the dict that carries voxtral_noise_seed.
eq("the noise seed survives cfg_alpha being off",
   [(b.get("extra_params") or {}).get("voxtral_noise_seed") for b in off], [43, 44])

# --- the floor is a clamp, not a suggestion -----------------------------------------
# --n-improv is free, so a descending ladder runs off the bottom of the usable band on a
# long run: ten draws at -0.2 from 2.0 would reach 0.2, and 1.0 is already CFG off,
# where the decoder stops following the text. The clamp is what keeps a long redo from
# spending its last draws there.
deep = _capture_bodies(10, cfg_alpha_start=2.0, cfg_alpha_step=-0.2, cfg_alpha_floor=1.2)
eq("the ladder stops at the floor", [_alpha(b) for b in deep],
   [2.0, 1.8, 1.6, 1.4, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
eq("no draw is ever at or below CFG-off", [a for a in (_alpha(b) for b in deep) if a <= 1.0], [])

# --- speed jitter, if someone turns it on -------------------------------------------
# It cycles around the base, but note it is NOT a second source of variation on this
# backend: vllm-omni applies speed as a phase vocoder over the finished waveform.
jit = _capture_bodies(3, tts_speed=1.0, speed_jitter=0.05)
eq("speed cycles around the base", [round(b["speed"], 4) for b in jit], [1.0, 1.05, 0.95])

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
