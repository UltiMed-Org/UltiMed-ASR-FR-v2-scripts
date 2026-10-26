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


def _capture_bodies(n_improv: int, *, clip_voice: str | None = None, **overrides) -> list[dict]:
    """Run n_improv drafts with the network stubbed, returning the request bodies.

    `clip_voice` is the PER-CLIP voice, the one `_synthesize_draft` takes as an argument
    and the one the driver passes from the voice assignment; `tts_voice` in the overrides
    is only the fallback for a clip that carries none. The two differ on purpose, since
    the cfg_alpha floor has to follow the voice that is actually synthesized.
    """
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
            _imp._synthesize_draft(attempt, cfg, "clip", "bonjour", 999, "nohash", set(),
                                   voice=clip_voice)
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

# --- and the floor is PER VOICE ------------------------------------------------------
# The 2026-10-13 audition measured the usable band as 1.2-2.0 on the two shipped `fr_*`
# voices but 1.5-2.0 on the hybrids, and seven of the nine v2 voices are hybrids. With a
# single global 1.2 a hybrid clip could be "repaired" by a draw two rungs below its own
# floor, and nothing downstream would notice: the winner is picked on CER, and CER cannot
# hear the articulation the audition rejected that band for. So the ladder asks
# utils/voice_assignment.py, the one home of those numbers.
import sys  # noqa: E402

sys.path.insert(0, str(HERE.parent / "utils"))
import voice_assignment as _va  # noqa: E402

hyb = _capture_bodies(10, clip_voice="h_frf_itm",
                      cfg_alpha_start=2.0, cfg_alpha_step=-0.2, cfg_alpha_floor=1.2)
eq("a hybrid clip stops at its own 1.5, not the global 1.2", [_alpha(b) for b in hyb],
   [2.0, 1.8, 1.6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
eq("the hybrid floor is the one from voice_assignment", _va.FLOOR_HYBRID, 1.5)
# The per-clip voice wins over cfg.tts_voice, which is only the fallback. This is the
# shape the driver actually runs: one --tts-voice on the command line, the real voice per
# clip out of the assignment, so reading the fallback would floor every clip at 1.2.
eq("the per-clip voice is what sets the floor",
   _alpha(_capture_bodies(5, clip_voice="h_itf_arm", cfg_alpha_start=2.0,
                          cfg_alpha_step=-0.2, cfg_alpha_floor=1.2,
                          tts_voice="fr_female")[-1]), 1.5)
# An unknown voice gets the STRICTER floor, because a floor set too high only loses a rung
# of the retry ladder while one set too low ships audio an audition already called unusable.
eq("an unknown voice is treated as a hybrid",
   _alpha(_capture_bodies(5, clip_voice="h_future_voice", cfg_alpha_start=2.0,
                          cfg_alpha_step=-0.2, cfg_alpha_floor=1.2)[-1]), 1.5)
# `max` composes the two, so an explicitly stricter --cfg-alpha-floor still wins on a base
# voice (otherwise the flag would be silently capped at 1.2 and sweeps could not raise it),
eq("an explicit stricter floor still wins", [_alpha(b) for b in
   _capture_bodies(5, cfg_alpha_start=2.0, cfg_alpha_step=-0.2, cfg_alpha_floor=1.8)],
   [2.0, 1.8, 1.8, 1.8, 1.8])
# and a negative floor (which reaches here as None, see --cfg-alpha-floor's help) still
# removes the clamp entirely, hybrid or not: it is the documented escape hatch for a sweep.
eq("a removed floor stays removed on a hybrid too", [_alpha(b) for b in
   _capture_bodies(5, clip_voice="h_frf_itm", cfg_alpha_start=2.0, cfg_alpha_step=-0.2,
                   cfg_alpha_floor=None)],
   [2.0, 1.8, 1.6, 1.4, 1.2])

# --- speed jitter, if someone turns it on -------------------------------------------
# It cycles around the base, but note it is NOT a second source of variation on this
# backend: vllm-omni applies speed as a phase vocoder over the finished waveform.
jit = _capture_bodies(3, tts_speed=1.0, speed_jitter=0.05)
eq("speed cycles around the base", [round(b["speed"], 4) for b in jit], [1.0, 1.05, 0.95])

# --- the v2 generation guards, on a redraw ------------------------------------------
# A redo used to send cfg_alpha and the noise seed and nothing else, so a looping draw
# ran to the server's 327.68 s cap: ~6 minutes of GPU to produce a take the duration
# check then threw away, five times per clip. These sections pin that each draw now
# carries the same per-clip frame budget the generation pass sends, and that the two
# cheap rejections (duration band, code gate) happen BEFORE the draw costs an STT call.
import array  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import wave  # noqa: E402

# Imported by name, not by path: code_loops defines dataclasses, and a module loaded by
# path is absent from sys.modules, which @dataclass needs to resolve its own
# annotations. These are also the instances the stages themselves import, so the
# numbers asserted here are the ones that actually gate.
sys.path.insert(0, str(HERE.parent / "utils"))
import code_loops  # noqa: E402
import code_sidecar  # noqa: E402
import frame_budget as fb  # noqa: E402

# OFF unless asked for, which is deliberate and worth a line: _synthesize_draft reads
# the flag with getattr defaulting to False, exactly as stage 05's build_body does, so a
# caller that never asked for a budget (a test, anything outside main()) keeps sending
# what it used to. main() passes --auto-token-budget, whose click default is True.
eq("no budget is sent unless the pass asks for one", "max_new_tokens" in bodies[0], False)

budgeted = _capture_bodies(2, auto_token_budget=True)
# The number is not restated here: it comes from utils/frame_budget.py, through the
# VOICE's measured band (fr_female reads at 9.9 chars/s at its slow end, where the
# unlisted-voice fallback is 5.0), which is the same call stage 05 makes. A copy of the
# arithmetic in this file would keep passing after a voice was re-measured.
_slow, _fast = fb.rates_for_voice("fr_female")
eq("the frame budget rides flat, where vllm-omni reads it",
   budgeted[0].get("max_new_tokens"), fb.budget_tokens("bonjour", fb.BUDGET_FIXED_S, _slow))
eq("every draw carries it", [b.get("max_new_tokens") for b in budgeted],
   [fb.budget_tokens("bonjour", fb.BUDGET_FIXED_S, _slow)] * 2)
# Flat, not nested: unlike cfg_alpha, max_new_tokens is applied by a generic vllm-omni
# override gated on _SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES (which includes voxtral_tts),
# after prompt building. Nesting it in extra_params would be silently ignored.
eq("the budget is not sent nested", "max_new_tokens" in (budgeted[0].get("extra_params") or {}),
   False)

# The code gate's tag: nested (only the patched fork reads it) and unique per ATTEMPT,
# because the server APPENDS to <tag>.i16. A shared tag would read attempt 2 as attempt 1
# concatenated with itself and condemn a clean draw.
with tempfile.TemporaryDirectory() as _td:
    tagged = _capture_bodies(3, code_dump_dir=_td)
_tags = [(b.get("extra_params") or {}).get("voxtral_dump_codes") for b in tagged]
eq("a dump tag per draw, nested", _tags, ["clip__redo0s43", "clip__redo1s44", "clip__redo2s45"])
eq("the tags are distinct", len(set(_tags)), 3)
eq("no dump tag when the gate is off",
   (bodies[0].get("extra_params") or {}).get("voxtral_dump_codes"), None)


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x00\x00" * int(24000 * seconds))
    return buf.getvalue()


def _write_dump(path: Path, track: list[int]) -> None:
    """A dump as the server writes it: 37 int16 per frame, semantic in column 0."""
    rows = [[c] + [0] * 36 for c in track] + [[code_loops.SEMANTIC_END_AUDIO] + [0] * 36]
    path.write_bytes(array.array("h", [v for r in rows for v in r]).tobytes())


def _guarded_draft(text: str, seconds: float, *, track: list[int] | None = None,
                   attempt: int = 0, **overrides) -> tuple[dict, Path]:
    """One draw against a stub that answers `seconds` of audio (and writes `track` as
    its code dump when given). Returns the draft dict and the clip's directory, so the
    caller can read the code-QC sidecar the draw left beside it."""
    tmp = Path(tempfile.mkdtemp())
    dumps = tmp / "dumps"
    dumps.mkdir()
    audio = tmp / "000001_0001_terme.flac"
    audio.write_bytes(b"the original clip")
    cfg = SimpleNamespace(
        start_seed=43, tts_speed=None, speed_jitter=0.0,
        cfg_alpha_start=2.0, cfg_alpha_step=-0.2, cfg_alpha_floor=1.2, n_improv=5,
        tts_url="http://localhost:8003", tts_format="flac", tts_voice="fr_female",
        tts_model=None, tts_timeout=30.0, tts_max_retries=1,
        max_audio_seconds=327.68, tts_sem=threading.Semaphore(1),
        auto_token_budget=True, code_dump_dir=(str(dumps) if track else None),
        store_semantic_codes=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)

    def fake_post(url, body, timeout):
        if track is not None:
            _write_dump(dumps / f"{body['extra_params']['voxtral_dump_codes']}.i16", track)
        return 200, _wav(seconds), 1.0

    orig_post = _imp._tts.post_json
    _imp._tts.post_json = fake_post
    try:
        draft = _imp._synthesize_draft(attempt, cfg, "clip", text, 999, "nohash", set(),
                                       audio_path=audio)
    finally:
        _imp._tts.post_json = orig_post
    return draft, tmp


# 60 characters read by fr_female: plausible between 2.02 s (its fastest take ever
# measured, times 1.45) and 9.06 s (3 s of head/tail plus 60/9.9). The three draws below
# sit above, below and inside that band, with the same text each time.
_TEXT = "a" * 60
_over, _ = _guarded_draft(_TEXT, 30.0)
eq("a draw past its per-clip budget is rejected", _over["reject"], "over")
# The budget, not the 327.68 s cap: at the cap a 30 s draw is perfectly fine, which is
# exactly how a looping repair used to be promoted over the clip it was repairing.
eq("and it is the budget it was judged against", "budget" in (_over["reject_detail"] or ""), True)
_short, _ = _guarded_draft(_TEXT, 1.0)
eq("a draw too short for its text is rejected", _short["reject"], "short")
eq("the floor fires with the budget off too",
   _guarded_draft(_TEXT, 1.0, auto_token_budget=False)[0]["reject"], "short")
_ok, _ = _guarded_draft(_TEXT, 4.0)
eq("a plausible draw survives both bounds", _ok["reject"], None)

# The code gate: the half of the defects a duration check cannot see. Both draws below
# are the same plausible 4 s for the same text, so only the codes tell them apart.
_COLLAPSED = [10, 855, 4007] * 40          # 8 s of audio built from three codes
_CLEAN = list(range(300, 500))
_derailed, _dir = _guarded_draft(_TEXT, 4.0, track=_COLLAPSED)
eq("a derailed draw is rejected on its codes alone", _derailed["reject"], "derailed")
eq("with the verdict in the line", "collapse" in (_derailed["reject_detail"] or ""), True)
# Consumed: 925 B per second of audio would be ~11.7 GB over a corpus-wide pass.
eq("the dump is deleted once read", list((_dir / "dumps").iterdir()), [])
_rows = [json.loads(ln) for ln in
         (_dir / code_sidecar.SIDECAR_FILENAME).read_text(encoding="utf-8").splitlines()]
eq("one sidecar row per draw", len(_rows), 1)
# `stage` and `seed` are what make the row readable later: stage 05 and stage 06 append
# to the SAME file, and without them attempt 1 of a redo and attempt 1 of the original
# generation are the same row for the same stem.
eq("the row says which stage drew it", _rows[0]["stage"], "redo")
eq("and which seed", _rows[0]["seed"], 43)
eq("and that it was not kept", _rows[0]["accepted"], False)
eq("the codes themselves are stored, so the thresholds can be re-decided later",
   code_sidecar.decode_track(_rows[0]["semantic"]), _COLLAPSED)
_kept, _dir_ok = _guarded_draft(_TEXT, 4.0, track=_CLEAN)
eq("a clean draw passes the code gate", _kept["reject"], None)
_rows_ok = [json.loads(ln) for ln in
            (_dir_ok / code_sidecar.SIDECAR_FILENAME).read_text(encoding="utf-8").splitlines()]
eq("and is recorded as accepted", _rows_ok[0]["accepted"], True)
# A stock, unpatched server writes no dump at all. That has to read as "no opinion", or
# turning the gate on against the wrong server would reject every draw of every clip.
_no_dump, _dir_nd = _guarded_draft(_TEXT, 4.0, code_dump_dir=str(Path(tempfile.mkdtemp())))
eq("a missing dump rejects nothing", _no_dump["reject"], None)
eq("and records nothing", (_dir_nd / code_sidecar.SIDECAR_FILENAME).exists(), False)

# Both guards have to be reachable from the driver, which is the only way the pass is
# actually launched. A flag the driver never passes is a guard that is off in production.
_driver_text = DRIVER_PATH.read_text(encoding="utf-8")
for _flag in ("--auto-token-budget", "--no-auto-token-budget", "--code-dump-dir"):
    eq(f"driver.sh can pass {_flag}", _flag in _driver_text, True)
eq("and it puts them on the command line",
   'GUARD_ARGS[@]+"${GUARD_ARGS[@]}"' in _driver_text, True)

if _failures:
    print("FAILED:")
    for f in _failures:
        print("  -", f)
    raise SystemExit(1)
print("ALL PASSED")
