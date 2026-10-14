#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["loguru", "soundfile", "tqdm"]
# ///
"""Simple client for the local TTS server (POST /v1/audio/speech).

Both TTS services bind :8003 by design (only one up at a time) and this
client covers whichever answers: `crispasr-tts` (TADA backend),
`voxtral-tts` (vLLM + vllm-omni serving mistralai/Voxtral-4B-TTS-2603),
or a qwen3-tts VoiceDesign deployment (which is what --instruction is
for). Against Voxtral, `input` / `voice` / `response_format` / `speed` are
honoured per request, and so is `--seed`, but only through the nested
`extra_params.voxtral_noise_seed` this client also sends (see build_body);
its sampling knobs really are startup-time env vars. Pass --voice one of
the built-in embedding names in VOXTRAL_VOICES and, if a modelless request
is rejected by vLLM, add --model mistralai/Voxtral-4B-TTS-2603.

Two modes, selected by what --input is:

* literal text: one synthesis, --output is the audio file to write
  (default tts_out.wav).
* path to a .jsonl file: batch mode. Each row's `asr_training_source` is
  synthesized and written into the --output folder as
  {term_index:06d}_{variant_index:04d}_{normalized_term}.{format} (the two
  indices are zero-padded so the files sort in order across hundreds of
  thousands of clips; --format defaults to flac). A clip already present
  is skipped unless --overwrite is given, matched by the index/term stem
  regardless of extension, so switching --format (e.g. wav to flac) does
  not re-synthesize clips you already have and an interrupted run resumes
  by re-running the same command. Rows are synthesized
  --concurrency at a time (default 4): a vLLM/Voxtral server batches those
  concurrent requests server-side, so raising it speeds the run up, while
  the single-context TADA/crispasr backend wants 1. A tqdm bar tracks the
  remaining work; per-row failures are logged and skipped, and the run
  exits non-zero if any row failed (130 if you interrupted it).

  Stopping is safe at any moment: each clip is written to a .tmp sibling and
  renamed, so there is no half-written file for a resume to mistake for a
  finished one, and Ctrl-C cancels the queued clips instead of draining them
  (it only waits for the requests already in flight). Re-running the same
  command picks up exactly the clips that are missing, including the ones a
  previous run failed or rejected as truncated.

Voices in batch mode: v1 read the whole corpus with one voice (fr_female),
v2 spreads it over the ten French voices in utils/voice_assignment.py, which
is also what stage 99 re-derives the manifest's `voice` column from. The
default is --voice-mode group: the voice is a hash of the row's group key, so
all variants of a term (and all chunks of a PARHAF/PARROT document) are read
by one speaker, and a given group keeps its voice when the corpus is edited
or a run resumes. The roster, the mode and the group field are recorded in
voice_assignment.json in the output folder for stage 99 to check. Because the
voice is a per-request field, one run covers all ten with no restart. Pass
--voice X (or --voice-mode single) for the v1 single-voice behaviour, and
--voices-file to use a different roster.

Every knob (--speed, the talker sampling flags, --instruction, --voice,
--seed, --model) is sent only when set, so the server keeps its startup
defaults otherwise (same per-request contract as the full CrispASR client
this is simplified from, and what keeps TADA-only fields out of Voxtral
requests).

Truncation guard: /v1/audio/speech has no `finish_reason`, so a generation
that stops on the backend's token limit instead of at the end of the text
still answers 200 with a clip cut off mid-sentence. Any clip landing within
half a second of the frame cap (--max-audio-seconds, default 327.68s = the
served 4096 frames at 12.5 Hz) is therefore rejected as "stopped for
length": nothing is written, the row counts as failed, and the run exits
non-zero. Keep --max-audio-seconds in step with the server's actual
max_new_tokens, and shorten the input text rather than retrying, since the
same text truncates every time.

Logs go to stderr and are appended to a local file (see --log-file).

Simplified from CrispASR/tts.py (which absorbed the old voxtral query
client). Written with Claude Code.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from voice_assignment import (  # noqa: E402
    MODES,
    VOICES_FR_V2,
    group_field,
    load_voices,
    voice_for_row,
    write_provenance,
)
# The duration sanity rules live in utils/ because three callers need the SAME
# numbers: this script (which sends the budget as max_new_tokens and rejects a take
# that comes back outside it), 02_audit_durations.py (which re-checks clips already on
# disk) and eventually 06_hotfixes. Re-exported under the names the tests use.
from frame_budget import (  # noqa: E402
    BUDGET_FIXED_S,
    FAST_CHARS_PER_S,
    SLOW_CHARS_PER_S as BUDGET_CHARS_PER_S,
    budget_seconds,
    budget_tokens,
    classify_duration,
    floor_seconds,
)

DEFAULT_URL = "http://127.0.0.1:8003"
DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "tts_generate.log"

# Built-in voice embeddings shipped with mistralai/Voxtral-4B-TTS-2603, only
# relevant when voxtral-tts is the service on :8003. DELIBERATE COPY of
# VOXTRAL_VOICES in CrispASR/tts.py (separate repo, standalone tool); if the
# served voices change, update both sides. Reference list, not argparse
# choices, since --voice is backend-dependent free text.
VOXTRAL_VOICES = [
    "ar_male",
    "casual_female",
    "casual_male",
    "cheerful_female",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "neutral_female",
    "neutral_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]

# Voxtral decodes audio as discrete frames at a fixed 12.5 Hz and stops when it hits the
# server's max_new_tokens, WITHOUT saying so: /v1/audio/speech answers a plain 200 with a
# clip that simply ends mid-sentence (there is no `finish_reason` in the audio API, unlike
# the chat API where we guard on it). The only observable of a "length" stop is a duration
# pinned at the frame cap, which is what is_truncated() checks. DELIBERATE COPY of the
# serving constants (frame rate, VOXTRAL_MAX_TOKENS) from the voxtral-tts service in the
# separate CrispASR repo: if the server's cap changes, pass the matching
# --max-audio-seconds here (or update this default).
#
# The served value is 4096 frames (327.68 s), NOT the packaged default of 2048 (163.84 s):
# the cap was raised precisely because 2048 was cutting long clips off mid-sentence, and
# the stage-0 context was raised alongside it (VOXTRAL_MAX_MODEL_LEN=6144) since prompt
# tokens and generated frames share it, so 4096 frames are actually reachable.
VOXTRAL_FRAME_RATE_HZ = 12.5
VOXTRAL_DEFAULT_MAX_NEW_TOKENS = 4096  # -> 327.68 s of audio
TRUNCATION_TOLERANCE_S = 0.5
# How often the run writes a progress line (with an absolute ETA) to the log, in
# clips. The tqdm bar is for the terminal; this is for a run measured in days,
# where the bar is gone after a tmux detach or a scrollback wipe and the only
# durable answer to "when does this finish" has to be in the log file.
PROGRESS_LOG_EVERY = 500
# Extra attempts for a clip whose duration is rejected; see synthesize_with_retries
# for why a plain re-request is a genuinely different take.
DEFAULT_RETRIES = 2


class TruncatedAudioError(RuntimeError):
    """The backend stopped on its token limit instead of at the end of the text, so the
    clip is cut short. Subclasses RuntimeError so batch mode counts it as a per-row
    failure (nothing written, a resume retries) rather than aborting the run.

    With --auto-token-budget (the default) that limit is the per-clip budget, so this
    is the runaway detector: the take was still talking when it ran out of frames,
    i.e. it was looping rather than reading. Without it, it means the server's global
    cap, i.e. genuinely too much text."""


class ImplausiblyShortError(RuntimeError):
    """The clip is far SHORTER than its text can be read in, so part of the text was
    not spoken (or the take is near-silence). The mirror image of
    TruncatedAudioError, and a RuntimeError for the same reason: one failed row,
    nothing written, retried in place and again by the next resume.

    This failure has no `finish_reason` either: the backend emits a stop token early
    and answers 200 with a clip that sounds complete in isolation. Only comparing the
    duration to the text's length catches it, which is why the floor is checked on
    every clip and not just on budgeted runs."""


def floor_rate(args: argparse.Namespace) -> float:
    """`--floor-chars-per-second` off a namespace that may not define it."""
    return getattr(args, "floor_chars_per_second", FAST_CHARS_PER_S)


def budget_terms(args: argparse.Namespace) -> tuple[float, float]:
    """The upper bound's two terms off a namespace that may not define them.

    Needed even when the budget is OFF, because the floor check runs either way and
    goes through the same classifier: a caller that never asked for a budget (a test,
    stage 06's synthetic namespace) must not have to carry its fields.
    """
    return (getattr(args, "budget_fixed_seconds", BUDGET_FIXED_S),
            getattr(args, "budget_chars_per_second", BUDGET_CHARS_PER_S))


def attempt_budget(args: argparse.Namespace) -> int:
    """How many attempts one clip gets in total, i.e. `--retries` + 1."""
    return max(1, getattr(args, "retries", DEFAULT_RETRIES) + 1)


def frame_cap_seconds(max_new_tokens: int = VOXTRAL_DEFAULT_MAX_NEW_TOKENS) -> float:
    """Longest clip the backend can emit under a given frame budget, in seconds."""
    return max_new_tokens / VOXTRAL_FRAME_RATE_HZ



def is_truncated(duration: float | None, cap_seconds: float | None,
                 tolerance: float = TRUNCATION_TOLERANCE_S) -> bool:
    """Whether a clip looks cut off at the backend's frame cap. A generation that stops
    naturally lands wherever the text ends; one that stops on the token limit lands on the
    cap itself, so a duration within `tolerance` of it means "stopped for length".
    A non-positive cap disables the check (and an unparseable duration cannot be judged)."""
    if duration is None or not cap_seconds or cap_seconds <= 0:
        return False
    return duration >= cap_seconds - tolerance


def post_json(url: str, body: dict, timeout: float) -> tuple[int, bytes, float]:
    """POST `body` as JSON, return (status, payload, elapsed seconds)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0


def build_body(text: str, args: argparse.Namespace, voice: str | None = None) -> dict:
    """Request body for /v1/audio/speech; unset knobs are omitted so the
    server falls back to its startup defaults.

    `voice` overrides `args.voice` for this one row. v2 reads the corpus with ten
    voices, assigned per group by utils/voice_assignment, and the voice is a
    per-request field on every backend we use, so one run covers all ten with no
    restart: see run_batch.
    """
    body: dict = {"input": text, "response_format": args.format}
    # A per-clip frame budget turns a runaway take into a cheap, detected failure: the
    # server stops it at ~1.3x the plausible duration instead of generating up to
    # 327.68 s of looping syllables, which costs a slot ~6 minutes and (when it stops
    # just under the cap) silently lands rambling audio in the dataset under a short
    # label. ON by default since 2026-10-14: 0.52% of the clips generated at
    # cfg_alpha 2.0 exceed the budget and the worst exceeded it 4.7x (317 characters
    # of text, 202 s of audio), so leaving it off costs ~2,700 rambling clips over the
    # dictionary stage alone. --no-auto-token-budget disables it for a run at a lower
    # cfg_alpha, where the rambling is rare (v1 at 1.3: 4 clips in 1,068,676 over
    # 41.2 s). getattr rather than args.auto_token_budget so a synthetic namespace
    # that does not define the field still builds a valid body.
    if getattr(args, "auto_token_budget", False):
        body["max_new_tokens"] = budget_tokens(text, *budget_terms(args))
    optional = {
        "model": args.model,
        "instructions": args.instruction,
        "voice": voice or args.voice,
        "seed": args.seed,
        "speed": args.speed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    # The seed is sent a SECOND time, nested, because the flat field above cannot
    # reach Voxtral: it lands in the vLLM sampler's params, and Voxtral's stage 0
    # feeds that sampler a vocab-wide -inf row with one finite entry, so the token
    # is forced whatever the seed says. The audio's only randomness is the
    # flow-matching Gaussian, which the CrispASR voxtral-tts service draws per
    # request from extra_params.voxtral_noise_seed. Backends that read the flat
    # field (TADA, qwen3-tts) keep working, and a server with no reader for the
    # nested key merges and ignores it, so sending both is always safe. Same wire
    # format as CrispASR/tts.py and stage 06's tts_synthesize.
    # cfg_alpha rides the same nested route: stock vllm-omni drops a flat one silently.
    extra = {"cfg_alpha": args.cfg_alpha, "voxtral_noise_seed": args.seed}
    extra = {k: v for k, v in extra.items() if v is not None}
    if extra:
        body["extra_params"] = extra
    return body


def normalize_term(term: str) -> str:
    """Filesystem-safe slug: accents stripped, lowercased, runs of anything
    non-alphanumeric collapsed to a single underscore."""
    ascii_term = unicodedata.normalize("NFKD", term).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_term.lower()).strip("_")
    return (slug or "term")[:80]


def payload_duration(payload: bytes) -> float | None:
    """Duration in seconds of an in-memory audio payload, read from its header (no
    full decode). None when libsndfile cannot parse the container (raw pcm/f32, and
    mp3/aac depending on the local libsndfile build)."""
    try:
        info = sf.info(io.BytesIO(payload))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:  # noqa: BLE001 - any unparseable container, just report unknown
        return None


def synthesize(base_url: str, text: str, out: Path,
               args: argparse.Namespace, voice: str | None = None) -> tuple[float, float | None]:
    """Synthesize `text` into `out`; return (backend elapsed seconds,
    clip duration seconds or None when unparseable).

    Raises RuntimeError on a non-200 response and lets URLError propagate
    (an unreachable server should abort a batch, not fail every row).
    The audio is written to a .tmp sibling then renamed, so a killed run
    never leaves a truncated file that a resume would skip as done.

    Raises TruncatedAudioError when the clip comes back at or past its upper bound
    (the per-clip budget with --auto-token-budget, the server's global frame cap
    without it), and ImplausiblyShortError when it is below the floor its text can be
    read in. Both are RuntimeErrors, both leave nothing written, so the row stays "to
    generate" for the in-run retry and for the next resume.
    """
    code, payload, elapsed = post_json(f"{base_url}/v1/audio/speech",
                                       build_body(text, args, voice), args.timeout)
    if code != 200:
        raise RuntimeError(f"server returned {code} after {elapsed:.2f}s: "
                           f"{payload[:500].decode('utf-8', 'replace')}")
    dur = payload_duration(payload)
    # With a per-clip budget the cap to compare against is that budget, not the
    # server's global one: the whole point is to catch the clip that ran long for its
    # text, which the 327.68 s cap only catches in the worst case.
    budget_on = getattr(args, "auto_token_budget", False)
    fixed, slow = budget_terms(args)
    cap = (budget_seconds(text, fixed, slow) if budget_on else args.max_audio_seconds)
    # max() only to keep the ratio in the message below finite when the floor is
    # disabled with --floor-chars-per-second 0 (no clip can then be "short").
    floor = max(1e-9, floor_seconds(text, floor_rate(args)))
    # One classifier for both bounds, shared with 02_audit_durations.py so a clip
    # rejected here is exactly a clip that script would delete from disk, and vice
    # versa. `budget=None` lets it recompute the per-clip bound from the text.
    verdict = classify_duration(
        text, dur, budget=None if budget_on else args.max_audio_seconds,
        fixed=fixed, slow_chars_per_second=slow,
        fast_chars_per_second=floor_rate(args),
        tolerance=TRUNCATION_TOLERANCE_S)
    if verdict == "over":
        # Spelled out because this line is what a grep over a multi-day log has to
        # answer with: how far over, against which bound, and for how much text. The
        # ratio is the honest severity measure (2.0x is a loop, 1.05x is a long take).
        raise TruncatedAudioError(
            f"over budget: {dur:.2f}s of audio for {len(text)} chars, "
            f"{'budget' if budget_on else 'server frame cap'} {cap:.2f}s "
            f"({dur / cap:.2f}x), {'ran out of frames while still talking' if budget_on else 'stopped for length'}, "
            f"so the take was looping rather than reading. Not written."
        )
    if verdict == "short":
        raise ImplausiblyShortError(
            f"implausibly short: {dur:.2f}s of audio for {len(text)} chars, "
            f"floor {floor:.2f}s ({dur / floor:.2f}x), i.e. "
            f"{len(text) / max(dur, 1e-9):.0f} chars/s where the model reads at 16.6, "
            f"so part of the text was not spoken. Not written."
        )
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(out)
    return elapsed, dur


def synthesize_with_retries(base_url: str, text: str, out: Path,
                            args: argparse.Namespace,
                            voice: str | None = None) -> tuple[float, float | None, int]:
    """`synthesize` plus in-run retries on a duration rejection; returns
    (backend elapsed seconds of the ACCEPTED take, its duration, attempts used).

    Why retrying at all helps: the only randomness in Voxtral's pipeline is the fresh
    Gaussian each frame's flow-matching ODE starts from, so an unseeded re-request on
    the SAME text is a genuinely new take, not a replay. A take that rambled or
    dropped half the sentence is usually a sampling accident, and the measured rates
    (0.52% over budget at cfg_alpha 2.0, 0.001% under the floor on real v1 clips) say
    independent attempts rarely fail twice.

    Deliberately at the SAME cfg_alpha every time. Lowering it per attempt would
    improve the odds (the stage 06 ladder walks 2.0 -> 1.8 -> 1.6 -> 1.4 for its own
    QC failures), but then the clip on disk no longer matches the flat `cfg_alpha`
    column stage 06 writes into the manifest, and nothing here records a per-clip
    value. Add a sidecar first if that trade ever looks worth it.

    Only duration rejections are retried. A non-200, a timeout or an unreachable
    server are not sampling accidents: they are handled by the caller, which can
    abort the whole run rather than hammer a sick server row by row.
    """
    attempts = attempt_budget(args)
    for attempt in range(1, attempts + 1):
        try:
            elapsed, dur = synthesize(base_url, text, out, args, voice)
            if attempt > 1:
                logger.warning("{}: attempt {}/{} accepted ({:.2f}s of audio)",
                               out.name, attempt, attempts,
                               dur if dur is not None else float("nan"))
            return elapsed, dur, attempt
        except (TruncatedAudioError, ImplausiblyShortError) as e:
            # Logged per attempt, at warning while there is still a retry left and
            # left to the caller to log as an error once there is not, so a grep for
            # ERROR counts clips actually lost and a grep for the reason counts takes.
            if attempt >= attempts:
                raise
            logger.warning("{}: attempt {}/{} rejected, retrying: {}",
                           out.name, attempt, attempts, e)
    raise AssertionError("unreachable")  # pragma: no cover


def cmd_check(base_url: str, args: argparse.Namespace) -> int:
    """Probe /health, then synthesize a tiny clip on the server's defaults."""
    logger.info("GET {}/health", base_url)
    req = urllib.request.Request(f"{base_url}/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            code, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    except urllib.error.URLError as e:
        logger.error("unreachable: {}", e)
        return 2
    logger.info("  -> {}: {}", code, body[:200].decode("utf-8", "replace"))
    if code != 200:
        return 1

    logger.info("POST {}/v1/audio/speech (smoke test)", base_url)
    smoke = {"input": "Test.", "response_format": "wav"}
    if args.model:
        smoke["model"] = args.model
    code, payload, elapsed = post_json(f"{base_url}/v1/audio/speech", smoke, timeout=args.timeout)
    if code != 200:
        logger.error("  -> {}: {}", code, payload[:500].decode("utf-8", "replace"))
        return 1
    logger.info("  -> {}, {} bytes; model OK, backend took {:.2f}s", code, len(payload), elapsed)
    return 0


def run_single(base_url: str, args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        elapsed, dur, attempts = synthesize_with_retries(base_url, args.input, out, args)
    except urllib.error.URLError as e:
        logger.error("unreachable: {}", e)
        return 2
    except RuntimeError as e:
        logger.error("{}", e)
        return 1
    dur_info = f", {dur:.2f}s of audio" if dur is not None else ""
    logger.info("wrote {}{} (backend took {:.2f}s{})", out.resolve(), dur_info, elapsed,
                f", {attempts} attempts" if attempts > 1 else "")
    return 0


def resolve_roster(args: argparse.Namespace) -> tuple[tuple[str, ...] | None, str]:
    """Return (roster, mode) for this run: the voices to spread rows over, or None.

    Parameters
    ----------
    args
        Parsed CLI args; reads `--voice`, `--voices-file` and `--voice-mode`.

    Returns
    -------
    tuple of (tuple of str or None, str)
        `(None, "single")` means every row uses `args.voice` (or, if that is unset
        too, whatever voice the server booted with), which is exactly v1 behaviour.
        Otherwise the roster and the assignment mode.

    `--voice` wins over the roster rather than erroring, so the v1 invocations and
    the one-off re-synthesis of a single clip keep working unchanged.
    """
    if args.voice_mode == "single" or args.voice:
        return None, "single"
    voices = load_voices(args.voices_file) if args.voices_file else VOICES_FR_V2
    return voices, args.voice_mode


def run_batch(base_url: str, jsonl_path: Path, args: argparse.Namespace) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ten voices by default (v2), one voice when --voice or --voice-mode single says so
    # (v1). The voice is a per-request field, so a single run covers the whole roster
    # with no restart and no second pass.
    voices, voice_mode = resolve_roster(args)
    field: str | None = None
    per_voice: Counter = Counter()

    # Existing clips are matched by their index/term stem, not their full
    # filename, so a clip already synthesized as .wav counts as done even when
    # this run writes .flac. Scanned once up front (a single directory read) so
    # the resume check stays O(rows) instead of a glob per row. A leftover
    # ".flac.tmp" has stem "....flac" and so never masks a real stem.
    existing_stems = {p.stem for p in out_dir.iterdir() if p.is_file()}

    todo: list[tuple[Path, str, str | None]] = []
    planned: set[str] = set()
    skipped_existing = bad_rows = 0
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("term_index", "variant_index", "term", "asr_training_source")
                       if k not in row]
            if missing:
                logger.error("{}:{} missing {}, row skipped", jsonl_path, lineno, missing)
                bad_rows += 1
                continue
            stem = (f"{int(row['term_index']):06d}_{int(row['variant_index']):04d}_"
                    f"{normalize_term(row['term'])}")
            if stem in planned:
                logger.warning("{}:{} duplicate target {}, row skipped", jsonl_path, lineno, stem)
                bad_rows += 1
                continue
            planned.add(stem)
            # The group field is decided ONCE, from the first valid row, so one row
            # missing source_id cannot regroup the rest of the file onto term_index.
            if voices and field is None:
                field = group_field(row)
            voice = voice_for_row(row, field=field, mode=voice_mode,
                                  voices=voices) if voices else None
            if voice:
                # Counted for every planned row, not just the ones generated now, so
                # the provenance describes the corpus and not the tail of a resume.
                per_voice[voice] += 1
            if stem in existing_stems and not args.overwrite:
                skipped_existing += 1
                continue
            todo.append((out_dir / f"{stem}.{args.format}", row["asr_training_source"], voice))

    logger.info("{}: {} rows planned, {} already done (skipped), {} to generate",
                jsonl_path, len(planned), skipped_existing, len(todo))
    if voices:
        spread = ", ".join(f"{v}={per_voice.get(v, 0)}" for v in voices)
        logger.info("voices: {} of them, by {} ({} mode): {}",
                    len(voices), field, voice_mode, spread)
        # Beside the audio, not in this stage's code: stage 99 re-derives each clip's
        # voice from the same hash and needs to know which roster/mode/field produced
        # the clips it is reading. utils/voice_assignment owns the format for both.
        write_provenance(out_dir, jsonl_path, voices=voices, mode=voice_mode,
                         field=field, counts=per_voice)
    else:
        logger.info("voices: single ({})", args.voice or "server default")
        # Single-voice runs are recorded too, with a one-name roster, so stage 99 can
        # still label the clips (hashing a one-name roster always returns that name).
        # Only when --voice was given, though: without it the voice is whatever the
        # server booted with and this process never learns its name, so there is
        # nothing truthful to write and stage 99 falls back to no voice column.
        if args.voice:
            write_provenance(out_dir, jsonl_path, voices=(args.voice,), mode="single",
                             field=None, counts=Counter({args.voice: len(planned)}))

    failed = generated = 0
    # The two duration rejections are counted apart from the other failures because
    # they mean opposite things and have opposite fixes: "over" is a runaway take
    # (retry, or lower cfg_alpha), "short" is dropped text (retry, or check the
    # respelling). `retried_ok` is what says whether the in-run retries are worth
    # their GPU time: it counts clips that failed at least once and still landed.
    over_budget = too_short = retried_ok = 0
    total_audio = 0.0
    aborted = False
    interrupted = False
    # Fan out over a thread pool: each synthesize() call is an independent HTTP
    # request writing its own deterministic filename, so it is thread-safe. Running
    # several in flight lets a vLLM/Voxtral server batch them together server-side
    # (big speedup); for the single-context TADA/crispasr backend pass
    # --concurrency 1. as_completed yields in finish order, so log lines interleave,
    # but each row still lands in its own file.
    pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))
    futures = {pool.submit(synthesize_with_retries, base_url, text, out, args, voice): out
               for out, text, voice in todo}
    t_start = time.monotonic()
    # smoothing=0 makes tqdm report the CUMULATIVE average rate instead of its
    # default EMA (smoothing=0.3, which weights the last few clips heavily). That
    # default is unreadable here: clip duration spans 12.5 s (p5) to 24.1 s (p95) on
    # the dictionary stage, so the instantaneous rate swings by a third between
    # refreshes and the ETA jumps by hours. The cumulative rate converges within a
    # few hundred clips and is also the honest basis for an ETA over a multi-day run.
    # mininterval raises the refresh floor to 2 s: at ~1 clip/s the bar otherwise
    # redraws on every single completion for days.
    bar = tqdm(as_completed(futures), total=len(futures), unit="clip", desc="TTS",
               smoothing=0, mininterval=2.0)
    try:
        for fut in bar:
            out = futures[fut]
            try:
                elapsed, dur, attempts = fut.result()
            except urllib.error.URLError as e:
                logger.error("server unreachable ({}), aborting; re-run to resume", e)
                aborted = True
                break
            except TimeoutError as e:
                # ONE slow clip, not a dead server, so it must not abort the batch and
                # above all must not escape this loop: TimeoutError is not a URLError,
                # so before this branch existed it propagated out of the `for`, killed
                # the progress bar and ALL logging, and the `finally` below then let the
                # pool work through the entire remaining queue in silence (observed
                # 2026-10-14: logging died at 01:26 and the run was still writing clips
                # at 07:00, 22k clips later, with nothing in the log). The clip stays
                # "to generate", so a resume retries it.
                logger.error("{}: no response within {:.0f}s ({}), counted as failed; "
                             "a resume retries it", out.name, args.timeout, e)
                failed += 1
                continue
            except TruncatedAudioError as e:
                # Already warned once per rejected attempt by the wrapper; this is the
                # line that says the clip was given up on, after all its retries.
                logger.error("{}: giving up after {} attempts, {}",
                             out.name, attempt_budget(args), e)
                over_budget += 1
                failed += 1
                continue
            except ImplausiblyShortError as e:
                logger.error("{}: giving up after {} attempts, {}",
                             out.name, attempt_budget(args), e)
                too_short += 1
                failed += 1
                continue
            except RuntimeError as e:
                logger.error("{}: {}", out.name, e)
                failed += 1
                continue
            generated += 1
            if attempts > 1:
                retried_ok += 1
            if dur is not None:
                total_audio += dur
            logger.debug("{} ({}{:.2f}s backend)", out.name,
                         f"{dur:.2f}s audio, " if dur is not None else "", elapsed)
            # Two cumulative figures the bar's own rate cannot give. "xRT" is audio
            # seconds produced per wall second: it is the only number comparable
            # across stages and across server settings (concurrency, cfg_alpha,
            # euler steps), because clips/h depends on how long the stage's clips
            # happen to be (dictionary median 18.3 s vs acronyms 8.7 s).
            elapsed_run = max(1e-9, time.monotonic() - t_start)
            rate = (generated + failed) / elapsed_run
            bar.set_postfix_str(f"{rate * 3600:5.0f} clips/h, "
                                f"{total_audio / elapsed_run:4.1f}x RT", refresh=False)
            if generated % PROGRESS_LOG_EVERY == 0:
                left = len(futures) - generated - failed
                logger.info(
                    "progress: {}/{} ({:.1f}%), {:.0f} clips/h, {:.1f}x realtime, "
                    "{:.1f} h of audio written; {} left, ETA {:%Y-%m-%d %H:%M}",
                    generated + failed, len(futures),
                    100.0 * (generated + failed) / max(1, len(futures)),
                    rate * 3600, total_audio / elapsed_run, total_audio / 3600, left,
                    dt.datetime.now() + dt.timedelta(seconds=left / rate))
    except KeyboardInterrupt:
        # Ctrl-C has to stop the run in seconds, and without this it does not. Every
        # row is submitted up front, so the plain shutdown(cancel_futures=False) below
        # keeps the pool working through the ENTIRE remaining queue while the terminal
        # looks hung: measured on a 2000-task stand-in, one interrupt still executed
        # 2000/2000 tasks, which on a 534k-clip stage is days of "stopping". Cancelling
        # the queued futures leaves only the in-flight requests, i.e. a few seconds, and
        # each of those writes through a .tmp rename, so nothing is left half-written
        # and the next run resumes on exactly the clips that are missing.
        interrupted = True
        logger.warning("interrupted: {} clips written, dropping the queued ones; "
                       "re-run the same command to resume", generated)
    except Exception as e:  # noqa: BLE001
        # Belt and braces for the failure mode described in the TimeoutError branch
        # above: whatever the exception, stop the run loudly instead of letting
        # shutdown(wait=True) churn through the queue with no bar and no log lines.
        # `aborted` is what makes the shutdown below cancel the queued futures.
        aborted = True
        logger.opt(exception=True).error(
            "unexpected {} after {} clips, aborting; re-run the same command to resume",
            type(e).__name__, generated)
    finally:
        # On abort or interrupt, drop queued-but-unstarted work so the run stops
        # instead of grinding through every remaining row; in-flight requests are
        # left to finish (wait=True) because they are seconds away and atomic. On
        # success, wait for the pool normally.
        pool.shutdown(wait=not aborted, cancel_futures=aborted or interrupted)
        # break/KeyboardInterrupt leave the iterator unexhausted, so tqdm never
        # closes itself and the last bar line stays stuck on the terminal.
        bar.close()
    if aborted:
        # Everything not written counts as failed so the run exits non-zero; a
        # re-run resumes cheaply via the skip-existing logic above.
        failed += len(todo) - generated - failed

    audio_info = f", {total_audio / 60:.1f} min of audio" if total_audio else ""
    logger.info("done{}: {} generated{}, {} failed, {} skipped as existing, {} bad rows",
                " (interrupted)" if interrupted else "",
                generated, audio_info, failed, skipped_existing, bad_rows)
    # Separate line, always printed, because these are the numbers that decide whether
    # to re-run the stage and whether to touch cfg_alpha, and they must survive a grep
    # of a multi-day log that has 500k DEBUG lines in it.
    logger.info("duration rejections: {} over budget, {} implausibly short, "
                "{} saved by a retry (up to {} attempts each)",
                over_budget, too_short, retried_ok, attempt_budget(args))
    # 130 is the shell's conventional "killed by SIGINT", and generate_v2.sh keys on it
    # to stop the whole run instead of moving to the next stage (a stage exiting 1 for a
    # handful of failed clips must NOT stop it, since a re-run retries exactly those).
    if interrupted:
        return 130
    return 1 if (failed or bad_rows) else 0


def build_parser() -> argparse.ArgumentParser:
    """The CLI, separate from main() so tests can assert its defaults.

    Several of them are measured decisions rather than conveniences
    (--auto-token-budget, --timeout, --max-audio-seconds), so a test that pins them
    is pinning the reasoning in the help text, not the plumbing.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"Base URL of the TTS server (default: {DEFAULT_URL}).")
    p.add_argument("--check", action="store_true",
                   help="Probe /health and run a tiny synthesis to confirm the model is loaded.")
    p.add_argument("--input",
                   help="Text to read aloud, or a path to a .jsonl file whose rows "
                        "carry asr_training_source (batch mode).")
    p.add_argument("--output", default="tts_out.flac",
                   help="Output audio file (text mode, default tts_out.flac) or output "
                        "folder (jsonl batch mode).")
    p.add_argument("--overwrite", action="store_true",
                   help="Batch mode: regenerate audio files that already exist "
                        "(default: skip them).")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Batch mode: number of requests in flight at once "
                        "(default 4). A vLLM/Voxtral server batches concurrent "
                        "requests server-side, so >1 is a big speedup there; for "
                        "the single-context TADA/crispasr backend use 1.")
    p.add_argument("--instruction", default=None,
                   help="Voice-direction prose (qwen3-tts VoiceDesign only; TADA and "
                        "Voxtral ignore it). Not sent unless given.")
    p.add_argument("--voice", default=None,
                   help="ONE voice for the whole run, passed through verbatim (this is "
                        "v1 behaviour, and it overrides the roster below). TADA: a "
                        "tada-ref-*.gguf path or registry name. Voxtral: a built-in "
                        "embedding name, e.g. fr_male, fr_female, neutral_female (see "
                        "VOXTRAL_VOICES in this file).")
    p.add_argument("--voice-mode", default="group", choices=(*MODES, "single"),
                   help="How batch mode picks a voice per row (default: group). "
                        "'group' spreads the roster over groups, so all clips of a "
                        "term (or of a source document) share one voice: this is v2's "
                        "mode. 'row' gives every clip its own draw, spreading a term's "
                        "variants over speakers. 'single' is v1: one voice for "
                        "everything, from --voice or the server's default.")
    p.add_argument("--voices-file", default=None, metavar="FILE",
                   help="Roster to spread rows over, one voice name per line, '#' "
                        "comments ignored (default: the ten names in "
                        "utils/voice_assignment.py, i.e. the v2 roster). Accepts the "
                        "CrispASR repo's voices_fr_production.txt as is. The FILE ORDER "
                        "is part of the mapping, so re-ordering it re-assigns every "
                        "voice; changing the roster after generating invalidates the "
                        "voice column of the clips already written.")
    p.add_argument("--model", default=None,
                   help="OpenAI `model` field (default: not sent). CrispASR does not "
                        "need it; vLLM validates it, so against voxtral-tts pass "
                        "mistralai/Voxtral-4B-TTS-2603 if a modelless request is rejected.")
    p.add_argument("--seed", type=int, default=None,
                   help="Integer seed for reproducible synthesis (default: not sent). "
                        "Sent both flat (TADA, qwen3-tts) and nested as "
                        "extra_params.voxtral_noise_seed, the only route that reaches "
                        "Voxtral's flow-matching noise. NOTE in batch mode every clip "
                        "gets this same seed: identical text would then give identical "
                        "audio, which is the point for reproducibility, not a bug.")
    p.add_argument("--speed", type=float, default=None,
                   help="Speech speed, 0.25 to 4.0 (default: not sent, server default "
                        "applies). Honoured per request by both TADA and Voxtral.")
    p.add_argument("--cfg-alpha", type=float, default=None,
                   help="Voxtral classifier-free guidance, sent as extra_params.cfg_alpha (default: "
                        "not sent, the server's VOXTRAL_CFG_ALPHA applies, 2.0 in the CrispASR "
                        "compose since the 2026-10-13 sweep, which is what v2 is generated at). "
                        "Higher follows the text more closely, but the measured usable band is "
                        "1.2-2.0 (1.5-2.0 for the hybrid voices): 1.0 is CFG off and the decoder "
                        "stops following the text, 3.0+ distorts and often never terminates.")
    p.add_argument("--format", default="flac",
                   choices=("wav", "pcm", "f32", "flac", "mp3", "aac", "opus"),
                   help="Audio response format and file extension (default: flac, "
                        "lossless and about half the size of wav at the same rate). "
                        "CrispASR accepts wav/pcm/f32; Voxtral (vLLM) accepts "
                        "wav/flac/mp3/aac/opus/pcm, so against a wav-only backend "
                        "generate wav and convert afterwards.")
    # 600 s, not 300: the server's frame cap is 327.68 s of AUDIO
    # (VOXTRAL_MAX_TOKENS 4096 / 12.5 Hz), and a clip that runs to that cap costs
    # ~48 ms/frame at 16 requests in flight (~200 s) but ~89 ms/frame at 32
    # (~365 s, measured 2026-10-14), so 300 s times out on the longest clips at
    # high concurrency and reports a working server as a failure.
    p.add_argument("--timeout", type=float, default=600.0,
                   help="HTTP timeout in seconds (default: 600, which clears the "
                        "server's 327.68s frame cap even at 32 requests in flight).")
    p.add_argument("--max-audio-seconds", type=float, default=frame_cap_seconds(),
                   help="Backend frame cap in seconds: a clip landing within "
                        f"{TRUNCATION_TOLERANCE_S}s of it stopped for length (truncated "
                        "mid-sentence) and is rejected instead of written. Default "
                        f"{frame_cap_seconds():.2f} (the served "
                        f"{VOXTRAL_DEFAULT_MAX_NEW_TOKENS} frames at "
                        f"{VOXTRAL_FRAME_RATE_HZ} Hz); must match the server's actual "
                        "VOXTRAL_MAX_TOKENS. 0 disables the check.")
    p.add_argument("--auto-token-budget", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="ON by default (--no-auto-token-budget to disable). "
                        "Ask the server for only as many frames as the text can "
                        "plausibly need (max_new_tokens = "
                        f"{BUDGET_FIXED_S:.0f}s + chars/{BUDGET_CHARS_PER_S:.0f} at "
                        f"{VOXTRAL_FRAME_RATE_HZ} Hz) and judge truncation against that "
                        "budget instead of the global cap. Stops a looping take in "
                        "seconds instead of letting it run to 327.68s, and keeps "
                        "rambling audio out of the dataset. The rate is the SLOWEST "
                        "voice's slow tail, not the average one (see "
                        "utils/frame_budget.py): the ten v2 voices read at medians "
                        "10.7 to 16.4 chars/s, so a bound set on the average rejects "
                        "the slow voice for being slow.")
    p.add_argument("--budget-fixed-seconds", type=float, default=BUDGET_FIXED_S,
                   help="Head/tail room term of --auto-token-budget (default: "
                        f"{BUDGET_FIXED_S}).")
    p.add_argument("--budget-chars-per-second", type=float, default=BUDGET_CHARS_PER_S,
                   help="Slowest plausible reading speed in characters per second of "
                        f"audio, the other term of --auto-token-budget (default: "
                        f"{BUDGET_CHARS_PER_S}, below the slowest voice h_frm_chf at "
                        "p0.1 of 5.58; its median is 10.7 and the fastest voice's is "
                        "16.4).")
    p.add_argument("--floor-chars-per-second", type=float, default=FAST_CHARS_PER_S,
                   help="Fastest plausible reading speed in characters per second of "
                        "audio: a clip shorter than chars/this did not speak all of "
                        f"its text and is rejected (default: {FAST_CHARS_PER_S}, above "
                        "the fastest clip ever measured: 21.3 chars/s over 28,749 v2 "
                        "clips and 28.5 over 602,813 v1 ones, so it flags none of "
                        "them). Checked "
                        "on every run, budget or not, since it needs nothing from the "
                        "server. 0 disables it.")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="Extra attempts for a clip whose DURATION is rejected (over "
                        "budget or implausibly short), default 2, i.e. 3 attempts. "
                        "Each attempt is a genuinely new take (the flow-matching noise "
                        "is fresh per request), at the same cfg_alpha, so a rambling "
                        "take is usually fixed by one retry. Does not retry non-200s, "
                        "timeouts or an unreachable server.")
    p.add_argument("--log-file", default=str(DEFAULT_LOG_FILE),
                   help=f"Loguru log file, appended to (default: {DEFAULT_LOG_FILE}).")

    sampling = p.add_argument_group(
        "talker sampling (TADA only)",
        "Per-request talker knobs; unset = server startup default "
        "(do_sample on, temperature 0.6, top_p 0.9, top_k 0, repetition_penalty 1.1). "
        "Leave unset against Voxtral: its sampling is startup-time only and the "
        "vllm-omni request schema does not define these fields.")
    sampling.add_argument("--temperature", type=float, default=None,
                          help="Talker sampling temperature (0 = greedy).")
    sampling.add_argument("--top-p", type=float, default=None,
                          help="Talker nucleus-sampling cutoff.")
    sampling.add_argument("--top-k", type=int, default=None,
                          help="Talker top-k cutoff (0 = disabled).")
    sampling.add_argument("--repetition-penalty", type=float, default=None,
                          help="Talker repetition penalty (1.0 = none).")
    sampling.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None,
                          help="Enable talker sampling; --no-do-sample forces greedy decode.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # stderr through tqdm.write so log lines don't mangle the progress bar.
    logger.remove()
    logger.add(lambda m: tqdm.write(m, end="", file=sys.stderr), level="INFO", colorize=True)
    logger.add(args.log_file, level="DEBUG")

    base_url = args.url.rstrip("/")
    if args.check:
        return cmd_check(base_url, args)
    if not args.input:
        p.error("--input is required unless --check is given")

    in_path = Path(args.input)
    if in_path.suffix == ".jsonl" and in_path.is_file():
        return run_batch(base_url, in_path, args)
    return run_single(base_url, args)


if __name__ == "__main__":
    sys.exit(main())
