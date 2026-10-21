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
"""Transcribe a dataset AND regenerate the worst TTS clips, in one combined pass.

This is a superset of ``01_compute_stt.py``: it runs the same STT + CER scoring
over the whole dataset, and interleaved with it (same parallel pool, one clip at a
time) it tries to repair the clips the TTS engine derailed on. Doing both together
means the "before" transcript needed to measure the TTS gain is computed right
where the improvement happens, no separate scoring run.

It reuses ``01_compute_stt.py``'s orchestration verbatim (its ``process_file``: the
dual clip / audio-seconds progress bars, resume, atomic flushing, thread pool)
through injected per-clip hooks, so none of that is duplicated. With ``--n-improv 0``
it degenerates to exactly ``01_compute_stt.py``.

STT and TTS run in parallel against their own servers. Each has an independent
concurrency cap (``--stt-parallel`` / ``--tts-parallel``): every transcription passes
through one semaphore, every regeneration through another, so a clip busy on TTS
never holds a transcription slot and both GPUs stay saturated. The worker pool is
sized ``stt-parallel + tts-parallel`` so both caps can be full at once.

``--mode`` picks how STT and TTS are combined, so this runs even when the two servers
cannot share VRAM:

* ``both`` (default): the pass above, both servers up, regeneration scores each draw
  inline and stops early once one clears the threshold.
* ``stt`` (only the STT server up): transcribe + score originals, flag the bad ones
  ``pending_tts``; AND score the candidates a prior ``tts`` pass persisted
  (``pending_stt``), promoting the best draw or exhausting the clip. Deletes the
  candidate files once picked. It only picks from a COMPLETE set: a clip whose
  candidates are short of ``--n-improv`` (the tts pass ran with a smaller one, or a
  sidecar has gone missing) goes back to ``pending_tts`` rather than being judged, and
  settled, on a partial set. "Complete" means the tts pass delivered every draw it was
  asked for, not one file per draw: a draw it skipped as a duplicate or on one of the
  generation guards of step 2a is skipped deterministically (the seed per attempt is
  fixed), so demanding a file for it would bounce the clip between the two passes
  forever.
* ``tts`` (only the TTS server up): for every ``pending_tts`` clip, synthesize all
  ``--n-improv`` candidates (no early stop, since scoring needs STT) and persist each
  as a ``<clip>.draft<seed>.<fmt>`` sidecar, marking the clip ``pending_stt``.

A driver alternates ``stt`` and ``tts`` runs (bringing the matching server up between
them); the state lives in the row's ``improvement.status`` so each pass resumes where
the last left off. Convergence is signalled by ``stt`` exiting 10 when no clip anywhere
still needs either pass. The split flow loses only ``both``'s per-clip early stop (the
whole candidate set is generated before any is scored), which is the right trade when
minimizing server swaps dominates.

Failures retry with exponential backoff (STT and TTS alike). Nothing that fails is
silently skipped: a request that keeps failing after every retry ABORTS the run.
Exit codes: 2 the seed had no effect (SeedDeterminismError), 3 a generation came
back empty / duration-0 (GenerationError: a real clip must never be replaced with
silence), 4 a server stayed unreachable (TranscriptionError / TTSRequestError), 5 a
dataset clip transcribed blank after every retry (EmptyTranscriptError: an empty
transcript is never written, it stops the run so the bad clip is surfaced), 10 (``stt``
mode only) a clean "no improvement work left, dataset converged" signal for the driver
loop, 11 (split modes) "this mode has run dry": the pass completed nothing, so repeating
it changes nothing, but clips are still waiting on the OTHER mode. 11 is what lets a
driver run one mode on its own (an STT-only night while the TTS server is down) and
still stop by itself. The one non-fatal case is a regenerated DRAW that transcribes blank: that seed is
discarded and the next is tried (the point of trying several seeds), the original clip
is untouched.

Progress is visible live across three bars: the clip bar (STT sweep) and the
audio-seconds bar are inherited from ``01_compute_stt.py``; a third TTS bar shows
the repair backlog, its total is the clips found bad (high CER) and it fills as they
are successfully improved, so the gap between the two reveals how far the slow TTS
engine is lagging the fast STT pass. The clip bar postfix shows the last clip's CER,
running improved / exhausted tallies, and a rolling STT real-time factor (rtf = STT
seconds / audio seconds, < 1 = faster than real time). Every scored clip logs one
CER line (with its STT time), and a clip over the threshold additionally dumps its
source / target / transcript, exactly like ``01_compute_stt.py``. Every clip being
redone also logs its per-attempt seed, CER and STT time above the bar (REDO /
IMPROVED / EXHAUSTED lines). Per-clip stt_seconds / stt_rtf are also stored in the
output jsonl. Every row also carries a top-level ``cfg_alpha`` recording the TTS
cfg_alpha behind its CURRENT audio: a regenerated clip stores its winning draw's
cfg_alpha, while a clip that keeps its original audio (passed under the threshold, or
exhausted) stores the value stage 05 drew it at: the per-clip rung recorded in
``cfg_alpha_sidecar.jsonl`` beside the audio when there is one (stage 05 walks the
same ladder down when a take is rejected for its duration), else
``--original-cfg-alpha`` (2.0 by default, the value v2 audio is generated at; v1 audio
was drawn at 1.3, so pass it when running over v1). Audio is
replaced through atomic writes (temp + fsync + os.replace)
for both the ``.bak`` backup and the new clip, so an interrupted run can never
corrupt a clip.

Per clip (in the pool):

1. Transcribe the original audio (reusing a stored transcript on resume) and score
   it by CER only, against BOTH ``asr_training_source`` and ``asr_training_target``,
   taking the lower of the two (``min``). CER uses the exact same normalization as
   script 1 (NFC, lowercase, punctuation stripped but accents kept, whitespace
   collapsed), applied to source, target and transcript alike.
1a. Score the ENDING too. A whole-clip CER averages over the whole reference, so a
   defect confined to the last seconds of a long clip is diluted by everything that came
   out right: a chunk truncated at the TTS output cap keeps ~85% of its text and lands at
   CER ~0.13, under any sane gate. So every clip longer than 30s is also scored over its
   last ~30s (approximated by character count at the corpus median pace, see
   ``_stt.tail_cer``), stored as ``cer_tail`` on the transcription entry
   (``--tail-cer-threshold`` 0.12, looser than ``--cer-threshold`` 0.08 because the
   window boundary cuts mid-sentence).
1b. Score the WORST STRETCH too, which is what catches a defect in the MIDDLE of a
   long clip. The tail only looks at the ending and the whole-clip CER averages the
   defect away: five words dropped out of a 48 s PARHAF chunk are ~30 normalized
   characters, i.e. CER 0.04, under any gate that does not also flag every clean long
   clip. So two more scores come from ``_stt.window_metrics``: ``cer_window``, the CER of
   the worst ~15 s (213-character) window, and ``bad_run_chars``, the longest contiguous
   run of reference characters the recognizer never said at all. A clip is bad when ANY
   of the four scores is over its threshold (``--window-cer-threshold`` 0.15,
   ``--bad-run-threshold`` 30 characters: measured p50 1, p99 5, p99.9 9, max 26 over
   19,958 v1 clips, so 30 flagged none of them). ``bad_run_chars`` is a count and not a
   rate on purpose: a skipped phrase has a size, and dividing it by the clip's length is
   exactly what hides it.

1c. Triple-check a bad reading before trusting it. STT can hallucinate a wildly wrong
   transcript on perfectly fine audio (Whisper's "Sous-titrage ..." caption is the
   classic case), which scores a false-bad CER and would waste a regeneration. So while
   the clip reads bad on ANY gate and has been read fewer than
   ``STT_CHECK_TARGET`` (3) times, re-transcribe it, each recheck one step further up the
   ``--stt-recheck-temperature`` ladder (default base 0.5, distinct from
   ``--stt-temperature``, so the two rechecks run at 0.5 then 1.0), and keep whichever
   reading scored the LOWEST CER. The reading count lives in ``n_stt_check`` on the
   transcription entry, so a clip still bad afterwards is never re-read again (it proceeds
   to step 2) and a resumed run only tops a partially checked clip back up to 3; a clip a
   recheck rescued drops the marker and is treated as a normal good clip (no regeneration).
2. If the clip still reads bad on any of the four gates, improve it: up to ``--n-improv``
   attempts, re-synthesize via the TTS server (seed incremented each attempt,
   TTS ``cfg_alpha`` walked from ``--cfg-alpha-start`` by ``--cfg-alpha-step`` each
   attempt, optional ``--speed-jitter``), re-transcribe, re-score. The goal is a draw
   UNDER EVERY GATE: stop early the moment one is reached. If no draw gets there, keep
   the lowest-CER draw as long as it still beats the original by ``--min-improvement``
   (otherwise keep the original untouched).
2a. Every draw is held to the v2 GENERATION guards before it is worth transcribing, the
   same three stage 05 generates under and with the same numbers (they live in
   ``utils/frame_budget.py`` and ``utils/code_loops.py``, called through ``_tts``, never
   copied): it is sent a per-clip frame budget as ``max_new_tokens``
   (``--auto-token-budget``, on) and discarded if it comes back past that budget (it was
   looping, not reading) or below the floor its text can be read in (part of the text was
   never spoken), and with ``--code-dump-dir`` also if its own semantic codes say it
   derailed. Before this, a redo sent no budget and only checked the server's global
   327.68 s cap after the fact, so repairing a looping clip could spend ~6 minutes of GPU
   producing exactly the failure the generator would have killed at 50 s.
3. On a win: back the current clip up before overwriting (the first win writes the
   pristine original to ``<clip>.flac.bak``, later wins write the prior version to
   ``<clip>.flac.bak2`` / ``.bak3`` / ..., so every intermediate audio is kept),
   atomically overwrite the original with the better audio, re-probe and update the
   row's ``duration``,
   update the stored transcription, and append a record to ``<name>.improved.jsonl``.

A per-row ``improvement`` marker records the outcome so a resumed run skips clips
already improved / exhausted (override with ``--force``). Output mirrors script 1:
``<name>.stt.jsonl`` under ``--output``.

``--rescore`` re-derives every stored ``cer`` / ``cer_tail`` under the current scoring
rules and re-applies the gates. Changing ``normalize_for_scoring`` or a threshold makes
every score already on disk stale, while leaving the TRANSCRIPTS themselves perfectly
valid, so re-transcribing the corpus would be pure waste: the worker reuses the stored
transcript and only a row that now reads bad (and has fewer than ``STT_CHECK_TARGET``
readings) spends an STT call. It also re-judges the REGENERATION QUEUE the old rules
built: a ``pending_tts`` row that now reads clean drops its marker instead of costing a
regeneration, and one that stays bad has its ``orig_cer`` / ``orig_tail`` refreshed, so
a candidate is later compared against today's number and not a stale inflated one.

``<name>.improved.jsonl`` is a durable audit trail of every win: it is loaded and
merged (never deleted) on start, and each record keeps the before / after CER
(``orig_cer`` / ``new_cer`` / ``cer_delta``), the before / after transcript
(``orig_transcript`` / ``new_transcript``), the before / after duration
(``orig_duration`` / ``new_duration``), the row ``index`` and audio path, the
winning ``seed`` / ``speed`` / ``attempts``, the ``.bak`` path, and timestamps
(``first_ts`` / ``ts`` epoch plus ``*_iso``). The pristine baseline
(``orig_cer_pristine``) and ``first_ts`` survive re-improvement so history is never
lost.

Seed: retries walk seeds from ``--start-seed`` (default 43), incrementing per attempt.
It reaches Voxtral's flow-matching noise only on the forked vllm-omni the CrispASR image
is built from, which routes it to stage 0's ``voxtral_noise_seed``; on a stock server it
is accepted and inert (see ``tts_synthesize``). As a safety net, a regenerated clip
byte-identical to the original (checked by length then SHA1) means the seed had no
effect, so the run CRASHES immediately (SeedDeterminismError) instead of wasting a
whole run. ``--speed-jitter`` is NOT a second lever on this backend: vllm-omni applies
``speed`` after generation, as a torchaudio phase vocoder over the finished waveform,
so it time-stretches identical audio rather than drawing it differently. Leave it at 0.

In ``--mode both`` two servers must be up at once: the STT endpoint
(``--stt-endpoint``) and the TTS server (``--tts-url``, default
http://127.0.0.1:8003). The split ``stt`` / ``tts`` modes need only their own server
up (that is their whole point). Manifest audio paths are resolved via ``--audio-root``
(usually required), then absolute / CWD.

``--category`` (repeatable) restricts ALL work to rows whose ``category`` matches (e.g.
``--category parhaf``). Rows in every other category are passed through UNCHANGED: their
already-computed transcriptions / CER are preserved, never re-run, so you can focus the
expensive TTS regeneration on the one noisy subset now and resume the rest later without
losing anything. Convergence (the ``stt`` exit 10) is then judged over the selected
categories only, so the driver's alternating loop stops when just those subsets are done,
ignoring any pending clips a prior full run left in the other categories.

``--stems-from`` (repeatable) is the same restriction one CLIP at a time: a QC sidecar
jsonl contributes its flagged rows' stems, anything else is read as a plain list of
stems, and every row outside the list is passed through unchanged, convergence included.
It reads the same two file shapes as ``05_generate_audio/03_qc_cpu_stt.py --stems-from``
(both call ``utils/stem_requests.py``), so one stage's findings are another's worklist.
That is what makes a second opinion affordable: the v2 plan sweeps all 602k clips with
the fast whisper model, then re-judges only the ~12k it flagged with the slower,
stronger one before a single clip is regenerated, which is an hour instead of three days.

Usage (one command does STT + improvement over the whole manifest):

    uv run 01_recursive_improvement.py \
        --input ../99_hf_release/data/NeMO_files/full.jsonl \
        --output ./stt_out \
        --audio-root ../99_hf_release/data/NeMO_files \
        --stt-endpoint http://localhost:8000/v1/audio/transcriptions \
        --stt-model whisper-large-v3 \
        --tts-url http://127.0.0.1:8003 --tts-voice fr_male \
        --stt-parallel 8 --tts-parallel 8 --n-improv 5 --cer-threshold 0.08

When STT and TTS cannot share VRAM, alternate the split modes instead (a driver script
brings the matching server up before each), same flags plus ``--mode``:

    uv run 01_recursive_improvement.py ... --mode stt   # transcribe + score + promote
    uv run 01_recursive_improvement.py ... --mode tts   # generate candidates
    # repeat until the --mode stt run exits 10 (converged), or 11 (that mode ran dry)

Every pass appends one line per dataset to ``rounds.jsonl`` beside its output (where the
dataset stands, what the pass did, the gates it judged by), and the history is read back
with no other option and no server, from another tab while a pass is running:

    uv run 01_recursive_improvement.py --report          # here and one level down
    uv run 01_recursive_improvement.py --report improved # one output dir

This file was written with Claude Code.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import click
import soundfile as sf
from loguru import logger
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent


class SeedDeterminismError(RuntimeError):
    """Raised when a regenerated clip is byte-identical to the original, which
    means the request seed did not take effect. We abort the whole run rather than
    waste it producing no change."""

    def __init__(self, ident: str, seed: int):
        super().__init__(
            f"regenerated clip for {ident!r} is BYTE-IDENTICAL to the original "
            f"(same length and SHA1) with seed {seed}: the seed had no effect, so "
            f"retries cannot change anything. Check for a fixed seed env var on the "
            f"TTS server overriding the request, or bump --start-seed if the "
            f"originals were generated with this seed."
        )


class GenerationError(RuntimeError):
    """A TTS generation came back empty (0 bytes) or with duration 0. Raised so the
    run aborts rather than replacing a real clip with silence or nothing."""


class TTSRequestError(RuntimeError):
    """The TTS request kept failing (network / HTTP) after every retry. Raised so
    the run aborts rather than silently skipping the clip: a dead TTS server would
    otherwise make every regeneration a no-op with nobody noticing."""


def _load_module(path: Path, name: str):
    """Import a sibling script whose filename is not a valid module name (the
    ``01_`` prefix), so we can reuse its helpers instead of duplicating them.
    Importing runs the module body only; its ``main()`` is guarded by
    ``if __name__ == '__main__'`` so nothing executes on import."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse script 1's STT + scoring + io helpers and orchestration, and the stage-05
# audio client's HTTP layer, rather than copying them (the repo forbids duplication).
_stt = _load_module(_HERE / "01_compute_stt.py", "compute_stt_mod")
_tts = _load_module(
    _HERE.parent / "05_generate_audio" / "01_generate_audio.py", "generate_audio_mod"
)
# The stage-05 clip-name convention lives with the manifest builder that already reads it
# (stdlib-only import, so it costs nothing here).
sys.path.insert(0, str(_HERE.parent / "utils"))
from progress import bar_safe_sink  # noqa: E402
from nemo_manifest import parse_clip_name  # noqa: E402
# "Work on exactly these clips" lists, shared with the live CPU QC that produces them
# (05_generate_audio/03_qc_cpu_stt.py --stems-from reads the same two file shapes).
import stem_requests as _stems  # noqa: E402
# The cfg_alpha ladder and the per-clip sidecar are shared with stage 05 (which walks
# the same rungs when a take is rejected for its duration): one copy of the numbers, one
# file format, so the `cfg_alpha` column this stage writes can tell the truth about a
# clip stage 05 happened to draw below the run's value.
from cfg_sidecar import (  # noqa: E402
    CFG_ALPHA_FLOOR,
    CFG_ALPHA_START,
    CFG_ALPHA_STEP,
    cfg_alpha_for_clip,
    ladder_cfg_alpha,
)
# Who reads which clip is decided in one place for the whole repo (stage 05 at synthesis
# time, stage 99 at release time). This stage only needs the two things that place can
# tell it: the roster a directory of audio was generated with, and which OTHER voice to
# hand a clip whose own voice could not read it (see `swap_voice_for`).
import voice_assignment as _voices  # noqa: E402
# The per-round history (`improved/rounds.jsonl`) and its report. Format, reader and
# renderer live in one module so `--report` cannot drift from what the passes append,
# and so a future stage can append to the same ledger without redefining it.
import rounds_ledger as _ledger  # noqa: E402


# How many independent STT readings a bad-CER clip gets before its CER is trusted: the
# first pass plus (STT_CHECK_TARGET - 1) rechecks at rising temperatures. 3 = the
# triple-check of step 1b above. The count is stored per clip in ``n_stt_check``.
STT_CHECK_TARGET = 3


def recheck_temperature(base: float, readings_done: int) -> float:
    """Temperature for the recheck that follows ``readings_done`` readings (>= 1): the
    ladder walks ``base``, ``2 * base``, ... capped at 1.0 (Whisper's own fallback
    ceiling), so each recheck samples the audio differently instead of repeating the
    previous reading. With the default base 0.5 the two rechecks run at 0.5 then 1.0."""
    return round(min(1.0, base * readings_done), 4)


def best_cer(transcript: str | None, source: str | None, target: str | None) -> float | None:
    """Lower of the CER against source and against target, both fully normalized
    by script 1's ``compute_metrics`` (which normalizes reference and hypothesis
    identically). None when there is nothing scorable."""
    if not transcript:
        return None
    vals: list[float] = []
    for ref in (source, target):
        if ref:
            _wer, cer = _stt.compute_metrics(ref, transcript)
            if cer is not None:
                vals.append(cer)
    return min(vals) if vals else None


def best_tail_cer(
    transcript: str | None, source: str | None, target: str | None, duration: float | None
) -> float | None:
    """``best_cer`` restricted to the last ~30s of the clip (see ``_stt.tail_cer``): the
    lower of the two tail CERs, or None on a clip too short to have a tail."""
    if not transcript:
        return None
    vals = [
        cer for ref in (source, target) if ref
        for cer in [_stt.tail_cer(ref, transcript, duration or 0.0)] if cer is not None
    ]
    return min(vals) if vals else None


def best_window_metrics(
    transcript: str | None, source: str | None, target: str | None
) -> tuple[float | None, int | None]:
    """``best_cer``'s treatment of the two WINDOW scores (see ``_stt.window_metrics``):
    the worst ~15s window's CER, and the longest contiguous run of reference characters
    the recognizer never said, each taken against whichever reference reads it more
    favorably. Taken independently of each other, exactly as ``cer`` and ``tail``
    already are: a defect counts only when NEITHER spelling of the reference explains
    it."""
    if not transcript:
        return None, None
    pairs = [_stt.window_metrics(ref, transcript) for ref in (source, target) if ref]
    windows = [w for w, _run in pairs if w is not None]
    runs = [run for _w, run in pairs if run is not None]
    return (min(windows) if windows else None), (min(runs) if runs else None)


class Scores(NamedTuple):
    """Every number one reading of a clip is judged on, travelling as a single value.

    They answer four different questions and no one of them subsumes the others:
    ``cer`` the whole clip, ``tail`` its last ~30s (an ending cut off at the TTS frame
    cap keeps ~85% of its text, so it lands at whole-clip CER ~0.13), ``window`` the
    worst ~15s stretch (five words dropped mid-clip are ~30 normalized characters:
    0.14 of a 213-char window but 0.04 of a 48s PARHAF chunk) and ``bad_run`` the
    longest span of reference text that went unsaid, in characters (the SHAPE of a
    skip rather than an average, which is why it is a count and not a rate).

    Every field is optional because each has a reading that cannot produce it: no
    transcript at all, a clip too short to have a tail, an empty reference."""

    cer: float | None = None
    tail: float | None = None
    window: float | None = None
    bad_run: int | None = None


def score_reading(
    transcript: str | None, source: str | None, target: str | None, duration: float | None
) -> Scores:
    """All four scores of one reading, each against whichever reference reads it best."""
    window, bad_run = best_window_metrics(transcript, source, target)
    return Scores(cer=best_cer(transcript, source, target),
                  tail=best_tail_cer(transcript, source, target, duration),
                  window=window, bad_run=bad_run)


def scores_from_state(state: dict) -> Scores:
    """The scores a prior pass stored beside a clip's ``improvement`` marker. A key a
    marker written before that gate existed never carried reads as None, i.e. the same
    as "this reading cannot produce that score", which costs the clip nothing: the stt
    pass re-derives all four from the transcript before it judges anything, and these
    only travel so the tts pass can log what flagged the clip and the promote
    comparison has a number to beat."""
    return Scores(cer=state.get("orig_cer"), tail=state.get("orig_tail"),
                  window=state.get("orig_window"), bad_run=state.get("orig_bad_run"))


def scores_to_state(scores: Scores) -> dict:
    """``scores_from_state``'s inverse, spread into an ``improvement`` marker."""
    return {"orig_cer": scores.cer, "orig_tail": scores.tail,
            "orig_window": scores.window, "orig_bad_run": scores.bad_run}


#: The two marker fields a state hop must NOT drop: the voice a redraw has to use and
#: how many rungs of the escalation ladder the clip has burned (see `swap_voice_for`).
#: Everything else in the marker is rewritten from scratch at every hop, which is what
#: keeps a stale score out of the next round; these two are the clip's voice history,
#: so they have to outlive it.
CARRIED_FIELDS = ("swap_voice", "voice_tier")


def set_improvement(row: dict, status: str, orig: Scores, **extra) -> dict:
    """Write ``row['improvement']``, carrying over the marker a state hop must not drop.

    Every state the clip passes through (pending_tts, pending_stt, exhausted, improved)
    replaces the whole marker, which is what keeps a stale field from a previous round
    out of the next one. `CARRIED_FIELDS` are the exception: the stt pass decides the
    next voice, the tts pass two steps later is what draws with it (`voice_for_redraw`),
    and the pending_stt hop in between would otherwise lose both and silently redraw
    with the voice that already failed, at a rung the clip had already climbed.
    """
    state = {"status": status, **scores_to_state(orig), **extra}
    prev = row.get("improvement") or {}
    for field in CARRIED_FIELDS:
        # Truthiness, not `in`: a `voice_tier` of 0 says the ladder is untouched, which
        # is exactly what its absence says, so there is nothing to carry.
        if prev.get(field) and field not in state:
            state[field] = prev[field]
    row["improvement"] = state
    return state


def voice_for_redraw(row: dict) -> str | None:
    """The voice a redraw of this clip must use: its swap voice once one was decided,
    else the voice that generated it. None on a v1 (single-voice) manifest, where the
    request simply carries ``--tts-voice`` as before. Named apart from
    `voice_assignment.voice_for_row`, which answers the different question of which
    voice a row was ASSIGNED at synthesis time."""
    return (row.get("improvement") or {}).get("swap_voice") or row.get("voice")


@functools.lru_cache(maxsize=None)
def _swap_roster(audio_dir: str) -> tuple[tuple[str, ...], str] | None:
    """``(roster, mode)`` for one audio directory, from the ``voice_assignment.json``
    stage 05 left beside the clips, or None when that file cannot answer.

    Cached per directory (five of them for the whole corpus, against 602k rows) and
    logged once per directory, so a run says exactly once why a subset will or will not
    swap voices. Called from worker threads; `lru_cache` only ever recomputes an
    idempotent read, so a race costs a duplicate log line at worst.
    """
    entry = _voices.read_dir_provenance(audio_dir)
    if entry is None:
        logger.info(
            f"{audio_dir}: no usable {_voices.PROVENANCE_FILENAME} (v1 audio, or two "
            f"stages generated here with different rosters), so no voice swap: an "
            f"exhausted clip keeps its original audio")
        return None
    roster = tuple(entry.get("voices") or ())
    mode = str(entry.get("mode") or "")
    if not roster or mode not in _voices.MODES:
        logger.info(f"{audio_dir}: generated in {mode or 'single'} mode "
                    f"({len(roster)} voice(s)), so no voice swap here")
        return None
    # Nested quotes kept out of the f-string: PEP 701 would allow them, but this file
    # declares requires-python >=3.10, where they are a syntax error.
    scope = ("one new reader per clip" if mode == "row" else
             "one new reader per document, shared by every chunk needing repair")
    logger.info(f"{audio_dir}: {len(roster)} voice(s), {mode} mode, "
                f"voice swap ON ({scope})")
    return roster, mode


def swap_voice_for(row: dict, audio_path: Path,
                   cfg: SimpleNamespace) -> tuple[str, int] | None:
    """``(voice, rungs_used)`` for one more round of draws of a clip every draw failed,
    or None when the clip has run out of voices.

    A clip that fails `--n-improv` draws is a text its own voice cannot read: the draws
    differ only by noise seed and guidance, so a sixth one of the same kind buys little.
    A different voice is a genuinely different draw, and it is the one lever left before
    keeping the known-bad original.

    WHICH voice is `voice_assignment.VOICE_LADDER_FR_V2`, walked one rung per exhausted
    round: ``--n-improv`` draws at the clip's own voice, then that many at h_itf_arm or
    h_itf_def, then that many at fr_male or fr_female, then the clip exhausts. Those
    four are the voices the first full QC sweep measured as able to read what the other
    five could not (the rates are quoted where the constant is defined), so each
    escalation is to a MEASURED voice rather than to any other roster entry: a redraw at
    one of the five that fail 40 to 77 pct of long clips is not a repair attempt, it is
    a coin flip. The ladder replaced a single swap over the whole roster on 2026-10-20.

    None, and the clip exhausts as it did in v1, when:

    * ``--no-voice-swap``;
    * the ladder is used up (``voice_tier`` has reached its length), or every voice on
      the next rung is one this clip already failed at;
    * the row carries no ``voice`` (v1 single-voice audio);
    * stage 05 recorded no usable roster beside that audio, or one that does not contain
      the rung's voices. Read from what stage 05 recorded for the directory rather than
      from a list of stage names here, so the rule follows the audio.

    The rung's two voices are chosen between by the shared assignment hash, keyed on the
    clip stem in ``row`` mode and on the GROUP id in ``group`` mode (`03_PARHAF`,
    `04_PARROT`), so every repaired chunk of one document lands on the SAME new reader:
    a document gains at most one reader per rung instead of one per repaired chunk.
    Group mode used to refuse to swap at all, to keep one reader per document. The first
    full sweep then measured 22.6 pct of PARHAF clips over the gates against ~1 pct
    elsewhere, and audio that does not say its text is worse label noise for an ASR
    corpus than a document changing reader once, so the rule is now about keeping that
    change as small as possible rather than forbidding it.

    `cfg.term_voices` (harvested in `make_hooks`) is still honoured as a preference, so
    a rung prefers the voice not already reading the same term when it can afford to.
    """
    if not getattr(cfg, "voice_swap", False):
        return None
    current = row.get("voice")
    if not current:
        return None
    roster = _swap_roster(str(audio_path.parent))
    if roster is None:
        return None
    voices, mode = roster
    imp = row.get("improvement") or {}
    rung = int(imp.get("voice_tier") or 0)
    # The clip STEM (or its group id), not the key synthesis hashed: the voice is
    # recorded on the row (and carried into the manifests by
    # 99_hf_release/03_sync_hotfix_results.py), never re-derived, so it only has to be
    # stable per clip. See `ladder_voice`.
    key = str(row.get("group_id") or audio_path.stem) if mode == "group" \
        else audio_path.stem
    avoid = (getattr(cfg, "term_voices", None) or {}).get(row.get("group_id")) or ()
    voice = _voices.ladder_voice(key, rung, roster=voices, avoid=avoid,
                                 tried=(current, imp.get("swap_voice")))
    return None if voice is None else (voice, rung + 1)


def finish_or_swap(row: dict, result: dict, orig: Scores, imp: dict,
                   cfg: SimpleNamespace, stats: dict) -> str:
    """Resolve a clip no draw could repair: one more round at the next voice on the
    escalation ladder when the clip has one left (`swap_voice_for`), else exhausted.

    Returns the status written, so a caller can tell the two apart. Either way the
    clip's audio is untouched: a swap only queues more draws, and the original still
    ships if those fail too. Every state is written through `set_improvement`, so the
    swap marker and the rung count survive the pending_stt hop in between.
    """
    attempts = imp.get("attempts", 0)
    audio_path = result.get("audio_path")
    nxt = swap_voice_for(row, Path(audio_path), cfg) if audio_path else None
    if nxt is None:
        set_improvement(row, "exhausted", orig, attempts=attempts)
        stats["exhausted"] += 1
        return "exhausted"
    swap, rung = nxt
    rungs = len(_voices.VOICE_LADDER_FR_V2)
    # The voice that just failed is the previous rung's, not the row's: the row still
    # names the voice of the audio on disk, which is what it shipped with.
    failed = imp.get("swap_voice") or row.get("voice")
    set_improvement(row, "pending_tts", orig, swap_voice=swap, voice_tier=rung)
    stats["swapped"] += 1
    if rung >= rungs:
        # The last rung, so this clip either comes back fixed or exhausts. Counted
        # apart because it is the number that says whether the ladder is long enough.
        stats["swapped_last"] += 1
    logger.info(
        f"SWAP {Path(audio_path).name}: {failed} -> {swap} (ladder rung {rung}/{rungs})"
        f" after {attempts} attempt(s) left it at {_fmt_scores(orig)}; queued for one "
        f"more round of draws with the new voice"
    )
    return "pending_tts"


def is_bad(scores: Scores, cfg: SimpleNamespace) -> bool:
    """Whether a reading fails quality: ANY of the four gates. The whole clip is off
    (``cer >= --cer-threshold``), or only its ending is (``tail >=
    --tail-cer-threshold``), or only one stretch of it is (``window >=
    --window-cer-threshold``), or a contiguous span of the reference went unsaid
    (``bad_run >= --bad-run-threshold``). The three local gates are looser than the
    whole-clip one because their windows are shorter, hence noisier; ``bad_run`` is a
    character count rather than a rate because the defect it names (a phrase skipped
    mid-clip) has a size, and dividing it by the clip's length is exactly what hides
    it."""
    return ((scores.cer is not None and scores.cer >= cfg.cer_threshold)
            or (scores.tail is not None and scores.tail >= cfg.tail_cer_threshold)
            or (scores.window is not None and scores.window >= cfg.window_cer_threshold)
            or (scores.bad_run is not None and scores.bad_run >= cfg.bad_run_threshold))


def _fmt_scores(scores: Scores) -> str:
    """`cer=0.123 tail=0.456 window=0.789 bad_run=42` for the logs, each score omitted
    when the reading could not produce it, and ``bad_run`` also when it is 0 (nothing
    went missing at all, which is the common case)."""
    cer_s = f"{scores.cer:.3f}" if scores.cer is not None else "n/a"
    return (f"cer={cer_s}"
            + (f" tail={scores.tail:.3f}" if scores.tail is not None else "")
            + (f" window={scores.window:.3f}" if scores.window is not None else "")
            + (f" bad_run={scores.bad_run}" if scores.bad_run else ""))


def badness(scores: Scores, cfg: SimpleNamespace) -> tuple[bool, float]:
    """Sort key picking the most favorable of several readings / candidate draws: one that
    passes every gate beats one that does not, and CER breaks the tie within each class.
    A reading whose only defect is a derailed tail therefore loses to a clean one, which is
    the whole point of re-reading a clip before condemning it."""
    return (is_bad(scores, cfg), scores.cer if scores.cer is not None else float("inf"))


def flac_duration(path: Path) -> float | None:
    """Clip length in seconds from the audio header (no full decode)."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception as exc:  # noqa: BLE001 - any unreadable file, just skip refresh
        logger.warning(f"could not probe duration of {path.name}: {exc}")
        return None


def attempt_speed(base_speed: float | None, jitter: float, attempt: int) -> float | None:
    """Speed for a given retry: cycle 0, +j, -j, +2j, -2j ... around the base. Returns
    None (knob omitted, server default) when jitter is off and no base speed is set.

    NOT a source of generation variation on the Voxtral backend: vllm-omni applies
    ``speed`` to the finished waveform (a torchaudio phase vocoder), so jitter yields
    time-stretched copies of the SAME draw, with vocoder artifacts on top. Off by
    default (``--speed-jitter 0``) and best left there; the seed is the real lever."""
    if jitter <= 0:
        return base_speed
    base = base_speed if base_speed is not None else 1.0
    step = (attempt + 1) // 2
    sign = 1 if attempt % 2 == 1 else -1
    return round(base + sign * step * jitter, 4)


def tts_synthesize(
    tts_url: str,
    text: str,
    *,
    response_format: str,
    seed: int | None,
    voice: str | None,
    speed: float | None,
    model: str | None,
    timeout: float,
    max_retries: int = 4,
    cfg_alpha: float | None = None,
    max_new_tokens: int | None = None,
    dump_tag: str | None = None,
) -> bytes:
    """POST one synthesis request and return the raw audio bytes. Unset knobs are
    omitted so the server keeps its startup defaults (same contract as stage 05).

    The per-draw knobs are sent nested in ``extra_params`` (``cfg_alpha``,
    ``voxtral_noise_seed``), the route both upstream and the forked vllm-omni of the
    CrispASR ``voxtral-tts`` image read; the seed is also sent flat for TADA /
    qwen3-tts. Same wire format as stage 05's ``build_body`` (pinned by
    ``tests/test_tts_wire_format.py``). Why flat alone is not enough upstream:

    * ``cfg_alpha`` is a top-level field there, promoted precisely so a request that
      sets it cannot silently do nothing. Upstream has no such field and its request
      model declares no ``model_config``, so pydantic's default ``extra=ignore`` DROPS
      a flat one without an error, and every draw runs at the server's startup value.
    * ``seed`` exists upstream but cannot reach Voxtral's audio: stage 0 hands vLLM's
      sampler ``fake_logits_for_audio_tokens()`` (a vocab-wide -inf row with one finite
      entry), so sampling params are arithmetically inert there, and the only randomness
      is the flow-matching Gaussian off the global torch RNG. The fork routes the flat
      seed into stage 0's ``extra_args["voxtral_noise_seed"]``, which does pin it.

    So against a STOCK vllm-omni both knobs are silently ignored and the draws differ
    only by global-RNG drift, which is what this stage used to do without knowing it.
    The fork reads the flat fields too, but nesting works on both. One more
    server-side requirement: stage 0's deploy YAML must keep a default
    ``cfg_alpha`` under ``default_sampling_params.extra_args``, or vllm-omni
    sets ``has_sampling_extra_args=False`` and drops every per-request extra.

    The two v2 generation guards ride opposite routes, and neither spelling is a
    style choice either:

    * ``max_new_tokens`` is FLAT. vllm-omni applies it after prompt building, from a
      generic override gated on ``_SAMPLING_MAX_TOKENS_TTS_MODEL_TYPES``, which does
      include ``voxtral_tts``, so the flat field is the one that is read (and it is
      server-capped at 4096 frames, 327.68 s). It is the per-draw frame budget: the
      server stops a looping draw at ~1.3x its text's plausible duration instead of
      generating minutes of syllables that the duration check below then throws away.
    * ``voxtral_dump_codes`` is NESTED, like the other two, and only the forked
      vllm-omni reads it; it makes the server append the codes stage 0 emitted to
      ``<VOXTRAL_CODE_DUMP_DIR>/<tag>.i16``. A stock server merges and ignores it, the
      audio is byte-identical either way, and the absence of a dump has to read as
      "no opinion" rather than as a rejection (see ``_tts.read_codes``).

    Retries transient failures (connection errors, 429, 5xx) with exponential
    backoff, mirroring the STT client. A non-retryable status (e.g. 4xx) stops
    early. On exhaustion raises TTSRequestError so the run aborts rather than
    silently skipping the clip."""
    body: dict = {"input": text, "response_format": response_format}
    for key, value in (("seed", seed), ("voice", voice), ("speed", speed),
                       ("model", model), ("max_new_tokens", max_new_tokens)):
        if value is not None:
            body[key] = value
    extra = {k: v for k, v in (("cfg_alpha", cfg_alpha), ("voxtral_noise_seed", seed),
                               ("voxtral_dump_codes", dump_tag))
             if v is not None}
    if extra:
        body["extra_params"] = extra

    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            code, payload, elapsed = _tts.post_json(f"{tts_url}/v1/audio/speech", body, timeout)
        except Exception as exc:  # noqa: BLE001 - connection/timeout, retry with backoff
            last = exc
        else:
            if code == 200:
                return payload
            last = RuntimeError(
                f"TTS server returned {code} after {elapsed:.2f}s: "
                f"{payload[:300].decode('utf-8', 'replace')}"
            )
            if code not in (429, 500, 502, 503, 504):
                break  # not transient (bad request etc): retrying will not help
        if attempt < max_retries - 1:
            backoff = 2.0 * (2 ** attempt)
            logger.debug(f"TTS retry {attempt + 1}/{max_retries} after {backoff}s: {last}")
            time.sleep(backoff)
    raise TTSRequestError(f"TTS failed after {max_retries} attempts: {last}")


def _synthesize_draft(
    attempt: int, cfg: SimpleNamespace, ident: str, text: str,
    orig_len: int, orig_hash: str, seen_hashes: set[str], voice: str | None = None,
    audio_path: Path | None = None,
) -> dict:
    """Synthesize ONE candidate for the given attempt index (the TTS-only half of a
    regeneration, shared by ``both`` mode's inline loop and ``tts`` mode's draft
    generation). Returns ``{"payload", "seed", "speed", "cfg_alpha", "dur",
    "duplicate", "reject", "reject_detail"}``: ``duplicate`` is True when the draw
    byte-repeats an earlier one (seeds collided), and ``reject`` names the guard that
    threw the draw away before it was worth transcribing, ``None`` when it survived
    them all. Either way the caller skips the draw; ``reject_detail`` is the numbers
    behind the verdict, ready to log.

    The three rejections are the v2 GENERATION guards, run here so that a repair pass
    is held to the same standard as the pass that made the clip (before, a redraw sent
    no frame budget and checked only the global frame cap after the fact, so repairing
    a looping clip could spend 327 s of GPU producing the exact failure stage 05 would
    have killed at 50 s):

    * ``"over"``: the draw ran past the upper bound its text can plausibly take, i.e.
      it was looping rather than reading. The bound is the per-clip budget when
      ``--auto-token-budget`` is on (the normal case, and the same number is sent as
      ``max_new_tokens`` so the server stops there instead of at 327.68 s), else the
      server's global frame cap, where it means the text is simply too long.
    * ``"short"``: the draw came back below the floor its text can be read in, so part
      of the text was never spoken. Checked whether or not the budget is on, since it
      needs no cooperation from the server, and rejecting here also saves the STT call.
    * ``"derailed"``: the draw's own semantic codes say it collapsed, held one code or
      looped a unit, which is the half of the defects a duration check cannot see (a
      plausible length for its text, and still not a reading). Only when
      ``--code-dump-dir`` is set AND the server is the patched fork; a missing dump
      reads as "no opinion" and rejects nothing.

    All three verdicts come from the stage-05 module imported here as ``_tts``, which
    reads the bands from ``utils/frame_budget.py`` and the code thresholds from
    ``utils/code_loops.py``. Nothing is re-derived: a draw rejected here is exactly a
    clip ``05_generate_audio/02_audit_durations.py`` would delete from disk.

    ``audio_path`` is the clip being repaired, used for the dump tag and to put the
    code-QC row beside the audio (the same ``code_qc_sidecar.jsonl`` stage 05 writes,
    tagged ``"stage": "redo"``). None skips the sidecar, not the gate.

    ``voice`` is the voice THIS clip was originally read by, from the manifest row's
    ``voice`` field (v2 spreads the corpus over nine voices, one per term/document).
    A regeneration has to use it, or the replacement clip would be spoken by someone
    else while the manifest, the Parquet and any per-voice evaluation still say the
    original name: the voice column is re-derived from the group key, so it cannot
    follow a regeneration. None falls back to ``--tts-voice``, which is the v1 case
    (one voice for the whole corpus, no per-row record).

    Only the network call is gated by ``tts_sem`` so a clip busy on STT never holds a
    TTS slot. ``tts_synthesize`` retries transient failures itself and raises
    ``TTSRequestError`` on exhaustion, which we let propagate to abort the run (never
    skip silently). Empty / duration-0 audio raises ``GenerationError``; a draw
    byte-identical to the original raises ``SeedDeterminismError`` (the seed had no
    effect), both aborting the run rather than replacing a real clip with silence or
    wasting the whole run producing no change."""
    seed = cfg.start_seed + attempt
    speed = attempt_speed(cfg.tts_speed, cfg.speed_jitter, attempt)
    # Walk the TTS guidance DOWN on each re-attempt: cfg_alpha starts at
    # --cfg-alpha-start (the value the dataset was generated at, the top of the usable
    # band) and moves by --cfg-alpha-step each attempt, which is negative, so the draws
    # descend toward --cfg-alpha-floor. Descending rather than ascending because the
    # measured band is 1.2-2.0 and 2.0 is its TOP: there is room below the generation
    # value and almost none above it, and a clip that derailed at 2.0 is usually
    # over-guided (clipped, rushed, distorted) rather than under-guided.
    #
    # The floor is a hard clamp, not a hint, because --n-improv is user-settable: five
    # draws from 2.0 at -0.2 land exactly on 1.2, but ten draws would walk into 0.2,
    # i.e. CFG off, where the flow-matching decoder stops following the text and
    # invents phonetically plausible words. That is the defect this stage repairs, so
    # it must never be a draw. None start -> knob omitted (server keeps its default).
    # utils/cfg_sidecar owns the arithmetic: stage 05 walks the same rungs for its
    # duration rejections, and two copies of "round(start + step * attempt)" would
    # drift the moment one of them learned about, say, a per-voice floor.
    # getattr with the shared defaults rather than cfg.<field>, because the synthetic
    # namespaces the tests build (and any caller reusing this function outside main())
    # carry only the knobs they care about; before the ladder moved to utils/, the
    # conditional expression here read those fields lazily and so tolerated their
    # absence by accident.
    cfg_alpha = ladder_cfg_alpha(
        start=getattr(cfg, "cfg_alpha_start", CFG_ALPHA_START), attempt=attempt,
        step=getattr(cfg, "cfg_alpha_step", CFG_ALPHA_STEP),
        floor=getattr(cfg, "cfg_alpha_floor", CFG_ALPHA_FLOOR))
    # The per-clip duration band this draw is held to, and the frame budget derived
    # from its slow end. budget_terms / budget_tokens are stage 05's, which read
    # utils/frame_budget.py: the per-voice bands have ONE home, because a copy here
    # would drift the day a voice is re-measured, and the two stages would then reject
    # different clips while both claiming to apply "the" bound.
    #
    # getattr rather than cfg.auto_token_budget, defaulting to OFF rather than to the
    # CLI's ON, exactly as stage 05's build_body does it: a synthetic namespace (a
    # test, any caller outside main()) must not have to carry the field, and silently
    # budgeting a caller that never asked for one would change what it sends. main()
    # always sets it, so a real pass does get the budget.
    budget_on = getattr(cfg, "auto_token_budget", False)
    fixed, slow, fast = _tts.budget_terms(cfg, voice or cfg.tts_voice)
    # The code gate is on only when the operator gave the HOST path of the server's
    # $VOXTRAL_CODE_DUMP_DIR (the two are different strings for the same directory, so
    # nothing can derive it). The tag must be unique per ATTEMPT because the server
    # APPENDS to <tag>.i16: a shared tag would read attempt 2 as attempt 1 twice over
    # and condemn a clean draw. The seed is in it as well as the attempt index, so a
    # retry run with a different --start-seed cannot collide with a crashed run's
    # leftover either, and `redo` keeps these dumps apart from stage 05's
    # `<stem>__t<attempt>` in the directory both stages share.
    stem = audio_path.stem if audio_path is not None else _tts.normalize_term(ident)
    dump_tag = (f"{stem}__redo{attempt}s{seed}"
                if getattr(cfg, "code_dump_dir", None) else None)
    with cfg.tts_sem:
        payload = tts_synthesize(
            cfg.tts_url, text,
            response_format=cfg.tts_format, seed=seed, voice=voice or cfg.tts_voice,
            speed=speed, model=cfg.tts_model, timeout=cfg.tts_timeout,
            max_retries=cfg.tts_max_retries, cfg_alpha=cfg_alpha,
            max_new_tokens=(_tts.budget_tokens(text, fixed, slow) if budget_on else None),
            dump_tag=dump_tag,
        )
    # Read the dump (which also unlinks it) on every path out of here, including the
    # two aborts below: at 925 B per second of audio, leaving them behind would cost
    # ~11.7 GB over a corpus-wide pass and put 600k files in one directory.
    codes = _tts.read_codes(cfg, dump_tag)

    # A generation must never be empty or duration-0: that would mean replacing a
    # real clip with silence / nothing. Abort the run so the user notices.
    if not payload:
        raise GenerationError(f"{ident}: TTS returned empty audio (0 bytes) at seed {seed}")
    dur = _tts.payload_duration(payload)
    if dur is not None and dur <= 0:
        raise GenerationError(f"{ident}: TTS returned duration-0 audio at seed {seed}")

    # Both duration bounds, from the one classifier stage 05 and 02_audit_durations.py
    # use, so a draw rejected here is exactly a clip those two would reject. Why a
    # duration is the only observable at all: /v1/audio/speech has no finish_reason, so
    # a draw that stopped for LENGTH is indistinguishable from one that ended on its
    # own except by where it landed. With the budget on, the bound to compare against
    # is that per-clip budget and not the server's global cap, which is the whole point:
    # the clip that ran long for ITS text is the looping one, and the 327.68 s cap only
    # catches the worst of them. The floor is checked either way, since it needs no
    # cooperation from the server.
    cap = _tts.budget_seconds(text, fixed, slow) if budget_on else cfg.max_audio_seconds
    # max() only to keep the ratio in the message finite when the floor is disabled.
    floor = max(1e-9, _tts.floor_seconds(text, fast))
    verdict = _tts.classify_duration(
        text, dur, budget=None if budget_on else cfg.max_audio_seconds,
        fixed=fixed, slow_chars_per_second=slow, fast_chars_per_second=fast,
        tolerance=_tts.TRUNCATION_TOLERANCE_S)
    derailed = codes is not None and codes[0].is_bad
    # One sidecar row per attempt, accepted or not, beside the clip and before any
    # verdict is acted on, in the same file stage 05 writes (`stage` tells them apart).
    # A file holding only the rejects could not answer "what do the features look like
    # on the draw that WON", which is the question that lets these thresholds, which
    # were calibrated on 30 takes, be re-decided later without regenerating anything.
    if codes is not None and audio_path is not None:
        _tts.record_code_take(
            audio_path.parent, stem=audio_path.stem, take=codes[0], attempt=attempt + 1,
            accepted=(verdict == "ok" and not derailed),
            track=codes[1] if getattr(cfg, "store_semantic_codes", True) else None,
            voice=voice or cfg.tts_voice, cfg_alpha=cfg_alpha, stage="redo", seed=seed)
    # Spelled out because these lines are what a grep over a multi-day log has to
    # answer with: how far over or under, against which bound, and for how much text.
    # The ratio is the honest severity measure (2.0x is a loop, 1.05x is a long take).
    voice_s = voice or cfg.tts_voice or "server default voice"
    reject = detail = None
    if verdict == "over":
        reject, detail = "over", (
            f"{dur:.2f}s of audio for {len(text)} chars ({voice_s} at {slow:.1f} "
            f"chars/s), {'budget' if budget_on else 'server frame cap'} {cap:.2f}s "
            f"({dur / cap:.2f}x), "
            + ("ran out of frames while still talking, so the draw was looping rather "
               "than reading" if budget_on else
               "stopped for length, so the text is too long for the backend to speak "
               "in full and no seed can fix this clip: re-chunk the source text"))
    elif verdict == "short":
        reject, detail = "short", (
            f"{dur:.2f}s of audio for {len(text)} chars, floor {floor:.2f}s "
            f"({dur / floor:.2f}x), i.e. {len(text) / max(dur, 1e-9):.0f} chars/s where "
            f"the fastest take ever measured for {voice_s} reaches {fast / 1.45:.1f}, "
            f"so part of the text was not spoken")
    elif derailed:
        reject, detail = "derailed", (
            f"its own codes say {','.join(codes[0].verdicts)} over "
            f"{codes[0].duration_s:.2f}s of audio for {len(text)} chars: a plausible "
            f"length for the text and still not a reading of it")

    # Seed check: only bother hashing when the length matches the original (a cheap
    # gate). A byte-identical regen means the seed did not take effect, so we crash
    # the whole run rather than waste it producing no change.
    h = hashlib.sha1(payload).hexdigest()
    if len(payload) == orig_len and h == orig_hash:
        raise SeedDeterminismError(ident, seed)
    duplicate = h in seen_hashes
    if not duplicate:
        seen_hashes.add(h)
    return {"payload": payload, "seed": seed, "speed": speed, "cfg_alpha": cfg_alpha,
            "dur": dur, "duplicate": duplicate, "reject": reject,
            "reject_detail": detail}


#: The guards ``_synthesize_draft`` can throw a draw away on, in the order it checks
#: them. Used to build the per-clip tallies, so adding a guard cannot leave a counter
#: behind that nobody increments.
REJECT_KINDS = ("over", "short", "derailed")


def reject_summary(ident: str, counts: dict[str, int], attempts: int) -> str | None:
    """One line for the draws a redo threw away before they were ever scored, or None
    when every draw survived the guards.

    Worth its own line rather than being left to the per-draw warnings: the per-draw
    ones say what happened to one seed, this one says whether the clip is repairable
    at all. Five "over" rejections out of five attempts is not bad luck, it is a clip
    whose text no seed can fit, and the operator reading a multi-day log needs that
    distinction without counting WARNING lines by hand."""
    total = sum(counts.get(kind, 0) for kind in REJECT_KINDS)
    if not total:
        return None
    detail = ", ".join(f"{counts[kind]} {kind}" for kind in REJECT_KINDS
                       if counts.get(kind))
    return (f"{ident}: {total}/{attempts} draw(s) discarded by the generation guards "
            f"({detail}); see the per-draw lines above for the numbers")


def _score_audio_file(
    audio_file: Path, cfg: SimpleNamespace, source: str | None, target: str | None,
    duration: float | None = None,
) -> dict | None:
    """Transcribe an audio file and score the reading on all four gates
    (``score_reading``; the tail only when ``duration`` says the draw is long enough):
    the STT-only half of a regeneration, shared by ``both`` mode's inline loop and
    ``tts`` mode's draft scoring. Returns ``{"transcript", "scores", "stt_seconds"}``,
    or None when the clip transcribed blank.

    A blank transcription of a fresh DRAW is a bad draw, not the dataset clip: the
    caller discards it and tries the next (the whole point of trying several seeds).
    Persistent network failure is a ``TranscriptionError``, which propagates and
    aborts the run; a blank ORIGINAL clip is fatal, handled in ``combined_worker``.
    Only the call is gated by ``stt_sem`` so a clip busy on TTS never holds an STT
    slot."""
    try:
        with cfg.stt_sem:
            transcript, stt_seconds = _stt.transcribe_timed(
                audio_file, cfg.stt_endpoint, cfg.stt_token, cfg.stt_model, cfg.stt_temperature,
                cfg.stt_language, cfg.stt_response_format, cfg.stt_extra_params,
                cfg.stt_timeout, cfg.stt_max_retries,
            )
    except _stt.EmptyTranscriptError:
        return None
    return {"transcript": transcript,
            "scores": score_reading(transcript, source, target, duration),
            "stt_seconds": stt_seconds}


def run_regeneration(
    pos: int, ident: str, audio_path: Path, text: str,
    source: str | None, target: str | None, orig: Scores, cfg: SimpleNamespace,
    voice: str | None = None,
) -> dict:
    """Regenerate one bad clip, trying up to --n-improv draws to clear EVERY gate (CER,
    tail CER, window CER and the unsaid-run length, see ``is_bad``), stopping early the
    moment a draw does. If no draw clears them, keep the lowest-CER draw when it still
    beats the original by --min-improvement, otherwise keep the original. Returns a dict;
    when a draw wins it carries the winning ``payload`` (bytes) + metadata, else
    ``payload`` is None. Raises SeedDeterminismError if a regen comes back byte-identical
    to the original."""
    try:
        orig_bytes = audio_path.read_bytes()
    except OSError as exc:
        logger.warning(f"{ident}: cannot read {audio_path} to regenerate: {exc}")
        return {"payload": None, "attempts": 0, "all_duplicates": False}
    orig_len = len(orig_bytes)
    orig_hash = hashlib.sha1(orig_bytes).hexdigest()

    seen_hashes = {orig_hash}
    dup_count = 0
    rejects = dict.fromkeys(REJECT_KINDS, 0)
    attempts = 0
    best = None  # best draw seen so far (kept if none clears the gates)
    logger.info(
        f"REDO {ident}: {_fmt_scores(orig)} over threshold (cer {cfg.cer_threshold} / "
        f"tail {cfg.tail_cer_threshold} / window {cfg.window_cer_threshold} / "
        f"bad_run {cfg.bad_run_threshold}), up to {cfg.n_improv} attempt(s)"
    )
    for attempt in range(cfg.n_improv):
        attempts = attempt + 1
        # Synthesize one candidate (TTS only). Empty / duration-0 audio and a draw
        # byte-identical to the original abort the run (GenerationError /
        # SeedDeterminismError); a draw that merely repeats an earlier one is skipped.
        draft = _synthesize_draft(attempt, cfg, ident, text, orig_len, orig_hash,
                                  seen_hashes, voice=voice, audio_path=audio_path)
        seed, speed, cfg_alpha, dur = draft["seed"], draft["speed"], draft["cfg_alpha"], draft["dur"]
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if draft["reject"]:
            rejects[draft["reject"]] += 1
            logger.warning(
                f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
                f"{draft['reject']}: {draft['reject_detail']}, discarding draw"
            )
            continue
        if draft["duplicate"]:
            dup_count += 1
            logger.debug(f"{ident}: attempt {attempt + 1} repeated an earlier draw, skipped")
            continue

        # Score the fresh clip (STT): write to a temp file (transcribe reads a path),
        # transcribe, delete. A unique temp name keeps concurrent clips isolated. A
        # blank transcript is a bad draw (discard and try the next), never fatal here.
        tmp = Path(tempfile.gettempdir()) / f"improve_{os.getpid()}_{pos}_{attempt}.{cfg.tts_format}"
        try:
            tmp.write_bytes(draft["payload"])
            scored = _score_audio_file(tmp, cfg, source, target, dur)
        finally:
            tmp.unlink(missing_ok=True)
        if scored is None:
            logger.info(
                f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> transcribed empty, discarding draw"
            )
            continue

        transcript, stt_seconds = scored["transcript"], scored["stt_seconds"]
        cand = scored["scores"]
        # Real-time factor of this draw's transcription (STT seconds / clip seconds).
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf_s += f" rtf={stt_seconds / dur:.3f}"
        if cand.cer is None:
            logger.info(f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> cer=n/a{rtf_s}")
            continue
        # Keep the best draw so far (one that clears both gates beats one that does not,
        # CER breaks the tie), so we can fall back to it if none clears them.
        is_best = best is None or badness(cand, cfg) < badness(best["scores"], cfg)
        if is_best:
            best = {"payload": draft["payload"], "new_transcript": transcript,
                    "scores": cand, "seed": seed, "speed": speed, "cfg_alpha": cfg_alpha}
        below = not is_bad(cand, cfg)
        if below:
            note = " (under every threshold, done)"
        elif is_best:
            note = f" (new best, still over threshold, trying next)"
        else:
            note = " (not better, trying next)"
        logger.info(
            f"  {ident} attempt {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
            f"{_fmt_scores(cand)}{rtf_s}{note}"
        )
        if below:
            # Goal reached: every gate cleared. Stop early with this draw (which is also
            # the best so far, since the original failed at least one gate).
            logger.info(
                f"IMPROVED {ident} {_fmt_scores(orig)} -> "
                f"{_fmt_scores(cand)} "
                f"(attempt {attempt + 1}, seed {seed}, under every threshold)"
            )
            return {**best, "attempts": attempts, "all_duplicates": False}

    # Exhausted every attempt without clearing the gates. Keep the best draw if it still
    # beats the original by --min-improvement, otherwise leave the original untouched.
    # A missing original CER (nothing scorable) leaves nothing to beat, so the original
    # stands, which is what `or 0.0` says: no draw can be 0.02 below zero.
    if best is not None and best["scores"].cer <= (orig.cer or 0.0) - cfg.min_improvement:
        logger.info(
            f"IMPROVED {ident} {_fmt_scores(orig)} -> "
            f"{_fmt_scores(best['scores'])} "
            f"(best of {attempts}, still over threshold)"
        )
        return {**best, "attempts": attempts, "all_duplicates": False}

    logger.info(
        f"EXHAUSTED {ident}: kept {_fmt_scores(orig)} after {attempts} attempt(s)"
        + (f"; best draw {_fmt_scores(best['scores'])} did not beat it"
           if best is not None else "")
    )
    summary = reject_summary(ident, rejects, attempts)
    if summary:
        logger.warning(summary)
    return {"payload": None, "attempts": attempts,
            "all_duplicates": bool(dup_count and dup_count == attempts)}


def draft_path_for(audio_path: Path, seed: int, fmt: str) -> Path:
    """Sidecar path for one persisted TTS candidate, deterministic by seed so a
    re-run of the tts pass overwrites the same file instead of piling up. Lives next
    to the clip, like the ``.bak`` backups."""
    return audio_path.with_name(f"{audio_path.name}.draft{seed}.{fmt}")


def clear_drafts(audio_path: Path) -> None:
    """Delete any candidate sidecars next to a clip (leftovers from an interrupted or
    superseded tts pass) so drafts never accumulate or get mistaken for fresh ones."""
    for p in audio_path.parent.glob(f"{audio_path.name}.draft*"):
        try:
            p.unlink()
        except OSError:
            pass


def generate_drafts(
    pos: int, ident: str, audio_path: Path, text: str, orig: Scores, cfg: SimpleNamespace,
    voice: str | None = None,
) -> dict:
    """TTS-only half of a split regeneration (``--mode tts``): synthesize up to
    ``--n-improv`` candidates for one bad clip and persist each as a sidecar draft
    file, WITHOUT scoring (the STT server is down in this mode). Returns
    ``{"drafts": [...], "attempts": n, "all_duplicates": bool}`` where each draft is
    ``{"path", "seed", "speed", "cfg_alpha", "dur"}``; the next ``--mode stt`` pass
    scores them and promotes the best. Raises SeedDeterminismError / GenerationError
    exactly like the inline loop (byte-identical draw / empty audio abort the run)."""
    try:
        orig_bytes = audio_path.read_bytes()
    except OSError as exc:
        logger.warning(f"{ident}: cannot read {audio_path} to regenerate: {exc}")
        return {"drafts": [], "attempts": 0, "all_duplicates": False}
    orig_len = len(orig_bytes)
    orig_hash = hashlib.sha1(orig_bytes).hexdigest()

    # Drop any stale candidates from a previous, interrupted tts pass before writing a
    # fresh set (draft names are seed-deterministic, but a smaller --n-improv than last
    # time would otherwise orphan the higher-seed files).
    clear_drafts(audio_path)

    seen_hashes = {orig_hash}
    dup_count = 0
    rejects = dict.fromkeys(REJECT_KINDS, 0)
    attempts = 0
    drafts: list[dict] = []
    logger.info(
        f"REDO {ident}: {_fmt_scores(orig)} over threshold, generating up to "
        f"{cfg.n_improv} candidate(s)"
    )
    for attempt in range(cfg.n_improv):
        attempts = attempt + 1
        draft = _synthesize_draft(attempt, cfg, ident, text, orig_len, orig_hash,
                                  seen_hashes, voice=voice, audio_path=audio_path)
        seed, cfg_alpha = draft["seed"], draft["cfg_alpha"]
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if draft["reject"]:
            rejects[draft["reject"]] += 1
            logger.warning(
                f"  {ident} candidate {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> "
                f"{draft['reject']}: {draft['reject_detail']}, not saved"
            )
            continue
        if draft["duplicate"]:
            dup_count += 1
            logger.debug(f"{ident}: attempt {attempt + 1} repeated an earlier draw, skipped")
            continue
        draft_path = draft_path_for(audio_path, seed, cfg.tts_format)
        _stt.atomic_write_bytes(draft_path, draft["payload"])
        drafts.append({"path": str(draft_path), "seed": seed, "speed": draft["speed"],
                       "cfg_alpha": cfg_alpha, "dur": draft["dur"]})
        logger.info(
            f"  {ident} candidate {attempt + 1}/{cfg.n_improv} seed={seed}{alpha_s} -> saved {draft_path.name}"
        )
    summary = reject_summary(ident, rejects, attempts)
    if summary:
        logger.warning(summary)
    return {"drafts": drafts, "attempts": attempts,
            "all_duplicates": bool(dup_count and dup_count == attempts and not drafts)}


def score_and_pick(
    pos: int, ident: str, drafts: list[dict],
    source: str | None, target: str | None, orig: Scores, cfg: SimpleNamespace,
    voice: str | None = None,
) -> dict:
    """STT-only half of a split regeneration (``--mode stt``): transcribe every
    candidate a prior ``--mode tts`` pass persisted, score each, and pick the best.
    No early stop (all candidates already exist), so we keep the best draw (one clearing
    every gate beats one that does not, CER breaks the tie) and promote
    it when it clears them, or when it still beats the original by ``--min-improvement``,
    else keep the original. Returns a dict shaped like ``run_regeneration``'s: a winner
    carries ``payload`` (read back from its draft file) + metadata, else ``payload`` is
    None. The draft files are removed by ``apply_result`` once the pick is applied."""
    best = None
    n = len(drafts)
    logger.info(
        f"SCORE {ident}: {n} candidate(s) from the tts pass, "
        f"original {_fmt_scores(orig)}"
    )
    for i, d in enumerate(drafts):
        draft_path = Path(d["path"])
        seed, cfg_alpha, dur = d.get("seed"), d.get("cfg_alpha"), d.get("dur")
        alpha_s = f" alpha={cfg_alpha}" if cfg_alpha is not None else ""
        if not draft_path.is_file():
            logger.warning(f"  {ident} candidate {i + 1}/{n} seed={seed}: draft file missing, skipping")
            continue
        scored = _score_audio_file(draft_path, cfg, source, target, dur)
        if scored is None:
            logger.info(f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> transcribed empty, discarding draw")
            continue
        transcript, stt_seconds = scored["transcript"], scored["stt_seconds"]
        cand = scored["scores"]
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf_s += f" rtf={stt_seconds / dur:.3f}"
        if cand.cer is None:
            logger.info(f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> cer=n/a{rtf_s}")
            continue
        is_best = best is None or badness(cand, cfg) < badness(best["scores"], cfg)
        if is_best:
            best = {"path": str(draft_path), "new_transcript": transcript,
                    "scores": cand, "seed": seed, "speed": d.get("speed"),
                    "cfg_alpha": cfg_alpha}
        below = not is_bad(cand, cfg)
        note = (" (under every threshold)" if below
                else (" (new best, still over threshold)" if is_best else " (not better)"))
        logger.info(
            f"  {ident} candidate {i + 1}/{n} seed={seed}{alpha_s} -> "
            f"{_fmt_scores(cand)}{rtf_s}{note}"
        )

    # Pick the best candidate: promote it when it clears every gate, or when it still
    # beats the original by --min-improvement; otherwise keep the original.
    if best is not None and (not is_bad(best["scores"], cfg)
                             or best["scores"].cer <= (orig.cer or 0.0) - cfg.min_improvement):
        reached = not is_bad(best["scores"], cfg)
        logger.info(
            f"IMPROVED {ident} {_fmt_scores(orig)} -> "
            f"{_fmt_scores(best['scores'])} "
            + (f"(best of {n}, under every threshold)" if reached
               else f"(best of {n}, still over threshold)")
        )
        return {**best, "payload": Path(best["path"]).read_bytes(), "attempts": n,
                "all_duplicates": False}

    logger.info(
        f"EXHAUSTED {ident}: kept {_fmt_scores(orig)} after {n} candidate(s)"
        + (f"; best draw {_fmt_scores(best['scores'])} did not beat it"
           if best is not None else "")
    )
    return {"payload": None, "attempts": n, "all_duplicates": False}


def _transcribe_original(audio_path: Path, cfg: SimpleNamespace, temperature: float) -> tuple[str, float]:
    """One STT pass over a DATASET clip at ``temperature``, gated by ``stt_sem`` so a clip
    busy on TTS never holds an STT slot, and timed inside the semaphore so the RTF excludes
    the queue wait. Shared by the first transcription and the bad-CER recheck (only the
    temperature differs). EmptyTranscriptError (a dataset clip blank after every retry) and
    TranscriptionError (a dead server) propagate: both are fatal, we never skip a clip we
    cannot transcribe. The recheck caller catches EmptyTranscriptError to fall back to the
    first reading instead of aborting."""
    with cfg.stt_sem:
        return _stt.transcribe_timed(
            audio_path, cfg.stt_endpoint, cfg.stt_token, cfg.stt_model, temperature,
            cfg.stt_language, cfg.stt_response_format, cfg.stt_extra_params,
            cfg.stt_timeout, cfg.stt_max_retries,
        )


def combined_worker(pos: int, rec: dict, cfg: SimpleNamespace, model_key: str, input_file: Path) -> dict:
    """Thread-side per-clip work: transcribe the original (reusing a stored
    transcript on resume), score it, and if it is bad regenerate it. Returns the
    transcription entry plus any improvement payload; the main thread applies the
    disk changes. Raises only to abort the whole run on a fatal condition:
    SeedDeterminismError, GenerationError, or a persistent server failure
    (TranscriptionError / TTSRequestError after backoff)."""
    ident = _stt.short_ident(rec, cfg.audio_key)
    entry: dict = {"text": None, "wer": None, "cer": None}
    # ``action`` tells apply_result which disk mutation to run: "scored" (store a
    # transcription, maybe regenerate inline in both mode / flag pending_tts in stt
    # mode), "drafted" (a tts pass persisted candidates), "picked" (an stt pass scored
    # candidates and chose a winner), or None (nothing to do this pass).
    out = {"pos": pos, "entry": entry, "improvement": None, "scores": Scores(),
           "audio_path": None, "action": None, "drafts": None,
           # The voice every draw of this clip uses this pass: its swap voice once a
           # prior pass decided one, else the voice that generated it. Carried on the
           # result (not re-derived) so apply_result records the voice the audio it
           # promotes was actually read by.
           "voice": voice_for_redraw(rec)}

    audio_path = _stt.resolve_audio(rec, input_file, cfg.audio_key, cfg.audio_root)
    if audio_path is None:
        entry["error"] = f"audio not found for {rec.get(cfg.audio_key)!r}"
        logger.warning(f"{ident}: {entry['error']} (try --audio-root)")
        return out
    out["audio_path"] = str(audio_path)

    source = rec.get("asr_training_source")
    target = rec.get("asr_training_target")
    text = source or target  # what the TTS was fed is asr_training_source
    imp_state = rec.get("improvement") or {}
    status = imp_state.get("status")

    # --- Split mode: tts (generate candidates only, STT server down) ----------
    # Only clips a prior --mode stt pass flagged bad (pending_tts) do work here; we
    # never transcribe, just synthesize and persist candidates for the next stt pass.
    if cfg.mode == "tts":
        if status == "pending_tts" and cfg.n_improv > 0 and text:
            out["action"] = "drafted"
            out["scores"] = scores_from_state(imp_state)
            out["improvement"] = generate_drafts(
                pos, ident, audio_path, text, out["scores"], cfg,
                voice=out["voice"],
            )
        return out

    # --- Split mode: stt, candidates waiting to be scored (TTS server down) ----
    # A prior tts pass left persisted candidates (pending_stt); score them and pick
    # the winner. The original transcript / cer are already on the row from the pass
    # that first flagged this clip, so the original is not re-transcribed.
    if cfg.mode == "stt" and status == "pending_stt":
        drafts = imp_state.get("drafts") or []
        # Only judge a COMPLETE candidate set. Picking the best of 3 when 5 were asked
        # for settles for a draw that may not be the best available, and the clip is
        # then marked improved / exhausted and never revisited. So a set left short by
        # the tts pass (it ran with a smaller --n-improv, or its sidecars have since
        # gone missing) goes BACK to that pass to be made in full.
        # The bar is "the tts pass delivered everything it was asked for", not
        # "len(drafts) == --n-improv": a draw it skipped as a duplicate or on one of the
        # generation guards is skipped deterministically (each attempt's seed is fixed), so
        # demanding a file for it would bounce the clip between the two passes forever.
        missing = [d for d in drafts if not Path(str(d.get("path") or "")).is_file()]
        asked = int(imp_state.get("attempts") or 0)
        if (not drafts) or missing or asked < cfg.n_improv:
            out["action"] = "requeue_tts"
            out["scores"] = scores_from_state(imp_state)
            why = (f"{len(missing)} of its {len(drafts)} candidate file(s) are gone"
                   if missing else
                   f"only {asked} draw(s) were tried, --n-improv is {cfg.n_improv}"
                   if asked < cfg.n_improv else "it has no candidate at all")
            logger.warning(
                f"{ident}: incomplete candidate set ({why}), back to the tts pass "
                f"instead of picking from a partial set"
            )
            return out
        out["action"] = "picked"
        out["scores"] = scores_from_state(imp_state)
        out["drafts"] = drafts
        out["improvement"] = score_and_pick(
            pos, ident, out["drafts"], source, target, out["scores"], cfg,
        )
        return out

    # Reuse a good transcript already stored on this row (resume); else transcribe.
    # A blank stored transcript is treated as missing so it is redone, never kept.
    prior = (rec.get("transcriptions") or {}).get(model_key) or {}
    transcript = prior.get("text")
    if prior.get("error") or not (transcript and transcript.strip()):
        transcript = None
    stt_seconds = None
    # How many STT requests this clip actually costs this pass. 0 means the score was
    # re-derived from the stored transcript alone (a rescore), which is what the
    # RESCORED tag below reports: it is the cheap, offline half of a --rescore run, and
    # seeing it on a clip you expected to be transcribed means something is wrong.
    stt_calls = 0
    if transcript is None:
        if cfg.rescore_only:
            # Offline pass: never open a connection. needs_work already filters these
            # out, so this is just a belt-and-braces guard.
            return out
        # Both fatal, on purpose: EmptyTranscriptError (a dataset clip that stays blank
        # through every retry) and TranscriptionError (a persistent network failure) abort
        # the run. We never skip past a clip we cannot transcribe.
        transcript, stt_seconds = _transcribe_original(audio_path, cfg, cfg.stt_temperature)
        stt_calls += 1

    duration = rec.get("duration") or 0.0
    orig = score_reading(transcript, source, target, duration)

    # Triple-check a bad CER at DIFFERENT STT temperatures before trusting it: a
    # hallucinated transcript (e.g. Whisper's "Sous-titrage ...") scores a spuriously high
    # CER on audio that is actually fine, which would waste a regeneration on a good clip.
    # While the clip reads bad, re-transcribe it until it has been read STT_CHECK_TARGET
    # times in total (the first pass plus whatever a previous run already did, per
    # n_stt_check), each recheck one step up the temperature ladder, keeping whichever
    # reading scored the LOWEST CER. A clip still bad at the end records its reading count
    # so it is never re-read again (it now genuinely goes to regeneration); one a recheck
    # rescued drops the marker and is treated as the good clip it is.
    checks = max(int(prior.get("n_stt_check") or 0), 1)
    while (is_bad(orig, cfg)
           and checks < STT_CHECK_TARGET and cfg.stt_recheck_temperature is not None
           # An offline rescore never re-reads a clip: a newly-bad score is left on the
           # row and the next normal stt pass does the triple-check.
           and not cfg.rescore_only):
        temp = recheck_temperature(cfg.stt_recheck_temperature, checks)
        stt_calls += 1
        try:
            re_transcript, re_seconds = _transcribe_original(audio_path, cfg, temp)
        except _stt.EmptyTranscriptError:
            # A blank recheck is inconclusive (an earlier pass already produced text), not a
            # broken dataset clip: keep the reading we have rather than abort the run.
            re_transcript, re_seconds = None, None
        checks += 1
        re_scores = score_reading(re_transcript, source, target, duration)
        re_s = _fmt_scores(re_scores)
        if re_transcript and badness(re_scores, cfg) < badness(orig, cfg):
            logger.info(
                f"{ident}  RECHECK {checks}/{STT_CHECK_TARGET} temp={temp}: "
                f"{_fmt_scores(orig)} -> {re_s} (kept the better reading)"
            )
            transcript, orig, stt_seconds = re_transcript, re_scores, re_seconds
        else:
            logger.info(
                f"{ident}  RECHECK {checks}/{STT_CHECK_TARGET} temp={temp}: {re_s} "
                f"no better than {_fmt_scores(orig)} (kept the previous reading)"
            )

    entry["text"] = transcript
    entry["cer"] = orig.cer
    # The three local scores under the very keys 01_compute_stt.py writes, and only when
    # the reading produced one: needs_work reads them back to re-judge a row without
    # re-deriving anything, so an absent key must mean "this clip cannot fail that gate"
    # and never "scored under rules that had no such gate".
    for _key, _value in (("cer_tail", orig.tail), ("cer_window", orig.window),
                         ("bad_run_chars", orig.bad_run)):
        if _value is not None:
            entry[_key] = _value
    out["scores"] = orig
    # A clip still bad records how many readings it has had, so it is not re-read past
    # STT_CHECK_TARGET (and a later run tops up a partially checked one); a clip that came
    # back good drops the marker.
    if is_bad(orig, cfg):
        entry["n_stt_check"] = checks
    # Real-time factor for this clip (STT seconds / audio duration), when we actually
    # transcribed it this run (not on a resumed transcript). process_file rolls these
    # up into the bar's rtf= readout.
    dur = rec.get("duration") or 0
    rtf_s = ""
    if stt_seconds is not None:
        entry["stt_seconds"] = round(stt_seconds, 3)
        rtf_s = f" stt={stt_seconds:.2f}s"
        if dur and dur > 0:
            rtf = stt_seconds / dur
            entry["stt_rtf"] = round(rtf, 3)
            rtf_s += f" rtf={rtf:.3f}"
    elif stt_calls == 0:
        # Reused transcript (a rescore): the timing already on the row measured THIS
        # transcript, so carry it forward. The entry is rebuilt from scratch each pass,
        # so without this a rescore would silently erase the run's RTF telemetry from
        # every row it touched.
        for key in ("stt_seconds", "stt_rtf"):
            if prior.get(key) is not None:
                entry[key] = prior[key]

    # One log line per clip scored (like 01_compute_stt.py), so every STT result and
    # its CER is visible above the bars; a clip over the threshold additionally dumps
    # its texts so the derailed ones are readable. Runs in a pool thread; loguru +
    # tqdm.write keep this from tearing the progress bars.
    # RESCORED = this clip cost no STT request, its score was re-derived from the
    # transcript already stored on the row. Tagged in the line itself so a rescore pass
    # is distinguishable from a transcription pass at a glance.
    out["rescored"] = stt_calls == 0
    scores_s = _fmt_scores(orig)
    tag = "RESCORED " if out["rescored"] else ""
    logger.info(f"{ident}  {tag}{scores_s}{rtf_s}")
    if is_bad(orig, cfg):
        logger.warning(
            f"LOW QUALITY {ident} {scores_s}\n"
            f"  source    : {source}\n"
            f"  target    : {target}\n"
            f"  transcript: {transcript}"
        )

    # We transcribed / scored the original this pass. apply_result stores the entry
    # and (both mode) regenerates inline, or (stt mode) flags a bad clip pending_tts.
    out["action"] = "scored"
    if (is_bad(orig, cfg) and cfg.n_improv > 0 and text
            and cfg.mode == "both" and not cfg.rescore_only):
        out["improvement"] = run_regeneration(
            pos, ident, audio_path, text, source, target, orig, cfg,
            voice=out["voice"])
    return out


def next_backup_path(audio_path: Path) -> Path:
    """The ``.bak`` path to write for THIS overwrite, keeping every prior version.

    The first improvement stores the pristine original at ``<clip>.bak``. Every
    later improvement stores the current (about-to-be-overwritten) clip at
    ``<clip>.bak2``, ``<clip>.bak3``, ... (first free slot), so no intermediate
    version is ever lost across repeated ``--force`` runs."""
    base = audio_path.with_suffix(audio_path.suffix + ".bak")
    if not base.exists():
        return base
    n = 2
    while True:
        cand = audio_path.with_suffix(audio_path.suffix + f".bak{n}")
        if not cand.exists():
            return cand
        n += 1


def _apply_improvement(
    row: dict, result: dict, imp: dict, model_key: str, improved_records: dict, cfg: SimpleNamespace
) -> None:
    """Single-threaded disk mutation for a winning clip: back up, overwrite,
    refresh duration, update the row, record the win. Both the backup and the
    replacement are written atomically (temp + fsync + os.replace) so an interrupted
    run can never leave a half-written or corrupted clip."""
    audio_path = Path(result["audio_path"])
    # Always back up the current clip before overwriting it: the first win writes the
    # pristine original to .bak, later wins write the prior version to .bak2/.bak3/...
    # so every intermediate audio is kept.
    backup = next_backup_path(audio_path)
    try:
        _stt.atomic_write_bytes(backup, audio_path.read_bytes())
    except OSError as exc:
        logger.error(f"{audio_path.name}: could not write backup, skipping replace: {exc}")
        return

    # Capture the pre-replacement transcript / duration for the audit record before
    # we overwrite them below.
    orig_transcript = row["transcriptions"][model_key].get("text")
    orig_duration = row.get("duration")

    _stt.atomic_write_bytes(audio_path, imp["payload"])

    if cfg.refresh_duration:
        new_dur = flac_duration(audio_path)
        if new_dur is not None:
            row["duration"] = round(new_dur, 4)

    new = imp["scores"]
    row["transcriptions"][model_key]["text"] = imp["new_transcript"]
    row["transcriptions"][model_key]["cer"] = new.cer
    # Each local score of the winning draw, or no key at all when the draw cannot produce
    # one (a regenerated clip can be shorter than the tail window the original needed; an
    # unsaid run of 0 characters is no run at all). Popped and not merely skipped, because
    # the ORIGINAL's score is sitting under that key and it is what needs_work would
    # re-judge this now-replaced clip on.
    for key, value in (("cer_tail", new.tail), ("cer_window", new.window),
                       ("bad_run_chars", new.bad_run)):
        if value is not None:
            row["transcriptions"][model_key][key] = value
        else:
            row["transcriptions"][model_key].pop(key, None)
    # The bad-CER recheck-count marker described the ORIGINAL reading; the clip now has
    # fresh, good audio + transcript, so drop it (harmless if it was never set).
    row["transcriptions"][model_key].pop("n_stt_check", None)
    # Record the cfg_alpha behind the clip's CURRENT (just-overwritten) audio at the
    # row level, so every clip carries the value that synthesized its shipping audio.
    # A regeneration draw's cfg_alpha is authoritative here (may be None if the knob
    # was omitted); clips that keep their original audio get --original-cfg-alpha in
    # apply_result instead.
    row["cfg_alpha"] = imp.get("cfg_alpha")
    # The voice that actually read the audio now on disk. It only ever changes on a
    # voice swap (see `swap_voice_for`), and recording it here is what keeps the
    # dataset honest: the manifest's `voice` column is RE-DERIVED from the assignment
    # hash every time 99_hf_release/01_build_nemo_manifest.py runs, so a swapped clip
    # is correctly labelled only because 03_sync_hotfix_results.py copies this value
    # back over that re-derivation, exactly as it does for `duration` and `cfg_alpha`.
    won_voice = result.get("voice")
    if won_voice and won_voice != row.get("voice"):
        logger.info(f"VOICE {audio_path.name}: {row.get('voice')} -> {won_voice} "
                    f"(the swapped voice read it well enough to ship)")
        row["voice"] = won_voice
    row["improvement"] = {
        "status": "improved",
        "attempts": imp["attempts"],
        **scores_to_state(result["scores"]),
        "best_cer": new.cer,
        "best_tail": new.tail,
        "best_window": new.window,
        "best_bad_run": new.bad_run,
        "seed": imp["seed"],
        "speed": imp["speed"],
        "cfg_alpha": imp.get("cfg_alpha"),
    }

    now = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    orig = result["scores"]
    orig_cer = orig.cer
    cer_delta = round(orig_cer - new.cer, 4) if orig_cer is not None else None

    key = row.get(cfg.audio_key) or result["audio_path"]
    # Merge onto any prior win for this clip so the pristine baseline and first-seen
    # time survive re-improvement (--force), while this run's before/after is recorded.
    prev = improved_records.get(key) or {}
    index, term = clip_identity(row, cfg.audio_key)
    improved_records[key] = {
        "audio_filepath": row.get(cfg.audio_key),
        "audio_path": result["audio_path"],
        "index": index,
        "term": term,
        # Which voice read the promoted draw, so the audit trail says whether a win
        # came from another seed or from another reader.
        "voice": row.get("voice"),
        "orig_cer": orig_cer,
        "new_cer": new.cer,
        "cer_delta": cer_delta,
        # The other three gates' before / after, so the audit trail says which defect a
        # win actually repaired (a clip flagged for a dropped phrase and a clip flagged
        # for a bad average are the same row here otherwise).
        "orig_tail": orig.tail,
        "new_tail": new.tail,
        "orig_window": orig.window,
        "new_window": new.window,
        "orig_bad_run": orig.bad_run,
        "new_bad_run": new.bad_run,
        "orig_transcript": orig_transcript,
        "new_transcript": imp["new_transcript"],
        "orig_duration": orig_duration,
        "new_duration": row.get("duration"),
        "seed": imp["seed"],
        "speed": imp["speed"],
        "cfg_alpha": imp.get("cfg_alpha"),
        "attempts": imp["attempts"],
        "backup": str(backup),
        # Preserved across reruns so the audit trail is never lost:
        "orig_cer_pristine": prev.get("orig_cer_pristine", orig_cer),
        "n_improvements": prev.get("n_improvements", 0) + 1,
        "first_ts": prev.get("first_ts", now),
        "first_ts_iso": prev.get("first_ts_iso", now_iso),
        "ts": now,
        "ts_iso": now_iso,
    }


def load_existing_improved(path: Path) -> dict[str, dict]:
    """Prior ``.improved.jsonl`` keyed by audio path, so re-runs merge wins."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get("audio_filepath") or rec.get("audio_path")
            if key:
                out[key] = rec
    return out


def clip_identity(row: dict, audio_key: str = "audio_filepath") -> tuple:
    """The ``(index, term)`` pair naming a clip in the dataset it came from.

    A stage's own ``generated_dataset.jsonl`` carries both fields and they are used as-is.
    The NeMo manifests stage 99 builds (what the driver actually runs on) do NOT: they keep
    ``group_id`` / ``item_index`` and drop the term, so reading ``row["term"]`` there wrote
    a null into every audit record. Fall back to the clip's own name, which stage 05 built
    as ``{term_index}_{variant_index}_{slug}`` and is the one per-clip identity a manifest
    row still carries. For PARHAF / PARROT the slug is the document + chunk id rather than
    a term, those stages having none, but it is still what names the clip.
    """
    index, term = row.get("index"), row.get("term")
    if index is not None and term is not None:
        return index, term
    parsed = parse_clip_name(Path(row.get(audio_key) or "").name)
    if parsed is None:
        return index, term
    term_index, _variant_index, slug = parsed
    return (index if index is not None else term_index,
            term if term is not None else (slug or None))


def backfill_audit_identity(improved_file: Path, improved_records: dict) -> int:
    """Fill ``index`` / ``term`` on audit records written before ``clip_identity`` existed,
    which stored two nulls for every manifest-driven win. Derived from each record's own
    ``audio_filepath``, so it needs no join with the scored rows. Returns how many records
    were repaired; the file is rewritten only if that is non-zero (the pass would write it
    at its next flush anyway, but a converged run never flushes)."""
    repaired = 0
    for rec in improved_records.values():
        if rec.get("index") is not None and rec.get("term") is not None:
            continue
        index, term = clip_identity(rec)
        if (index, term) == (rec.get("index"), rec.get("term")):
            continue
        rec["index"], rec["term"] = index, term
        repaired += 1
    if repaired:
        _stt.atomic_write_jsonl(improved_file, list(improved_records.values()))
    return repaired


def original_cfg_alpha_for(rec: dict, cfg: SimpleNamespace, input_file: Path) -> float:
    """cfg_alpha the clip's CURRENT (not regenerated) audio was drawn at.

    ``--original-cfg-alpha`` is the run-wide answer (2.0 for v2), but stage 05 may have
    drawn an individual clip lower: since 2026-10-14 a take rejected for its duration
    is re-drawn one rung down the shared ladder (2.0 -> 1.8 -> 1.6 ...), and the rung
    it landed on is recorded per clip in ``cfg_alpha_sidecar.jsonl`` beside the audio.
    Reading that here is what keeps the published ``cfg_alpha`` column true for those
    clips instead of flattening every un-regenerated row to one value.

    Falls back to ``cfg.original_cfg_alpha`` when the clip has no sidecar row (the
    overwhelmingly common case: well under 1% of clips descend) and when the clip
    cannot be located on disk at all, which is exactly what the column said before
    the sidecar existed.
    """
    path = _stt.resolve_audio(rec, input_file, cfg.audio_key, cfg.audio_root)
    if path is None:
        return cfg.original_cfg_alpha
    return cfg_alpha_for_clip(path, default=cfg.original_cfg_alpha)


def backfill_cfg_alpha(out_file: Path, improved_records: dict, cfg: SimpleNamespace,
                       input_file: Path) -> int:
    """Add the top-level ``cfg_alpha`` to any row in a prior ``.stt.jsonl`` written before
    the field existed, so a plain re-run populates it without needing ``--force``. Returns
    the number of rows backfilled (0 if the file is absent or every row already has it).

    Runs before ``process_file`` on the same output file, so the value it writes is loaded
    back by ``load_existing`` and preserved on skipped (already-resolved) rows. Resolves
    each clip's value the same way the live paths do:

    * a clip whose audio was replaced by a win carries that win's cfg_alpha, taken from
      the ``improvement`` marker (status ``improved``) or, when a later ``exhausted`` marker
      hides it, from the ``.improved.jsonl`` audit (which tracks the audio actually on
      disk); both may legitimately be ``None`` if the knob was omitted at generation;
    * every other clip still has its original audio, so it gets ``--original-cfg-alpha``.
    """
    if not out_file.is_file():
        return 0
    rows: list[dict] = []
    changed = 0
    with out_file.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"backfill: skipping unreadable line {i} in {out_file.name}")
                continue
            rows.append(rec)
            if "cfg_alpha" in rec:
                continue
            imp = rec.get("improvement") or {}
            key = rec.get(cfg.audio_key)
            audit = improved_records.get(key) if key else None
            # A win's cfg_alpha (marker or audit) means the audio on disk is that draw;
            # anything else still has the original audio -> --original-cfg-alpha.
            if imp.get("cfg_alpha") is not None:
                rec["cfg_alpha"] = imp["cfg_alpha"]
            elif audit is not None:
                rec["cfg_alpha"] = audit.get("cfg_alpha")
            else:
                rec["cfg_alpha"] = original_cfg_alpha_for(rec, cfg, input_file)
            changed += 1
    if changed:
        _stt.atomic_write_jsonl(out_file, rows)
    return changed


class Snapshot(NamedTuple):
    """The whole dataset as it stands in a written ``.stt.jsonl``, not just the rows one
    pass touched. Two consumers, one parse: the convergence check the driver loops on,
    and the round line appended to ``rounds.jsonl``."""

    rows: int = 0            # rows the filters let through
    scored: int = 0          # of those, the ones carrying a usable transcript
    bad: int = 0             # scored rows failing a gate under THIS pass's gates
    pending_tts: int = 0     # flagged, waiting for a tts pass to draft candidates
    pending_stt: int = 0     # candidates drafted, waiting for an stt pass to judge them
    improved: int = 0        # a draw replaced the original
    exhausted: int = 0       # flagged, no draw was good enough, original ships
    clean: int = 0           # scored, passes every gate, never carried a marker
    audio_seconds: float = 0.0
    # Score key -> {"n", "p50", "p90"}, or None for a score nothing carried. Both
    # constructors below always pass it; the default is None rather than {} because a
    # mutable default on a NamedTuple is shared by every instance that omits it.
    scores: dict | None = None


#: Gates for a scan whose caller has no cfg (the thin ``count_pending`` wrapper, and any
#: outside caller): the stage's own defaults, read from the script that owns them, so a
#: snapshot taken without a cfg is judged by the same numbers as a pass.
_DEFAULT_GATES = SimpleNamespace(
    cer_threshold=_stt.DEFAULT_CER_GATE,
    tail_cer_threshold=_stt.DEFAULT_TAIL_CER_GATE,
    window_cer_threshold=_stt.DEFAULT_WINDOW_CER_GATE,
    bad_run_threshold=_stt.DEFAULT_BAD_RUN_GATE,
)


def scan_output(out_file: Path, *, model_key: str = "default",
                cfg: SimpleNamespace | None = None,
                categories: set[str] | None = None,
                stem_requests: dict[str, set[str]] | None = None,
                audio_key: str = "audio_filepath") -> Snapshot:
    """One parse of a written output file, returning everything that describes the
    dataset as a whole.

    The scores are read back, never re-derived: this runs after a pass that already
    stored all four on every row it scored, and re-deriving 602k rows costs ~1.8 h
    (jiwer's character alignment) for numbers that are already on disk. A row last
    scored before a gate existed carries None for it, which ``is_bad`` treats as "this
    reading cannot produce that score", exactly as the pass itself does, so the ``bad``
    count here is the same verdict the pass would reach on the same row.

    When ``categories`` or ``stem_requests`` is given only those rows are counted, so a
    restricted run converges on its own subset instead of being held open forever by a
    pending row it is never going to look at, and its round line describes the slice it
    actually worked on rather than the corpus around it.

    Percentiles come from ``utils/rounds_ledger.py``, which is also what renders them."""
    gates = cfg if cfg is not None else _DEFAULT_GATES
    counts = dict(rows=0, scored=0, bad=0, pending_tts=0, pending_stt=0,
                  improved=0, exhausted=0, clean=0)
    audio_seconds = 0.0
    # array('d') rather than a list: 602k rows x 4 scores is ~19 MB packed against
    # ~77 MB of boxed floats, and the only thing done with them is sorting.
    collected = {key: array("d") for key in _ledger.SCORE_KEYS}
    if not out_file.is_file():
        return Snapshot(scores={key: None for key in _ledger.SCORE_KEYS})
    with out_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if categories and (rec.get("category") not in categories):
                continue
            if stem_requests and not _stems.row_is_requested(
                    stem_requests, rec.get(audio_key) or ""):
                continue
            counts["rows"] += 1
            audio_seconds += float(rec.get("duration") or 0.0)
            status = (rec.get("improvement") or {}).get("status")
            if status in counts:
                counts[status] += 1
            reading = (rec.get("transcriptions") or {}).get(model_key) or {}
            text = reading.get("text")
            if reading.get("error") or not (text and text.strip()):
                continue
            counts["scored"] += 1
            for key in _ledger.SCORE_KEYS:
                value = reading.get(key)
                if value is not None:
                    collected[key].append(float(value))
            bad = is_bad(Scores(reading.get("cer"), reading.get("cer_tail"),
                                reading.get("cer_window"), reading.get("bad_run_chars")),
                         gates)
            if bad:
                counts["bad"] += 1
            elif status is None:
                counts["clean"] += 1
    return Snapshot(audio_seconds=audio_seconds,
                    scores={key: _ledger.summarize(vals)
                            for key, vals in collected.items()},
                    **counts)


def count_pending(out_file: Path, categories: set[str] | None = None,
                  stem_requests: dict[str, set[str]] | None = None,
                  audio_key: str = "audio_filepath") -> tuple[int, int]:
    """(pending_tts, pending_stt) counts in a written ``.stt.jsonl``, so the driver /
    main can tell when the alternating passes have fully converged (both zero). A
    pending_tts clip still needs a tts pass to draft candidates; a pending_stt clip
    still needs an stt pass to score them. The two numbers out of ``scan_output``,
    for callers that want nothing else from it."""
    snap = scan_output(out_file, categories=categories, stem_requests=stem_requests,
                       audio_key=audio_key)
    return snap.pending_tts, snap.pending_stt


def record_round(out_file: Path, snap: Snapshot, stats: dict, cfg: SimpleNamespace,
                 wall_s: float) -> dict | None:
    """Append this pass's line to ``rounds.jsonl`` beside the output file.

    One line per pass per dataset, describing the DATASET (from `snap`) and the WORK
    (from the pass's own counters), because the two answer different questions: the
    states say how far along the run is, the work says what this pass actually spent to
    get there. The gates are written down with them, since `bad` is only interpretable
    against the numbers it was judged by, and a run that lowers a gate mid-sweep would
    otherwise look like it had made the corpus worse.

    Best effort: a pass that has just spent hours of GPU must not die because a log file
    could not be appended to, so an OSError is reported and swallowed."""
    entry = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": out_file.name,
        "mode": cfg.mode,
        "wall_s": round(wall_s, 1),
        "audio_hours": round(snap.audio_seconds / 3600.0, 3),
        "states": {"rows": snap.rows, "scored": snap.scored, "bad": snap.bad,
                   "pending_tts": snap.pending_tts, "pending_stt": snap.pending_stt,
                   "improved": snap.improved, "exhausted": snap.exhausted,
                   "clean": snap.clean},
        "scores": snap.scores or {},
        # What this pass did, as opposed to where the dataset now stands.
        "work": {key: stats.get(key, 0) for key in
                 ("queued", "done", "transcribed", "rescored", "improved", "exhausted",
                  "swapped", "swapped_last", "requeued", "unflagged",
                  "incomplete_drafts",
                  "deterministic")},
        "gates": {"cer": cfg.cer_threshold, "tail": cfg.tail_cer_threshold,
                  "window": cfg.window_cer_threshold, "bad_run": cfg.bad_run_threshold,
                  "min_improvement": cfg.min_improvement},
        "n_improv": cfg.n_improv,
        "rescore_only": bool(cfg.rescore_only),
    }
    path = _ledger.ledger_path(out_file.parent)
    try:
        entry["round"], entry["pass"] = _ledger.next_round(
            _ledger.load_rounds(path), entry["dataset"], cfg.mode)
        _ledger.append_round(path, entry)
    except OSError as exc:
        logger.warning(f"could not append to {path}: {exc} (the pass itself is fine)")
        return None
    return entry


def make_hooks(cfg: SimpleNamespace, model_key: str, input_file: Path,
               improved_file: Path, improved_records: dict):
    """Build the (worker, apply_result, needs_work, on_flush) hooks that plug this
    stage's transcribe+improve behavior into 01_compute_stt.py's process_file."""
    # `queued` counts the rows this pass picked up, `done` the ones it actually
    # completed. A split-mode pass that completes nothing has made no progress, which
    # is how the driver knows an stt-only (or tts-only) loop has run dry: waiting for
    # the OTHER mode is not a reason to keep spinning.
    stats = {"improved": 0, "exhausted": 0, "deterministic": 0, "queued": 0, "done": 0,
             # swapped = clips handed to ANOTHER voice after every draw of their own
             # failed (plan item 4). They are not resolved: each is back in the
             # pending_tts queue for one more round, and lands on improved or
             # exhausted after it.
             "swapped": 0,
        # swapped_last = of those, the ones that went to the LAST rung of the ladder
        # (the fr_* bases), i.e. the clips one more rung would have to be invented for.
        "swapped_last": 0,
             # rescored = scored from the transcript already on the row (no STT request);
             # transcribed = a real server round trip. rescored_bad is the subset that now
             # reads bad under the current rules, i.e. what a rescore actually surfaced.
             # unflagged / requeued = pending_tts rows a rescore re-judged: the queue was
             # built under the old rules, so some entries no longer belong in it.
             "rescored": 0, "transcribed": 0, "rescored_bad": 0,
             "unflagged": 0, "requeued": 0,
             # incomplete_drafts = pending_stt clips handed BACK to the tts pass because
             # their candidate set was short of --n-improv (see combined_worker).
             "incomplete_drafts": 0}

    # Third progress bar, the TTS repair backlog: its total is the clips found bad
    # (high CER, sent to regeneration) and it fills as they are successfully improved.
    # The total grows as the STT sweep discovers bad clips, so the gap between n and
    # total shows how far the slow TTS engine is lagging the fast STT pass. Sits below
    # process_file's clip (pos 0) and audio-seconds (pos 1) bars. Only meaningful in
    # 'both' mode (one pass discovers and repairs); the split modes each do only half
    # the work per pass, so the backlog is tracked across passes by the driver instead.
    tts_bar = (
        tqdm(total=0, desc=f"{input_file.name} tts", unit="clip", position=2)
        if cfg.n_improv > 0 and cfg.mode == "both" else None
    )

    def worker(pos, rec):
        return combined_worker(pos, rec, cfg, model_key, input_file)

    def apply_result(pos, result, results):
        stats["done"] += 1
        row = results[pos]
        action = result.get("action")
        imp = result.get("improvement") or {}
        # Carried as one value through every state hop, so the tts pass and the
        # candidate scoring know WHICH gate flagged a clip (its ending, one stretch of
        # it, a phrase it skipped) and not merely that something was wrong.
        orig = result.get("scores") or Scores()

        # tts pass: candidates were generated for a bad clip. Record them (await the
        # next stt pass to score) and do NOT touch the stored transcription.
        if action == "drafted":
            drafts = imp.get("drafts") or []
            if drafts:
                set_improvement(row, "pending_stt", orig,
                                attempts=imp.get("attempts", 0), drafts=drafts)
            else:
                # No usable candidate (unreadable clip / every draw a duplicate): try
                # another voice if the clip has one coming, else give up on it so the
                # alternation can still converge.
                if imp.get("all_duplicates"):
                    stats["deterministic"] += 1
                finish_or_swap(row, result, orig, imp, cfg, stats)
            return result["entry"]

        # stt pass: the candidate set is short of what --n-improv asks for (see
        # combined_worker), so the clip goes back to the tts pass to have it made in
        # full rather than being judged on a partial set. The leftover sidecars stay
        # where they are: the next tts pass clears them before regenerating.
        if action == "requeue_tts":
            set_improvement(row, "pending_tts", orig)
            stats["incomplete_drafts"] += 1
            return result["entry"]

        # stt pass: candidates from a prior tts pass were scored. Promote the winner or
        # exhaust, then delete every candidate file (win or lose).
        if action == "picked":
            # Kept-original fallback: unless a candidate wins (below), the clip still
            # carries the audio the original dataset synthesized, which is
            # --original-cfg-alpha unless stage 05's sidecar recorded a lower rung for
            # this very clip. Guarded by `not in` rather than setdefault so the lookup
            # (which touches the filesystem) only runs for a row that needs it.
            if "cfg_alpha" not in row:
                row["cfg_alpha"] = original_cfg_alpha_for(row, cfg, input_file)
            if imp.get("payload") is not None:
                _apply_improvement(row, result, imp, model_key, improved_records, cfg)
                stats["improved"] += 1
            else:
                finish_or_swap(row, result, orig, imp, cfg, stats)
            for d in result.get("drafts") or []:
                Path(d["path"]).unlink(missing_ok=True)
            return result["entry"]

        # scored (both / stt original transcription): store the transcription entry.
        entry = result["entry"]
        row.setdefault("transcriptions", {})[model_key] = entry
        # Split the tally by what the clip actually cost: a score re-derived from the
        # stored transcript (no server round trip) versus a real transcription.
        if result.get("rescored"):
            stats["rescored"] += 1
            if is_bad(orig, cfg):
                stats["rescored_bad"] += 1
        else:
            stats["transcribed"] += 1
        # Every clip records the cfg_alpha behind its CURRENT audio. A clip that keeps
        # its original audio (passed with low CER, exhausted, or flagged pending_tts)
        # carries the dataset's original synthesis cfg_alpha: --original-cfg-alpha (2.0
        # for v2, 1.3 for v1) unless stage 05's sidecar recorded a lower rung for this
        # clip; a win overwrites this with the draw's cfg_alpha in _apply_improvement.
        # `not in` rather than setdefault, so a prior improved value is never clobbered
        # AND the filesystem lookup is skipped for the rows that already have a value.
        if "cfg_alpha" not in row:
            row["cfg_alpha"] = original_cfg_alpha_for(row, cfg, input_file)

        # A regeneration queue built under the OLD scoring rules is stale: this row was
        # flagged pending_tts by a threshold or a normalization that no longer applies,
        # and the orig_cer stored beside that flag is the number a candidate would later
        # have to beat. Nothing else ever revisits it (an stt pass hands pending_tts to
        # the tts pass and moves on), so a rescore settles it here: a row that now reads
        # clean leaves the queue instead of costing a regeneration, and one that stays bad
        # carries the refreshed score with it, so the promote comparison is against
        # today's number instead of a stale inflated one that any draw would beat.
        # Skipped when this pass regenerates inline (both mode), which resolves the row
        # itself just below.
        regenerates_inline = (cfg.mode == "both" and cfg.n_improv > 0
                              and not cfg.rescore_only)
        # --retry-exhausted opens the same door for the clips that gave up: they are the
        # only other rows holding a score computed under the old rules that nothing would
        # revisit, and the whole point of retrying them is to re-judge first (the ones a
        # scoring change makes clean cost no regeneration at all) and redraw second.
        reopened = ("pending_tts", "exhausted") if cfg.retry_exhausted else ("pending_tts",)
        if (cfg.rescore and not regenerates_inline
                and (row.get("improvement") or {}).get("status") in reopened):
            if is_bad(orig, cfg):
                # Back to the queue rather than left resolved, so the next tts pass draws
                # for it again. The attempt count starts over: what matters downstream is
                # the score a new candidate has to beat, and that is refreshed here.
                set_improvement(row, "pending_tts", orig)
                stats["requeued"] += 1
            else:
                row.pop("improvement", None)
                stats["unflagged"] += 1
            return entry

        if imp.get("payload") is not None:
            # both mode won inline.
            _apply_improvement(row, result, imp, model_key, improved_records, cfg)
            stats["improved"] += 1
            if tts_bar is not None:
                tts_bar.total += 1   # a bad clip, now counted in the backlog...
                tts_bar.update(1)    # ...and successfully repaired, so fill it
        elif is_bad(orig, cfg) and cfg.n_improv > 0 and not cfg.rescore_only:
            if cfg.mode == "both":
                # both mode tried and exhausted inline.
                if imp.get("all_duplicates"):
                    stats["deterministic"] += 1
                if tts_bar is not None:
                    tts_bar.total += 1   # bad clip not repaired (yet): total only
                    tts_bar.refresh()
                finish_or_swap(row, result, orig, imp, cfg, stats)
            else:
                # stt mode: flag the bad clip for the next tts pass to regenerate.
                set_improvement(row, "pending_tts", orig)
        elif (cfg.retry_exhausted and not is_bad(orig, cfg)
              and (row.get("improvement") or {}).get("status") == "exhausted"):
            # Reopened, re-read, and it clears the gates now (a scoring change, or a
            # regenerated clip whose promotion the old rules undersold). Nothing is left
            # to retry, so drop the marker instead of leaving a stale "gave up" on a
            # clip that passes.
            row.pop("improvement", None)
            stats["unflagged"] += 1
        return entry

    def needs_work(out_rec):
        # Category restriction: rows outside the selected categories are never queued,
        # so they pass through untouched (existing transcription / CER preserved). This
        # is what lets a parhaf-only run keep the dictionary/drugs work already computed.
        if cfg.categories and (out_rec.get("category") not in cfg.categories):
            return False

        # Clip restriction (--stems-from), the same filter one clip at a time instead of
        # one subset at a time, and with the same pass-through semantics. It is what lets
        # a second, stronger judge re-score ONLY what the first one flagged: 12k clips
        # rather than 602k, which is the difference between an hour and a day and a half.
        if cfg.stem_requests and not _stems.row_is_requested(
                cfg.stem_requests, out_rec.get(cfg.audio_key) or ""):
            return False

        # --rescore-only: an OFFLINE pass that re-derives stored scores and opens no
        # connection at all, so it can run with both servers down and finishes in
        # minutes. Only a row that already carries a usable transcript qualifies;
        # everything a normal pass would have to transcribe or synthesize (never
        # transcribed, blank, errored, awaiting a tts/stt half) is deliberately left
        # alone for that pass to pick up afterwards from the refreshed score.
        if cfg.rescore_only:
            # A row anywhere in the improvement state machine belongs to the normal
            # passes: pending_stt would drag us into scoring persisted candidates (an
            # STT call), and improved / exhausted are resolved. pending_tts is the one
            # exception: it is a QUEUE ENTRY written under the old rules, its original
            # clip already has a transcript, so refreshing it here costs nothing and
            # keeps the tts pass from regenerating clips that now read clean.
            # --retry-exhausted adds `exhausted` to that exception, for the same reason:
            # its stored score is old-rules too, and re-judging it offline is free.
            reopened = ((None, "pending_tts", "exhausted") if cfg.retry_exhausted
                        else (None, "pending_tts"))
            if (out_rec.get("improvement") or {}).get("status") not in reopened:
                return False
            prior = (out_rec.get("transcriptions") or {}).get(model_key) or {}
            text = prior.get("text")
            if prior.get("error") or not (text and text.strip()):
                return False
            return _stt.resolve_audio(
                out_rec, input_file, cfg.audio_key, cfg.audio_root
            ) is not None

        imp = out_rec.get("improvement") or {}
        status = imp.get("status")

        # tts pass: only clips a prior stt pass flagged bad and awaiting candidates.
        if cfg.mode == "tts":
            return status == "pending_tts" and cfg.n_improv > 0

        # stt (and both) passes below.
        # Candidates a prior tts pass persisted always need this stt pass to score them.
        if cfg.mode == "stt" and status == "pending_stt":
            return True
        # Resolved clips are skipped unless --force, or, for the ones that gave up
        # without a good draw, unless --retry-exhausted reopens them specifically.
        resolved = ("improved",) if cfg.retry_exhausted else ("improved", "exhausted")
        if status in resolved and not cfg.force:
            return False
        # A clip already handed to the tts pass has nothing to do in an stt pass until
        # candidates exist (the tts pass owns it), except under --rescore: the flag that
        # queued it was computed under the old rules, so it is exactly what a rescore
        # exists to re-derive (see the pending_tts branch in apply_result).
        if cfg.mode == "stt" and status == "pending_tts" and not cfg.force and not cfg.rescore:
            return False
        prior = (out_rec.get("transcriptions") or {}).get(model_key)
        if prior is None or prior.get("error"):
            return True  # not transcribed yet -> transcribe (and maybe flag / improve)
        text = prior.get("text")
        if not (text and text.strip()):
            return True  # blank transcript -> redo it, never keep empty
        # --rescore: re-derive every stored score under the CURRENT scoring rules. A
        # change to normalize_for_scoring or to a threshold makes every cer / cer_tail
        # already on disk stale, and nothing else would ever revisit a row that still
        # reads clean. This costs no STT call: the worker reuses the stored transcript
        # (only a row that now reads BAD and has fewer than STT_CHECK_TARGET readings
        # spends a recheck, which is real work either way).
        # Gated on the clip still being on disk: a row whose audio is gone comes back
        # from the worker as an error entry, which would REPLACE a perfectly good stored
        # transcript with nothing. That is fine when the audio is genuinely missing and
        # you are transcribing, but a rescore must never destroy the very transcripts it
        # exists to preserve (e.g. a subset whose clips are being regenerated).
        if cfg.rescore:
            return _stt.resolve_audio(
                out_rec, input_file, cfg.audio_key, cfg.audio_root
            ) is not None
        # Already transcribed: only revisit if it is bad (both mode regenerates inline;
        # stt mode flags it pending_tts). The stored score is READ BACK, not re-derived:
        # it is exactly what this gate would compute (the worker stores score_reading's
        # four numbers), and re-deriving it costs ~3 ms per row (normalize + jiwer,
        # against source and target), i.e. ~22 minutes of silence on a 600k-row corpus
        # before a single clip is transcribed. Scores going stale under a scoring-rule
        # change is what --rescore above is for.
        # The three LOCAL scores are the only thing still derived here, and only when
        # one is MISSING: a row scored before its gate existed must not escape that gate
        # just because the field was never written.
        source = out_rec.get("asr_training_source")
        target = out_rec.get("asr_training_target")
        cer = prior.get("cer")
        if cer is None:
            cer = best_cer(text, source, target)
        tail = prior.get("cer_tail")
        if tail is None and (out_rec.get("duration") or 0.0) > _stt.TAIL_SECONDS:
            tail = best_tail_cer(text, source, target, out_rec.get("duration") or 0.0)
        # The two window scores get the same treatment, but gated on "could either of
        # them even fire?" rather than on the tail's duration, because deriving them
        # costs ~11 ms per row (jiwer's character alignment), i.e. ~1.8 h of silence on a
        # 600k-row corpus before the first clip is transcribed, on every pass. Both are
        # provably redundant on a short reference: the window IS the whole clip once the
        # reference is no longer than WINDOW_CHARS (`width = min(WINDOW_CHARS, len(ref))`
        # in window_score), so window_cer == cer there; and an unsaid run of B characters
        # out of L forces cer >= B / L, so it cannot hide behind a clean CER unless
        # L > B / cer_gate (375 chars at the defaults, ~26 s of speech). The RAW reference
        # length bounds the normalized one from above (normalization only ever shrinks
        # text), so this skips nothing a derivation would have flagged. In practice the
        # sweep writes both keys on every row it scores, which makes this the safety net
        # for rows last scored before the window gates existed, not the common path.
        window, bad_run = prior.get("cer_window"), prior.get("bad_run_chars")
        ref_len = max(len(source or ""), len(target or ""))
        if (window is None and bad_run is None
                and (ref_len > _stt.WINDOW_CHARS
                     or ref_len * cfg.cer_threshold > cfg.bad_run_threshold)):
            window, bad_run = best_window_metrics(text, source, target)
        return is_bad(Scores(cer, tail, window, bad_run), cfg)

    # {group_id: {voice, ...}} for the voice swap, or None when no swap can happen
    # anyway (--no-voice-swap, or a caller whose namespace predates the option): then
    # the harvest below is skipped entirely rather than paying for 602k set inserts.
    term_voices = (cfg.term_voices if getattr(cfg, "voice_swap", False)
                   and getattr(cfg, "term_voices", None) is not None else None)

    def counting_needs_work(out_rec):
        """Tally what this pass picks up, so main can tell a pass that found nothing
        to do from one that did work (see the `queued` / `done` note above).

        Also harvests which voices read each term, which is where a voice swap gets its
        accent spread from (`swap_voice_for`). This is the one hook called on EVERY row
        of the manifest, and process_file calls it during the scan that happens before
        any worker starts, so the index is complete and read-only by the time a thread
        asks for it, and no pass pays for a second parse of 602k rows to learn what
        this scan already sees. Rows this pass will not touch are harvested too, on
        purpose: the other variants of a flagged term are usually not in the flagged
        slice, and they are exactly the voices a swap has to avoid.
        """
        if term_voices is not None:
            voice = out_rec.get("voice")
            group = out_rec.get("group_id")
            if voice and group is not None:
                term_voices.setdefault(group, set()).add(voice)
        if needs_work(out_rec):
            stats["queued"] += 1
            return True
        return False

    def on_flush():
        _stt.atomic_write_jsonl(improved_file, list(improved_records.values()))

    def postfix_fn(entry):
        # Live readout on the bar: the just-finished clip's CER plus running tallies.
        cer = entry.get("cer")
        cer_s = f"{cer:.3f}" if cer is not None else "n/a"
        return f"cer={cer_s} improved={stats['improved']} exhausted={stats['exhausted']}"

    return worker, apply_result, counting_needs_work, on_flush, postfix_fn, stats, tts_bar


def _print_report(ctx, _param, value):
    """``--report`` handler. Eager and `expose_value=False`, the way click's own
    --version works, so it prints and exits BEFORE the required run options
    (--input / --output / --stt-endpoint) are checked: reading the history has to work
    from another tab with no run arguments at hand, while a pass is running."""
    if not value or ctx.resilient_parsing:
        return
    print(_ledger.render_report(Path(value)))
    ctx.exit()


@click.command(context_settings={"show_default": True})
@click.option("--report", is_flag=False, flag_value=".", default=None, metavar="[DIR]",
              is_eager=True, expose_value=False, callback=_print_report,
              help="Print the per-round history and exit, instead of running a pass. "
                   "Every pass appends one line per dataset to `rounds.jsonl` beside "
                   "its output (counts by state, p50 / p90 of each of the four scores, "
                   "audio hours, wall seconds), and this renders them as a table with "
                   "the delta against the first round on record. Bare `--report` looks "
                   "for ledgers in the current directory and one level under it, which "
                   "from 06_hotfixes/ finds improved/ and improved_parrot/ with no "
                   "hardcoded dataset list; `--report DIR` looks under DIR. Read-only, "
                   "so it is safe to run while a pass is working.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path),
              help="Manifest jsonl (or a dir searched recursively for *.jsonl, skipping *.stt.jsonl).")
@click.option("--output", "output_root", required=True, type=click.Path(path_type=Path),
              help="Output root; mirrors input hierarchy, files get .stt.jsonl (same as 01_compute_stt.py).")
@click.option("--audio-root", default=None, type=click.Path(path_type=Path),
              help="Base dir to resolve each row's audio path against. Usually required.")
@click.option("--audio-key", default="audio_filepath", help="JSONL key holding the audio path.")
@click.option("--category", "categories", multiple=True,
              help="Restrict work to rows whose `category` is one of these (repeatable, e.g. "
                   "--category parhaf). Rows in other categories are passed through UNCHANGED "
                   "(existing transcriptions / CER preserved, never re-run), so a focused run "
                   "does not lose the rest of the dataset's work. Convergence (exit 10) is then "
                   "judged over the selected categories only. Default: all categories.")
@click.option("--stems-from", "stems_from", multiple=True,
              type=click.Path(exists=True, path_type=Path),
              help="Restrict work to the clips named in this file (repeatable): either a "
                   "QC sidecar jsonl (its flagged rows' `stem`s) or a plain list of stems, "
                   "one per line. Same two shapes as 03_qc_cpu_stt.py --stems-from, and "
                   "the same pass-through semantics as --category: every other row keeps "
                   "what it already has. This is how a second, stronger judge re-scores "
                   "only what the first one flagged.")
# STT.
@click.option("--stt-endpoint", required=True, help="OpenAI-compatible /v1/audio/transcriptions URL.")
@click.option("--stt-api-token", default="", envvar="STT_API_TOKEN", help="Bearer token; falls back to $STT_API_TOKEN.")
@click.option("--stt-model", default=None, help="STT model name; also the transcriptions key (keyed 'default' when unset).")
@click.option("--stt-temperature", default=0.0, type=float, help="STT temperature.")
@click.option("--stt-recheck-temperature", default=0.5, type=float, help="Base temperature of the bad-CER recheck ladder (default 0.5, distinct from --stt-temperature). A bad clip is re-transcribed until it has been read 3 times in total, at this temperature then twice it (capped at 1.0), keeping the LOWEST CER: a hallucinated transcript at one temperature (e.g. Whisper's 'Sous-titrage ...') scores a false-bad CER on audio that is actually fine, which would waste a regeneration. A clip still bad after its rechecks records its reading count in `n_stt_check` and is never re-read again; one that came back good drops the marker. Pass a negative value to disable the rechecks.")
@click.option("--stt-language", default="fr", help="STT language hint (empty to omit).")
@click.option("--stt-response-format", default="json", help="STT response_format (json / verbose_json / text).")
@click.option("--stt-param", "stt_params", multiple=True, help="Extra STT form field KEY=VALUE (repeatable).")
@click.option("--stt-timeout", default=120.0, type=float, help="STT per-request timeout (seconds).")
@click.option("--stt-max-retries", default=4, type=int, help="STT retries on transient errors.")
# TTS (should match how the dataset audio was originally synthesized).
@click.option("--tts-url", default="http://127.0.0.1:8003", help="Base URL of the TTS server.")
@click.option("--tts-voice", default=None, help="Fallback TTS voice, used only for clips whose manifest row carries no `voice` field (v1 audio). A v2 row is regenerated with ITS OWN voice, since the voice column is re-derived from the group key and cannot follow a regeneration. Unset keeps the server default.")
@click.option("--tts-speed", default=None, type=float, help="Base TTS speed (unset = server default).")
@click.option("--tts-model", default=None, help="TTS `model` field (e.g. mistralai/Voxtral-4B-TTS-2603 if vLLM rejects a modelless request).")
@click.option("--tts-format", default="flac", help="TTS response format / clip extension (should match the dataset, default flac).")
@click.option("--tts-timeout", default=300.0, type=float, help="TTS per-request timeout (seconds).")
@click.option("--tts-max-retries", default=4, type=int, help="TTS retries on transient errors (connection / 429 / 5xx) before the run aborts.")
@click.option("--max-audio-seconds", default=_tts.frame_cap_seconds(), type=float,
              help="TTS frame cap in seconds: a draw landing within half a second of it stopped for LENGTH (cut off mid-sentence) rather than at the end of the text, since /v1/audio/speech has no finish_reason. Such draws are discarded, never promoted. Default is the served 4096 frames at 12.5 Hz; must match the server's actual VOXTRAL_MAX_TOKENS. 0 disables the check.")
# Improvement policy.
@click.option("--mode", type=click.Choice(["both", "stt", "tts"]), default="both",
              help="both = STT+TTS in one pass (needs both servers up, default). Split modes for when the two servers cannot share VRAM: 'stt' transcribes originals, scores any candidates generated by a prior 'tts' pass, and promotes/exhausts them (only STT server up); 'tts' generates candidates for the clips a prior 'stt' pass flagged bad (only TTS server up). Alternate stt/tts runs until 'stt' reports no work left (exit code 10).")
@click.option("--n-improv", default=5, type=int, help="Max regeneration attempts per bad clip (0 = transcribe only, i.e. behave like 01_compute_stt.py).")
@click.option("--cer-threshold", default=_stt.DEFAULT_CER_GATE, type=float, help="A clip is 'bad' (regenerated) when its best CER (vs source or target) is >= this; regeneration also aims to get a draw BELOW this (stops early once one is). This is a SWEEP DEPTH, not a quality verdict: the corpus was first taken to 0.15 (~P99.95, 71 clips of 601k), then to 0.10 (487), and 0.08 is the next ~1550 clips. Below it the diffs are increasingly things no seed can fix (French agreement homophony, Whisper's spelling of rare terms, notation the scorer does not fold), so pair a low gate with --min-improvement or the winner is picked on judge noise rather than on the audio.")
@click.option("--tail-cer-threshold", default=_stt.DEFAULT_TAIL_CER_GATE, type=float, help="Second gate, on the CER of the clip's LAST ~30s only (stored as `cer_tail`, computed on clips longer than that): a clip is 'bad' when EITHER its whole-clip CER reaches --cer-threshold OR its tail CER reaches this. Catches endings a whole-clip average dilutes (truncated at the TTS output cap, derailed, looped). Looser than --cer-threshold on purpose (the window boundary cuts mid-sentence, which adds noise) but only 1.5x: the window is ~500 characters, the size of a whole PARHAF chunk, not a fifth of one. Measured over 23.6k long clips: tail P95 0.076, P99 0.193, so 0.12 sits around P98. Pass a negative value to disable the tail gate.")
@click.option("--window-cer-threshold", default=_stt.DEFAULT_WINDOW_CER_GATE, type=float,
              help="A clip is also 'bad' when the WORST ~15s window of it scores a CER >= "
                   "this (see 01_compute_stt.py's window_score). This is the only one of the "
                   "four gates that sees a phrase dropped in the MIDDLE of a long clip: five "
                   "lost words are ~30 normalized characters, which is 0.14 of a 213-char "
                   "window but 0.04 of a 48s PARHAF chunk, so no whole-clip threshold can "
                   "catch it without flagging every clean long clip too. Looser than "
                   "--cer-threshold because a ~5x shorter window is ~5x noisier. Pass a "
                   "negative value to disable this gate.")
@click.option("--bad-run-threshold", default=_stt.DEFAULT_BAD_RUN_GATE, type=int,
              help="A clip is also 'bad' when the longest CONTIGUOUS run of reference "
                   "characters the recognizer never said is >= this many characters. A count, "
                   "not a rate, because a skipped phrase has a size: dividing it by the clip "
                   "length is what hides it. Measured on 19,958 v1 clips: p50 1, p99 5, p99.9 "
                   "9, max 26, so 30 flagged zero clean clips. Pass a negative value to "
                   "disable this gate.")
@click.option("--min-improvement", default=0.02, type=float, help="When no draw reaches --cer-threshold, keep the lowest-CER draw only if it beats the original CER by at least this. A draw below the threshold is always kept. Non-zero matters once the gate is low: a clip flagged at 0.09 whose best draw lands at 0.088 was not repaired, it was fitted to the judge, and a clean clip's CER moves that much on Whisper's own spelling choices. Pass 0 to accept any gain.")
@click.option("--start-seed", default=43, type=int, help="First seed; each attempt uses start-seed + attempt. Always sent (Voxtral respects the request seed).")
@click.option("--speed-jitter", default=0.0, type=float, help="Optional: also vary speed by +/- this each attempt (e.g. 0.03). Not needed when the seed works.")
@click.option("--cfg-alpha-start", default=CFG_ALPHA_START, type=float, help="TTS cfg_alpha sent on the first re-attempt. Default 2.0, the value the dataset itself is generated at and the TOP of the measured usable band (1.2-2.0); --cfg-alpha-step then walks the later draws DOWN from it. Pass a negative value to omit cfg_alpha entirely.")
@click.option("--cfg-alpha-step", default=CFG_ALPHA_STEP, type=float, help="Added to cfg_alpha on each successive re-attempt, normally NEGATIVE so the ladder descends. With the defaults, five draws walk 2.0, 1.8, 1.6, 1.4, 1.2. Prefer the equals form for a negative value (--cfg-alpha-step=-0.2): this click version accepts the bare form too, but the equals form cannot be read as another option whatever the version.")
@click.option("--cfg-alpha-floor", default=CFG_ALPHA_FLOOR, type=float, help="Hard lower clamp on the ladder (default 1.2, the measured floor of the usable band). It matters because --n-improv is free: without it, ten draws at the default step would reach 0.2, i.e. CFG off, where the decoder stops following the text. Pass a negative value to remove the clamp.")
@click.option("--original-cfg-alpha", default=CFG_ALPHA_START, type=float, help="cfg_alpha the ORIGINAL dataset audio was synthesized with, for every clip stage 05 did NOT record a per-clip value for (a clip whose rung is in cfg_alpha_sidecar.jsonl beside the audio uses that instead, since stage 05 re-draws a duration rejection one rung lower). Default 2.0, the value v2 is generated at (VOXTRAL_CFG_ALPHA in the CrispASR compose since the 2026-10-13 sweep); pass 1.3 when running over v1 audio, which was drawn at that. Written as each clip's `cfg_alpha` when it keeps its original audio (passed / exhausted); a regenerated clip records its winning draw's cfg_alpha instead.")
@click.option("--auto-token-budget/--no-auto-token-budget", default=True,
              help="ON by default. Send each draw a per-clip frame budget "
                   "(max_new_tokens) sized from its text and the VOICE's measured "
                   "reading speed, and reject a draw that comes back past it. This is "
                   "the same guard stage 05 generates under, with the same numbers "
                   "(utils/frame_budget.py, per-voice bands; stage 05's "
                   "--budget-chars-per-second style overrides are deliberately NOT "
                   "repeated here, so there is one place to change a band). Without "
                   "it a looping draw runs to the server's 327.68s cap, i.e. ~6 "
                   "minutes of GPU to produce a take the duration check then throws "
                   "away. The floor check (a draw too SHORT for its text, so part of "
                   "it was never spoken) runs either way: it needs nothing from the "
                   "server.")
@click.option("--code-dump-dir", metavar="DIR", default=None,
              help="HOST path of the directory the TTS server writes code dumps into, "
                   "i.e. the host side of $VOXTRAL_CODE_DUMP_DIR in the CrispASR "
                   "compose (container /cache/cloning/dumps). Setting it turns on the "
                   "code gate on every draw: the draw's semantic codes are read back "
                   "and a derailed one (collapsed onto a few codes, held one, looped a "
                   "unit) is discarded before it costs an STT call, which is the half "
                   "of the defects a duration check cannot see. Off when unset, and a "
                   "stock (unpatched) server writes no dump, which reads as no opinion "
                   "and rejects nothing. One row per attempt is appended to the "
                   "code_qc_sidecar.jsonl beside the audio, tagged \"stage\": \"redo\" "
                   "so a redraw is distinguishable from stage 05's own take.")
@click.option("--store-semantic-codes/--no-store-semantic-codes", default=True,
              help="Store each draw's semantic codes in that sidecar row, base64 int16 "
                   "(default on, ~500 bytes per 15s draw). It is what lets the code "
                   "gate's thresholds, calibrated on 30 takes, be re-decided later "
                   "without regenerating anything. No effect without --code-dump-dir.")
@click.option("--voice-swap/--no-voice-swap", default=True,
              help="ON by default. When every draw of a clip fails, hand the clip to "
                   "the next voice on the MEASURED escalation ladder and give it one "
                   "more round of draws, instead of keeping the known-bad original: "
                   "h_itf_arm or h_itf_def first, then fr_male or fr_female, then "
                   "exhausted (voice_assignment.VOICE_LADDER_FR_V2, the four voices the "
                   "first full QC sweep found able to read what the other five could "
                   "not). The roster and the mode are read from the voice_assignment."
                   "json beside the clips, so the rule follows the audio: one new "
                   "reader per clip for the term stages, one per DOCUMENT for 03_PARHAF "
                   "/ 04_PARROT (every chunk of it that needs repairing gets the same "
                   "new reader). Deterministic, prefers a voice not already reading the "
                   "same term, and the new voice is written to the row's `voice` only "
                   "if its draw actually wins (99_hf_release/03_sync_hotfix_results.py "
                   "carries it into the manifests).")
@click.option("--refresh-duration/--no-refresh-duration", default=True, help="Re-probe and update each row's `duration` when its audio is replaced (keeps stage-99 splits correct).")
@click.option("--force", is_flag=True, help="Re-process clips already marked improved/exhausted in a previous run.")
@click.option("--retry-exhausted", is_flag=True, help="Give the clips marked `exhausted` (flagged bad, no draw good enough) another run: they are treated as if they carried no marker, so a rescore can clear them and an stt pass can send the ones that still read bad back through the tts queue. Unlike --force this leaves the clips that already passed and the ones already improved alone, so a retry costs only the known-weak tail (173 clips after the 0.08 sweep) instead of re-transcribing the corpus. Use it after a scoring-rule change or with a different seed / cfg_alpha ladder.")
@click.option("--rescore", is_flag=True, help="Re-derive all four scores (cer, cer_tail, cer_window, bad_run_chars) for EVERY row that already has a stored transcript, under the current scoring rules, and re-apply the gates. Use after changing normalize_for_scoring or a threshold, which makes every score already on disk stale while leaving the transcripts themselves perfectly valid. Costs no STT call (the stored transcript is reused); only a row that now reads bad and has fewer than 3 readings spends a recheck. Also re-judges the regeneration queue: a pending_tts row that now reads clean drops its marker, one that stays bad gets a refreshed orig_cer.")
@click.option("--rescore-only", is_flag=True, help="Offline rescore pass: like --rescore but makes NO request at all (no STT, no TTS), so it runs with both servers down. Only rows that already carry a transcript are touched: scores are refreshed in place, nothing new is flagged for regeneration and nothing is transcribed (rows the OLD rules already queued are re-judged, since that queue is exactly what a rules change invalidates). Run it once after a scoring-rule change, then run the normal passes, which pick up whatever now reads bad. Implies --rescore.")
# Runtime. STT and TTS run in parallel against their own servers: each has its own
# concurrency cap so a clip busy on TTS never blocks a transcription slot, and vice
# versa. The worker pool is sized stt-parallel + tts-parallel so both can be full.
@click.option("--stt-parallel", default=4, type=int, help="Max concurrent STT (transcription) requests in flight. Size to saturate the STT server.")
@click.option("--tts-parallel", default=4, type=int, help="Max concurrent TTS (regeneration) requests in flight, independent of STT. Size to saturate the TTS server.")
@click.option("--shuffle/--no-shuffle", default=False, help="Randomize processing order (not output order) so the ETA reflects a representative clip-length mix early.")
@click.option("--flush-every", default=50, type=int, help="Atomically write outputs every N processed clips (0 = disable this cap).")
@click.option("--flush-interval", default=0.0, type=float, help="Also flush every this many seconds (e.g. 300 = every 5 min); 0 = disabled.")
@click.option("--limit", default=0, type=int, help="Process at most this many lines per file (0 = all), for testing.")
def main(
    input_path, output_root, audio_root, audio_key, categories, stems_from,
    stt_endpoint, stt_api_token, stt_model,
    stt_temperature, stt_recheck_temperature, stt_language, stt_response_format, stt_params,
    stt_timeout, stt_max_retries, tts_url, tts_voice, tts_speed, tts_model, tts_format, tts_timeout,
    tts_max_retries, max_audio_seconds, mode, n_improv, cer_threshold, tail_cer_threshold,
    window_cer_threshold, bad_run_threshold, min_improvement,
    start_seed, speed_jitter,
    cfg_alpha_start, cfg_alpha_step, cfg_alpha_floor, original_cfg_alpha,
    auto_token_budget, code_dump_dir, store_semantic_codes, voice_swap, refresh_duration,
    force, retry_exhausted,
    rescore, rescore_only,
    stt_parallel, tts_parallel, shuffle, flush_every, flush_interval, limit,
):
    """Transcribe a dataset and regenerate the worst TTS clips, in one combined pass."""
    logger.remove()
    bar_safe_sink(logger)

    cfg = SimpleNamespace(
        audio_root=audio_root, audio_key=audio_key,
        # Empty tuple -> None so "no filter" is a simple falsy check everywhere.
        categories=(set(categories) or None),
        stem_requests=_stems.load_stem_requests(
            stems_from,
            on_file=lambda path, rows, plain: logger.info(
                f"--stems-from {path.name}: {rows} flagged row(s), {plain} bare stem(s)"),
        ),
        stt_endpoint=stt_endpoint, stt_token=stt_api_token, stt_model=stt_model,
        stt_temperature=stt_temperature,
        # A negative recheck temperature disables the bad-CER triple-check.
        stt_recheck_temperature=(None if stt_recheck_temperature < 0 else stt_recheck_temperature),
        stt_language=stt_language,
        stt_response_format=stt_response_format,
        stt_extra_params=_stt._parse_params(stt_params),
        stt_timeout=stt_timeout, stt_max_retries=stt_max_retries,
        tts_url=tts_url.rstrip("/"), tts_voice=tts_voice, tts_speed=tts_speed,
        tts_model=tts_model, tts_format=tts_format, tts_timeout=tts_timeout,
        max_audio_seconds=max_audio_seconds,
        tts_max_retries=tts_max_retries, mode=mode,
        n_improv=n_improv, cer_threshold=cer_threshold,
        # A negative threshold disables that one gate (inf never compares true), which
        # is how a sweep isolates a single gate without touching the other three.
        tail_cer_threshold=(float("inf") if tail_cer_threshold < 0 else tail_cer_threshold),
        window_cer_threshold=(float("inf") if window_cer_threshold < 0 else window_cer_threshold),
        bad_run_threshold=(float("inf") if bad_run_threshold < 0 else bad_run_threshold),
        min_improvement=min_improvement,
        start_seed=start_seed, speed_jitter=speed_jitter,
        # cfg_alpha walks down on each re-attempt; a negative start disables the knob
        # and a negative floor removes the clamp (both are "no value" sentinels, since
        # cfg_alpha itself is only meaningful above 0).
        cfg_alpha_start=(None if cfg_alpha_start < 0 else cfg_alpha_start),
        cfg_alpha_step=cfg_alpha_step,
        cfg_alpha_floor=(None if cfg_alpha_floor < 0 else cfg_alpha_floor),
        # cfg_alpha the original dataset audio was made with, written on every clip that
        # keeps its original audio (99_hf_release/README.md documents it as 1.3).
        original_cfg_alpha=original_cfg_alpha,
        # The v2 generation guards, applied to every draw (see _synthesize_draft). The
        # three names are the ones stage 05 reads off its own namespace, because
        # _tts.budget_terms / dump_path / read_codes are called with THIS cfg: one
        # spelling, or the guard would be silently off here while looking configured.
        auto_token_budget=auto_token_budget,
        code_dump_dir=code_dump_dir,
        store_semantic_codes=store_semantic_codes,
        # Voice swap on exhaustion (see swap_voice_for). `term_voices` is filled during
        # each file's scan, by the needs_work hook, and is only read afterwards: one
        # dict per pass, {group_id: {voice, ...}}, so a swap can avoid the voices
        # already reading the same term.
        voice_swap=voice_swap,
        term_voices={},
        refresh_duration=refresh_duration,
        force=force,
        retry_exhausted=retry_exhausted,
        # --rescore-only is --rescore plus "never open a connection", so it implies it.
        rescore=(rescore or rescore_only),
        rescore_only=rescore_only,
        # Independent concurrency caps: STT calls pass through stt_sem, TTS through
        # tts_sem, so the two servers saturate without waiting on each other.
        stt_sem=threading.Semaphore(stt_parallel), tts_sem=threading.Semaphore(tts_parallel),
    )
    # One thread pool feeds both semaphores; size it so stt_parallel STT calls and
    # tts_parallel TTS calls can be in flight at the same time.
    pool_parallel = stt_parallel + tts_parallel
    model_key = stt_model or "default"

    input_root, input_files = _stt.discover_inputs(input_path)
    if not input_files:
        logger.warning(f"no *.jsonl files found under {input_path}")
        return
    if audio_root is None:
        logger.warning("no --audio-root given; audio will only resolve via absolute / CWD paths")
    if shuffle:
        import random
        random.shuffle(input_files)
    logger.info(
        f"{len(input_files)} input file(s); stt_model={model_key} tts_url={cfg.tts_url} "
        f"n_improv={n_improv} (0 = transcribe only); "
        f"parallel: {stt_parallel} STT + {tts_parallel} TTS = {pool_parallel} workers"
    )
    if cfg.categories:
        logger.info(
            f"category filter: only {sorted(cfg.categories)} "
            f"(rows in other categories are passed through unchanged)"
        )
    if cfg.stem_requests:
        n_stems = sum(len(s) for s in cfg.stem_requests.values())
        logger.info(
            f"clip filter: only {n_stems} stem(s) from --stems-from "
            f"(every other row is passed through unchanged)"
        )
    if cfg.rescore_only:
        if mode == "tts":
            raise click.UsageError("--rescore-only is an stt-side pass; use --mode stt (or both).")
        logger.info(
            "--rescore-only: offline pass, no STT or TTS request will be made. Only rows "
            "with a stored transcript are rescored; nothing is transcribed, flagged or "
            "regenerated. Run the normal passes afterwards."
        )

    # Split modes only: how much improvement work is still outstanding across all
    # files, so the driver knows when the alternating stt/tts passes have converged.
    total_pending_tts = total_pending_stt = 0
    # ... and how much THIS mode picked up and completed, so a driver running one mode
    # on its own (no TTS server up yet, say) can stop when that mode has run dry
    # instead of looping on work only the other mode can clear.
    total_queued = total_done = 0

    for input_file in input_files:
        out_file = _stt.output_path_for(input_file, input_root, output_root)
        improved_file = out_file.with_name(out_file.name.replace(".stt.jsonl", "") + ".improved.jsonl")
        improved_records = load_existing_improved(improved_file)
        # Backfill cfg_alpha onto any rows an earlier (pre-field) run resolved, so a plain
        # re-run populates them without --force. Runs before process_file on the same file,
        # so load_existing carries the values forward onto the skipped rows.
        n_backfilled = backfill_cfg_alpha(out_file, improved_records, cfg, input_file)
        if n_backfilled:
            logger.info(f"{out_file.name}: backfilled cfg_alpha on {n_backfilled} pre-existing row(s)")
        n_named = backfill_audit_identity(improved_file, improved_records)
        if n_named:
            logger.info(f"{improved_file.name}: filled index/term on {n_named} audit record(s)")
        worker, apply_result, needs_work, on_flush, postfix_fn, stats, tts_bar = make_hooks(
            cfg, model_key, input_file, improved_file, improved_records
        )
        t_start = time.monotonic()
        try:
            _stt.process_file(
                input_file, out_file,
                endpoint=stt_endpoint, token=stt_api_token, model=stt_model,
                temperature=stt_temperature, language=stt_language,
                response_format=stt_response_format, extra_params=cfg.stt_extra_params,
                timeout=stt_timeout, max_retries=stt_max_retries, audio_key=audio_key,
                reference_key="", audio_root=audio_root, wer_threshold=1.0,
                cer_threshold=cer_threshold, flush_every=flush_every,
                flush_interval=flush_interval, limit=limit, n_parallel=pool_parallel,
                shuffle=shuffle,
                worker=worker, apply_result=apply_result, needs_work=needs_work, on_flush=on_flush,
                postfix_fn=postfix_fn,
            )
        except SeedDeterminismError as exc:
            logger.error(str(exc))
            raise SystemExit(2)
        except GenerationError as exc:
            logger.error(str(exc))
            raise SystemExit(3)
        except (_stt.TranscriptionError, TTSRequestError) as exc:
            # A server stayed unreachable through every retry / backoff: abort rather
            # than grind out a run full of skipped clips against a dead endpoint.
            logger.error(f"aborting: {exc}")
            raise SystemExit(4)
        except _stt.EmptyTranscriptError as exc:
            # A dataset clip transcribed blank on every retry: fatal so a silent /
            # broken clip is surfaced, never written as an empty transcript.
            logger.error(f"aborting: clip transcribed empty after every retry: {exc}")
            raise SystemExit(5)
        finally:
            if tts_bar is not None:
                tts_bar.close()

        if stats["deterministic"]:
            logger.warning(
                f"{stats['deterministic']} clip(s) drew the SAME audio on every attempt (seeds "
                f"collided) yet none matched the original: the seed space may be too small. "
                f"Raise --n-improv, change --start-seed, or add --speed-jitter."
            )
        # Split by cost, so a pass that was supposed to be a cheap rescore and instead
        # transcribed thousands of clips (or the reverse) is obvious from the summary.
        if stats["rescored"] or stats["transcribed"]:
            logger.info(
                f"{out_file.name}: {stats['rescored']} rescored from stored transcripts "
                f"(no STT call), {stats['transcribed']} transcribed"
            )
        if cfg.rescore_only and stats["rescored_bad"]:
            logger.info(
                f"{out_file.name}: {stats['rescored_bad']} clip(s) now read bad under the "
                f"current rules; they are NOT flagged here, the next stt pass picks them up"
            )
        # The other half of a rules change: entries the OLD rules queued for the tts pass.
        if stats["unflagged"] or stats["requeued"]:
            logger.info(
                f"{out_file.name}: regeneration queue re-judged, {stats['unflagged']} clip(s) "
                f"left it (they read clean now, no regeneration needed), "
                f"{stats['requeued']} stayed with a refreshed score"
            )
        if stats["incomplete_drafts"]:
            logger.info(
                f"{out_file.name}: {stats['incomplete_drafts']} clip(s) went back to the tts "
                f"pass, their candidate set was short of --n-improv ({cfg.n_improv}); they are "
                f"scored once it has been made in full"
            )
        # A swapped clip is neither improved nor exhausted yet: it went back to the
        # queue with another voice, so it is reported on its own line rather than
        # inside a total that says resolved.
        if stats["swapped"]:
            logger.info(
                f"{out_file.name}: {stats['swapped']} clip(s) were handed to another "
                f"voice after every draw of their own failed; they are back in the tts "
                f"queue for one more round (grep SWAP for the per-clip lines)"
            )
        logger.success(
            f"{out_file.name}: {stats['improved']} improved, {stats['exhausted']} exhausted; "
            f"{len(improved_records)} total wins in {improved_file.name}"
        )
        total_queued += stats["queued"]
        total_done += stats["done"]
        # One parse of the written output, for both the round line and the convergence
        # check the driver loops on. `both` mode pays for it too (it used to skip the
        # count entirely): the dataset-wide numbers are the whole point of the ledger,
        # and a scan of 602k rows is seconds against a pass that runs for hours.
        snap = scan_output(out_file, model_key=model_key, cfg=cfg,
                           categories=cfg.categories, stem_requests=cfg.stem_requests,
                           audio_key=cfg.audio_key)
        entry = record_round(out_file, snap, stats, cfg, time.monotonic() - t_start)
        if entry is not None:
            cer = (snap.scores or {}).get("cer") or {}
            cer_s = ("" if cer.get("p50") is None else
                     f", cer p50 {cer['p50']:.4f} p90 {cer['p90']:.4f}")
            logger.success(
                f"{out_file.name}: round {entry['round']} (pass {entry['pass']}, "
                f"{cfg.mode}) recorded in {_ledger.LEDGER_FILENAME}: {snap.bad:,} of "
                f"{snap.scored:,} scored clip(s) still fail a gate{cer_s}"
            )
        if cfg.mode != "both":
            total_pending_tts += snap.pending_tts
            total_pending_stt += snap.pending_stt
            if snap.pending_tts or snap.pending_stt:
                logger.info(f"{out_file.name}: {snap.pending_tts} pending_tts, "
                            f"{snap.pending_stt} pending_stt remaining")

    # Split flow: the driver alternates stt/tts passes. Signal convergence from the stt
    # pass (the only one that resolves clips) so the driver loop can stop: exit 10 when
    # no clip anywhere still needs a tts or stt pass.
    if cfg.mode in ("stt", "tts"):
        logger.info(
            f"pending across all files: {total_pending_tts} pending_tts (need a tts pass), "
            f"{total_pending_stt} pending_stt (need an stt pass)"
        )
    if cfg.mode == "stt" and total_pending_tts == 0 and total_pending_stt == 0:
        logger.success("no improvement work left: dataset has converged")
        raise SystemExit(10)
    # This mode alone has run dry: it either queued nothing or completed nothing, so
    # repeating it cannot change anything. Distinct from 10, because clips are still
    # waiting on the OTHER mode. A driver running a single mode stops here; the
    # alternating driver treats it as "the work is on the other side" and runs the
    # other pass (an stt pass legitimately has nothing to do whenever a previous run
    # stopped right after flagging clips pending_tts), and only stops when BOTH
    # passes report it in one round.
    if cfg.mode in ("stt", "tts") and total_done == 0:
        logger.success(
            f"no {cfg.mode} work left ({total_queued} queued, 0 completed): "
            f"{total_pending_tts} clip(s) still need a tts pass, "
            f"{total_pending_stt} an stt pass"
        )
        raise SystemExit(11)


if __name__ == "__main__":
    main()
