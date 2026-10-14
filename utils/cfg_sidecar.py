"""Per-clip cfg_alpha: the ladder a retry descends, and the sidecar recording where
each clip actually landed.

Two callers need the SAME numbers and the SAME file format, so neither owns them:
`05_generate_audio/01_generate_audio.py` (walks the ladder when a take is rejected for
its duration, and appends a row for every clip that landed off the run's base value)
and `06_hotfixes/01_recursive_improvement.py` (walks the same ladder for its own
STT-QC failures, and must write the TRUE cfg_alpha of the audio it ships into the
manifest's `cfg_alpha` column, which means reading this sidecar instead of assuming
`--original-cfg-alpha` for every clip it did not regenerate itself).

Why a ladder on a duration rejection
------------------------------------
A take rejected by the frame budget was a runaway: the clip ran far longer than its
text can plausibly be read in, i.e. the decoder looped syllables instead of reading.
Re-requesting the same text is already a genuinely new take (the only randomness in
Voxtral's pipeline is the fresh Gaussian each frame's flow-matching ODE starts from),
and that alone saves most rejected clips: measured 2026-10-14 over a live stage-05
run, 13 rejected attempts became 10 clips saved by a plain same-cfg_alpha retry. What
it cannot save is a text that derails *reproducibly*: `000863_0005_acineux` (339
characters, voice h_frm_chf) ran over budget on all three attempts in two separate
runs. Those clips need a different draw, not another sample of the same one, and
cfg_alpha is the one per-request quality lever Voxtral honours. 2.0 is the TOP of the
measured usable band (1.2-2.0, sweep of 2026-10-13), so the room is below: hence a
DESCENDING ladder, 2.0 -> 1.8 -> 1.6 ..., floored at 1.2 where the flow-matching
decoder starts inventing phonetically-plausible-but-wrong audio.

Why a sidecar was the blocker
-----------------------------
Before this file, stage 05 deliberately retried at the SAME cfg_alpha: the dataset
records one flat `cfg_alpha` per row (99_hf_release/03_sync_hotfix_results.py ships
it), so a clip quietly drawn at 1.6 would have been published as a 2.0 clip. A
provenance lie is worse than a missing clip, which is why the ladder waited for
somewhere to write the truth. That somewhere is this sidecar, which lives beside the
audio (like voice_assignment.json) so it travels with the clips it describes and
survives the clips being regenerated, audited or deleted.

Format: JSON Lines, append-only, one row per accepted clip that landed OFF the run's
base cfg_alpha (so a corpus generated entirely at the default leaves no file at all):

    {"stem": "000863_0005_acineux", "cfg_alpha": 1.6, "voice": "h_frm_chf",
     "attempt": 3, "attempts": 3, "reason": "over budget", "ts": "2026-10-14T12:00:00Z"}

Append-only rather than rewritten because stage 05 writes it from a thread pool
mid-run and must never lose rows to a crash or a Ctrl-C; the readers take the LAST
row for a stem, so a clip deleted by `02_audit_durations.py` and regenerated later
simply gets a newer row. Rows for clips no longer on disk are harmless (lookups are
by stem, driven by the manifest).

Written with Claude Code.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

#: Top of the measured usable band (1.2-2.0, the 2026-10-13 sweep of 16 voices x 10
#: values) and the value the v2 corpus is generated at: VOXTRAL_CFG_ALPHA in the
#: CrispASR compose. Both stages start their ladder here.
CFG_ALPHA_START = 2.0

#: Added to cfg_alpha on each successive re-attempt. NEGATIVE, so the ladder descends:
#: a clip that derailed at the top of the band is over-guided (clipped, rushed,
#: looping), and there is room below 2.0 but almost none above it.
CFG_ALPHA_STEP = -0.2

#: Hard clamp, not a hint. The number of attempts is user-settable, so without it a
#: long ladder walks into cfg_alpha ~0, i.e. CFG off, where the decoder stops
#: following the text and invents phonetically plausible words: exactly the defect
#: these retries exist to repair, so it must never be a draw.
CFG_ALPHA_FLOOR = 1.2

#: Beside the audio, next to voice_assignment.json.
SIDECAR_FILENAME = "cfg_alpha_sidecar.jsonl"

# One lock for all appends: stage 05 writes from a ThreadPoolExecutor, and a line
# written under the lock with a single write() call stays intact. Module-level
# (not per-directory) because contention is nil: only off-base clips write, ~0.1% of
# a run at a few clips per second.
_APPEND_LOCK = threading.Lock()

# dir -> {stem: cfg_alpha}, so stage 06 reads each stage's sidecar once per run
# instead of once per row (602k rows).
_CACHE: dict[Path, dict[str, float]] = {}
_CACHE_LOCK = threading.Lock()


def ladder_cfg_alpha(*, start: float | None, attempt: int,
                     step: float = CFG_ALPHA_STEP,
                     floor: float | None = CFG_ALPHA_FLOOR) -> float | None:
    """cfg_alpha for re-attempt number `attempt`, walking down from `start`.

    Parameters
    ----------
    start
        cfg_alpha of attempt 0. None means the caller never sets cfg_alpha (the knob
        is omitted from the request and the server keeps its own default), in which
        case there is no ladder to walk and None comes back.
    attempt
        0 for the first attempt (which is `start` itself), 1 for the first retry, ...
    step
        Added per attempt; negative descends.
    floor
        Hard lower clamp, or None for no clamp.

    Returns
    -------
    float | None
        The value to send, rounded to 4 decimals so 2.0 + 3 * -0.2 is 1.4 and not
        1.4000000000000004 (the value is logged, recorded in the sidecar and
        published, so float noise would leak into the dataset).

    Examples
    --------
    >>> [ladder_cfg_alpha(start=2.0, attempt=i) for i in range(6)]
    [2.0, 1.8, 1.6, 1.4, 1.2, 1.2]
    >>> ladder_cfg_alpha(start=None, attempt=3) is None
    True
    >>> ladder_cfg_alpha(start=2.0, attempt=2, step=0.0)
    2.0
    """
    if start is None:
        return None
    value = round(start + step * attempt, 4)
    if floor is not None:
        value = max(value, floor)
    return value


def sidecar_path(audio_dir: Path | str) -> Path:
    """Where the sidecar of an audio directory lives."""
    return Path(audio_dir) / SIDECAR_FILENAME


def record_cfg_alpha(audio_dir: Path | str, *, stem: str, cfg_alpha: float,
                     voice: str | None = None, attempt: int | None = None,
                     attempts: int | None = None,
                     reason: str | None = None) -> Path:
    """Append one row saying this clip's audio was drawn at `cfg_alpha`.

    Only call it for a clip that landed OFF the run's base value: a sidecar that
    repeats the default for 600k rows would cost more to read than it says, and an
    absent stem already means "the base value" to every reader.

    Returns the sidecar path (also when the append fails, see below).

    An OSError is swallowed with no re-raise: the clip itself is already on disk and
    valid, so losing its provenance row must not fail the run. The caller logs.
    """
    row = {
        "stem": stem,
        "cfg_alpha": cfg_alpha,
        "voice": voice,
        "attempt": attempt,
        "attempts": attempts,
        "reason": reason,
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = sidecar_path(audio_dir)
    line = json.dumps({k: v for k, v in row.items() if v is not None},
                      ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return path


def load_cfg_alphas(audio_dir: Path | str, *, use_cache: bool = True) -> dict[str, float]:
    """{clip stem: cfg_alpha} from one audio directory's sidecar.

    A missing or unreadable file gives {} (the overwhelmingly common case: no sidecar
    means every clip in that directory was drawn at the run's base value). Unparseable
    lines are skipped rather than raising, because the file is appended to live and a
    run killed mid-write can leave a half line as its last one.

    The LAST row for a stem wins, which is what makes a regenerated clip (new row
    appended) override its own earlier draw.
    """
    key = Path(audio_dir).resolve()
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached
    out: dict[str, float] = {}
    try:
        text = sidecar_path(key).read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            out[str(row["stem"])] = float(row["cfg_alpha"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = out
    return out


def cfg_alpha_for_clip(audio_path: Path | str, *, default: float | None = None,
                       use_cache: bool = True) -> float | None:
    """cfg_alpha of one clip, by its path: its sidecar row's value, else `default`.

    This is the function the manifest side wants: give it the clip's audio path and it
    answers what that audio was actually drawn at, with the stage's own base value as
    the fallback.
    """
    path = Path(audio_path)
    table = load_cfg_alphas(path.parent, use_cache=use_cache)
    return table.get(path.stem, default)


def clear_cache() -> None:
    """Forget the loaded sidecars (tests, and any caller that appends then re-reads)."""
    with _CACHE_LOCK:
        _CACHE.clear()


if __name__ == "__main__":  # pragma: no cover - doctests, run via `uv run`
    import doctest

    failures, _ = doctest.testmod(verbose=False)
    raise SystemExit(1 if failures else 0)
