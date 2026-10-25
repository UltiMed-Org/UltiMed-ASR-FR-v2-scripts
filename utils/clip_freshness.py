"""Is a recorded QC verdict still about the audio that is on disk right now?

Every QC stage writes append-only sidecars keyed by stem, and stage 06 can REPLACE a
clip's audio in place (a redraw keeps the filename). So a verdict is only valid for the
take it heard, and a resume that keys on the stem alone will happily carry a pre-redraw
verdict forever. That is not hypothetical: it shipped. The acoustic scan of 2026-10-21
resumed on stem alone, so 9,829 of its 12,881 loop flags still described audio that the
T3 repair round had already replaced, and a clip could neither lose a stale flag nor
gain a fresh one.

The rule is the same everywhere, which is why it lives here rather than in each stage:
compare the clip's mtime to the moment the row was written, and treat the row as stale
when the clip is newer.

This file was written by Claude Code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

#: Seconds a clip may be newer than its row before the row is called stale. Covers the
#: sub-second gap between statting a clip and writing its row, and filesystem mtime
#: granularity on a copy; a real redraw is minutes or days newer, never one second.
MTIME_SLACK = 2.0


def clip_mtime(path: Path) -> float | None:
    """The clip's mtime, or None when it cannot be statted (deleted, bad mount).

    None is NOT folded into 0.0 here: "the clip is gone" and "the clip is ancient" call
    for opposite decisions (nothing left to rescore vs rescore it), so the caller has to
    see the difference.
    """
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def parse_row_ts(value: object) -> float | None:
    """A sidecar timestamp as a POSIX float, whichever of the two spellings it uses.

    Sidecars written by the acoustic scan carry a float `mtime`; the STT sidecars carry
    an ISO-8601 UTC string `ts` (`2026-10-25T20:26:15Z`). Accepting both is what lets
    one staleness rule serve both, instead of each stage inventing its own.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def is_stale(clip: Path, scored_at: object, *, slack: float = MTIME_SLACK) -> bool:
    """True when `clip` has been rewritten since a row claiming to describe it.

    A row with no usable timestamp is stale by definition: it cannot prove it heard the
    current take, and calling it fresh is exactly the bug this module exists to stop. A
    clip that cannot be statted is NOT stale: there is no newer audio to describe.
    """
    mtime = clip_mtime(clip)
    if mtime is None:
        return False
    written = parse_row_ts(scored_at)
    if written is None:
        return True
    return mtime > written + slack
