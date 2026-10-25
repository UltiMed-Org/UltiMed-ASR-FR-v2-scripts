#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The shared staleness rule (`utils/clip_freshness.py`).

Stage 06 redraws a clip by OVERWRITING it, so a QC verdict is only ever about the take
it heard. Keying a resume on the stem alone is what let 9,829 acoustic loop flags
describe audio that had already been replaced, a corpus-wide wrong answer that produced
no error of any kind. Every property below is one half of that bug.

This file was written by Claude Code.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

import clip_freshness as cf  # noqa: E402

NOW = 1_700_000_000.0


def _clip(tmp: Path, name: str, mtime: float) -> Path:
    """A file standing in for a clip, with its mtime set to `mtime`."""
    path = tmp / name
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


def test_a_rewritten_clip_is_stale() -> None:
    with tempfile.TemporaryDirectory() as d:
        clip = _clip(Path(d), "redrawn.flac", NOW + 600.0)
        assert cf.is_stale(clip, NOW), "a clip rewritten after the row must be stale"


def test_an_untouched_clip_is_fresh() -> None:
    with tempfile.TemporaryDirectory() as d:
        clip = _clip(Path(d), "quiet.flac", NOW - 600.0)
        assert not cf.is_stale(clip, NOW), "a clip older than its row is still described"


def test_slack_absorbs_the_scan_write_gap() -> None:
    """A row is written a moment AFTER its clip is statted, so sub-second newer is normal.

    Without the slack every row written during a scan would re-queue its own clip on the
    next run and the scan would never converge.
    """
    with tempfile.TemporaryDirectory() as d:
        clip = _clip(Path(d), "justnow.flac", NOW + 1.0)
        assert not cf.is_stale(clip, NOW), "1 s newer is the write gap, not a redraw"
        clip = _clip(Path(d), "later.flac", NOW + 3.0)
        assert cf.is_stale(clip, NOW), "3 s newer is past the slack"


def test_a_row_with_no_timestamp_is_stale() -> None:
    """It cannot prove it heard the current take, so it does not get the benefit."""
    with tempfile.TemporaryDirectory() as d:
        clip = _clip(Path(d), "undated.flac", NOW)
        for bad in (None, "", "not a date", {}, []):
            assert cf.is_stale(clip, bad), f"{bad!r} must not read as a fresh verdict"


def test_a_missing_clip_is_not_stale() -> None:
    """There is no newer audio to describe, so re-scoring it would only fail."""
    with tempfile.TemporaryDirectory() as d:
        assert not cf.is_stale(Path(d) / "gone.flac", NOW)
        assert cf.clip_mtime(Path(d) / "gone.flac") is None


def test_both_sidecar_timestamp_spellings_parse() -> None:
    """The acoustic sidecar stores a float, the STT sidecars an ISO-8601 UTC string.

    One rule has to serve both or each stage invents its own, which is how they drift.
    """
    assert cf.parse_row_ts(NOW) == NOW
    iso = cf.parse_row_ts("2026-10-25T20:26:15Z")
    assert iso is not None and abs(iso - 1_792_959_975.0) < 1.0, iso
    # A naive stamp is read as UTC, not as local time: the sidecars are written in UTC
    # and guessing the writer's timezone would shift a verdict by hours.
    assert cf.parse_row_ts("2026-10-25T20:26:15") == cf.parse_row_ts("2026-10-25T20:26:15Z")
    assert cf.parse_row_ts("whenever") is None


def test_the_rule_is_shared_not_copied() -> None:
    """The acoustic scanner must USE this module, not keep its own slack constant.

    Two copies of a staleness rule is how one stage keeps clearing flags the other
    re-adds, so this asserts the import rather than the behaviour.
    """
    scanner = (Path(__file__).resolve().parent.parent
               / "05_generate_audio" / "04_qc_acoustic_loops.py").read_text()
    assert "from clip_freshness import" in scanner, "scanner no longer shares the rule"
    assert "_MTIME_SLACK = " not in scanner, "scanner grew its own copy of the slack"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
