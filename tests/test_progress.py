#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tqdm>=4.66",
#   "loguru>=0.7",
# ]
# ///
"""The shared progress plumbing (`utils/progress.py`).

Two things are worth a test here. `resumed_bar` exists because a bar built on the work
that is LEFT claims 0% on a job that is nearly done, which is the one question a bar is
asked, so the arithmetic that places its starting point is the whole point of it.
`bar_safe_sink` exists because three scripts had copied the loguru-through-tqdm.write
line; a helper only replaces those copies honestly if it keeps both shapes they used,
the explicit format and loguru's own default.

Written with Claude Code.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from progress import DEFAULT_FORMAT, bar_safe_sink, resumed_bar  # noqa: E402


def test_a_resumed_bar_starts_where_the_last_run_stopped() -> None:
    """236500 of 533675 clips scored is 44%, not 0%, and ends at 100%."""
    with resumed_bar(total=533675, remaining=297175, desc="01_dictionnary loops",
                     file=io.StringIO()) as bar:
        assert bar.total == 533675, bar.total
        assert bar.n == 236500, bar.n
        bar.update(297175)
        assert bar.n == bar.total, f"{bar.n} != {bar.total}"


def test_a_fresh_bar_starts_at_zero() -> None:
    """The calibration path scores its whole sample, so nothing is carried."""
    with resumed_bar(total=1000, remaining=1000, desc="x", file=io.StringIO()) as bar:
        assert bar.n == 0, bar.n


def test_a_bar_never_starts_negative() -> None:
    """`remaining` larger than `total` is a caller's miscount (a clip appearing between
    the listing and the sidecar read, say). A bar that starts at -3 crashes tqdm's
    rate maths; clamping costs one wrong percent and keeps the run going."""
    with resumed_bar(total=5, remaining=8, desc="x", file=io.StringIO()) as bar:
        assert bar.n == 0, bar.n


def test_the_sink_writes_through_tqdm_so_a_bar_survives_a_log_line() -> None:
    """The reason the helper exists: the sink must go through `tqdm.write`, which parks
    the bar, prints, and redraws. Asserted by capturing tqdm's own write, not by eyeing
    a terminal."""
    import tqdm as tqdm_mod
    from loguru import logger

    seen: list[str] = []
    original = tqdm_mod.tqdm.write
    tqdm_mod.tqdm.write = classmethod(lambda cls, s, **kw: seen.append(s))
    try:
        logger.remove()
        sink_id = bar_safe_sink(logger, colorize=False)
        logger.info("one flagged clip")
        logger.remove(sink_id)
    finally:
        tqdm_mod.tqdm.write = original

    assert len(seen) == 1, seen
    assert "one flagged clip" in seen[0], seen
    # DEFAULT_FORMAT is level then message, with no timestamp: a bar already shows time.
    assert seen[0].startswith("INFO"), seen
    assert ":" not in seen[0].split("|")[0], seen


def test_fmt_none_keeps_loguru_s_own_format() -> None:
    """01_generate_audio.py's sink passed no `format` at all, so its lines carry a
    timestamp and the module name. `fmt=None` is what keeps that output identical."""
    import tqdm as tqdm_mod
    from loguru import logger

    seen: list[str] = []
    original = tqdm_mod.tqdm.write
    tqdm_mod.tqdm.write = classmethod(lambda cls, s, **kw: seen.append(s))
    try:
        logger.remove()
        sink_id = bar_safe_sink(logger, fmt=None, colorize=False)
        logger.info("probing durations")
        logger.remove(sink_id)
    finally:
        tqdm_mod.tqdm.write = original

    assert len(seen) == 1, seen
    line = seen[0]
    assert "probing durations" in line, line
    # loguru's default: "<time> | LEVEL | module:function:line - message".
    assert " | INFO " in line, line
    assert line.split(" | ")[0][:2].isdigit(), line
    assert DEFAULT_FORMAT not in line, "the explicit format leaked into the default path"


def main() -> None:
    test_a_resumed_bar_starts_where_the_last_run_stopped()
    print("test_a_resumed_bar_starts_where_the_last_run_stopped: OK")
    test_a_fresh_bar_starts_at_zero()
    print("test_a_fresh_bar_starts_at_zero: OK")
    test_a_bar_never_starts_negative()
    print("test_a_bar_never_starts_negative: OK")
    test_the_sink_writes_through_tqdm_so_a_bar_survives_a_log_line()
    print("test_the_sink_writes_through_tqdm_so_a_bar_survives_a_log_line: OK")
    test_fmt_none_keeps_loguru_s_own_format()
    print("test_fmt_none_keeps_loguru_s_own_format: OK")
    print("test_progress: OK")


if __name__ == "__main__":
    main()
