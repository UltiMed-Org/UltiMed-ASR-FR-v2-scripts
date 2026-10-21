"""Progress-bar plumbing shared by the pipeline's long-running scripts.

Every script here that shows a `tqdm` bar needs the same thing first: loguru must print
through `tqdm.write`, or each log line is written straight to the stream the bar is
redrawing on and tears it in half. That was three near-identical copies
(`05_generate_audio/01_generate_audio.py`, `06_hotfixes/01_compute_stt.py`,
`06_hotfixes/01_recursive_improvement.py`) before this module, and the acoustic scan
would have been a fourth.

`tqdm` is imported inside the function, the convention `nemo_manifest` already follows,
so importing this module costs nothing and the unit tests do not need the dependency.
The caller still declares `tqdm` and `loguru` in its own PEP 723 header, because that is
what `uv run` reads.
"""
from __future__ import annotations

#: What the three existing call sites print: the level, then the message. The timestamp
#: is left out on purpose, since a bar already shows elapsed and remaining time.
DEFAULT_FORMAT = "<level>{level: <8}</level> | {message}"


def bar_safe_sink(logger, *, level: str = "INFO", fmt: str | None = DEFAULT_FORMAT,
                  stream=None, colorize: bool = True) -> int:
    """Route `logger` through `tqdm.write` so log lines cannot tear a progress bar.

    Adds a sink; it does not remove existing ones, so a caller that also wants a log
    FILE adds that separately, and one that wants the bar to own the terminal calls
    `logger.remove()` first (every current caller does).

    Parameters
    ----------
    logger
        The loguru logger to add a sink to.
    level
        Minimum level for this sink.
    fmt
        loguru format string. Defaults to `DEFAULT_FORMAT`; None keeps loguru's own
        default (timestamp, level, module, message), which one caller relies on.
    stream
        Passed to `tqdm.write` as its `file`. None means tqdm's default, stdout. Pass
        `sys.stderr` to keep the log off a stdout that something else is reading.
    colorize
        Whether loguru applies colour tags. Keep it on for a terminal.

    Returns
    -------
    int
        The sink id, so the caller can `logger.remove(id)` later.
    """
    from tqdm import tqdm

    kwargs = {} if fmt is None else {"format": fmt}
    return logger.add(lambda m: tqdm.write(m, end="", file=stream),
                      level=level, colorize=colorize, **kwargs)


def resumed_bar(*, total: int, remaining: int, desc: str, unit: str = "clip", **kwargs):
    """A `tqdm` bar whose percentage counts work a previous run already did.

    A resumable scan iterates only over what is left, so a bar built on that count
    alone claims 0% when the job is in fact nearly finished, which is exactly the
    question a bar is there to answer. `total` is the whole job, `remaining` is what
    this process will actually do, and the difference is where the bar starts.

    Parameters
    ----------
    total
        Units of work in the whole job, done and not done.
    remaining
        Units this run will process. `total - remaining` is taken as already done.
    desc
        Bar label.
    unit
        Unit name for the rate display.
    **kwargs
        Passed through to `tqdm`.

    Returns
    -------
    tqdm.tqdm
        The bar, for use as a context manager.
    """
    from tqdm import tqdm

    return tqdm(total=total, initial=max(total - remaining, 0), desc=desc, unit=unit,
                **kwargs)
