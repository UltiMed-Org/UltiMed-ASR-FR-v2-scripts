"""The per-round history of a QC run: one append-only JSONL line per pass, and a report.

Why it exists: `06_hotfixes/driver.sh` alternates stt and tts passes for hours or days,
and each pass logs plenty about what it is doing right now but nothing that survives it.
"How much did this improve the corpus" was answered by diffing two Markdown reports by
eye, and "how far along are we" by scrolling a log. So every pass now appends ONE line
describing the whole dataset as it stands when that pass ends (not just the rows it
touched), and `01_recursive_improvement.py --report` renders the history as a table with
the delta against the first round on record.

The line is a snapshot, deliberately: counts by state, the median and p90 of each of the
four scores, audio hours and wall seconds. That makes a round interpretable on its own
years later, and it makes the delta a subtraction rather than a reconstruction. It also
means the ledger is useless as a source of truth about any individual clip, which is the
point: `full.stt.jsonl` owns that, and this file must never become a second copy of it.

Append-only, one writer at a time (a pass is a single process, and the append happens
after its thread pool is done), and a line that cannot be parsed is skipped rather than
fatal: a report is a convenience and must never be what stops a run from finishing.

Stdlib only, no logging dependency, so any stage can append without pulling in loguru.

This file was written with Claude Code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

#: One ledger per output directory, beside the `*.stt.jsonl` it describes.
LEDGER_FILENAME = "rounds.jsonl"

#: The four scores, in the order every table prints them. Keys are the ones the ledger
#: line uses, which are also the keys a row's stored reading carries, so there is one
#: spelling of each score across the stage.
SCORE_KEYS = ("cer", "cer_tail", "cer_window", "bad_run_chars")

#: Counts by state, in print order. `clean` is "scored and passes every gate with no
#: marker", `bad` is "fails a gate under the gates of THIS pass, whatever its marker":
#: the two overlap with the markers on purpose, because what the operator wants to see
#: shrink is `bad`, and what tells them the run is still working is the pendings.
STATE_KEYS = ("rows", "scored", "bad", "pending_tts", "pending_stt",
              "improved", "exhausted", "clean")


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile of an ALREADY SORTED sequence, 0 <= q <= 1.

    Nearest rank rather than interpolation because two of the four scores
    (`bad_run_chars`, and in practice `cer` on short clips) are small integers or
    coarse ratios, where an interpolated p90 invents a value no clip has.

    >>> percentile([0.0, 1.0, 2.0, 3.0], 0.5)
    2.0
    >>> percentile([0.0, 1.0, 2.0, 3.0], 0.9)
    3.0
    >>> percentile([], 0.5) is None
    True
    """
    if not values:
        return None
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def summarize(values: Iterable[float]) -> dict | None:
    """``{"n": count, "p50": ..., "p90": ...}`` for one score, or None when nothing
    carried it (a corpus of clips too short for a tail score, say).

    >>> summarize([0.1, 0.2, 0.3, 0.4, 0.5]) == {"n": 5, "p50": 0.3, "p90": 0.5}
    True
    >>> summarize([]) is None
    True
    """
    ordered = sorted(values)
    if not ordered:
        return None
    return {"n": len(ordered), "p50": percentile(ordered, 0.5),
            "p90": percentile(ordered, 0.9)}


def ledger_path(out_dir: Path) -> Path:
    """The ledger of an output directory (`improved/`, `improved_parrot/`, ...)."""
    return Path(out_dir) / LEDGER_FILENAME


def load_rounds(path: Path) -> list[dict]:
    """Every parseable line of one ledger, oldest first. A missing file is an empty
    history, not an error: the first pass of a run legitimately has none."""
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            rows.append(rec)
    return rows


def next_round(prior: Iterable[dict], dataset: str, mode: str) -> tuple[int, int]:
    """``(round, pass)`` for the pass about to be recorded.

    A round is an stt pass plus the tts pass it feeds, because that pair is what the
    driver alternates and what "one more round of draws" means to the operator. So an
    `stt` or `both` pass OPENS a round and a `tts` pass joins the open one; a `tts` pass
    with no history opens round 1 anyway, since a run started by hand on the tts half is
    still a first round.

    >>> next_round([], "full.stt.jsonl", "stt")
    (1, 1)
    >>> hist = [{"dataset": "full.stt.jsonl", "round": 1, "pass": 1, "mode": "stt"}]
    >>> next_round(hist, "full.stt.jsonl", "tts")
    (1, 2)
    >>> next_round(hist, "full.stt.jsonl", "stt")
    (2, 2)
    >>> next_round(hist, "other.stt.jsonl", "tts")
    (1, 1)
    """
    mine = [r for r in prior if r.get("dataset") == dataset]
    n_pass = len(mine) + 1
    current = max((int(r.get("round") or 0) for r in mine), default=0)
    if mode in ("stt", "both") or current == 0:
        return current + 1, n_pass
    return current, n_pass


def append_round(path: Path, entry: dict) -> None:
    """Append one line. Best effort: a ledger that cannot be written must not take down
    a pass that has just spent hours of GPU, so the OSError is swallowed by the caller's
    logging path rather than raised here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_ledgers(root: Path) -> list[Path]:
    """Every ledger at or one level under ``root``, sorted.

    One level is enough and is on purpose: the convention is one output directory per
    dataset beside the stage (`06_hotfixes/improved/`, `06_hotfixes/improved_parrot/`),
    so running `--report` from the stage directory finds both with no hardcoded list of
    dataset names to keep in step with the driver."""
    root = Path(root)
    found = []
    if (root / LEDGER_FILENAME).is_file():
        found.append(root / LEDGER_FILENAME)
    found.extend(p for p in root.glob(f"*/{LEDGER_FILENAME}") if p.is_file())
    return sorted(set(found))


def _fmt(value, width: int, spec: str = "") -> str:
    if value is None:
        return "-".rjust(width)
    return format(value, spec).rjust(width)


def _delta(now, before, spec: str = "+.3f") -> str:
    if now is None or before is None:
        return ""
    diff = now - before
    if not diff:
        return "0"   # spelled as a number, so a row of deltas reads as one list
    return format(diff, spec)


def render_dataset(path: Path, rows: Sequence[dict]) -> str:
    """The table for one ledger: one line per pass, plus the delta against round 1."""
    out = [f"{path}: {len(rows)} pass(es), "
           f"{max((int(r.get('round') or 0) for r in rows), default=0)} round(s)"]
    if not rows:
        out.append("  (empty)")
        return "\n".join(out)
    header = (f"  {'rnd':>3} {'pass':>4} {'mode':<5} {'rows':>8} {'bad':>7} "
              f"{'pend_tts':>8} {'pend_stt':>8} {'impr':>6} {'rq':>5} {'exh':>6} "
              f"{'swap':>5} "
              + " ".join(f"{k.replace('cer_', ''):>13}" for k in SCORE_KEYS)
              + f" {'audio_h':>8} {'wall_h':>7}")
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for rec in rows:
        states = rec.get("states") or {}
        work = rec.get("work") or {}
        scores = rec.get("scores") or {}
        cells = [f"  {_fmt(rec.get('round'), 3, 'd')} {_fmt(rec.get('pass'), 4, 'd')} "
                 f"{str(rec.get('mode') or '?'):<5} "
                 f"{_fmt(states.get('rows'), 8, ',d')} {_fmt(states.get('bad'), 7, ',d')} "
                 f"{_fmt(states.get('pending_tts'), 8, ',d')} "
                 f"{_fmt(states.get('pending_stt'), 8, ',d')} "
                 f"{_fmt(work.get('improved'), 6, ',d')} "
                 # rq = of those wins, the ones only good enough to requeue (better
                 # audio, still over a gate), so `impr` is never read as "now clean".
                 f"{_fmt(work.get('improved_requeued'), 5, ',d')} "
                 f"{_fmt(work.get('exhausted'), 6, ',d')} "
                 f"{_fmt(work.get('swapped'), 5, ',d')}"]
        for key in SCORE_KEYS:
            s = scores.get(key) or {}
            spec = ".0f" if key == "bad_run_chars" else ".3f"
            p50, p90 = s.get("p50"), s.get("p90")
            cell = ("-" if p50 is None else format(p50, spec)) + "/" + \
                   ("-" if p90 is None else format(p90, spec))
            cells.append(f" {cell:>13}")
        cells.append(f" {_fmt(rec.get('audio_hours'), 8, ',.1f')}")
        wall = rec.get("wall_s")
        cells.append(f" {_fmt(None if wall is None else wall / 3600.0, 7, ',.2f')}")
        out.append("".join(cells))

    first, last = rows[0], rows[-1]
    fs, ls = first.get("states") or {}, last.get("states") or {}
    bits = [f"bad {_delta(ls.get('bad'), fs.get('bad'), '+,d')}"]
    if fs.get("bad"):
        bits[-1] += f" ({100.0 * ((ls.get('bad') or 0) - fs['bad']) / fs['bad']:+.1f} pct)"
    for key in SCORE_KEYS:
        a = (first.get("scores") or {}).get(key) or {}
        b = (last.get("scores") or {}).get(key) or {}
        spec = "+.0f" if key == "bad_run_chars" else "+.4f"
        bits.append(f"{key.replace('cer_', '')} p50 {_delta(b.get('p50'), a.get('p50'), spec)}"
                    f" p90 {_delta(b.get('p90'), a.get('p90'), spec)}")
    out.append(f"  since round {first.get('round')} ({first.get('ts')}): " + ", ".join(bits))
    total_wall = sum(float(r.get("wall_s") or 0.0) for r in rows)
    out.append(f"  total wall: {total_wall / 3600.0:,.2f} h over {len(rows)} pass(es)")
    remaining = (ls.get("pending_tts") or 0) + (ls.get("pending_stt") or 0)
    out.append("  converged: nothing pending" if not remaining
               else f"  still open: {ls.get('pending_tts') or 0:,} clip(s) need a tts "
                    f"pass, {ls.get('pending_stt') or 0:,} an stt pass")
    return "\n".join(out)


def render_report(root: Path) -> str:
    """Every ledger at or under ``root``, one table each."""
    ledgers = find_ledgers(root)
    if not ledgers:
        return (f"no {LEDGER_FILENAME} found at or under {root}. A pass writes one when "
                f"it finishes, so either none has run yet or this is the wrong directory "
                f"(run it from 06_hotfixes/, or pass --report DIR).")
    blocks = []
    for path in ledgers:
        rows = load_rounds(path)
        by_dataset: dict[str, list[dict]] = {}
        for rec in rows:
            by_dataset.setdefault(str(rec.get("dataset") or "?"), []).append(rec)
        for dataset, recs in sorted(by_dataset.items()):
            blocks.append(render_dataset(path.parent / dataset, recs))
    return "\n\n".join(blocks)


if __name__ == "__main__":
    import doctest
    import sys

    failures, _ = doctest.testmod()
    if len(sys.argv) > 1:
        print(render_report(Path(sys.argv[1])))
    raise SystemExit(1 if failures else 0)
