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
"""The per-round ledger of 06_hotfixes: `rounds.jsonl`, and `--report`.

Plan item 5 of docs/V2_QC_PHASE_PLAN.md. A run alternates stt and tts passes for days,
so each pass appends one line describing the whole dataset as it stands when that pass
ends, and `--report` renders the history with the delta against the first round. What is
easy to get silently wrong, and is therefore pinned here:

  * the snapshot describes the DATASET, not the rows one pass touched, and it reads the
    scores back instead of re-deriving them (deriving 602k rows costs ~1.8 h);
  * `bad` is the verdict of the gates of THAT pass, so the gates travel in the line;
  * the row filters (--category, --stems-from) narrow the snapshot too, or a restricted
    run would report the corpus around its slice;
  * `count_pending`, which the driver's convergence depends on, still answers exactly
    what it did before it became a wrapper over the one scan;
  * a round is an stt pass plus the tts pass it feeds, numbered per dataset;
  * `--report` works with none of the required run options, since it is read from
    another tab while a pass is running, and does not need a server either.

Run it with uv:

    uv run tests/test_rounds_ledger.py

This file was written with Claude Code.
"""

from __future__ import annotations

import doctest
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

import rounds_ledger as ledger  # noqa: E402
import test_recursive_improvement_modes as harness  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if not ok else ""))
    if not ok:
        failures.append(label)


def row(stem: str, *, cer=None, tail=None, window=None, bad_run=None, status=None,
        duration=12.0, category="dictionary", text="bonjour") -> dict:
    """One output row as a pass writes it: the four scores live in the reading."""
    rec = {
        "audio_filepath": f"audio/{stem}.flac",
        "duration": duration,
        "category": category,
        "asr_training_source": "bonjour",
        "asr_training_target": "bonjour",
        "transcriptions": {"default": {
            "text": text, "cer": cer, "cer_tail": tail,
            "cer_window": window, "bad_run_chars": bad_run,
        }},
    }
    if status is not None:
        rec["improvement"] = {"status": status}
    return rec


def write_out(tmp: Path, rows: list[dict], name: str = "full.stt.jsonl") -> Path:
    out = tmp / name
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return out


def gates() -> SimpleNamespace:
    """The shipped gate values, so `bad` here means what it means in production."""
    return SimpleNamespace(cer_threshold=0.08, tail_cer_threshold=0.12,
                           window_cer_threshold=0.15, bad_run_threshold=30)


def test_module_doctests() -> None:
    print("\nthe ledger module's own doctests")
    got = doctest.testmod(ledger, verbose=False)
    check("percentile / summarize / next_round doctests pass", got.failed == 0,
          f"{got.failed} of {got.attempted} failed")


def test_snapshot(mod) -> None:
    """scan_output: states, the four distributions, audio hours, and the filters."""
    print("\nthe snapshot of a written output file")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows = [
            row("a", cer=0.01, tail=0.02, window=0.03, bad_run=0),
            row("b", cer=0.02, tail=0.04, window=0.05, bad_run=1),
            row("c", cer=0.03, tail=0.06, window=0.07, bad_run=2),
            # Each of these trips exactly one gate, so all four gates are exercised.
            row("d", cer=0.40, tail=0.02, window=0.03, bad_run=0, status="pending_tts"),
            row("e", cer=0.01, tail=0.90, window=0.03, bad_run=0, status="exhausted"),
            row("f", cer=0.01, tail=0.02, window=0.99, bad_run=0, status="pending_stt"),
            row("g", cer=0.01, tail=0.02, window=0.03, bad_run=99),
            # Resolved and clean: an improved clip is not `clean`, it carries a marker.
            row("h", cer=0.01, tail=0.02, window=0.03, bad_run=0, status="improved"),
            # Never transcribed: counted as a row, not as a scored one.
            {"audio_filepath": "audio/i.flac", "duration": 8.0, "category": "dictionary"},
        ]
        out = write_out(tmp, rows)
        snap = mod.scan_output(out, model_key="default", cfg=gates())
        check("every row is counted", snap.rows == 9, str(snap.rows))
        check("only the transcribed ones are scored", snap.scored == 8, str(snap.scored))
        check("each gate catches its own clip", snap.bad == 4, str(snap.bad))
        check("a marker is not `clean` even when the clip now reads fine",
              snap.clean == 3, str(snap.clean))
        check("the markers are counted by state",
              (snap.pending_tts, snap.pending_stt, snap.improved, snap.exhausted)
              == (1, 1, 1, 1),
              str((snap.pending_tts, snap.pending_stt, snap.improved, snap.exhausted)))
        check("audio seconds are summed over every row, scored or not",
              abs(snap.audio_seconds - (8 * 12.0 + 8.0)) < 1e-9, str(snap.audio_seconds))
        cer = snap.scores["cer"]
        check("the cer distribution covers the scored rows", cer["n"] == 8, str(cer))
        # Sorted cer: 0.01 x5, 0.02, 0.03, 0.40. Nearest rank puts p90 at index
        # round(0.9 * 7) = 6, i.e. 0.03: the 0.40 outlier is the max, not the p90,
        # which is exactly the point of reporting p90 beside the gate counts.
        check("p50 and p90 are nearest-rank, not interpolated",
              (cer["p50"], cer["p90"]) == (0.01, 0.03), str(cer))
        check("bad_run_chars gets the same treatment (0 x5, 1, 2, 99 -> p90 2)",
              snap.scores["bad_run_chars"]["p90"] == 2.0,
              str(snap.scores["bad_run_chars"]))

        # A score nothing carries must answer None rather than 0, or a corpus of short
        # clips would report a perfect tail CER it never measured.
        tailless = write_out(tmp, [row("j", cer=0.01)], name="tailless.stt.jsonl")
        snap2 = mod.scan_output(tailless, model_key="default", cfg=gates())
        check("a score no row carries is None, not zero",
              snap2.scores["cer_tail"] is None, str(snap2.scores))

        # Filters: a restricted run must describe its slice.
        mixed = write_out(tmp, [
            row("k", cer=0.40, category="dictionary", status="pending_tts"),
            row("l", cer=0.40, category="drugs", status="pending_tts"),
        ], name="mixed.stt.jsonl")
        snap3 = mod.scan_output(mixed, model_key="default", cfg=gates(),
                                categories={"drugs"})
        check("--category narrows the snapshot",
              (snap3.rows, snap3.bad, snap3.pending_tts) == (1, 1, 1), str(snap3))
        snap4 = mod.scan_output(mixed, model_key="default", cfg=gates(),
                                stem_requests={"": {"k"}})
        check("--stems-from narrows it one clip at a time",
              (snap4.rows, snap4.pending_tts) == (1, 1), str(snap4))

        # The regression that matters most: the driver stops looping on this pair.
        check("count_pending still answers the same pair over the same scan",
              mod.count_pending(out) == (1, 1), str(mod.count_pending(out)))
        missing = mod.scan_output(tmp / "nope.stt.jsonl", model_key="default", cfg=gates())
        check("a file that does not exist is an empty snapshot, not a crash",
              (missing.rows, missing.scores["cer"]) == (0, None), str(missing))


def test_round_numbering(mod) -> None:
    """record_round: one line per pass, rounds paired stt + tts, per dataset."""
    print("\nthe round line")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out = write_out(tmp, [row("a", cer=0.01), row("b", cer=0.40, status="pending_tts")])
        stats = {"queued": 2, "done": 1, "improved": 0, "exhausted": 0, "swapped": 1}
        seq = []
        for mode in ("stt", "tts", "stt"):
            cfg = harness.make_cfg(mode, tmp)
            cfg.cer_threshold, cfg.tail_cer_threshold = 0.08, 0.12
            cfg.window_cer_threshold, cfg.bad_run_threshold = 0.15, 30
            snap = mod.scan_output(out, model_key="default", cfg=cfg)
            seq.append(mod.record_round(out, snap, stats, cfg, wall_s=61.5))
        check("an stt pass opens a round, the tts pass joins it, the next stt opens one",
              [(e["round"], e["pass"], e["mode"]) for e in seq]
              == [(1, 1, "stt"), (1, 2, "tts"), (2, 3, "stt")],
              str([(e["round"], e["pass"], e["mode"]) for e in seq]))

        path = ledger.ledger_path(tmp)
        lines = ledger.load_rounds(path)
        check("one line per pass, appended", len(lines) == 3, str(len(lines)))
        first = lines[0]
        check("the line carries the dataset's state", first["states"]["bad"] == 1,
              str(first["states"]))
        check("and this pass's own work, which is a different question",
              first["work"]["swapped"] == 1, str(first["work"]))
        check("the gates travel with it, since `bad` means nothing without them",
              first["gates"]["cer"] == 0.08, str(first["gates"]))
        check("wall seconds and audio hours are recorded",
              (first["wall_s"], first["audio_hours"]) == (61.5, round(24 / 3600, 3)),
              str((first["wall_s"], first["audio_hours"])))

        # A second dataset in the same directory keeps its own numbering: the driver
        # runs several manifests, and round 3 of one says nothing about the other.
        other = tmp / "parrot.stt.jsonl"
        other.write_text(json.dumps(row("z", cer=0.01)) + "\n", encoding="utf-8")
        cfg = harness.make_cfg("stt", tmp)
        cfg.cer_threshold, cfg.tail_cer_threshold = 0.08, 0.12
        cfg.window_cer_threshold, cfg.bad_run_threshold = 0.15, 30
        entry = mod.record_round(other, mod.scan_output(other, model_key="default", cfg=cfg),
                                 stats, cfg, wall_s=1.0)
        check("a second dataset starts at round 1 in the same ledger",
              (entry["round"], entry["pass"]) == (1, 1), str(entry))

        # A corrupt line must not take the report down with it.
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{not json at all\n")
        check("an unparseable line is skipped", len(ledger.load_rounds(path)) == 4,
              str(len(ledger.load_rounds(path))))


def test_report(mod) -> None:
    """--report: the table, the delta, and that it needs no run options."""
    print("\nthe report")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dataset = tmp / "improved"
        dataset.mkdir()
        base = {"dataset": "full.stt.jsonl", "mode": "stt", "wall_s": 3600.0,
                "audio_hours": 3115.2,
                "scores": {"cer": {"n": 10, "p50": 0.02, "p90": 0.06},
                           "cer_tail": None, "cer_window": None,
                           "bad_run_chars": {"n": 10, "p50": 1.0, "p90": 3.0}},
                "work": {"improved": 0, "exhausted": 0, "swapped": 0}}
        ledger.append_round(ledger.ledger_path(dataset), dict(
            base, ts="2026-10-19T10:00:00+02:00", round=1, **{"pass": 1},
            states={"rows": 1000, "scored": 1000, "bad": 100, "pending_tts": 100,
                    "pending_stt": 0, "improved": 0, "exhausted": 0, "clean": 900}))
        ledger.append_round(ledger.ledger_path(dataset), dict(
            base, ts="2026-10-20T10:00:00+02:00", round=2, **{"pass": 3},
            scores={"cer": {"n": 10, "p50": 0.015, "p90": 0.04},
                    "cer_tail": None, "cer_window": None,
                    "bad_run_chars": {"n": 10, "p50": 1.0, "p90": 2.0}},
            work={"improved": 80, "exhausted": 20, "swapped": 5},
            states={"rows": 1000, "scored": 1000, "bad": 20, "pending_tts": 0,
                    "pending_stt": 0, "improved": 80, "exhausted": 20, "clean": 900}))

        text = ledger.render_report(tmp)
        check("the report finds a ledger one level down",
              "improved/full.stt.jsonl" in text.replace("\\\\", "/"), text[:120])
        check("it prints one line per pass", text.count("\n  ") >= 2, text)
        check("the delta against the first round is spelled out",
              "bad -80" in text and "-80.0 pct" in text,
              [ln for ln in text.splitlines() if "since round" in ln])
        check("a score nothing carried prints as - rather than 0.000",
              "-/-" in text, text)
        check("and it says whether anything is still open",
              "converged: nothing pending" in text, text.splitlines()[-1])

        empty = ledger.render_report(tmp / "nowhere")
        check("an empty history explains itself instead of printing a blank table",
              "no rounds.jsonl found" in empty, empty)

        # The whole point of --report is being runnable from another tab: no --input,
        # no --output, no --stt-endpoint, no server.
        from click.testing import CliRunner
        got = CliRunner().invoke(mod.main, ["--report", str(tmp)])
        check("--report runs with none of the required run options",
              got.exit_code == 0, f"exit {got.exit_code}: {got.output[:200]}")
        check("and prints the same table", "full.stt.jsonl" in got.output,
              got.output[:200])

        # The driver is the only way a run is actually launched, so the table has to be
        # reachable from it: it prints one per dataset when that dataset's loop ends.
        driver = (Path(__file__).resolve().parent.parent
                  / "06_hotfixes" / "driver.sh").read_text(encoding="utf-8")
        check("driver.sh prints the report when a dataset finishes",
              '--report "${CUR_OUTPUT}"' in driver, "absent")
        check("and never lets a missing ledger fail a finished dataset",
              '--report "${CUR_OUTPUT}" || true' in driver, "no || true")


if __name__ == "__main__":
    module = harness.load_module()
    test_module_doctests()
    test_snapshot(module)
    test_round_numbering(module)
    test_report(module)
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)
