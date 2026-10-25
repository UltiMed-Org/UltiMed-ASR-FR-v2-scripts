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
"""`06_hotfixes/10_compare_judges.py`: reading out the T2 second opinion.

The phase scores every clip with `ggml-large-v3-turbo` and then re-reads the flagged
slice with non-turbo `ggml-large-v3`. Both readings sit on the same row under their own
key, and the question that justifies the second pass is the asymmetric one: a clip the
better judge calls bad and the faster one waved through is a false negative that ships,
whereas the reverse is only a flag the better judge forgives. So the two cells must never
be summed into one "disagreements" number, which is what this file pins, along with the
thing the next command consumes: the stem list, which must name what the SECOND judge
rejects (whether or not the first agreed) and nothing else, and must parse as a stem list.

Run it with uv (the tool imports the pass, which imports soundfile):
    uv run tests/test_compare_judges.py

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from click.testing import CliRunner

HERE = Path(__file__).resolve().parent
TOOL_PATH = HERE.parent / "06_hotfixes" / "10_compare_judges.py"
UTILS = HERE.parent / "utils"

KEY_A = "whisper.cpp/ggml-large-v3-turbo"
KEY_B = "whisper.cpp/ggml-large-v3"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reading(text: str = "bonjour le patient", **scores) -> dict:
    """One entry of a row's `transcriptions` map, as 01_compute_stt.py writes it: the
    score keys are optional, because a short clip has no tail and an empty reference has
    no CER at all."""
    entry = {"text": text, "wer": None}
    entry.update(scores)
    return entry


def row(name: str, *, a: dict | None, b: dict | None, duration: float = 10.0,
        status: str | None = None) -> dict:
    rec = {
        "audio_filepath": f"../PARHAF/{name}.flac",
        "duration": duration,
        "text": "bonjour le patient prend du paracetamol",
        "asr_training_source": "bonjour le patient prend du paracetamol",
        "transcriptions": {},
    }
    if a is not None:
        rec["transcriptions"][KEY_A] = a
    if b is not None:
        rec["transcriptions"][KEY_B] = b
    if status is not None:
        rec["improvement"] = {"status": status}
    return rec


def write_state(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def build(tmp: Path) -> tuple[Path, Path]:
    """Two state files, one clip per cell of the agreement table."""
    main_state = tmp / "improved" / "full.stt.jsonl"
    parrot_state = tmp / "improved_parrot" / "full.stt.jsonl"
    write_state(main_state, [
        # both judges are happy: not a disagreement, and must not reach either artefact.
        row("agree_clean", a=reading(cer=0.01, cer_window=0.01, bad_run_chars=0),
            b=reading(cer=0.02, cer_window=0.02, bad_run_chars=0), status="improved"),
        # the cell the whole pass exists for: turbo waved it through.
        row("turbo_missed", a=reading(cer=0.01, cer_window=0.01, bad_run_chars=0),
            b=reading(cer=0.20, cer_window=0.22, bad_run_chars=4), status="improved"),
        # the mirror: the better judge forgives what turbo flagged. Still a disagreement
        # worth recording, but NOT something to redraw.
        row("turbo_overflagged", a=reading(cer=0.10, cer_window=0.11, bad_run_chars=0),
            b=reading(cer=0.01, cer_window=0.01, bad_run_chars=0), status="improved"),
        # five rounds could not fix it and both judges still reject it.
        row("still_bad", a=reading(cer=0.15, cer_window=0.16, bad_run_chars=2),
            b=reading(cer=0.25, cer_window=0.26, bad_run_chars=3), status="exhausted"),
        # only one gate fires, and it is not the whole-clip one: a phrase dropped
        # mid-clip is exactly what `cer` dilutes away on a long clip.
        row("window_only", a=reading(cer=0.01, cer_window=0.02, bad_run_chars=1),
            b=reading(cer=0.03, cer_window=0.19, bad_run_chars=31), duration=48.0,
            status="improved"),
        # a short clip has no tail score at all: None must not read as "fails the gate".
        row("short_no_tail", a=reading(cer=0.01), b=reading(cer=0.01), duration=2.0,
            status="improved"),
        # T2 never got to it (it is not in the flag list, or its reading was lost).
        row("unread_by_b", a=reading(cer=0.01), b=None),
        # T2 tried and failed: an errored reading judges nothing either way.
        row("b_errored", a=reading(cer=0.01),
            b={"text": "", "error": "connection reset"}),
    ])
    write_state(parrot_state, [
        row("parrot_bad", a=reading(cer=0.02, cer_window=0.02, bad_run_chars=0),
            b=reading(cer=0.12, cer_window=0.13, bad_run_chars=0), status="improved"),
    ])
    return main_state, parrot_state


def run_tool(mod, states: list[Path], out_dir: Path, *extra: str):
    result = CliRunner().invoke(mod.main, [
        *sum((["--state", str(s)] for s in states), []),
        "--out-jsonl", str(out_dir / "judge_disagreements.jsonl"),
        "--out-stems", str(out_dir / "second_opinion_flags.txt"),
        *extra,
    ], catch_exceptions=False)
    return result


def test_the_agreement_table(mod) -> None:
    print("\nthe agreement table keeps the two directions apart")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        main_state, parrot_state = build(tmp)
        result = run_tool(mod, [main_state, parrot_state], tmp, "--samples", "0")
        check("the tool exits 0", result.exit_code == 0, result.output)

        measured = mod.compare([main_state, parrot_state], KEY_A, KEY_B,
                               mod.gates_from(0.08, 0.12, 0.15, 30))
        counts = measured["counts"]
        check("every row is read", counts["rows"] == 9, str(counts))
        check("only the rows with two usable readings are compared",
              counts["both_judges"] == 7, str(counts))
        check("a missing or errored second reading is not a disagreement",
              counts["only_a_read"] == 2, str(counts))
        check("both clean counts the agreements", counts["both_clean"] == 2, str(counts))
        check("the false negatives are their own cell", counts["only_b_bad"] == 3,
              str(counts))
        check("so are the flags the better judge forgives", counts["only_a_bad"] == 1,
              str(counts))
        check("and the clips both judges reject", counts["both_bad"] == 1, str(counts))
        flagged = {d["stem"] for d in measured["detail"]}
        check("a clip with no tail score is not called bad by the tail gate",
              "short_no_tail" not in flagged, str(flagged))
        check("nor does an agreement reach the detail file", "agree_clean" not in flagged,
              str(flagged))


def test_the_stem_list_is_what_the_repair_round_takes(mod) -> None:
    print("\nthe stem list names what the SECOND judge rejects, worst first")
    sys.path.insert(0, str(UTILS))
    from stem_requests import load_stem_requests, row_is_requested  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        main_state, parrot_state = build(tmp)
        result = run_tool(mod, [main_state, parrot_state], tmp)
        check("the tool exits 0", result.exit_code == 0, result.output)

        stems_file = tmp / "second_opinion_flags.txt"
        lines = stems_file.read_text(encoding="utf-8").splitlines()
        stems = [ln for ln in lines if ln and not ln.startswith("#")]
        check("it is commented, so the file explains itself in six months",
              all(ln.startswith("#") for ln in lines[:3]), "\n".join(lines[:3]))
        check("it names every clip the second judge rejects",
              set(stems) == {"still_bad", "turbo_missed", "window_only", "parrot_bad"},
              str(stems))
        check("and not the flag the second judge forgave",
              "turbo_overflagged" not in stems, str(stems))
        check("worst first, so an interrupted repair round fixes the worst clips",
              stems[:2] == ["still_bad", "turbo_missed"], str(stems))

        # The contract that makes it a one-liner: STEMS_FROM reads this file as it stands.
        requests = load_stem_requests([stems_file])
        check("utils/stem_requests parses it as a bare stem list",
              requests.get("") == {"still_bad", "turbo_missed", "window_only",
                                   "parrot_bad"}, str(requests))
        check("and matches it against a manifest row's audio_filepath",
              row_is_requested(requests, "../PARHAF/turbo_missed.flac")
              and not row_is_requested(requests, "../PARHAF/agree_clean.flac"))

        detail = [json.loads(ln) for ln
                  in (tmp / "judge_disagreements.jsonl").read_text().splitlines()]
        check("the jsonl keeps BOTH directions, including the forgiven flag",
              {d["stem"] for d in detail} == {"still_bad", "turbo_missed", "window_only",
                                              "parrot_bad", "turbo_overflagged"},
              str([d["stem"] for d in detail]))
        check("each entry carries both transcripts and the source, to be read not trusted",
              all(d["text_a"] and d["text_b"] and d["source"] for d in detail))
        window_only = next(d for d in detail if d["stem"] == "window_only")
        check("and names which gates fired, since bad_run and cer mean different repairs",
              sorted(window_only["gates_b"]) == ["bad_run", "window"],
              str(window_only["gates_b"]))
        check("the clip's provenance travels too, so a redraw can change voice",
              window_only["improvement"] == {"status": "improved"},
              str(window_only["improvement"]))


def test_dry_run_writes_nothing(mod) -> None:
    print("\n--dry-run reports without touching the artefacts")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        main_state, _parrot = build(tmp)
        result = run_tool(mod, [main_state], tmp, "--dry-run")
        check("it exits 0", result.exit_code == 0, result.output)
        check("no stem list", not (tmp / "second_opinion_flags.txt").exists())
        check("no detail file", not (tmp / "judge_disagreements.jsonl").exists())


def test_the_gates_are_the_passs_own(mod) -> None:
    print("\nthe gates come from the pass, and can still be tightened by hand")
    stt = mod._load_pass()._stt
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        main_state, _parrot = build(tmp)
        # Two clips straddling the pass's own whole-clip gate, run through the CLI with
        # no threshold flags at all: if the defaults were not wired to the pass, one of
        # these two would land on the wrong side.
        edge = tmp / "edge" / "full.stt.jsonl"
        write_state(edge, [
            row("exactly_at_the_gate", a=reading(cer=0.0),
                b=reading(cer=stt.DEFAULT_CER_GATE)),
            row("just_under_the_gate", a=reading(cer=0.0),
                b=reading(cer=stt.DEFAULT_CER_GATE - 0.001)),
        ])
        result = run_tool(mod, [edge], tmp)
        stems = [ln for ln in (tmp / "second_opinion_flags.txt")
                 .read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
        check("the gate defaults are the pass's own, with no flags passed",
              stems == ["exactly_at_the_gate"], str(stems))
        # A stricter run must find more, which also proves the thresholds are plumbed.
        strict = mod.compare([main_state], KEY_A, KEY_B,
                             mod.gates_from(0.015, 0.12, 0.15, 30))["counts"]
        loose = mod.compare([main_state], KEY_A, KEY_B,
                            mod.gates_from(0.08, 0.12, 0.15, 30))["counts"]
        check("a stricter cer gate moves clips out of 'both clean'",
              strict["both_clean"] < loose["both_clean"],
              f"{strict['both_clean']} vs {loose['both_clean']}")


def main() -> int:
    mod = load_module(TOOL_PATH, "compare_judges_under_test")
    test_the_agreement_table(mod)
    test_the_stem_list_is_what_the_repair_round_takes(mod)
    test_dry_run_writes_nothing(mod)
    test_the_gates_are_the_passs_own(mod)
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
