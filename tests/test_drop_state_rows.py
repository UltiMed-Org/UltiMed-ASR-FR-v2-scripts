#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""`06_hotfixes/20_drop_state_rows.py`: which state rows it drops, and that it loses none.

The script exists because stage 06's merge lets an existing state row win over the
rebuilt manifest, so a clip regenerated under it after a voice retirement would keep the
old take's verdicts forever. Its two failure modes are both silent: dropping too much
(clips re-judged for nothing, or worse, a row gone that was not saved) and dropping too
little (stale verdicts shipped). Pinned here:

  * a stems list is read both as plain lines and as the audit's JSONL report;
  * `--voice` drops by stored reader, alone or on top of a stems list;
  * a dry run leaves the state byte-identical and writes no dropped file;
  * `--apply` removes exactly the named rows, keeps them verbatim in the dropped file,
    and a requested stem absent from the state is reported.

    uv run tests/test_drop_state_rows.py

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from click.testing import CliRunner
from loguru import logger

HERE = Path(__file__).resolve().parent
DROP_PATH = HERE.parent / "06_hotfixes" / "20_drop_state_rows.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ROWS = [
    {"audio_filepath": "../PARHAF/000001_0000_a.flac", "voice": "h_frf_itm",
     "text": "le compte rendu", "improvement": {"status": "improved", "orig_cer": 0.2}},
    {"audio_filepath": "../PARHAF/000002_0000_b.flac", "voice": "fr_male",
     "text": "la biopsie", "improvement": {"status": "exhausted"}},
    {"audio_filepath": "../PARHAF/000003_0000_c.flac", "voice": "h_frf_itm",
     "text": "l'examen"},
    {"audio_filepath": "../dictionary/000004_0000_d.flac", "voice": "fr_male",
     "text": "myocardite"},
    {"audio_filepath": "../dictionary/000005_0000_e.flac", "voice": "h_frf_itm",
     "text": "thyroïde", "improvement": {"status": "pending_tts"}},
    {"audio_filepath": "../dictionary/000006_0000_f.flac", "voice": "fr_male",
     "text": "érythème", "improvement": {"status": "improved"}},
]
#: The serialisation the state uses, kept odd on purpose (extra space after the colon
#: is not json.dumps' default) so a dropped row that is re-dumped rather than copied
#: verbatim shows up.
LINES = [json.dumps(r, ensure_ascii=False, separators=(",", ":  ")) + "\n" for r in ROWS]


def make_state(tmp: Path) -> Path:
    for old in tmp.iterdir():
        old.unlink()
    state = tmp / "full.stt.jsonl"
    state.write_text("".join(LINES), encoding="utf-8")
    return state


def stems_of(state: Path) -> list[str]:
    return [Path(json.loads(ln)["audio_filepath"]).stem
            for ln in state.read_text(encoding="utf-8").splitlines() if ln.strip()]


def dropped_files(tmp: Path) -> list[Path]:
    return sorted(tmp.glob("full.stt.jsonl.dropped-*.jsonl"))


def run_main(mod, args: list[str]) -> list[str]:
    """Run the CLI, returning its log lines."""
    logs: list[str] = []
    sink = logger.add(lambda m: logs.append(str(m)), level="INFO")
    try:
        result = CliRunner().invoke(mod.main, args, catch_exceptions=False)
    finally:
        logger.remove(sink)
    assert result.exit_code == 0, result.output
    return logs


def test_plain_stems_list(mod, tmp: Path) -> None:
    stems_file = tmp.parent / "stems.txt"
    stems_file.write_text("000001_0000_a\n\n000004_0000_d\n", encoding="utf-8")
    assert mod.read_stems(stems_file) == {"000001_0000_a", "000004_0000_d"}


def test_jsonl_report_with_path(mod, tmp: Path) -> None:
    report = tmp.parent / "retired.jsonl"
    report.write_text(
        "\n".join(json.dumps({"stage": "03_PARHAF", "voice": "h_frf_itm",
                              "verdict": "retired_voice", "path": p})
                  for p in ("/x/03_PARHAF/generated_audios/000001_0000_a.flac",
                            "/x/03_PARHAF/generated_audios/000003_0000_c.flac"))
        + "\n" + json.dumps({"audio_filepath": "../dictionary/000005_0000_e.flac"}) + "\n",
        encoding="utf-8")
    assert mod.read_stems(report) == {"000001_0000_a", "000003_0000_c", "000005_0000_e"}


def test_voice_selects_by_stored_reader(mod, tmp: Path) -> None:
    state = make_state(tmp)
    tally = {k: Counter() for k in ("counts", "voice", "category", "status")}
    found: set[str] = set()
    n = mod.process(state, set(), {"h_frf_itm"}, True, tally, found)
    assert n == 3 and stems_of(state) == ["000002_0000_b", "000004_0000_d",
                                          "000006_0000_f"], stems_of(state)
    assert tally["category"] == Counter({"PARHAF": 2, "dictionary": 1}), tally
    assert tally["status"] == Counter({"improved": 1, "no marker": 1,
                                       "pending_tts": 1}), tally
    # stems and voices add up: a stem read by a survivor is dropped too
    state = make_state(tmp)
    n = mod.process(state, {"000006_0000_f"}, {"h_frf_itm"}, True,
                    {k: Counter() for k in tally}, set())
    assert n == 4 and stems_of(state) == ["000002_0000_b", "000004_0000_d"], stems_of(state)


def test_dry_run_writes_nothing(mod, tmp: Path) -> None:
    state = make_state(tmp)
    before = state.read_bytes()
    logs = run_main(mod, ["--state", str(state), "--voice", "h_frf_itm"])
    assert state.read_bytes() == before, "a dry run must leave the state byte-identical"
    assert not dropped_files(tmp), f"a dry run wrote {dropped_files(tmp)}"
    assert sorted(p.name for p in tmp.iterdir()) == ["full.stt.jsonl"], \
        f"a dry run left files behind: {list(tmp.iterdir())}"
    assert any("3 to drop" in ln for ln in logs), logs


def test_apply_drops_exactly_and_keeps_them(mod, tmp: Path) -> None:
    state = make_state(tmp)
    stems_file = tmp.parent / "apply_stems.txt"
    stems_file.write_text("000002_0000_b\n000005_0000_e\n999999_0000_absent\n",
                          encoding="utf-8")
    logs = run_main(mod, ["--state", str(state), "--stems-from", str(stems_file),
                          "--apply"])
    assert stems_of(state) == ["000001_0000_a", "000003_0000_c", "000004_0000_d",
                               "000006_0000_f"], stems_of(state)
    kept = state.read_text(encoding="utf-8").splitlines(keepends=True)
    assert [json.loads(ln) for ln in kept] == [ROWS[0], ROWS[2], ROWS[3], ROWS[5]], \
        "a kept row must come back unchanged"
    files = dropped_files(tmp)
    assert len(files) == 1, files
    assert files[0].read_text(encoding="utf-8") == LINES[1] + LINES[4], \
        "the dropped file must hold the rows verbatim, in state order"
    assert not list(tmp.glob("*.drop.tmp")), "the temp file must be renamed over the state"
    assert any("1 requested stem(s) found in no state file" in ln
               and "999999_0000_absent" in ln for ln in logs), logs


if __name__ == "__main__":
    drop = load_module(DROP_PATH, "drop_under_test")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "state"
        tmp.mkdir()
        for fn in (test_plain_stems_list, test_jsonl_report_with_path,
                   test_voice_selects_by_stored_reader, test_dry_run_writes_nothing,
                   test_apply_drops_exactly_and_keeps_them):
            fn(drop, tmp)
            print(f"  ok  {fn.__name__}")
    print("OK: exactly the named rows are dropped, and kept verbatim beside the state")
