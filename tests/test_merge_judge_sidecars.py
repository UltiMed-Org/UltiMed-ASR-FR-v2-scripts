#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""The stage-05 sidecar -> stage-06 state join (`06_hotfixes/12_merge_judge_sidecars.py`).

This script's whole job is a join, and a join that matches nothing looks exactly like a
join that had nothing to do: both print zero and exit 0. So the key itself is pinned
here, along with the three ways a reading can be wrong in a way no error would reveal:
the metric names differ on the two sides (`window_cer` vs `cer_window`), a sidecar is
append-only so a stem's LAST row is the current one, and a row whose clip was redrawn
after it was read describes audio nobody ships.

This file was written by Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "06_hotfixes" / "12_merge_judge_sidecars.py"
    spec = importlib.util.spec_from_file_location("merge_judge_sidecars", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load()

NOW = 1_700_000_000.0
STAMP = "2026-10-25T20:00:00Z"
STAMP_POSIX = 1_792_958_400.0


def _stage(tmp: Path, rows: list[dict], *, clip_mtime: float | None = None) -> Path:
    """A stage audio directory holding one sidecar and a clip per row."""
    audio = tmp / "PARHAF"
    audio.mkdir(exist_ok=True)
    with (audio / "qc.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for row in rows:
        clip = audio / f"{row['stem']}.flac"
        clip.write_bytes(b"")
        when = clip_mtime if clip_mtime is not None else STAMP_POSIX - 600.0
        os.utime(clip, (when, when))
    return audio


def _row(stem: str, **over) -> dict:
    row = {"stem": stem, "stage": "03_PARHAF", "duration": 20.0, "cer": 0.10,
           "wer": 0.2, "tail_cer": 0.3, "window_cer": 0.4, "bad_run_chars": 5,
           "stt_s": 1.0, "transcript": " heard it ", "ts": STAMP}
    row.update(over)
    return row


def test_the_metric_names_are_translated() -> None:
    """The sidecar's `tail_cer`/`window_cer` must land as `cer_tail`/`cer_window`.

    A judge whose window CER arrived under the wrong name would be read as having no
    window reading at all, which is indistinguishable from a judge that never ran.
    """
    reading = mod.to_reading(_row("x"), n_readings=1)
    assert reading["cer_tail"] == 0.3, reading
    assert reading["cer_window"] == 0.4, reading
    assert reading["bad_run_chars"] == 5, reading
    assert reading["text"] == "heard it", "the transcript must be stripped"
    assert reading["stt_seconds"] == 1.0
    assert reading["stt_rtf"] == 0.05, "1.0 s of STT over 20 s of audio"
    assert "n_stt_check" not in reading, "a single reading does not need a count"


def test_rtf_is_omitted_rather_than_invented() -> None:
    """No duration means no ratio: a fabricated one would look measured."""
    for bad in (None, 0, 0.0):
        reading = mod.to_reading(_row("x", duration=bad), n_readings=1)
        assert "stt_rtf" not in reading, f"duration={bad!r} must not yield an rtf"
        assert reading["stt_seconds"] == 1.0, "the raw measurement still survives"


def test_the_last_row_for_a_stem_wins() -> None:
    """A sidecar is append-only, so a re-read leaves both rows in the file."""
    with tempfile.TemporaryDirectory() as d:
        audio = _stage(Path(d), [_row("a", cer=0.90), _row("a", cer=0.02)])
        index, stats = mod.load_sidecars({"03_PARHAF": audio}, "qc.jsonl",
                                         ts_field="ts", check_freshness=True)
        assert index[("PARHAF", "a")]["cer"] == 0.02, "the later reading is current"
        assert index[("PARHAF", "a")]["n_stt_check"] == 2, "both readings are counted"
        assert stats["rows"] == 2 and stats["stems"] == 1


def test_a_redrawn_clip_is_not_re_certified() -> None:
    """A reading of the OLD take must not be merged onto the new one.

    This is the bug that put 9,829 stale loop flags in the corpus, in the other
    direction: here a stale row would certify audio it never heard.
    """
    with tempfile.TemporaryDirectory() as d:
        audio = _stage(Path(d), [_row("fresh"), _row("redrawn")])
        later = STAMP_POSIX + 600.0
        os.utime(audio / "redrawn.flac", (later, later))
        index, stats = mod.load_sidecars({"03_PARHAF": audio}, "qc.jsonl",
                                         ts_field="ts", check_freshness=True)
        assert ("PARHAF", "fresh") in index
        assert ("PARHAF", "redrawn") not in index, "the clip is newer than the reading"
        assert stats["stale"] == 1
        # ...and the escape hatch has to actually let it through, for a sidecar that
        # predates the stamp.
        index, _ = mod.load_sidecars({"03_PARHAF": audio}, "qc.jsonl",
                                      ts_field="ts", check_freshness=False)
        assert ("PARHAF", "redrawn") in index


def test_a_torn_final_line_is_survived() -> None:
    """A killed sweep leaves half a line; the other rows still have to merge."""
    with tempfile.TemporaryDirectory() as d:
        audio = _stage(Path(d), [_row("a")])
        with (audio / "qc.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"stem": "b", "cer": 0.1')
        index, stats = mod.load_sidecars({"03_PARHAF": audio}, "qc.jsonl",
                                         ts_field="ts", check_freshness=True)
        assert ("PARHAF", "a") in index and stats["torn_line"] == 1


def test_the_join_key_is_the_audio_directory() -> None:
    """A state row says `../PARHAF/<stem>.flac`, whatever base it is relative to.

    Keying on the parent directory is what lets one index serve both state files, whose
    paths are relative to different bases.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        audio = _stage(tmp, [_row("a", cer=0.42)])
        index, _ = mod.load_sidecars({"03_PARHAF": audio}, "qc.jsonl",
                                     ts_field="ts", check_freshness=True)
        state = tmp / "full.stt.jsonl"
        rows = [{"audio_filepath": "../PARHAF/a.flac",
                 "transcriptions": {"whisper.cpp/ggml-large-v3-turbo": {"cer": 0.01}}},
                {"audio_filepath": "../../PARHAF/a.flac", "transcriptions": {}},
                {"audio_filepath": "../dictionary/a.flac", "transcriptions": {}}]
        state.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        # Dry run first: it must report the matches and change nothing.
        before = state.read_text()
        stats = mod.merge_state(state, index, "judge-x", False)
        assert stats["matched"] == 2, stats
        assert state.read_text() == before, "a dry run must not write"

        stats = mod.merge_state(state, index, "judge-x", True)
        assert stats["matched"] == 2 and stats["added"] == 2
        out = [json.loads(line) for line in state.read_text().splitlines()]
        assert out[0]["transcriptions"]["judge-x"]["cer"] == 0.42
        # The existing judge is untouched: this script adds a reading, never replaces one.
        assert out[0]["transcriptions"]["whisper.cpp/ggml-large-v3-turbo"]["cer"] == 0.01
        assert out[1]["transcriptions"]["judge-x"]["cer"] == 0.42, "base-insensitive"
        assert "judge-x" not in out[2]["transcriptions"], "another stage must not match"

        # Idempotent: a second apply replaces its own reading rather than duplicating it.
        stats = mod.merge_state(state, index, "judge-x", True)
        assert stats["replaced"] == 2 and stats["added"] == 0, stats


def test_a_failed_write_leaves_the_state_intact() -> None:
    """The rewrite is to a temp file and renamed, so a crash cannot truncate the state."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        state = tmp / "full.stt.jsonl"
        state.write_text(json.dumps({"audio_filepath": "../PARHAF/a.flac"}) + "\n"
                         + "{ this line is not json\n")
        before = state.read_text()
        try:
            mod.merge_state(state, {("PARHAF", "a"): {"cer": 1.0}}, "judge-x", True)
        except ValueError:
            pass
        else:
            raise AssertionError("a malformed state line should have raised")
        assert state.read_text() == before, "the state must survive a failed merge"
        assert not list(tmp.glob("*.merge.tmp")), "the temp file must be cleaned up"


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
