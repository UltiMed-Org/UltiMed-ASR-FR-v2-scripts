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
"""The per-clip cfg_alpha sidecar, end to end: stage 05 writes it, stage 06 publishes it.

The dataset ships ONE `cfg_alpha` per row (99_hf_release/03_sync_hotfix_results.py),
so the moment stage 05 is allowed to re-draw a rejected clip one rung lower
(2.0 -> 1.8 -> 1.6 ...), that column becomes a lie for those clips unless something
records the rung per clip and something else reads it back. `utils/cfg_sidecar.py` is
the first half; `06_hotfixes/01_recursive_improvement.py::original_cfg_alpha_for` is
the second. This test pins the contract between them, because the failure mode is
silent: the audio is fine, only its published provenance is wrong.

What is covered here:

  * the ladder arithmetic and its floor (shared by both stages, so a drift in one
    would publish numbers the other never drew),
  * the sidecar round trip: append, last-row-wins, a half-written final line, and a
    missing file meaning "everything was drawn at the run's value",
  * stage 06 answering with the sidecar's value for a clip that has a row and with
    --original-cfg-alpha for a clip that does not.

Stage 05's own half (the descent and the row it appends) is pinned in
tests/test_tts_wire_format.py, where the request bodies are already faked.

Run it with uv, since the stage-06 module imports soundfile/click at import time:

    uv run tests/test_cfg_sidecar.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SIDECAR_PATH = HERE.parent / "utils" / "cfg_sidecar.py"
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ladder_rungs_and_floor() -> None:
    """The rungs both stages draw at, and the clamp that keeps CFG from reaching off."""
    cs = load_module(SIDECAR_PATH, "cfg_sidecar_ladder")

    # The measured band is 1.2-2.0 with 2.0 at the TOP, so the ladder descends from
    # the generation value and stops at the bottom of the band rather than walking out
    # of it: at CFG ~0 the flow-matching decoder stops following the text entirely,
    # which is the defect these retries exist to repair.
    assert [cs.ladder_cfg_alpha(start=2.0, attempt=i) for i in range(7)] == [
        2.0, 1.8, 1.6, 1.4, 1.2, 1.2, 1.2]

    # Rounded, because the value is logged, written to the sidecar and published:
    # 2.0 + 3 * -0.2 must be 1.4, not 1.4000000000000004.
    assert cs.ladder_cfg_alpha(start=2.0, attempt=3) == 1.4

    # No --cfg-alpha means the knob is never sent, so there is no rung to start from
    # and nothing to record. Guessing 2.0 here would be a CLIMB against a server
    # booted at 1.3 (v1), which is why it stays None.
    assert cs.ladder_cfg_alpha(start=None, attempt=2) is None

    # Both ends are overridable: step 0 pins every attempt, no floor lets it run.
    assert cs.ladder_cfg_alpha(start=2.0, attempt=5, step=0.0) == 2.0
    assert cs.ladder_cfg_alpha(start=2.0, attempt=5, floor=None) == 1.0
    print("OK the ladder descends from the top of the band and is floored at 1.2")


def test_sidecar_round_trip() -> None:
    """Append, read back, last row wins, and survive a half-written line."""
    cs = load_module(SIDECAR_PATH, "cfg_sidecar_rt")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        # No file at all is the normal case for a corpus generated entirely at the
        # run's cfg_alpha, and it must read as "no per-clip values", not as an error.
        assert cs.load_cfg_alphas(d, use_cache=False) == {}
        assert cs.cfg_alpha_for_clip(d / "000001_0000_abces.flac",
                                     default=2.0, use_cache=False) == 2.0

        cs.record_cfg_alpha(d, stem="000863_0005_acineux", cfg_alpha=1.6,
                            voice="h_frm_chf", attempt=3, attempts=3,
                            reason="over budget")
        cs.record_cfg_alpha(d, stem="001355_0006_adherence", cfg_alpha=1.8,
                            voice="h_frm_chf", attempt=2, attempts=3,
                            reason="over budget")
        # The same clip regenerated later (02_audit_durations.py deletes a bad clip and
        # the next resume re-draws it) appends a NEWER row, which must win.
        cs.record_cfg_alpha(d, stem="000863_0005_acineux", cfg_alpha=1.4,
                            voice="h_frm_chf", attempt=4, attempts=4,
                            reason="over budget")

        table = cs.load_cfg_alphas(d, use_cache=False)
        assert table == {"000863_0005_acineux": 1.4, "001355_0006_adherence": 1.8}, table

        rows = [json.loads(ln) for ln in
                cs.sidecar_path(d).read_text().splitlines()]
        assert len(rows) == 3 and rows[0]["voice"] == "h_frm_chf"
        assert rows[0]["reason"] == "over budget" and rows[0]["attempt"] == 3
        assert "ts" in rows[0], "a row with no timestamp cannot be audited later"

        # The file is appended to live from a thread pool, so a run killed mid-write
        # can leave a truncated last line. That must cost that one row, not the file.
        with cs.sidecar_path(d).open("a") as fh:
            fh.write('{"stem": "009999_0000_trunc", "cfg_al')
        assert cs.load_cfg_alphas(d, use_cache=False) == table

        # By path, which is how the manifest side asks.
        assert cs.cfg_alpha_for_clip(d / "000863_0005_acineux.flac", default=2.0,
                                     use_cache=False) == 1.4
        assert cs.cfg_alpha_for_clip(d / "000001_0000_abces.flac", default=2.0,
                                     use_cache=False) == 2.0
    print("OK the sidecar round trips, newest row wins, a torn line costs one row")


def test_stage06_publishes_the_per_clip_value() -> None:
    """stage 06's cfg_alpha column follows the sidecar, not just --original-cfg-alpha."""
    improve = load_module(IMPROVE_PATH, "improve_cfg_sidecar")
    cs = load_module(SIDECAR_PATH, "cfg_sidecar_stage06")

    with tempfile.TemporaryDirectory() as td:
        audio = Path(td) / "dictionary"
        audio.mkdir()
        # The clips must EXIST: resolve_audio only returns a path it can stat, which is
        # also what keeps a stale sidecar row from being attached to nothing.
        for stem in ("000863_0005_acineux", "000001_0000_abces"):
            (audio / f"{stem}.flac").write_bytes(b"audio")
        cs.record_cfg_alpha(audio, stem="000863_0005_acineux", cfg_alpha=1.6,
                            voice="h_frm_chf", attempt=3, attempts=3,
                            reason="over budget")
        cs.clear_cache()  # the reader caches per directory; this test just wrote one

        manifest = Path(td) / "train.jsonl"
        manifest.write_text("")
        cfg = SimpleNamespace(audio_key="audio_filepath", audio_root=None,
                              original_cfg_alpha=2.0)

        descended = {"audio_filepath": "dictionary/000863_0005_acineux.flac"}
        plain = {"audio_filepath": "dictionary/000001_0000_abces.flac"}
        missing = {"audio_filepath": "dictionary/000777_0000_absent.flac"}

        assert improve.original_cfg_alpha_for(descended, cfg, manifest) == 1.6, \
            "a clip stage 05 re-drew at 1.6 must not be published as a 2.0 clip"
        assert improve.original_cfg_alpha_for(plain, cfg, manifest) == 2.0
        # A row whose audio cannot be found falls back to the run-wide value, i.e. to
        # exactly what the column said before the sidecar existed.
        assert improve.original_cfg_alpha_for(missing, cfg, manifest) == 2.0
        cs.clear_cache()
    print("OK stage 06 publishes the per-clip cfg_alpha when stage 05 recorded one")


if __name__ == "__main__":
    test_ladder_rungs_and_floor()
    test_sidecar_round_trip()
    test_stage06_publishes_the_per_clip_value()
    print("OK all")
