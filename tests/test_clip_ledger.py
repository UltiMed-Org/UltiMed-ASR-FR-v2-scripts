"""Stdlib test for `utils/clip_ledger.py`, the one-line-per-clip QC view.

It pins the failure that motivated the ledger (2026-10-28): a clip the acoustic scan
flagged `looped` on its CURRENT take must carry `acoustic:loop`, while a flag about a
take that has since been redrawn must not, and the reader must not have to know which
of seven sidecars to open to learn either.

Run: python tests/test_clip_ledger.py

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import clip_ledger as L  # noqa: E402

NOW = 1_700_000_000.0


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _clip(cat: Path, stem: str, mtime: float) -> None:
    path = cat / f"{stem}.flac"
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))


def _gates(scores, gates) -> list[str]:
    return ["cer"] if scores.cer is not None and scores.cer >= gates.cer_threshold else []


def _judge_fn(readings: dict) -> dict:
    return L.judge_block(readings, scores_from_reading=lambda r: SimpleNamespace(cer=r.get("cer")),
                         which_gates=_gates, gates=SimpleNamespace(cer_threshold=0.08))


def _fixture(root: Path) -> Path:
    cat = root / "PARHAF"
    cat.mkdir()
    _clip(cat, "looped_now", NOW)           # flagged on the take on disk
    _clip(cat, "redrawn", NOW + 600.0)      # flagged, then redrawn: the flag is stale
    _clip(cat, "orphan", NOW)               # on disk, in no manifest, never dropped
    _write(cat / "qc_acoustic_sidecar.jsonl", [
        {"stem": "looped_now", "mtime": NOW, "similarity": 0.86, "looped": True},
        {"stem": "redrawn", "mtime": NOW, "similarity": 0.91, "looped": True},
    ])
    _write(cat / "qc_parakeet_ultra.jsonl", [
        {"stem": "looped_now", "ts": NOW + 5, "cer": 0.11, "over_gate": True,
         "reference": "la source épelée"},
        {"stem": "looped_now", "ts": NOW - 5, "cer": 0.50, "over_gate": True},  # older take
    ])
    return cat


def test_flags_describe_the_current_take_only() -> None:
    with tempfile.TemporaryDirectory() as d:
        cat = _fixture(Path(d))
        state = {"PARHAF/looped_now.flac": {
            "text": "la source", "source": "la source épelée", "readings": {
                "turbo": {"text": "ok", "cer": 0.06}, "parakeet": {"text": "ok ok", "cer": 0.11}}}}
        release = {"PARHAF/looped_now.flac": {"splits": ["train"], "dropped": None, "asr_flag": None},
                   "PARHAF/redrawn.flac": {"splits": ["val"], "dropped": None, "asr_flag": None}}
        rows = {r["stem"]: r for r in L.build_category(cat, state, {}, release, _judge_fn)}
        assert set(rows) == {"looped_now", "redrawn", "orphan"}, "one line per clip, orphans included"
        respelled = L.build_category(cat, {"PARHAF/looped_now.flac": {"text": "LA SOURCE ÉPELÉE"}},
                                     {}, release, _judge_fn, normalize=str.lower)
        assert "parakeet_ultra:reference_differs" not in respelled[0]["flags"], \
            "a label only respelled since QC does not invalidate the verdict"
        rewritten = L.build_category(cat, {"PARHAF/looped_now.flac": {"text": "autre chose"}},
                                     {}, release, _judge_fn, normalize=str.lower)
        assert rewritten[0]["stem"] == "looped_now"
        assert "parakeet_ultra:reference_differs" in rewritten[0]["flags"]
        now = rows["looped_now"]
        for flag in ("acoustic:loop", "parakeet_ultra:over_gate", "parakeet:cer"):
            assert flag in now["flags"], f"{flag} missing from {now['flags']}"
        assert "turbo:cer" not in now["flags"], "turbo passed it, which is the whole trap"
        assert now["qc"]["parakeet_ultra"]["cer"] == 0.11, "the latest fresh row wins"
        assert now["qc"]["parakeet_ultra"]["stale_takes"] == 1, "the pre-redraw row is counted"
        assert "parakeet_ultra:reference_differs" not in now["flags"], \
            "scoring against asr_training_source is not a stale reference"
        old = rows["redrawn"]
        assert "acoustic:loop" not in old["flags"], "a verdict about replaced audio must not flag"
        assert old["qc"]["acoustic"] == {"stale_takes": 1}, "but the old look stays countable"
        assert "unreleased" in rows["orphan"]["flags"]
        assert "no_state" in rows["orphan"]["flags"]


def test_an_unknown_sidecar_is_reported_not_skipped() -> None:
    with tempfile.TemporaryDirectory() as d:
        cat = _fixture(Path(d))
        assert L.check_known(cat) == []
        _write(cat / "qc_new_detector.jsonl", [{"stem": "looped_now", "bad": True}])
        assert L.check_known(cat) == ["qc_new_detector.jsonl"]


def test_parrot_double_state_merges_and_flags_a_conflict() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a" / "b").mkdir(parents=True)
        one, two = root / "one.jsonl", root / "two.jsonl"
        _write(one, [{"audio_filepath": "../PARROT/x.flac", "text": "t",
                      "improvement": {"status": "improved"},
                      "transcriptions": {"whisper.cpp/ggml-large-v3-turbo": {"text": "a"}}}])
        _write(two, [{"audio_filepath": "../../PARROT/x.flac", "text": "t",
                      "transcriptions": {"whisper.cpp/ggml-large-v3-turbo": {"text": "b"},
                                         "onnx-asr/parakeet-tdt-0.6b-v3-ultra": {"text": "c"}}}])
        state = L.load_state([(one, root / "a"), (two, root / "a" / "b")])
        assert list(state) == ["PARROT/x.flac"], "both spellings are one clip"
        x = state["PARROT/x.flac"]
        assert x["improvement"] == {"status": "improved"}, "an absent marker never erases one"
        assert set(x["readings"]) == {"turbo", "parakeet"}
        assert x["conflict"], "two different turbo transcripts of one take must be surfaced"


def test_release_fate_and_drop_staleness() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        nemo = root / "NeMO_files"
        nemo.mkdir()
        _write(nemo / "train.jsonl", [{"audio_filepath": "../PARHAF/a.flac"}])
        _write(root / "drops.jsonl", [{"clip": "PARHAF/b.flac", "reason": "exhausted",
                                       "audio_mtime_ns": "1"}])
        _write(root / "flags.jsonl", [{"clip": "PARHAF/b.flac", "reason": "repeated", "words": "w"}])
        rel = L.load_release(nemo, root / "drops.jsonl", root / "flags.jsonl",
                             still_dropped=lambda entry, audio: audio.endswith("PARHAF/b.flac"))
        assert rel["PARHAF/a.flac"]["splits"] == ["train"]
        assert rel["PARHAF/b.flac"]["dropped"] == {"reason": "exhausted", "applies": True}
        assert rel["PARHAF/b.flac"]["asr_flag"]["reason"] == "repeated"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok {name}")
    print("all ok")
