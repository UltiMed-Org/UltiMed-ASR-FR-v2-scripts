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
#   "numpy",
# ]
# ///
"""The CPU QC tail (`05_generate_audio/03_qc_cpu_stt.py`), minus the model.

The script's job is to pick the RIGHT ~5% of a running generation to listen to on the
spare CPU cores, and everything that can go wrong with it is in the picking, not in the
transcription: a sample that silently correlates with the voice assignment would make
the per-voice CER table it prints meaningless, a stem-keyed dedup would skip the second
take of a clip that was deleted and re-drawn (which is exactly the clip worth hearing),
and a queue without a cap would grow for eight days. So this test covers the selection,
the dedup, the resume and the backlog policy, and never loads onnxruntime.

What is NOT covered: `load_cpu_model` and `check_clip`, which need the 619 MB int8 ONNX
export and real audio. Those were verified by hand against the live run instead.

    uv run tests/test_qc_cpu_stt.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
QC_PATH = HERE.parent / "05_generate_audio" / "03_qc_cpu_stt.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qc = load_module(QC_PATH, "qc_cpu_stt_under_test")
sys.path.insert(0, str(HERE.parent / "utils"))
from voice_assignment import VOICES_FR_V2, voice_for_key  # noqa: E402


def make_cfg(**over) -> SimpleNamespace:
    """The subset of the CLI the pollers and the queue actually read."""
    base = dict(sample_percent=5.0, max_queue=10, cer_threshold=0.08,
                tail_cer_threshold=0.12, reference_key="asr_training_source",
                out_name="qc_stt_sidecar.jsonl", workers=1, keep_transcripts=False,
                summary_every=0)
    base.update(over)
    return SimpleNamespace(**base)


def make_watch(tmp: Path, *, rows: list[dict] | None = None) -> qc.StageWatch:
    """A stage directory with an audio dir and, optionally, a manifest."""
    audio = tmp / "generated_audios"
    audio.mkdir(parents=True, exist_ok=True)
    jsonl = tmp / "generated_dataset.jsonl"
    with jsonl.open("w") as fh:
        for row in rows or []:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return qc.StageWatch(name="01_dictionnary", mode="row", jsonl=jsonl, audio=audio)


def test_sample_is_deterministic_and_the_right_size() -> None:
    """The draw is a pure function of the stem, so a restart re-draws the same clips."""
    stems = [f"{i:06d}_0001_terme" for i in range(20_000)]

    drawn = [s for s in stems if qc.is_sampled(s, percent=5.0)]
    # Binomial(20000, 0.05) has sd ~31, so +/-150 is ~5 sd: wide enough never to flake,
    # tight enough to catch a draw that is off by a factor.
    assert 850 <= len(drawn) <= 1150, len(drawn)
    assert [s for s in stems if qc.is_sampled(s, percent=5.0)] == drawn

    # The two extremes have to be exact, because --sample-percent 0 is how the script
    # is told "only the retried clips" and 100 is how a one-shot audit is run.
    assert not any(qc.is_sampled(s, percent=0.0) for s in stems)
    assert all(qc.is_sampled(s, percent=100.0) for s in stems)


def test_sample_does_not_correlate_with_the_voice_draw() -> None:
    """The salt is load-bearing: the sample must not favour any voice.

    `voice_assignment.voice_for_key` also hashes with blake2b, and the whole point of
    the 5% sample is to compare voices against each other, so a sample that over-draws
    one voice would show CER drift that is really sampling bias.
    """
    stems = [f"{i:06d}_0001_terme" for i in range(40_000)]
    drawn = [s for s in stems if qc.is_sampled(s, percent=5.0)]
    counts = Counter(voice_for_key(s, VOICES_FR_V2) for s in drawn)

    assert len(counts) == len(VOICES_FR_V2), counts
    share = 1.0 / len(VOICES_FR_V2)
    for voice, n in counts.items():
        assert 0.5 * share < n / len(drawn) < 1.5 * share, (voice, n, len(drawn))


def test_poll_audio_draws_clips_only() -> None:
    """A clip in flight, the sidecars and the provenance file are not audio to check."""
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        ctx = qc.Context(make_cfg(sample_percent=100.0), {watch.name: watch})
        for name in ("000001_0001_un.flac", "000002_0001_deux.flac",
                     "000003_0001_trois.flac.tmp", "cfg_alpha_sidecar.jsonl",
                     "qc_stt_sidecar.jsonl", "voice_assignment.json"):
            (watch.audio / name).write_text("x")

        assert qc.poll_audio(watch, ctx) == 2
        assert sorted(j.stem for j in ctx.samples) == ["000001_0001_un",
                                                       "000002_0001_deux"]
        # Second poll: the directory is unchanged, so nothing is queued twice.
        assert qc.poll_audio(watch, ctx) == 0
        assert len(ctx.samples) == 2


def test_poll_sidecar_queues_every_take_and_survives_a_torn_line() -> None:
    """Retried clips: one job per sidecar ROW, including a re-draw of the same stem."""
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        ctx = qc.Context(make_cfg(sample_percent=0.0), {watch.name: watch})
        sidecar = watch.audio / qc.SIDECAR_FILENAME
        (watch.audio / "000001_0001_un.flac").write_text("x")
        (watch.audio / "000002_0001_deux.flac").write_text("x")

        def row(stem: str, cfg_alpha: float, ts: str) -> str:
            return json.dumps({"stem": stem, "cfg_alpha": cfg_alpha, "ts": ts}) + "\n"

        sidecar.write_text(row("000001_0001_un", 1.8, "2026-10-14T10:47:29Z"))
        qc.poll_sidecar(watch, ctx)
        assert [(j.stem, j.cfg_alpha, j.reason) for j in ctx.retries] == [
            ("000001_0001_un", 1.8, "retry")]

        # Nothing new: no second job for the same take.
        qc.poll_sidecar(watch, ctx)
        assert len(ctx.retries) == 1

        # Stage 05 is mid-append. The half-written row must be re-read whole on the
        # next poll, not parsed now and not skipped.
        with sidecar.open("a") as fh:
            fh.write('{"stem": "000002_0001_deux", "cfg_alp')
        qc.poll_sidecar(watch, ctx)
        assert len(ctx.retries) == 1
        with sidecar.open("a") as fh:
            fh.write('ha": 1.6, "ts": "2026-10-14T11:00:00Z"}\n')
        qc.poll_sidecar(watch, ctx)
        assert [j.stem for j in ctx.retries] == ["000001_0001_un", "000002_0001_deux"]

        # The duration auditor deleted a clip and stage 05 re-drew it, so the stem gets
        # a SECOND row with a new timestamp. That is a different take of the same label
        # and nobody has heard it either, so it is a new job: a dedup on the stem alone
        # would drop exactly the clip this script exists for.
        sidecar.write_text(sidecar.read_text()
                           + row("000001_0001_un", 2.0, "2026-10-15T09:00:00Z"))
        qc.poll_sidecar(watch, ctx)
        assert [j.stem for j in ctx.retries] == ["000001_0001_un", "000002_0001_deux",
                                                 "000001_0001_un"]

        # A row whose clip is gone (deleted by the auditor, not yet re-drawn) is not a
        # job: there is nothing on disk to transcribe.
        (watch.audio / "000003_0001_trois.flac").unlink(missing_ok=True)
        sidecar.write_text(sidecar.read_text()
                           + row("000003_0001_trois", 1.4, "2026-10-15T09:30:00Z"))
        qc.poll_sidecar(watch, ctx)
        assert len(ctx.retries) == 3


def test_retries_are_served_first_and_never_dropped() -> None:
    """The backlog policy: samples are interchangeable, retried takes are not."""
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        ctx = qc.Context(make_cfg(max_queue=3), {watch.name: watch})
        for i in range(6):
            ctx.enqueue(qc.Job(stage=watch.name, stem=f"s{i}", path=Path("x"),
                               reason="sample", key=f"s{i}"))
        for i in range(4):
            ctx.enqueue(qc.Job(stage=watch.name, stem=f"r{i}", path=Path("x"),
                               reason="retry", key=f"r{i}@t"))

        # The cap applies to samples only, and it is the OLDEST that go: a clip from an
        # hour ago says less about the run's current state than the one just written.
        assert [j.stem for j in ctx.samples] == ["s3", "s4", "s5"]
        assert ctx.stats.dropped == 3
        assert len(ctx.retries) == 4

        assert [ctx.next_job().stem for _ in range(4)] == ["r0", "r1", "r2", "r3"]
        assert ctx.next_job().stem == "s3"


def test_resume_reads_its_own_output() -> None:
    """A restart must not re-check what the previous process already wrote."""
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        ctx = qc.Context(make_cfg(sample_percent=100.0), {watch.name: watch})
        out = watch.audio / "qc_stt_sidecar.jsonl"
        (watch.audio / "000001_0001_un.flac").write_text("x")
        (watch.audio / "000002_0001_deux.flac").write_text("x")
        sidecar = watch.audio / qc.SIDECAR_FILENAME
        sidecar.write_text(json.dumps({"stem": "000002_0001_deux", "cfg_alpha": 1.8,
                                       "ts": "2026-10-14T10:47:29Z"}) + "\n")

        out.write_text(
            json.dumps({"stem": "000001_0001_un", "reason": "sample"}) + "\n"
            + json.dumps({"stem": "000002_0001_deux", "reason": "retry",
                          "src_ts": "2026-10-14T10:47:29Z"}) + "\n"
            # A run killed mid-write leaves a torn last line, which must not stop the
            # resume from reading the rows before it.
            + '{"stem": "000009_0001_neuf", "reas')
        assert qc.seed_handled(watch, out) == 2

        qc.poll_audio(watch, ctx)
        qc.poll_sidecar(watch, ctx)
        assert not ctx.samples and not ctx.retries

        # ...but a NEW take of that same retried clip still gets checked.
        sidecar.write_text(sidecar.read_text()
                           + json.dumps({"stem": "000002_0001_deux", "cfg_alpha": 1.6,
                                         "ts": "2026-10-15T09:00:00Z"}) + "\n")
        watch.sidecar_offset = 0
        qc.poll_sidecar(watch, ctx)
        assert [j.stem for j in ctx.retries] == ["000002_0001_deux"]


def test_manifest_row_is_found_by_the_clip_name() -> None:
    """The join between a clip on disk and the text it was asked to say.

    The manifest is read by byte offset, keyed on the two zero-padded indexes in the
    clip name, so the ~600k rows of `asr_training_source` never sit in memory for the
    5% this script reads.
    """
    rows = [
        {"term_index": 4534, "variant_index": 6, "term": "apophyse",
         "asr_training_source": "Apophyse unciforme des vertebres cervicales.",
         "source_id": "DOC1"},
        {"term_index": 12, "variant_index": 1, "term": "autre",
         "asr_training_source": "Autre chose.", "source_id": "DOC2"},
    ]
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td), rows=rows)
        qc.ensure_index(watch)
        assert watch.ready and watch.voice_field is not None

        got = qc.manifest_row(watch, "004534_0006_apophyse_unciforme_des_vertebres")
        assert got is not None
        assert got["asr_training_source"].startswith("Apophyse unciforme")
        # The slug is NOT part of the key: it is re-derived from the term and a clip
        # whose slug was truncated by the filesystem would still find its row.
        assert qc.manifest_row(watch, "000012_0001_autre")["asr_training_source"] \
            == "Autre chose."
        assert qc.manifest_row(watch, "999999_0001_absent") is None
        # A second call is a no-op rather than a second pass over the manifest.
        handle = watch.handle
        qc.ensure_index(watch)
        assert watch.handle is handle


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall CPU QC tail tests passed")


if __name__ == "__main__":
    main()
