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

What is NOT covered: `load_cpu_model`, which needs the 619 MB int8 ONNX export. The
scoring half of `check_clip` IS covered, by handing it a stub recognizer and a silent
clip: the transcript is the only thing the model contributes, so the gates, the stored
row and the defect shape can all be exercised without it.

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
    base = dict(sample_percent=5.0, max_queue=10, backfill=False, cer_threshold=0.08,
                tail_cer_threshold=0.12, window_cer_threshold=0.15, bad_run_threshold=30,
                repeat_threshold=qc.DEFAULT_REPEAT_GATE,
                reference_key="asr_training_source",
                out_name="qc_stt_sidecar.jsonl", workers=1, keep_transcripts=False,
                summary_every=0, stem_requests={})
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


def test_stems_from_checks_exactly_what_the_acoustic_filter_flagged() -> None:
    """--stems-from routes the pre-filter's flags past the sample, and nothing else.

    The economics of the two-stage QC live in this test: with the sample off, a clip is
    transcribed because the acoustic pass flagged it, and a clip the acoustic pass
    cleared costs nothing. The `looped: false` row and the stem belonging to another
    stage are the two ways a caller could accidentally queue the whole corpus.
    """
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        flags = Path(td) / "qc_acoustic_sidecar.jsonl"
        flags.write_text("\n".join([
            json.dumps({"stem": "000001_0001_un", "stage": watch.name,
                        "similarity": 0.91, "looped": True}),
            json.dumps({"stem": "000002_0001_deux", "stage": watch.name,
                        "similarity": 0.31, "looped": False}),
            json.dumps({"stem": "000003_0001_trois", "stage": "04_PARROT",
                        "similarity": 0.95, "looped": True}),
            "{ torn",
        ]) + "\n")
        requests = qc.load_stem_requests((flags,))
        assert requests == {watch.name: {"000001_0001_un"},
                            "04_PARROT": {"000003_0001_trois"}}, requests

        # sample_percent 0: the ONLY reason to transcribe anything is the flag.
        ctx = qc.Context(make_cfg(sample_percent=0.0, stem_requests=requests),
                         {watch.name: watch})
        for stem in ("000001_0001_un", "000002_0001_deux", "000003_0001_trois"):
            (watch.audio / f"{stem}.flac").write_text("x")
        assert qc.poll_audio(watch, ctx) == 3            # three clips seen...
        queued = sorted((j.stem, j.reason) for j in ctx.samples)
        assert queued == [("000001_0001_un", "acoustic")], queued   # ...one checked

        # A bare stem list applies to whatever stage is being followed.
        plain = Path(td) / "stems.txt"
        plain.write_text("# the ones I want\n000002_0001_deux\n\n")
        watch2 = make_watch(Path(td) / "second")
        ctx2 = qc.Context(make_cfg(sample_percent=0.0,
                                   stem_requests=qc.load_stem_requests((plain,))),
                          {watch2.name: watch2})
        (watch2.audio / "000002_0001_deux.flac").write_text("x")
        (watch2.audio / "000009_0001_neuf.flac").write_text("x")
        qc.poll_audio(watch2, ctx2)
        assert sorted(j.stem for j in ctx2.samples) == ["000002_0001_deux"]


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


def test_a_backfill_queue_is_uncapped_and_says_so_when_it_is_not() -> None:
    """`--max-queue 0` means no cap, which is what a backfill needs.

    The cap is a backlog policy: while following a live run, a clip from an hour ago
    says less about the run's current state than the one just written, so dropping the
    oldest is right. A --backfill is the opposite situation: the queue is not a backlog
    but the known population on disk, and trimming it returns incomplete coverage while
    still exiting 0. That shipped: a --backfill --once sweep of 602,789 clips under the
    old default of 2000 checked 2000 of them, logged "63795 sample(s) dropped to keep
    up", and looked finished.
    """
    with tempfile.TemporaryDirectory() as td:
        watch = make_watch(Path(td))
        ctx = qc.Context(make_cfg(max_queue=0, backfill=True), {watch.name: watch})
        for i in range(50):
            ctx.enqueue(qc.Job(stage=watch.name, stem=f"s{i}", path=Path("x"),
                               reason="sample", key=f"s{i}"))
        assert len(ctx.samples) == 50, "a cap of 0 must not drop anything"
        assert ctx.stats.dropped == 0

        # A cap that IS set during a backfill is a mistake worth one loud line, not a
        # silent 2% sample. One line, not one per dropped clip.
        captured: list[str] = []
        sink = qc.logger.add(lambda m: captured.append(m), level="WARNING")
        try:
            tight = qc.Context(make_cfg(max_queue=3, backfill=True),
                               {watch.name: watch})
            for i in range(10):
                tight.enqueue(qc.Job(stage=watch.name, stem=f"t{i}", path=Path("x"),
                                     reason="sample", key=f"t{i}"))
        finally:
            qc.logger.remove(sink)
        assert tight.stats.dropped == 7, tight.stats.dropped
        warnings = [m for m in captured if "max-queue" in m]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"

        # Following a live run keeps the old behaviour, warning included: silence there
        # is correct, the drop is the policy working.
        captured.clear()
        sink = qc.logger.add(lambda m: captured.append(m), level="WARNING")
        try:
            tail = qc.Context(make_cfg(max_queue=3, backfill=False),
                              {watch.name: watch})
            for i in range(10):
                tail.enqueue(qc.Job(stage=watch.name, stem=f"f{i}", path=Path("x"),
                                    reason="sample", key=f"f{i}"))
        finally:
            qc.logger.remove(sink)
        assert tail.stats.dropped == 7
        assert not [m for m in captured if "max-queue" in m], \
            "a tail run drops by design and must not cry about it"


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


def test_classify_defect_shapes() -> None:
    """Each failure shape is named from the two texts, and a clean-ish clip is not a
    loop just because clinical French repeats itself."""
    ref = ("Le patient presente une douleur thoracique. Le patient a recu "
           "du paracetamol et de la morphine.")
    # A derailed take says the same clause over and over.
    assert qc.classify_defect(ref, ref + " et de la morphine et de la morphine "
                              "et de la morphine") == "loop"
    # The LABEL's own repetition must not count: here both texts repeat "de la"
    # exactly as often, so the difference is wording, not a loop.
    assert qc.classify_defect("de la fievre de la toux de la douleur",
                              "de la fievre de la tour de la douceur") == "wrong_words"
    # Stopped early / kept going, with no single clause dominating.
    assert qc.classify_defect(ref, "Le patient presente une douleur") == "truncated"
    assert qc.classify_defect(
        ref, ref + " Il a egalement signale des nausees, une asthenie marquee, "
        "une anorexie et des sueurs nocturnes abondantes depuis plusieurs jours "
        "sans fievre associee ni perte de poids documentee.") == "overrun"
    # A clause said TWICE is a loop: this is the common derailment, and it used to be
    # filed as `wrong_words` (which is what made the mix read as "mostly no acoustic
    # signature").
    doubled = ("La patiente recoit deux grammes par jour pendant dix jours "
               "recoit deux grammes par jour pendant dix jours puis une surveillance.")
    long_ref = ("La patiente recoit deux grammes par jour pendant dix jours "
                "puis une surveillance.")
    assert qc.classify_defect(long_ref, doubled) == "loop"
    # But the LABEL's own repetition does not become a loop: a dosage list that says the
    # clause twice is matched by a take that says it twice, EVEN when the recognizer
    # mishears the same word in both copies. That misread window is a span the label
    # never contains, so comparing windows verbatim would call it a loop; the shared
    # rule forgives a window the label writes that often within one word.
    twice = long_ref + " " + long_ref
    assert qc.classify_defect(twice, twice.replace("recoit", "recevait")) == "wrong_words"
    # A four-word stutter is NOT named a loop, and that is a measured choice rather than
    # an oversight: a window of four words cannot see one word said four times, and an
    # arm for it would have added 10 clips per 19,120 to the repair list of which 4 were
    # real, its misfires being normalization artefacts (a label's "100000" read back as
    # "nil nil nil"). A word said over is also the most self-similar waveform there is,
    # so the acoustic scan is its detector; this one keeps a single rule. Longer runs,
    # which is the shape a derailed take actually produces, still read as loops.
    assert qc.classify_defect("le patient tousse",
                              "le patient tousse tousse tousse tousse") == "overrun"
    assert qc.classify_defect("le patient tousse",
                              "le patient" + " tousse" * 9) == "loop"
    # Same length, different words: the shape that no acoustic detector can see.
    assert qc.classify_defect("le patient tousse", "la patiente touche") == "wrong_words"
    # An empty transcript is a truncation, not a crash.
    assert qc.classify_defect(ref, "") == "truncated"
    assert qc.classify_defect(ref, None) == "truncated"
    # Every answer is one of the documented shapes.
    assert set(qc.DEFECT_SHAPES) >= {
        qc.classify_defect(ref, h) for h in (ref, "", "a a a a a a", ref * 3)}


def test_defect_mix_report_reads_existing_rows() -> None:
    """The report classifies rows written before the `defect` field existed, recomputes
    a stored shape rather than trusting it, and survives a torn final line."""
    rows = [
        # Clean rows are counted but never classified.
        {"stem": "a", "over_gate": False, "cer": 0.01, "voice": "v1"},
        # Flagged, no text: counted as flagged, not classifiable.
        {"stem": "b", "over_gate": True, "cer": 0.4, "voice": "v1"},
        # Flagged with text, no stored shape: classified on the fly.
        {"stem": "c", "over_gate": True, "cer": 0.4, "voice": "v1",
         "reference": "le patient tousse depuis trois jours",
         "transcript": "le patient tousse depuis trois jours tousse depuis trois "
                       "jours tousse depuis trois jours"},
        # A stored shape is recomputed, not trusted: this row claims `overrun` while
        # its two texts are identical, so the current rule says `wrong_words`. That is
        # what lets a classifier change be re-measured over rows already on disk.
        {"stem": "d", "over_gate": True, "cer": 0.3, "voice": "v2",
         "defect": "overrun", "reference": "x y z", "transcript": "x y z"},
        # Below the severity split, so it lands in the overall mix only.
        {"stem": "e", "over_gate": True, "cer": 0.09, "voice": "v2",
         "reference": "le patient tousse", "transcript": "la patiente touche"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "qc_stt_sidecar.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows)
                        + '{"stem": "torn", "over_g')
        captured: list[str] = []
        sink = qc.logger.add(lambda message: captured.append(message), level="INFO")
        try:
            assert qc.report_defect_mix({"st": path}) == 3
        finally:
            qc.logger.remove(sink)
        # Row "d"'s stored `overrun` must not survive into the printed mix.
        assert "overrun" not in "".join(captured)
        # A missing file is reported, not fatal.
        assert qc.report_defect_mix({"st": Path(td) / "nope.jsonl"}) == 0


def test_a_loop_is_flagged_even_when_every_cer_gate_is_happy() -> None:
    """The gate that exists because the other four cannot see a loop.

    Whisper deduplicates repeated speech, so a derailed take comes back from the primary
    judge reading like a clean one and every CER gate passes it. This is the whole
    reason the Parakeet sweep exists, and a metric computed but not gated on would have
    left the clip in the dataset: before this gate, `classify_defect` was only ever
    consulted AFTER a CER gate had already fired, so a loop that scored clean was never
    named. Measured on real rows, the separation is not available to a threshold on the
    four: a confirmed loop scores CER 0.338 while "monsieur renaud" read as "monsieur
    renault" scores 0.321.
    """
    import numpy as np
    import soundfile as sf

    # Long label, short repeat: that ratio is the blind spot, because an insertion costs
    # the CER gates only its own length. Here one four-word clause is read twice over a
    # 385-character label, which is CER 0.052, worst window 0.094, nothing missing.
    label = ("le patient de soixante douze ans est hospitalise pour une pneumopathie "
             "aigue communautaire de la base droite il recoit de l amoxicilline un "
             "gramme par voie intraveineuse toutes les huit heures pendant sept jours "
             "puis un relais par voie orale la surveillance comporte la temperature la "
             "saturation en oxygene et la numeration formule sanguine tous les deux "
             "jours jusqu a la sortie du service de medecine interne")
    looped = label.replace(" pendant sept jours",
                           " toutes les huit heures pendant sept jours", 1)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stem = "000001_0001_paracetamol"
        watch = make_watch(tmp, rows=[{"term_index": 1, "variant_index": 1,
                                       "term": "paracetamol", "source_id": "DOC1",
                                       "asr_training_source": label}])
        sf.write(watch.audio / f"{stem}.flac", np.zeros(24_000, "float32"), 24_000)
        ctx = qc.Context(make_cfg(), {"01_dictionnary": watch})
        job = qc.Job(stem=stem, stage="01_dictionnary", reason="sample",
                     path=watch.audio / f"{stem}.flac", key=stem)
        qc.check_clip(job, SimpleNamespace(recognize=lambda *a, **k: looped),
                      watch, ctx)
        row = json.loads((watch.audio / "qc_stt_sidecar.jsonl").read_text().splitlines()[0])

    # Every distance gate is satisfied, and the clip is flagged anyway.
    assert row["cer"] < 0.08, row
    # None where the clip is shorter than the tail window, which is not a failure.
    assert row["tail_cer"] is None or row["tail_cer"] < 0.12, row
    assert row["window_cer"] < 0.15 and row["bad_run_chars"] < 30, row
    assert row["repeat_excess"] > 0 and row["over_gate"] is True, row
    assert row["defect"] == "loop", row


def test_the_loop_count_is_stored_on_clean_rows_too() -> None:
    """Stored on every row for the same reason the window numbers are: so the gate can
    be re-swept over the sidecar instead of transcribing the corpus a second time."""
    import numpy as np
    import soundfile as sf

    label = "la tension arterielle est a cent trente sur quatre vingt"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stem = "000002_0001_tension"
        watch = make_watch(tmp, rows=[{"term_index": 2, "variant_index": 1,
                                       "term": "tension", "source_id": "DOC1",
                                       "asr_training_source": label}])
        sf.write(watch.audio / f"{stem}.flac", np.zeros(24_000, "float32"), 24_000)
        ctx = qc.Context(make_cfg(), {"01_dictionnary": watch})
        job = qc.Job(stem=stem, stage="01_dictionnary", reason="sample",
                     path=watch.audio / f"{stem}.flac", key=stem)
        qc.check_clip(job, SimpleNamespace(recognize=lambda *a, **k: label), watch, ctx)
        row = json.loads((watch.audio / "qc_stt_sidecar.jsonl").read_text().splitlines()[0])
    assert row["over_gate"] is False and row["repeat_excess"] == 0, row
    # And `--repeat-threshold 0` turns the gate off without removing the measurement.
    assert make_cfg(repeat_threshold=0).repeat_threshold == 0


def make_prep(seconds: float, *, sr: int = 24_000) -> qc.ClipInput:
    """A ClipInput with only what the grouping reads: a waveform length and a rate."""
    import numpy as np

    return qc.ClipInput(row={}, reference="x", voice="v",
                        wav=np.zeros(int(seconds * sr), "float32"), sr=sr,
                        duration=seconds, t0=0.0)


def test_batches_are_capped_by_padded_seconds_not_by_clip_count() -> None:
    """The cap that actually binds, and the reason `--batch` can sit at 64.

    What the encoder allocates for is the batch PADDED to its longest clip, so the cost
    of a batch is clips x longest, not clips. With this corpus' p50 of 20.3 s a 600 s cap
    is reached at ~29 clips, which is why `--batch 16` and `--batch 64` measured the same
    rate: the seconds cap had already cut every batch. A count-only cap is what walked
    the card to 23.4 GB and started losing batches to failed allocations.
    """
    # Ten 30 s clips, cap 120 s: four fit (4 x 30 = 120), a fifth would make 150.
    groups = list(qc.batch_groups([(i, make_prep(30.0)) for i in range(10)],
                                  batch=64, max_sec=120.0))
    assert [len(g) for g in groups] == [4, 4, 2], [len(g) for g in groups]

    # The clip count still caps, when it is the smaller of the two.
    groups = list(qc.batch_groups([(i, make_prep(1.0)) for i in range(7)],
                                  batch=3, max_sec=600.0))
    assert [len(g) for g in groups] == [3, 3, 1], [len(g) for g in groups]

    # One long clip takes a batch to itself rather than being dropped for exceeding the
    # cap on its own: every clip must come back with a row.
    groups = list(qc.batch_groups([(0, make_prep(5.0)), (1, make_prep(900.0)),
                                   (2, make_prep(5.0))], batch=64, max_sec=600.0))
    assert [len(g) for g in groups] == [1, 1, 1], [len(g) for g in groups]
    assert sum(len(g) for g in groups) == 3

    # A rate change splits too: `recognize()` takes ONE sample_rate for the whole batch,
    # so a 16 kHz clip in a 24 kHz batch would be resampled as if it were 24 kHz.
    groups = list(qc.batch_groups([(0, make_prep(1.0)), (1, make_prep(1.0)),
                                   (2, make_prep(1.0, sr=16_000))],
                                  batch=64, max_sec=600.0))
    assert [len(g) for g in groups] == [2, 1], [len(g) for g in groups]


def test_a_batch_that_cannot_allocate_is_halved_not_lost() -> None:
    """The regression test for the first full-corpus launch, which lost 576 clips.

    `batch_groups`' cap is sized from a measured peak, but onnxruntime's BFC arena never
    returns GPU memory, so thousands of distinct padded shapes walk its high-water mark
    up for hours: at a 900 s cap it reached 23.4 GB of a 24.6 GB card within 2500 clips
    and the widest batches died on a single 1.47 GB conv buffer. The handler then sat
    around the whole 192-clip sort window, so each failure charged 192 clips to `failed`
    and left them unchecked. Halving keeps the cost at a repeated encode.
    """
    import numpy as np
    import soundfile as sf

    label = "la tension arterielle est a cent trente sur quatre vingt"

    class ArenaBoundModel:
        """Fails any batch wider than `width`, the way a full arena does."""

        def __init__(self, width: int) -> None:
            self.width, self.widths = width, []

        def recognize(self, waveforms, sample_rate=None):
            self.widths.append(len(waveforms))
            if len(waveforms) > self.width:
                raise RuntimeError("Failed to allocate memory for requested buffer "
                                   "of size 1473806336")
            return [label] * len(waveforms)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows, jobs = [], []
        watch = make_watch(tmp, rows=[
            {"term_index": i, "variant_index": 1, "term": f"t{i}",
             "source_id": "DOC1", "asr_training_source": label} for i in range(1, 5)])
        ctx = qc.Context(make_cfg(), {"01_dictionnary": watch})
        for i in range(1, 5):
            stem = f"00000{i}_0001_t{i}"
            sf.write(watch.audio / f"{stem}.flac", np.zeros(24_000, "float32"), 24_000)
            job = qc.Job(stem=stem, stage="01_dictionnary", reason="sample",
                         path=watch.audio / f"{stem}.flac", key=stem)
            jobs.append((job, qc.prepare_clip(job, watch, ctx)))

        model = ArenaBoundModel(width=2)
        qc.run_group(model, jobs, ctx)
        rows = [json.loads(line) for line in
                (watch.audio / "qc_stt_sidecar.jsonl").read_text().splitlines()]

    # Four clips in, four rows out, nothing charged to `failed`.
    assert len(rows) == 4, rows
    assert {r["stem"] for r in rows} == {j.stem for j, _ in jobs}
    assert ctx.stats.failed == 0 and ctx.stats.checked == 4
    # And the retry is a halving, not a fall back to one clip at a time.
    assert model.widths == [4, 2, 2], model.widths

    # A single clip that still fails is the clip's own problem, and is counted once.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        watch = make_watch(tmp, rows=[{"term_index": 1, "variant_index": 1, "term": "t",
                                       "source_id": "DOC1",
                                       "asr_training_source": label}])
        ctx = qc.Context(make_cfg(), {"01_dictionnary": watch})
        stem = "000001_0001_t"
        sf.write(watch.audio / f"{stem}.flac", np.zeros(24_000, "float32"), 24_000)
        job = qc.Job(stem=stem, stage="01_dictionnary", reason="sample",
                     path=watch.audio / f"{stem}.flac", key=stem)
        prep = qc.prepare_clip(job, watch, ctx)
        model = ArenaBoundModel(width=0)
        qc.run_group(model, [(job, prep)], ctx)
        assert not (watch.audio / "qc_stt_sidecar.jsonl").exists()
    assert ctx.stats.failed == 1 and ctx.stats.checked == 0


def test_a_batched_clip_is_scored_exactly_like_a_lone_one() -> None:
    """The batched path must be a throughput change and nothing else.

    It exists because the GPU sweep is 9x faster in batches (4.3 clips/s against 35.7),
    and the way it stays only that is by reusing `score_clip`: both paths read the same
    manifest row, compute the same five gates and write the same row. Verified end to end
    too, on 318 real acronym clips, where 317 rows matched on every field and
    `over_gate` matched on all 318 (the one difference was a hyphen, from fp16).
    """
    import numpy as np
    import soundfile as sf

    label = "le patient recoit de l amoxicilline un gramme trois fois par jour"
    heard = "le patient recoit de l amoxicilline un gramme deux fois par jour"

    def one_clip(batched: bool) -> dict:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stem = "000001_0001_amoxicilline"
            watch = make_watch(tmp, rows=[{"term_index": 1, "variant_index": 1,
                                           "term": "amoxicilline", "source_id": "DOC1",
                                           "asr_training_source": label}])
            sf.write(watch.audio / f"{stem}.flac", np.zeros(48_000, "float32"), 24_000)
            ctx = qc.Context(make_cfg(keep_transcripts=True),
                             {"01_dictionnary": watch})
            job = qc.Job(stem=stem, stage="01_dictionnary", reason="sample",
                         path=watch.audio / f"{stem}.flac", key=stem)
            if batched:
                # The batch hands `recognize()` a LIST and gets a list back, which is
                # the only difference in the call.
                model = SimpleNamespace(recognize=lambda w, **k: [heard] * len(w))
                qc.run_group(model, [(job, qc.prepare_clip(job, watch, ctx))], ctx)
            else:
                qc.check_clip(job, SimpleNamespace(recognize=lambda *a, **k: heard),
                              watch, ctx)
            line = (watch.audio / "qc_stt_sidecar.jsonl").read_text().splitlines()[0]
            return json.loads(line)

    lone, batched = one_clip(False), one_clip(True)
    # `stt_s` is per clip in one path and a batch's cost divided by its clips in the
    # other, so it is the one field that is expected to differ.
    for field_name in sorted(set(lone) | set(batched)):
        if field_name == "stt_s":
            continue
        assert lone.get(field_name) == batched.get(field_name), (
            field_name, lone.get(field_name), batched.get(field_name))
    # Not a pair of empty rows: the gates really ran and really stored their evidence.
    assert batched["transcript"] == heard and batched["cer"] > 0
    assert "window_cer" in batched and "repeat_excess" in batched


def test_rescore_adds_the_loop_gate_without_transcribing_anything() -> None:
    """A gate added mid-sweep must be recoverable from the rows already written.

    49,339 clips were judged before the loop gate existed and carry no `repeat_excess`,
    i.e. no answer to the one question this judge was added to ask. Re-transcribing them
    is ~33 min of GPU; re-scoring them is seconds, and it is only possible because
    `score_clip` stores the window numbers on clean rows too and keeps the transcript and
    reference under --keep-transcripts. This checks the recovery, that a loop the old row
    missed is now caught, and that a dry run writes nothing.
    """
    label = ("le patient de soixante douze ans est hospitalise pour une pneumopathie "
             "aigue communautaire de la base droite il recoit de l amoxicilline un "
             "gramme par voie intraveineuse toutes les huit heures pendant sept jours "
             "puis un relais par voie orale la surveillance comporte la temperature la "
             "saturation en oxygene et la numeration formule sanguine tous les deux "
             "jours jusqu a la sortie du service de medecine interne")
    looped = label.replace(" pendant sept jours",
                           " toutes les huit heures pendant sept jours", 1)

    # Two rows in the shape the old sweep wrote: no `repeat_excess` at all. The first is
    # the blind spot, a loop that every distance gate passes.
    old_rows = [
        {"stem": "a", "stage": "01_dictionnary", "reason": "sample", "voice": "v",
         "duration": 30.0, "cer": 0.052, "wer": 0.1, "tail_cer": None,
         "window_cer": 0.094, "bad_run_chars": 0, "over_gate": False,
         "stt_s": 1.0, "ts": "2026-10-25T20:00:00Z",
         "transcript": looped, "reference": label},
        {"stem": "b", "stage": "01_dictionnary", "reason": "sample", "voice": "v",
         "duration": 30.0, "cer": 0.0, "wer": 0.0, "tail_cer": None,
         "window_cer": 0.0, "bad_run_chars": 0, "over_gate": False,
         "stt_s": 1.0, "ts": "2026-10-25T20:00:00Z",
         "transcript": label, "reference": label},
        # A row with no transcript cannot be rescored and must survive untouched.
        {"stem": "c", "stage": "01_dictionnary", "reason": "sample", "voice": "v",
         "duration": 30.0, "cer": 0.5, "over_gate": True, "stt_s": 1.0,
         "ts": "2026-10-25T20:00:00Z"},
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "qc.jsonl"
        with path.open("w") as fh:
            for row in old_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        original = path.read_text()
        cfg = make_cfg()

        dry = qc.rescore_sidecar(path, cfg, apply=False)
        assert path.read_text() == original, "a dry run must not touch the file"
        assert dry["rows"] == 3 and dry["rescored"] == 2 and dry["skipped"] == 1
        assert dry["gained_gate"] == 2, dry
        # The loop is the one that flips, and only it.
        assert dry["flipped_on"] == 1 and dry["flipped_off"] == 0, dry

        applied = qc.rescore_sidecar(path, cfg, apply=True)
        assert applied == dry, (applied, dry)
        rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 3
    by_stem = {r["stem"]: r for r in rows}
    # The loop now carries the gate that can see it, and is flagged on that alone.
    assert by_stem["a"]["repeat_excess"] > 0 and by_stem["a"]["over_gate"] is True
    assert by_stem["a"]["cer"] < 0.08 and by_stem["a"]["window_cer"] < 0.15
    assert by_stem["a"]["defect"] == "loop", by_stem["a"]
    # The clean row gains the measurement and stays clean.
    assert by_stem["b"]["repeat_excess"] == 0 and by_stem["b"]["over_gate"] is False
    # The row that could not be rescored is byte-for-byte what it was.
    assert "rescored_ts" not in by_stem["c"] and by_stem["c"]["cer"] == 0.5
    assert "repeat_excess" not in by_stem["c"]
    # And a rescored row says when, so a later pass can tell the two generations apart.
    assert by_stem["a"]["rescored_ts"].endswith("Z")


def test_rescore_leaves_rows_that_already_carry_every_gate_alone() -> None:
    """The default pass must cost nothing on the rows the sweep scored itself.

    Scoring one row is ~26 ms of pure Python (the sliding window dominates), so over
    602,789 rows a blanket pass is ~4.3 h single-threaded while the rows that actually
    need it, the 49,339 predating the loop gate, are ~21 min. A row that already carries
    `repeat_excess` was scored by `score_clip` with these same functions on these same
    texts, so recomputing it can only reproduce what is there. `--rescore-all` is the
    escape hatch for a threshold or normalization change, which does invalidate them.
    """
    label = ("le patient recoit de l amoxicilline un gramme par voie intraveineuse "
             "toutes les huit heures pendant sept jours puis un relais par voie orale")
    new_row = {"stem": "new", "stage": "01_dictionnary", "reason": "sample", "voice": "v",
               "duration": 30.0, "cer": 0.0, "wer": 0.0, "tail_cer": None,
               "window_cer": 0.0, "bad_run_chars": 0, "repeat_excess": 0,
               "over_gate": False, "stt_s": 1.0, "ts": "2026-10-26T01:00:00Z",
               "transcript": label, "reference": label}
    # Deliberately wrong stored numbers on a row that HAS the gate: the default pass must
    # not silently correct them, which is what proves it never scored the row at all.
    stale_row = dict(new_row, stem="stale", cer=0.9, window_cer=0.9, over_gate=True)
    old_row = {"stem": "old", "stage": "01_dictionnary", "reason": "sample", "voice": "v",
               "duration": 30.0, "cer": 0.0, "wer": 0.0, "tail_cer": None,
               "window_cer": 0.0, "bad_run_chars": 0, "over_gate": False,
               "stt_s": 1.0, "ts": "2026-10-25T20:00:00Z",
               "transcript": label, "reference": label}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "qc.jsonl"
        with path.open("w") as fh:
            for row in (new_row, stale_row, old_row):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        cfg = make_cfg()

        got = qc.rescore_sidecar(path, cfg, apply=True)
        assert got["rows"] == 3, got
        assert got["kept"] == 2 and got["rescored"] == 1, got
        assert got["gained_gate"] == 1, got
        by_stem = {json.loads(l)["stem"]: json.loads(l)
                   for l in path.read_text().splitlines()}
        assert "rescored_ts" not in by_stem["new"] and "rescored_ts" not in by_stem["stale"]
        assert by_stem["stale"]["cer"] == 0.9, "a kept row must be passed through verbatim"
        assert by_stem["old"]["repeat_excess"] == 0 and by_stem["old"]["rescored_ts"]

        # --rescore-all is the opposite contract: it touches everything, and the wrong
        # numbers on the stale row are then corrected.
        allgot = qc.rescore_sidecar(path, cfg, apply=True, all_rows=True)
        assert allgot["rescored"] == 3 and allgot["kept"] == 0, allgot
        assert allgot["flipped_off"] == 1, allgot
        fixed = {json.loads(l)["stem"]: json.loads(l)
                 for l in path.read_text().splitlines()}
        assert fixed["stale"]["cer"] == 0.0 and fixed["stale"]["over_gate"] is False


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall CPU QC tail tests passed")


if __name__ == "__main__":
    main()
