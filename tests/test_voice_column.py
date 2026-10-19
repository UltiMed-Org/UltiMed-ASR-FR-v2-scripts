#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "soundfile>=0.12",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "requests>=2.31",
#   "jiwer>=3.0",
# ]
# ///
"""End to end: the voice stage 05 SENT is the voice stage 99 WRITES.

`utils/voice_assignment.py` is unit-tested by `tests/test_voice_assignment.py`. This
file tests the thing those unit tests cannot see: the two stages agree in practice.
They have to, and nothing about the audio reveals it if they do not. Stage 05 picks a
voice per row and sends it in the request; the clip lands on disk with a filename that
does not mention the voice, and stage 99 re-derives the name months later from the
roster recorded beside the audio. A mismatch anywhere in that chain (the group field,
the roster order, the mode, which row of the todo list a worker got) produces a
`voice` column that is confidently wrong for 600k clips, trains fine, and ruins every
per-voice number computed from the release.

So this runs the real `run_batch` against a faked HTTP layer, with the real
`build_body` and the real request plumbing, and then the real `build_dataset` over the
clips it wrote, and compares the two. Four cases:

1. **The dictionary shape** (`term_index` / `variant_index`): all variants of a term
   share one voice, several voices are used across terms, and every manifest row's
   `voice` is the one that was actually requested for that clip.
2. **The document shape** (`source_id` + `chunk_index`): the group is the document, so
   its chunks share a voice even though their `term_index` differs. This is the case
   the group field exists for.
3. **`--group-by` at stage 99 does not move the voice.** Splitting choices are a
   release-time decision; who read the clip is a fact about the audio, recorded at
   synthesis.
4. **The v1 shapes.** `--voice X` labels every row X (a one-name roster), a run with
   no voice at all writes no `voice` field rather than guessing one, and stage 99's
   `--voice-fallback` is what labels that unrecorded audio when the operator knows
   the name (a v1 rebuild: `fr_female` throughout).
5. **A stage-06 regeneration keeps the clip's own voice.** The column cannot follow a
   regeneration (it is re-derived from the group key), so a replacement clip spoken by
   a different voice is a clip whose label lies. `--tts-voice` is only the fallback for
   rows without the field.

No server and no GPU: `post_json` is replaced by one that returns a real one-second
FLAC, which matters because stage 99 then probes real durations off real files. The
dependency list is the union of the three stage scripts' own, since importing stage 06
pulls in its STT half (jiwer, requests). Run it under uv, like the other tests that
import a stage module:

    uv run tests/test_voice_column.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf

HERE = Path(__file__).resolve().parent
TTS_PATH = HERE.parent / "05_generate_audio" / "01_generate_audio.py"
MANIFEST_PATH = HERE.parent / "99_hf_release" / "01_build_nemo_manifest.py"
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"

sys.path.insert(0, str(HERE.parent / "utils"))
import voice_assignment as va  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one check, print it, and keep going so one run reports everything."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else f'  <- {detail}'}")
    if not ok:
        failures.append(label)


def load_module(path: Path, name: str):
    """Import a stage script by path (they are not importable as modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flac_seconds(seconds: float) -> bytes:
    """A real 24 kHz mono FLAC of that length, so both stages parse real headers.

    Stage 05 reads the duration out of the payload (truncation guard) and stage 99
    probes it off the written file, so a fake byte string would fail for the wrong
    reason.
    """
    buf = io.BytesIO()
    sf.write(buf, [0.0] * int(24000 * seconds), 24000, format="FLAC")
    return buf.getvalue()


# 2.0 s, not 1.0: the texts below are ~29 characters, and stage 05 now rejects a clip
# that is implausibly SHORT for its text (below chars / the voice's fastest measured
# reading speed, 27-31 chars/s). A 1 s clip for 29 characters is 29 chars/s, which the
# per-voice floor flags for the slower voices, so the old stub made this test fail on a
# guard it is not about. 2.0 s is 14.5 chars/s, the roster's median pace.
FLAC = flac_seconds(2.0)
# A second, different payload for stage 06's regeneration: a draw byte-identical to
# the original clip raises SeedDeterminismError there (the seed had no effect), which
# is a guard worth not tripping over in a test about voices.
FLAC_REDO = flac_seconds(2.4)


def tts_args(out_dir: Path, **over) -> SimpleNamespace:
    """A stage-05 namespace with every knob run_batch/synthesize/build_body read."""
    args = dict(
        output=str(out_dir), format="flac", overwrite=False, concurrency=4,
        timeout=30.0, max_audio_seconds=327.68,
        voice=None, voice_mode="group", voices_file=None,
        model=None, instruction=None, seed=None, speed=None, cfg_alpha=None,
        temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        do_sample=None,
    )
    args.update(over)
    return SimpleNamespace(**args)


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write a stage's generated_dataset.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def dictionary_rows() -> list[dict]:
    """Six terms x three variants, the 01_dictionnary / 02_drugs / 07_acronyms shape.

    Six terms because the roster has nine voices and the point is to see several of
    them used; three variants because the point is that a term's variants do NOT
    differ. The text is unique per row, which is how the request is matched back to
    the clip (the request body carries the text, not the filename).
    """
    rows = []
    for term_index in range(6):
        for variant_index in range(3):
            rows.append({
                "term_index": term_index,
                "variant_index": variant_index,
                "term": f"terme{term_index}",
                "asr_training_source": f"Le terme numero {term_index} variante {variant_index}.",
                "asr_training_target": f"Le terme numero {term_index} variante {variant_index}.",
                "category": "dictionary",
            })
    return rows


def document_rows() -> list[dict]:
    """Four documents x three chunks, the 03_PARHAF / 04_PARROT shape.

    `term_index` increases across the whole file (as it does in the real stages), so a
    voice that followed term_index instead of source_id would split every document
    across three speakers: exactly the failure this shape is here to catch.
    """
    rows = []
    term_index = 0
    for doc in range(4):
        for chunk_index in range(3):
            rows.append({
                "source_id": f"PARHAF_{doc:04d}",
                "chunk_index": chunk_index,
                "term_index": term_index,
                "variant_index": 0,
                "term": f"doc{doc}",
                "asr_training_source": f"Document {doc}, passage {chunk_index}.",
                "asr_training_target": f"Document {doc}, passage {chunk_index}.",
                "category": "PARHAF",
            })
            term_index += 1
    return rows


def generate(tts, rows: list[dict], input_path: Path, audio_dir: Path,
             **over) -> dict[str, str | None]:
    """Run the real run_batch over `rows`; return {text: voice requested}.

    Only the HTTP call is faked, so the row -> voice_for_row -> todo -> thread pool
    -> build_body -> request chain is the real one, and what is recorded is what a
    server would have received.
    """
    write_rows(input_path, rows)
    sent: dict[str, str | None] = {}

    def fake_post_json(url: str, body: dict, timeout: float):
        sent[body["input"]] = body.get("voice")
        return 200, FLAC, 0.01

    tts.post_json = fake_post_json
    rc = tts.run_batch("http://unused", input_path, tts_args(audio_dir, **over))
    assert rc == 0, f"run_batch failed with {rc}"
    return sent


def manifest_voices(m99, input_path: Path, audio_dir: Path, out_root: Path,
                    name: str, **over) -> dict[str, str | None]:
    """Run the real build_dataset; return {asr_training_source: voice in full.jsonl}.

    Keyed by the TTS input text because that is the one field both sides carry: the
    manifest row has `asr_training_source`, the request body has it as `input`.
    """
    opts = dict(group_by=None, group_mode=None, voice_fallback=None)
    opts.update(over)
    m99.build_dataset(
        name, input_path, audio_dir, out_root, m99.parse_split("80/10/10"),
        True, True, False, 2, False, opts["group_by"], opts["group_mode"], None,
        opts["voice_fallback"],
    )
    full = out_root / name / "full.jsonl"
    rows = [json.loads(line) for line in full.read_text().splitlines() if line.strip()]
    assert rows, "build_dataset wrote no rows"
    return {r["asr_training_source"]: r.get("voice") for r in rows}


tts = load_module(TTS_PATH, "tts_under_test")
m99 = load_module(MANIFEST_PATH, "manifest_under_test")
# Both stages log at INFO through loguru; the checks below are the output of this run.
tts.logger.remove()
m99.logger.remove()

# ------------------------------------------------ 1. the dictionary shape (term_index)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = dictionary_rows()
    audio = td / "audio" / "dictionary"
    inp = td / "01_dictionnary" / "generated_dataset.jsonl"
    sent = generate(tts, rows, inp, audio)

    check("every row was synthesized", len(sent) == len(rows), f"{len(sent)}/{len(rows)}")
    check("every request carried a voice from the roster",
          all(v in va.VOICES_FR_V2 for v in sent.values()), str(sorted(set(sent.values()))))
    per_term = {}
    for r in rows:
        per_term.setdefault(r["term_index"], set()).add(sent[r["asr_training_source"]])
    check("a term's three variants share one voice",
          all(len(v) == 1 for v in per_term.values()), str(per_term))
    check("several voices are used across the six terms",
          len({next(iter(v)) for v in per_term.values()}) > 1, str(per_term))

    prov = va.read_provenance(audio, inp)
    check("stage 05 recorded the roster beside the audio",
          prov is not None and prov["voices"] == list(va.VOICES_FR_V2)
          and prov["mode"] == "group" and prov["group_field"] == "term_index",
          json.dumps(prov))
    check("the recorded counts cover every row",
          prov is not None and sum(prov["rows_per_voice"].values()) == len(rows),
          json.dumps(prov["rows_per_voice"]) if prov else "no provenance")

    got = manifest_voices(m99, inp, audio, td / "NeMO", "dictionary")
    check("stage 99 labels every clip with the voice stage 05 requested",
          got == sent,
          str({k: (sent[k], v) for k, v in got.items() if sent.get(k) != v}))

    # 3. A splitting override must not move the voice: the recorded field wins.
    forced = manifest_voices(m99, inp, audio, td / "NeMO_forced", "dictionary",
                             group_by="variant_index", group_mode="distribute")
    check("--group-by at stage 99 does not relabel the voices", forced == sent,
          str({k: (sent[k], v) for k, v in forced.items() if sent.get(k) != v}))

# ------------------------------------------------- 2. the document shape (source_id)
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = document_rows()
    audio = td / "audio" / "PARHAF"
    inp = td / "03_PARHAF" / "generated_dataset.jsonl"
    sent = generate(tts, rows, inp, audio)

    prov = va.read_provenance(audio, inp)
    check("the group field is source_id for documents",
          prov is not None and prov["group_field"] == "source_id",
          json.dumps(prov))
    per_doc = {}
    for r in rows:
        per_doc.setdefault(r["source_id"], set()).add(sent[r["asr_training_source"]])
    check("a document's chunks share one voice despite differing term_index",
          all(len(v) == 1 for v in per_doc.values()), str(per_doc))

    got = manifest_voices(m99, inp, audio, td / "NeMO", "PARHAF")
    check("stage 99 agrees on the document shape too", got == sent,
          str({k: (sent[k], v) for k, v in got.items() if sent.get(k) != v}))

# ----------------------------------------------------------------- 4. the v1 shapes
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = dictionary_rows()
    audio = td / "audio" / "single"
    inp = td / "01_dictionnary" / "generated_dataset.jsonl"
    sent = generate(tts, rows, inp, audio, voice="fr_female")
    check("--voice X sends X for every row",
          set(sent.values()) == {"fr_female"}, str(set(sent.values())))
    got = manifest_voices(m99, inp, audio, td / "NeMO", "dictionary")
    check("a one-name roster labels every row with that name",
          set(got.values()) == {"fr_female"}, str(set(got.values())))

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rows = dictionary_rows()
    audio = td / "audio" / "server_default"
    inp = td / "01_dictionnary" / "generated_dataset.jsonl"
    sent = generate(tts, rows, inp, audio, voice_mode="single")
    check("--voice-mode single with no --voice sends no voice at all",
          set(sent.values()) == {None}, str(set(sent.values())))
    check("and records nothing, since the name is the server's secret",
          va.read_provenance(audio, inp) is None)
    got = manifest_voices(m99, inp, audio, td / "NeMO", "dictionary")
    check("stage 99 then writes no voice field rather than guessing",
          set(got.values()) == {None}, str(set(got.values())))
    # ... unless the operator supplies the name the files do not carry, which is how a
    # v1 rebuild gets a truthful column instead of an empty one.
    got = manifest_voices(m99, inp, audio, td / "NeMO_fallback", "dictionary",
                          voice_fallback="fr_female")
    check("--voice-fallback labels unrecorded audio",
          set(got.values()) == {"fr_female"}, str(set(got.values())))

# ------------------------------------- 5. stage 06 regenerates with the clip's voice
m06 = load_module(IMPROVE_PATH, "improve_under_test")
m06.logger.remove()
asked: list[str | None] = []


def fake_tts_synthesize(url, text, **kw):
    """Stand in for the retrying HTTP helper; record the voice it was asked for."""
    asked.append(kw.get("voice"))
    return FLAC_REDO


m06.tts_synthesize = fake_tts_synthesize


class _NullSem:
    """The TTS concurrency semaphore, which _synthesize_draft enters."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def improve_cfg(**over) -> SimpleNamespace:
    """A stage-06 cfg with the fields the TTS half reads."""
    cfg = dict(
        tts_url="http://unused", tts_format="flac", tts_voice="fr_female",
        tts_model=None, tts_speed=None, tts_timeout=30.0, tts_max_retries=1,
        tts_sem=_NullSem(), speed_jitter=0.0, start_seed=43, n_improv=1,
        cfg_alpha_start=None, cfg_alpha_step=0.0, max_audio_seconds=327.68,
        mode="tts", audio_key="audio_filepath", audio_root=None,
        cer_threshold=0.08, tail_cer_threshold=0.12,
        window_cer_threshold=0.15, bad_run_threshold=30, min_improvement=0.02,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


# The voice the row carries wins over --tts-voice ...
asked.clear()
m06._synthesize_draft(0, improve_cfg(), "clip", "texte", 1, "nothash", set(),
                      voice="h_frf_itm")
check("stage 06 regenerates with the row's voice", asked == ["h_frf_itm"], str(asked))

# ... and --tts-voice is what a row without the field falls back to (v1 audio).
asked.clear()
m06._synthesize_draft(0, improve_cfg(), "clip", "texte", 1, "nothash", set())
check("--tts-voice is the fallback, not the rule", asked == ["fr_female"], str(asked))

# The whole path, from the manifest row to the request: combined_worker in tts mode
# goes straight to generate_drafts, no STT server involved.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    clip = td / "000001_0000_terme.flac"
    clip.write_bytes(FLAC)
    rec = {
        "audio_filepath": clip.name,
        "asr_training_source": "Un texte a refaire.",
        "voice": "h_frm_ptf",
        "improvement": {"status": "pending_tts", "orig_cer": 0.4},
    }
    asked.clear()
    out = m06.combined_worker(0, rec, improve_cfg(), "whisper", td / "manifest.jsonl")
    check("the row's voice reaches the request through combined_worker",
          asked == ["h_frm_ptf"], f"{asked} action={out.get('action')}")

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
