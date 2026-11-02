#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Carry stage 06's quality-control results into the NeMo manifests.

Run AFTER an improvement run and BEFORE ``scripts/build_parquet.py``:

    uv run 03_sync_hotfix_results.py            # dry run, reports what it would change
    uv run 03_sync_hotfix_results.py --apply

Stage 06 regenerates bad clips by OVERWRITING the audio in place, so the clips on the
SSD are already the good ones, but the manifests built before that run still describe the
old audio: a regenerated draw is a different length (deltas up to ~115 s observed, and a
handful of clips cross the trainer's 45 s ``max_duration`` in one direction or the other).
Rebuilding the manifests instead of patching them is a second, slower route to the
same durations (``probe_durations`` validates its per-name cache against each clip's
size and mtime, so an overwritten clip is re-probed rather than served stale), but it
would not carry over the CER, the transcript or the ``qc_status`` below.

So this reads stage 06's scored output (which carries the transcript of the audio that
actually shipped) and, joining on the resolved absolute audio path, writes into every
manifest row:

    duration          the state row's, which is only re-probed on a REDRAW (see below)
    cer               CER of the shipped audio against its label
    cer_tail          CER of the last ~30 s, null on clips shorter than that
    stt_transcript    what the STT model heard, so every CER above is auditable
    n_stt_check       how many readings the clip needed (null unless it was rechecked)
    stt_model         which model produced them, spelled out rather than an API alias
    cfg_alpha         TTS guidance the shipped audio was drawn at
    voice             the voice that actually read the shipped audio (see below)
    regenerated       true if stage 06 replaced this clip
    qc_status         improved / exhausted / original (see below)

and, for each judge that read the corpus, one column group of its own, so a consumer
can see WHICH model disagreed rather than a single blended number:

    whisper_stt       what ggml-large-v3-turbo heard (the judge over all 602,789 clips)
    whisper_cer       its CER, whisper_cer_window its worst-window CER
    whisper_repeat_excess   null on EVERY row, kept only for column symmetry: the
                      whisper scorer in `06_hotfixes/01_compute_stt.py` deliberately
                      does not compute `repeat_excess` (see the comment there and the
                      decoder reason below), so there is no count to copy, not merely
                      a weak one. Measured over the v2 release: the field is on 100%
                      of the 604,337 parakeet readings and 0% of the whisper ones.
                      `whisper_v3_repeat_excess` is null for the same reason, so the
                      loop gate rides on `parakeet_repeat_excess` (>= 1 on 4,098
                      readings) plus the acoustic self-similarity scan, and on nothing
                      else.
    whisper_v3_*      the same four from non-turbo ggml-large-v3, the second opinion
                      run only over what turbo flagged, so null on most rows
    parakeet_*        the same four from parakeet-tdt-0.6b-v3-ultra, and the only
                      group whose repeat_excess is populated
    cer_max           the worst CER any judge gave the clip
    cer_window_max    the worst window CER any judge gave it

Two judges rather than one because they fail differently, and a dataset that ships only
the blend hides that. Whisper's decoder suppresses repeated n-grams, so a clip that says
the same clause three times comes back transcribed ONCE and scores clean: on 269 clips
the acoustic scan had flagged, whisper read CER 0.0167 where parakeet read 0.1796 on the
same audio, a 10.7x gap. Parakeet has the opposite bias: it is a general French model
with no exposure to this corpus's drug brand names, so it mishears clean audio at CER
0.08-0.14. Neither is the truth on its own, which is why both ship: `cer_max` is the
pessimistic read for filtering, and the per-judge columns are what let a user see why.

``qc_status`` is ``improved`` when the clip was flagged and a better draw replaced it,
``exhausted`` when it was flagged and no draw was good enough (the ORIGINAL audio ships,
so these are the known-weak clips), and ``original`` when it never tripped a gate.

``duration`` is the one column copied from a field stage 06 does NOT keep current, which
is why ``06_hotfixes/13_refresh_stale_durations.py --apply`` must run immediately before
this script, and why ``driver.sh`` now runs the pair under one step header. Only a redraw
writes a state row's duration (``01_recursive_improvement.py``); a clip the GENERATOR
regenerated keeps whatever length it had when the sweep first scored it. Measured on
2026-10-26: 208,235 of 604,337 state rows disagreed with the manifest, by up to 53 s. Run
after a manifest rebuild without the refresh, the copy below therefore replaces re-probed
truth with a v1-era number, and ``cer_tail``, which is null at or under 30 s, goes blind
on every clip whose real length crosses that window (5,250 of them). With the refresh
first, the copy is a no-op and this script keeps doing what it says.

``voice`` is the one column here that stage 06 can CHANGE rather than measure: a term clip
no draw of its own voice could fix is redrawn by another voice of the roster
(``01_recursive_improvement.py --voice-swap``), and ``01_build_nemo_manifest.py``
RE-DERIVES that column from the assignment hash, which cannot know about the swap. So
without this copy every manifest rebuild would relabel a swapped clip with the voice that
never read it. Unlike the QC fields it is written only when stage 06 has one: a missing
value means "stage 06 does not know" (a v1 row), not "no voice".

Idempotent: re-run it after a rescore or another improvement run and it simply re-copies
the current values. It refuses to run while a stage-06 pass is alive, since that pass
rewrites its own output file wholesale at every flush.

This file was written with Claude Code.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from hotfix_states import STATE_AUDIO_BASES  # noqa: E402
from nemo_manifest import MANIFEST_GLOBS, read_jsonl, resolve_audio, write_jsonl  # noqa: E402

# The STT side of the corpus, named as it actually is rather than as the OpenAI-compatible
# alias the client sends (`whisper-1`). Confirmed against the serving endpoint itself,
# which reports `{"id": "/cache/ggml-large-v3-turbo.bin", "backend": "whisper"}`, and
# against the CrispASR compose that starts it (`--hf-repo
# ggerganov/whisper.cpp:ggml-large-v3-turbo.bin`).
STT_MODEL = "whisper.cpp/ggml-large-v3-turbo"
# Key under which stage 06 stored that model's readings. NOT the same thing as STT_MODEL
# above: the pass stores a reading under `transcriptions[<its --stt-model> or "default"]`,
# and v1 ran it with the `whisper-1` alias, hence this default. The v2 driver passes
# $STT_MODEL instead, so the QC phase's two judges (turbo over everything, then large-v3
# over what turbo flagged) land in different keys on one row rather than overwriting each
# other. Use --stt-key to read the other judge.
STT_KEY = "whisper-1"

# Stage 06's scored outputs. Each was produced from one manifest, so its rows' relative
# audio paths resolve against that manifest's directory (the pairing is
# hotfix_states.STATE_AUDIO_BASES), spelled relative to this directory.
HOTFIX_SOURCES = tuple((f"../06_hotfixes/{state}", base)
                       for state, base in STATE_AUDIO_BASES.items())

# Every judge whose reading reaches a released row, and the column prefix it gets.
# Adding a judge is one line here plus its sweep; nothing else in this file knows a
# model name. The keys are `transcriptions` keys, which `12_merge_judge_sidecars.py`
# normalizes (stage 05's sidecars spell the same two fields `window_cer` / `tail_cer`),
# so every judge's reading uses the stage-06 names by the time it arrives here.
#
# `whisper-1` is deliberately NOT here. It is the v1-era OpenAI alias the client used
# before the real model name was recorded, and its readings describe v1 AUDIO: shipping
# them as a judge column would put a stale number next to a v2 clip with nothing on the
# row to say so. It still exists on old rows, hence `--stt-key`'s default.
JUDGES = (
    ("whisper.cpp/ggml-large-v3-turbo", "whisper"),
    ("whisper.cpp/ggml-large-v3", "whisper_v3"),
    ("onnx-asr/parakeet-tdt-0.6b-v3-ultra", "parakeet"),
)
# reading key -> column suffix. `text` is renamed because `whisper_text` would read like
# a label; everything else keeps its name so a column and the reading it came from are
# searchable by the same word.
JUDGE_FIELDS = (("text", "stt"), ("cer", "cer"), ("cer_window", "cer_window"),
                ("repeat_excess", "repeat_excess"))
JUDGE_COLUMNS = tuple(f"{prefix}_{col}" for _, prefix in JUDGES for _, col in JUDGE_FIELDS)

QC_FIELDS = (("cer", "cer_tail", "stt_transcript", "n_stt_check", "stt_model",
              "cfg_alpha", "regenerated", "qc_status")
             + JUDGE_COLUMNS + ("cer_max", "cer_window_max"))


def judge_columns(all_readings: dict) -> dict:
    """One flat column per judge per field, plus the across-judge maxima.

    Flat rather than a nested `{"whisper": {...}}` struct because this lands in a
    released parquet: a struct whose keys vary row to row makes the schema depend on
    which rows the writer happened to see first, and the non-turbo second opinion ran
    only over the flagged slice, so its keys are absent on most rows. Flat columns are
    declared for every row (null where a judge did not read the clip), which is also
    what makes `cer_max` safe to compute downstream.

    The maxima ignore nulls and are None when no judge scored the clip, never 0.0: a
    0.0 would read as "every judge agreed it is perfect" on a clip nobody judged.
    """
    out: dict = {}
    cers: list[float] = []
    windows: list[float] = []
    for key, prefix in JUDGES:
        reading = all_readings.get(key) or {}
        for field, col in JUDGE_FIELDS:
            value = reading.get(field)
            if field == "text":
                value = (value or "").strip() or None
            out[f"{prefix}_{col}"] = value
        if isinstance(reading.get("cer"), (int, float)):
            cers.append(float(reading["cer"]))
        if isinstance(reading.get("cer_window"), (int, float)):
            windows.append(float(reading["cer_window"]))
    out["cer_max"] = max(cers) if cers else None
    out["cer_window_max"] = max(windows) if windows else None
    return out


PASS_SCRIPT = "01_recursive_improvement.py"


def is_pass_cmdline(cmdline: str) -> bool:
    """True when this /proc cmdline IS the stage-06 pass, not merely a mention of it.

    /proc cmdlines are NUL-separated, so the pass itself always has one ARGUMENT whose
    basename is the script (`uv run .../01_recursive_improvement.py --mode stt`). Anything
    that just names it carries the name inside a longer argument instead: a shell running
    `-c '... 01_recursive_improvement.py ...'`, a `grep`, or a wait loop polling
    `pgrep -f "01_recursive_improvement.py --mode stt"`. Substring matching counted all of
    those as a live pass, and a wait loop matches its OWN pattern, so one stale loop
    blocked this script indefinitely with no pass running anywhere.
    """
    return any(os.path.basename(arg) == PASS_SCRIPT for arg in cmdline.split("\0"))


def running_passes() -> list[int]:
    """PIDs of any live stage-06 pass, which would overwrite what we read."""
    me = os.getpid()
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if is_pass_cmdline(cmdline):
            pids.append(int(entry.name))
    return sorted(pids)


def merge_hotfix(prior: dict, fresh: dict) -> dict:
    """One clip described by two of `HOTFIX_SOURCES`, merged rather than overwritten.

    The PARROT clips sit in both sources: the root run reads them through
    `data/NeMO_files/PARROT/<clip>` and the PARROT-only run through its own manifest, so
    the two resolve to the SAME absolute path and collide in the index. The later source
    used to win outright, and it is the one that knows less: its state file dropped the
    stored state of 908 of its 1548 rows when their text changed (the pass logs "manifest
    rebuilt?" and starts those clips over), so 126 clips whose audio the root run had
    regenerated shipped as `regenerated: false` / `qc_status: "original"`. Their CER
    columns were right, which is exactly what kept it quiet: the numbers described the
    improved audio while the provenance denied the improvement had happened.

    The rule, field by field:

    * measurements (duration, cer, stt_transcript, voice, cfg_alpha, ...) take the later
      source's value when it HAS one, since that source read the audio now on disk, and
      fall back to the earlier source's when it is null;
    * ``regenerated`` is the OR of the two: the audio was replaced if either run replaced
      it, and neither run can un-replace it;
    * ``qc_status`` takes the later source's only when that source carries a marker at
      all. A pass never DROPS a marker on a clip it re-scores clean, so an absent marker
      here means "this source knows nothing about the clip's history", not "this clip was
      never flagged". A marker that IS there is the fresher judgement and wins, which is
      what lets a clip the root run improved be labelled `exhausted` if a later run
      re-judged the shipped draw and no redraw could save it.
    """
    merged = dict(prior)
    for key, value in fresh.items():
        if value is not None:
            merged[key] = value
    merged["regenerated"] = bool(prior.get("regenerated")) or bool(fresh.get("regenerated"))
    if fresh["qc_status"] == "original" and prior["qc_status"] != "original":
        merged["qc_status"] = prior["qc_status"]
    return merged


def load_hotfix(root: Path, stt_model: str, stt_key: str = STT_KEY) -> dict[str, dict]:
    """Everything stage 06 knows, keyed by absolute audio path.

    ``stt_key`` selects WHICH judge's reading to carry over when a row holds several;
    ``stt_model`` is only the name written into the manifest.

    A clip described by more than one source goes through `merge_hotfix` instead of
    being overwritten by whichever source happens to be read last.
    """
    index: dict[str, dict] = {}
    for rel_source, rel_base in HOTFIX_SOURCES:
        source = (root / rel_source).resolve()
        if not source.is_file():
            logger.warning(f"{source} not found, skipping (was that dataset never run?)")
            continue
        base = root / rel_base
        n = pending = read = overlap = 0
        seen_keys: set[str] = set()
        for row in read_jsonl(source):
            all_readings = row.get("transcriptions") or {}
            seen_keys.update(all_readings)
            reading = all_readings.get(stt_key) or {}
            read += bool(reading)
            imp = row.get("improvement") or {}
            status = imp.get("status")
            entry = {
                "duration": row.get("duration"),
                "cer": reading.get("cer"),
                "cer_tail": reading.get("cer_tail"),
                "stt_transcript": (reading.get("text") or "").strip() or None,
                "n_stt_check": reading.get("n_stt_check"),
                "stt_model": stt_model,
                "cfg_alpha": row.get("cfg_alpha"),
                "voice": row.get("voice"),
                "regenerated": status == "improved",
                # pending_* means a run was interrupted mid-flight: the clip is whatever
                # is on disk right now, which is the original, so it ships as one.
                "qc_status": status if status in ("improved", "exhausted") else "original",
                # Per-judge columns, from the SAME row: every judge that read this clip
                # wrote into the one `transcriptions` dict, so no second join is needed.
                **judge_columns(all_readings),
            }
            key = resolve_audio(base, row["audio_filepath"])
            prior = index.get(key)
            index[key] = entry if prior is None else merge_hotfix(prior, entry)
            overlap += prior is not None
            n += 1
            if status in ("pending_tts", "pending_stt"):
                pending += 1
        logger.info(f"{source.name}: {n} scored rows, {read} with a reading under "
                    f"'{stt_key}' (base {rel_base})")
        if overlap:
            # Not a warning: the overlap is by design (the root set covers every clip,
            # the per-dataset sets re-run a subset), and `merge_hotfix` is what keeps the
            # one that knows less from erasing the one that knows more.
            logger.info(f"{source.name}: {overlap} clip(s) were already described by an "
                        f"earlier source and were merged, not overwritten")
        if n and not read:
            # Every QC column would be written as null, which reads exactly like "this
            # clip was never judged". Say so instead, and name the keys that ARE there.
            logger.warning(
                f"{source.name}: no row carries a reading under '{stt_key}'. The keys "
                f"present are {sorted(seen_keys) or ['none']}: pass --stt-key with one "
                f"of those, or the QC columns all get written empty."
            )
        if pending:
            # These ship their ORIGINAL audio and are labelled `original`, which is true
            # of the audio but hides that the clip is known-weak: stage 06 flagged it and
            # was interrupted before resolving it. Finish the improvement run first, or
            # the release calls a bad clip a clean one.
            logger.warning(
                f"{source.name}: {pending} row(s) are still mid-flight (pending_tts / "
                f"pending_stt) and would ship as `original`. Finish the stage-06 run "
                f"before releasing, or they lose their `exhausted` label."
            )
    return index


def sync_manifest(path: Path, hotfix: dict[str, dict], apply: bool) -> tuple[int, int, int, int, int]:
    """Returns (rows, matched, duration changes, voice changes, field changes) for one manifest."""
    rows = read_jsonl(path)
    matched = redur = revoice = changed = 0
    for row in rows:
        rel = row.get("audio_filepath")
        if rel is None:
            continue
        qc = hotfix.get(resolve_audio(path.parent, rel))
        if qc is None:
            continue
        matched += 1
        touched = False
        if qc["duration"] is not None and abs((row.get("duration") or 0.0) - qc["duration"]) > 5e-4:
            row["duration"] = qc["duration"]
            redur += 1
            touched = True
        if qc["voice"] and row.get("voice") != qc["voice"]:
            # Conditional, unlike QC_FIELDS: a row stage 06 has no voice for keeps the one
            # the manifest derived, instead of being blanked. See the module docstring.
            row["voice"] = qc["voice"]
            revoice += 1
            touched = True
        for key in QC_FIELDS:
            # Written even when the value is null (a clip too short for a tail score, a
            # clip never rechecked), so every row declares the same keys and the parquet
            # schema is not inferred from whichever rows happen to have them.
            if key not in row or row[key] != qc[key]:
                row[key] = qc[key]
                touched = True
        changed += int(touched)
    if apply and changed:
        write_jsonl(path, rows)
    return len(rows), matched, redur, revoice, changed


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--stt-model", default=STT_MODEL, show_default=True,
              help="Model name written to every row. The default is what the serving "
                   "endpoint reports, not the `whisper-1` alias the client sends.")
@click.option("--stt-key", default=STT_KEY, show_default=True,
              help="Which judge's reading to carry over, i.e. the `transcriptions` key "
                   "stage 06 stored it under, which is the --stt-model that pass ran "
                   "with. driver.sh passes $STT_MODEL.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
def main(manifests: tuple[Path, ...], root: str, stt_model: str, stt_key: str,
         apply: bool) -> None:
    """Refresh manifest durations and attach stage 06's QC columns."""
    root_path = Path(root).resolve()
    if apply:
        pids = running_passes()
        if pids:
            raise SystemExit(
                f"a stage-06 pass is running (pid {pids}); let it finish first, or it "
                "will keep writing the very file this reads"
            )

    hotfix = load_hotfix(root_path, stt_model, stt_key)
    if not hotfix:
        raise SystemExit("no stage-06 output found, nothing to sync")
    logger.info(f"{len(hotfix)} scored clips, {sum(1 for q in hotfix.values() if q['regenerated'])} "
                f"regenerated, {sum(1 for q in hotfix.values() if q['qc_status'] == 'exhausted')} exhausted")

    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    total_rows = total_matched = total_changed = 0
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        rows, matched, redur, revoice, changed = sync_manifest(path, hotfix, apply)
        total_rows += rows
        total_matched += matched
        total_changed += changed
        gap = rows - matched
        logger.info(
            f"{path.relative_to(root_path)}: {rows} rows, {matched} matched, "
            f"{redur} duration(s) refreshed, "
            + (f"{revoice} voice(s) swapped, " if revoice else "")
            + f"{changed} row(s) updated"
            + (f", {gap} WITHOUT a stage-06 score" if gap else "")
        )
    logger.info(f"total: {total_matched}/{total_rows} rows matched, {total_changed} updated")
    if not apply:
        logger.warning("dry run, nothing written (pass --apply)")


if __name__ == "__main__":
    main()
