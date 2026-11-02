#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "jiwer>=3.0", "requests>=2.31", "tqdm>=4.66"]
#   # jiwer/requests/tqdm are not used here: they are what importing stage 06's
#   # scorer costs, and the scorer owns `repeat_excess` (see `_load_stt`).
# ///
"""Carry a stage-05 QC sidecar's readings into stage 06's state as another judge.

Stage 06 already stores EVERY judge's reading on the clip's own row, keyed by model:

    row["transcriptions"]["whisper.cpp/ggml-large-v3-turbo"] = {"cer": 0.01, ...}
    row["transcriptions"]["whisper.cpp/ggml-large-v3"]       = {"cer": 0.09, ...}

so a new judge needs no new column anywhere: it needs its readings written into that
dict. Stage 05's judge (`05_generate_audio/03_qc_cpu_stt.py`) instead writes per-stage
sidecars beside the audio, keyed by stem, because it is built to follow a generation run
clip by clip. This script is the join between the two, and nothing else: it changes no
verdict and recomputes no score.

Why it exists at all. Whisper's decoder suppresses repeated n-grams, so a clip that says
the same clause three times comes back transcribed ONCE and scores clean. Measured on
269 clips the acoustic scan had flagged: whisper read them at CER 0.0167 while parakeet
read the same audio at 0.1796, a 10.7x gap, and whisper's transcript was SHORTER than
the reference (the collapsing signature). Every CER gate in this repo is therefore blind
to loops as long as whisper is the only judge, which is why a second, non-collapsing
judge has to reach the row.

Read the result as a loop detector, not as a second quality gate. Parakeet is a general
model and this corpus is full of French drug brand names, so it mishears them at a rate
no TTS defect explains: `zymad` as "Zimade", `xelevia` as "Zolevia", `bactrim forte` as
"bactrium fort". Those land as `wrong_words` with CER 0.08-0.14 and are not defects.
The shape worth acting on is `loop`.

    uv run 12_merge_judge_sidecars.py                     # dry run, reports what it would do
    uv run 12_merge_judge_sidecars.py --apply
    uv run 12_merge_judge_sidecars.py --out-name qc_acoustic_sidecar.jsonl --judge x  # any sidecar

Idempotent: re-running it re-copies the current values. A sidecar row whose clip has been
redrawn since it was written is SKIPPED rather than merged, via the shared staleness rule
in utils/clip_freshness.py, because a reading only ever describes the take it heard.

This file was written by Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "utils"))

from clip_freshness import is_stale  # noqa: E402

_STT = None


def _load_stt():
    """Stage 06's scorer, loaded on first use.

    Imported rather than reimplemented because `repeat_excess` has to see exactly the
    text the gates see: it compares word windows AFTER `normalize_for_scoring`, so a
    second spelling of that normalizer here would count windows the pass does not. The
    import is lazy because it costs jiwer and a few hundred regexes, and a `--dry-run`
    that only wants the match counts should not pay for them.
    """
    global _STT
    if _STT is None:
        import importlib.util

        path = ROOT / "06_hotfixes" / "01_compute_stt.py"
        spec = importlib.util.spec_from_file_location("stt_for_merge", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _STT = mod
    return _STT

#: What stage 05's judge calls its output. Overridable so this script can also merge the
#: acoustic sidecar, or a sweep parked under its own name.
DEFAULT_OUT_NAME = "qc_parakeet_ultra.jsonl"

#: The model name written into `transcriptions`, spelled out rather than as a path or an
#: API alias, matching `whisper.cpp/ggml-large-v3-turbo`. The sweep runs the unsuffixed
#: (fp32) ONNX export: the fp16 folder of that repo ships an encoder with no decoder, so
#: fp32 is the only pair that resolves.
DEFAULT_JUDGE = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"

#: Stage 06 state files, same pairing `99_hf_release/03_sync_hotfix_results.py` uses.
from hotfix_states import DEFAULT_STATES  # noqa: E402

#: sidecar field -> reading field. The two sides spell the same four metrics
#: differently, which is the whole reason this mapping is explicit and in one place.
FIELD_MAP = {
    "transcript": "text",
    "cer": "cer",
    "wer": "wer",
    "tail_cer": "cer_tail",
    "window_cer": "cer_window",
    "bad_run_chars": "bad_run_chars",
}


def load_stage_dirs() -> dict[str, Path]:
    """Stage name -> the resolved directory its clips and sidecars live in.

    Discovered from the filesystem rather than from stage 05's STAGES table, which maps
    a stage to its VOICE MODE: a different fact, and executing that module to read it
    would drag numpy and soundfile into a script that only joins two JSONL files. What
    is needed here is just "which stage directories exist", and `<stage>/generated_audios`
    being present is the definition of that. The symlink is resolved because it points at
    the external SSD, which is where the sidecars actually are.
    """
    dirs: dict[str, Path] = {}
    for audio in sorted(ROOT.glob("*/generated_audios")):
        if audio.exists():
            dirs[audio.parent.name] = audio.resolve()
    return dirs


def to_reading(row: dict, *, n_readings: int) -> dict:
    """One sidecar row as a `transcriptions` reading.

    `stt_rtf` and `repeat_excess` are derived rather than copied. The first only when the
    duration is usable: a zero would divide, and a missing one would make an invented
    ratio look measured. The second because it is the reading that justifies this judge
    existing (see the module docstring) and the sweep's sidecars predate it.
    """
    reading: dict[str, object] = {}
    for src, dst in FIELD_MAP.items():
        value = row.get(src)
        if src == "transcript":
            value = (value or "").strip() or None
        if value is not None:
            reading[dst] = value
    # The loop count is computed here rather than copied, because the sweep that wrote
    # these rows predates the metric and its sidecars carry no such field. Computing it
    # at merge time also keeps one rule for every row, old and new: a row swept in
    # October and a row swept after a repair round get the same number from the same
    # function, which is the only way the two are comparable.
    reference, transcript = row.get("reference"), row.get("transcript")
    if reference and transcript:
        reading["repeat_excess"] = _load_stt().repeat_excess(reference, transcript)
    seconds = row.get("stt_s")
    duration = row.get("duration")
    if seconds is not None:
        reading["stt_seconds"] = seconds
        if isinstance(duration, (int, float)) and duration > 0:
            reading["stt_rtf"] = round(float(seconds) / float(duration), 5)
    if n_readings > 1:
        reading["n_stt_check"] = n_readings
    return reading


def load_sidecars(stage_dirs: dict[str, Path], out_name: str, *, ts_field: str,
                  check_freshness: bool) -> tuple[dict[tuple[str, str], dict], Counter]:
    """Every sidecar's readings, keyed by (audio directory name, stem).

    Keyed on the DIRECTORY rather than the stage name because that is what a state row
    can tell us about itself: its `audio_filepath` is `../PARHAF/<stem>.flac`, so the
    parent directory is the join key whichever state file the row came from.

    A sidecar is append-only, so a stem can appear several times: a later row is a later
    reading of a later take, and the last one wins. The count of rows per stem is kept as
    `n_stt_check`, which is what the whisper readings mean by it.
    """
    index: dict[tuple[str, str], dict] = {}
    stats = Counter()
    for stage, audio in sorted(stage_dirs.items()):
        path = audio / out_name
        if not path.is_file():
            logger.info(f"{stage}: no {out_name} in {audio}, nothing to merge")
            continue
        rows_here = seen = stale = 0
        per_stem: dict[str, int] = {}
        for raw in path.open(encoding="utf-8"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                stats["torn_line"] += 1      # a killed run's final, half-written line
                continue
            stem = row.get("stem")
            if not stem:
                continue
            rows_here += 1
            if check_freshness and is_stale(audio / f"{stem}.flac", row.get(ts_field)):
                # The clip was redrawn after this reading: it describes audio nobody
                # ships any more. Merging it would re-certify the OLD take.
                stale += 1
                continue
            per_stem[stem] = per_stem.get(stem, 0) + 1
            index[(audio.name, stem)] = to_reading(row, n_readings=per_stem[stem])
        seen = len(per_stem)
        stats["rows"] += rows_here
        stats["stems"] += seen
        stats["stale"] += stale
        logger.info(f"{stage}: {rows_here} sidecar row(s) -> {seen} stem(s)"
                    + (f", {stale} skipped as stale" if stale else ""))
    return index, stats


def merge_state(path: Path, index: dict[tuple[str, str], dict], judge: str,
                apply: bool, matched: set[tuple[str, str]] | None = None) -> Counter:
    """Write the judge's readings into one state file, streaming.

    Streamed line by line rather than loaded: this file is 909 MB and holding 602k
    parsed rows would cost several GB for a join that needs one row at a time. The
    rewrite goes to a temporary file in the same directory and is renamed over the
    original, so an interrupted run leaves the state intact rather than truncated.
    """
    stats = Counter()
    tmp = path.with_suffix(path.suffix + ".merge.tmp")
    out = tmp.open("w", encoding="utf-8") if apply else None
    try:
        for raw in path.open(encoding="utf-8"):
            if not raw.strip():
                continue
            row = json.loads(raw)
            stats["rows"] += 1
            audio_path = Path(row.get("audio_filepath") or "")
            key = (audio_path.parent.name, audio_path.stem)
            reading = index.get(key)
            if reading is not None:
                readings = row.setdefault("transcriptions", {})
                stats["replaced" if judge in readings else "added"] += 1
                readings[judge] = reading
                stats["matched"] += 1
                if matched is not None:
                    matched.add(key)
            if out is not None:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        if out is not None:
            out.close()
            tmp.unlink(missing_ok=True)
        raise
    if out is not None:
        out.flush()
        os.fsync(out.fileno())
        out.close()
        tmp.replace(path)
    return stats


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--state", "states", multiple=True, type=click.Path(path_type=Path),
              help=f"Stage 06 state file, repeatable (default: {', '.join(DEFAULT_STATES)}).")
@click.option("--out-name", default=DEFAULT_OUT_NAME, show_default=True,
              help="Sidecar filename to read from each stage's audio directory.")
@click.option("--judge", default=DEFAULT_JUDGE, show_default=True,
              help="Key to write the readings under in `transcriptions`.")
@click.option("--ts-field", default="ts", show_default=True,
              help="Sidecar field holding when the row was written, for the staleness "
                   "check. The STT sidecars stamp an ISO-8601 `ts`, the acoustic one a "
                   "float `mtime`; both parse.")
@click.option("--no-freshness-check", is_flag=True,
              help="Merge every row, even one whose clip was redrawn after it was read. "
                   "Only for a sidecar that predates the `ts` field.")
@click.option("--apply", is_flag=True, help="Write. Without it, report only.")
def main(states: tuple[Path, ...], out_name: str, judge: str, ts_field: str,
         no_freshness_check: bool, apply: bool) -> None:
    """Merge stage-05 QC sidecars into stage 06's state as another judge's readings."""
    stage_dirs = load_stage_dirs()
    if not stage_dirs:
        raise click.ClickException("no stage directory found, is the SSD mounted?")
    index, side = load_sidecars(stage_dirs, out_name, ts_field=ts_field,
                                check_freshness=not no_freshness_check)
    if not index:
        raise click.ClickException(
            f"no sidecar row to merge: looked for {out_name} in "
            f"{', '.join(str(p) for p in stage_dirs.values())}")
    logger.info(f"{len(index)} stem(s) to merge under '{judge}' "
                f"from {side['rows']} sidecar row(s)"
                + (f", {side['stale']} stale row(s) skipped" if side["stale"] else "")
                + (f", {side['torn_line']} torn line(s)" if side["torn_line"] else ""))

    paths = states or tuple(_HERE / rel for rel in DEFAULT_STATES)
    total = Counter()
    # Distinct stems, not a sum of per-file counts: a PARROT clip is described by
    # BOTH state files, so summing would report more matches than there are stems
    # and could hide a genuinely unmatched set behind a negative remainder.
    matched: set[tuple[str, str]] = set()
    for path in paths:
        path = Path(path)
        if not path.is_file():
            logger.warning(f"{path} not found, skipping")
            continue
        stats = merge_state(path, index, judge, apply, matched)
        total.update(stats)
        logger.info(f"{path}: {stats['rows']} row(s), {stats['matched']} matched "
                    f"({stats['added']} new, {stats['replaced']} replaced)")
    unmatched = len(index) - len(matched)
    if unmatched > 0:
        # Not fatal: a sweep can cover clips a state file does not describe (a draft
        # pruned since, a stage whose state lives elsewhere). Said out loud because the
        # other reading of it is a broken join key, which would look identical.
        logger.warning(f"{unmatched} swept stem(s) matched no state row")
    if not apply:
        logger.warning("dry run: nothing written, pass --apply")


if __name__ == "__main__":
    main()
