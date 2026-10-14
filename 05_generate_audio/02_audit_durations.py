#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "soundfile", "numpy", "tqdm"]
# ///
"""Re-check the duration of clips ALREADY on disk, and optionally delete the bad ones.

Why this exists: `01_generate_audio.py` only grew its per-clip frame budget and its
too-short floor on 2026-10-14, after ~29k v2 clips had already been generated without
them. Those clips were accepted whatever their duration, so a share of them hold a
runaway take (the model looping syllables for minutes, which lands in the dataset
under a short label) or a truncated one (part of the text never spoken). This script
applies the SAME bounds as the generator, from the same `utils/frame_budget.py`, to
clips that are already written.

The repair is deliberately "delete, then resume" rather than "regenerate here": the
generator's resume already skips every clip whose stem exists and regenerates every
stem that does not, so removing a bad clip is the whole fix and this script needs no
TTS client, no server and no voice logic of its own. Run it, then re-run stage 05.

The deleted clip's row in `cfg_alpha_sidecar.jsonl` (the per-clip cfg_alpha stage 05
records for a clip it had to draw below the run's value) is left alone on purpose:
nothing here can know what the replacement will be drawn at, and stage 05 appends a
fresh row for any stem that already has one, whatever rung it lands on. Since the
readers take the LAST row for a stem, the regenerated clip's value always wins.

    uv run 05_generate_audio/02_audit_durations.py                    # dry run, all stages
    uv run 05_generate_audio/02_audit_durations.py --stage 01_dictionnary
    uv run 05_generate_audio/02_audit_durations.py --apply            # actually delete
    uv run 05_generate_audio/02_audit_durations.py --report bad.jsonl # keep the list

Written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from frame_budget import (  # noqa: E402
    BUDGET_FIXED_S,
    budget_seconds,
    classify_duration,
    floor_seconds,
    rates_for_voice,
)
from voice_assignment import VOICES_FR_V2, group_field, voice_for_row  # noqa: E402

#: Stage directory -> voice-assignment mode, the same pairing `generate_v2.sh` uses.
#: The voice is needed to judge a clip at all: the bounds are PER VOICE, since the ten
#: v2 voices read at medians 10.7 to 16.4 chars/s (see utils/frame_budget.py). Getting
#: the mode wrong here would judge clips against another voice's bounds, which is why
#: this table mirrors the runner's rather than being a flag.
STAGES: dict[str, str] = {
    "01_dictionnary": "row",
    "02_drugs": "row",
    "03_PARHAF": "group",
    "04_PARROT": "group",
    "07_acronyms": "row",
}


def _load_generator():
    """Import `01_generate_audio.py` by path to borrow `normalize_term`.

    The module name starts with a digit, so a plain import is impossible; this is the
    same spec_from_file_location trick the tests use. Borrowing rather than copying
    matters more here than anywhere else in the repo: the slug rule IS the join key
    between a row and its file, so a copy that drifts by one character would make
    every clip look missing and this script would report a clean corpus.
    """
    # NB this pulls in the generator's own imports (tqdm, soundfile), which is why
    # they appear in this script's PEP 723 dependency list although nothing here uses
    # them directly.
    path = Path(__file__).resolve().parent / "01_generate_audio.py"
    spec = importlib.util.spec_from_file_location("stage05_generator", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def probe(path: Path) -> float | None:
    """Clip duration in seconds from the container header, None if unreadable."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:  # noqa: BLE001 - an unreadable clip is reported, not raised
        return None


def audit_stage(stage: Path, mode: str, normalize_term, *, fixed: float,
                slow: float | None, fast: float | None, workers: int) -> list[dict]:
    """Join every row of one stage to its clip on disk and classify the durations.

    Returns
    -------
    list of dict
        One entry per clip that is NOT ok, each carrying the verdict, the path, the
        duration, the two bounds and the voice, ready to be reported or deleted.
    """
    jsonl = stage / "generated_dataset.jsonl"
    audio = stage / "generated_audios"
    if not jsonl.exists():
        logger.warning("{}: no generated_dataset.jsonl, skipped", stage.name)
        return []
    if not audio.exists():
        logger.warning("{}: no generated_audios, skipped", stage.name)
        return []
    # resolve() because generated_audios is a symlink to the SSD, and iterdir() on the
    # symlink itself is what makes `ls | wc -l` report 1 elsewhere in this repo.
    on_disk = {p.stem: p for p in Path(audio.resolve()).iterdir() if p.is_file()
               and not p.name.endswith(".tmp")}

    rows: list[tuple[str, str, Path]] = []  # (voice, text, path)
    field: str | None = None
    missing = 0
    for line in jsonl.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(k not in row for k in ("term_index", "variant_index", "term",
                                      "asr_training_source")):
            continue
        if field is None:
            field = group_field(row)
        stem = (f"{int(row['term_index']):06d}_{int(row['variant_index']):04d}_"
                f"{normalize_term(row['term'])}")
        path = on_disk.get(stem)
        if path is None:
            missing += 1
            continue
        voice = voice_for_row(row, field=field, mode=mode, voices=VOICES_FR_V2)
        rows.append((voice, row["asr_training_source"], path))

    logger.info("{}: {} clips on disk joined to their text, {} not generated yet",
                stage.name, len(rows), missing)
    if not rows:
        return []

    # Threads because this is pure I/O: sf.info reads a header, so 600k files are
    # minutes rather than hours, and the GIL is released during the read.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        durations = list(pool.map(lambda r: probe(r[2]), rows))

    verdicts: Counter = Counter()
    per_voice: defaultdict[str, Counter] = defaultdict(Counter)
    rates: defaultdict[str, list[float]] = defaultdict(list)
    bad: list[dict] = []
    for (voice, text, path), dur in zip(rows, durations):
        if dur is None:
            verdicts["unreadable"] += 1
            per_voice[voice]["unreadable"] += 1
            # Unreadable is reported but NOT deleted: a container this script cannot
            # parse may still be a perfectly good clip (libsndfile build differences),
            # and deleting it would cost a regeneration for nothing.
            continue
        # Per voice, and resolved exactly as 01_generate_audio.py resolves it, so a
        # clip this script deletes is exactly a clip the generator would have refused
        # to write (and vice versa): slow/fast are None unless the user overrode them
        # on the command line, in which case the override wins for every voice.
        v_slow, v_fast = rates_for_voice(voice, slow=slow, fast=fast)
        verdict = classify_duration(text, dur, fixed=fixed,
                                    slow_chars_per_second=v_slow,
                                    fast_chars_per_second=v_fast)
        verdicts[verdict] += 1
        per_voice[voice][verdict] += 1
        rates[voice].append(len(text) / dur)
        if verdict != "ok":
            bad.append({
                "stage": stage.name, "path": str(path), "voice": voice,
                "verdict": verdict, "duration": round(dur, 2), "chars": len(text),
                "chars_per_second": round(len(text) / dur, 2),
                "slow_chars_per_second": v_slow,
                "fast_chars_per_second": v_fast,
                "budget_s": round(budget_seconds(text, fixed, v_slow), 2),
                "floor_s": round(floor_seconds(text, v_fast), 2),
                "text": text[:200],
            })

    total = sum(verdicts.values())
    logger.info("{}: {} ok, {} over budget ({:.3f}%), {} implausibly short ({:.3f}%), "
                "{} unreadable",
                stage.name, verdicts["ok"], verdicts["over"],
                100.0 * verdicts["over"] / max(1, total), verdicts["short"],
                100.0 * verdicts["short"] / max(1, total), verdicts["unreadable"])

    # Per voice, sorted slowest median first: that is the order in which a voice
    # becomes suspect, and the column that explains an outlier rejection rate.
    logger.info("{}: per voice (median chars/s, the bounds it was judged by, "
                "over, short, n)", stage.name)
    for voice in sorted(rates, key=lambda v: float(np.median(rates[v]))):
        c = per_voice[voice]
        v_slow, v_fast = rates_for_voice(voice, slow=slow, fast=fast)
        logger.info("  {:12s} {:5.2f} chars/s  bounds {:4.1f}-{:4.1f}  "
                    "over {:4d} ({:5.2f}%)  short {:3d}  n={}",
                    voice, float(np.median(rates[voice])), v_slow, v_fast, c["over"],
                    100.0 * c["over"] / max(1, sum(c.values())), c["short"],
                    sum(c.values()))

    # The worst offenders by how far out of bounds they are, because a list of
    # filenames says nothing: the ratio is what distinguishes a loop from a long take.
    for verdict, label in (("over", "worst runaways"), ("short", "worst truncations")):
        worst = sorted((b for b in bad if b["verdict"] == verdict),
                       key=lambda b: (b["duration"] / b["budget_s"] if verdict == "over"
                                      else -b["duration"] / max(b["floor_s"], 1e-9)),
                       reverse=True)[:5]
        for b in worst:
            logger.info("  {}: {} {:.1f}s for {} chars ({:.2f} chars/s, bound {:.1f}s): {!r}",
                        label, Path(b["path"]).name, b["duration"], b["chars"],
                        b["chars_per_second"],
                        b["budget_s"] if verdict == "over" else b["floor_s"],
                        b["text"][:60])
    return bad


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stage", "stages", multiple=True, type=str,
              help="Stage directory name to audit, repeatable (default: all five, "
                   f"{', '.join(STAGES)}).")
@click.option("--apply", "apply_", is_flag=True,
              help="Actually DELETE the offending clips. Without it this is a dry run "
                   "that only reports. Deleting is the repair: stage 05's resume "
                   "regenerates exactly the stems that are missing.")
@click.option("--report", type=click.Path(dir_okay=False, path_type=Path),
              help="Write one JSON line per offending clip here (verdict, duration, "
                   "bounds, voice, text), so the list survives the deletion.")
@click.option("--budget-fixed-seconds", default=BUDGET_FIXED_S, show_default=True,
              help="Head/tail room term of the upper bound.")
@click.option("--budget-chars-per-second", default=None, type=float,
              help="Override the slowest plausible reading speed for EVERY voice. "
                   "Unset (the default) uses each voice's own measured rate from "
                   "utils/frame_budget.py, which is what the generator does.")
@click.option("--floor-chars-per-second", default=None, type=float,
              help="Override the fastest plausible reading speed for every voice; a "
                   "clip shorter than chars/this did not speak all of its text. Unset "
                   "uses the per-voice table, 0 disables the floor.")
@click.option("--workers", default=16, show_default=True,
              help="Threads probing audio headers (pure I/O).")
def main(stages: tuple[str, ...], apply_: bool, report: Path | None,
         budget_fixed_seconds: float, budget_chars_per_second: float,
         floor_chars_per_second: float, workers: int) -> None:
    """Audit, and optionally delete, clips whose duration does not match their text."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    names = stages or tuple(STAGES)
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise click.BadParameter(f"unknown stage(s) {unknown}, known: {list(STAGES)}")

    generator = _load_generator()
    logger.info("bounds: over if duration >= {:.0f}s + chars/slow, short if duration "
                "< chars/fast, with (slow, fast) {}",
                budget_fixed_seconds,
                "per voice from utils/frame_budget.py"
                if budget_chars_per_second is None and floor_chars_per_second is None
                else f"overridden to ({budget_chars_per_second}, "
                     f"{floor_chars_per_second})")

    bad: list[dict] = []
    for name in names:
        bad += audit_stage(ROOT / name, STAGES[name], generator.normalize_term,
                           fixed=budget_fixed_seconds, slow=budget_chars_per_second,
                           fast=floor_chars_per_second, workers=workers)

    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w") as f:
            for b in bad:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")
        logger.info("wrote {} offending clips to {}", len(bad), report)

    kinds = Counter(b["verdict"] for b in bad)
    logger.info("total: {} offending clips ({} over budget, {} implausibly short)",
                len(bad), kinds["over"], kinds["short"])
    if not bad:
        return
    if not apply_:
        logger.warning("dry run, nothing deleted. Re-run with --apply to delete these "
                       "{} clips, then re-run stage 05 to regenerate them.", len(bad))
        return
    deleted = 0
    for b in bad:
        try:
            Path(b["path"]).unlink()
            deleted += 1
        except OSError as e:
            logger.error("could not delete {}: {}", b["path"], e)
    logger.info("deleted {} clips; re-run stage 05 to regenerate them (its resume "
                "picks up exactly the missing stems)", deleted)


if __name__ == "__main__":
    main()
