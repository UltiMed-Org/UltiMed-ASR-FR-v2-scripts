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

**Which voice read a clip.** By default a clip is attributed to its hash voice, the
pre-retirement `voice_for_row(..., retired=())`, which is what stage 05 drew. But stage
06 rewrites clips IN PLACE with another voice (a swap or a ladder rung) and records the
reader in the `voice` field of its state file (`06_hotfixes/improved/full.stt.jsonl`).
Pass that file with `--state` and its voice replaces the hash voice for every clip it
names, both for the `retired_voice` verdict and for the per-voice duration bounds, since
it is the voice that produced the audio. Without it, after a retirement, a clip stage 06
re-voiced to a retired voice is missed and one it re-voiced away from a retired voice is
deleted for nothing (3,015 and 3,834 clips respectively on 2026-11-03).

    uv run 05_generate_audio/02_audit_durations.py                    # dry run, all stages
    uv run 05_generate_audio/02_audit_durations.py --stage 01_dictionnary
    uv run 05_generate_audio/02_audit_durations.py --apply            # actually delete
    uv run 05_generate_audio/02_audit_durations.py --report bad.jsonl # keep the list
    uv run 05_generate_audio/02_audit_durations.py \
        --state 06_hotfixes/improved/full.stt.jsonl --report retired.jsonl

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
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
from nemo_manifest import clip_key, probe_duration  # noqa: E402
from frame_budget import (  # noqa: E402
    BUDGET_FIXED_S,
    budget_seconds,
    classify_duration,
    floor_seconds,
    rates_for_voice,
)
from voice_assignment import (RETIRED_VOICES_FR_V2, VOICES_FR_V2,  # noqa: E402
                              group_field, voice_for_row)

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


def _clip_id(path: str | Path) -> tuple[str, str]:
    """`(audio folder name, stem)` of a clip path, the key shared with stage 06's state.

    Built on `clip_key`, so `../PARHAF/000123_0004_x.flac` from a state row and the
    resolved `/media/.../v2/PARHAF/000123_0004_x.flac` of a file on disk meet on
    `("PARHAF", "000123_0004_x")`. The extension is dropped because stage 05 joins on
    stems too.
    """
    key = Path(clip_key(str(path)))
    return key.parent.name, key.stem


def load_state_voices(states: tuple[Path, ...] | list[Path]) -> dict[tuple[str, str], str]:
    """`{(audio folder name, stem): voice}` recorded by stage 06, streamed.

    Each file is read line by line (the real state is ~800 MB), and a later row or a
    later file wins, as in `utils/clip_ledger.load_state`. Rows with no `voice` (v1
    audio) are skipped, so those clips keep their hash voice.
    """
    out: dict[tuple[str, str], str] = {}
    for path in states:
        n = 0
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                voice, audio = row.get("voice"), row.get("audio_filepath")
                if voice and audio:
                    out[_clip_id(audio)] = voice
                    n += 1
        logger.info("state {}: {} clip voices", path, n)
    return out


def audit_stage(stage: Path, mode: str, normalize_term, *, fixed: float,
                slow: float | None, fast: float | None, workers: int,
                stored: dict[tuple[str, str], str] | None = None,
                retired: tuple[str, ...] | None = None) -> list[dict]:
    """Join every row of one stage to its clip on disk and classify the durations.

    Parameters
    ----------
    stored
        From `load_state_voices`: the voice stage 06 recorded for a clip, which wins
        over the hash voice when present. None or empty keeps the hash voice for all.
    retired
        Voices whose clips are condemned; None means `RETIRED_VOICES_FR_V2`.

    Returns
    -------
    list of dict
        One entry per clip that is NOT ok, each carrying the verdict, the path, the
        duration, the two bounds and the voice, ready to be reported or deleted.
    """
    retired = RETIRED_VOICES_FR_V2 if retired is None else tuple(retired)
    stored = stored or {}
    origin: Counter = Counter()
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
        # `retired=()`: this joins clips that are ALREADY on disk, so the voice that
        # matters is the one that actually read them, which is the pre-retirement
        # mapping. Under the live mapping a retired voice's clips would be attributed
        # to whichever voice now owns the row, and the very clips that have to be
        # deleted would become invisible.
        voice = voice_for_row(row, field=field, mode=mode, voices=VOICES_FR_V2,
                              retired=())
        # ...unless stage 06 re-voiced the clip in place, in which case the voice it
        # recorded is the one that read the audio now on disk.
        recorded = stored.get(_clip_id(path))
        if recorded is not None:
            origin["state"] += 1
            if recorded != voice:
                origin["differ"] += 1
                origin["stored_retired"] += recorded in retired and voice not in retired
                origin["hash_retired"] += voice in retired and recorded not in retired
            voice = recorded
        rows.append((voice, row["asr_training_source"], path))

    logger.info("{}: {} clips on disk joined to their text, {} not generated yet",
                stage.name, len(rows), missing)
    if stored:
        logger.info("{}: {} clip voice(s) taken from the stage-06 state, {} differing "
                    "from the hash voice ({} now retired where the hash voice is not, "
                    "{} spared where only the hash voice is retired)",
                    stage.name, origin["state"], origin["differ"],
                    origin["stored_retired"], origin["hash_retired"])
    if not rows:
        return []

    # Threads because this is pure I/O: sf.info reads a header, so 600k files are
    # minutes rather than hours, and the GIL is released during the read.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        durations = list(pool.map(lambda r: probe_duration(r[2]), rows))

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
        # A clip read by a retired voice is wrong whatever its duration says, because
        # the row now belongs to another voice: deleting it is what makes the next
        # resume of 01_generate_audio.py re-read it in that voice. Checked before the
        # duration verdict so a retired clip is reported once, under the reason that
        # actually condemns it.
        if voice in retired:
            verdict = "retired_voice"
        else:
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
    if verdicts["retired_voice"]:
        logger.warning("{}: {} clip(s) ({:.3f}%) were read by a retired voice ({}) "
                       "and must be regenerated; --apply deletes them and the next "
                       "run of 01_generate_audio.py refills them",
                       stage.name, verdicts["retired_voice"],
                       100.0 * verdicts["retired_voice"] / max(1, total),
                       ", ".join(retired))

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
@click.option("--state", "states", multiple=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Stage-06 state file (06_hotfixes/improved/full.stt.jsonl), "
                   "repeatable, later files win. The voice it records for a clip "
                   "replaces the hash voice, since stage 06 re-voices clips in "
                   "place. Needed whenever a voice is retired.")
def main(stages: tuple[str, ...], apply_: bool, report: Path | None,
         budget_fixed_seconds: float, budget_chars_per_second: float,
         floor_chars_per_second: float, workers: int,
         states: tuple[Path, ...] = ()) -> None:
    """Audit, and optionally delete, clips whose duration does not match their text."""
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    names = stages or tuple(STAGES)
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        raise click.BadParameter(f"unknown stage(s) {unknown}, known: {list(STAGES)}")

    generator = _load_generator()
    if RETIRED_VOICES_FR_V2 and not states:
        logger.warning("voices are retired ({}) but no --state was given: clips stage "
                       "06 re-voiced in place are attributed to their hash voice, so "
                       "some retired readings are missed and some survivors' audio is "
                       "deleted. Pass --state 06_hotfixes/improved/full.stt.jsonl",
                       ", ".join(RETIRED_VOICES_FR_V2))
    stored = load_state_voices(states)
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
                           fast=floor_chars_per_second, workers=workers,
                           stored=stored)

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
