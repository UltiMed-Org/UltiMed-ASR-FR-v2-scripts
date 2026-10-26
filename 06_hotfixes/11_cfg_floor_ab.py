#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
# ]
# ///
"""Is the audio generated below the measured cfg_alpha floor actually worse?

The problem, in numbers. The usable cfg_alpha band was fixed by listening on 2026-10-13:
1.2 to 2.0 for the two shipped Voxtral voices, **1.5 to 2.0 for the hybrids**, which
"need more guidance to hold their articulation". The compose default only moved to 2.0
that day, so 165,066 clips (27% of the corpus) are hybrid voices drawn at cfg_alpha 1.3,
under their own floor. Their Whisper CER is indistinguishable from the in-band hybrids
(p50 0.0118 against 0.0127, p90 0.0336 against 0.0342), which proves nothing either way:
Whisper has a strong language model and recovers the words from mumbled speech, so the
instrument is blind to exactly the property the floor protects.

The experiment. Parakeet TDT carries a far weaker implicit language model, so its CER
tracks acoustic intelligibility much more closely than Whisper's does. Score matched
cohorts with it:

* if the below-floor cohort scores WORSE under Parakeet while scoring the same under
  Whisper, the articulation really is degraded and 165k clips need a decision;
* if Parakeet sees no difference either, the floor was conservative and the clips are
  fine, which is the cheapest possible answer to the question.

Matching matters more than sample size here, because CER varies far more by voice and by
text category than by anything this experiment is measuring. So cohorts are drawn
stratified by (voice, category, duration bucket) and only from strata that hold clips on
BOTH sides, which makes it a paired comparison rather than two averages that differ for
the wrong reason.

This script only picks the clips and writes the stem lists. The scoring is stage 05's CPU
judge, which already owns the model loading, the sidecar and the resume, pointed at the
IMPARTIAL base export rather than the UltiMed finetune (which was trained on v1 of this
corpus and would snap unclear speech back onto expected French, destroying the signal):

    # 1. pick the cohorts
    uv run 06_hotfixes/11_cfg_floor_ab.py --per-arm 600

    # 2. score them both with base Parakeet on CPU (~2 s/clip, 4 workers, no GPU)
    cd 05_generate_audio && uv run ./03_qc_cpu_stt.py --backfill --once \
        --sample-percent 0 --no-retries --keep-transcripts \
        --stems-from ../06_hotfixes/cfg_ab_stems.txt \
        --model-dir ../../parakeet-tdt-0.6b-v3-ultra-onnx \
        --out-name qc_cfg_ab.jsonl --log-file qc_cfg_ab.log --workers 4 --threads 3

    # 3. read the result out
    uv run 06_hotfixes/11_cfg_floor_ab.py --report

Step 2 is now usually FREE. The full-corpus Parakeet sweep of 2026-10-26 scores every
one of the 602,789 clips with this exact judge (`qc_parakeet_ultra.jsonl` per stage), so
the cohorts are already scored and `--report` can read those sidecars instead:

    uv run 06_hotfixes/11_cfg_floor_ab.py --per-arm 20000
    uv run 06_hotfixes/11_cfg_floor_ab.py --report \
        --sidecar /media/$USER/ssd1TO/audio_datasets/v2/dictionary/qc_parakeet_ultra.jsonl \
        --sidecar .../drugs/qc_parakeet_ultra.jsonl   # and the other three stages

With the scoring free, `--per-arm` is bounded by how many clips the matched strata hold
rather than by compute, so raise it: the whole point of the stratification is that a
bigger arm narrows the interval without unbalancing the comparison.

Written with Claude Code.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent

#: Hybrid voices are the ones whose floor is 1.5 rather than 1.2, and they are named for
#: it: `h_<source><target>` against the two shipped `fr_*` voices.
def is_hybrid(voice: str) -> bool:
    return bool(voice) and voice.startswith("h_")


#: The floor each voice family was measured to need, from the 2026-10-13 audition.
FLOOR_HYBRID, FLOOR_BASE = 1.5, 1.2

#: Duration buckets, in seconds. A defect's visibility to CER depends strongly on clip
#: length (a dropped phrase is diluted in a long clip), so a cohort that is shorter on
#: average than its counterpart would show a CER difference that says nothing about cfg.
DURATION_EDGES = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 1e9)


def bucket_of(duration: float) -> int:
    for i in range(len(DURATION_EDGES) - 1):
        if DURATION_EDGES[i] <= duration < DURATION_EDGES[i + 1]:
            return i
    return len(DURATION_EDGES) - 2


def stratum(row: dict) -> tuple:
    return (row.get("voice"), row.get("category"), bucket_of(row.get("duration") or 0.0))


def floor_for(voice: str) -> float:
    return FLOOR_HYBRID if is_hybrid(voice) else FLOOR_BASE


def pick(manifest: Path, per_arm: int, seed: int, hybrids_only: bool) -> dict:
    """Draw two matched cohorts: below the voice's floor, and at or above it."""
    below: dict[tuple, list] = defaultdict(list)
    above: dict[tuple, list] = defaultdict(list)
    total = manifest.stat().st_size
    with manifest.open("rb") as fh, tqdm(total=total, desc="read manifest", unit="B",
                                         unit_scale=True, unit_divisor=1024,
                                         leave=False) as bar:
        for raw in fh:
            bar.update(len(raw))
            if not raw.strip():
                continue
            r = json.loads(raw)
            voice, cfg = r.get("voice"), r.get("cfg_alpha")
            if not voice or cfg is None:
                continue
            if hybrids_only and not is_hybrid(voice):
                continue
            # 1.0 is a different question (CFG disabled outright, a known hallucination
            # risk) and so are the values above the band; this compares in-band-vs-just-
            # under, which is the only comparison the 165k clips actually pose.
            if cfg <= 1.0 or cfg > 2.0:
                continue
            (below if cfg < floor_for(voice) else above)[stratum(r)].append(
                {"stem": Path(r["audio_filepath"]).stem, "voice": voice,
                 "cfg_alpha": cfg, "duration": r.get("duration"),
                 "category": r.get("category"), "cer_whisper": r.get("cer")})

    shared = sorted(set(below) & set(above))
    logger.info(f"{len(shared)} stratum/strata hold clips on both sides "
                f"({len(below)} below-floor strata, {len(above)} at-or-above)")
    if not shared:
        logger.error("no stratum has clips on both sides: nothing to compare")
        sys.exit(2)

    rng = random.Random(seed)
    # Spread the budget over strata in proportion to how many BOTH sides can supply, so
    # the two arms end up with the same stratum mix, which is the whole point.
    capacity = {s: min(len(below[s]), len(above[s])) for s in shared}
    room = sum(capacity.values())
    take = {s: max(1, round(per_arm * capacity[s] / room)) for s in shared}
    arms = {"below": [], "above": []}
    for s in shared:
        k = min(take[s], capacity[s])
        arms["below"].extend(rng.sample(below[s], k))
        arms["above"].extend(rng.sample(above[s], k))
    logger.info(f"drew {len(arms['below'])} below-floor and {len(arms['above'])} "
                f"at-or-above clips over {len(shared)} matched strata")
    return arms


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=str(_HERE.parent / "99_hf_release" / "data" / "NeMO_files"
                          / "full.jsonl"), show_default=True)
@click.option("--per-arm", default=600, show_default=True, type=int,
              help="Clips per arm. The arms are matched stratum by stratum, so the "
                   "realised size is close to but not exactly this.")
@click.option("--hybrids-only/--all-voices", default=True, show_default=True,
              help="The 1.5 floor is a hybrid finding, so the shipped fr_* voices only "
                   "add noise unless you are checking the 1.2 floor too.")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--stems-out", type=click.Path(dir_okay=False, path_type=Path),
              default=str(_HERE / "cfg_ab_stems.txt"), show_default=True,
              help="Stem list for 03_qc_cpu_stt.py --stems-from (both arms, since the "
                   "judge must not know which arm a clip is in).")
@click.option("--cohorts-out", type=click.Path(dir_okay=False, path_type=Path),
              default=str(_HERE / "cfg_ab_cohorts.jsonl"), show_default=True,
              help="Which arm each clip is in, read back by --report.")
@click.option("--sidecar", "sidecars", multiple=True,
              type=click.Path(path_type=Path),
              help="With --report: the qc_cfg_ab.jsonl sidecar(s) the judge wrote. "
                   "Default: every stage's, under the audio root.")
@click.option("--audio-root", type=click.Path(file_okay=False, path_type=Path),
              default=str(_HERE.parent / "99_hf_release" / "data"), show_default=True)
@click.option("--report", "do_report", is_flag=True,
              help="Read the judge's sidecar and print the comparison instead of "
                   "drawing new cohorts.")
def main(manifest: Path, per_arm: int, hybrids_only: bool, seed: int, stems_out: Path,
         cohorts_out: Path, sidecars: tuple[Path, ...], audio_root: Path,
         do_report: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | <level>{level: <7}</level> | "
                                  "{message}", colorize=True)

    if not do_report:
        arms = pick(manifest, per_arm, seed, hybrids_only)
        with stems_out.open("w") as fh:
            fh.write("# cfg_alpha floor A/B cohorts, from 11_cfg_floor_ab.py\n")
            fh.write(f"# {len(arms['below'])} below-floor + {len(arms['above'])} "
                     f"at-or-above, matched by (voice, category, duration bucket)\n")
            for arm in ("below", "above"):
                for c in arms[arm]:
                    fh.write(c["stem"] + "\n")
        with cohorts_out.open("w") as fh:
            for arm in ("below", "above"):
                for c in arms[arm]:
                    fh.write(json.dumps({"arm": arm, **c}, ensure_ascii=False) + "\n")
        logger.success(f"wrote {stems_out} and {cohorts_out}")
        return

    cohorts = {}
    for line in cohorts_out.open():
        if line.strip():
            c = json.loads(line)
            cohorts[c["stem"]] = c
    if not cohorts:
        logger.error(f"{cohorts_out} is empty: draw the cohorts first")
        sys.exit(2)

    paths = list(sidecars) or list(audio_root.glob("*/qc_cfg_ab.jsonl"))
    readings: dict[str, dict] = {}
    for p in paths:
        if not p.is_file():
            continue
        for line in p.open():
            if line.strip():
                row = json.loads(line)
                stem = row.get("stem") or Path(row.get("clip", "")).stem
                if stem:
                    readings[stem] = row
    logger.info(f"{len(readings)} reading(s) from {len(paths)} sidecar(s), "
                f"{len(cohorts)} clip(s) in the cohorts")

    scored = defaultdict(lambda: defaultdict(list))
    paired: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for stem, c in cohorts.items():
        r = readings.get(stem)
        if not r or r.get("cer") is None:
            continue
        key = (c["voice"], c["category"], bucket_of(c["duration"] or 0.0))
        scored[c["arm"]]["parakeet"].append(r["cer"])
        scored[c["arm"]]["whisper"].append(c["cer_whisper"])
        paired[key][c["arm"]].append(r["cer"])

    if not scored:
        logger.error("no cohort clip has a Parakeet reading yet: run the judge first")
        sys.exit(2)

    print()
    print(f"{'arm':<22}{'n':>7}{'parakeet p50':>14}{'p90':>9}{'mean':>9}"
          f"{'whisper p50':>14}{'mean':>9}")
    for arm in ("below", "above"):
        pk = sorted(x for x in scored[arm]["parakeet"] if x is not None)
        wh = sorted(x for x in scored[arm]["whisper"] if x is not None)
        if not pk:
            continue
        label = "below floor (cfg<1.5)" if arm == "below" else "in band (cfg>=1.5)"
        print(f"{label:<22}{len(pk):>7}{st.median(pk):>14.4f}"
              f"{pk[int(.9 * len(pk))]:>9.4f}{st.fmean(pk):>9.4f}"
              f"{st.median(wh):>14.4f}{st.fmean(wh):>9.4f}")

    # The paired read: within each matched stratum, which arm has the higher median.
    wins = {"below": 0, "above": 0, "tie": 0}
    deltas = []
    for key, arms_ in paired.items():
        b, a = arms_.get("below"), arms_.get("above")
        if not b or not a:
            continue
        db, da = st.median(b), st.median(a)
        deltas.append(db - da)
        if abs(db - da) < 1e-9:
            wins["tie"] += 1
        elif db > da:
            wins["below"] += 1
        else:
            wins["above"] += 1
    print()
    print(f"paired over {len(deltas)} matched stratum/strata: below-floor worse in "
          f"{wins['below']}, better in {wins['above']}, tied in {wins['tie']}")
    if deltas:
        print(f"median per-stratum CER delta (below minus in-band): "
              f"{st.median(deltas):+.4f}")
        print("A positive delta means the below-floor clips are harder for an acoustic "
              "recognizer,\nwhich is the mumbling the Whisper CER cannot see.")


if __name__ == "__main__":
    main()
