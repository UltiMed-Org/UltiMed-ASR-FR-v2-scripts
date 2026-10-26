#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jiwer>=3.0",
#   "requests>=2.31",  # unused here; stage 06's scorer imports it at module scope
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "soundfile>=0.12",
# ]
# ///
"""Refuse to ship a manifest that does not describe the audio it points at.

Why this exists: every other script in this repo WRITES a manifest, and each one is
careful about its own columns. Nothing reads the finished article back and asks whether
it is internally consistent, so the failure modes that survive are exactly the ones no
single writer can see: a `cer` left behind by a redraw that replaced the audio, a row
whose `duration` came from the clip that was deleted rather than the one that replaced
it, a `voice` column that disagrees with the sidecar, the same document's chunks split
across train and val. Each of those trains the model on a lie, and none of them shows up
as a crash.

Five groups of checks, each independently selectable so a 30-second structural pass is
not held hostage by the 20-minute acoustic one:

``structure``
    Every row carries every expected column, filled, of the right type, with values in
    range: text non-empty, duration positive, voice on the roster, cfg_alpha inside the
    measured usable band, qc_status a known value. Cheap, reads nothing but the jsonl.

``files``
    The audio each row points at exists, and `soundfile` agrees with the stored
    `duration` to within `--duration-tolerance`. Header reads only, no decode. Also
    reports stray files beside the audio (`*.bak`, `*.tmp`, `*.flac.tmp`), which are the
    residue of an interrupted write, and clips on disk that no manifest mentions.

``scores``
    Recompute `cer`, `cer_tail`, `cer_window` and `bad_run_chars` from the row's OWN
    stored `stt_transcript` against its `text` (the label; see the comment at the
    recompute for why the label and not `asr_training_source`), with stage 06's scorer
    (imported, not re-spelled), and compare to what the row claims. A mismatch means the
    stored score describes a different transcript than the stored one, i.e. a stale score,
    which is the bug class that bit this corpus twice in 2026-10. Expensive (jiwer per
    row), so it runs on every row that is not pristine plus `--sample-percent` of the rest.

``splits``
    train/val/test must partition `full.jsonl` exactly: no clip in two splits, no clip in
    none, nothing in a split that is not in full. Then the leakage checks that matter for
    a believable val WER: the same `text` appearing in two splits, and the same `group_id`
    (one PARHAF document, chunked) straddling two splits.

``coverage``
    Per-split voice and category tables, because a val set missing a voice cannot measure
    that voice, and v2's whole point is that there are nine of them.

    uv run 99_hf_release/08_validate_manifests.py                       # everything
    uv run 99_hf_release/08_validate_manifests.py --only structure --only splits
    uv run 99_hf_release/08_validate_manifests.py --sample-percent 100  # all scores
    uv run 99_hf_release/08_validate_manifests.py --report audit.jsonl  # keep the rows

Exit status is 1 when any check produced an ERROR, 0 when only warnings, so it can gate a
release in a shell `&&` chain. Written with Claude Code.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
import soundfile as sf
from loguru import logger
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_STT = None


def _load_stt():
    """Import `06_hotfixes/01_compute_stt.py` (its filename is not a module name).

    The scorer lives with the pass that WRITES the scores, which is what makes this a
    check rather than a second opinion: if the gate moves, this script moves with it.
    """
    global _STT
    if _STT is None:
        import importlib.util

        path = _HERE.parent / "06_hotfixes" / "01_compute_stt.py"
        spec = importlib.util.spec_from_file_location("stt_for_validate", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _STT = mod
    return _STT


#: Columns every row of a release manifest must carry. A row missing one of these is an
#: ERROR, not a warning: the trainer reads the first three and the QC story needs the rest.
REQUIRED_KEYS = ("audio_filepath", "duration", "text", "asr_training_source")

#: Columns that must be PRESENT but may legitimately be null, with why.
OPTIONAL_KEYS = {
    "cer_tail": "None on clips at or under 30 s: there is no tail to isolate",
    "n_stt_check": "None until a clip has been re-checked at least once",
    # The per-judge groups `03_sync_hotfix_results.py` writes. Present on every row and
    # null where that judge did not read the clip, which for the non-turbo second
    # opinion is most of the corpus (it ran only over what turbo flagged).
    "whisper_stt": "None if the turbo judge has no reading for the clip",
    "whisper_cer": "same",
    "whisper_cer_window": "same",
    "whisper_repeat_excess": "same",
    "whisper_v3_stt": "None unless turbo flagged the clip and the second opinion ran",
    "whisper_v3_cer": "same",
    "whisper_v3_cer_window": "same",
    "whisper_v3_repeat_excess": "same",
    "parakeet_stt": "None until the parakeet sweep's sidecar is merged in",
    "parakeet_cer": "same",
    "parakeet_cer_window": "same",
    "parakeet_repeat_excess": "same",
    "cer_max": "None only when no judge scored the clip at all",
    "cer_window_max": "same",
}

#: `cer_max` / `cer_window_max` must be the max over the per-judge columns of the same
#: name, so a consumer filtering on one of them cannot be shown a number that no judge
#: produced. Keyed by the maximum, valued by the columns it summarizes.
JUDGE_MAXIMA = {
    "cer_max": ("whisper_cer", "whisper_v3_cer", "parakeet_cer"),
    "cer_window_max": ("whisper_cer_window", "whisper_v3_cer_window",
                       "parakeet_cer_window"),
}

#: Everything else a v2 row is expected to carry, with the type it must have when set.
EXPECTED_KEYS = {
    "category": str, "group_id": str, "group_mode": str, "item_index": int,
    "voice": str, "cer": float, "stt_transcript": str, "stt_model": str,
    "cfg_alpha": float, "regenerated": bool, "qc_status": str,
}

#: The v2 roster, as `full.jsonl` actually spells it on 2026-10-25: the two shipped
#: Voxtral voices that survived the measurement, plus the seven hybrids. A voice outside
#: this set means the manifest and the generator disagree about the roster.
KNOWN_VOICES = frozenset({
    "fr_male", "fr_female",
    "h_itf_arm", "h_itf_def", "h_frf_esm", "h_frm_ptf", "h_frf_itm", "h_frm_nlf",
    "h_frf_ptm",
})

#: `cfg_alpha` band measured on 2026-10-13 (16 voices x 10 values, listened to): 1.2 to
#: 2.0 for the shipped voices, 1.5 to 2.0 for the hybrids, and 1.0 disables CFG outright.
CFG_FLOOR, CFG_CEILING = 1.2, 2.0

#: How far the stored score may sit from the recomputed one before it is a stale score
#: rather than a float-formatting difference.
SCORE_EPSILON = 1e-4

#: Stray-file suffixes that mean an interrupted write, not a clip.
STRAY_SUFFIXES = (".bak", ".tmp", ".flac.tmp", ".part")


class Findings:
    """Errors and warnings, deduplicated by (check, kind) with up to `keep` examples.

    A corpus-wide audit that printed one line per bad row would print 602,789 lines on a
    manifest whose every row is bad, so a finding is a COUNT plus a few examples, and the
    full list goes to `--report` when one is asked for.
    """

    def __init__(self, keep: int = 5):
        self.keep = keep
        self.errors: dict[tuple[str, str], list] = defaultdict(list)
        self.warnings: dict[tuple[str, str], list] = defaultdict(list)
        self.counts: Counter = Counter()
        self.rows: list[dict] = []

    def _add(self, bucket, check, kind, example, record):
        key = (check, kind)
        self.counts[key] += 1
        if len(bucket[key]) < self.keep:
            bucket[key].append(example)
        if record is not None:
            self.rows.append({"check": check, "kind": kind, **record})

    def error(self, check, kind, example, record=None):
        self._add(self.errors, check, kind, example, record)

    def warn(self, check, kind, example, record=None):
        self._add(self.warnings, check, kind, example, record)

    def render(self) -> bool:
        """Log everything found, worst first. Returns True when there were errors."""
        for label, bucket, log in (("ERROR", self.errors, logger.error),
                                   ("WARN", self.warnings, logger.warning)):
            keys = sorted(bucket, key=lambda k: -self.counts[k])
            for check, kind in keys:
                n = self.counts[(check, kind)]
                log(f"[{label}] {check}: {kind} ({n} row(s))")
                for ex in bucket[(check, kind)]:
                    log(f"         e.g. {ex}")
        if not self.errors and not self.warnings:
            logger.success("no findings")
        return bool(self.errors)


def read_manifest(path: Path, *, bar: bool = True) -> list[dict]:
    """Parse a manifest, barred by bytes so a 950 MB read is not a silent minute."""
    rows: list[dict] = []
    total = path.stat().st_size
    with path.open("rb") as fh, tqdm(total=total, desc=f"read {path.name}", unit="B",
                                     unit_scale=True, unit_divisor=1024,
                                     disable=not bar, leave=False) as pb:
        for i, raw in enumerate(fh):
            pb.update(len(raw))
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.error(f"{path.name}:{i + 1} is not valid JSON: {exc}")
    return rows


def check_structure(rows: list[dict], name: str, f: Findings) -> None:
    """Columns present, filled, typed, and in range."""
    for i, r in enumerate(rows):
        ident = r.get("audio_filepath") or f"{name}:{i + 1}"
        for key in REQUIRED_KEYS:
            if key not in r:
                f.error(name, f"missing required column '{key}'", ident)
            elif r[key] is None or (isinstance(r[key], str) and not r[key].strip()):
                f.error(name, f"empty required column '{key}'", ident)
        for key in OPTIONAL_KEYS:
            if key not in r:
                f.warn(name, f"missing column '{key}'", ident)
        for key, typ in EXPECTED_KEYS.items():
            if key not in r:
                f.error(name, f"missing column '{key}'", ident)
                continue
            v = r[key]
            if v is None:
                f.error(name, f"null column '{key}'", ident)
            elif typ is float and not isinstance(v, (int, float)):
                f.error(name, f"column '{key}' is {type(v).__name__}, want number", ident)
            elif typ is not float and not isinstance(v, typ):
                f.error(name, f"column '{key}' is {type(v).__name__}, "
                              f"want {typ.__name__}", ident)

        # A maximum that no judge produced is worse than a missing one: a consumer
        # filtering `cer_max < 0.08` would be trusting a number with no source. Checked
        # per row rather than in aggregate because that is the only way to name the clip.
        for top, parts in JUDGE_MAXIMA.items():
            if top not in r:
                continue
            seen = [r[p] for p in parts if isinstance(r.get(p), (int, float))]
            want = max(seen) if seen else None
            got = r[top]
            if want is None:
                if got is not None:
                    f.error(name, f"'{top}' is {got} but no judge scored this clip", ident)
            elif not isinstance(got, (int, float)) or abs(got - want) > 1e-9:
                f.error(name, f"'{top}' is {got}, want {want} "
                              f"(max of {', '.join(parts)})", ident)

        dur = r.get("duration")
        if isinstance(dur, (int, float)) and dur <= 0:
            f.error(name, "duration is not positive", f"{ident} duration={dur}")

        cer = r.get("cer")
        if isinstance(cer, (int, float)) and cer < 0:
            f.error(name, "cer is negative", f"{ident} cer={cer}")

        voice = r.get("voice")
        if isinstance(voice, str) and voice and voice not in KNOWN_VOICES:
            f.error(name, "voice is not on the v2 roster", f"{ident} voice={voice!r}")

        cfg = r.get("cfg_alpha")
        if isinstance(cfg, (int, float)):
            if cfg == 1.0:
                f.error(name, "cfg_alpha 1.0 disables CFG (hallucination risk)", ident)
            elif not (CFG_FLOOR <= cfg <= CFG_CEILING):
                f.warn(name, f"cfg_alpha outside the measured {CFG_FLOOR}-{CFG_CEILING} "
                             f"band", f"{ident} cfg_alpha={cfg}")

        # A row that says it was regenerated but still calls itself pristine (or the
        # reverse) means one of the two writers did not run.
        qc, regen = r.get("qc_status"), r.get("regenerated")
        if qc == "original" and regen is True:
            f.error(name, "regenerated=True with qc_status='original'", ident)

    logger.info(f"{name}: structure checked over {len(rows)} row(s)")
    for key in ("qc_status", "group_mode", "stt_model"):
        tally = Counter(r.get(key) for r in rows)
        logger.info(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in tally.most_common()))


def check_files(rows: list[dict], name: str, manifest_dir: Path, f: Findings, *,
                tolerance: float, scan_strays: bool) -> None:
    """The audio exists and is as long as the row claims."""
    audio_dirs: set[Path] = set()
    seen_on_disk: set[Path] = set()
    for r in tqdm(rows, desc=f"{name} audio", unit="clip", leave=False):
        rel = r.get("audio_filepath")
        if not rel:
            continue
        path = (manifest_dir / rel).resolve()
        audio_dirs.add(path.parent)
        if not path.is_file():
            f.error(name, "audio file does not exist", str(path),
                    {"audio_filepath": rel})
            continue
        seen_on_disk.add(path)
        try:
            info = sf.info(str(path))
        except Exception as exc:  # a truncated or corrupt container
            f.error(name, "audio header is unreadable", f"{path}: {exc}",
                    {"audio_filepath": rel})
            continue
        if info.frames == 0 or info.samplerate == 0:
            f.error(name, "audio holds zero frames", str(path), {"audio_filepath": rel})
            continue
        real = info.frames / info.samplerate
        claimed = r.get("duration")
        if isinstance(claimed, (int, float)) and abs(real - claimed) > tolerance:
            f.error(name, f"duration disagrees with the file by more than {tolerance}s",
                    f"{path} manifest={claimed:.3f} file={real:.3f}",
                    {"audio_filepath": rel, "duration": claimed, "real_duration": real})

    if not scan_strays:
        return
    for d in sorted(audio_dirs):
        strays = [p.name for p in d.iterdir()
                  if p.is_file() and p.name.endswith(STRAY_SUFFIXES)]
        if strays:
            f.warn(name, f"stray files beside the audio in {d.name}",
                   f"{len(strays)} file(s), e.g. {strays[:3]}")
        orphans = [p for p in d.glob("*.flac") if p.resolve() not in seen_on_disk]
        if orphans:
            f.warn(name, f"clips on disk that {name} does not mention, in {d.name}",
                   f"{len(orphans)} file(s), e.g. {[p.name for p in orphans[:3]]}")


def check_scores(rows: list[dict], name: str, f: Findings, *, sample_percent: float,
                 seed: int) -> None:
    """Recompute each row's scores from its own stored transcript and compare.

    Every row that is not pristine is checked whatever the sample says, because those are
    the rows a redraw touched and therefore the only ones that CAN carry a stale score.
    """
    stt = _load_stt()
    rng = random.Random(seed)
    todo = [r for r in rows
            if r.get("qc_status") != "original"
            or (r.get("cer") or 0) > 0
            or rng.uniform(0, 100) < sample_percent]
    logger.info(f"{name}: recomputing scores for {len(todo)} of {len(rows)} row(s) "
                f"(all non-pristine, plus {sample_percent}% of the rest)")

    checked = 0
    for r in tqdm(todo, desc=f"{name} scores", unit="row", leave=False):
        # `text`, the LABEL, not `asr_training_source`. Stage 06 scores against the source
        # (what the TTS read) at QC time, and the two normally fold to the same string
        # under `normalize_for_scoring` (it already folds `%` vs `pour cent`, digits vs
        # words, accents). But 05_normalize_text.py rewrites the SOURCE afterwards with
        # pronunciation respellings the fold does not reach: `oestrogènes` ->
        # `éstrogènes`, `foetal` -> `fétal`, which is deliberate (it is what the TTS is
        # given) and leaves the shipped source one character from the audio's spelling.
        # Recomputing against it charges that respelling as a recognition error. Measured
        # on this validator's own 3,686 flagged v2 rows: 0 match a recompute against
        # `asr_training_source`, 3,684 match one against `text`, and all 318 tail scores
        # match `text`. So the source was the reason this check fired, not stale scores,
        # and the 2 rows left over are the only genuinely stale ones in the release.
        ref, hyp = r.get("text"), r.get("stt_transcript")
        if not ref or not hyp:
            # Not an error here: a row with no transcript is structure's business.
            continue
        checked += 1
        ident = r.get("audio_filepath")
        _, cer = stt.compute_metrics(ref, hyp)
        stored = r.get("cer")
        if cer is not None and isinstance(stored, (int, float)) \
                and abs(cer - stored) > SCORE_EPSILON:
            f.error(name, "stored cer does not match the stored transcript",
                    f"{ident} stored={stored:.6f} recomputed={cer:.6f}",
                    {"audio_filepath": ident, "cer": stored, "recomputed_cer": cer})

        dur = r.get("duration")
        if isinstance(dur, (int, float)):
            tail = stt.tail_cer(ref, hyp, dur)
            stored_tail = r.get("cer_tail")
            if tail is None and stored_tail is not None:
                f.warn(name, "cer_tail stored on a clip too short to have a tail",
                       f"{ident} duration={dur:.1f}s cer_tail={stored_tail}")
            elif tail is not None and isinstance(stored_tail, (int, float)) \
                    and abs(tail - stored_tail) > SCORE_EPSILON:
                f.error(name, "stored cer_tail does not match the stored transcript",
                        f"{ident} stored={stored_tail:.6f} recomputed={tail:.6f}",
                        {"audio_filepath": ident, "cer_tail": stored_tail,
                         "recomputed_cer_tail": tail})

        # The gates, applied to what the row actually says. A row whose own transcript
        # trips a gate while the row calls itself fine is the one that reaches training.
        window, bad_run = stt.window_metrics(ref, hyp)
        tripped = []
        if cer is not None and cer >= stt.DEFAULT_CER_GATE:
            tripped.append(f"cer={cer:.3f}")
        if window is not None and window >= stt.DEFAULT_WINDOW_CER_GATE:
            tripped.append(f"window={window:.3f}")
        if bad_run is not None and bad_run >= stt.DEFAULT_BAD_RUN_GATE:
            tripped.append(f"bad_run={bad_run}")
        if tripped:
            f.warn(name, "row trips a QC gate on its own stored transcript",
                   f"{ident} {' '.join(tripped)} qc_status={r.get('qc_status')!r}",
                   {"audio_filepath": ident, "tripped": tripped,
                    "qc_status": r.get("qc_status")})
    logger.info(f"{name}: {checked} row(s) had a transcript to recompute")


def check_splits(full: list[dict], splits: dict[str, list[dict]], f: Findings) -> None:
    """train/val/test must partition full, and must not leak text or documents."""
    name = "splits"
    full_keys = {r.get("audio_filepath") for r in full}
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        for r in rows:
            key = r.get("audio_filepath")
            if key in seen:
                f.error(name, "clip appears in two splits",
                        f"{key} in {seen[key]} and {split}")
            else:
                seen[key] = split
            if key not in full_keys:
                f.error(name, "clip is in a split but not in full.jsonl",
                        f"{key} ({split})")
    missing = full_keys - set(seen)
    if missing:
        f.error(name, "clip is in full.jsonl but in no split",
                f"{len(missing)} clip(s), e.g. {sorted(missing)[:3]}")
    logger.info(f"{name}: {len(full_keys)} clip(s) in full, {len(seen)} across "
                f"{len(splits)} split(s)")

    # Leakage. Identical text in two splits makes the val WER optimistic; a chunked
    # document straddling two splits does the same thing one level up, because the
    # chunks share vocabulary, reader and recording session.
    for field, label in (("text", "identical text"), ("group_id", "the same document")):
        where: dict[str, set[str]] = defaultdict(set)
        for split, rows in splits.items():
            for r in rows:
                v = r.get(field)
                if v:
                    where[v].add(split)
        shared = {k: v for k, v in where.items() if len(v) > 1}
        if shared:
            examples = [f"{k[:60]!r} in {sorted(v)}" for k, v in
                        list(shared.items())[:3]]
            f.warn(name, f"{label} appears in more than one split",
                   f"{len(shared)} value(s), e.g. {examples}")
        else:
            logger.success(f"{name}: no {label} shared across splits")


def check_coverage(splits: dict[str, list[dict]], f: Findings) -> None:
    """Every split must see every voice and every category, or it cannot measure them."""
    name = "coverage"
    per_split_voices: dict[str, Counter] = {}
    for split, rows in splits.items():
        per_split_voices[split] = Counter(r.get("voice") for r in rows)
        cats = Counter(r.get("category") for r in rows)
        logger.info(f"{name}: {split} categories: "
                    + ", ".join(f"{k}={v}" for k, v in cats.most_common()))
    all_voices = set().union(*(set(c) for c in per_split_voices.values()))
    for split, tally in per_split_voices.items():
        logger.info(f"{name}: {split} voices: "
                    + ", ".join(f"{k}={v}" for k, v in tally.most_common()))
        absent = all_voices - set(tally)
        if absent:
            f.error(name, f"{split} contains no clip for some voice(s)",
                    f"{sorted(absent)}")
        # A voice present but vanishingly rare cannot carry a per-voice WER either.
        thin = {v: n for v, n in tally.items() if n < 100}
        if thin and split != "train":
            f.warn(name, f"{split} has fewer than 100 clips for some voice(s)",
                   f"{thin}")


ALL_CHECKS = ("structure", "files", "scores", "splits", "coverage")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--manifest-dir", type=click.Path(file_okay=False, path_type=Path),
              default=str(_HERE / "data" / "NeMO_files"), show_default=True,
              help="Directory holding full/train/val/test .jsonl. Audio paths are "
                   "resolved relative to it, which is how the trainer reads them.")
@click.option("--only", "only", multiple=True, type=click.Choice(ALL_CHECKS),
              help="Run just these check groups, repeatable. Default: all of them.")
@click.option("--sample-percent", default=2.0, show_default=True, type=float,
              help="Percent of PRISTINE rows to recompute scores for. Every row that "
                   "is not pristine is recomputed whatever this says.")
@click.option("--duration-tolerance", default=0.05, show_default=True, type=float,
              help="Seconds the stored duration may differ from the file's own.")
@click.option("--no-strays", is_flag=True,
              help="Skip the stray-file and orphan-clip scan, which lists the audio "
                   "directories and is the slow part of --only files.")
@click.option("--examples", default=5, show_default=True, type=int,
              help="Example rows to print per finding.")
@click.option("--seed", default=0, show_default=True, type=int,
              help="Sampling seed, so a re-run checks the same rows.")
@click.option("--report", type=click.Path(dir_okay=False, path_type=Path),
              help="Write every offending row as jsonl here, beyond the examples.")
def main(manifest_dir: Path, only: tuple[str, ...], sample_percent: float,
         duration_tolerance: float, no_strays: bool, examples: int, seed: int,
         report: Path | None) -> None:
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | <level>{level: <7}</level> | "
                                  "{message}", colorize=True)
    checks = set(only) if only else set(ALL_CHECKS)
    manifest_dir = manifest_dir.resolve()
    f = Findings(keep=examples)

    full_path = manifest_dir / "full.jsonl"
    if not full_path.is_file():
        logger.error(f"no full.jsonl in {manifest_dir}")
        sys.exit(2)
    full = read_manifest(full_path)
    logger.info(f"full.jsonl: {len(full)} row(s)")

    splits = {}
    for split in ("train", "val", "test"):
        p = manifest_dir / f"{split}.jsonl"
        if p.is_file():
            splits[split] = read_manifest(p)
            logger.info(f"{split}.jsonl: {len(splits[split])} row(s)")
        else:
            f.warn("splits", f"{split}.jsonl is missing", str(p))

    dupes = [k for k, n in Counter(r.get("audio_filepath") for r in full).items() if n > 1]
    if dupes:
        f.error("structure", "duplicate audio_filepath in full.jsonl",
                f"{len(dupes)} path(s), e.g. {dupes[:3]}")

    if "structure" in checks:
        check_structure(full, "full.jsonl", f)
    if "files" in checks:
        check_files(full, "full.jsonl", manifest_dir, f,
                    tolerance=duration_tolerance, scan_strays=not no_strays)
    if "scores" in checks:
        check_scores(full, "full.jsonl", f, sample_percent=sample_percent, seed=seed)
    if "splits" in checks and splits:
        check_splits(full, splits, f)
    if "coverage" in checks and splits:
        check_coverage(splits, f)

    if report and f.rows:
        with report.open("w") as fh:
            for row in f.rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"wrote {len(f.rows)} offending row(s) to {report}")

    failed = f.render()
    total_err = sum(n for k, n in f.counts.items() if k in f.errors)
    total_warn = sum(n for k, n in f.counts.items() if k in f.warnings)
    logger.info(f"{total_err} error row(s), {total_warn} warning row(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
