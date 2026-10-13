"""Shared NeMo-manifest assembly for the ``99_hf_release`` stage.

Turns each text stage's ``generated_dataset.jsonl`` plus its synthesized ``.flac``
clips into NeMo ASR manifests (one JSON object per line with ``audio_filepath``,
``duration``, ``text``, plus grouping provenance) and owns the grouped,
duration-aware train/val/test splitter that both stage-99 scripts share:

* ``01_build_nemo_manifest.py`` (per-dataset builder, one subfolder each), and
* ``02_combine_nemo_manifests.py`` (parent combiner, global re-split).

Kept in ``utils/`` so neither script forks the split policy. The split logic and
the small helpers (``parse_split``, ``assign_splits``, ``summarize``,
``relativize``, jsonl io) are stdlib-only *at import time*: ``soundfile`` and
``tqdm`` are imported lazily inside the audio-probing helpers, so the unit tests
can import and exercise the splitter without those wheels.

Two group behaviours, chosen per dataset by the builder:

* **distribute** (dictionary / drugs): the N ``asr_training_target`` variants of a
  term are spread across splits. A group of >=2 keeps at least one variant in
  train; under strict coverage a group of >=3 also keeps at least one in *every*
  enabled split, so each split sees the term. This is augmentation-style data,
  spreading it is wanted. The per-split coverage can be made best-effort (see
  ``assign_splits(strict_coverage=False)``) so val/test hit their target ratio.
* **atomic** (PARHAF / PARROT): every chunk of one source document (``source_id``)
  goes to the *same* split, so no clinical document leaks across train/test.

Singletons (a one-variant term, a one-chunk document) are placed by ratio like any
other free item. The greedy placement fills the currently most under-target split,
weighted by clip duration when ``by_duration`` is set (duration is the proxy for
"amount of speech / tokens" the split should balance), else by row count. The whole
procedure is deterministic (no RNG): same inputs give the same split.

Written with Claude Code.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

log = logging.getLogger("nemo_manifest")

# Fixed split order; also the deterministic tiebreak when two splits are equally
# under-target. train first so an all-empty tie seeds train.
SPLITS: tuple[str, ...] = ("train", "val", "test")

# A NeMo training row reads only these three; the rest are provenance the parent
# combiner re-uses to re-split without re-reading the source datasets.
NEMO_CORE = ("audio_filepath", "duration", "text")
# Every manifest the release stage (99_hf_release/03 to 06) reads or rewrites, top level
# and per dataset, relative to the stage root.
MANIFEST_GLOBS = ("data/NeMO_files/*.jsonl", "data/NeMO_files/*/*.jsonl")


# --------------------------------------------------------------------------- #
# split-spec parsing
# --------------------------------------------------------------------------- #
def parse_split(spec: str) -> dict[str, float]:
    """Parse a ``"train/val/test"`` spec into normalised fractions.

    Accepts percentages or fractions in any scale (``"80/10/10"``,
    ``"8/1/1"``, ``"0.8/0.1/0.1"`` all give the same result); the three values
    are divided by their sum. A value may be ``0`` to disable that split
    (e.g. ``"90/10/0"`` for train/val only). Raises ``ValueError`` on anything
    that is not three non-negative numbers summing to > 0.
    """
    parts = [p for p in spec.replace(",", "/").split("/") if p.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"--split needs 3 values train/val/test, got {spec!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--split values must be numbers, got {spec!r}") from exc
    if any(v < 0 for v in vals):
        raise ValueError(f"--split values must be >= 0, got {spec!r}")
    total = sum(vals)
    if total <= 0:
        raise ValueError(f"--split values must sum to > 0, got {spec!r}")
    return {s: v / total for s, v in zip(SPLITS, vals)}


# --------------------------------------------------------------------------- #
# the grouped, duration-aware splitter
# --------------------------------------------------------------------------- #
def assign_splits(
    rows: Sequence[dict],
    ratios: dict[str, float],
    *,
    stratify: bool = True,
    by_duration: bool = True,
    strict_coverage: bool = True,
) -> list[str]:
    """Assign each row to ``"train"`` / ``"val"`` / ``"test"``.

    ``rows`` is a sequence of dicts; each must carry ``group_id`` (hashable),
    ``group_mode`` (``"atomic"`` or ``"distribute"``) and ``duration`` (float).
    Returns a list of split labels parallel to ``rows``.

    * ``stratify=False`` ignores groups entirely: every row is placed
      independently by ratio (still duration-weighted when ``by_duration``).
    * ``stratify=True`` honours the per-group guarantees described in the module
      docstring: atomic groups stay whole in one split; distribute groups of >=2
      keep >=1 in train.
    * ``strict_coverage=True`` additionally forces a distribute group of >=3 to
      put >=1 in *every* enabled split. This guarantees each split sees the term,
      but floors val/test at one sample per such group, so a dataset with many
      small term-groups overshoots the target ratio (e.g. val/test ~15% instead
      of 10%). ``strict_coverage=False`` makes that per-split coverage
      best-effort: only >=1-in-train is kept as a hard guarantee and the rest of
      the group is placed greedily, so val/test converge on their target ratio
      (big groups still tend to span all splits naturally). ``by_duration`` and
      the atomic guarantee are unaffected.

    Deterministic: groups are processed largest-weight first (a
    longest-processing-time greedy that balances well), ties broken by
    ``str(group_id)``; within a group items are processed largest-first, ties
    broken by original position.
    """
    enabled = [s for s in SPLITS if ratios[s] > 0]
    if not enabled:  # parse_split guarantees at least one, but stay defensive
        raise ValueError("no split has a positive ratio")

    totals: dict[str, float] = {s: 0.0 for s in SPLITS}

    def weight(i: int) -> float:
        return float(rows[i]["duration"]) if by_duration else 1.0

    def pick(candidates: Iterable[str]) -> str:
        # Most under-target = smallest achieved/target ratio; SPLITS order breaks
        # ties (so the first placements seed train, then val, then test).
        return min(candidates, key=lambda s: (totals[s] / ratios[s], SPLITS.index(s)))

    labels: list[str | None] = [None] * len(rows)

    def place(i: int, s: str) -> None:
        totals[s] += weight(i)
        labels[i] = s

    if not stratify:
        # Every row independent; largest-first for balance.
        order = sorted(range(len(rows)), key=lambda i: (-weight(i), i))
        for i in order:
            place(i, pick(enabled))
        return [s for s in labels]  # type: ignore[misc]

    # Bucket rows into their groups (insertion order preserved for the tiebreak).
    groups: dict[object, list[int]] = {}
    modes: dict[object, str] = {}
    for i, row in enumerate(rows):
        gid = row["group_id"]
        groups.setdefault(gid, []).append(i)
        modes[gid] = row.get("group_mode", "distribute")

    def group_weight(idxs: list[int]) -> float:
        return sum(weight(i) for i in idxs)

    # Largest groups first (LPT); str(gid) tiebreak keeps it deterministic.
    ordered_gids = sorted(groups, key=lambda g: (-group_weight(groups[g]), str(g)))

    for gid in ordered_gids:
        idxs = sorted(groups[gid], key=lambda i: (-weight(i), i))
        mode = modes[gid]

        if len(idxs) == 1:
            place(idxs[0], pick(enabled))
            continue

        if mode == "atomic":
            # Whole document to one split (no chunk leakage across splits).
            s = pick(enabled)
            for i in idxs:
                place(i, s)
            continue

        # distribute, len >= 2
        if len(enabled) == 1:
            for i in idxs:
                place(i, enabled[0])
            continue

        # Coverage guarantee: >=1 in train always; and, under strict_coverage,
        # >=1 in every enabled split when the group has >=3 items. Assign the
        # largest items to the required splits, each to whichever required split
        # is currently most under-target. With strict_coverage off, only train is
        # reserved and the rest fall through to the greedy fill, so val/test are
        # not floored above their target ratio.
        cover_all = strict_coverage and len(idxs) >= 3
        required = list(enabled) if cover_all else (
            ["train"] if "train" in enabled else [enabled[0]]
        )
        covered = 0
        to_cover = list(required)
        for i in idxs:
            if not to_cover:
                break
            s = pick(to_cover)
            place(i, s)
            to_cover.remove(s)
            covered += 1
        for i in idxs[covered:]:
            place(i, pick(enabled))

    return [s for s in labels]  # type: ignore[misc]


def summarize(rows: Sequence[dict], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-split ``{count, duration, count_pct, duration_pct}`` for logging."""
    agg = {s: {"count": 0, "duration": 0.0} for s in SPLITS}
    for row, s in zip(rows, labels):
        agg[s]["count"] += 1
        agg[s]["duration"] += float(row.get("duration") or 0.0)
    tot_c = sum(a["count"] for a in agg.values()) or 1
    tot_d = sum(a["duration"] for a in agg.values()) or 1.0
    for a in agg.values():
        a["count_pct"] = 100.0 * a["count"] / tot_c
        a["duration_pct"] = 100.0 * a["duration"] / tot_d
    return agg


# A TTS backend that stops on its token limit produces clips cut off at exactly the
# same length, because the limit is a frame count: Voxtral emits one audio frame every
# 1/12.5 s, so a truncated clip lands on the cap to the frame. The signature is a PILE-UP
# at the longest duration, where a healthy distribution tails off with a single clip at
# its max. This is the audit that would have caught the 163.84 s truncation, and it lives
# here so both the release statistics (99_hf_release/scripts/get_statistics.py) and the
# hotfix statistics (06_hotfixes/02_statistics.py) ask the question the same way.
CEILING_TOL_S = 0.08          # one Voxtral frame
CEILING_WARN_SHARE = 0.002    # 0.2% of a dataset sitting on its own maximum is not chance


def duration_ceiling(durs: Sequence[float]) -> dict | None:
    """Look for a hard cap at the top of a duration distribution. Returns the longest
    clip, how many clips sit within one frame of it, and whether that pile-up is big
    enough to mean a cap rather than coincidence."""
    if not durs:
        return None
    top = max(durs)
    at_top = sum(1 for d in durs if d >= top - CEILING_TOL_S)
    share = at_top / len(durs)
    return {"max": top, "at_max": at_top, "share": share,
            "capped": at_top > 1 and share >= CEILING_WARN_SHARE}


def format_summary(agg: dict[str, dict[str, float]]) -> str:
    """One-line-per-split human summary of a ``summarize`` result."""
    lines = []
    for s in SPLITS:
        a = agg[s]
        lines.append(
            f"  {s:5s}: {a['count']:>8d} rows ({a['count_pct']:5.1f}%)  "
            f"{a['duration'] / 3600.0:8.2f} h ({a['duration_pct']:5.1f}%)"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# pinning a split to an earlier generation of the audio
# --------------------------------------------------------------------------- #
# Pull just the one field out of a manifest line. Pinning reads 2+ GB of v1
# manifests and needs ONE key per row, so parsing every row into a dict (with its
# text, transcript and QC columns) would dominate the build for nothing.
_AUDIO_FIELD = re.compile(r'"audio_filepath"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _audio_filepath(line: str) -> str | None:
    """``audio_filepath`` of a manifest line, or None for a blank/odd line."""
    m = _AUDIO_FIELD.search(line)
    if m:
        return m.group(1)
    if not line.strip():
        return None
    try:  # a value with escapes the regex gave up on: pay for the full parse
        return json.loads(line).get("audio_filepath")
    except json.JSONDecodeError:
        return None


def load_pinned_splits(root: Path) -> dict[str, str]:
    """``{clip_key: split}`` read from an EARLIER generation's manifests.

    Why this exists: ``assign_splits`` is duration-greedy, so re-synthesizing an
    unchanged corpus (v2 reads every clip with a new voice) yields different
    durations and therefore a different train/val/test. Any comparison between the
    two generations then leaks, in the direction that flatters the new one: a
    sentence the old test set measured can land in the new train set, so the new
    model is scored on text it was trained on. Pinning each clip to the split it
    already had keeps the two generations comparable.

    **Only the split label travels.** Nothing else of the old row is read, so no
    stale duration, CER, transcript or ``qc_status`` can ride along; those describe
    audio that no longer exists, which is exactly what the rest of stage 99 is
    careful about.

    Both levels are scanned: ``<root>/{train,val,test}.jsonl`` (the release-wide
    split, what the trainer reads) and ``<root>/<dataset>/{train,val,test}.jsonl``
    (per dataset). The release-wide label wins where the two disagree, and the
    disagreements are counted in a warning, since ``02_combine_nemo_manifests.py``
    syncs the two levels and a synced tree has none.

    Parameters
    ----------
    root : Path
        Manifest root of the previous generation (v1's ``NeMO_files/``).

    Returns
    -------
    dict[str, str]
        ``clip_key`` (``<source folder>/<file>``) to split name. The key is a folder
        plus a file name, not an absolute path, so it still matches after the audio
        has moved to another root, which is what v2 did.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"no manifest root to pin splits from: {root}")
    pins: dict[str, str] = {}

    def scan(manifest: Path, split: str) -> int:
        n = 0
        with manifest.open() as f:
            for line in f:
                rel = _audio_filepath(line)
                if rel:
                    pins[clip_key(resolve_audio(manifest.parent, rel))] = split
                    n += 1
        return n

    # Per dataset first, then release-wide on top, so the release-wide label wins.
    per_dataset = 0
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        for split in SPLITS:
            manifest = sub / f"{split}.jsonl"
            if manifest.is_file():
                per_dataset += scan(manifest, split)
    before = dict(pins)
    release = 0
    for split in SPLITS:
        manifest = root / f"{split}.jsonl"
        if manifest.is_file():
            release += scan(manifest, split)
    if not pins:
        raise FileNotFoundError(f"{root} holds no train/val/test.jsonl to pin splits from")
    conflicts = sum(1 for k, v in pins.items() if k in before and before[k] != v)
    if conflicts:
        log.warning("pinned splits: %d clips sat in a different split per dataset than "
                    "release-wide in %s; the release-wide one wins", conflicts, root)
    log.info("pinned splits: %d clips from %s (%d per-dataset rows, %d release-wide rows)",
             len(pins), root, per_dataset, release)
    return pins


def assign_splits_pinned(
    rows: Sequence[dict],
    ratios: dict[str, float],
    pins: Mapping[str, str],
    *,
    key: Callable[[dict], str],
    stratify: bool = True,
    by_duration: bool = True,
    strict_coverage: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """Like ``assign_splits``, but keep the split an earlier generation gave a clip.

    Resolution order per row:

    1. the clip's own pin (``pins[key(row)]``);
    2. failing that, the split its GROUP's pinned clips are in. A row the old
       generation did not have (a restored clip, a newly added variant) must not be
       placed on its own: that could put it in another split than the rest of its
       term, breaking the atomic-group guarantee the splitter exists to make. A
       ``distribute`` group that legitimately spans splits contributes several
       votes and the largest wins, ties going to the earliest of ``SPLITS``;
    3. failing that, ``assign_splits`` over the remaining rows alone, which keeps
       the usual group guarantees among them and gives them the ordinary ratios.

    Parameters
    ----------
    rows, ratios, stratify, by_duration, strict_coverage
        As ``assign_splits``; the last three apply to step 3 only.
    pins : Mapping[str, str]
        From ``load_pinned_splits``.
    key : Callable[[dict], str]
        Row to ``clip_key``. The caller owns it because the two stage-99 scripts
        hold the audio path differently (``_audio_abs`` vs an ``audio_filepath``
        relative to the manifest).

    Returns
    -------
    labels : list[str]
        Split per row, parallel to ``rows``.
    stats : dict[str, int]
        ``{"clip": pinned by their own key, "group": pinned by their group,
        "fresh": split by assign_splits}``. Worth logging: a build meant to
        reproduce a previous split that reports a large ``fresh`` is pinning
        against the wrong root, or against a corpus whose file names moved.
    """
    labels: list[str | None] = [None] * len(rows)
    votes: dict[object, Counter] = defaultdict(Counter)
    for i, row in enumerate(rows):
        pin = pins.get(key(row))
        if pin in SPLITS:
            labels[i] = pin
            votes[row.get("group_id")][pin] += 1
    n_clip = sum(1 for lab in labels if lab is not None)

    n_group = 0
    for i, row in enumerate(rows):
        if labels[i] is not None:
            continue
        group = votes.get(row.get("group_id"))
        if group:
            labels[i] = min(group.items(), key=lambda kv: (-kv[1], SPLITS.index(kv[0])))[0]
            n_group += 1

    todo = [i for i, lab in enumerate(labels) if lab is None]
    if todo:
        fresh = assign_splits([rows[i] for i in todo], ratios, stratify=stratify,
                              by_duration=by_duration, strict_coverage=strict_coverage)
        for i, lab in zip(todo, fresh):
            labels[i] = lab
    return [lab for lab in labels], {"clip": n_clip, "group": n_group, "fresh": len(todo)}  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# audio index + duration probing
# --------------------------------------------------------------------------- #
def _stem_key(term_index: int, variant_index: int) -> str:
    """The stage-05 filename prefix, zero-padded exactly as
    ``05_generate_audio/01_generate_audio.py`` writes it. Six/four digits make
    the prefix fixed-width, so it can never be ambiguous between rows."""
    return f"{int(term_index):06d}_{int(variant_index):04d}"


def parse_clip_name(name: str) -> tuple[int, int, str] | None:
    """Split a stage-05 clip filename into ``(term_index, variant_index, slug)``.

    Stage 05 names every clip ``{term_index:06d}_{variant_index:04d}_{slug}.flac`` (see
    ``05_generate_audio/01_generate_audio.py``), which makes the name the only per-clip
    identity a NeMo manifest row carries: the manifest keeps ``group_id`` / ``item_index``
    and drops the source term. Returns ``None`` for anything that is not a clip (a
    ``*.flac.tmp`` leftover, a name without the two zero-padded index fields).
    """
    if not name.endswith(".flac"):
        return None
    parts = name[: -len(".flac")].split("_", 2)
    if len(parts) < 2 or not (
        len(parts[0]) == 6 and parts[0].isdigit()
        and len(parts[1]) == 4 and parts[1].isdigit()
    ):
        return None
    return int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""


def audio_index(audio_dir: Path) -> dict[str, Path]:
    """Scan ``audio_dir`` once and map each clip's ``termidx_varidx`` prefix to
    its path, so a row is matched by its two indices rather than by re-deriving
    the term slug. A leftover ``*.tmp`` or a name without the two leading
    zero-padded index fields is skipped. First file wins on a duplicate prefix
    (warned)."""
    index: dict[str, Path] = {}
    for entry in os.scandir(audio_dir):
        name = entry.name
        # Only real clips: this also skips a "*.flac.tmp" leftover, a stray
        # sub-symlink (e.g. a self-referential loop) and dotfolders, without ever
        # stat-ing them (following a symlink loop would raise OSError).
        parsed = parse_clip_name(name)
        if parsed is None:
            continue
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue  # broken/looping symlink named *.flac
        key = f"{parsed[0]:06d}_{parsed[1]:04d}"
        if key in index:
            log.warning("duplicate audio prefix %s: %s and %s", key, index[key].name, entry.name)
            continue
        index[key] = Path(entry.path)
    return index


def probe_durations(
    paths: Sequence[Path],
    cache_path: Path | None = None,
    jobs: int = 8,
    progress: bool = True,
) -> dict[str, float]:
    """Return ``{filename: duration_seconds}`` for ``paths``.

    Durations are read from the FLAC header only (``soundfile.info``), never a
    full decode. Results are cached in ``cache_path`` as JSON, one
    ``name: [duration_seconds, size_bytes, mtime_ns]`` entry per clip: still
    keyed by *file name* (location-independent, so the cache survives the audio
    folder being moved or re-symlinked) but each hit is **validated against the
    file's current size and mtime**, and only reused for bytes that still look
    like the ones that were probed.

    That validation is the point. The same file name carries different audio in
    two routine situations: stage 06 regenerates a bad clip by overwriting it in
    place (duration deltas up to ~115 s observed, enough to cross the trainer's
    ``max_duration``), and a new generation of the dataset re-synthesizes the
    whole corpus under the very same names. A name-only cache answered both with
    a silently wrong duration. A stale entry, or a legacy duration-only one
    written before the stamp existed, is simply re-probed and counted in a
    warning. Corrupt/unreadable clips are logged and omitted from the result.
    """
    import soundfile as sf  # lazy: keeps the module importable without the wheel

    cache: dict[str, list] = {}
    if cache_path and cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("could not read duration cache %s, reprobing", cache_path)
            raw = {}
        # A bare float is a pre-stamp entry: unverifiable, so it is dropped
        # rather than trusted (it may well describe an older generation's audio).
        cache = {k: v for k, v in raw.items() if isinstance(v, list) and len(v) == 3}
        legacy = len(raw) - len(cache)
        if legacy:
            log.warning("duration cache %s: %d entries carry no size/mtime stamp, reprobing them",
                        cache_path, legacy)

    # One stat per clip (no decode, microseconds each) is what lets a hit be
    # verified; it is the whole cost of the safety above.
    stamps: dict[Path, tuple[int, int] | None] = {}
    for p in paths:
        try:
            st = p.stat()
            stamps[p] = (st.st_size, st.st_mtime_ns)
        except OSError:
            stamps[p] = None  # missing/broken clip: let the probe report it

    out: dict[str, float] = {}
    todo: list[Path] = []
    stale = 0
    for p in paths:
        hit = cache.get(p.name)
        stamp = stamps[p]
        if hit is not None and stamp is not None and (int(hit[1]), int(hit[2])) == stamp:
            out[p.name] = float(hit[0])
            continue
        if hit is not None:
            stale += 1
            cache.pop(p.name, None)
        todo.append(p)
    if stale:
        log.warning("duration cache %s: %d of %d clips changed on disk since they were "
                    "probed, reprobing them", cache_path, stale, len(paths))

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def one(p: Path) -> tuple[str, float | None]:
            try:
                info = sf.info(str(p))
                return p.name, (info.frames / info.samplerate if info.samplerate else None)
            except Exception as exc:  # noqa: BLE001 - any decode error drops the clip
                log.warning("cannot read duration of %s: %s", p.name, exc)
                return p.name, None

        bar = None
        if progress:
            try:
                from tqdm import tqdm
                bar = tqdm(total=len(todo), unit="clip", desc="probing durations")
            except ImportError:
                bar = None
        by_name = {p.name: p for p in todo}
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for fut in as_completed(pool.submit(one, p) for p in todo):
                name, dur = fut.result()
                stamp = stamps[by_name[name]]
                if dur is not None:
                    out[name] = dur
                    # No stamp means the clip could not be stat-ed, so there is
                    # nothing to validate a future hit against: do not cache it.
                    if stamp is not None:
                        cache[name] = [dur, stamp[0], stamp[1]]
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(cache_path)

    return out


# --------------------------------------------------------------------------- #
# path handling + jsonl io
# --------------------------------------------------------------------------- #
def relativize(audio_abs: str | Path, manifest_dir: Path) -> str:
    """POSIX path of ``audio_abs`` relative to the directory the manifest lives
    in. Relative (not absolute) on purpose: absolute paths under the audio SSD
    embed the machine's home/username, which must never be baked into a file
    that could be committed or published."""
    return Path(os.path.relpath(Path(audio_abs), manifest_dir)).as_posix()


def resolve_audio(manifest_dir: Path, rel: str) -> str:
    """A row's audio path as one absolute string, so manifests that store it relative to
    different directories still join onto each other. Pure string work (no
    ``Path.resolve``): it runs on every row of every manifest, and touching the SSD
    600k times per manifest would dominate the run."""
    return os.path.normpath(os.path.join(manifest_dir, rel))


def clip_key(path: str) -> str:
    """``<source folder>/<file>`` of an audio path (``dictionary/012095_0006_x.flac``):
    unique across the corpus (stage-05 names are unique within a source) and free of the
    machine's absolute path, so it can key a committed list (06's ASR-flagged clips)."""
    return "/".join(os.path.normpath(path).split(os.sep)[-2:])


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write ``rows`` as JSONL (utf-8, non-ascii kept). Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    tmp.replace(path)
    return n


def write_splits(
    out_dir: Path,
    rows: Sequence[dict],
    labels: Sequence[str],
    manifest_dir: Path | None = None,
    absolute: bool = False,
    audio_key: str = "_audio_abs",
) -> dict[str, int]:
    """Write ``full/train/val/test.jsonl`` into ``out_dir``.

    Each row is emitted with only the NeMo fields plus provenance, and its
    ``audio_filepath`` set from ``row[audio_key]`` (an absolute path kept
    internally): relative to ``manifest_dir`` (default ``out_dir``) unless
    ``absolute``. Returns ``{split_or_full: count}``.
    """
    manifest_dir = manifest_dir or out_dir

    def emit(row: dict) -> dict:
        audio_abs = row[audio_key]
        out = {
            "audio_filepath": str(audio_abs) if absolute else relativize(audio_abs, manifest_dir),
            "duration": round(float(row["duration"]), 3),
            "text": row["text"],
        }
        # The deterministic TTS input (asr_training_source) alongside the label
        # (text): passed through when present so downstream artifacts (the Parquet
        # release) ship both. NeMo training ignores it.
        if "asr_training_source" in row:
            out["asr_training_source"] = row["asr_training_source"]
        # `voice` is which of the ten v2 voices read the clip (absent on v1 audio,
        # which is single-voice): passed through like the other provenance fields, so
        # the per-dataset manifests, the combined ones and the Parquet all carry it.
        for k in ("category", "group_id", "group_mode", "item_index", "voice"):
            if k in row:
                out[k] = row[k]
        return out

    counts: dict[str, int] = {}
    counts["full"] = write_jsonl(out_dir / "full.jsonl", (emit(r) for r in rows))
    for s in SPLITS:
        sel = [emit(r) for r, lab in zip(rows, labels) if lab == s]
        counts[s] = write_jsonl(out_dir / f"{s}.jsonl", sel)
    return counts
