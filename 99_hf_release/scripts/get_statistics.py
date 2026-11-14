#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""Report Markdown statistics over the NeMo manifests built by stage 99.

Reads the `{train,val,test}.jsonl` manifests under `data/NeMO_files/` and prints,
as Markdown, the character / word / duration distribution, file & duration counts,
and on-disk audio **size** (MB/GB), broken down **per split**, **per dataset**, and
**per dataset x split**, plus, when the rows carry a `voice` field (v2 audio),
**per voice**, **per voice x dataset** and **per voice x split**.

Two sources (the same rows, partitioned differently):

* `--source per-dataset` (default): each dataset's own split under
  `<root>/<dataset>/{train,val,test}.jsonl` (dataset = subfolder name). "By split"
  is then the concatenation of every dataset's split.
* `--source combined`: the release-wide global re-split at `<root>/{train,val,test}.jsonl`
  (dataset = each row's `category`). Use this to describe the actual combined
  train/val/test the model trains on.

Per-dataset character/word/duration stats are identical under either source (a
dataset's row set is the same); only the split-level breakdown differs.

    uv run scripts/get_statistics.py                     # Markdown to stdout
    uv run scripts/get_statistics.py --source combined
    uv run scripts/get_statistics.py --output stats.md
    uv run scripts/get_statistics.py --no-sizes          # skip on-disk size probe

Stdlib only (`statistics`, `os`, `concurrent.futures`). Durations come from the
manifest; the audio itself is never decoded. Sizes are read with `os.stat` on the
resolved (relative) `audio_filepath` and cached in `<root>/.size_cache.json`, so
`--sizes` needs the clips present; `--no-sizes` drops the size columns. Written
with Claude Code.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parent
# The truncation audit is shared with 06_hotfixes/02_statistics.py, so it lives in utils/.
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "utils"))
from nemo_manifest import CEILING_TOL_S, duration_ceiling  # noqa: E402

DEFAULT_ROOT = SCRIPT_DIR.parent / "data" / "NeMO_files"
SPLITS = ("train", "val", "test")
# Duration histogram edges (seconds); last bucket is the open-ended tail.
DUR_EDGES = [0, 1, 2, 3, 5, 8, 12, 20, 40, 80, float("inf")]
DEFAULT_JOBS = 16
# Bucket for rows with no `voice` field, i.e. v1 manifests (single voice, never
# recorded). Kept out of the per-voice sections rather than renamed to a guess.
UNRECORDED_VOICE = "(unrecorded)"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _iter_rows(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _triple(row: dict) -> tuple[float, int, int]:
    """(duration_seconds, n_chars, n_words) for one manifest row."""
    text = row.get("text", "") or ""
    return float(row.get("duration") or 0.0), len(text), len(text.split())


def _abspath(base: Path, row: dict) -> str | None:
    """Resolve a row's (relative) audio_filepath against its manifest dir."""
    ap = row.get("audio_filepath")
    if not ap:
        return None
    return os.path.normpath(os.path.join(base, ap))


def _voice(row: dict) -> str:
    """The voice that read this clip, or UNRECORDED_VOICE when the row predates it.

    v1 manifests have no `voice` field at all (one voice read the whole corpus and
    nothing recorded which), so a report over v1 audio must not invent a name: those
    rows land in one bucket and the per-voice sections are skipped entirely."""
    v = row.get("voice")
    return v if isinstance(v, str) and v else UNRECORDED_VOICE


def _iter_records(root: Path, source: str):
    """Yield (dataset, split, voice, group_id_or_None, abspath_or_None, dur, nchars,
    nwords) per row.

    The manifest file's own directory is the base for resolving the row's
    relative audio_filepath (per-dataset paths are relative to the subfolder,
    combined paths to the root)."""
    if source == "combined":
        for split in SPLITS:
            f = root / f"{split}.jsonl"
            if not f.is_file():
                continue
            for r in _iter_rows(f):
                yield (r.get("category", "?"), split, _voice(r), r.get("group_id"),
                       _abspath(f.parent, r), *_triple(r))
    else:  # per-dataset
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for split in SPLITS:
                f = sub / f"{split}.jsonl"
                if not f.is_file():
                    continue
                for r in _iter_rows(f):
                    yield (sub.name, split, _voice(r), r.get("group_id"),
                           _abspath(f.parent, r), *_triple(r))


def probe_sizes(paths: set[str], cache_path: Path, jobs: int) -> dict[str, int | None]:
    """Byte size per audio path, cached in ``cache_path`` (filename -> bytes).

    Missing/unreadable files map to ``None`` and are not cached (cheap to retry).
    The cache lives under the untracked ``data/`` tree, so absolute keys are fine."""
    cache: dict[str, int] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    todo = [p for p in paths if p not in cache]
    if todo:
        def _stat(p: str):
            try:
                return p, os.path.getsize(p)
            except OSError:
                return p, None

        with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            for p, sz in ex.map(_stat, todo):
                if sz is not None:
                    cache[p] = sz
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass  # caching is best-effort; do not fail the report over it

    return {p: cache.get(p) for p in paths}


def load_cells(root: Path, source: str, show_sizes: bool, jobs: int):
    """Return (cells, groups, missing).

    `cells` maps `(dataset, split, voice)` to a list of `(dur, nchars, nwords,
    size_bytes)`; `groups` maps `(dataset, voice)` to the set of `group_id`s seen
    there; `missing` counts clips with no readable file on disk.

    The voice is part of the cell key rather than a second parallel dict, so every
    breakdown below (per dataset, per split, per voice, and the crossings) reads the
    same cells through `select` and no two sections can disagree about a total.
    `groups` is separate because it is a set, not a list of quads: it answers how many
    distinct keys were hashed to produce the voice spread, which is what makes a large
    spread on a small dataset legible instead of alarming."""
    records = list(_iter_records(root, source))
    sizes: dict[str, int | None] = {}
    if show_sizes:
        uniq = {r[4] for r in records if r[4]}
        sizes = probe_sizes(uniq, root / ".size_cache.json", jobs)

    cells: dict[tuple[str, str, str], list] = {}
    groups: dict[tuple[str, str], set] = {}
    missing = 0
    for ds, split, voice, gid, ap, dur, nchars, nwords in records:
        if gid is not None:
            groups.setdefault((ds, voice), set()).add(gid)
        sz = 0
        if show_sizes:
            got = sizes.get(ap) if ap else None
            if got is None:
                missing += 1
            else:
                sz = got
        cells.setdefault((ds, split, voice), []).append((dur, nchars, nwords, sz))
    return cells, groups, missing


def select(cells: dict, *, dataset: str | None = None, split: str | None = None,
           voice: str | None = None) -> list:
    """Every quad whose cell matches the dimensions given; None means "any".

    One accessor for all the breakdowns, so a section cannot quietly filter on a
    different rule than its percentages are computed from."""
    return [t for (ds, sp, vc), v in cells.items()
            if (dataset is None or ds == dataset)
            and (split is None or sp == split)
            and (voice is None or vc == voice)
            for t in v]


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def describe(quads: list[tuple[float, int, int, int]]) -> dict | None:
    """min/mean/median/max of duration, chars, words + file/duration/byte totals."""
    if not quads:
        return None
    durs = [t[0] for t in quads]
    chars = [t[1] for t in quads]
    words = [t[2] for t in quads]

    def mmmm(xs):
        return (min(xs), statistics.mean(xs), statistics.median(xs), max(xs))

    return {
        "files": len(quads),
        "dur_total": sum(durs),
        "bytes_total": sum(t[3] for t in quads),
        "dur": mmmm(durs),
        "chars": mmmm(chars),
        "words": mmmm(words),
    }


def histogram(quads: list[tuple[float, int, int, int]]) -> list[dict]:
    """Per duration-bucket file count, summed duration, and summed byte size."""
    bins = [{"lo": DUR_EDGES[i], "hi": DUR_EDGES[i + 1], "files": 0, "dur": 0.0, "bytes": 0}
            for i in range(len(DUR_EDGES) - 1)]
    for dur, _, _, sz in quads:
        for b in bins:
            if b["lo"] <= dur < b["hi"]:
                b["files"] += 1
                b["dur"] += dur
                b["bytes"] += sz
                break
    return bins


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def _i(x) -> str:
    return f"{x:,.0f}"


def _h(sec: float) -> str:
    return f"{sec / 3600:.2f}"


def _size(nbytes: float) -> str:
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024 ** 3:.2f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes:.0f} B"


def _pct(part: float, whole: float) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "-"


def _spread(values) -> str:
    """`max - min` over a set of per-voice totals, as a share of their mean.

    The one number that says whether the voices are balanced. A share of the mean
    rather than of the max, so it reads the same whether the corpus has 10 clips per
    voice or 60,000, and the raw extremes are printed with it because a 50% spread
    over 116 to 194 clips (one small dataset) and over 300k clips mean very
    different things."""
    xs = list(values)
    if not xs:
        return "-"
    mean = sum(xs) / len(xs)
    if not mean:
        return "-"
    # Precision follows the data, because this formats both clip counts (integers in
    # the tens of thousands) and hours per voice (fractions, where ",.0f" printed the
    # useless "0 to 0").
    lo, hi = min(xs), max(xs)
    prec = 0 if all(float(x).is_integer() for x in xs) else (2 if hi < 10 else 1)
    return f"{100.0 * (hi - lo) / mean:.1f}% ({lo:,.{prec}f} to {hi:,.{prec}f})"


def _mmmm_int(t) -> str:
    return f"{t[0]:,.0f} / {t[1]:,.1f} / {t[2]:,.1f} / {t[3]:,.0f}"


def _mmmm_dur(t) -> str:
    return f"{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} / {t[3]:.2f}"


def _header(label: str, show_sizes: bool) -> str:
    cols = ["Files", "% files", "Dur (h)", "% dur"]
    if show_sizes:
        cols += ["Size", "% size"]
    cols += ["Chars (min/mean/med/max)", "Words (min/mean/med/max)", "Dur s (min/mean/med/max)"]
    return "| " + label + " | " + " | ".join(cols) + " |"


def _rule(show_sizes: bool) -> str:
    n = 1 + 4 + (2 if show_sizes else 0) + 3
    return "|" + "|".join(["---"] * n) + "|"


def _row(label: str, d: dict, tot_files: int, tot_dur: float, tot_bytes: int,
         show_sizes: bool) -> str:
    cells = [_i(d["files"]), _pct(d["files"], tot_files),
             _h(d["dur_total"]), _pct(d["dur_total"], tot_dur)]
    if show_sizes:
        cells += [_size(d["bytes_total"]), _pct(d["bytes_total"], tot_bytes)]
    cells += [_mmmm_int(d["chars"]), _mmmm_int(d["words"]), _mmmm_dur(d["dur"])]
    return "| " + label + " | " + " | ".join(cells) + " |"


def _voice_sections(cells: dict, groups: dict, voices: list[str],
                    datasets: list[str], grand: dict, show_sizes: bool) -> list[str]:
    """Render the per-voice breakdowns: totals, balance per dataset, and per split.

    Parameters
    ----------
    cells
        From `load_cells`, keyed `(dataset, split, voice)`.
    groups
        From `load_cells`, `(dataset, voice)` to the set of group ids there.
    voices
        The recorded voices, in name order; never includes UNRECORDED_VOICE.
    datasets
        Dataset names, in the order the rest of the report uses.
    grand
        `describe` of every row, for the shares of the grand total.
    show_sizes
        Whether the on-disk size columns are present.

    Returns
    -------
    list of str
        Markdown lines, appended to the report by `render`.

    Why these three tables and not one: the voice is a hash of each clip's group key
    (see utils/voice_assignment.py), so balance is approximate BY DESIGN, and "how
    approximate" is the only thing a reader needs to decide whether a model trained on
    this will favour a speaker. The overall table answers it for the corpus, the
    per-dataset one is where imbalance actually lives (a dataset with a few hundred
    groups cannot spread evenly over five voices, while the big ones are within a few
    percent), and the per-split one catches the case that would quietly bias
    evaluation: a voice over-represented in val/test.
    """
    tot_files, tot_dur = grand["files"], grand["dur_total"]
    tot_bytes = grand["bytes_total"]
    out: list[str] = []

    out.append("## By voice")
    out.append("")
    out.append("Who read what. The voice is a blake2b hash of each clip's group key "
               "(a term, or a source document), so every clip of a group shares a "
               "voice and the counts are only approximately equal: the hash buys edit "
               "and resume stability, see `utils/voice_assignment.py`. `Spread` below "
               "is `max - min` as a share of the mean.")
    out.append("")
    out.append(_header("Voice", show_sizes))
    out.append(_rule(show_sizes))
    per_voice = {v: describe(select(cells, voice=v)) for v in voices}
    for v in voices:
        out.append(_row(v, per_voice[v], tot_files, tot_dur, tot_bytes, show_sizes))
    out.append(_row("**all**", grand, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append("")
    out.append(f"- Spread, files: {_spread(d['files'] for d in per_voice.values())}")
    out.append(f"- Spread, hours: "
               f"{_spread(d['dur_total'] / 3600 for d in per_voice.values())}")
    out.append("")

    out.append("## Voice balance per dataset")
    out.append("")
    out.append("`Groups` counts the distinct `group_id`s the dataset was hashed on, "
               "which is the number the assignment actually divides: a dataset with a "
               "few hundred groups cannot land evenly on five voices, so a wide spread "
               "there is arithmetic, not a bug, while a wide spread on a dataset with "
               "tens of thousands of groups would mean the hash is not uniform. "
               "`Clips per voice` inherits that plus the variation in group size.")
    out.append("")
    out.append("| Dataset | Clips | Groups | Voices used | Groups per voice | "
               "Clips per voice | Hours per voice |")
    out.append("|---|---:|---:|---:|---|---|---|")
    for ds in datasets:
        used = [d for d in (describe(select(cells, dataset=ds, voice=v))
                            for v in voices) if d]
        ds_d = describe(select(cells, dataset=ds))
        g_sets = [groups.get((ds, v), set()) for v in voices]
        g_sizes = [len(s) for s in g_sets]
        # Union, not sum: in the default "group" mode a group has exactly one voice so
        # the two agree, but under `--voice-mode row` one group is spread over several
        # and summing would count it once per voice.
        n_groups = len(set().union(*g_sets)) if g_sets else 0
        out.append(f"| {ds} | {_i(ds_d['files'])} | "
                   f"{_i(n_groups) if n_groups else 'n/a'} | "
                   f"{len(used)}/{len(voices)} | "
                   f"{_spread(g for g in g_sizes if g) if n_groups else 'n/a'} | "
                   f"{_spread(d['files'] for d in used)} | "
                   f"{_spread(d['dur_total'] / 3600 for d in used)} |")
    out.append("")

    out.append("## By voice x split")
    out.append("")
    out.append("Shares **within each split**, so each column sums to 100%. A voice "
               "over-represented in val or test biases evaluation toward it, which no "
               "other table in this report would show.")
    out.append("")
    out.append("| Voice | " + " | ".join(f"{s} files | % of {s}" for s in SPLITS) + " |")
    out.append("|---" + "|---:" * (2 * len(SPLITS)) + "|")
    split_tot = {s: describe(select(cells, split=s)) for s in SPLITS}
    for v in voices:
        row = [v]
        for s in SPLITS:
            d = describe(select(cells, split=s, voice=v))
            tot = split_tot[s]
            row += [_i(d["files"]) if d else "0",
                    _pct(d["files"] if d else 0, tot["files"] if tot else 0)]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def render(cells: dict, groups: dict, root: Path, source: str, show_sizes: bool,
           missing: int) -> str:
    datasets = sorted({ds for ds, _, _ in cells})
    # Only the voices that were actually recorded: a v1 report (every row in the
    # UNRECORDED_VOICE bucket) then skips the per-voice sections instead of printing a
    # one-row table claiming the corpus has one voice called "(unrecorded)".
    voices = sorted({vc for _, _, vc in cells} - {UNRECORDED_VOICE})
    all_quads = [t for v in cells.values() for t in v]
    grand = describe(all_quads)
    tot_files, tot_dur, tot_bytes = grand["files"], grand["dur_total"], grand["bytes_total"]

    out: list[str] = []
    out.append("# NeMo manifest statistics")
    out.append("")
    out.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    out.append(f"- Root: `{root}`")
    out.append(f"- Source: **{source}** "
               f"({'per-dataset splits' if source != 'combined' else 'combined release re-split'})")
    out.append(f"- Datasets: {', '.join(datasets)}")
    if voices:
        out.append(f"- Voices: {len(voices)} ({', '.join(voices)})")
        unrec = describe(select(cells, voice=UNRECORDED_VOICE))
        if unrec:
            out.append(f"- Note: {_i(unrec['files'])} clip(s) carry no `voice` field "
                       f"(v1-era audio); they are in every total but in no voice row.")
    total_line = f"- Total: **{_i(tot_files)} files**, **{_h(tot_dur)} h** audio"
    if show_sizes:
        total_line += f", **{_size(tot_bytes)}** on disk"
    out.append(total_line)
    if show_sizes and missing:
        out.append(f"- Note: {_i(missing)} clip(s) had no readable file on disk; "
                   f"counted as 0 bytes in the size columns.")
    out.append("")
    out.append("Column groups `min / mean / median / max`. `Dur s` is per-clip "
               "duration in seconds; `Size` is on-disk audio; `% files` / `% dur` / "
               "`% size` are shares of the grand total unless noted.")
    out.append("")

    # By dataset
    out.append("## By dataset")
    out.append("")
    out.append(_header("Dataset", show_sizes))
    out.append(_rule(show_sizes))
    for ds in datasets:
        d = describe(select(cells, dataset=ds))
        out.append(_row(ds, d, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append(_row("**all**", grand, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append("")

    # Truncation audit: is any dataset's duration distribution sitting on a ceiling?
    out.append("## Duration ceiling check")
    out.append("")
    out.append("A TTS generation stopped by its token limit is cut off mid-sentence at "
               "exactly the cap, so truncated clips pile up on one duration instead of "
               "tailing off. `At max` counts the clips within one 12.5 Hz frame "
               f"({CEILING_TOL_S:.2f} s) of the longest one: more than a couple, and the "
               "dataset was generated against a cap, not to the end of its text.")
    out.append("")
    out.append("| Dataset | Longest clip | At max | Share | Verdict |")
    out.append("|---|---:|---:|---:|---|")
    capped_any = []
    for ds in datasets + ["**all**"]:
        quads = all_quads if ds == "**all**" else select(cells, dataset=ds)
        c = duration_ceiling([t[0] for t in quads])
        if not c:
            continue
        if c["capped"] and ds != "**all**":
            capped_any.append(ds)
        verdict = "**CAPPED, clips are truncated**" if c["capped"] else "no ceiling"
        out.append(f"| {ds} | {c['max']:.2f} s | {_i(c['at_max'])} | "
                   f"{c['share'] * 100:.3f}% | {verdict} |")
    out.append("")
    if capped_any:
        out.append(f"> **Warning:** {', '.join(capped_any)} shows a duration ceiling. Those "
                   "clips stopped for length, not at the end of their text, so their audio "
                   "does not match their transcript. Raise the server's max_new_tokens or "
                   "shorten the source text, then regenerate them.")
        out.append("")

    # By split
    out.append("## By split")
    out.append("")
    out.append(_header("Split", show_sizes))
    out.append(_rule(show_sizes))
    for split in SPLITS:
        d = describe(select(cells, split=split))
        if d:
            out.append(_row(split, d, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append(_row("**all**", grand, tot_files, tot_dur, tot_bytes, show_sizes))
    out.append("")

    # By dataset x split (percentages are within each dataset, so splits sum to 100%)
    out.append("## By dataset x split")
    out.append("")
    out.append("`% files` / `% dur` / `% size` are shares **within the dataset** (so the "
               "three splits sum to 100%), which shows each dataset's split ratio.")
    out.append("")
    out.append(_header("Dataset / split", show_sizes))
    out.append(_rule(show_sizes))
    for ds in datasets:
        ds_d = describe(select(cells, dataset=ds))
        for split in SPLITS:
            d = describe(select(cells, dataset=ds, split=split))
            if not d:
                continue
            out.append(_row(f"{ds} / {split}", d, ds_d["files"], ds_d["dur_total"],
                            ds_d["bytes_total"], show_sizes))
    out.append("")

    if voices:
        out += _voice_sections(cells, groups, voices, datasets, grand, show_sizes)

    # Duration distribution (histogram)
    out.append("## Duration distribution")
    out.append("")
    hcols = ["Bucket (s)", "Files", "% files", "Dur (h)", "% dur"]
    if show_sizes:
        hcols += ["Size", "% size"]
    out.append("| " + " | ".join(hcols) + " |")
    out.append("|" + "|".join(["---"] * len(hcols)) + "|")
    for b in histogram(all_quads):
        hi = "inf" if b["hi"] == float("inf") else f"{b['hi']:g}"
        label = f"{b['lo']:g}-{hi}"
        cells_h = [label, _i(b["files"]), _pct(b["files"], tot_files),
                   _h(b["dur"]), _pct(b["dur"], tot_dur)]
        if show_sizes:
            cells_h += [_size(b["bytes"]), _pct(b["bytes"], tot_bytes)]
        out.append("| " + " | ".join(cells_h) + " |")
    out.append("")

    return "\n".join(out)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--root", type=click.Path(path_type=Path), default=DEFAULT_ROOT,
              show_default=True, help="Folder holding the NeMo manifests.")
@click.option("--source", type=click.Choice(["per-dataset", "combined"]), default="per-dataset",
              show_default=True, help="Which split files to read (see --help).")
@click.option("--sizes/--no-sizes", "show_sizes", default=True, show_default=True,
              help="Probe on-disk audio size (needs the clips present).")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True,
              help="Threads for the size probe.")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Write the Markdown here instead of stdout.")
def main(root: Path, source: str, show_sizes: bool, jobs: int, output: Path | None):
    root = root.resolve()
    if not root.is_dir():
        raise click.UsageError(f"root not found: {root} (run 01_build_nemo_manifest.py first)")
    cells, groups, missing = load_cells(root, source, show_sizes, jobs)
    if not cells:
        raise click.UsageError(
            f"no {'/'.join(SPLITS)} manifests found under {root} for source={source!r}; "
            f"run 01_build_nemo_manifest.py (per-dataset) or 02_combine_nemo_manifests.py (combined)")
    md = render(cells, groups, root, source, show_sizes, missing)
    if output:
        Path(output).write_text(md, encoding="utf-8")
        click.echo(f"wrote {output}", err=True)
    else:
        click.echo(md)


if __name__ == "__main__":
    main()
