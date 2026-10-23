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
# ]
# ///
"""Delete the TTS candidate sidecars (``<clip>.draft<seed>.<fmt>``) nothing needs any more.

The alternating improvement loop already cleans up after itself on the happy path:
``01_recursive_improvement.py`` deletes every candidate of a clip the moment an ``stt``
pass judges it (``apply_result``, win or lose), and clears the leftovers of a clip
before redrawing it (``clear_drafts``). So this script is NOT part of the normal cycle.
It exists for the two cases that path cannot reach:

* **An interrupted or abandoned round.** A ``tts`` pass that was stopped (or whose
  dataset later converged by another route) leaves a full set of candidates on disk
  that no later pass will ever look at, because no pass visits a clip that is no
  longer waiting on anything.
* **A requeue under changed gates.** When the scoring rules move, a queue built under
  the old ones is dropped (see the "regeneration queue built under the OLD scoring
  rules is stale" branch), and the sidecars of those clips are orphaned the same way.

That garbage is not small: a candidate is a full clip, measured at ~0.84 MiB mean, and
a flagged clip gets up to ``--n-improv`` of them, so one round over 12.4k flagged clips
is ~49 GiB.

**What it keeps, and why that rule is the safe one.** A clip whose row says
``pending_stt`` is waiting to be judged, so its candidates ARE the work product of the
last round: every draft of such a clip is kept, including a file the row does not list
(a pass that died between writing a sidecar and recording it). Everything else goes:

* ``pending_tts`` means the next ``tts`` pass will ``clear_drafts`` and redraw from
  scratch, so whatever sits there now is already dead (deleting it early only saves
  the disk; the pass behaves identically).
* ``improved`` / ``exhausted`` / no marker at all means the clip is settled.
* a draft whose clip is in none of the ``--state`` files is an **orphan**: reported,
  but only deleted with ``--delete-orphans``, because the usual reason for one is that
  the dataset that owns the clip was simply not passed in.

**Two guards, because this deletes data that costs GPU hours to make.** It is a dry run
unless ``--apply`` is given. And it refuses to ``--apply`` while a pass is live, since
the state file lags the filesystem: the output is rewritten on an adaptive flush (5%
duty cycle, so every few minutes on a 900 MB file), and in that window a clip can own
fresh candidates while its row still reads ``pending_tts``. Deleting those would not
corrupt anything (the clip is simply redrawn), but it would throw away real GPU work,
so the default is to wait for the run to be idle. ``--allow-running`` overrides it, and
``--min-age-min`` (60 by default) stays in force either way as the second belt: a file
younger than that is never touched.

``<clip>.bak`` backups are a different thing and are never considered here: they hold
the only remaining copy of the audio an improvement overwrote.

Usage (the audio root is the same value the driver passes, i.e. the manifest's own
directory):

    uv run 06_hotfixes/09_prune_drafts.py \
        --state 06_hotfixes/improved/full.stt.jsonl \
        --audio-root 99_hf_release/data/NeMO_files
    # ... then the same command with --apply

Both datasets in one sweep (one --state each, one --audio-root each, paired in order;
a single --audio-root applies to all of them). Passing every dataset that shares an
audio directory is what makes the orphan count meaningful.

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import posixpath
import re
import time
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    """Import a sibling script whose ``01_`` filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The naming convention, the state field names and the status values all belong to the
# pass; importing it is what keeps this script from growing a second copy of them.
_imp = _load_module(_HERE / "01_recursive_improvement.py", "recursive_improvement_mod")

#: Inverse of ``_imp.draft_path_for``, which names a candidate
#: ``<clip>.draft<seed>.<fmt>``. Only the pass can produce these, so the seed is
#: digits and the extension a bare audio suffix; `test_prune_drafts.py` pins the round
#: trip against `draft_path_for` so the two cannot drift.
DRAFT_RE = re.compile(r"^(?P<clip>.+)\.draft(?P<seed>\d+)\.(?P<fmt>[A-Za-z0-9]+)$")

#: The status whose candidates are still needed: the clip is waiting for an stt pass.
LIVE_STATUS = "pending_stt"

#: What a live pass looks like in /proc. Checked by name rather than by pid file
#: because the pass is started by the driver under `runuser`, several layers down.
PASS_SCRIPT = "01_recursive_improvement.py"

#: argv[0] basenames that can be RUNNING the pass. The script name has to appear as
#: its own argv element under one of these, because a plain substring test matches any
#: shell (`zsh -c "... 01_recursive_improvement.py ..."`) that merely mentions it,
#: including the terminal this tool was launched from. Same trap as `pkill -f`.
INTERPRETERS = ("python", "uv", "env", PASS_SCRIPT)


def clip_for_draft(draft: Path) -> Path | None:
    """The clip a candidate sidecar belongs to, or None if the name is not one."""
    m = DRAFT_RE.match(draft.name)
    return draft.with_name(m.group("clip")) if m else None


def running_passes() -> list[tuple[int, str]]:
    """``(pid, cmdline)`` of every live improvement pass, as seen in /proc.

    Reads /proc directly rather than shelling out to pgrep: pgrep's pattern would also
    match this script's own command line (and, in a sandboxed shell, pgrep sees a
    private pid namespace), whereas here the caller is skipped by pid.
    """
    found = []
    me = os.getpid()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmd = (Path(entry.path) / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue  # the process exited while we looked at it
        argv = [a for a in cmd.split("\0") if a]
        if not argv:
            continue  # a kernel thread
        exe = os.path.basename(argv[0])
        if not exe.startswith(INTERPRETERS):
            continue
        if exe == PASS_SCRIPT or any(os.path.basename(a) == PASS_SCRIPT for a in argv[1:]):
            found.append((int(entry.name), " ".join(argv)))
    return found


def read_state(state: Path, audio_root: Path, audio_key: str = "audio_filepath") -> tuple[dict, set]:
    """``(owners, live_clips)`` for one dataset's state file.

    ``owners`` maps every clip's real path to its improvement status (``""`` when the
    row carries no marker at all, i.e. the clip never needed work), and ``live_clips``
    is the subset whose candidates must be kept.

    The real path is built by resolving the clip's DIRECTORY once per distinct
    directory and joining the basename, instead of calling ``Path.resolve()`` 600k
    times: the manifest holds a handful of stage directories (``../PARHAF/...``), and
    the audio root is a symlink into the dataset SSD, so the resolve has to happen, but
    it has to happen six times rather than once per row.
    """
    root = audio_root.resolve()
    dir_cache: dict[str, Path] = {}
    owners: dict[Path, str] = {}
    live: set[Path] = set()
    rows = 0
    # Streamed, not read_text().splitlines(): one dataset's state file is ~900 MB, and
    # there is no reason for a disk-cleanup tool to hold two copies of it in memory.
    with state.open(encoding="utf-8") as fh:
        lines = (ln.strip() for ln in fh)
        for line in lines:
            if not line:
                continue
            row = json.loads(line)
            raw = row.get(audio_key)
            if not raw:
                continue
            rows += 1
            head, tail = posixpath.split(raw)
            real_dir = dir_cache.get(head)
            if real_dir is None:
                real_dir = dir_cache[head] = (root / head).resolve()
            clip = real_dir / tail
            status = (row.get("improvement") or {}).get("status") or ""
            owners[clip] = status
            if status == LIVE_STATUS:
                live.add(clip)
    logger.info(f"{state.name}: {rows} row(s), {len(live)} awaiting judgement "
                f"({LIVE_STATUS}), {len(dir_cache)} audio dir(s)")
    return owners, live


def scan_drafts(dirs: set[Path]):
    """Yield ``(path, size, mtime)`` for every candidate sidecar in these directories.

    One ``scandir`` per audio directory rather than a recursive walk of the dataset:
    a candidate always sits beside its clip, so the directories the state files name
    are exactly the places one can be.
    """
    for d in sorted(dirs):
        try:
            entries = list(os.scandir(d))
        except OSError as exc:
            logger.warning(f"cannot scan {d}: {exc}")
            continue
        found = 0
        for entry in entries:
            if ".draft" not in entry.name or not entry.is_file(follow_symlinks=False):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            found += 1
            yield Path(entry.path), st.st_size, st.st_mtime
        logger.info(f"{d}: {len(entries)} entr(ies), {found} candidate sidecar(s)")


def classify(path: Path, mtime: float, owners: dict, live: set, cutoff: float) -> str:
    """Which bucket a candidate falls in: see the module docstring for the rule."""
    clip = clip_for_draft(path)
    if clip is None:
        return "not a candidate name"
    if mtime > cutoff:
        return "too new"
    if clip in live:
        return f"live ({LIVE_STATUS})"
    if clip in owners:
        return f"stale ({owners[clip] or 'no marker'})"
    return "orphan (clip in no state file)"


def human(n: int) -> str:
    return f"{n / 2 ** 30:.2f} GiB"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--state", "states", multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="A pass's .stt.jsonl state file. Repeatable, once per dataset.")
@click.option("--audio-root", "audio_roots", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Where that state file's relative audio paths start, i.e. the "
                   "manifest's own directory (what the driver passes as --audio-root). "
                   "Give one per --state, or a single one for all of them.")
@click.option("--apply", "apply_", is_flag=True,
              help="Actually delete. Without it this only reports what it would do.")
@click.option("--delete-orphans", is_flag=True,
              help="Also delete candidates whose clip is in none of the --state files. "
                   "Off by default: the usual cause is a dataset not passed in.")
@click.option("--min-age-min", default=60.0, show_default=True, type=float,
              help="Never touch a candidate younger than this many minutes.")
@click.option("--allow-running", is_flag=True,
              help="Prune even while an improvement pass is live. The state file lags "
                   "the filesystem by one flush, so this can discard fresh GPU work.")
@click.option("--audio-key", default="audio_filepath", show_default=True,
              help="JSONL key holding the clip path.")
def main(states, audio_roots, apply_, delete_orphans, min_age_min, allow_running, audio_key):
    """Delete the TTS candidate sidecars nothing needs any more."""
    if len(audio_roots) == 1:
        audio_roots = audio_roots * len(states)
    if len(audio_roots) != len(states):
        raise click.UsageError(
            f"{len(states)} --state but {len(audio_roots)} --audio-root: give one "
            "root per state file, or a single root for all of them")

    live_passes = running_passes()
    if live_passes:
        for pid, cmd in live_passes:
            logger.warning(f"an improvement pass is live: pid {pid}: {cmd[:160]}")
        if apply_ and not allow_running:
            raise SystemExit(
                "refusing to delete while a pass is running: its state file lags the "
                "filesystem by one flush, so candidates it just drew can still look "
                "settled. Wait for the run to go idle, or pass --allow-running "
                "(--min-age-min still applies).")

    owners: dict[Path, str] = {}
    live: set[Path] = set()
    for state, root in zip(states, audio_roots):
        o, l = read_state(state, root, audio_key)
        owners.update(o)
        live |= l

    dirs = {clip.parent for clip in owners}
    cutoff = time.time() - min_age_min * 60
    buckets: dict[str, list] = {}
    for path, size, mtime in scan_drafts(dirs):
        buckets.setdefault(classify(path, mtime, owners, live, cutoff), []).append((path, size))

    logger.info(f"{sum(len(v) for v in buckets.values())} candidate(s) in "
                f"{len(dirs)} dir(s), {len(owners)} known clip(s)")
    deletable = []
    for reason in sorted(buckets):
        items = buckets[reason]
        total = sum(s for _, s in items)
        goes = reason.startswith("stale") or (reason.startswith("orphan") and delete_orphans)
        logger.info(f"  {'DELETE' if goes else 'keep  '} {reason:38s} "
                    f"{len(items):7d} file(s)  {human(total):>12s}")
        if goes:
            deletable += items
    if "orphan (clip in no state file)" in buckets and not delete_orphans:
        logger.warning("orphans kept: pass every dataset that shares these audio "
                       "directories, then --delete-orphans if the count is still > 0")

    freeable = sum(s for _, s in deletable)
    if not deletable:
        logger.info("nothing to prune")
        return
    if not apply_:
        logger.info(f"dry run: {len(deletable)} file(s), {human(freeable)} would be "
                    f"freed. Re-run with --apply to delete them.")
        return

    freed = failed = 0
    for path, size in deletable:
        try:
            path.unlink()
        except OSError as exc:
            failed += 1
            logger.warning(f"cannot delete {path.name}: {exc}")
            continue
        freed += size
    logger.info(f"deleted {len(deletable) - failed} file(s), freed {human(freed)}"
                + (f", {failed} failed" if failed else ""))


if __name__ == "__main__":
    main()
