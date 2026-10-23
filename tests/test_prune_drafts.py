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
"""`06_hotfixes/09_prune_drafts.py`: which TTS candidate sidecars may be deleted.

The pass itself deletes a clip's candidates as soon as an stt pass judges it, so this
tool only ever sees the leftovers of a round that was interrupted or superseded. That
makes its rule the whole risk: delete one file too many and a clip's redraw (up to 5
Voxtral takes, minutes of GPU) is thrown away, or worse, a clip waiting to be judged
loses the candidates it is about to be judged on. So this file pins the rule itself:

  * a clip in `pending_stt` keeps EVERY candidate beside it, including one its row does
    not list (a pass that died between writing a sidecar and recording it);
  * `pending_tts` / `improved` / `exhausted` / no marker all mean the files are dead;
  * a candidate whose clip is in no state file is an orphan: never deleted unless
    asked for, since the usual cause is a dataset that was not passed in;
  * nothing younger than --min-age-min is touched, and nothing at all is deleted
    without --apply;
  * a live pass blocks --apply, because the state file lags the filesystem by a flush;
  * the name inverse (`clip_for_draft`) round-trips the pass's own `draft_path_for`,
    which is what keeps the two conventions from drifting, and it must not mistake a
    `.bak` backup for a candidate.

Run it with uv (the tool imports the pass, which imports soundfile):

    uv run tests/test_prune_drafts.py

This file was written with Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click.testing import CliRunner
from loguru import logger

HERE = Path(__file__).resolve().parent
PRUNE_PATH = HERE.parent / "06_hotfixes" / "09_prune_drafts.py"
IMPROVE_PATH = HERE.parent / "06_hotfixes" / "01_recursive_improvement.py"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if not ok else ""))
    if not ok:
        failures.append(label)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tree(tmp: Path) -> tuple[Path, Path, dict]:
    """A miniature corpus: clips in one stage dir, the state file in another directory
    and reaching them through `..`, exactly as the real manifests do
    (`audio_filepath: "../PARHAF/x.flac"` against `--audio-root .../NeMO_files`).

    Returns `(state_file, audio_root, files)` where `files` maps a label to the
    candidate sidecars written for it.
    """
    stage = tmp / "v2" / "stage"
    root = tmp / "v2" / "NeMO_files"
    out = tmp / "out"
    for d in (stage, root, out):
        d.mkdir(parents=True, exist_ok=True)

    # label -> (improvement status or None, how many candidates to write)
    plan = {
        "live": ("pending_stt", 2),
        "settled": ("improved", 1),
        "given_up": ("exhausted", 1),
        "queued": ("pending_tts", 1),
        "clean": (None, 1),
        "fresh": ("pending_tts", 1),
    }
    files: dict[str, list[Path]] = {}
    rows = []
    for label, (status, n) in plan.items():
        clip = stage / f"{label}.flac"
        clip.write_bytes(b"clip")
        files[label] = []
        for seed in range(43, 43 + n):
            draft = clip.with_name(f"{clip.name}.draft{seed}.flac")
            draft.write_bytes(b"x" * 1024)
            files[label].append(draft)
        row = {"audio_filepath": f"../stage/{label}.flac", "duration": 1.0}
        if status is not None:
            row["improvement"] = {"status": status, "orig_cer": 0.3}
            if status == "pending_stt":
                # Only the FIRST candidate is recorded, so the keep rule is exercised at
                # clip granularity: the unrecorded sibling must survive too.
                row["improvement"]["drafts"] = [{"path": str(files[label][0]), "seed": 43}]
                row["improvement"]["attempts"] = n
        rows.append(row)

    # A candidate with no clip and no row anywhere: the orphan case.
    orphan = stage / "ghost.flac.draft43.flac"
    orphan.write_bytes(b"x" * 1024)
    files["orphan"] = [orphan]
    # A backup of an improved clip: never a candidate, must be invisible to the tool.
    bak = stage / "settled.flac.bak"
    bak.write_bytes(b"original")
    files["bak"] = [bak]

    state = out / "full.stt.jsonl"
    state.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # Everything except `fresh` is old enough to be prunable.
    old = time.time() - 6 * 3600
    for label, paths in files.items():
        if label == "fresh":
            continue
        for p in paths:
            os.utime(p, (old, old))
    return state, root, files


def run(prune, *args):
    """Invoke the CLI, returning `(result, log_text)`: the report goes through loguru,
    which writes to the real stderr, so it is captured with a sink of our own."""
    captured: list[str] = []
    sink = logger.add(lambda m: captured.append(str(m)), format="{message}", level="DEBUG")
    try:
        result = CliRunner().invoke(prune.main, [str(a) for a in args])
    finally:
        logger.remove(sink)
    return result, "".join(captured)


def test_name_inverse(prune, imp) -> None:
    print("test_name_inverse")
    for name, seed, fmt in (("x.flac", 43, "flac"), ("a.b.c.flac", 47, "wav"),
                            ("000007_0000_anatomopathologie_00002_t0_c3.flac", 44, "flac")):
        clip = Path("/data/stage") / name
        draft = imp.draft_path_for(clip, seed, fmt)
        check(f"{draft.name} -> {name}", prune.clip_for_draft(draft) == clip,
              str(prune.clip_for_draft(draft)))
    for not_a_draft in ("x.flac", "x.flac.bak", "x.flac.bak2", "x.flac.draft43.flac.bak"):
        check(f"{not_a_draft} is not a candidate",
              prune.clip_for_draft(Path("/data/stage") / not_a_draft) is None)


def test_the_rule(prune, tmp: Path) -> None:
    print("test_the_rule")
    state, root, files = make_tree(tmp)
    alive = lambda label: all(p.is_file() for p in files[label])  # noqa: E731

    # --allow-running throughout: this test is about the rule, and the dev machine may
    # well have a real pass in flight (the guard itself is covered below).
    res, log = run(prune, "--state", state, "--audio-root", root)
    check("the dry run exits 0", res.exit_code == 0, f"{res.exit_code} {res.exception}")
    check("the dry run deletes nothing", all(alive(k) for k in files))
    check("and says what it would free", "dry run:" in log and "would be freed" in log, log[-300:])
    check("it counts 4 stale candidates", "4 file(s)" in log, log)

    res, log = run(prune, "--state", state, "--audio-root", root, "--apply", "--allow-running")
    check("--apply exits 0", res.exit_code == 0, f"{res.exit_code} {res.exception}")
    check("a judged clip's candidate goes", not alive("settled"))
    check("an exhausted clip's candidate goes", not alive("given_up"))
    check("a clip queued for a redraw loses its stale candidate", not alive("queued"))
    check("a clip that never needed work loses its candidate", not alive("clean"))
    check("a clip awaiting judgement keeps BOTH its candidates", alive("live"))
    check("including the one its row does not list", files["live"][1].is_file())
    check("a candidate younger than --min-age-min is untouched", alive("fresh"))
    check("an orphan is kept without --delete-orphans", alive("orphan"))
    check("and the report says so", "orphans kept" in log, log[-400:])
    check("a .bak backup is never a candidate", alive("bak"))

    res, log = run(prune, "--state", state, "--audio-root", root, "--apply",
                   "--allow-running", "--delete-orphans")
    check("--delete-orphans deletes it", not alive("orphan"))
    check("and still leaves the live clip alone", alive("live"))

    res, log = run(prune, "--state", state, "--audio-root", root, "--apply",
                   "--allow-running", "--min-age-min", "0")
    check("--min-age-min 0 reaches the fresh candidate", not alive("fresh"))
    check("the clip files themselves are never touched",
          (tmp / "v2" / "stage" / "live.flac").is_file()
          and (tmp / "v2" / "stage" / "settled.flac").is_file())

    res, log = run(prune, "--state", state, "--audio-root", root, "--apply", "--allow-running")
    check("a second run has nothing left to do", "nothing to prune" in log, log[-200:])


def test_a_live_pass_blocks_it(prune, tmp: Path) -> None:
    print("test_a_live_pass_blocks_it")
    state, root, files = make_tree(tmp)
    # A process whose command line looks like the pass. Sleeping python, not the real
    # script: the guard reads /proc/<pid>/cmdline, so what matters is the argv text.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                             str(IMPROVE_PATH), "--mode", "tts"])
    try:
        for _ in range(50):  # the pid is in /proc immediately, but be kind to CI
            if prune.running_passes():
                break
            time.sleep(0.1)
        check("the guard sees the live pass", bool(prune.running_passes()))
        res, log = run(prune, "--state", state, "--audio-root", root, "--apply")
        check("--apply refuses", res.exit_code != 0, str(res.exit_code))
        check("and says why", "refusing to delete while a pass is running"
              in str(res.exception), str(res.exception)[:200])
        check("nothing was deleted", all(p.is_file() for ps in files.values() for p in ps))
        res, log = run(prune, "--state", state, "--audio-root", root)
        check("a dry run is still allowed", res.exit_code == 0, str(res.exception))
        res, log = run(prune, "--state", state, "--audio-root", root, "--apply",
                       "--allow-running")
        check("--allow-running overrides it", not files["settled"][0].is_file())
        check("but --min-age-min still protects the fresh one", files["fresh"][0].is_file())
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_the_root_pairing(prune, tmp: Path) -> None:
    print("test_the_root_pairing")
    state, root, _ = make_tree(tmp)
    res, _ = run(prune, "--state", state, "--state", state,
                 "--audio-root", root, "--audio-root", root, "--audio-root", root)
    check("3 roots for 2 states is a usage error", res.exit_code != 0, str(res.exit_code))
    res, log = run(prune, "--state", state, "--state", state, "--audio-root", root)
    check("one root covers every state file", res.exit_code == 0, str(res.exception))


if __name__ == "__main__":
    import tempfile

    prune = load_module(PRUNE_PATH, "prune_under_test")
    imp = load_module(IMPROVE_PATH, "recimp_for_names")
    test_name_inverse(prune, imp)
    with tempfile.TemporaryDirectory() as td:
        test_the_rule(prune, Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_a_live_pass_blocks_it(prune, Path(td))
    with tempfile.TemporaryDirectory() as td:
        test_the_root_pairing(prune, Path(td))
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)
