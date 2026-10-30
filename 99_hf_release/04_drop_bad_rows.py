#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Drop the rows that must not ship from every NeMo manifest of the release.

Run AFTER ``03_sync_hotfix_results.py --apply`` (which is what writes ``qc_status``
into the manifests) and BEFORE ``05_normalize_text.py`` and ``scripts/build_parquet.py``:

    uv run 04_drop_bad_rows.py            # dry run, reports what would be dropped
    uv run 04_drop_bad_rows.py --apply

Seven kinds of rows are dropped:

1. **Exhausted clips.** A clip is ``exhausted`` when stage 06 flagged it (its STT
   reading disagreed too much with its label) and no regenerated draw was good
   enough, so the ORIGINAL, known-bad audio is still what sits on disk. UltiMed v1
   shipped those rows, which was a mistake: a clip whose audio does not say its label
   is label noise in train and a wrong reference in val/test. They are removed rather
   than patched because by definition no good audio exists for them. Matched on the
   row's own ``qc_status`` field, not by joining against another manifest, so a
   manifest that was never synced (no ``qc_status`` at all) is reported rather than
   silently treated as clean.

2. **Eval clips whose text is also a training text.** The release-wide split
   (``02_combine_nemo_manifests.py``) groups rows by document / term, not by text, so
   two PARHAF documents that happen to contain the same sentence can put it in train
   AND in val/test (UltiMed v1 had one such sentence in val and one in test, differing
   only in punctuation or case). The eval side is dropped, since train loses nothing
   and the eval set would otherwise score a sentence the model was trained on.
   Texts are compared after ``normalize_text`` (NFKC, lowercase, punctuation to space,
   whitespace collapsed), the same key the trainer's ``data_leak_check`` uses (see
   below). Fixing it here rather than in the splitter keeps the v1 split stable: a
   splitter change would reshuffle every row.

3. **Labels with an ellipsis.** ``...`` (or ``…``) in a label marks text that is not
   there: an LLM output cut short (``rénine active à 12 milliu...``) or a date PARHAF
   anonymized away (``La date de l'intervention est le ...``). The audio says the
   truncated text, so the pair is consistent, but it teaches the model to write an
   ellipsis for a trailing-off sentence, which no dictation wants. UltiMed v1 had 2.

4. **Leaked LLM reasoning.** A label holding the generator's own chatter instead of
   a sentence: ``tags Let me create varied contexts: - Sentence 1: 300 mg comprimé -
   initial prescription ...``, found 2026-09-28 in the hand-made ``drug_sentence``
   train set (the TTS then read it all). Matched on English meta-talk markers only
   (``let me``, ``I'll``, ``here are``, ``Sentence 1:``, a ``<t>`` tag), never on a
   lone English word: UltiMed holds legitimate English terms (``bed nucleus of the
   accessory olfactory tract``) and French ``sentence arbitrale``. UltiMed v1.2 has 0.

5. **Impossible speaking rate.** More than ``MAX_WORDS_PER_SECOND`` words per second
   of the manifest ``duration``: the audio cannot say the label. Found 2026-09-28 in
   ``drug_sentence`` train: a 16-word label on a 0.3 s clip (53 words/s), a broken
   TTS output. Real speech in UltiMed tops out at 4.2 words/s (99.99th percentile
   3.6), so 8 leaves a wide margin and drops no UltiMed row.

6. **ASR-flagged clips.** The clips listed in ``asr_flagged_clips.jsonl`` (committed,
   next to this script), written by ``06_hotfixes/04_flag_asr_defects.py`` from a
   fine-tuned Parakeet transcription of the corpus: a TTS preamble (voxtral babbled
   words before the sentence, which Whisper QC skipped) or a phrase both Parakeet and
   Whisper agree the audio lacks. See that script for the rules. Matched on
   ``nemo_manifest.clip_key`` (``<source>/<file>``). A missing list drops nothing.
   Because the key is a name and not the audio, the list only means something for the
   audio it was measured on: regenerate the corpus (v2 reuses v1's file names) and it
   would drop good clips for defects that no longer exist. So the run warns when the
   list is older than the clips it flags, and ``--no-asr-flagged`` turns it off until
   ``06_hotfixes/04_flag_asr_defects.py`` has re-run on the new audio
   (``--asr-flagged-list PATH`` points at another one).
(Rule 7, ``cfg-off``, dropped every take at exactly ``cfg_alpha`` 1.0. Retired on
2026-10-30: the T9c redraw at cfg 1.0 fixed 59 of the 143 hardest clips, loops 100 ->
38, and those takes pass the same per-clip judges as any other, so the judges alone
decide. Old ``cfg-off`` entries in the drop ledger are no longer re-applied.)

Two of those reasons (exhausted, ASR-flagged) are not in the row itself:
they come from stage 06's state files, through the sync or the flag list. And stage
06's state follows the manifest it is pointed at, so a QC round run AFTER this script
forgets every clip this script removed, and the next ``01_build_nemo_manifest.py``
rebuild finds the audio on disk and brings the row back with no record left to drop
it on. That happened on 2026-10-27: 71 rows (17 exhausted, 54 ASR-flagged) came back
with six empty QC columns, and 63 earlier ones (54 CFG off, 9 exhausted) came back,
got re-read by a Whisper judge, and passed. So every such drop is also written to
``dropped_clips.jsonl`` (committed, next to this script), with the audio's mtime, and
re-applied as ``previously-dropped`` for as long as the audio is the same bytes. A
clip redrawn after its drop has newer audio, so its old verdict no longer applies and
it is judged afresh.

A dropped clip is dropped from EVERY manifest (matched on its resolved audio path),
including the per-source ``<source>/*.jsonl`` partitions and the down-sampled
``*.down-N`` fixtures, since each is a copy of rows and the parquet / trainer read
them directly. The ``.flac`` files are NOT deleted: nothing references them once the
manifests are filtered (``build_parquet.py`` embeds only manifest rows), and keeping
them lets a later stage-06 retry still reach them.

The hand-made NeMo-repo sets (``perso/drug_sentence_dataset/*.json``, the private
``perso/oli_spoken_dataset`` manifests) are filtered by passing them explicitly:
``uv run 04_drop_bad_rows.py --apply <manifest> ...`` (the eval-duplicate check still
compares against the release train split only).

Idempotent: a second run finds nothing left to drop, and writes nothing new to the ledger.

This file was written by Claude Code.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections.abc import Collection, Mapping
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import MANIFEST_GLOBS, clip_key, read_jsonl, resolve_audio, write_jsonl  # noqa: E402

# The release-wide split the parquet and the trainer read. PARROT is eval-only and
# lives outside it, but it must not repeat a training text either.
TRAIN_MANIFEST = "data/NeMO_files/train.jsonl"
EVAL_MANIFESTS = ("data/NeMO_files/val.jsonl", "data/NeMO_files/test.jsonl",
                  "data/NeMO_files/PARROT/full.jsonl")

EXHAUSTED = "exhausted"
DUPLICATE = "duplicate-of-train"
ELLIPSIS = "ellipsis"
_ELLIPSIS = re.compile(r"\.\.\.|…")
LLM_LEAK = "llm-leak"
_LLM_LEAK = re.compile(r"(?i:\blet me\b|\bI'll\b|\bI will\b|\bhere (?:is|are)\b|\bas requested\b)"
                       r"|\bSentence \d+\s*:|</?t>")
TOO_FAST = "too-fast"
MAX_WORDS_PER_SECOND = 8.0
ASR_FLAGGED = "asr-flagged"
ASR_FLAGGED_LIST = _HERE / "asr_flagged_clips.jsonl"
PREVIOUSLY_DROPPED = "previously-dropped"
DROP_LEDGER = _HERE / "dropped_clips.jsonl"
#: The reasons whose evidence lives outside the row (stage 06's state, the flag list),
#: so a later QC round can erase it. The others are re-derived from the row's own
#: text, duration and split on every run and need no memory.
REMEMBERED_REASONS = frozenset({EXHAUSTED, ASR_FLAGGED})

_PUNCT = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Comparison key for "same sentence": NFKC, lowercase, punctuation to space,
    whitespace collapsed, so ``Sur le plan cardiovasculaire, il n'y a pas...`` and
    ``sur le plan cardiovasculaire il n'y a pas...`` collide.

    Deliberate cross-repo copy of ``normalize_text`` in the NeMo fork's
    ``nemo/collections/asr/parts/utils/data_leak_check.py`` (a separate repo, so it
    cannot be imported). Keep the two identical: if this one is weaker, the trainer
    refuses a release this script called clean.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def eval_duplicates(train_rows: list[dict],
                    eval_manifests: dict[Path, list[dict]]) -> dict[str, str]:
    """Find the eval rows whose normalized text also appears in train.

    Parameters
    ----------
    train_rows : list[dict]
        Rows of the release-wide train manifest.
    eval_manifests : dict[Path, list[dict]]
        Eval manifest path -> its rows (the path resolves relative audio paths).

    Returns
    -------
    dict[str, str]
        Resolved audio path of each eval clip to drop -> its text, for the log.
    """
    train_keys = {normalize_text(r["text"]) for r in train_rows}
    return {resolve_audio(path.parent, r["audio_filepath"]): r["text"]
            for path, rows in eval_manifests.items() for r in rows
            if normalize_text(r["text"]) in train_keys}


def _too_fast(row: dict) -> bool:
    """More words than the clip can hold (``MAX_WORDS_PER_SECOND``); a row without a
    ``duration`` cannot be judged and is kept."""
    duration = row.get("duration")
    return bool(duration) and len(row.get("text", "").split()) / duration > MAX_WORDS_PER_SECOND


def load_asr_flagged(path: Path = ASR_FLAGGED_LIST) -> set[str]:
    """Clip keys listed by ``06_hotfixes/04_flag_asr_defects.py``; empty if no list."""
    return {r["clip"] for r in read_jsonl(path)} if path.is_file() else set()


def asr_flagged_stale(flagged: Collection[str], root_path: Path,
                      list_path: Path = ASR_FLAGGED_LIST) -> tuple[int, int]:
    """How many flagged clips the list can no longer be speaking about.

    The list names clip keys, not audio, so it keeps "matching" after the audio
    under those names has been replaced: a regenerated clip, or a whole new
    generation of the dataset synthesized under v1's file names, would be dropped
    for a defect that was measured on bytes that are gone. The check is the same
    one ``probe_durations`` makes for durations: compare each clip's mtime with the
    list's.

    Parameters
    ----------
    flagged : Collection[str]
        Clip keys from ``load_asr_flagged``.
    root_path : Path
        Release stage root; clip keys resolve under ``<root>/data/``.
    list_path : Path
        The list itself, whose mtime is the reference.

    Returns
    -------
    n_newer : int
        Clips whose audio was written after the list, so its verdict is stale.
    n_missing : int
        Clips whose audio is not under ``<root>/data`` at all, so nothing can be
        checked (an unmounted symlink, or a list from another corpus).
    """
    if not flagged or not list_path.is_file():
        return 0, 0
    listed_at = list_path.stat().st_mtime_ns
    n_newer = n_missing = 0
    for clip in flagged:
        audio = root_path / "data" / clip
        try:
            if audio.stat().st_mtime_ns > listed_at:
                n_newer += 1
        except OSError:
            n_missing += 1
    return n_newer, n_missing


def audio_mtime_ns(audio: str) -> int | None:
    """The audio file's mtime, or None when it is not on disk."""
    try:
        return os.stat(audio).st_mtime_ns
    except OSError:
        return None


def load_drop_ledger(path: Path = DROP_LEDGER) -> dict[str, dict]:
    """``clip_key`` -> ledger entry (``reason``, ``audio_mtime_ns``); empty if no ledger."""
    return {r["clip"]: r for r in read_jsonl(path)} if path.is_file() else {}


def still_dropped(entry: dict, audio: str) -> bool:
    """Whether a remembered drop still applies: the audio is the bytes it was made on.

    A clip redrawn since has a newer mtime, so the verdict describes audio that is gone.
    A clip missing from disk cannot be shipped either way, and is not dropped here.
    """
    mtime = audio_mtime_ns(audio)
    return mtime is not None and mtime <= entry["audio_mtime_ns"]


def record_drops(ledger: dict[str, dict], dropped: list[tuple[dict, str, str]]) -> int:
    """Add the drops whose reason a later QC round could erase to ``ledger``, in place.

    Parameters
    ----------
    ledger : dict[str, dict]
        From ``load_drop_ledger``.
    dropped : list[tuple[dict, str, str]]
        ``(row, reason, resolved audio path)`` for every row dropped this run.

    Returns
    -------
    int
        How many entries were added or refreshed.
    """
    n = 0
    for _, reason, audio in dropped:
        if reason not in REMEMBERED_REASONS:
            continue
        mtime = audio_mtime_ns(audio)
        key = clip_key(audio)
        if mtime is None or ledger.get(key, {}).get("audio_mtime_ns") == mtime:
            continue
        ledger[key] = {"clip": key, "reason": reason, "audio_mtime_ns": mtime}
        n += 1
    return n


def filter_rows(rows: list[dict], manifest_dir: Path, drop_audio: Collection[str] = (),
                asr_flagged: Collection[str] = (), ledger: Mapping[str, dict] | None = None,
                ) -> tuple[list[dict], list[tuple[dict, str]], int]:
    """Split manifest rows into kept and dropped ones.

    Parameters
    ----------
    rows : list[dict]
        NeMo manifest rows, as read by ``read_jsonl``.
    manifest_dir : Path
        Directory of the manifest, which relative ``audio_filepath`` values resolve against.
    drop_audio : Collection[str]
        Resolved audio paths to drop whatever their status (from ``eval_duplicates``).
    asr_flagged : Collection[str]
        ``clip_key`` values to drop (from ``load_asr_flagged``).
    ledger : Mapping[str, dict] | None
        Earlier drops (from ``load_drop_ledger``), re-applied while the audio is unchanged.

    Returns
    -------
    kept : list[dict]
        Rows that stay (order preserved).
    dropped : list[tuple[dict, str]]
        Each dropped row with its reason: ``EXHAUSTED``, ``DUPLICATE``, ``ELLIPSIS``,
        ``LLM_LEAK``, ``TOO_FAST``, ``ASR_FLAGGED`` or
        ``"previously-dropped (<reason>)"``.
    n_unsynced : int
        How many rows carry no ``qc_status`` key at all, i.e. were never touched by
        ``03_sync_hotfix_results.py``: their status is unknown, so they are kept, but
        the caller should warn since an exhausted clip could hide among them.
    """
    kept: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    n_unsynced = 0
    for row in rows:
        if "qc_status" not in row:
            n_unsynced += 1
        if row.get("qc_status") == EXHAUSTED:
            dropped.append((row, EXHAUSTED))
        elif drop_audio and resolve_audio(manifest_dir, row["audio_filepath"]) in drop_audio:
            dropped.append((row, DUPLICATE))
        elif _ELLIPSIS.search(row.get("text", "")):
            dropped.append((row, ELLIPSIS))
        elif _LLM_LEAK.search(row.get("text", "")):
            dropped.append((row, LLM_LEAK))
        elif _too_fast(row):
            dropped.append((row, TOO_FAST))
        elif asr_flagged and clip_key(resolve_audio(manifest_dir, row["audio_filepath"])) in asr_flagged:
            dropped.append((row, ASR_FLAGGED))
        elif ledger and (entry := ledger.get(clip_key(resolve_audio(manifest_dir, row["audio_filepath"])))) \
                and entry["reason"] in REMEMBERED_REASONS \
                and still_dropped(entry, resolve_audio(manifest_dir, row["audio_filepath"])):
            dropped.append((row, f"{PREVIOUSLY_DROPPED} ({entry['reason']})"))
        else:
            kept.append(row)
    return kept, dropped, n_unsynced


@click.command()
@click.argument("manifests", nargs=-1, type=click.Path(path_type=Path))
@click.option("--root", default=str(_HERE), show_default="99_hf_release/",
              help="Release stage root, holding the `data` symlink.")
@click.option("--apply", is_flag=True, help="Write. Without it this is a dry run.")
@click.option("--asr-flagged-list", "asr_flagged_list", default=str(ASR_FLAGGED_LIST),
              show_default="99_hf_release/asr_flagged_clips.jsonl", type=click.Path(path_type=Path),
              help="List of ASR-flagged clip keys to drop (from 06_hotfixes/04_flag_asr_defects.py). "
                   "A list built on an earlier generation of the audio must be rebuilt or disabled.")
@click.option("--drop-ledger", "drop_ledger", default=str(DROP_LEDGER),
              show_default="99_hf_release/dropped_clips.jsonl", type=click.Path(path_type=Path),
              help="Memory of earlier drops whose evidence a QC round can erase; read every run, "
                   "written with --apply.")
@click.option("--no-asr-flagged", is_flag=True,
              help="Ignore the ASR-flagged list entirely. What to pass on a freshly "
                   "regenerated corpus until 04_flag_asr_defects.py has re-run on it.")
def main(manifests: tuple[Path, ...], root: str, apply: bool,
         asr_flagged_list: Path, drop_ledger: Path, no_asr_flagged: bool) -> None:
    """Remove exhausted clips, eval duplicates of training texts, ellipsis labels, leaked LLM
    reasoning, impossibly fast labels and ASR-flagged clips from the release manifests (or the
    given ones)."""
    root_path = Path(root).resolve()
    targets = [Path(m) for m in manifests]
    if not targets:
        for pattern in MANIFEST_GLOBS:
            targets.extend(sorted(root_path.glob(pattern)))
    if not targets:
        raise SystemExit(f"no manifest found under {root_path} (is the `data` symlink mounted?)")

    duplicates = eval_duplicates(
        read_jsonl(root_path / TRAIN_MANIFEST),
        {root_path / m: read_jsonl(root_path / m) for m in EVAL_MANIFESTS if (root_path / m).is_file()},
    )
    for audio, text in duplicates.items():
        logger.info(f"eval clip repeats a training text: {audio} ({text[:80]!r})")

    list_path = Path(asr_flagged_list)
    asr_flagged = set() if no_asr_flagged else load_asr_flagged(list_path)
    if no_asr_flagged:
        logger.warning("--no-asr-flagged: the ASR-flagged list is ignored, no clip is dropped for it")
    else:
        logger.info(f"{len(asr_flagged)} ASR-flagged clips listed in {list_path.name}")
        n_newer, n_missing = asr_flagged_stale(asr_flagged, root_path, list_path)
        if n_newer or n_missing:
            # Loud rather than fatal: on a mixed corpus a few regenerated clips are
            # expected, while "all of them" means the list belongs to a previous
            # generation of the audio and would drop good clips for old defects.
            logger.warning(
                f"{list_path.name} is OLDER than the audio of {n_newer} of the {len(asr_flagged)} "
                f"clips it flags ({n_missing} not found under {root_path / 'data'}): those verdicts "
                "were measured on audio that is gone. Rebuild the list with "
                "06_hotfixes/04_flag_asr_defects.py, or pass --no-asr-flagged.")

    ledger_path = Path(drop_ledger)
    ledger = load_drop_ledger(ledger_path)
    logger.info(f"{len(ledger)} earlier drops remembered in {ledger_path.name}")

    total_dropped = 0
    this_run: list[tuple[dict, str, str]] = []
    for path in targets:
        if not path.is_file() or path.name.startswith("."):
            continue
        rows = read_jsonl(path)
        kept, dropped, n_unsynced = filter_rows(rows, path.parent, duplicates, asr_flagged, ledger)
        this_run.extend((row, reason, resolve_audio(path.parent, row["audio_filepath"]))
                        for row, reason in dropped)
        label = path.relative_to(root_path) if path.is_relative_to(root_path) else path
        if n_unsynced:
            logger.warning(f"{label}: {n_unsynced}/{len(rows)} rows WITHOUT qc_status, run "
                           f"03_sync_hotfix_results.py --apply first or they cannot be judged")
        if not dropped:
            continue
        total_dropped += len(dropped)
        logger.info(f"{label}: dropping {len(dropped)}/{len(rows)} rows")
        for row, reason in dropped:
            logger.info(f"    {reason}: {row.get('audio_filepath')} (cer={row.get('cer')})")
        if apply:
            write_jsonl(path, kept)

    logger.info(f"total: {total_dropped} rows dropped across all manifests")
    if apply and (n_new := record_drops(ledger, this_run)):
        write_jsonl(ledger_path, sorted(ledger.values(), key=lambda r: r["clip"]))
        logger.info(f"{ledger_path.name}: {n_new} drop(s) remembered, {len(ledger)} in total")
    if total_dropped and not apply:
        logger.warning("dry run, nothing written (pass --apply)")
    elif total_dropped:
        logger.info("next: uv run 05_normalize_text.py --apply, then uv run scripts/build_parquet.py")


if __name__ == "__main__":
    main()
