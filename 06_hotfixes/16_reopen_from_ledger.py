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
"""Queue for redraw every clip the clip ledger says is still wrong (round T6).

`14_reopen_loop_clips.py` reopens one defect from one judge's reading. The clip ledger
(`utils/clip_ledger.py`, rebuilt by `15_build_clip_ledger.py`) puts EVERY verdict on the
current take side by side, and that is what showed the gaps on 2026-10-28: clips the
acoustic scan had flagged `looped` were shipped because the keep/redraw decision only
read whisper's verdict, wrong numbers both judges heard were invisible to the CER gates,
and so were clauses said twice. This script turns the ledger's flags into the stage's
own `pending_tts` marker, so the normal pass redraws them; it writes no audio and judges
nothing new.

**Rules** (a clip is queued when any fires; all that fire are recorded as its reasons):

* `loop`: `acoustic:loop` AND a parakeet-side verdict (a stage-06 parakeet gate, or an
  `over_gate` from the parakeet sweep, its refill, or live QC). The acoustic scan alone
  over-fires on legitimately repeated phrases; parakeet hears the repeat verbatim.
* `number`: ANY judge (turbo or parakeet) heard a number the label does not hold
  (`utils/number_swaps.swaps`). Deliberately looser than the ledger's own `text:number`
  flag (both judges, same wrong number): on clips holding a 70-79 / 90-99 number read by
  an `h_itf_*` voice, 20.4 pct are flagged by both and a further 11.7 pct by one judge
  only, against 0.7 pct one-judge-only for the fr voices, which is the judges' own
  noise level. The excess is mumbled numbers, and the user's call (2026-10-28) was that
  a redraw costs less than shipping one.
* `inserted`: `text:inserted`, a clause of 3+ words both judges heard at the same place
  that the label lacks (`04_flag_asr_defects.py::_inserted_by_both`).
* `window`: the worst ~15 s window over `WINDOW_BOTH` on turbo AND parakeet. One judge
  over the stage's window gate is mostly that judge's blind spot (French medical terms,
  language-id flips); both at once is audio.
* `asr`: the release's own ASR-flag rules (`text:preamble`, `text:repeated`).
* `dropped`: dropped by the release for an audio reason (`REMEMBERED_REASONS` of
  `99_hf_release/04_drop_bad_rows.py`: exhausted, asr-flagged, cfg-off). A redraw can
  bring those back; a text-side drop (ellipsis, llm-leak, too-fast) it cannot, so those
  are never queued.

**Voice rule for `number`.** The two `h_itf_*` voices are the worst at French numbers by
far (rates in `voice_assignment.NUMBER_LADDER_FR_V2`: 25.7 and 12.9 pct on 70s/90s
labels, against 0.34-0.69 for the other hybrids), and they are also the first rung of
the stage's escalation ladder. So a number clip:

* read by an `h_itf_*` voice goes to one of the five other hybrids right away
  (`NUMBER_LADDER_FR_V2`, keeping the accent spread), and to the fr pair if that fails;
* read by any other voice keeps it with a fresh seed (its errors are seed noise, not an
  accent), and escalates straight to the fr pair, skipping the itf rung.

Both go through `01_recursive_improvement.swap_voice_for`, so a document's chunks still
share one new reader. Applies to every number, not only 70s/90s: the itf voices trip
elsewhere too (63 -> 73, 6 -> 16, 100 -> 5). T6 (2026-10-28) ran an earlier version that
sent itf number clips to the fr pair directly; its 4,050 such clips keep that.

**Marker scores** are the parakeet reading's, as in 14 and for the same reason: a draw
is promoted on beating `orig_cer`, and whisper's view of a looping clip is already near
0, which would exhaust every one of them.

Run order (the ledger must be rebuilt after anything touched the state):

    uv run 15_build_clip_ledger.py
    uv run 16_reopen_from_ledger.py                                   # census only
    uv run 16_reopen_from_ledger.py --apply --out-stems t6_stems.txt
    sudo -E MODE=alternate STEMS_FROM=t6_stems.txt NUMBER_THRESHOLD=1 CFG_ALPHA_FLOOR=1.6 ./driver.sh

`NUMBER_THRESHOLD=1` turns on the repair pass's number gate, so a redraw that repeats
the wrong number is not promoted.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
RELEASE = ROOT / "99_hf_release"
sys.path.insert(0, str(ROOT / "utils"))
from clip_ledger import LEDGER_NAME  # noqa: E402
from nemo_manifest import clip_key, resolve_audio  # noqa: E402
from number_swaps import swaps  # noqa: E402
from voice_assignment import NUMBER_LADDER_FR_V2  # noqa: E402
from script_module import load_script  # noqa: E402

CATEGORIES = ("PARHAF", "PARROT", "acronyms", "dictionary", "drugs")
REASONS = ("loop", "number", "inserted", "window", "asr", "dropped")
#: Both judges' worst window at once (see module docstring).
WINDOW_BOTH = 0.12
#: Parakeet-side verdicts that confirm an acoustic loop.
PARAKEET_OVER = ("parakeet_ultra:over_gate", "parakeet_refill:over_gate", "live_stt:over_gate")
ITF_PREFIX = "h_itf_"


def reasons_of(rec: dict, audio_drops) -> list[str]:
    """Every rule that fires on one ledger record, in `REASONS` order.

    >>> base = {"audio": {"exists": True}, "release": {"splits": ["train"], "dropped": None},
    ...         "judges": {}, "flags": []}
    >>> reasons_of({**base, "flags": ["acoustic:loop"]}, ())
    []
    >>> reasons_of({**base, "flags": ["acoustic:loop", "parakeet:cer", "text:number"]}, ())
    ['loop', 'number']
    >>> heard = lambda t, p: {**base, "text": "75 ans", "judges": {"turbo": {"text": t}, "parakeet": {"text": p}}}
    >>> reasons_of(heard("75 ans", "65 ans"), ()), reasons_of(heard("75 ans", "75 ans"), ())
    (['number'], [])
    >>> reasons_of({**base, "judges": {"turbo": {"window": 0.2}, "parakeet": {"window": 0.13}}}, ())
    ['window']
    >>> reasons_of({**base, "judges": {"turbo": {"window": 0.2}, "parakeet": {"window": 0.05}}}, ())
    []
    >>> gone = {**base, "release": {"splits": [], "dropped": {"reason": "cfg-off", "applies": True}}}
    >>> reasons_of(gone, {"cfg-off"}), reasons_of({**gone, "flags": ["text:number"]}, ())
    (['dropped'], [])
    >>> reasons_of({**base, "audio": {"exists": False}, "flags": ["text:number"]}, ())
    []
    """
    if not rec["audio"]["exists"]:
        return []
    rel, flags = rec["release"], set(rec["flags"])
    dropped = rel.get("dropped") or {}
    audio_drop = bool(dropped.get("applies")) and dropped.get("reason") in audio_drops
    if not rel["splits"] and not audio_drop:
        return []  # unreleased, or dropped for its text: a redraw changes nothing
    judges = rec.get("judges") or {}
    heard = [(judges.get(j) or {}).get("text") for j in ("turbo", "parakeet")]
    label = rec.get("text") or ""
    windows = [(judges.get(j) or {}).get("window") for j in ("turbo", "parakeet")]
    fired = {
        "loop": "acoustic:loop" in flags and (any(f.startswith("parakeet:") for f in flags)
                                              or any(f in flags for f in PARAKEET_OVER)),
        "number": "text:number" in flags or any(h and swaps(label, h) for h in heard),
        "inserted": "text:inserted" in flags,
        "window": all(w is not None and w >= WINDOW_BOTH for w in windows),
        "asr": bool(flags & {"text:preamble", "text:repeated"}),
        "dropped": audio_drop,
    }
    return [r for r in REASONS if fired[r]]


def read_queue(data: Path, audio_drops, only: set[str]) -> dict[str, list[str]]:
    """`clip_key -> reasons` over every category's ledger, streamed (600k records)."""
    queue = {}
    for cat in CATEGORIES:
        path = data / cat / LEDGER_NAME
        if not path.exists():
            raise click.ClickException(f"{path} missing: run 15_build_clip_ledger.py first")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    reasons = [r for r in reasons_of(rec, audio_drops) if r in only]
                    if reasons:
                        queue[rec["clip"]] = reasons
    return queue


def number_voice(mod, row: dict, audio_path: Path) -> dict:
    """The marker fields of the number voice rule (module docstring).

    Rung 0 of the stage ladder is the itf pair, so claiming it spent (`voice_tier=1`)
    makes every later escalation start at the fr pair. A clip read by an itf voice right
    now is also swapped at once to a `NUMBER_LADDER_FR_V2` hybrid, through the stage's
    own `swap_voice_for`; that ladder has one rung, so the tier it returns is 1 as well.
    """
    imp = row.get("improvement") or {}
    tier = max(int(imp.get("voice_tier") or 0), 1)
    if not (mod.voice_for_redraw(row) or "").startswith(ITF_PREFIX):
        return {"voice_tier": tier}
    probe = {**row, "improvement": {**imp, "voice_tier": 0}}
    nxt = mod.swap_voice_for(probe, audio_path, SimpleNamespace(voice_swap=True, term_voices={}),
                             ladder=NUMBER_LADDER_FR_V2)
    if nxt is None:
        return {"voice_tier": tier}
    return {"swap_voice": nxt[0], "voice_tier": tier}


def new_tally() -> SimpleNamespace:
    """The census `reopen_state` adds to, one Counter per breakdown."""
    return SimpleNamespace(counts=Counter(), by_voice=Counter(), by_cat=Counter(),
                           by_reason=Counter(), voice_moves=Counter())


def reopen_state(state: Path, base: Path, queue: dict[str, list[str]], reopen, mod,
                 apply: bool, tally: SimpleNamespace) -> list[dict]:
    """Mark every row of one stage-06 state file whose clip is in `queue`.

    `base` is the directory the rows' `audio_filepath` is relative to (the state's
    `HOTFIX_SOURCES` entry), so a row is matched on the same `clip_key` the ledger uses.
    The streaming, the reopenable statuses and the counts are `14.process`'s; this only
    supplies which rows (`select`) and the marker (`mark`). Returns one record per
    marked row: clip, reasons, and the voice swapped to if any.
    """
    picked, current = [], {}

    def select(row):
        audio = Path(resolve_audio(base, row.get("audio_filepath") or ""))
        reasons = queue.get(clip_key(audio))
        if not reasons:
            return None
        current.update(audio=audio, reasons=reasons)
        reading = (row.get("transcriptions") or {}).get(reopen.DEFAULT_JUDGE_KEY) or {}
        return mod.scores_from_reading(reading) if reading.get("text") else mod.Scores()

    def mark(row, scores):
        reasons = current["reasons"]
        extra = number_voice(mod, row, current["audio"]) if "number" in reasons else {}
        if extra.get("swap_voice"):
            tally.voice_moves[f"{mod.voice_for_redraw(row)} -> {extra['swap_voice']}"] += 1
        elif extra:
            tally.voice_moves["keeps voice, itf rung skipped"] += 1
        tally.by_reason.update(reasons)
        picked.append({"clip": clip_key(current["audio"]), "reasons": reasons,
                       **({"swap_voice": extra["swap_voice"]} if extra.get("swap_voice") else {})})
        mod.set_improvement(row, "pending_tts", scores, **extra)

    reopen.process(state, mod, select, apply, tally.counts, tally.by_voice, tally.by_cat,
                   mark=mark)
    return picked


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--reason", "only", multiple=True, type=click.Choice(REASONS),
              help="Only queue for these rules (repeatable). Default: all of them.")
@click.option("--out-stems", type=click.Path(path_type=Path),
              help="Write the queued stems here (one per line), for the driver's STEMS_FROM.")
@click.option("--out-queue", type=click.Path(path_type=Path),
              help="Write one JSON line per queued clip: clip, reasons, voice change.")
@click.option("--allow-stale-ledger", is_flag=True,
              help="Accept a ledger older than a state file (it may queue on old verdicts).")
@click.option("--apply", is_flag=True,
              help="Write the markers. Without it: census only, nothing is changed.")
def main(only, out_stems, out_queue, allow_stale_ledger, apply) -> None:
    reopen = load_script(_HERE / "14_reopen_loop_clips.py", "reopen_for_t6")
    sync = load_script(RELEASE / "03_sync_hotfix_results.py", "sync_for_t6")
    drop = load_script(RELEASE / "04_drop_bad_rows.py", "drop_for_t6")
    mod = reopen._load_pass()
    data = RELEASE / "data"
    states = [((RELEASE / src).resolve(), RELEASE / base) for src, base in sync.HOTFIX_SOURCES]

    ledgers = [data / c / LEDGER_NAME for c in CATEGORIES]
    oldest = min((p.stat().st_mtime for p in ledgers if p.exists()), default=0)
    newer = [p.name for p, _ in states if p.exists() and p.stat().st_mtime > oldest]
    if newer and not allow_stale_ledger:
        raise click.ClickException(f"state {newer} changed after the ledger was built: "
                                   "rerun 15_build_clip_ledger.py (or --allow-stale-ledger)")

    queue = read_queue(data, drop.REMEMBERED_REASONS, set(only or REASONS))
    logger.info(f"ledger: {len(queue)} clip(s) to redraw; by reason: "
                f"{dict(Counter(r for rs in queue.values() for r in rs).most_common())}")
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")

    tally = new_tally()
    picked = []
    for state, base in states:
        if not state.exists():
            logger.warning(f"{state} does not exist, skipping it")
            continue
        found = reopen_state(state, base, queue, reopen, mod, apply, tally)
        logger.info(f"{state.parent.name}/{state.name}: {len(found)} clip(s) marked pending_tts")
        picked.extend(found)
    counts = tally.counts

    # A PARROT clip sits in both state files (see clip_ledger's "Stage-06 overlap"), so
    # it is marked twice, once per file; the stems list must still name it once.
    stems = sorted({Path(p["clip"]).stem for p in picked})
    logger.info(f"{counts['rows']} row(s) read, {counts['reopened']} marked "
                f"({len(stems)} distinct clip(s)), {counts['already_queued']} already queued")
    for key in sorted(k for k in counts if k.startswith(("from::", "left::"))):
        logger.info(f"    {key}: {counts[key]}")
    logger.info(f"by reason: {dict(tally.by_reason.most_common())}")
    logger.info(f"by voice: {dict(tally.by_voice.most_common())}")
    logger.info(f"by category: {dict(tally.by_cat.most_common())}")
    logger.info(f"number voice rule: {dict(tally.voice_moves.most_common())}")
    missing = len(queue) - len(stems)
    if missing:
        logger.warning(f"{missing} queued clip(s) not marked (already pending, or not "
                       "reopenable from their status, or absent from the state)")
    for path, text in ((out_stems, "".join(f"{s}\n" for s in stems)),
                       (out_queue, "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked))):
        if path:
            path = path if path.is_absolute() else _HERE / path
            path.write_text(text, encoding="utf-8")
            logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
