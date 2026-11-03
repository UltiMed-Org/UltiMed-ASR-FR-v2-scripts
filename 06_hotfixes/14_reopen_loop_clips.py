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
"""Queue the clips that LOOP for a redraw: the one defect no whisper judge can flag.

A voxtral take sometimes says a clause twice. Whisper's decoder collapses repeated
speech, so such a clip comes back as a clean transcript and the four CER gates see
nothing: measured on 2026-10-27, 1,272 looping clips of the v2 corpus read clean to
turbo, which is the key the release's `stt_transcript` / `cer` columns are synced from.
Parakeet transcribes the repeat verbatim, and `repeat_excess` counts it.

So the clips exist, the evidence for them is already on their rows, and no pass can
reopen them: `needs_work` queues a clip at `pending_tts`, nothing puts a clip there but
a judge calling it bad, and the judge that does the calling cannot see this defect. That
is the gap this script closes. It writes no audio and judges nothing new; it moves the
verdict parakeet already reached into the marker the tts pass reads.

**The rule is a conjunction, and the measurement is why.** Neither half works alone:

  * parakeet's CER gates alone flag 8,096 clips corpus-wide, mostly its known
    false-positive families (French medical mishearings, mid-sentence English
    language-id flips). Redrawing those would churn good audio.
  * `repeat_excess >= 4` alone flags 1,391, of which 160 have clean parakeet CER, and
    those 160 are not defects: at equal duration their chars/second tracks the rx=0
    controls to within a few percent (-1.2% in the 20-30 s band, -1.0% at 30-40 s,
    -3.3% at 40-50 s), which is the signature of the metric's deliberate tolerance for
    labels that legitimately repeat a phrase ("1 comprime matin midi et soir", once per
    drug), not of a take that spent extra audio.

Their intersection, 1,244 clips, is where the excess audio actually is. chars/second
against length-matched rx=0 controls, which is judge-free (it compares the LABEL's
length to the clip's duration and asks no recognizer anything):

    band        n    p50 chars/s   vs control
    20-30 s    62      12.12         -14.7%
    30-40 s   207      11.36         -17.9%
    40-50 s   394      11.29         -18.3%
    50-65 s   418      10.71         -17.5%
    65 s +    158       9.78         -13.2%

A clip that needs 18% more audio than its length-matched peers to say the same number of
characters is saying something twice. Read one and it is plain:
`000766_0000_anatomopathologie_00193_t0_c0` (52.24 s, h_frm_ptf) has turbo and parakeet
both hearing "L'examen macroscopique retrouve trois fragments biopsiques de 9 mm ...
inclus en totalite dans un bloc." twice, where the non-turbo arbiter that promoted it
reads it once at CER 0.02.

**What the marker gets, and why the scores are parakeet's.** `set_improvement(row,
"pending_tts", orig)` is the pass's own entry point, so the clip re-enters the normal
alternation and nothing here duplicates its state machine. `orig` is the FLAGGING
judge's reading, which is what the marker is defined to hold ("what the clip looked like
when it was flagged", `scores_from_state`), and it is also what makes the repair
arithmetically possible: a draw is promoted when it beats `orig_cer` by
`--min-improvement` (0.02), so writing whisper's view of a looping clip (CER ~0.02,
because whisper cannot see the loop) would make every clip exhaust and then be DROPPED
by `99_hf_release/04_drop_bad_rows.py`. Parakeet's view (CER 0.276 at the median here)
is beaten by any clean draw.

`seeds_spent` is carried, never reset: the TTS is deterministic per (voice, seed), so a
reset seed ladder would redraw byte-identical audio and fail five times over.

The round this queues is a normal one, and the confirmation is a second parakeet pass,
because the judge that runs inside it still cannot see the defect it is repairing:

    uv run 06_hotfixes/14_reopen_loop_clips.py                      # dry run, counts only
    uv run 06_hotfixes/14_reopen_loop_clips.py --apply --out-stems loop_flags.txt
    sudo -E MODE=alternate STEMS_FROM=loop_flags.txt CFG_ALPHA_FLOOR=1.6 ./driver.sh
    # then refill parakeet on those stems and read the result out:
    uv run 06_hotfixes/10_compare_judges.py --key-b onnx-asr/parakeet-tdt-0.6b-v3-ultra \
        --repeat-threshold 4 --dry-run

`CFG_ALPHA_FLOOR=1.6` because 1,037 of the 1,244 are hybrid voices, whose usable cfg
band is 1.5-2.0 (measured 2026-10-13), and the default ladder descends to 1.2.

This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent

#: The judge whose readings carry the loop count. Only parakeet's do: `score_texts`
#: deliberately does not compute it, so every whisper reading in the corpus lacks the
#: field, and `12_merge_judge_sidecars.py` derives it when it merges the sweep.
DEFAULT_JUDGE_KEY = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
sys.path.insert(0, str(_HERE.parent / "utils"))
from hotfix_states import DEFAULT_STATES  # noqa: E402
#: The measured cut (see the module docstring). The live QC's gate is 1, which is right
#: for "ask a human to listen" and too tight for "spend five TTS draws".
DEFAULT_REPEAT_GATE = 4
#: Statuses a clip may be reopened FROM. `pending_tts` and `pending_stt` are already
#: queued, so re-marking them would reset nothing and only reshuffle their marker.
REOPENABLE = ("improved", "exhausted", "unflagged", "deterministic", None, "")
# A clip a human listened to and kept (`18_manual_pass.py`): no judge's flag reopens it.
MANUAL_PASS = "manual_pass"


def _load_pass():
    """Import `01_recursive_improvement.py`, which owns the marker and the gates."""
    path = _HERE / "01_recursive_improvement.py"
    spec = importlib.util.spec_from_file_location("recimp_for_reopen", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["recimp_for_reopen"] = mod
    spec.loader.exec_module(mod)
    return mod


def gates_of(stt, repeat: int) -> SimpleNamespace:
    """The stage's own four gates, plus the loop gate at the requested cut."""
    return SimpleNamespace(cer_threshold=stt.DEFAULT_CER_GATE,
                           tail_cer_threshold=stt.DEFAULT_TAIL_CER_GATE,
                           window_cer_threshold=stt.DEFAULT_WINDOW_CER_GATE,
                           bad_run_threshold=stt.DEFAULT_BAD_RUN_GATE,
                           repeat_threshold=repeat)


def loops(mod, row: dict, judge_key: str, gates: SimpleNamespace) -> object:
    """The flagging judge's Scores when this row's clip loops, else None.

    The conjunction of the docstring: the reading must trip at least one of the stage's
    four gates (so the judge genuinely rejects the clip, not just notices a repeat the
    label itself contains) AND its `repeat_excess` must reach the loop gate (so what the
    judge is rejecting is a repeat and not one of its own misreadings)."""
    reading = (row.get("transcriptions") or {}).get(judge_key)
    if not isinstance(reading, dict) or reading.get("error"):
        return None
    if not (reading.get("text") or "").strip():
        return None
    repeat = reading.get("repeat_excess")
    if repeat is None or repeat < gates.repeat_threshold:
        return None
    scores = mod.scores_from_reading(reading)
    # Without the loop gate: does the judge reject the clip on its own four numbers?
    if not mod.is_bad(scores, gates_of_four(gates)):
        return None
    return scores


def gates_of_four(gates: SimpleNamespace) -> SimpleNamespace:
    """`gates` with the loop gate off, i.e. the stage's four as the pass applies them."""
    return SimpleNamespace(cer_threshold=gates.cer_threshold,
                           tail_cer_threshold=gates.tail_cer_threshold,
                           window_cer_threshold=gates.window_cer_threshold,
                           bad_run_threshold=gates.bad_run_threshold,
                           repeat_threshold=0)


def category_of(audio: str) -> str:
    """`../PARHAF/000085_0000_x.flac` -> `PARHAF`, for the report's breakdown."""
    parts = Path(audio or "").parts
    return parts[-2] if len(parts) >= 2 else "?"


def stream_rewrite(state: Path, apply: bool, edit, tmp_suffix: str = ".reopen.tmp") -> None:
    """Feed every row of `state` to `edit(row, line)` and, under `apply`, rewrite the file.

    `edit` returns the row to keep, written back re-serialised (so it may mutate it), or
    None to drop it from the file; `line` is the row's raw text, for a caller that has to
    keep a dropped row verbatim (`20_drop_state_rows.py`). Kept rows go to
    `<state><tmp_suffix>`, which replaces `state` once the whole file has been read.

    Streamed rather than read_jsonl'd for the same reason `13_refresh_stale_durations.py`
    streams: `improved/full.stt.jsonl` is ~800 MB, and holding it as parsed JSON costs
    several GB of RAM.
    """
    tmp = state.with_suffix(state.suffix + tmp_suffix)
    out = tmp.open("w", encoding="utf-8") if apply else None
    try:
        with state.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = edit(json.loads(line), line)
                if out is not None and row is not None:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out is not None:
            out.close()
    if apply:
        tmp.replace(state)


def process(state: Path, mod, select, apply: bool, counts: Counter, by_voice: Counter,
            by_cat: Counter, mark=None) -> list[str]:
    """One state file: mark every clip `select` picks, stream-rewritten like its siblings.

    `select(row)` returns the scores to record as the clip's "when flagged" view (see the
    module docstring on why they must be the flagging judge's), or None to leave the row
    alone; here that is `loops`, and `16_reopen_from_ledger.py` passes its own. `mark(row,
    scores)` writes the marker, `pending_tts` with those scores by default; a caller that
    also has to pick the redraw's voice passes its own. The file walk is `stream_rewrite`.
    """
    if mark is None:
        def mark(row, scores):
            mod.set_improvement(row, "pending_tts", scores)
    stems: list[str] = []

    def edit(row: dict, _line: str) -> dict:
        counts["rows"] += 1
        scores = select(row)
        if scores is not None:
            status = (row.get("improvement") or {}).get("status")
            if (row.get("improvement") or {}).get(MANUAL_PASS):
                counts["left::manual_pass"] += 1
            elif status in REOPENABLE:
                counts["reopened"] += 1
                counts[f"from::{status or 'no marker'}"] += 1
                mark(row, scores)
                audio = row.get("audio_filepath") or ""
                stems.append(Path(audio).stem)
                by_voice[row.get("voice") or "?"] += 1
                by_cat[category_of(audio)] += 1
            else:
                counts["already_queued"] += 1
                counts[f"left::{status}"] += 1
        return row

    stream_rewrite(state, apply, edit)
    return stems


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--state", "states", multiple=True, type=click.Path(path_type=Path),
              help=f"State file(s). Default: {', '.join(DEFAULT_STATES)}")
@click.option("--judge-key", default=DEFAULT_JUDGE_KEY, show_default=True,
              help="The judge whose reading carries the loop count.")
@click.option("--repeat-threshold", default=DEFAULT_REPEAT_GATE, show_default=True,
              type=int, help="`repeat_excess` at or above which a clip is a loop.")
@click.option("--out-stems", default=None, type=click.Path(path_type=Path),
              help="Write the stem list here, for the round's STEMS_FROM.")
@click.option("--apply", is_flag=True,
              help="Actually rewrite the state files (default: report only).")
def main(states, judge_key, repeat_threshold, out_stems, apply) -> None:
    """Mark looping clips `pending_tts` so the next tts pass redraws them."""
    mod = _load_pass()
    gates = gates_of(mod._stt, repeat_threshold)
    logger.info(f"judge '{judge_key}', loop gate repeat_excess>={repeat_threshold}, "
                f"confirmed by the stage's four gates "
                f"(cer>={gates.cer_threshold} tail>={gates.tail_cer_threshold} "
                f"window>={gates.window_cer_threshold} "
                f"bad_run>={gates.bad_run_threshold})")
    if not apply:
        logger.warning("dry run: nothing will be written (pass --apply)")
    counts, by_voice, by_cat = Counter(), Counter(), Counter()
    stems: list[str] = []
    for state in ([Path(s) for s in states] or [_HERE / s for s in DEFAULT_STATES]):
        if not state.exists():
            logger.warning(f"{state} does not exist, skipping it")
            continue
        found = process(state, mod, lambda row: loops(mod, row, judge_key, gates), apply,
                        counts, by_voice, by_cat)
        logger.info(f"{state.name}: {len(found)} clip(s) marked pending_tts")
        stems.extend(found)
    logger.info(f"{counts['rows']} row(s) read, {counts['reopened']} reopened, "
                f"{counts['already_queued']} already queued")
    for key in sorted(k for k in counts if k.startswith(("from::", "left::"))):
        logger.info(f"  {key}: {counts[key]}")
    logger.info(f"by voice: {dict(by_voice.most_common())}")
    logger.info(f"by category: {dict(by_cat.most_common())}")
    if out_stems and stems:
        out_stems = out_stems if out_stems.is_absolute() else _HERE / out_stems
        out_stems.write_text("".join(f"{s}\n" for s in sorted(stems)), encoding="utf-8")
        logger.info(f"wrote {len(stems)} stem(s) to {out_stems}")
    elif out_stems:
        logger.warning("nothing to write: no clip matched")


if __name__ == "__main__":
    main()
