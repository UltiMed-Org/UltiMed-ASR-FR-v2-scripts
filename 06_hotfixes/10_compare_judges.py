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
"""Compare two judges' readings of the same clips, i.e. read out the T2 second opinion.

The QC phase scores every clip with whisper `ggml-large-v3-turbo` (T1), then re-reads the
clips that pass flagged with non-turbo `ggml-large-v3` (T2), which is slower and better.
Both readings land on the same row, under their own key in `transcriptions`, because the
pass stores readings by model name:

    row["transcriptions"]["whisper.cpp/ggml-large-v3-turbo"] = {"cer": 0.01, ...}
    row["transcriptions"]["whisper.cpp/ggml-large-v3"]       = {"cer": 0.09, ...}

Nothing reads those two side by side, which is the whole point of the second opinion, so
this script does: it puts each reading through the same four gates the pass judges on
(`is_bad`, imported, not re-spelled) and sorts the clips into the four cells of the
agreement table. The cell that matters is **only the second judge says bad**: those are
the clips T1 waved through and a better judge would not, i.e. the false negatives the
whole T2 pass exists to find.

The second judge does not have to be the non-turbo whisper. `--key-b
onnx-asr/parakeet-tdt-0.6b-v3-ultra` reads out the parakeet sweep instead, and that is
the pairing that finds a defect class no whisper can see: a take that says a clause
twice. Whisper's decoder collapses repeated speech, so a looping clip reads CER-clean to
both whisper judges, while parakeet writes the repetition out and its `repeat_excess`
counts it. That count is the fifth gate (`--repeat-threshold`, default
`DEFAULT_REPEAT_GATE`), and it is why this script's default `--key-b` is not the whole
story:

    # the loop sweep: 1,391 clips of the v2 corpus at rx>=4, 1,272 of them invisible to turbo
    uv run 10_compare_judges.py --key-b onnx-asr/parakeet-tdt-0.6b-v3-ultra \
        --repeat-threshold 4 --out-stems loop_flags.txt

It changes no state. The two artefacts it writes are a detailed `.jsonl` (one entry per
disagreement, both transcripts next to the source text, so the call can be made by
reading rather than by trusting a number) and a plain stem list that the repair round
takes verbatim:

    uv run 10_compare_judges.py                       # report + both artefacts
    uv run 10_compare_judges.py --dry-run --samples 20  # just look
    # then, to redraw what the second judge rejected:
    sudo -E MODE=alternate STEMS_FROM=second_opinion_flags.txt ./driver.sh

Note the stem list names clips whose SHIPPED audio the second judge rejects, which is not
the same as T1's flag list: a clip T1 flagged and then successfully redrew reads clean
here, and a clip T1 never flagged cannot appear at all (T2 only re-read the flagged
slice). So this list is what is still wrong after one full alternation, not a re-run of it.

This file was written with Claude Code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent

# The gates, the Scores tuple and the reading -> Scores mapper all live with the pass that
# writes the readings; importing them is what keeps a second judge judged by the same
# rules as the first (the repo forbids duplicating them).
_recimp = None  # filled by _load_pass(), kept module-level so tests can stub it


def _load_pass():
    """Import `01_recursive_improvement.py` (its filename is not a module name)."""
    global _recimp
    if _recimp is None:
        import importlib.util

        path = _HERE / "01_recursive_improvement.py"
        spec = importlib.util.spec_from_file_location("recimp_for_compare", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _recimp = mod
    return _recimp


DEFAULT_STATES = ("improved/full.stt.jsonl", "improved_parrot/full.stt.jsonl")
KEY_TURBO = "whisper.cpp/ggml-large-v3-turbo"
KEY_LARGE_V3 = "whisper.cpp/ggml-large-v3"


def reading_is_usable(reading: dict) -> bool:
    """A reading with no transcript judges nothing: the clip is unscored by that judge,
    which is neither agreement nor disagreement and must not be counted as either."""
    text = reading.get("text")
    return not reading.get("error") and bool(text and text.strip())


def gates_from(cer: float, tail: float, window: float, bad_run: int,
               repeat: int = 0) -> SimpleNamespace:
    """`is_bad` reads its thresholds off a cfg object; this is the smallest one."""
    return SimpleNamespace(cer_threshold=cer, tail_cer_threshold=tail,
                           window_cer_threshold=window, bad_run_threshold=bad_run,
                           repeat_threshold=repeat)


def which_gates(scores, gates) -> list[str]:
    """Which of the four gates this reading trips, for the report. `is_bad` is the OR of
    them; a repair decision wants to know WHICH, because `bad_run` (a phrase skipped) and
    `cer` (the whole clip off) call for different things."""
    fired = []
    if scores.cer is not None and scores.cer >= gates.cer_threshold:
        fired.append("cer")
    if scores.tail is not None and scores.tail >= gates.tail_cer_threshold:
        fired.append("tail")
    if scores.window is not None and scores.window >= gates.window_cer_threshold:
        fired.append("window")
    if scores.bad_run is not None and scores.bad_run >= gates.bad_run_threshold:
        fired.append("bad_run")
    if (gates.repeat_threshold > 0 and scores.repeat is not None
            and scores.repeat >= gates.repeat_threshold):
        fired.append("repeat")
    return fired


def compare(states: list[Path], key_a: str, key_b: str,
            gates: SimpleNamespace) -> dict:
    """Walk the state files once and sort every row carrying BOTH readings into the
    agreement table. Returns the counts plus the per-clip detail of every disagreement."""
    mod = _load_pass()
    counts = {"rows": 0, "both_judges": 0, "only_a_read": 0, "neither_read": 0,
              "both_clean": 0, "both_bad": 0, "only_b_bad": 0, "only_a_bad": 0}
    detail: list[dict] = []
    for state in states:
        if not state.exists():
            logger.warning(f"{state} does not exist, skipping it")
            continue
        with state.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["rows"] += 1
                readings = row.get("transcriptions") or {}
                a, b = readings.get(key_a) or {}, readings.get(key_b) or {}
                a_ok, b_ok = reading_is_usable(a), reading_is_usable(b)
                if not b_ok:
                    counts["only_a_read" if a_ok else "neither_read"] += 1
                    continue
                if not a_ok:
                    # B read it and A did not: no comparison to make, and T2 was pointed
                    # at A's flags, so this means A's reading was lost, not never taken.
                    counts["neither_read"] += 1
                    continue
                counts["both_judges"] += 1
                sa, sb = mod.scores_from_reading(a), mod.scores_from_reading(b)
                bad_a, bad_b = mod.is_bad(sa, gates), mod.is_bad(sb, gates)
                if bad_a and bad_b:
                    cell = "both_bad"
                elif bad_b:
                    cell = "only_b_bad"
                elif bad_a:
                    cell = "only_a_bad"
                else:
                    cell = "both_clean"
                counts[cell] += 1
                if cell == "both_clean":
                    continue
                detail.append({
                    "audio_filepath": row.get("audio_filepath"),
                    "stem": Path(row.get("audio_filepath") or "").stem,
                    "state_file": str(state),
                    "cell": cell,
                    "duration": row.get("duration"),
                    "improvement": row.get("improvement"),
                    "voice": row.get("voice"),
                    "cfg_alpha": row.get("cfg_alpha"),
                    "source": row.get("asr_training_source") or row.get("text"),
                    "gates_a": which_gates(sa, gates),
                    "gates_b": which_gates(sb, gates),
                    "scores_a": sa._asdict(),
                    "scores_b": sb._asdict(),
                    "text_a": a.get("text"),
                    "text_b": b.get("text"),
                })
    return {"counts": counts, "detail": detail}


def worst_first(detail: list[dict]) -> list[dict]:
    """Sort by the second judge's whole-clip CER, worst first, so `--samples` shows the
    clips most worth listening to rather than the first ones on disk."""
    return sorted(detail, key=lambda d: (d["scores_b"].get("cer") or 0.0), reverse=True)


def render(result: dict, key_a: str, key_b: str, samples: int) -> None:
    c = result["counts"]
    logger.info(f"{c['rows']} row(s) read; {c['both_judges']} carry both judges "
                f"({c['only_a_read']} only '{key_a}', {c['neither_read']} neither)")
    logger.info("agreement table over the clips both judges read:")
    logger.info(f"  both clean                 : {c['both_clean']:>6}")
    logger.info(f"  only '{key_b}' says bad : {c['only_b_bad']:>6}"
                "   <- the false negatives T2 exists to find")
    logger.info(f"  only '{key_a}' says bad : {c['only_a_bad']:>6}"
                "   <- the first judge flagged it, the better one forgives")
    logger.info(f"  both bad                   : {c['both_bad']:>6}"
                "   <- still defective after a full alternation")
    gate_tally: dict[str, int] = {}
    for entry in result["detail"]:
        if entry["cell"] in ("only_b_bad", "both_bad"):
            for gate in entry["gates_b"]:
                gate_tally[gate] = gate_tally.get(gate, 0) + 1
    if gate_tally:
        pretty = ", ".join(f"{g}={n}" for g, n in sorted(gate_tally.items()))
        logger.info(f"  gates the second judge trips (a clip can trip several): {pretty}")
    for entry in worst_first([e for e in result["detail"]
                              if e["cell"] in ("only_b_bad", "both_bad")])[:samples]:
        sb = entry["scores_b"]
        logger.warning(
            f"SECOND OPINION {entry['stem']} cell={entry['cell']} "
            f"gates={'+'.join(entry['gates_b']) or 'none'} "
            f"cer={sb.get('cer')} window={sb.get('window')} "
            f"bad_run={sb.get('bad_run')} repeat={sb.get('repeat')}")
        print(f"  source     : {entry['source']}")
        print(f"  {key_a:<34}: {entry['text_a']}")
        print(f"  {key_b:<34}: {entry['text_b']}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--state", "states", multiple=True, type=click.Path(path_type=Path),
              help=f"State file(s) to read. Default: {', '.join(DEFAULT_STATES)}")
@click.option("--key-a", default=KEY_TURBO, show_default=True,
              help="The first judge, i.e. the key T1's readings are stored under.")
@click.option("--key-b", default=KEY_LARGE_V3, show_default=True,
              help="The second judge (T2).")
@click.option("--cer-threshold", default=None, type=float,
              help="Whole-clip CER gate (default: the pass's own).")
@click.option("--tail-cer-threshold", default=None, type=float)
@click.option("--window-cer-threshold", default=None, type=float)
@click.option("--bad-run-threshold", default=None, type=int)
@click.option("--repeat-threshold", default=None, type=int,
              help="Loop gate: `repeat_excess` >= this many word windows heard twice. "
                   "Only parakeet readings carry the count, so it is inert on a "
                   "whisper-vs-whisper comparison (default: the stage's own gate).")
@click.option("--samples", default=10, show_default=True,
              help="How many of the worst disagreements to print in full (0 = none).")
@click.option("--out-jsonl", default="judge_disagreements.jsonl", show_default=True,
              type=click.Path(path_type=Path),
              help="Per-clip detail of every disagreement.")
@click.option("--out-stems", default="second_opinion_flags.txt", show_default=True,
              type=click.Path(path_type=Path),
              help="Stem list of the clips the second judge rejects, for STEMS_FROM.")
@click.option("--dry-run", is_flag=True, help="Report only, write nothing.")
def main(states, key_a, key_b, cer_threshold, tail_cer_threshold,
         window_cer_threshold, bad_run_threshold, repeat_threshold, samples, out_jsonl,
         out_stems, dry_run) -> None:
    """Read out the T2 second opinion: where do the two judges disagree?"""
    stt = _load_pass()._stt
    gates = gates_from(
        stt.DEFAULT_CER_GATE if cer_threshold is None else cer_threshold,
        stt.DEFAULT_TAIL_CER_GATE if tail_cer_threshold is None else tail_cer_threshold,
        stt.DEFAULT_WINDOW_CER_GATE if window_cer_threshold is None
        else window_cer_threshold,
        stt.DEFAULT_BAD_RUN_GATE if bad_run_threshold is None else bad_run_threshold,
        stt.DEFAULT_REPEAT_GATE if repeat_threshold is None else repeat_threshold)
    logger.info(f"gates: cer>={gates.cer_threshold} tail>={gates.tail_cer_threshold} "
                f"window>={gates.window_cer_threshold} "
                f"bad_run>={gates.bad_run_threshold} "
                f"repeat>={gates.repeat_threshold}")
    paths = [Path(s) for s in states] or [_HERE / s for s in DEFAULT_STATES]
    result = compare(paths, key_a, key_b, gates)
    render(result, key_a, key_b, samples)

    rejected = [e for e in result["detail"] if e["cell"] in ("only_b_bad", "both_bad")]
    if dry_run:
        logger.info("--dry-run: writing nothing")
        return
    out_jsonl = out_jsonl if out_jsonl.is_absolute() else _HERE / out_jsonl
    out_stems = out_stems if out_stems.is_absolute() else _HERE / out_stems
    out_jsonl.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                for e in worst_first(result["detail"])),
                         encoding="utf-8")
    logger.info(f"wrote {len(result['detail'])} disagreement(s) to {out_jsonl}")
    header = (f"# clips '{key_b}' rejects, from 10_compare_judges.py\n"
              f"# {len(rejected)} clip(s): "
              f"{result['counts']['only_b_bad']} the first judge waved through, "
              f"{result['counts']['both_bad']} both judges reject\n"
              "# feed it back with: STEMS_FROM=second_opinion_flags.txt\n")
    out_stems.write_text(header + "".join(f"{e['stem']}\n" for e in worst_first(rejected)),
                         encoding="utf-8")
    logger.info(f"wrote {len(rejected)} stem(s) to {out_stems}")


if __name__ == "__main__":
    main()
