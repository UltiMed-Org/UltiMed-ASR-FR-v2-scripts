#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1", "loguru>=0.7", "jiwer>=3.0", "requests>=2.31", "tqdm>=4.66",
#                 "soundfile>=0.12", "numpy"]
# ///
"""Does 13_refresh_stale_durations.py close the tail blind spot, and only that?

The bug it exists for: the sweeps rewrite a row's `transcriptions` but never its own
`duration`, so 208,235 of 604,337 v2 rows carried a v1-era length beside current
transcripts, and `_stt.tail_cer` returns None at or under 30 s. 5,250 clips that really
run past 30 s therefore never had their ENDING scored, which is the one defect shape the
whole-clip and window CERs dilute.

What is pinned here is both halves: the repair happens (duration, voice, tail, rtf) and
nothing else moves (`cer`, `cer_window`, `bad_run_chars`, `repeat_excess` are
length-independent and must come out byte-identical), plus the dry run really writes
nothing and a second run is a no-op.

Run: uv run tests/test_refresh_stale_durations.py
This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

from click.testing import CliRunner
from loguru import logger

HERE = Path(__file__).resolve().parent
_PATH = HERE.parent / "06_hotfixes" / "13_refresh_stale_durations.py"
_spec = importlib.util.spec_from_file_location("refresh_mod", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_stt = _mod._load_stt()

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


#: Long enough that the real clip has a tail (over `TAIL_SECONDS`) while the stale
#: duration claims it does not, which is the exact shape of the 5,250 missed clips.
LABEL = (
    "Compte rendu d'hospitalisation. Le patient a ete admis pour une douleur thoracique "
    "atypique evoluant depuis trois jours, sans irradiation ni signe vagal associe. "
    "L'electrocardiogramme est sans particularite et le dosage de troponine reste "
    "negatif a six heures. La radiographie thoracique ne montre pas de foyer. "
    "La sortie est autorisee le jour meme avec un controle en consultation a un mois "
    "et des consignes de surveillance remises au patient et a sa famille."
)
#: Same reading until the end, where it derails: the tail gate's whole purpose.
HEARD_BAD_END = LABEL[:-150] + "et des consignes de surveillance remises remises remises remises."


def rows_for(tmp: Path) -> tuple[Path, Path]:
    """A manifest and a state file that disagree, in the two directions that matter."""
    manifest = tmp / "full.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        # really long, the state row thinks it is short: tail never scored
        {"audio_filepath": "../PARHAF/000001_0000_x.flac", "duration": 61.4,
         "text": LABEL, "voice": "h_itf_arm"},
        # really short, the state row thinks it is long: tail scored but unscorable
        {"audio_filepath": "../PARROT/000002_0000_y.flac", "duration": 12.0,
         "text": LABEL, "voice": "fr_female"},
        # agrees already: must come out untouched
        {"audio_filepath": "../dictionary/000003_0000_z.flac", "duration": 20.0,
         "text": LABEL, "voice": "fr_male"},
    ]), encoding="utf-8")

    def reading(text, **over):
        base = {"text": text, "cer": 0.0123, "cer_window": 0.0456, "bad_run_chars": 7,
                "repeat_excess": 2, "stt_seconds": 10.0, "stt_rtf": 0.5}
        base.update(over)
        return base

    state = tmp / "full.stt.jsonl"
    state.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"audio_filepath": "../PARHAF/000001_0000_x.flac", "duration": 20.0, "text": LABEL,
         "transcriptions": {"turbo": reading(HEARD_BAD_END), "parakeet": reading(LABEL)}},
        {"audio_filepath": "../PARROT/000002_0000_y.flac", "duration": 63.3, "text": LABEL,
         "transcriptions": {"turbo": reading(LABEL, cer_tail=0.3333)}},
        {"audio_filepath": "../dictionary/000003_0000_z.flac", "duration": 20.0,
         "text": LABEL, "voice": "fr_male",
         "transcriptions": {"turbo": reading(LABEL)}},
    ]), encoding="utf-8")
    return manifest, state


def run(manifest: Path, state: Path, *, apply: bool) -> tuple[object, str]:
    """``(click result, everything the script logged)``.

    The log is captured through a loguru sink rather than read off `result.output`: the
    script reports with loguru, which writes to stderr, and CliRunner only collects
    stdout. Asserting on the sink is also what checks the real reporting path.
    """
    args = ["--manifest", str(manifest), "--state", str(state)] + (["--apply"] if apply else [])
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        result = CliRunner().invoke(_mod.main, args, catch_exceptions=False)
    finally:
        logger.remove(sink)
    return result, "".join(lines)


def read(state: Path) -> list[dict]:
    return [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    print("13_refresh_stale_durations: the tail blind spot, and nothing else")

    # Guard the fixture: if the derailed ending did not actually move the tail score
    # there would be nothing to detect, and a change to the scorer could make that true.
    tail = _stt.tail_cer(LABEL, HEARD_BAD_END, 61.4)
    check("the derailed ending scores over the tail gate",
          tail is not None and tail >= _stt.DEFAULT_TAIL_CER_GATE, str(tail))
    check("and at 20 s there is no tail to score at all",
          _stt.tail_cer(LABEL, HEARD_BAD_END, 20.0) is None)

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        manifest, state = rows_for(tmp)
        before = state.read_text(encoding="utf-8")

        result, log = run(manifest, state, apply=False)
        check("the dry run exits 0", result.exit_code == 0, str(result.exit_code))
        check("and writes nothing", state.read_text(encoding="utf-8") == before)
        check("and leaves no .tmp behind", not list(tmp.glob("*.tmp")))
        check("and still reports the missed tail", "tail_added_OVER_GATE" in log, log[-400:])

        check("--apply exits 0", run(manifest, state, apply=True)[0].exit_code == 0)
        rows = read(state)

        long_row, short_row, same_row = rows
        check("the stale long duration is refreshed", long_row["duration"] == 61.4,
              str(long_row["duration"]))
        check("the missing voice is filled from the manifest",
              long_row.get("voice") == "h_itf_arm", str(long_row.get("voice")))
        turbo = long_row["transcriptions"]["turbo"]
        check("the clip that was never tail-checked now is",
              turbo.get("cer_tail") is not None and turbo["cer_tail"] >= _stt.DEFAULT_TAIL_CER_GATE,
              str(turbo.get("cer_tail")))
        check("the clean reading of the same clip gets a tail under the gate",
              (long_row["transcriptions"]["parakeet"].get("cer_tail") or 0) < _stt.DEFAULT_TAIL_CER_GATE,
              str(long_row["transcriptions"]["parakeet"].get("cer_tail")))
        check("stt_rtf follows the new duration",
              turbo["stt_rtf"] == round(10.0 / 61.4, 5), str(turbo["stt_rtf"]))

        # The length-independent metrics have exactly one owner, and it is not this script.
        check("cer, cer_window, bad_run_chars and repeat_excess are untouched",
              (turbo["cer"], turbo["cer_window"], turbo["bad_run_chars"], turbo["repeat_excess"])
              == (0.0123, 0.0456, 7, 2), str(turbo))

        check("a row that is really short loses its unscorable tail",
              short_row["transcriptions"]["turbo"].get("cer_tail") is None,
              str(short_row["transcriptions"]["turbo"].get("cer_tail")))
        check("and its duration is refreshed down", short_row["duration"] == 12.0,
              str(short_row["duration"]))

        check("the already-correct row is left exactly as it was",
              same_row == json.loads(before.splitlines()[2]), str(same_row))

        after_first = state.read_text(encoding="utf-8")
        run(manifest, state, apply=True)
        check("a second --apply is a no-op", state.read_text(encoding="utf-8") == after_first)

        # A row the manifest does not know about must survive untouched rather than
        # losing its duration: the manifests are rebuilt per release, this file is not.
        orphan = tmp / "orphan.stt.jsonl"
        line = json.dumps({"audio_filepath": "../dictionary/999999_0000_gone.flac",
                           "duration": 9.5, "text": LABEL,
                           "transcriptions": {"turbo": {"text": LABEL, "cer": 0.0}}},
                          ensure_ascii=False) + "\n"
        orphan.write_text(line, encoding="utf-8")
        _, orphan_log = run(manifest, orphan, apply=True)
        check("a clip missing from the manifest is reported, not rewritten",
              orphan.read_text(encoding="utf-8") == line
              and "clip_not_in_manifest" in orphan_log, orphan_log[-300:])

    if failures:
        print(f"\nFAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
