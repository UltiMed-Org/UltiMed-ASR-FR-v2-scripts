#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1", "loguru>=0.7", "jiwer>=3.0", "requests>=2.31", "tqdm>=4.66", "soundfile>=0.12", "numpy"]
# ///
"""Which reference does 08_validate_manifests.py's `scores` check recompute against?

It must be `text`, the label. It used to be `asr_training_source`, and that made the
check fire on 3,678 rows of the v2 full manifest, every one of them a false alarm:
05_normalize_text.py rewrites the SOURCE with pronunciation respellings meant for the
TTS (`oestrogènes` -> `éstrogènes`, `foetal` -> `fétal`), which `normalize_for_scoring`
does not fold away, so a recompute against it charged the respelling as a recognition
error. Measured over those 3,686 rows: 0 matched a recompute against the source, 3,684
matched one against `text`, and all 318 tail scores matched `text`.

A release gate that always fails is a gate nobody reads, so what is pinned here is both
halves: the respelled row passes, and a genuinely stale score is still caught.

Run: uv run tests/test_validate_score_reference.py
This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PATH = HERE.parent / "99_hf_release" / "08_validate_manifests.py"
_spec = importlib.util.spec_from_file_location("validate_scores_mod", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_stt = _mod._load_stt()

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


#: The real divergence, taken from ../acronyms and ../PARHAF rows of the v2 release: the
#: label keeps the French spelling, the shipped source carries the TTS respelling, and
#: the ASR judge writes the label's spelling because that is what the audio sounds like.
LABEL = ("Les récepteurs aux oestrogènes sont positifs et le bilan foetal est normal.")
SOURCE = ("Les récepteurs aux éstrogènes sont positifs et le bilan fétal est normal.")
HEARD = ("Les récepteurs aux oestrogènes sont positifs et le bilan foetal est normal.")


def row(**over) -> dict:
    stored = _stt.compute_metrics(LABEL, HEARD)[1]
    base = {"audio_filepath": "../PARHAF/000013_0000_x.flac", "duration": 20.3,
            "text": LABEL, "asr_training_source": SOURCE, "stt_transcript": HEARD,
            "stt_model": "whisper.cpp/ggml-large-v3-turbo", "qc_status": "regenerated",
            "cer": stored, "cer_tail": None}
    base.update(over)
    return base


def score_errors(r: dict) -> list[str]:
    f = _mod.Findings()
    _mod.check_scores([r], "unit", f, sample_percent=100.0, seed=0)
    return [kind for _check, kind in f.errors]


def main() -> int:
    print("the reference the scores check recomputes against")

    # Guard the fixture: if the scorer's own folds erased the respelling there would be
    # nothing to test, and a later change to normalize_for_scoring could make that true.
    cer_label = _stt.compute_metrics(LABEL, HEARD)[1]
    cer_source = _stt.compute_metrics(SOURCE, HEARD)[1]
    check("the respelling really moves the CER", cer_label != cer_source,
          f"both {cer_label}")
    check("and against the label the judge is nearly perfect", cer_label == 0.0,
          str(cer_label))

    check("a row whose score matches its LABEL passes", score_errors(row()) == [],
          str(score_errors(row())))

    # The same row scored against the source is what used to fail. Stating it as a value
    # pins the direction: the source inflates the CER, it does not merely differ.
    check("the source would have read worse", cer_source > cer_label,
          f"{cer_source} vs {cer_label}")

    # The check must still earn its keep: a genuinely stale score is caught.
    stale = score_errors(row(cer=0.42))
    check("a stale cer is still reported", len(stale) == 1, str(stale))
    check("and it is reported as a cer mismatch",
          stale and "cer does not match" in stale[0], str(stale))

    # A stale tail is its own finding, on a clip long enough to have a tail (over 30 s).
    long_row = row(duration=42.1, cer_tail=0.42)
    tail = [k for k in score_errors(long_row) if "cer_tail" in k]
    check("a stale cer_tail is reported too", len(tail) == 1, str(tail))

    # A row with no transcript is the structure check's business, not this one's.
    check("a row with no transcript is skipped here",
          score_errors(row(stt_transcript=None)) == [])
    # And a row with no label cannot be scored at all (the label is REQUIRED elsewhere).
    check("a row with no label is skipped here", score_errors(row(text=None)) == [])

    if failures:
        print(f"\nFAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
