#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1", "loguru>=0.7", "jiwer>=3.0", "requests>=2.31", "tqdm>=4.66", "soundfile>=0.12", "numpy"]
# ///
"""Stdlib test for 08_validate_manifests.py's per-judge column checks.

The release ships one column group per judge (`whisper_*`, `whisper_v3_*`, `parakeet_*`)
plus `cer_max` / `cer_window_max` across them, because the two judges fail in opposite
directions: whisper's decoder collapses repeated speech, so a looping clip reads clean
(CER 0.0167 against parakeet's 0.1796 on the same 269 flagged clips), while parakeet
mishears this corpus's drug brand names at CER 0.08-0.14 on clean audio.

A maximum is a DERIVED column, which is the dangerous kind: nothing downstream
recomputes it, so a stale or invented value is a number a consumer filters on with no
judge behind it. This pins that the validator refuses one.

Run: uv run tests/test_validate_judge_maxima.py
This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PATH = HERE.parent / "99_hf_release" / "08_validate_manifests.py"
_spec = importlib.util.spec_from_file_location("validate_manifests_mod", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def row(**over) -> dict:
    """A row that passes every OTHER structural check, so only the maxima are on trial."""
    base = {
        "audio_filepath": "../PARHAF/000013_0000_x.flac",
        "duration": 20.3,
        "text": "le patient recoit de l amoxicilline",
        "asr_training_source": "le patient recoit de l amoxicilline",
        "whisper_cer": 0.02, "whisper_cer_window": 0.05,
        "whisper_v3_cer": None, "whisper_v3_cer_window": None,
        "parakeet_cer": 0.18, "parakeet_cer_window": 0.31,
        "cer_max": 0.18, "cer_window_max": 0.31,
    }
    base.update(over)
    return base


def errors_for(r: dict) -> list[str]:
    """Only the maxima errors, so an unrelated missing column cannot pass the test.

    `Findings` deduplicates by (check, kind) and only appends to `.rows` when a caller
    passes a structured record, which `check_structure` does not, so the findings have
    to be read off `.errors`' keys.
    """
    f = _mod.Findings()
    _mod.check_structure([r], "unit", f)
    return [kind for _check, kind in f.errors
            if "cer_max" in kind or "cer_window_max" in kind]


def main() -> int:
    print("per-judge maxima")
    check("a coherent row is accepted", errors_for(row()) == [], str(errors_for(row())))

    # The pessimistic read must come from a judge, not from the primary flat `cer`.
    bad = errors_for(row(cer_max=0.02))
    check("a maximum below a judge's own cer is refused", len(bad) == 1, str(bad))

    # Stale is the realistic failure: parakeet rescored, the maximum not recomputed.
    stale = errors_for(row(parakeet_cer=0.42))
    check("a maximum left stale by a rescore is refused", len(stale) == 1, str(stale))

    # A clip no judge read: null is the only honest value, 0.0 would read as unanimous.
    unjudged = dict(whisper_cer=None, whisper_v3_cer=None, parakeet_cer=None,
                    whisper_cer_window=None, whisper_v3_cer_window=None,
                    parakeet_cer_window=None)
    check("an unjudged clip may carry null maxima",
          errors_for(row(cer_max=None, cer_window_max=None, **unjudged)) == [])
    invented = errors_for(row(cer_max=0.0, cer_window_max=0.0, **unjudged))
    check("an unjudged clip may not carry 0.0 maxima", len(invented) == 2, str(invented))

    # A row from before the per-judge columns existed has no maxima at all, and must not
    # be reported as broken: that is what the OPTIONAL_KEYS warning is for.
    legacy = {k: v for k, v in row().items()
              if not k.startswith(("whisper", "parakeet")) and not k.endswith("_max")}
    check("a legacy row without the columns raises no maxima error",
          errors_for(legacy) == [], str(errors_for(legacy)))

    check("every judge column is declared optional-but-present",
          {"whisper_stt", "parakeet_cer", "whisper_v3_cer_window", "cer_max"}
          <= set(_mod.OPTIONAL_KEYS))

    if failures:
        print(f"\nFAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
