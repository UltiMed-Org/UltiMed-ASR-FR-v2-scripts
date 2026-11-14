"""Stdlib test: the Parquet export carries every column the QC sync writes, typed.

`99_hf_release/03_sync_hotfix_results.py` writes the per-judge columns (`whisper_*`,
`whisper_v3_*`, `parakeet_*`) plus `cer_max` / `cer_window_max` into the manifests, and
the dataset card documents them as shipped. Until 2026-11-14 `scripts/build_parquet.py`
exported only the eight v1-era QC columns, so the judges reached the manifests and then
stopped there: a rebuilt release would have silently lacked what its card promised.

This pins the contract: every field in 03's `QC_FIELDS` has a dtype in build_parquet's
`QC_COLUMNS`, and the judge columns get the dtype their values actually have.

Run: python tests/test_parquet_judge_columns.py

This file was written with Claude Code.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

from script_module import load_script  # noqa: E402
from test_sync_pass_guard import load_sync_module  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def main() -> None:
    sync = load_sync_module()
    build = load_script(HERE.parent / "99_hf_release" / "scripts" / "build_parquet.py",
                        "build_parquet_for_judge_test")
    cols = build.QC_COLUMNS
    missing = sorted(set(sync.QC_FIELDS) - set(cols))
    check("every field the sync writes is exported", not missing, str(missing))
    for name in ("whisper", "whisper_v3", "parakeet"):
        check(f"{name}_stt is a string", cols.get(f"{name}_stt") == "string")
        check(f"{name}_cer is a float", cols.get(f"{name}_cer") == "float32")
        check(f"{name}_cer_window is a float", cols.get(f"{name}_cer_window") == "float32")
        check(f"{name}_repeat_excess is an int", cols.get(f"{name}_repeat_excess") == "int32")
    check("cer_max is a float", cols.get("cer_max") == "float32")
    check("cer_window_max is a float", cols.get("cer_window_max") == "float32")
    try:
        build.judge_dtype("whisper_mystery")
        check("an unknown judge suffix is refused", False)
    except ValueError:
        check("an unknown judge suffix is refused", True)
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
