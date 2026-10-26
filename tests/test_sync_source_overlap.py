"""Stdlib test for the clips that two of 03_sync_hotfix_results.py's sources describe.

`HOTFIX_SOURCES` lists the root stage-06 output and the per-dataset ones. They OVERLAP:
the root run covers every clip of the corpus, and a per-dataset run re-reads a subset of
the same files, so a PARROT clip is reached both as `data/NeMO_files/PARROT/<clip>` and as
`<clip>` under `data/NeMO_files/PARROT`. `resolve_audio` normalises both to the same
absolute path, which is the point (the manifests join on it), and the index therefore has
one entry per clip no matter how many sources mention it.

What that collided with: `load_hotfix` used to assign `index[key] = {...}` outright, so the
source read LAST won every field. On 2026-10-25 the source read last was the one that knew
less, because 908 of its 1548 rows had lost their stored state when their text changed, and
126 clips the root run had regenerated shipped as `regenerated: false` /
`qc_status: "original"`. The CER columns were right, which is what kept it quiet: the
numbers described the improved audio while the provenance denied the improvement.

So this pins the merge rule `merge_hotfix` implements: fresher measurements win, a null
never erases a value, `regenerated` is the OR, and an absent marker never overrides a
present one while a present one does win.

Run: python tests/test_sync_source_overlap.py

This file was written with Claude Code.
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from test_sync_pass_guard import load_sync_module  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def state_row(name: str, *, status: str | None = None, cer: float = 0.01,
              text: str = "bonjour", voice: str | None = None,
              duration: float | None = 3.5, cfg_alpha: float | None = None) -> dict:
    """One row of a stage-06 `*.stt.jsonl`, as `load_hotfix` reads it."""
    row: dict = {
        "audio_filepath": name,
        "transcriptions": {"whisper-1": {"cer": cer, "cer_tail": None, "text": text,
                                         "n_stt_check": 1}},
    }
    if duration is not None:
        row["duration"] = duration
    if status is not None:
        row["improvement"] = {"status": status}
    if voice is not None:
        row["voice"] = voice
    if cfg_alpha is not None:
        row["cfg_alpha"] = cfg_alpha
    return row


def write_state(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build(tmp: Path, root_rows: list[dict], parrot_rows: list[dict]) -> Path:
    """The two state files where `HOTFIX_SOURCES` expects them, relative to the release
    stage root this returns."""
    root = tmp / "99_hf_release"
    root.mkdir(parents=True, exist_ok=True)
    write_state(tmp / "06_hotfixes" / "improved" / "full.stt.jsonl", root_rows)
    write_state(tmp / "06_hotfixes" / "improved_parrot" / "full.stt.jsonl", parrot_rows)
    return root


def test_the_later_source_cannot_erase_the_improvement(mod) -> None:
    """The exact 2026-10-25 case: the root run improved the clip, the PARROT run had
    dropped its state and reports nothing, and the clip must still ship as improved."""
    print("\na source that knows less cannot erase one that knows more")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = build(
            tmp,
            [state_row("PARROT/shared.flac", status="improved", cer=0.01,
                       text="the improved take", voice="fr_male", cfg_alpha=2.0),
             state_row("dictionary/root_only.flac", status="improved")],
            # No `improvement` key at all: this is a row whose stored state the pass
            # dropped ("manifest rebuilt?"), not a clip that was never flagged.
            [state_row("shared.flac", cer=0.02, text="re-read of the same audio"),
             state_row("parrot_only.flac", status="exhausted")],
        )
        index = mod.load_hotfix(root, "whisper.cpp/ggml-large-v3-turbo")
        shared = index[str(root / "data" / "NeMO_files" / "PARROT" / "shared.flac")]

        check("the overlapping clip is one entry, not two", len(index) == 3, str(len(index)))
        check("it still says the audio was regenerated", shared["regenerated"] is True,
              json.dumps(shared))
        check("and still carries the improved label", shared["qc_status"] == "improved",
              str(shared["qc_status"]))
        # The later source read the audio that is on disk now, so its measurement is the
        # one to publish. This half worked before the fix and must keep working.
        check("the fresher reading wins", shared["cer"] == 0.02, str(shared["cer"]))
        check("as does the fresher transcript",
              shared["stt_transcript"] == "re-read of the same audio",
              str(shared["stt_transcript"]))
        # ... but only where it HAS one. The PARROT row carries no voice and no
        # cfg_alpha, and a null there used to blank both columns for the release.
        check("a null does not erase the voice the earlier source knew",
              shared["voice"] == "fr_male", str(shared["voice"]))
        check("nor the cfg_alpha the clip was drawn at", shared["cfg_alpha"] == 2.0,
              str(shared["cfg_alpha"]))
        check("a clip only the root source has is untouched",
              index[str(root / "data" / "NeMO_files" / "dictionary" / "root_only.flac")]
              ["qc_status"] == "improved")
        check("a clip only the later source has is read normally",
              index[str(root / "data" / "NeMO_files" / "PARROT" / "parrot_only.flac")]
              ["qc_status"] == "exhausted")


def test_a_present_marker_is_the_fresher_judgement(mod) -> None:
    """The mirror case, which the fix must not break: when the later source DOES carry a
    marker it is the newer judgement and wins, so a clip whose shipped draw was re-judged
    and could not be saved ships as `exhausted` while still admitting it was regenerated.
    """
    print("\na present marker is the fresher judgement")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = build(
            tmp,
            [state_row("PARROT/rejudged.flac", status="improved", cer=0.01)],
            [state_row("rejudged.flac", status="exhausted", cer=0.21)],
        )
        index = mod.load_hotfix(root, "whisper.cpp/ggml-large-v3-turbo")
        entry = index[str(root / "data" / "NeMO_files" / "PARROT" / "rejudged.flac")]
        check("the newer label wins", entry["qc_status"] == "exhausted",
              str(entry["qc_status"]))
        check("while regenerated stays true, since the audio WAS replaced",
              entry["regenerated"] is True, json.dumps(entry))


def test_the_merge_rule_on_its_own(mod) -> None:
    """`merge_hotfix` directly, for the cases a two-file fixture states less clearly."""
    print("\nthe merge rule on its own")
    improved = {"cer": 0.01, "voice": "fr_male", "regenerated": True,
                "qc_status": "improved", "duration": 3.5}
    # A pending_* status is spelled `original` by load_hotfix (the clip on disk IS the
    # original mid-flight), so it must not be read as "never flagged" either.
    mid_flight = {"cer": 0.3, "voice": None, "regenerated": False,
                  "qc_status": "original", "duration": 3.5}
    merged = mod.merge_hotfix(improved, mid_flight)
    check("an interrupted run does not downgrade a finished one",
          merged["qc_status"] == "improved" and merged["regenerated"] is True,
          json.dumps(merged))
    check("the OR is symmetric",
          mod.merge_hotfix(mid_flight, improved)["regenerated"] is True)
    check("two originals stay original",
          mod.merge_hotfix(mid_flight, dict(mid_flight))["qc_status"] == "original")
    check("the prior is not mutated", improved["cer"] == 0.01, json.dumps(improved))


def test_each_judge_gets_its_own_columns(mod) -> None:
    """Per-judge columns, so a consumer sees WHICH model disagreed.

    The two judges fail in opposite directions: whisper's decoder collapses repeated
    speech, so a looping clip comes back transcribed once and scores clean (CER 0.0167
    where parakeet read 0.1796 on the same 269 flagged clips), while parakeet mishears
    this corpus's drug brand names at CER 0.08-0.14 on clean audio. A single blended
    number hides both, so each judge ships its own group and `cer_max` is the
    pessimistic read across them.
    """
    print("\neach judge gets its own columns")
    readings = {
        "whisper.cpp/ggml-large-v3-turbo": {
            "text": "  amoxicilline un gramme  ", "cer": 0.02, "cer_window": 0.05,
            "repeat_excess": 0,
        },
        "onnx-asr/parakeet-tdt-0.6b-v3-ultra": {
            "text": "amoxicilline un gramme un gramme un gramme", "cer": 0.18,
            "cer_window": 0.31, "repeat_excess": 4,
        },
        # The v1-era alias: present on old rows, and it must NOT become a column.
        "whisper-1": {"text": "v1 audio", "cer": 0.5, "cer_window": 0.6},
    }
    cols = mod.judge_columns(readings)

    check("whisper text is stripped", cols["whisper_stt"] == "amoxicilline un gramme",
          repr(cols["whisper_stt"]))
    check("whisper keeps its own cer", cols["whisper_cer"] == 0.02)
    check("parakeet keeps its own cer", cols["parakeet_cer"] == 0.18)
    check("the loop count comes from parakeet", cols["parakeet_repeat_excess"] == 4)
    check("whisper reads the loop as clean", cols["whisper_repeat_excess"] == 0)
    check("cer_max is the pessimistic read", cols["cer_max"] == 0.18, repr(cols["cer_max"]))
    check("cer_window_max is the pessimistic read", cols["cer_window_max"] == 0.31)
    # The non-turbo second opinion ran only over the flagged slice, so its columns must
    # be DECLARED and null rather than missing, or the parquet schema depends on which
    # rows the writer saw first.
    check("the second opinion is declared null",
          cols["whisper_v3_stt"] is None and cols["whisper_v3_cer"] is None)
    check("every judge column is present",
          set(mod.JUDGE_COLUMNS) <= set(cols),
          str(sorted(set(mod.JUDGE_COLUMNS) - set(cols))))
    check("the v1 alias is not a column",
          not any(k.startswith("whisper_1") or k.startswith("whisper-1") for k in cols))
    check("the v1 alias does not reach cer_max", cols["cer_max"] != 0.5)

    # A clip no judge read: nulls, and no 0.0 that would read as unanimous agreement.
    empty = mod.judge_columns({})
    check("an unjudged clip has no maxima",
          empty["cer_max"] is None and empty["cer_window_max"] is None,
          json.dumps(empty))
    check("the sync writes every judge column",
          set(mod.JUDGE_COLUMNS) <= set(mod.QC_FIELDS))


def main() -> None:
    mod = load_sync_module()
    test_the_later_source_cannot_erase_the_improvement(mod)
    test_a_present_marker_is_the_fresher_judgement(mod)
    test_the_merge_rule_on_its_own(mod)
    test_each_judge_gets_its_own_columns(mod)
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
