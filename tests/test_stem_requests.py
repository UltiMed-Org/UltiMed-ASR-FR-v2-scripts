#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The shared "work on exactly these clips" lists (`utils/stem_requests.py`).

This is the handoff between QC stages: a cheap pass flags clips, an expensive pass is
told to look at only those. Two properties are worth pinning, because getting either
wrong is silent:

* a missing or unreadable list must NOT read as "nothing to do". An empty result looks
  exactly like a clean corpus, and a QC stage that reports a clean corpus because it
  could not find its input is the worst failure this code can have.
* a list aimed at one stage must not reach a clip of another. Stages number their terms
  independently, so `000123_0001_<slug>` is a stem each of them can mint.

Run: `python tests/test_stem_requests.py` (stdlib only).
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import stem_requests as sr  # noqa: E402


def write(tmp: Path, name: str, lines: list[str]) -> Path:
    path = tmp / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sidecar_contributes_only_its_flagged_rows() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        path = write(tmp, "qc_acoustic_sidecar.jsonl", [
            json.dumps({"stem": "000001_0001_loop", "stage": "01_dictionnary",
                        "looped": True}),
            json.dumps({"stem": "000002_0001_clean", "stage": "01_dictionnary",
                        "looped": False}),
            # No `looped` key at all: a hand-filtered sidecar, so it counts.
            json.dumps({"stem": "000003_0001_hand", "stage": "02_drugs"}),
            '{"stem": "000004_0001_torn", "stage": "01_dic',   # killed mid-write
        ])
        got = sr.load_stem_requests([path])
    assert got == {"01_dictionnary": {"000001_0001_loop"},
                   "02_drugs": {"000003_0001_hand"}}, got


def test_plain_list_applies_to_every_stage() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        path = write(tmp, "flags.txt", ["# the T1 flags", "", "000001_0001_a",
                                        "000002_0001_b"])
        got = sr.load_stem_requests([path])
    assert got == {"": {"000001_0001_a", "000002_0001_b"}}, got
    assert sr.wanted_for(got, "01_dictionnary") == frozenset(
        {"000001_0001_a", "000002_0001_b"}), "a bare list must reach any stage"


def test_several_files_are_unioned_and_counted_per_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        a = write(tmp, "a.jsonl", [json.dumps({"stem": "s1", "stage": "01_dictionnary",
                                               "looped": True})])
        b = write(tmp, "b.txt", ["s2"])
        seen = []
        got = sr.load_stem_requests([a, b], on_file=lambda p, r, pl: seen.append(
            (p.name, r, pl)))
    assert got == {"01_dictionnary": {"s1"}, "": {"s2"}}, got
    assert seen == [("a.jsonl", 1, 0), ("b.txt", 0, 1)], seen
    assert sr.wanted_for(got, "01_dictionnary") == frozenset({"s1", "s2"})
    assert sr.wanted_for(got, "02_drugs") == frozenset({"s2"})


def test_a_missing_list_is_fatal_not_empty() -> None:
    try:
        sr.load_stem_requests([Path("/nonexistent/flags.txt")])
    except FileNotFoundError as exc:
        assert "flags.txt" in str(exc), exc
    else:
        raise AssertionError("a missing list must raise, not silently check nothing")


def test_row_filter_keeps_a_stage_list_inside_its_stage() -> None:
    requests = {"01_dictionnary": {"000123_0001_term"}}
    assert sr.row_is_requested(requests,
                               "../01_dictionnary/generated_audios/000123_0001_term.flac")
    # Same stem, different stage: two stages index their terms independently, so this is
    # a different clip that must not be reopened by the dictionary's flag list.
    assert not sr.row_is_requested(
        requests, "../02_drugs/generated_audios/000123_0001_term.flac")


def test_row_filter_is_a_no_op_when_no_list_was_given() -> None:
    assert sr.row_is_requested({}, "../01_dictionnary/generated_audios/x.flac"), \
        "no list must mean no restriction, not an empty allow-list"


def test_row_filter_matches_a_bare_stem_anywhere() -> None:
    requests = {"": {"000123_0001_term"}}
    for path in ("../01_dictionnary/generated_audios/000123_0001_term.flac",
                 "../../PARROT/audios/000123_0001_term.flac"):
        assert sr.row_is_requested(requests, path), path
    assert not sr.row_is_requested(requests, "../01_dictionnary/x/000999_0001_other.flac")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
