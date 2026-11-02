"""Stdlib tests for utils/hotfix_states: the one map from each stage-06 state file to the
directory its audio paths resolve against, and the scripts that import it instead of
carrying their own copy.

Run: python tests/test_hotfix_states.py

Written with Claude Code.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
import hotfix_states as hs  # noqa: E402
from nemo_manifest import resolve_audio  # noqa: E402


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_each_state_resolves_its_own_spelling_to_the_same_clip():
    """The PARROT state writes ``../../PARROT/x``, the root one ``../PARROT/x``: each,
    joined onto its own base, must name the same file, which is the property the
    2026-11-02 bug broke (a PARROT path joined onto the root base left the dataset)."""
    root = hs.RELEASE / hs.STATE_AUDIO_BASES["improved/full.stt.jsonl"]
    parrot = hs.RELEASE / hs.STATE_AUDIO_BASES["improved_parrot/full.stt.jsonl"]
    clip = "000041_0000_parrot_60_c3.flac"
    assert resolve_audio(root, f"../PARROT/{clip}") == resolve_audio(parrot, f"../../PARROT/{clip}")
    assert resolve_audio(root, f"../../PARROT/{clip}") != resolve_audio(parrot, f"../../PARROT/{clip}")


def test_default_states_and_paths_follow_the_map():
    assert hs.DEFAULT_STATES == ("improved/full.stt.jsonl", "improved_parrot/full.stt.jsonl")
    assert hs.default_state_paths() == [hs.HOTFIXES / s for s in hs.DEFAULT_STATES]


def test_audio_base_known_and_unknown_state():
    assert hs.audio_base(hs.HOTFIXES / "improved_parrot" / "full.stt.jsonl") == \
        hs.RELEASE / "data" / "NeMO_files" / "PARROT"
    # the same file named through a relative path still matches
    rel = os.path.relpath(hs.HOTFIXES / "improved" / "full.stt.jsonl")
    assert hs.audio_base(Path(rel)) == hs.RELEASE / "data" / "NeMO_files"
    with tempfile.TemporaryDirectory() as td:
        assert hs.audio_base(Path(td) / "full.stt.jsonl") is None


def test_importers_use_the_map():
    """03_sync's sources are derived from the map (same order, same bases), and the
    stage-06 scripts that used to carry a copy now hold the map's tuple."""
    sync = _load("99_hf_release/03_sync_hotfix_results.py", "sync_for_states")
    assert sync.HOTFIX_SOURCES == (
        ("../06_hotfixes/improved/full.stt.jsonl", "data/NeMO_files"),
        ("../06_hotfixes/improved_parrot/full.stt.jsonl", "data/NeMO_files/PARROT"),
    )
    refresh = _load("06_hotfixes/13_refresh_stale_durations.py", "refresh_for_states")
    assert refresh.DEFAULT_STATES is hs.DEFAULT_STATES


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
