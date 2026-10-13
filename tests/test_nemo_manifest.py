"""Stdlib tests for utils/nemo_manifest: split-spec parsing, the grouped,
duration-aware train/val/test splitter (atomic + distribute guarantees,
singletons by ratio, duration balance, determinism), and the duration cache's
size/mtime validation (a clip whose bytes changed must be re-probed).

Run: python tests/test_nemo_manifest.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from nemo_manifest import (  # noqa: E402
    SPLITS,
    assign_splits,
    audio_index,
    parse_split,
    probe_durations,
    relativize,
    summarize,
)


def _row(gid, mode, dur):
    return {"group_id": gid, "group_mode": mode, "duration": float(dur)}


def _by_group(rows, labels):
    """group_id -> set of split labels it landed in."""
    out = {}
    for r, lab in zip(rows, labels):
        out.setdefault(r["group_id"], set()).add(lab)
    return out


# --------------------------------------------------------------------------- #
# parse_split
# --------------------------------------------------------------------------- #
def test_parse_split_percentages():
    r = parse_split("80/10/10")
    assert abs(r["train"] - 0.8) < 1e-9 and abs(r["val"] - 0.1) < 1e-9 and abs(r["test"] - 0.1) < 1e-9


def test_parse_split_scale_invariant():
    assert parse_split("8/1/1") == parse_split("80/10/10") == parse_split("0.8/0.1/0.1")


def test_parse_split_disable_test():
    r = parse_split("90/10/0")
    assert r["test"] == 0.0 and abs(r["train"] - 0.9) < 1e-9


def test_parse_split_bad():
    for bad in ("80/20", "80/10/10/0", "a/b/c", "-1/1/1", "0/0/0"):
        try:
            parse_split(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


# --------------------------------------------------------------------------- #
# atomic: a whole group must land in exactly one split (no leakage)
# --------------------------------------------------------------------------- #
def test_atomic_groups_never_split():
    rows, i = [], 0
    for g in range(60):
        size = 1 + (g % 4)  # 1..4 chunks per document
        for _ in range(size):
            rows.append(_row(f"doc{g}", "atomic", 1.0 + (i % 7)))
            i += 1
    labels = assign_splits(rows, parse_split("80/10/10"))
    for gid, splits in _by_group(rows, labels).items():
        assert len(splits) == 1, f"atomic {gid} leaked across {splits}"
    # and all three splits are actually used
    assert set(labels) == set(SPLITS)


# --------------------------------------------------------------------------- #
# distribute: coverage guarantees
# --------------------------------------------------------------------------- #
def test_distribute_ge3_covers_every_split():
    rows = [_row("t", "distribute", 1.0) for _ in range(3)]
    labels = assign_splits(rows, parse_split("80/10/10"))
    assert set(labels) == set(SPLITS)  # one in each


def test_distribute_size2_has_train():
    rows = [_row("t", "distribute", 1.0), _row("t", "distribute", 1.0)]
    labels = assign_splits(rows, parse_split("80/10/10"))
    assert "train" in labels


def test_distribute_big_group_in_every_split():
    rows = [_row("t", "distribute", 1.0) for _ in range(11)]
    labels = assign_splits(rows, parse_split("80/10/10"))
    assert set(labels) == set(SPLITS)
    # bulk still goes to train
    assert labels.count("train") >= 7


def test_strict_vs_best_effort_coverage_ratio():
    # 1000 terms of 3 variants each. Strict floors val/test at one-per-term
    # (~33%); best-effort keeps only >=1-in-train, so val/test hit the target.
    rows = [_row(f"t{i}", "distribute", 1.0) for i in range(1000) for _ in range(3)]
    strict = summarize(rows, assign_splits(rows, parse_split("80/10/10"),
                                           by_duration=False, strict_coverage=True))
    best = summarize(rows, assign_splits(rows, parse_split("80/10/10"),
                                         by_duration=False, strict_coverage=False))
    assert strict["val"]["count_pct"] > 30 and strict["test"]["count_pct"] > 30
    assert abs(best["val"]["count_pct"] - 10) <= 3 and abs(best["test"]["count_pct"] - 10) <= 3
    assert abs(best["train"]["count_pct"] - 80) <= 3


def test_best_effort_still_guarantees_train():
    rows = [_row(f"t{g}", "distribute", 1.0) for g in range(500) for _ in range(2 + g % 3)]
    labels = assign_splits(rows, parse_split("80/10/10"), strict_coverage=False)
    for gid, splits in _by_group(rows, labels).items():
        assert "train" in splits, f"{gid} lost its train guarantee under best-effort"


def test_distribute_respects_disabled_split():
    # test disabled -> a >=3 group covers only train+val, never test
    rows = [_row("t", "distribute", 1.0) for _ in range(5)]
    labels = assign_splits(rows, parse_split("80/20/0"))
    assert "test" not in labels
    assert "train" in labels and "val" in labels


# --------------------------------------------------------------------------- #
# singletons distribute by ratio (not all forced to train)
# --------------------------------------------------------------------------- #
def test_singletons_follow_ratio():
    rows = [_row(f"s{i}", "distribute", 1.0) for i in range(1000)]
    labels = assign_splits(rows, parse_split("80/10/10"), by_duration=False)
    agg = summarize(rows, labels)
    assert 78 <= agg["train"]["count_pct"] <= 82
    assert 8 <= agg["val"]["count_pct"] <= 12
    assert 8 <= agg["test"]["count_pct"] <= 12


# --------------------------------------------------------------------------- #
# duration balance
# --------------------------------------------------------------------------- #
def test_duration_balance_singletons():
    # widely varying clip lengths; by_duration should balance seconds, not rows
    rows = [_row(f"s{i}", "distribute", 1 + (i % 20)) for i in range(2000)]
    labels = assign_splits(rows, parse_split("80/10/10"), by_duration=True)
    agg = summarize(rows, labels)
    for s, target in (("train", 80), ("val", 10), ("test", 10)):
        assert abs(agg[s]["duration_pct"] - target) <= 2.0, (s, agg[s]["duration_pct"])


def test_duration_mode_beats_count_on_skew():
    # one giant clip per group; balancing by count leaves duration lopsided
    rows = [_row(f"s{i}", "distribute", (100 if i % 10 == 0 else 1)) for i in range(500)]
    by_count = summarize(rows, assign_splits(rows, parse_split("80/10/10"), by_duration=False))
    by_dur = summarize(rows, assign_splits(rows, parse_split("80/10/10"), by_duration=True))
    off_count = abs(by_count["val"]["duration_pct"] - 10) + abs(by_count["test"]["duration_pct"] - 10)
    off_dur = abs(by_dur["val"]["duration_pct"] - 10) + abs(by_dur["test"]["duration_pct"] - 10)
    assert off_dur < off_count


# --------------------------------------------------------------------------- #
# mixed atomic + distribute (the parent-combiner case) + determinism
# --------------------------------------------------------------------------- #
def test_mixed_and_deterministic():
    rows = []
    for g in range(40):
        for _ in range(1 + g % 3):
            rows.append(_row(f"doc{g}", "atomic", 1.0 + g % 5))
    for t in range(40):
        for _ in range(2 + t % 4):
            rows.append(_row(f"term{t}", "distribute", 1.0 + t % 3))
    a = assign_splits(rows, parse_split("80/10/10"))
    b = assign_splits(rows, parse_split("80/10/10"))
    assert a == b  # deterministic
    for gid, splits in _by_group(rows, labels=a).items():
        if gid.startswith("doc"):
            assert len(splits) == 1  # atomic stays whole


def test_no_stratify_ignores_groups():
    rows = [_row("one", "atomic", 1.0) for _ in range(1000)]
    labels = assign_splits(rows, parse_split("80/10/10"), stratify=False, by_duration=False)
    # without stratify the single giant "group" is split by ratio
    agg = summarize(rows, labels)
    assert 78 <= agg["train"]["count_pct"] <= 82


# --------------------------------------------------------------------------- #
# relativize
# --------------------------------------------------------------------------- #
def test_relativize_up_and_over():
    assert relativize("/data/PARHAF/x.flac", Path("/data/NeMO_files/PARHAF")) == "../../PARHAF/x.flac"
    assert relativize("/data/PARHAF/x.flac", Path("/data/NeMO_files")) == "../PARHAF/x.flac"


# --------------------------------------------------------------------------- #
# audio_index: only real .flac clips, robust to stray/looping symlinks
# --------------------------------------------------------------------------- #
def test_audio_index_skips_non_flac_and_loops():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "000014_0000_abaissement.flac").write_bytes(b"x")
        (d / "000015_0001_foo.flac").write_bytes(b"x")
        (d / "000016_0002_bar.flac.tmp").write_bytes(b"x")   # aborted-synth leftover
        (d / ".claude").mkdir()                              # dotfolder
        os.symlink("voxtral_audios", d / "voxtral_audios")   # self-referential loop
        os.symlink("000099_0000_dead.flac", d / "000099_0000_dead.flac")  # .flac-named loop
        idx = audio_index(d)
    assert set(idx) == {"000014_0000", "000015_0001"}
    assert idx["000014_0000"].name == "000014_0000_abaissement.flac"


# --------------------------------------------------------------------------- #
# probe_durations: the cache must never answer for bytes it did not probe
# --------------------------------------------------------------------------- #
class _FakeInfo:
    """What ``soundfile.info`` gives ``probe_durations``: frames + samplerate."""

    def __init__(self, frames: int, samplerate: int) -> None:
        self.frames = frames
        self.samplerate = samplerate


class _FakeSF:
    """Stands in for the ``soundfile`` wheel so this stays a stdlib test.

    The reported duration is the clip file's own byte length (one second per
    byte), which is what lets a test rewrite a clip and check that the new
    "audio" is actually re-read. ``calls`` counts probes, so a cache HIT is
    observable rather than merely plausible.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def info(self, path: str) -> _FakeInfo:
        self.calls.append(Path(path).name)
        return _FakeInfo(frames=len(Path(path).read_bytes()), samplerate=1)


def _with_fake_sf():
    """Install the fake wheel and return it (``probe_durations`` imports lazily)."""
    fake = _FakeSF()
    sys.modules["soundfile"] = fake
    return fake


def test_probe_durations_cache_hit_and_invalidation():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        clip = d / "000014_0000_abaissement.flac"
        clip.write_bytes(b"xx")                     # 2 bytes -> 2.0 s
        cache = d / ".duration_cache.json"

        fake = _with_fake_sf()
        assert probe_durations([clip], cache_path=cache, progress=False) == {clip.name: 2.0}
        assert fake.calls == [clip.name]

        # Untouched clip: served from cache, never probed again.
        fake = _with_fake_sf()
        assert probe_durations([clip], cache_path=cache, progress=False) == {clip.name: 2.0}
        assert fake.calls == []

        # Same name, different audio (stage 06 overwrites in place, and a new
        # generation of the dataset reuses every file name): must be re-probed,
        # which is exactly what a name-only cache got wrong.
        clip.write_bytes(b"xxxxx")                  # 5 bytes -> 5.0 s
        fake = _with_fake_sf()
        assert probe_durations([clip], cache_path=cache, progress=False) == {clip.name: 5.0}
        assert fake.calls == [clip.name]

        # Byte-identical size but rewritten (mtime moved): still re-probed,
        # because size alone cannot tell two takes of equal length apart.
        st = clip.stat()
        clip.write_bytes(b"yyyyy")
        os.utime(clip, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        fake = _with_fake_sf()
        assert probe_durations([clip], cache_path=cache, progress=False) == {clip.name: 5.0}
        assert fake.calls == [clip.name]


def test_probe_durations_drops_legacy_unstamped_entries():
    """A cache written before the stamp existed (bare floats) carries v1
    durations under v2's file names, so it must be reprobed, not trusted."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        clip = d / "000014_0000_abaissement.flac"
        clip.write_bytes(b"xxx")                    # 3 bytes -> 3.0 s
        cache = d / ".duration_cache.json"
        cache.write_text(json.dumps({clip.name: 99.0}))

        fake = _with_fake_sf()
        assert probe_durations([clip], cache_path=cache, progress=False) == {clip.name: 3.0}
        assert fake.calls == [clip.name]
        assert json.loads(cache.read_text())[clip.name][0] == 3.0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
