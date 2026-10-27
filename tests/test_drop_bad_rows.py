"""Stdlib test for 99_hf_release/04_drop_bad_rows.py.

Checks that `qc_status == "exhausted"` rows are dropped, that rows never synced by
03_sync_hotfix_results.py (no `qc_status` key) are kept but counted, and that an eval
clip whose text repeats a training text (up to punctuation and case) is dropped from
every manifest that references its audio, whatever directory that manifest lives in,
that a label holding an ellipsis (truncated LLM output, anonymized date) is dropped,
and that leaked LLM reasoning and a label too long for its clip are dropped.
Also that the ASR-flagged list is recognized as stale once the audio it was measured
on has been replaced (the list keys on file names, which a regenerated corpus reuses),
and that a draw synthesized with CFG off (cfg_alpha 1.0) is dropped while the ladder's
1.2 floor is kept. And that a drop survives the QC round after it: stage 06's state
forgets a row the manifest no longer has, so the rebuilt row comes back with no
qc_status, and only dropped_clips.jsonl still knows why it went (until it is redrawn).

Run: python tests/test_drop_bad_rows.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "04_drop_bad_rows.py"


def test_filter_exhausted() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original"},
        {"audio_filepath": "b.flac", "qc_status": "exhausted"},
        {"audio_filepath": "c.flac", "qc_status": "improved"},
        {"audio_filepath": "d.flac"},
        {"audio_filepath": "e.flac", "qc_status": "exhausted"},
    ]
    kept, dropped, n_unsynced = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["a.flac", "c.flac", "d.flac"], kept
    assert [(r["audio_filepath"], why) for r, why in dropped] == [
        ("b.flac", mod.EXHAUSTED), ("e.flac", mod.EXHAUSTED)], dropped
    assert n_unsynced == 1, n_unsynced
    # Idempotent: filtering the kept rows again drops nothing.
    assert mod.filter_rows(kept, Path("/m"))[1] == []
    print("test_filter_exhausted: OK")


def test_eval_duplicates() -> None:
    mod = load_sync_module(script=SCRIPT)
    root = Path("/data/NeMO_files")
    train = [{"audio_filepath": "../PARHAF/t1.flac",
              "text": "Sur le plan cardiovasculaire, il n'y a pas de signe."}]
    evals = {
        root / "test.jsonl": [
            # Same sentence, punctuation and case differ: must be caught.
            {"audio_filepath": "../PARHAF/x1.flac",
             "text": "sur le plan cardiovasculaire il n'y a pas de signe"},
            {"audio_filepath": "../PARHAF/x2.flac", "text": "Une autre phrase."},
        ],
    }
    dup = mod.eval_duplicates(train, evals)
    assert list(dup) == ["/data/PARHAF/x1.flac"], dup

    # The same clip seen from a per-source manifest one level deeper is dropped too,
    # and the training twin (different audio) is kept.
    per_source = [
        {"audio_filepath": "../../PARHAF/x1.flac", "qc_status": "original"},
        {"audio_filepath": "../../PARHAF/t1.flac", "qc_status": "original"},
    ]
    kept, dropped, _ = mod.filter_rows(per_source, root / "PARHAF", dup)
    assert [r["audio_filepath"] for r in kept] == ["../../PARHAF/t1.flac"], kept
    assert [why for _, why in dropped] == [mod.DUPLICATE], dropped
    print("test_eval_duplicates: OK")


def test_ellipsis() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original",
         "text": "Les taux de rénine active à 12 milliu..."},
        {"audio_filepath": "b.flac", "qc_status": "original",
         "text": "La date de l'intervention est le …"},
        {"audio_filepath": "c.flac", "qc_status": "original", "text": "Tension à 12,8."},
    ]
    kept, dropped, _ = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["c.flac"], kept
    assert [why for _, why in dropped] == [mod.ELLIPSIS, mod.ELLIPSIS], dropped
    print("test_ellipsis: OK")


def test_llm_leak_and_too_fast() -> None:
    mod = load_sync_module(script=SCRIPT)
    rows = [
        # The two real drug_sentence train rows found on 2026-09-28.
        {"audio_filepath": "leak.flac", "qc_status": "original", "duration": 9.6,
         "text": "tags Let me create varied contexts: - Sentence 1: 300 mg comprimé - initial prescription"},
        {"audio_filepath": "fast.flac", "qc_status": "original", "duration": 0.3,
         "text": "Renouvellement du traitement par nitrofurantoïne une gélule de 50 milligrammes "
                 "quatre prises quotidiennes pendant 7 jours."},
        # Legitimate English terms and French "sentence" stay.
        {"audio_filepath": "ok1.flac", "qc_status": "original", "duration": 15.5,
         "text": "Hypersignal FLAIR au niveau du bed nucleus of the accessory olfactory tract."},
        {"audio_filepath": "ok2.flac", "qc_status": "original", "duration": 8.0,
         "text": "Le tribunal a rendu une sentence arbitrale le 15 mars 2024."},
        {"audio_filepath": "nodur.flac", "qc_status": "original", "text": "Pas de durée connue."},
    ]
    kept, dropped, _ = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["ok1.flac", "ok2.flac", "nodur.flac"], kept
    assert [why for _, why in dropped] == [mod.LLM_LEAK, mod.TOO_FAST], dropped
    print("test_llm_leak_and_too_fast: OK")


def test_asr_flagged() -> None:
    # Matched on <source>/<file> whatever directory the manifest resolves from.
    mod = load_sync_module(script=SCRIPT)
    rows = [{"audio_filepath": "../dictionary/a.flac", "text": "ok", "qc_status": "original"},
            {"audio_filepath": "../dictionary/b.flac", "text": "ok", "qc_status": "original"},
            {"audio_filepath": "../drugs/a.flac", "text": "ok", "qc_status": "original"}]
    kept, dropped, _ = mod.filter_rows(rows, Path("/x/NeMO_files"), asr_flagged={"dictionary/a.flac"})
    assert [r["audio_filepath"] for r in kept] == ["../dictionary/b.flac", "../drugs/a.flac"], kept
    assert [reason for _, reason in dropped] == [mod.ASR_FLAGGED]
    assert mod.load_asr_flagged(Path("/nonexistent.jsonl")) == set()
    print("test_asr_flagged: OK")


def test_asr_flagged_stale() -> None:
    """The list names clip keys, so it keeps matching after the audio was replaced.
    Each clip written after the list is one verdict measured on bytes that are gone."""
    mod = load_sync_module(script=SCRIPT)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "data" / "dictionary").mkdir(parents=True)
        old_clip = root / "data" / "dictionary" / "a.flac"
        new_clip = root / "data" / "dictionary" / "b.flac"
        for clip in (old_clip, new_clip):
            clip.write_bytes(b"x")
        listed = root / "asr_flagged_clips.jsonl"
        listed.write_text(json.dumps({"clip": "dictionary/a.flac"}) + "\n"
                          + json.dumps({"clip": "dictionary/b.flac"}) + "\n"
                          + json.dumps({"clip": "drugs/gone.flac"}) + "\n")
        flagged = mod.load_asr_flagged(listed)
        assert flagged == {"dictionary/a.flac", "dictionary/b.flac", "drugs/gone.flac"}

        # a.flac predates the list (it is what was measured), b.flac was regenerated
        # after it, drugs/gone.flac is not under data/ at all.
        stamp = listed.stat().st_mtime_ns
        os.utime(old_clip, ns=(stamp - 10**9, stamp - 10**9))
        os.utime(new_clip, ns=(stamp + 10**9, stamp + 10**9))
        assert mod.asr_flagged_stale(flagged, root, listed) == (1, 1), \
            mod.asr_flagged_stale(flagged, root, listed)

        # No list, or no flagged clips: nothing to be stale about.
        assert mod.asr_flagged_stale(flagged, root, root / "absent.jsonl") == (0, 0)
        assert mod.asr_flagged_stale(set(), root, listed) == (0, 0)
    print("test_asr_flagged_stale: OK")


def test_cfg_off() -> None:
    """A draw made with CFG disabled (cfg_alpha exactly 1.0) is dropped; the rest of the
    measured band is kept. The field holds the WINNING draw's value, so only a redrawn
    row can carry 1.0, and 1.2 (the ladder's floor) must survive: it is the bottom rung
    the improver is allowed to use, not a defect."""
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"audio_filepath": "a.flac", "qc_status": "original", "cfg_alpha": 2.0},
        {"audio_filepath": "b.flac", "qc_status": "improved", "cfg_alpha": 1.0},
        {"audio_filepath": "c.flac", "qc_status": "improved", "cfg_alpha": 1.2},
        {"audio_filepath": "d.flac", "qc_status": "original", "cfg_alpha": 1.3},
        {"audio_filepath": "e.flac", "qc_status": "original"},          # no field at all
        {"audio_filepath": "f.flac", "qc_status": "improved", "cfg_alpha": 1},  # int 1
    ]
    kept, dropped, _ = mod.filter_rows(rows, Path("/m"))
    assert [r["audio_filepath"] for r in kept] == ["a.flac", "c.flac", "d.flac", "e.flac"], kept
    assert [(r["audio_filepath"], why) for r, why in dropped] == [
        ("b.flac", mod.CFG_OFF), ("f.flac", mod.CFG_OFF)], dropped
    # Idempotent, like every other rule.
    assert mod.filter_rows(kept, Path("/m"))[1] == []
    print("test_cfg_off: OK")


def test_a_drop_survives_the_next_qc_round() -> None:
    """2026-10-27: 71 dropped clips came back with empty QC columns. The QC round after a
    drop runs on the trimmed manifest, so its state forgets them; the next manifest
    rebuild finds their audio on disk and re-adds them with no qc_status, and nothing
    was left to drop them on. The ledger is that memory, valid while the audio is the
    same bytes."""
    mod = load_sync_module(script=SCRIPT)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        nemo, audio_dir = root / "data" / "NeMO_files", root / "data" / "dictionary"
        nemo.mkdir(parents=True)
        audio_dir.mkdir(parents=True)
        for name in ("a", "b"):
            (audio_dir / f"{name}.flac").write_bytes(b"x")
        train, ledger = nemo / "train.jsonl", root / "dropped_clips.jsonl"

        def write_train(status_a: str | None) -> None:
            a = {"audio_filepath": "../dictionary/a.flac", "text": "une phrase", "duration": 3.0}
            if status_a:
                a["qc_status"] = status_a
            b = {"audio_filepath": "../dictionary/b.flac", "text": "une autre", "duration": 3.0,
                 "qc_status": "original"}
            train.write_text("".join(json.dumps(r) + "\n" for r in (a, b)))

        def run(apply: bool) -> list[str]:
            mod.main(manifests=(), root=str(root), apply=apply, asr_flagged_list=root / "none.jsonl",
                     drop_ledger=ledger, no_asr_flagged=False)
            return [json.loads(line)["audio_filepath"] for line in train.read_text().splitlines()]

        write_train("exhausted")
        assert run(apply=False) == ["../dictionary/a.flac", "../dictionary/b.flac"]
        assert not ledger.exists(), "a dry run wrote the ledger"
        assert run(apply=True) == ["../dictionary/b.flac"]
        assert mod.load_drop_ledger(ledger)["dictionary/a.flac"]["reason"] == mod.EXHAUSTED

        # The QC round forgot it and the rebuild brought it back, unjudged.
        write_train(None)
        assert run(apply=True) == ["../dictionary/b.flac"], "a forgotten drop shipped again"
        assert len(mod.load_drop_ledger(ledger)) == 1

        # Redrawn since: new bytes, so the old verdict no longer applies.
        stamp = (audio_dir / "a.flac").stat().st_mtime_ns + 10**9
        os.utime(audio_dir / "a.flac", ns=(stamp, stamp))
        write_train("improved")
        assert run(apply=True) == ["../dictionary/a.flac", "../dictionary/b.flac"]
    print("test_a_drop_survives_the_next_qc_round: OK")


if __name__ == "__main__":
    test_filter_exhausted()
    test_eval_duplicates()
    test_ellipsis()
    test_llm_leak_and_too_fast()
    test_asr_flagged()
    test_asr_flagged_stale()
    test_cfg_off()
    test_a_drop_survives_the_next_qc_round()
