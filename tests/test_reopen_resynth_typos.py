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
"""`06_hotfixes/21_reopen_resynth_typos.py`: the held-back spelling re-synthesis queue.

Runs the whole chain (``reopen_chain``) on a tiny temp tree laid out like the repo, with
a two-key table, so nothing real is read or written:

  * a dry run writes nothing but still writes the stem list and fills the census;
  * ``--apply`` rewrites the stage file, the release manifest and the state file, keeps
    the table casing, elides the article (``le cestradiol`` -> ``l'estradiol``) and marks the
    changed rows ``pending_tts``;
  * the generic-lab strip (``Simvastatine Téva``) queues its clip under its own census
    key, and is skipped with ``--no-generic-labs``;
  * a row with neither is left byte for byte, and a second run changes nothing;
  * every release manifest is rewritten, the splits included (``test.jsonl``, a
    per-source ``PARHAF/train.jsonl``), not only ``full.jsonl``: on 2026-11-17 the
    splits kept the old text of 329 re-synthesised clips;
  * ``rewrite_manifests`` alone (21's ``--manifests-only``) rewrites those manifests and
    leaves the stage and state files byte for byte, nothing queued;
  * the census credits a key per clip and lists keys that matched nothing;
  * with nothing held (both default tables empty, as since 2026-11-19) ``main`` exits 0
    at once instead of scanning every file.

    uv run tests/test_reopen_resynth_typos.py

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "utils"))
from script_module import load_script  # noqa: E402

failures = []

#: A small stand-in for RESYNTH_TYPOS: one plain fix, one that makes the word
#: vowel-initial (so the article must elide), one that no row carries.
TABLE = {"perindipril": "périndopril", "cestradiol": "estradiol", "nothere": "absent"}
#: A stand-in for RESYNTH_PHRASES, from its 2026-11-17 batch (the real table is empty once
#: its clips are redrawn): the first scrambled term, and the four context keys a bare
#: ``de Hutchinson melanotic freckle`` -> ``Hutchinson ...`` had to be split into, since
#: that draft was not stable on a second pass.
PHRASES = {"capsule de réfléchie Glisson": "capsule de Glisson réfléchie",
           "un de Hutchinson melanotic freckle": "un Hutchinson melanotic freckle",
           "du de Hutchinson melanotic freckle": "du Hutchinson melanotic freckle",
           "type de Hutchinson melanotic freckle": "type Hutchinson melanotic freckle",
           "de de Hutchinson melanotic freckle": "de Hutchinson melanotic freckle"}


def check(name: str, ok: bool, detail=None) -> None:
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    if not ok:
        failures.append(name)


def row(stem: str, text: str) -> dict:
    return {"audio_filepath": f"../PARHAF/{stem}.flac", "voice": "fr_male", "text": text,
            "asr_training_source": text, "transcriptions": {},
            "improvement": {"status": "improved", "cer": 0.01}}


TEXTS = {
    "a": "PERINDIPRIL 5 mg le matin.",
    "b": "Elle prend le cestradiol depuis mars.",
    "c": "Simvastatine Téva 20 mg.",
    "d": "Rien à corriger ici.",
}


#: Split manifests, which must follow full.jsonl (MANIFEST_GLOBS: top level and per source).
SPLITS = ("99_hf_release/data/NeMO_files/test.jsonl", "99_hf_release/data/NeMO_files/PARHAF/train.jsonl")
STAGE_AND_STATE = ("01_dictionnary/generated_dataset.jsonl", "06_hotfixes/improved/full.stt.jsonl")


def manifest_texts(root: Path) -> dict[str, tuple[str, str]]:
    """stem -> (text, asr_training_source) of every split manifest row."""
    return {Path(r["audio_filepath"]).stem: (r["text"], r["asr_training_source"])
            for rel in SPLITS for r in map(json.loads, (root / rel).read_text().splitlines())}


def write_tree(root: Path) -> None:
    """Lay out the files the chain reads, each carrying the four texts, plus two split
    manifests (top level and per source) holding a subset, as the release does."""
    for rel, rows in (
        ("01_dictionnary/generated_dataset.jsonl",
         [{"asr_training_target": t, "asr_training_source": t} for t in TEXTS.values()]),
        ("99_hf_release/data/NeMO_files/full.jsonl", [row(s, t) for s, t in TEXTS.items()]),
        (SPLITS[0], [row(s, TEXTS[s]) for s in "ad"]),
        (SPLITS[1], [row(s, TEXTS[s]) for s in "bc"]),
        ("06_hotfixes/improved/full.stt.jsonl", [row(s, t) for s, t in TEXTS.items()]),
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*.jsonl"))}


def main() -> int:
    t21 = load_script(ROOT / "06_hotfixes" / "21_reopen_resynth_typos.py", "t21_test")
    table = {k.lower(): v for k, v in TABLE.items()}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_tree(root)
        before = snapshot(root)
        stems_file = root / "stems.txt"

        observe, hits = t21.make_census(table, True)
        stems = t21.chain.reopen_chain(t21.make_rewrite(table, True), False, stems_file, "t21_dry",
                                       root=root, observe=observe)
        check("a dry run writes no jsonl", snapshot(root) == before)
        check("a dry run still writes the stem list",
              stems_file.read_text().split() == ["a", "b", "c"], stems_file.read_text())
        check("the census credits each key and the lab strip",
              {k: sorted(v) for k, v in hits.items()}
              == {"perindipril": ["a"], "cestradiol": ["b"], t21.GENERIC_LAB: ["c"]}, dict(hits))

        observe, hits = t21.make_census(table, True)
        stems = t21.chain.reopen_chain(t21.make_rewrite(table, True), True, None, "t21_apply",
                                       root=root, observe=observe)
        check("apply queues the same clips", stems == {"a", "b", "c"}, stems)
        state = {Path(r["audio_filepath"]).stem: r for r in map(
            json.loads, (root / "06_hotfixes/improved/full.stt.jsonl").read_text().splitlines())}
        # An all-caps typo takes the table's casing (capitalised at a sentence start),
        # the drug caser's rule in normalize_accents, not an all-caps fix.
        check("an all-caps typo takes the table casing", state["a"]["text"] == state["a"]["asr_training_source"]
              == "Périndopril 5 mg le matin.", state["a"])
        check("the article elides before the vowel-initial fix",
              state["b"]["asr_training_source"] == "Elle prend l'estradiol depuis mars.", state["b"])
        check("the generic lab name is stripped",
              "Téva" not in state["c"]["asr_training_source"], state["c"])
        check("changed rows are pending_tts with no reference score",
              all(state[s]["improvement"]["status"] == "pending_tts"
                  and state[s]["improvement"].get("cer") is None for s in "abc"),
              {s: state[s]["improvement"] for s in "abc"})
        check("the untouched row keeps its status",
              state["d"]["improvement"] == {"status": "improved", "cer": 0.01}, state["d"])
        manifest = [json.loads(line) for line in
                    (root / "99_hf_release/data/NeMO_files/full.jsonl").read_text().splitlines()]
        stage = [json.loads(line) for line in
                 (root / "01_dictionnary/generated_dataset.jsonl").read_text().splitlines()]
        check("manifest and stage file carry the new text",
              manifest[1]["text"] == stage[1]["asr_training_target"]
              == "Elle prend l'estradiol depuis mars.", (manifest[1], stage[1]))

        full = {Path(r["audio_filepath"]).stem: (r["text"], r["asr_training_source"])
                for r in manifest}
        splits = manifest_texts(root)
        check("every split manifest carries full.jsonl's new text",
              all(splits[s] == full[s] for s in splits) and splits["b"][0] == "Elle prend l'estradiol depuis mars.",
              (splits, full))

        after = snapshot(root)
        again = t21.chain.reopen_chain(t21.make_rewrite(table, True), True, None, "t21_again", root=root)
        check("a second run changes nothing", not again and snapshot(root) == after, again)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_tree(root)
        stems = t21.chain.reopen_chain(t21.make_rewrite(table, False), False, None, "t21_nolab", root=root)
        check("--no-generic-labs leaves the Téva clip alone", stems == {"a", "b"}, stems)

    check("load_table lowercases keys and unwraps {'rewrites': ...}",
          t21.load_table(_tmp_json({"rewrites": {"KAREGIC": "Kardegic"}})) == {"karegic": "Kardegic"})
    accent = load_script(ROOT / "06_hotfixes" / "07_build_accent_spellings.py", "accent_test")
    check("the default tables are RESYNTH_TYPOS and RESYNTH_PHRASES",
          t21.default_table() == accent.RESYNTH_TYPOS and t21.default_phrases() == accent.RESYNTH_PHRASES)
    check_nothing_held_exits_early(t21)

    # The repair for a round that rewrote only full.jsonl: manifests only, nothing else.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_tree(root)
        before = snapshot(root)
        counts = t21.chain.rewrite_manifests(t21.make_rewrite(table, True), False, root=root)
        check("a manifests-only dry run writes nothing", snapshot(root) == before)
        counts = t21.chain.rewrite_manifests(t21.make_rewrite(table, True), True, root=root)
        after = snapshot(root)
        check("manifests-only counts the changed rows per manifest",
              {str(p.relative_to(root)): n for p, n in counts.items() if n}
              == {"99_hf_release/data/NeMO_files/full.jsonl": 3, SPLITS[0]: 1, SPLITS[1]: 2}, counts)
        check("manifests-only leaves the stage and state files byte for byte",
              all(after[rel] == before[rel] for rel in STAGE_AND_STATE))
        check("manifests-only brings the splits to the new text",
              manifest_texts(root)["a"][0] == "Périndopril 5 mg le matin.", manifest_texts(root))

    # Multi-word fix (the scrambled "capsule de réfléchie Glisson" term, 2026-11-17).
    phrases = {k.lower(): v for k, v in PHRASES.items()}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        TEXTS["e"] = "La capsule de réfléchie Glisson est lisse."
        try:
            write_tree(root)
        finally:
            del TEXTS["e"]
        observe, hits = t21.make_census(table, True, phrases)
        stems = t21.chain.reopen_chain(t21.make_rewrite(table, True, phrases), True, None, "t21_phrase",
                                       root=root, observe=observe)
        state = {Path(r["audio_filepath"]).stem: r for r in map(
            json.loads, (root / "06_hotfixes/improved/full.stt.jsonl").read_text().splitlines())}
        check("a held phrase queues its clip", stems == {"a", "b", "c", "e"}, stems)
        check("the phrase is rewritten in label and source, pending_tts",
              state["e"]["text"] == state["e"]["asr_training_source"] == "La capsule de Glisson réfléchie est lisse."
              and state["e"]["improvement"]["status"] == "pending_tts", state["e"])
        check("the census credits the phrase",
              sorted(hits.get("capsule de réfléchie glisson", ())) == ["e"], dict(hits))
        check("without phrases the clip is left alone",
              t21.make_rewrite(table, True)(TEXTS["d"] + " capsule de réfléchie Glisson")
              == TEXTS["d"] + " capsule de réfléchie Glisson")

    # Every held phrase must be safe to apply twice (a resumed or repeated run), which
    # fails when a fix still contains a key, e.g. "de X" -> "X" after "antécédents de".
    # Checked on the stand-in and on the real table, which guards a future batch.
    for name, table_ in (("stand-in", phrases),
                         ("RESYNTH_PHRASES", {k.lower(): v for k, v in t21.default_phrases().items()})):
        pat = t21._phrase_re(table_)
        twice = {k: t21.fix_phrases(t21.fix_phrases(k, table_, pat), table_, pat) for k in table_}
        check(f"every {name} phrase is fixed in one pass and stable after",
              all(twice[k].lower() == table_[k].lower() for k in table_),
              {k: v for k, v in twice.items() if v.lower() != table_[k].lower()})
    check("a fix inside its usual sentence is stable too",
          t21.fix_phrases("aux antécédents de Hutchinson melanotic freckle du visage", phrases)
          == "aux antécédents de Hutchinson melanotic freckle du visage")

    print("FAILED" if failures else "all ok")
    return 1 if failures else 0


def check_nothing_held_exits_early(t21) -> None:
    """With both default tables empty (the state after the 2026-11-19 merge) ``main`` logs
    that nothing is held and returns at once, census or ``--manifests-only``, instead of
    scanning every stage file and manifest for 35 minutes; one held phrase is enough to scan."""
    from click.testing import CliRunner

    calls = []
    saved = (t21.default_table, t21.default_phrases, t21.chain.reopen_chain, t21.chain.rewrite_manifests)
    t21.chain.reopen_chain = lambda *a, **k: calls.append("reopen_chain") or set()
    t21.chain.rewrite_manifests = lambda *a, **k: calls.append("rewrite_manifests") or {}
    t21.default_table = dict
    t21.default_phrases = dict
    try:
        runs = [CliRunner().invoke(t21.main, args) for args in ([], ["--manifests-only"])]
        runs.append(CliRunner().invoke(t21.main, ["--table", str(_tmp_json({"rewrites": {}}))]))
        check("nothing held: exit 0 without scanning", all(r.exit_code == 0 for r in runs) and not calls,
              ([(r.exit_code, r.output, r.exception) for r in runs], calls))
        t21.default_phrases = lambda: dict(PHRASES)
        run = CliRunner().invoke(t21.main, [])
        check("one held phrase is enough to scan", run.exit_code == 0 and calls == ["reopen_chain"],
              (run.exit_code, run.exception, calls))
    finally:
        t21.default_table, t21.default_phrases, t21.chain.reopen_chain, t21.chain.rewrite_manifests = saved


def _tmp_json(obj: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "table.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
