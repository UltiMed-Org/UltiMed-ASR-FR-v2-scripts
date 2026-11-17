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
  * the census credits a key per clip and lists keys that matched nothing.

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


def write_tree(root: Path) -> None:
    """Lay out the three files the chain reads, each carrying the four texts."""
    for rel, rows in (
        ("01_dictionnary/generated_dataset.jsonl",
         [{"asr_training_target": t, "asr_training_source": t} for t in TEXTS.values()]),
        ("99_hf_release/data/NeMO_files/full.jsonl", [row(s, t) for s, t in TEXTS.items()]),
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
    check("the default table is RESYNTH_TYPOS and non-empty", len(t21.load_table(None)) > 0)

    # Multi-word fix (the scrambled "capsule de réfléchie Glisson" term, 2026-11-17).
    phrases = {k.lower(): v for k, v in t21.default_phrases().items()}
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

    print("FAILED" if failures else "all ok")
    return 1 if failures else 0


def _tmp_json(obj: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "table.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
