"""Stdlib test for 99_hf_release/05_normalize_text.py.

Checks that a v1 manifest row gets both of its texts normalized: the label through
``ParakeetTokenizer.clean_label`` + ``%``, the TTS source through ``pourcent``. The percent
rewrite itself is tested in ``tests/test_percent_normalize.py``, the label conventions
(titles, dates, clock, drug casing, applied to both texts) in ``tests/test_label_conventions.py``.

Run: python tests/test_normalize_text.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "05_normalize_text.py"

def test_normalize_row() -> None:
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {
        "audio_filepath": "a.flac",
        "text": "Laser CO₂ et séquence CUBE™,\nantigène Brª  espaçons 10 pour cent.",
        "asr_training_source": "Laser CO2,\nà 10 pour cent, puis 5%.",
    }
    out = mod.normalize_row(row, tok)
    assert out["text"] == "Laser CO2 et séquence CUBE, antigène Bra espaçons 10 %.", out["text"]
    assert out["asr_training_source"] == "Laser CO2, à 10 pourcent, puis 5 pourcent.", out["asr_training_source"]
    assert not tok.offending_chars(out["text"])
    assert mod.normalize_row(out, tok) == out  # idempotent
    assert row["text"].startswith("Laser CO₂")  # input not mutated
    print("test_normalize_row: OK")


def test_label_conventions_both_texts() -> None:
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {"audio_filepath": "a.flac",
           "text": "M. Petit prend du KARDEGIC depuis le quinze mars à 14h30.",
           "asr_training_source": "M. Petit prend du KARDEGIC depuis le quinze mars à 14h30."}
    out = mod.normalize_row(row, tok)
    want = "Monsieur Petit prend du Kardégic depuis le 15 mars à 14 heures 30."
    assert out["text"] == want, out["text"]
    assert out["asr_training_source"] == want, out["asr_training_source"]
    assert mod.normalize_row(out, tok) == out
    print("test_label_conventions_both_texts: OK")


def test_cp1252_mojibake() -> None:
    # The oli_drug_sentence manifest stored "œil" as U+009C + "il" (the cp1252 byte of
    # "œ" read as Latin-1), so the reference could never match a correct transcript.
    mod = load_sync_module(script=SCRIPT)
    tok = mod.ParakeetTokenizer(nfkc=False)
    row = {"audio_filepath": "a.flac", "text": "Une goutte dans chaque \u009cil, c\u009cur \u0081."}
    out = mod.normalize_row(row, tok)
    # U+0081 is undefined in cp1252: left for the <unk> gate, never guessed. The repaired
    # "œ" then follows the label convention like any other, hence "oe".
    assert out["text"] == "Une goutte dans chaque oeil, coeur \u0081.", out["text"]
    assert mod.normalize_row(out, tok) == out
    print("test_cp1252_mojibake: OK")


def test_main_dedup_parallel() -> None:
    """main normalizes each distinct (text, source) once, on several processes, and
    writes every manifest exactly as row-by-row normalize_row would."""
    import json
    import tempfile
    mod = load_sync_module(script=SCRIPT)
    sys.modules[mod.__name__] = mod  # the worker processes look the function up by module
    tok = mod.ParakeetTokenizer(nfkc=False)
    rows = [{"audio_filepath": f"{i}.flac", "text": t, "asr_training_source": t}
            for i, t in enumerate(["J5 post opération, post-chimiothérapie.", "M. Petit, 10 pour cent.",
                                   "Rien à changer."])]
    rows.append({"audio_filepath": "n.flac", "text": "Suites post opératoires."})  # no source
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "full.jsonl", Path(d) / "train.jsonl"
        for path, part in ((a, rows), (b, rows[1:3])):
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        mod.main((a, b), d, True, 2, 3)
        for path, part in ((a, rows), (b, rows[1:3])):
            got = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert got == [mod.normalize_row(r, tok) for r in part], got
    assert got[0]["text"] == "Monsieur Petit, 10 %.", got
    print("test_main_dedup_parallel: OK")


def test_stage_targets() -> None:
    """--stages rewrites asr_training_target of a stage file and nothing else: the source,
    the other fields and every row stay, an unchanged line byte for byte."""
    import json
    import tempfile
    mod = load_sync_module(script=SCRIPT)
    rows = [
        {"term": "spiramycine", "asr_training_target": "un comprimé de un million 500000 unités, à 94%.",
         "asr_training_source": "un comprimé de un million cinq cent mille unités, à 94 pourcent.",
         "other": "un million 500000"},
        {"term": "x", "asr_training_target": "Rien à changer.", "asr_training_source": "Rien à changer."},
        {"term": "no label"},
    ]
    lines = [json.dumps(rows[0], ensure_ascii=False) + "\n",
             # unusual spacing and escaping: kept verbatim since the row does not change
             '{"term":"x",  "asr_training_target": "Rien \\u00e0 changer.", "asr_training_source": "Rien à changer."}\n',
             json.dumps(rows[2]) + "\n"]
    with tempfile.TemporaryDirectory() as d:
        stage = Path(d) / "generated_dataset.jsonl"
        stage.write_text("".join(lines), encoding="utf-8")
        assert mod.normalize_stage_targets([stage], False, 1) == {stage: 1}
        assert stage.read_text(encoding="utf-8") == "".join(lines), "dry run wrote"
        assert mod.normalize_stage_targets([stage], True, 1) == {stage: 1}
        got = stage.read_text(encoding="utf-8").splitlines(keepends=True)
        assert len(got) == 3 and got[1:] == lines[1:], got
        out = json.loads(got[0])
        assert out == dict(rows[0], asr_training_target="un comprimé de 1500000 unités, à 94 %."), out
        assert not list(Path(d).glob("*.tmp")), "temp file left behind"
    print("test_stage_targets: OK")


if __name__ == "__main__":
    test_normalize_row()
    test_label_conventions_both_texts()
    test_cp1252_mojibake()
    test_main_dedup_parallel()
    test_stage_targets()
