#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""99_hf_release/07_write_back_texts.py: manifest texts land on the generated row of the
same clip, other fields are kept, unmatched rows drop, conflicting manifests are refused.

Run: uv run tests/test_write_back_texts.py (written with Claude Code)."""
import importlib.util
import json
import tempfile
from pathlib import Path

import click

spec = importlib.util.spec_from_file_location(
    "wb", Path(__file__).resolve().parent.parent / "99_hf_release" / "07_write_back_texts.py")
wb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)


def _manifest(tmp: Path, name: str, rows: list[tuple[str, str, str]]) -> Path:
    path = tmp / name
    path.write_text("".join(json.dumps({"audio_filepath": f, "text": t, "asr_training_source": s}) + "\n"
                            for f, t, s in rows))
    return path


def test_write_back() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        m = _manifest(tmp, "m.jsonl", [("../dictionary/000003_0001_foo.flac", "IgM positif.", "I-gé-M positif.")])
        texts = wb.manifest_texts([m])
        rows = [{"term_index": "3", "variant_index": "1", "term": "foo", "asr_training_target": "IgM +",
                 "asr_training_source": "IgM +"},
                {"term_index": "3", "variant_index": "2", "term": "foo", "asr_training_target": "x...",
                 "asr_training_source": "x..."}]
        out, dropped, changed, used = wb.write_back(rows, "dictionary", texts)
        assert out == [{**rows[0], "asr_training_target": "IgM positif.", "asr_training_source": "I-gé-M positif."}]
        assert dropped == [rows[1]] and changed == 1 and used == {("dictionary", 3, 1)}
        # same clip in another folder is another clip
        assert wb.write_back(rows[:1], "drugs", texts)[1] == rows[:1]
        bad = _manifest(tmp, "b.jsonl", [("x/dictionary/000003_0001_foo.flac", "autre", "autre")])
        try:
            wb.manifest_texts([m, bad])
        except click.ClickException:
            pass
        else:
            raise AssertionError("conflicting texts for one clip must be refused")
    print("test_write_back: OK")


if __name__ == "__main__":
    test_write_back()
