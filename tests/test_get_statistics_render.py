#!/usr/bin/env python
"""`99_hf_release/scripts/get_statistics.py` must actually render its report.

The truncation audit was moved into `utils/nemo_manifest.py` and shared with
`06_hotfixes/02_statistics.py`, but only `duration_ceiling` was imported back: the
tolerance constant `CEILING_TOL_S` stayed a bare global reference in the report text.
Nothing failed at import time, so the script died with `NameError` only once it reached
the "Duration ceiling check" section, i.e. after the whole (multi-minute) manifest scan
and at the exact moment the release numbers were wanted.

Every other test here imports a module and calls a function, which would NOT have caught
this: the reference is inside a rendering branch. So this one runs the real script, end
to end, over a synthetic two-dataset manifest tree, and only checks that it succeeds and
reports what it read. `--no-sizes` keeps it off the disk probe, so no audio is needed.

The per-voice sections are rendered the same way, in a branch taken only when the rows
carry a `voice` field, so they get the same treatment in a second tree. Both halves of
that branch are covered, because they are both wrong in a way nothing else would show:
a v1 tree (no `voice` anywhere) must NOT grow a one-row voice table about a voice called
"(unrecorded)", and a v2 tree must report every voice it was given. The numbers are
checked, not just the headings: the per-voice files have to add up to the corpus total,
which is what fails if `select` and the percentages ever disagree about a dimension.

    python tests/test_get_statistics_render.py

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "99_hf_release" / "scripts" / "get_statistics.py"
SPLITS = ("train", "val", "test")


def write_manifest(path: Path, n: int, dur: float, *, pile_up: int = 0,
                   voices: tuple[str, ...] = ()) -> None:
    """n rows of `dur` seconds, plus `pile_up` extra rows pinned at one longer value.

    With `voices`, every row also carries a `voice` and a `group_id`, cycling through
    the roster the way v2 audio does (one voice per group), which is what turns on the
    per-voice sections of the report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def extra(i: int) -> dict:
        if not voices:
            return {}
        return {"voice": voices[i % len(voices)],
                "group_id": f"{path.parent.name}:{path.stem}:{i}"}

    rows = [
        {"audio_filepath": f"{path.stem}_{i}.flac", "duration": dur + i * 0.1,
         "text": "un texte de test", "category": path.parent.name, **extra(i)}
        for i in range(n)
    ]
    rows += [
        {"audio_filepath": f"{path.stem}_cap_{i}.flac", "duration": 163.84,
         "text": "un texte tronque", "category": path.parent.name, **extra(n + i)}
        for i in range(pile_up)
    ]
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def test_render_runs_over_a_synthetic_tree() -> None:
    if shutil.which("uv") is None:
        print("SKIP test_render_runs_over_a_synthetic_tree (uv not on PATH)")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "NeMO_files"
        # One healthy dataset and one whose clips pile up on the old 2048-frame cap, so
        # BOTH branches of the ceiling section are rendered (quiet verdict and warning).
        for split, n in zip(SPLITS, (40, 5, 5)):
            write_manifest(root / "dictionary" / f"{split}.jsonl", n, 2.0)
            write_manifest(root / "parhaf" / f"{split}.jsonl", n, 12.0, pile_up=n)
        out = Path(td) / "stats.md"
        proc = subprocess.run(
            ["uv", "run", str(SCRIPT), "--root", str(root),
             "--source", "per-dataset", "--no-sizes", "--output", str(out)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert proc.returncode == 0, (
            f"get_statistics.py exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}"
        )
        md = out.read_text(encoding="utf-8")
        # The section that held the undefined name, and the numbers around it.
        assert "Duration ceiling check" in md, md[-2000:]
        assert "0.08 s" in md, "the frame tolerance must be spelled out in the report"
        assert "163.84" in md, "the pile-up dataset must be reported at its cap"
        assert "dictionary" in md and "parhaf" in md, "both datasets must appear"
        # v1 audio records no voice, so the per-voice sections must be absent rather
        # than present and empty, or claiming a voice named "(unrecorded)".
        assert "## By voice" not in md, "no voice field, so no per-voice section"
        assert "(unrecorded)" not in md, "the placeholder bucket must not be reported"
    print("OK test_render_runs_over_a_synthetic_tree")


VOICES = ("fr_female", "fr_male", "h_itf_arm", "h_frf_esm")


def test_render_reports_every_voice() -> None:
    """A v2 tree (rows carry `voice`) gets the three per-voice sections, and they add up."""
    if shutil.which("uv") is None:
        print("SKIP test_render_reports_every_voice (uv not on PATH)")
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "NeMO_files"
        for split, n in zip(SPLITS, (40, 8, 8)):
            write_manifest(root / "dictionary" / f"{split}.jsonl", n, 2.0, voices=VOICES)
            write_manifest(root / "parrot" / f"{split}.jsonl", n, 12.0, voices=VOICES)
        out = Path(td) / "stats.md"
        proc = subprocess.run(
            ["uv", "run", str(SCRIPT), "--root", str(root),
             "--source", "per-dataset", "--no-sizes", "--output", str(out)],
            capture_output=True, text=True, cwd=str(REPO),
        )
        assert proc.returncode == 0, (
            f"get_statistics.py exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}"
        )
        md = out.read_text(encoding="utf-8")
        for section in ("## By voice", "## Voice balance per dataset", "## By voice x split"):
            assert section in md, f"{section} missing\n{md[-3000:]}"
        assert f"- Voices: {len(VOICES)} (" in md, "the header must count the voices"
        for v in VOICES:
            assert v in md, f"voice {v} never reported"
        # The numbers, not just the headings: the per-voice rows of "By voice" have to
        # account for every clip. A dimension mismatch between `select` and the
        # percentages would leave this short (or double it) while still rendering.
        body = md[md.index("## By voice"):md.index("## Voice balance per dataset")]
        per_voice = {}
        for line in body.splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) > 2 and cols[0] in VOICES:
                per_voice[cols[0]] = int(cols[1].replace(",", ""))
        assert len(per_voice) == len(VOICES), f"rows found: {sorted(per_voice)}"
        total = sum(n for _, n in zip(SPLITS, (40, 8, 8)) for n in [n]) * 2
        assert sum(per_voice.values()) == total, f"{per_voice} does not sum to {total}"
        # Four voices cycled over the rows: balanced to within one clip, so the spread
        # line must say so rather than reporting a collapsed or empty roster.
        assert max(per_voice.values()) - min(per_voice.values()) <= 2, per_voice
    print("OK test_render_reports_every_voice")


if __name__ == "__main__":
    test_render_runs_over_a_synthetic_tree()
    test_render_reports_every_voice()
    print("all OK")
