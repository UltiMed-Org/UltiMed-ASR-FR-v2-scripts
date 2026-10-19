"""Read "work on exactly these clips" lists, shared by every stage that accepts one.

A clip list is how one QC stage hands its findings to a more expensive one without
either of them knowing about the other: `04_qc_acoustic_loops.py` scores all 602k clips
from the waveform alone and flags ~1%, `05_generate_audio/03_qc_cpu_stt.py` confirms
those flags with a real transcription (~2 h on one core against ~230 h for the whole
corpus), and `06_hotfixes/01_recursive_improvement.py` re-judges a flagged slice with a
second, stronger ASR model before anything is regenerated. Each of those is "run the
pass, but only over these stems".

Two file shapes are accepted because both are what a caller actually has:

* a **JSONL sidecar**, one object per clip, which contributes the stems of its `looped`
  rows and keys them by each row's own `stage`, so a file cannot leak stems into a
  sibling stage. A row with no `looped` key at all still counts, so a hand-filtered
  sidecar works.
* **anything else**, read as a plain list of stems, one per line, `#` comments allowed,
  applying to every stage.

Stdlib only, no logging dependency: the caller is handed per-file counts through
`on_file` and logs them in its own style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

#: Stage name -> stems to work on. The key `""` holds stems that apply to any stage.
StemRequests = dict[str, set[str]]


def load_stem_requests(
    paths: Iterable[Path], *, on_file: Callable[[Path, int, int], None] | None = None
) -> StemRequests:
    """Parse every file in `paths` into one stage -> stems mapping.

    A missing file is a hard error rather than an empty set: silently checking nothing
    would look exactly like a clean corpus, which is the one answer this must never
    invent. `on_file(path, jsonl_rows, bare_stems)` is called once per file read.
    """
    requests: StemRequests = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"stem list: no such file {path}")
        rows = plain = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                 # torn final line, as everywhere else here
                stem = row.get("stem")
                if not stem or ("looped" in row and not row["looped"]):
                    continue
                requests.setdefault(row.get("stage") or "", set()).add(stem)
                rows += 1
            else:
                requests.setdefault("", set()).add(line)
                plain += 1
        if on_file is not None:
            on_file(path, rows, plain)
    return requests


def wanted_for(requests: StemRequests, stage: str) -> frozenset[str]:
    """Stems to work on in `stage`: the ones aimed at it plus the stage-agnostic ones.

    Resolved once per caller loop rather than per clip, since the callers walk 100k-entry
    directories and 600k-row manifests.
    """
    return frozenset(requests.get(stage, frozenset()) | requests.get("", frozenset()))


def stem_of(audio_filepath: str) -> str:
    """Clip stem from a manifest's `audio_filepath`, i.e. the basename without suffix."""
    return Path(audio_filepath).stem


def row_is_requested(requests: StemRequests, audio_filepath: str) -> bool:
    """Whether a manifest row's clip is in a requested set.

    The stem is matched on its own for a stage-agnostic entry, and a stage-keyed entry
    additionally requires that stage's directory to appear in the clip's path. Stages
    number their terms independently, so `000123_0001_<slug>` is a stem each of them can
    mint, and a flag list aimed at the dictionary must not reopen a drugs clip that
    happens to share it. The slug makes a real collision very unlikely; the check costs
    one substring test and removes the question.
    """
    if not requests:
        return True                          # no list given = no restriction
    stem = stem_of(audio_filepath)
    if stem in requests.get("", frozenset()):
        return True
    return any(stem in stems and stage and f"/{stage}/" in f"/{audio_filepath}"
               for stage, stems in requests.items())
