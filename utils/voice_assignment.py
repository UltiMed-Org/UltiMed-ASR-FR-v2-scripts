"""Which of the ten French voices reads a given row, decided once and re-derivable.

v1 is single-voice (`fr_female` throughout), which makes a speaker-independent ASR
model learn one speaker's articulation. v2 spreads the same corpus over ten voices,
and two stages need to agree on which voice read which clip: `05_generate_audio`
sends it at synthesis time, and `99_hf_release` records it in the manifest and the
Parquet long after the audio was written. This module is the single place that
decides, so neither side can drift from the other (the repo rule: anything used by
more than one stage lives in `utils/`).

**The assignment is a hash of the row's group key, not a counter.** That is the whole
design, and the reason is edit stability. A round-robin over groups would be exactly
balanced, but adding or deleting one row shifts every later group to a different
voice, which silently invalidates the `voice` column of audio already on disk (the
clips are not regenerated: the resume check in `01_generate_audio.py` matches
filenames, and a filename does not depend on the voice). A hash gives each group key
its voice independently of every other row, so the corpus text can keep being edited,
rows can be added, and a half-finished run can resume, without any already-generated
clip becoming mislabelled. The price is imperfect balance, measured over the v1 inputs
at 2.8 pct spread on 01_dictionnary and up to 50 pct on 04_PARROT (475 document groups
over ten voices, so 116 to 194 clips per voice): acceptable, because the model sees
about 60k clips per voice overall and the skew lives in the two smallest stages.

`blake2b` rather than the builtin `hash()`: Python randomises string hashing per
process unless `PYTHONHASHSEED` is pinned, and the manifest builder runs in a
different process (usually a different day) than the generator, so `hash()` would
produce a `voice` column that does not match the audio. Tested by
`tests/test_voice_assignment.py`.

The group key, not the row, is what gets hashed, so every clip of a group shares a
voice. For PARHAF and PARROT a group is one source document cut into chunks, which
should sound like one person reading it; elsewhere it is one term's variants. The key
field mirrors `detect_grouping` in 99_hf_release/01_build_nemo_manifest.py
(`source_id` when the rows carry one, else `term_index`) so that both stages group
identically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The ten French voices cleared for v2, in roster order. Chosen by ear over two
# audition rounds, the second on medical text: FLEURS-clean prose did not predict
# behaviour on drug names and dosages and disagreed about 6 of 16 candidates. They are
# ten DISTINCT acoustic identities (five female, five male), not ten variants of three.
# Deliberate copy, like VOXTRAL_VOICES in 05_generate_audio/01_generate_audio.py: the
# roster is decided in the CrispASR harness repo (voxtral/cloning/voices_fr_production.txt,
# plan stages A9 and A10 there) and this repo must stay self-contained, so if that
# roster changes, update both sides. `--voices-file` overrides it without editing code.
VOICES_FR_V2 = (
    "fr_female",    # shipped base
    "fr_male",      # shipped base
    "h_itf_arm",    # ar_male
    "h_frf_esm",    # es_male
    "h_frm_nlf",    # nl_female
    "h_itf_def",    # de_female
    "h_frf_itm",    # it_male
    "h_frm_ptf",    # pt_female
    "h_frf_ptm",    # pt_male
    "h_frm_chf",    # cheerful_female
)

# How to turn a row into the thing that gets hashed. "group" keys on the group alone,
# "row" gives every clip its own draw. v2 uses BOTH, one per stage, because the group
# means two different things depending on the stage: for 01_dictionnary, 02_drugs and
# 07_acronyms a group is one TERM and its written variants, which are independent
# sentences, so "row" is used there to hear a given drug name in several accents and to
# keep a bad voice/term pairing from costing all ~8 clips of that term; for 03_PARHAF
# and 04_PARROT a group is one DOCUMENT and the clips are consecutive chunks of it, so
# "group" is used there and a report keeps one reader from start to end. Both modes are
# hashes, so edit and resume stability is identical either way (see module docstring);
# the mode a run used is recorded per audio folder in its voice_assignment.json.
MODES = ("group", "row")


def group_field(sample: dict) -> str:
    """Return the field whose value identifies a row's group.

    Parameters
    ----------
    sample
        Any row of the stage's `generated_dataset.jsonl`; only its keys are read.

    Returns
    -------
    str
        `"source_id"` if the rows carry one (PARHAF and PARROT: document chunks),
        else `"term_index"` (term variants).

    Mirrors the auto choice of `detect_grouping` in
    99_hf_release/01_build_nemo_manifest.py on purpose. If these two ever disagree, a
    clip's `group_id` and its `voice` would be grouped on different
    fields, so a document could be split across voices while still claiming to be one
    group.
    """
    return "source_id" if "source_id" in sample else "term_index"


def assignment_key(row: dict, *, field: str, mode: str = "group") -> str:
    """Return the string hashed to pick a voice for `row`.

    Parameters
    ----------
    row
        One row of a stage's `generated_dataset.jsonl`.
    field
        The group field, from `group_field`.
    mode
        `"group"` keys on the group alone, so all of a group's clips share a voice.
        `"row"` appends the clip's index within its group, giving each clip its own
        draw.

    Returns
    -------
    str
        The key. Always a string, since `source_id` is a string and `term_index` an
        int, and hashing must not depend on which.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    group = row.get(field, row.get("term_index"))
    if group is None:
        raise KeyError(f"row has neither {field!r} nor 'term_index': {sorted(row)}")
    if mode == "group":
        return str(group)
    # `chunk_index` FIRST, and the order is the whole correctness of "row" mode on the
    # document stages: 03_PARHAF and 04_PARROT rows carry BOTH fields, with
    # `variant_index` hard 0 on every row and the real position in `chunk_index`.
    # Reading `variant_index` first therefore collapsed every chunk of a document to
    # the key "<source_id>:0", which is exactly the group-mode key: "row" mode was a
    # silent no-op there, with no error and a plausible-looking assignment. The term
    # stages (dictionary, drugs, acronyms) have no `chunk_index` at all, so they are
    # unaffected by the order. This mirrors `item_index` in
    # 99_hf_release/01_build_nemo_manifest.py, which already preferred `chunk_index`.
    index = row.get("chunk_index", row.get("variant_index", 0))
    return f"{group}:{index}"


def voice_for_key(key: str, voices: tuple[str, ...] | list[str] = VOICES_FR_V2) -> str:
    """Return the voice that reads every clip whose assignment key is `key`.

    Parameters
    ----------
    key
        From `assignment_key`.
    voices
        The roster, in a fixed order. Order matters: it is part of the mapping, so
        re-ordering the roster re-assigns every voice.

    Returns
    -------
    str
        One name from `voices`.

    Deterministic across processes, machines and Python versions: an 8-byte blake2b
    digest of the UTF-8 key, big-endian, modulo the roster size. See the module
    docstring for why this is a hash and not a counter.
    """
    if not voices:
        raise ValueError("voices is empty")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return voices[int.from_bytes(digest, "big") % len(voices)]


def voice_for_row(row: dict, *, field: str | None = None, mode: str = "group",
                  voices: tuple[str, ...] | list[str] = VOICES_FR_V2) -> str:
    """Return the voice for one row, resolving its group field if not given.

    Parameters
    ----------
    row
        One row of a stage's `generated_dataset.jsonl`.
    field
        The group field; resolved from `row` itself when omitted. Pass it when
        iterating a whole file, so the choice is made once from the first row and a
        single row missing `source_id` cannot silently regroup the rest.
    mode
        See `assignment_key`.
    voices
        The roster.

    Returns
    -------
    str
        One name from `voices`.
    """
    field = field or group_field(row)
    return voice_for_key(assignment_key(row, field=field, mode=mode), voices)


def load_voices(path) -> tuple[str, ...]:
    """Read a roster file: one voice name per line, `#` comments and blanks ignored.

    Parameters
    ----------
    path
        Path to the roster (the format of
        `voxtral/cloning/voices_fr_production.txt` in the CrispASR repo, so that file
        can be passed directly).

    Returns
    -------
    tuple of str
        The names, in file order, which is part of the mapping.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    voices = tuple(s for line in lines if (s := line.strip()) and not s.startswith("#"))
    if not voices:
        raise ValueError(f"no voice names in {path}")
    if len(set(voices)) != len(voices):
        raise ValueError(f"duplicate voice names in {path}: {voices}")
    return voices


# The file stage 05 drops next to the audio and stage 99 reads back. It is the only
# record of WHICH roster a directory of clips was generated with: the voice is not
# stored per clip (602k rows of a value re-derivable from the key would be dead
# weight), so the release side re-runs the hash, and that is only sound while the
# roster, the mode and the group field are the ones synthesis used. Both sides go
# through the two functions below so the format cannot drift between them.
PROVENANCE_FILENAME = "voice_assignment.json"


def provenance_key(input_path) -> str:
    """Return the key one stage's entry is stored under in the provenance file.

    Parameters
    ----------
    input_path
        The stage's `generated_dataset.jsonl`.

    Returns
    -------
    str
        `"<stage folder>/<filename>"`, e.g. `"01_dictionnary/generated_dataset.jsonl"`.

    Not the filename alone, because every stage's input is called
    `generated_dataset.jsonl` and several stages can be generated into one output
    directory, and not the absolute path, because that bakes a home path into a file
    that sits beside the published audio.
    """
    p = Path(input_path)
    return f"{p.parent.name}/{p.name}"


def write_provenance(out_dir, input_path, *, voices, mode: str, field: str | None,
                     counts: dict) -> Path:
    """Record which roster this audio was generated with, beside the audio.

    Parameters
    ----------
    out_dir
        The directory the clips are written to.
    input_path
        The stage's `generated_dataset.jsonl`, used for the key.
    voices
        The roster, in order, or None for a single-voice run.
    mode
        `"group"`, `"row"` or `"single"`.
    field
        The group field actually used, or None in single mode.
    counts
        Rows per voice, over every planned row of the input (not just the rows
        generated in this run), so the entry describes the corpus and not the tail
        of a resume.

    Returns
    -------
    Path
        The file written.

    Merged into any existing file rather than overwriting it: the five stages share
    one `data/` parent in some layouts, and a second stage must not erase the first
    stage's entry.
    """
    path = Path(out_dir) / PROVENANCE_FILENAME
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        doc = {}
    key = provenance_key(input_path)
    doc[key] = {
        "input": key,
        "voices": list(voices) if voices else None,
        "mode": mode,
        "group_field": field,
        "rows_per_voice": dict(sorted(counts.items())),
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path


def read_provenance(audio_dir, input_path) -> dict | None:
    """Return the provenance entry for one stage's audio, or None if there is none.

    Parameters
    ----------
    audio_dir
        The directory holding that stage's clips.
    input_path
        The stage's `generated_dataset.jsonl`.

    Returns
    -------
    dict or None
        The entry written by `write_provenance` (`voices`, `mode`, `group_field`,
        `rows_per_voice`), or None when the file or the entry is absent, which is
        what v1 audio looks like: generated before this existed, single voice.

    A missing file is not an error. The caller decides what to do with it (stage 99
    leaves the `voice` column out rather than guessing a name).
    """
    path = Path(audio_dir) / PROVENANCE_FILENAME
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = doc.get(provenance_key(input_path))
    return entry if isinstance(entry, dict) else None
