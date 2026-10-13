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
field mirrors `nemo_manifest.detect_grouping` (`source_id` when the rows carry one,
else `term_index`) so that both stages group identically.
"""
from __future__ import annotations

import hashlib

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

# How to turn a row into the thing that gets hashed. "group" is the default and the
# only mode used for v2; "row" exists for the experiment where one term's variants are
# deliberately spread over several speakers.
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

    Mirrors `nemo_manifest.detect_grouping`'s auto choice on purpose. If these two
    ever disagree, a clip's `group_id` and its `voice` would be grouped on different
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
        `"row"` appends the variant index, giving each clip its own draw.

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
    return f"{group}:{row.get('variant_index', row.get('chunk_index', 0))}"


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
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    voices = tuple(s for line in lines if (s := line.strip()) and not s.startswith("#"))
    if not voices:
        raise ValueError(f"no voice names in {path}")
    if len(set(voices)) != len(voices):
        raise ValueError(f"duplicate voice names in {path}: {voices}")
    return voices
