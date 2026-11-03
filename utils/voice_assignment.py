"""Which of the nine French voices reads a given row, decided once and re-derivable.

v1 is single-voice (`fr_female` throughout), which makes a speaker-independent ASR
model learn one speaker's articulation. v2 spreads the same corpus over nine voices,
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
over the roster, which was ten voices when that was measured, so 116 to 194 clips per
voice): acceptable, because the model sees about 67k clips per voice overall and the
skew lives in the two smallest stages.

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

# The nine French voices cleared for v2, in roster order. Chosen by ear over two
# audition rounds, the second on medical text: FLEURS-clean prose did not predict
# behaviour on drug names and dosages and disagreed about 6 of 16 candidates. They are
# nine DISTINCT acoustic identities (four female, five male), not nine variants of
# three. That audition cleared TEN, five female and five male; `h_frm_chf` (heard as
# cheerful_female) was dropped on 2026-10-15 after the QC sample measured it derailing
# far more than the rest, which is why the roster is 4F/5M rather than even. See
# RETIRED_VOICES_FR_V2 below for those numbers and for why it was dropped outright
# instead of retired.
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
)

#: The lowest cfg_alpha each voice family was measured to stay usable at, from the
#: 2026-10-13 audition (16 French voices x 10 values from 0.1 to 5.0 on one dense
#: medical sentence, noise seed pinned; CrispASR `voxtral/cloning/cfg_sweep.sh`): the
#: usable band is 1.2-2.0 for the two shipped `fr_*` voices and **1.5-2.0 for the
#: hybrids**, which need more guidance to hold their articulation.
#:
#: This matters because the repair pass walks the guidance DOWN from 2.0 looking for a
#: draw that scores better, and seven of the nine v2 voices are hybrids. A single global
#: floor is wrong either way: 1.2 lets a hybrid clip be "repaired" by a draw two rungs
#: below its own floor (a better CER from audio the audition rejected), while 1.5 throws
#: away two usable rungs for the `fr_*` voices.
FLOOR_HYBRID, FLOOR_BASE = 1.5, 1.2


def is_hybrid(voice: str | None) -> bool:
    """True for the seven hybrid voices, which are named for it (`h_<source><target>`).

    >>> is_hybrid("h_frf_itm"), is_hybrid("fr_male"), is_hybrid(None)
    (True, False, False)
    """
    return bool(voice) and voice.startswith("h_")


def cfg_alpha_floor(voice: str | None, *, hybrid: float = FLOOR_HYBRID,
                    base: float = FLOOR_BASE) -> float:
    """The measured floor for one clip's voice. Unknown voices get the stricter one.

    Stricter rather than looser on an unknown name because the cost is asymmetric: a
    floor set too high loses a rung of a retry ladder, while one set too low ships audio
    a human audition already judged unusable, and nothing downstream re-auditions it.

    >>> cfg_alpha_floor("h_itf_arm"), cfg_alpha_floor("fr_female")
    (1.5, 1.2)
    >>> cfg_alpha_floor("some_new_voice")
    1.5
    """
    return base if voice in VOICES_FR_V2 and not is_hybrid(voice) else hybrid


#: Voices dropped from the roster after measurement. They stay listed in
#: `VOICES_FR_V2` on purpose: see `voice_for_key` for why deleting an entry is NOT
#: how a voice is retired, and what it would cost.
RETIRED_VOICES_FR_V2: tuple[str, ...] = ("h_frf_itm", "h_frm_ptf", "h_frf_ptm",
                                          "h_itf_arm")
#: Retired 2026-11-03 on a TRAINING measurement, not a QC one. A per-voice attribution
#: sweep ran nine 1500-step parakeet fine-tunes, each on one voice's UltiMed slice plus
#: the same constant non-UltiMed sets, and ranked them on the mean raw WER of the two
#: REAL-speech validation sets (oli and FLEURS-fr). The ranking split in two with
#: nothing in between: fr_female 0.0638, h_itf_def 0.0640, fr_male 0.0641, h_frm_nlf
#: 0.0648, h_frf_esm 0.0648, then a gap to h_frf_itm 0.0667, h_frm_ptf 0.0672,
#: h_frf_ptm 0.0674 and h_itf_arm 0.0687, the four retired here. Both shipped voices sit
#: in the good tier, so the gap is damage done by the hybrid surgery, not ordinary
#: spread between voices. The mixture arms agree: dropping voices took oli from 7.4 to
#: 4.9 raw WER at 3k steps.
#:
#: Why no QC judge saw it: every judge (stage 06's CER gate, the duration audit, the
#: synthetic val sets dictionary and parhaf) scores how well an ASR model reads the
#: TTS audio back, and the synthetic val sets are spoken by these same voices. They did
#: not separate the tiers. What these four teach a model transfers worse to real
#: speech, which only a fine-tune evaluated on real speech can measure.
#:
#: Retired by re-draw (see `voice_for_key`), so the keys whose first draw landed on one
#: of the four (~44 pct of the corpus, 4 of 9) move uniformly to the five survivors,
#: and every other key keeps its voice bit-identical.
#:
#: Before that, `h_frm_chf` (heard as cheerful_female) failed too, and on 2026-10-15
#: it was deleted from `VOICES_FR_V2` outright rather than
#: retired here. That is the move `voice_for_key` warns against, and it was safe
#: exactly once, for a reason that will not come back: it was decided while the whole
#: v2 corpus was being deleted anyway (148,303 dictionary clips, so that every clip
#: would carry the code sidecar the gate writes), so there was no audio on disk for the
#: re-draw to mislabel. The same edit DURING a run re-assigns every key and silently
#: mislabels every clip already written.
#:
#: The measurement that condemned it: 14 of 180 clips over stage 06's CER gate in the
#: unbiased 5% QC sample, against 4 of 165 for the next worst voice and zero for four
#: of the ten, every `loop` and `overrun` among the first 51 flagged clips belonging to
#: it, and p90 CER 0.195 against 0.014 to 0.036 for the rest. Its median CER (0.0069)
#: was mid-pack: it read most sentences well and then occasionally lost the text
#: entirely. It derailed, it was not bad, which is why the first plan was to keep it
#: and let the code gate reject its bad takes; the dataset was then held to a higher
#: bar than "recoverable by retry".
#:
#: Once audio exists retirement is the only safe tool: re-drawing leaves every key that
#: did not land on a retired voice bit-identical (~89 pct of the corpus per voice
#: retired at nine voices, ~56 pct for the four above), whereas editing the roster
#: means regenerating all of it.

#: How many re-draws `voice_for_key` tries before falling back to a direct draw over
#: the survivors. Only reachable when most of the roster is retired; see there.
_RETIRE_MAX_ATTEMPTS = 64

#: The escalation ladder for a clip no draw could repair: the voices stage 06 hands it
#: after its own, in order, one rung per exhausted round of `--n-improv` draws. Used by
#: `ladder_voice` below and by `06_hotfixes/01_recursive_improvement.py`.
#:
#: Measured, not chosen by ear. In the 40-60 s duration band of the first full QC sweep
#: (2026-10-20), the only band where enough clips fail to rank voices at all, the
#: per-voice gate-fail rates split the roster in two with nothing in between:
#: fr_female 0.0 pct (n=34), h_itf_arm 1.4 (69), h_itf_def 2.2 (46), fr_male 5.8 (52),
#: then a cliff to h_frf_esm 41.5 (94), h_frm_ptf 44.2 (104), h_frf_itm 48.3 (89),
#: h_frm_nlf 60.2 (88) and h_frf_ptm 77.4 (115). So a clip five draws could not fix is
#: never retried at a voice from the second half: those four survivors are the ladder,
#: and a clip that runs out of rungs keeps its original audio (as it did in v1).
#:
#: Hybrids before bases, on purpose. The hybrids are why the roster has nine voices
#: rather than two, so the repair keeps the corpus accent spread as long as it can and
#: only falls back to the two shipped voices, which are also the two the corpus already
#: has most of, once h_itf_* has failed too.
#:
#: Since 2026-11-03 rung 1 effectively offers only h_itf_def: h_itf_arm is in
#: `RETIRED_VOICES_FR_V2` and `ladder_voice` skips retired names. The tuple is left as
#: measured, since it is history.
VOICE_LADDER_FR_V2: tuple[tuple[str, ...], ...] = (
    ("h_itf_arm", "h_itf_def"),   # rung 1: the two best hybrids
    ("fr_male", "fr_female"),     # rung 2: the shipped bases, last resort
)

#: The first voice a clip with a WRONG NUMBER gets when its own voice is an `h_itf_*`
#: one: any hybrid but those two, so the repair keeps the accent spread. A one-rung
#: ladder for `ladder_voice` (via `01_recursive_improvement.swap_voice_for(ladder=...)`);
#: a clip it cannot fix then escalates on `VOICE_LADDER_FR_V2` from rung 2, the fr pair.
#:
#: Measured 2026-10-28 on the shipped corpus, clips whose label holds a 70-79 or 90-99
#: number (the French "soixante-quinze" family), share where turbo AND parakeet both
#: heard the same wrong number: h_itf_def 25.7 pct (n=6,446), h_itf_arm 12.9 (6,407),
#: these five 0.34 to 0.69 (5,455-5,925 each), fr_male 0.16 (6,188), fr_female 0.07
#: (5,974). So the two itf voices carry an accent error, not seed noise, and the
#: other hybrids sit far closer to the fr pair than to them.
#:
#: Since 2026-11-03 this effectively offers only h_frf_esm and h_frm_nlf: the other
#: three are in `RETIRED_VOICES_FR_V2` and `ladder_voice` skips retired names. The
#: tuple is left as measured, since it is history.
NUMBER_LADDER_FR_V2: tuple[tuple[str, ...], ...] = (
    ("h_frf_esm", "h_frf_itm", "h_frf_ptm", "h_frm_nlf", "h_frm_ptf"),
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


def voice_for_key(key: str, voices: tuple[str, ...] | list[str] = VOICES_FR_V2,
                  retired: tuple[str, ...] | list[str] = RETIRED_VOICES_FR_V2) -> str:
    """Return the voice that reads every clip whose assignment key is `key`.

    Parameters
    ----------
    key
        From `assignment_key`.
    voices
        The roster, in a fixed order. Order matters: it is part of the mapping, so
        re-ordering the roster re-assigns every voice.
    retired
        Voices that must never be returned although they are still in `voices`. Pass
        `()` to get the mapping as it was before any voice was retired, which is how
        a maintenance pass finds the clips a now-retired voice already read.

    Returns
    -------
    str
        One name from `voices` that is not in `retired`.

    Deterministic across processes, machines and Python versions: an 8-byte blake2b
    digest of the UTF-8 key, big-endian, modulo the roster size. See the module
    docstring for why this is a hash and not a counter.

    **A voice is retired by re-drawing, never by deleting its roster entry.** The
    mapping is `voices[digest % len(voices)]`, so dropping one name changes
    `len(voices)` and therefore re-assigns *every* key: with nine voices that would
    relabel ~89 pct of the corpus, and because the resume check in
    `01_generate_audio.py` matches filenames, which carry no voice, the audio already
    on disk would quietly become mislabelled rather than be regenerated. That is the
    same failure the module docstring rejects a counter for. Instead the first draw is
    made over the FULL roster exactly as before, and only when it lands on a retired
    voice is the key re-drawn with an attempt suffix until a survivor comes up. Three
    properties follow, and all three are what make retiring a voice mid-run safe:

    1. every key whose first draw survives keeps its voice bit for bit, so the audio
       already written stays correct and is never regenerated;
    2. the accepted draw is uniform over the survivors, because each attempt is
       uniform over the roster and conditioning on acceptance does not favour any
       survivor, so the retired voice's share is spread evenly rather than dumped on
       whichever name replaced it;
    3. retirements compose: retiring a second voice later leaves every key that never
       landed on it untouched.

    The cost is ~`1 / (1 - retired_fraction)` hashes per call, i.e. 1.125 for one
    voice in nine. After `_RETIRE_MAX_ATTEMPTS` failures it draws directly over the
    survivors instead, which keeps the function total when nearly the whole roster is
    retired (at eight of nine retired, a bounded loop alone would leave ~1e-3 of keys
    undecided).

    Deleting a roster entry was done once for real, deliberately, and only because the
    audio was being deleted in the same operation: see `RETIRED_VOICES_FR_V2` for why
    dropping `h_frm_chf` on 2026-10-15 was not an exception to the rule above, and for
    the four voices retired the proper way on 2026-11-03.

    Examples
    --------
    >>> voice_for_key("42", retired=()) in VOICES_FR_V2
    True
    >>> voice_for_key("42") not in RETIRED_VOICES_FR_V2
    True
    >>> # Retiring a voice leaves the other eight ninths of the keys alone. Passed
    >>> # explicitly so the example isolates one voice from the live retirements.
    >>> same = sum(voice_for_key(str(k), retired=("h_frm_ptf",))
    ...            == voice_for_key(str(k), retired=())
    ...            for k in range(10000))
    >>> 8500 < same < 9500
    True
    >>> # ...and the retired voice is then never drawn.
    >>> any(voice_for_key(str(k), retired=("h_frm_ptf",)) == "h_frm_ptf"
    ...     for k in range(10000))
    False
    >>> # The live retirement of four voices (2026-11-03) moves ~44 pct of the keys.
    >>> live = sum(voice_for_key(str(k)) == voice_for_key(str(k), retired=())
    ...            for k in range(10000))
    >>> 5100 < live < 6100
    True
    """
    if not voices:
        raise ValueError("voices is empty")
    retired = tuple(retired or ())
    survivors = tuple(v for v in voices if v not in retired)
    if not survivors:
        raise ValueError("every voice in the roster is retired")

    for attempt in range(_RETIRE_MAX_ATTEMPTS):
        # Attempt 0 hashes the bare key, which is what keeps the pre-retirement
        # mapping bit-identical for every key that does not land on a retired voice.
        # The NUL separator cannot occur in a key, so an attempt suffix can never
        # collide with another key's bare form.
        salted = key if attempt == 0 else f"{key}\x00{attempt}"
        digest = hashlib.blake2b(salted.encode("utf-8"), digest_size=8).digest()
        pick = voices[int.from_bytes(digest, "big") % len(voices)]
        if pick not in retired:
            return pick
    # Unreachable for any sane roster (1e-64 at one voice in ten). Deterministic, so
    # the few keys that get here still have one stable voice.
    digest = hashlib.blake2b(f"{key}\x00final".encode("utf-8"), digest_size=8).digest()
    return survivors[int.from_bytes(digest, "big") % len(survivors)]


def voice_for_row(row: dict, *, field: str | None = None, mode: str = "group",
                  voices: tuple[str, ...] | list[str] = VOICES_FR_V2,
                  retired: tuple[str, ...] | list[str] = RETIRED_VOICES_FR_V2) -> str:
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
    retired
        Voices never to return; see `voice_for_key`. Pass `()` for the mapping as it
        was before any retirement.

    Returns
    -------
    str
        One name from `voices` that is not in `retired`.
    """
    field = field or group_field(row)
    return voice_for_key(assignment_key(row, field=field, mode=mode), voices,
                         retired=retired)


def swap_voice(key: str, current: str | None, *,
               voices: tuple[str, ...] | list[str] = VOICES_FR_V2,
               retired: tuple[str, ...] | list[str] = RETIRED_VOICES_FR_V2,
               avoid: tuple[str, ...] | list[str] = ()) -> str | None:
    """Return a DIFFERENT voice for one clip, or None when the roster has no other.

    Parameters
    ----------
    key
        Any string that identifies the clip stably. Unlike `assignment_key` this does
        NOT have to be the key synthesis hashed: a swap is *recorded* (stage 06 writes
        the new name on the row and `99_hf_release/03_sync_hotfix_results.py` carries
        it into the manifests), never re-derived, so the key only has to give the same
        answer on a re-run. Stage 06 passes the clip stem, which is the one identifier
        its manifest row and the audio file agree on.
    current
        The voice that already read this clip and must not come back. None (v1
        single-voice audio, nothing recorded) still returns a voice, but the caller
        usually has no roster to trust in that case.
    voices
        The roster, in order. Pass the one the audio was generated with
        (`read_dir_provenance(...)["voices"]`), not the repo's current
        `VOICES_FR_V2`, for the same reason stage 99 does.
    retired
        Voices that must never be returned; see `voice_for_key`.
    avoid
        Voices to skip *if the roster can afford it*: a preference, not a rule. Stage
        06 passes the voices already reading the same term, so swapping a bad take
        keeps the term's accent spread instead of collapsing two of its clips onto one
        voice. When `avoid` would leave nothing, it is dropped and only `current` and
        `retired` are honoured, because getting a different voice matters more than
        the spread.

    Returns
    -------
    str or None
        A name from `voices`, never `current` and never in `retired`. None only when
        that leaves nothing at all (a one-voice roster).

    Uniform over the survivors and deterministic, because it is the same salted
    re-draw `voice_for_key` uses for a retired voice: `current` (and `avoid`) are
    simply passed as `retired` for this one call. Nothing is duplicated, and a clip
    swapped twice in two runs lands on the same voice both times.

    Examples
    --------
    >>> voice_for_key("42")
    'h_frf_esm'
    >>> swap_voice("42", "h_frf_esm")
    'h_frm_nlf'
    >>> # h_frf_itm and the other retired voices are skipped, so the pick lands on a
    >>> # survivor (h_frf_itm before the 2026-11-03 retirement).
    >>> swap_voice("42", "h_frf_esm", avoid=("h_frm_nlf",))
    'fr_female'
    >>> # `avoid` is a preference: asking it to skip the whole roster still answers.
    >>> swap_voice("42", "h_frf_esm", avoid=VOICES_FR_V2)
    'h_frm_nlf'
    >>> swap_voice("42", "only", voices=("only",)) is None
    True
    """
    # `current` is a hard exclusion like a retirement, `avoid` a soft one, so the two
    # tiers are the same draw with a longer and a shorter block list.
    hard = tuple(dict.fromkeys(v for v in (*tuple(retired or ()), current) if v))
    soft = tuple(dict.fromkeys(v for v in (avoid or ()) if v and v not in hard))
    for blocked in ((*hard, *soft), hard):
        if any(v not in blocked for v in voices):
            return voice_for_key(key, voices, retired=blocked)
    return None


def ladder_voice(key: str, rung: int, *,
                 ladder: tuple[tuple[str, ...], ...] = VOICE_LADDER_FR_V2,
                 roster: tuple[str, ...] | list[str] | None = None,
                 tried: tuple[str | None, ...] | list[str | None] = (),
                 retired: tuple[str, ...] | list[str] = RETIRED_VOICES_FR_V2,
                 avoid: tuple[str, ...] | list[str] = ()) -> str | None:
    """One rung of `VOICE_LADDER_FR_V2`: the next voice to try, or None when out.

    The sibling of `swap_voice`, for the other question. `swap_voice` answers "any
    other voice of this roster", which is what a one-shot swap needs; this answers
    "the next voice MEASURED to read what the clip's own voice could not", which is a
    shortlist of two per rung, so unlike `swap_voice` it can legitimately have no
    answer at all.

    Parameters
    ----------
    key
        Any string identifying the clip, hashed exactly as everywhere else so the pick
        is deterministic and resume stable. Stage 06 keys a `row`-mode clip on its stem
        and a `group`-mode clip on its group id, so that the repaired chunks of one
        document all land on the SAME new reader instead of one each.
    rung
        0-based index into `ladder`: how many rungs this clip has already used up.
    ladder
        Overridable for tests and for a future roster; the default is the measured one.
    roster
        The voices the audio directory was generated with, when known. A rung voice
        that directory never had is skipped rather than drawn, so a v1 or custom
        `--voices-file` run is never handed a name its generator does not know.
    tried
        Voices this clip already failed at (its original, plus whatever earlier rungs
        gave it). Hard exclusions, and None entries are ignored so a caller can pass
        `(row.get("voice"), imp.get("swap_voice"))` as is.
    retired
        Same meaning as everywhere else, and a hard exclusion here too: a retired voice
        on the ladder is skipped rather than re-drawn, since a two-name rung has no
        room to re-draw within.
    avoid
        Soft preference, same as in `swap_voice`: voices already reading the same term,
        dropped when honouring it would leave the rung empty.

    Returns
    -------
    str or None
        A name from `ladder[rung]`, or None when `rung` is past the end of the ladder
        or every voice on that rung is excluded. Both mean one thing to the caller:
        this clip is out of voices.

    Examples
    --------
    >>> # Live roster: h_itf_arm is retired (2026-11-03), so rung 1 only offers
    >>> # h_itf_def whatever the key.
    >>> (ladder_voice("42", 0), ladder_voice("13", 0))
    ('h_itf_def', 'h_itf_def')
    >>> ladder_voice("42", 1)
    'fr_male'
    >>> ladder_voice("42", 2) is None     # past the end of the ladder
    True
    >>> # The examples below pass `retired=()` to show the mechanism on a full rung.
    >>> # Another key, uniformly, lands on the other voice of each rung.
    >>> (ladder_voice("42", 0, retired=()), ladder_voice("13", 0, retired=()),
    ...  ladder_voice("13", 1))
    ('h_itf_arm', 'h_itf_def', 'fr_female')
    >>> # What the clip already failed at is skipped; None entries are ignored.
    >>> ladder_voice("13", 0, tried=("h_itf_def", None), retired=())
    'h_itf_arm'
    >>> ladder_voice("13", 0, tried=("h_itf_def", None)) is None   # live: rung empty
    True
    >>> ladder_voice("13", 0, tried=("h_itf_arm", "h_itf_def"), retired=()) is None
    True
    >>> # A directory whose roster never had the hybrids skips straight past them.
    >>> ladder_voice("42", 0, roster=("fr_male", "fr_female")) is None
    True
    >>> # `avoid` is a preference, not a rule: it is dropped before giving up.
    >>> ladder_voice("13", 0, avoid=("h_itf_def",), retired=())
    'h_itf_arm'
    >>> ladder_voice("13", 0, avoid=("h_itf_arm", "h_itf_def"), retired=())
    'h_itf_def'
    """
    if rung < 0 or rung >= len(ladder):
        return None
    blocked = {v for v in (*tuple(tried or ()), *tuple(retired or ())) if v}
    pool = [v for v in ladder[rung] if v not in blocked]
    if roster:
        known = set(roster)
        pool = [v for v in pool if v in known]
    if not pool:
        return None
    # Same two-tier draw as `swap_voice`: honour `avoid` when the rung can afford it,
    # drop it rather than answer None, and let the shared hash make the choice so a
    # rung reached twice in two runs lands on the same voice both times.
    soft = [v for v in pool if v not in set(avoid or ())]
    return voice_for_key(key, tuple(soft or pool), retired=())


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


def read_dir_provenance(audio_dir) -> dict | None:
    """Return the provenance entry for a directory of clips, when it is unambiguous.

    Parameters
    ----------
    audio_dir
        The directory holding the clips.

    Returns
    -------
    dict or None
        The single entry in that directory's provenance file, or, when several stages
        generated into it, the first entry *if they all agree* on `voices`, `mode` and
        `group_field`. None when there is no file, no entry, or the entries disagree.

    `read_provenance` is for a caller that knows which stage it is reading
    (`99_hf_release` iterates one `generated_dataset.jsonl` at a time). This one is for
    a caller that has only the audio: `06_hotfixes` works from the published manifest,
    where a row says which clip and which voice but not which stage's input produced
    it. Disagreement answers None rather than guessing, because the two facts it is
    asked for (which roster, and whether the voice was drawn per clip or per document)
    are exactly what a wrong pick would get wrong.
    """
    path = Path(audio_dir) / PROVENANCE_FILENAME
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    entries = [v for v in doc.values() if isinstance(v, dict)]
    if not entries:
        return None
    shape = {(tuple(e.get("voices") or ()), e.get("mode"), e.get("group_field"))
             for e in entries}
    return entries[0] if len(shape) == 1 else None
