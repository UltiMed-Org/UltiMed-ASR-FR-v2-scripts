#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Flag clips whose audio does not say their label, from a fine-tuned Parakeet sweep.

Stage 06's Whisper QC misses one TTS failure: voxtral sometimes babbles a few words
BEFORE the sentence ("Jackie Orbad décide sa voix de tension à T. Courrier de suivi...",
"Pardon, Madame Petit. Prescription de..."), and Whisper silently skips that preamble,
so the clip scored clean. A Parakeet fine-tuned on UltiMed transcribes it faithfully.
Found on 2026-09-28 by transcribing val/test/PARROT with run 1.6.0 step 8000 (~14 such
clips in 116k), then the whole corpus.

Input: the hypotheses of a recognizer that is not Whisper, plus the manifests whose
labels to check. Either shape is accepted (see ``load_hyps``): a hypotheses file
``{"audio": <path>, "hyp": <text>}`` per line, written by the NeMo repo's
``perso/transcribe_manifests.py``, or stage 06's own state file
(``improved/full.stt.jsonl``), where the parakeet sweep's readings already sit under
``transcriptions[<judge>]["text"]``. Only the last two path components
(``dictionary/012095_0006_sonde_de_churet.flac``) are compared, so the hypotheses may
have been made from another checkout or symlink.

The state file is the shape to reach for when the list is being REBUILT, which it has to
be whenever the audio is regenerated: these verdicts are keyed by clip name, so a list
measured on one generation of the audio would drop good clips of the next for defects
that no longer exist (``99_hf_release/04_drop_bad_rows.py::asr_flagged_stale`` is the
alarm for exactly that, and on 2026-10-26 it was ringing on 60 of the 63 listed clips).
Reading the state file costs no second transcription pass, and the readings there are the
ones the shipped audio produced.

A clip is flagged when, aligning label and hypothesis word by word:

- ``preamble``: the hypothesis STARTS with >= 2 words the label does not have (a pure
  insertion at position 0, not a replacement of the label's first word). The label always starts at the start of the audio, so words
  before it are TTS babble, never a model error on a correct clip.
- ``skipped``: >= 3 real words (>= 3 letters, not a spelled number or code letter, which
  recognizers glue into one token) of one label stretch are missing from the hypothesis,
  with less than half of the stretch's letters left in its place (so a compound written
  as one word is not a skip), AND at
  least half of them are missing from the stage-06 Whisper transcript too
  (``stt_transcript``). Two independent recognizers agreeing that the audio lacks a
  phrase means the TTS skipped it; one alone can be a recognition error.

Mid-sentence misreads (``2 grammes 10 grands cents`` for ``2,10 grammes``) are NOT
flagged: they cannot be told apart from a Parakeet error without listening.

The live CPU QC (``05_generate_audio/03_qc_cpu_stt.py``) flags the same ``skipped``
failure from a SINGLE recognizer, character-level: ``window_metrics`` in
``01_compute_stt.py`` returns the longest contiguous run of missing label characters,
gated at 30 (~5 French words). This file's output drops clips from the release, so it
waits for two recognizers to agree; the live one only asks for a listen, so it can
afford one and can therefore run during generation.

**The 2026-10-26 measurement**, which earlier versions of this docstring asked for ("run
both over the same hypotheses file and diff their catches"), is why ``skipped`` is now
OFF by default (`DEFAULT_REASONS`). Both rules were run over the first 19,738 v2 clips of
``improved/full.stt.jsonl``, with the parakeet-v3-ultra sweep as the hypothesis and the
whisper-turbo reading as the confirmation:

- ``preamble``: 7 clips, and all 7 are real, inspected one by one. Four are babble before
  the sentence ("El Rusem l'a donné ce jour, le 13 mars..." for a label starting "Ce
  jour le 13 mars"), three are the opening phrase spoken TWICE ("Sur le plan
  hématologique. Sur le plan hématologique, l'évolution..."). Every one of them passed
  the existing gates: whisper CER 0.0095 to 0.057 against a 0.08 threshold, and
  ``bad_run_chars`` 1 to 7 against 30, because an INSERTION at the start is what no
  current gate measures (``bad_run_chars`` counts missing reference characters only).
  Projected corpus-wide: ~210 clips, and they are repairable, so they belong in a redraw
  round before any drop.
- ``skipped``: 17 clips, and the character-level evidence contradicts most of them. Their
  spans are dosage and unit enumerations ("milligrammes est à porter de 8 heures à 20
  heures", "lisinopril 10 milligrammes deux comprimés le matin il prend lercan 20
  milligrammes un"), where both recognizers write digits for spelled numbers and
  ``_merged_diffs`` then chains a dozen one-word diffs into one long "missing" span. The
  tell is arithmetic: a clip that truly lost ten words of a ~500-character label cannot
  score CER 0.0049 with ``bad_run_chars`` 1, and that is exactly what clip
  ``006151_0000_cardiologie_00001_t0_c16`` scores. ``bad_run_chars`` was under 30 on all
  17. So on this corpus the word rule catches nothing real that the character gate
  misses, while it would drop ~520 long, dosage-dense PARHAF clips, which is the
  long-form data this dataset is built for. Re-enable it with ``--reason skipped`` on a
  corpus with fewer enumerations, or after teaching `_merged_diffs` a merge budget.

**Why ``--judge-key`` must stay on parakeet**, measured in the same run: the preamble
rule was also applied with the whisper-turbo reading as the hypothesis, over the same
19,729 clips. Turbo flags 9 clips, parakeet 7, and only 3 are the same clip. All 7
parakeet flags are real; of the 6 turbo-only ones, none is. They are whisper's own LM
pulling a filler to the front ("Il y a compte rendu de l'hospitalisation..." for a label
starting "Compte rendu", "C'est un traitement de sortie..." for "Traitement de sortie",
"Pour un patient de 41 ans" for "Patient de 41 ans") or re-spelling the first word
("Un decubitus ventral en observe" for "En décubitus ventral on observe"). That is the
same language prior that makes Whisper blind to real babble: it normalizes a disfluent
opening into something plausible instead of transcribing it. Parakeet-v3-ultra has no
such prior, which is exactly why it is the judge here. So a rebuild keyed on a Whisper
reading would queue good clips and still miss the defect, and a gate on this metric does
NOT belong in ``is_bad`` either: the alternating loop judges with Whisper, where the
metric is ~0 precision.

Consequence for the release, decided 2026-10-26: the flagged clips are DROPPED rather
than redrawn, although this repo's standing preference is repair. A redraw cannot work
here. ``01_recursive_improvement.py`` queues a clip from `is_bad` and promotes a draw
only when it beats the stored score by ``--min-improvement``; a babbling clip scores
CLEAN (CER 0.0095 to 0.057), so it is never queued, and a flawless new draw would not be
promoted either, because the number the promotion compares cannot see the defect. Making
repair possible would mean carrying this metric through scoring, queueing and promotion,
and no post-redraw check could confirm the fix without a second parakeet sweep. At ~210
clips of 602,789 (0.035%) the drop is the proportionate answer; the audio itself is left
on disk, so a later pass can revisit it.

Until then, do not add a third skipped-phrase detector.

**The ``repeated`` rule (2026-10-28), the loop both recognizers hear.** A clause said
twice ("hospitalisé du 14 août 2024 au 16 août 2024 au 16 août 2024", clip
``009599_0000_chir_ortho_et_traumato_00263_t0_c0``, voice h_frf_esm) costs a 40 to 90 s
PARHAF clip only ~0.05 of CER, under the 0.08 gate, and ``14_reopen_loop_clips.py``
asks for a parakeet CER trip on top of ``repeat_excess``, so the release audit of
2026-10-28 found such clips shipping after T5. The rule: a stretch of at least
``MIN_REPEATED_WORDS`` words the hypothesis says BEYOND the label, at least
``REPEATED_PAIR_SHARE`` of whose word pairs are the label's own (so it is the label
again, not a rewording or a babble, which the preamble rule owns), and Whisper's
transcript must hold such a stretch too. Two recognizers agreeing is the same bar as
``preamble``; this one is ON by default. Measured on the shipped ``full.jsonl``: 206
clips, and of 40 read side by side nearly all were real (the date range said twice is
the dominant family, then a dose or a finding said twice). Dropped, not redrawn, for the
reason given above for ``preamble``; the audio stays on disk.

**The number and inserted rules (2026-10-28), redrawn rather than dropped.** Two more
defects both recognizers hear, found when listening to clips the gates had passed:

  * ``number``: a label number both recognizers hear as the same OTHER number ("75 ans"
    said "soixante-cinq", "50 giga" said "cinq"). The comparison and its guards against
    formatting and alignment drift live in ``utils/number_swaps.py``, shared with the
    repair pass's number gate. 5,291 released clips, 2,643 of them h_itf_def and 1,404
    h_itf_arm; of 30 non-teen hits read in context nearly all were real TTS errors.
  * ``inserted``: a stretch of at least ``MIN_INSERTED_WORDS`` words carrying at least
    ``MIN_INSERTED_LETTERS`` letters more than what it replaced, heard by both at the
    same label position and sharing a content word. It catches the mutated repeat
    ``repeated`` cannot ("la calcémie est normale, la falcémie est normale, la
    magnésémie...", clip ``034548_0000_neurologie_00199_t0_c5``: the extra clause's
    word pairs are not the label's) and hallucinated additions ("importé du Cameroun du
    Cameroun"). 102 released clips; without the letter bar it was 366, mostly a
    recognizer splitting one compound into several words.

Both are OFF in ``DEFAULT_REASONS`` because this file's output DROPS clips, while these
labels are fine and a fresh draw usually says them right. ``utils/clip_ledger.py`` reports
them per clip (``text:number``, ``text:inserted``) for a repair round to queue.

Both texts are compared after lowercasing and dropping punctuation, like the NeMo
normalizer, which is enough since only whole missing or extra words matter here.

Output (default ``99_hf_release/asr_flagged_clips.jsonl``, committed): one row per
flagged clip with its key, reason and the offending words, sorted by key so a re-run
diffs cleanly. ``99_hf_release/04_drop_bad_rows.py`` drops the listed clips.

    uv run 06_hotfixes/04_flag_asr_defects.py <hyps.jsonl> <manifest> [<manifest> ...]

This file was written by Claude Code.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections.abc import Collection
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import clip_key, read_jsonl, resolve_audio  # noqa: E402
from number_swaps import agreed_swaps  # noqa: E402

DEFAULT_OUT = _HERE.parent / "99_hf_release" / "asr_flagged_clips.jsonl"
#: The judge whose readings `--judge-key` picks out of a state file: the parakeet sweep,
#: because the defect this script exists for is the one Whisper is blind to.
DEFAULT_JUDGE_KEY = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
PREAMBLE = "preamble"
SKIPPED = "skipped"
REPEATED = "repeated"
NUMBER = "number"
INSERTED = "inserted"
#: Which rules `--reason` may ask for, and which it asks for by default. `skipped` is OFF
#: by default: see "The 2026-10-26 measurement" in the module docstring. `number` and
#: `inserted` are off too, for the opposite reason: their clips are worth REDRAWING (the
#: label is fine, the take is not, and a fresh draw usually says it right), and this
#: file's output drops clips. `utils/clip_ledger.py` reports them so a repair round can
#: queue them; see "The number and inserted rules" in the module docstring.
REASONS = (PREAMBLE, SKIPPED, REPEATED, NUMBER, INSERTED)
DEFAULT_REASONS = (PREAMBLE, REPEATED)
MIN_PREAMBLE_WORDS = 2
MIN_SKIPPED_WORDS = 3
MIN_REPEATED_WORDS = 4
REPEATED_PAIR_SHARE = 0.75
MIN_INSERTED_WORDS = 3
#: Letters the inserted stretch must carry BEYOND what it replaced. Without it the rule
#: fires on a recognizer splitting one compound into several words
#: ("tomodensitométrique" -> "tomo dencytométrique"): more words, the same sounds.
MIN_INSERTED_LETTERS = 12
_PUNCT = re.compile(r"[^\w\s']")


def words(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


# Spelled numbers, which a recognizer writes as digits (one token for several words),
# and spoken signs it writes as symbols ("CD20 plus, CD5 moins" -> "CD20+, CD5-").
_NUMBER_WORDS = frozenset("""zéro un une deux trois quatre cinq six sept huit neuf dix onze douze
treize quatorze quinze seize vingt vingts trente quarante cinquante soixante cent cents mille
virgule et plus moins""".split())


def _is_content(word: str) -> bool:
    """A word of >= 3 letters that is not a spelled number or sign."""
    return len(word) >= 3 and word.isalpha() and word not in _NUMBER_WORDS


def _merged_diffs(ops: list[tuple]) -> list[tuple[int, int, int, int]]:
    """Adjacent non-equal opcodes merged into one ``(i1, i2, j1, j2)`` span: difflib may
    split one reworded stretch into an insert, one matching word, then a replace ("et
    typage hpa maternel et paternel" -> "étypage hpa maternelle" + "et" + "paternelle"),
    and the replace alone looks like 4 missing words. Spans at most one matching word
    apart are therefore merged."""
    spans: list[list[int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        if spans and i1 - spans[-1][1] <= 1 and j1 - spans[-1][3] <= 1:
            spans[-1][1], spans[-1][3] = i2, j2
        else:
            spans.append([i1, i2, j1, j2])
    return [tuple(sp) for sp in spans]


def _said_again(ref: list[str], hyp: list[str]) -> str | None:
    """The first stretch of at least `MIN_REPEATED_WORDS` words `hyp` says beyond `ref`
    that is built from `ref`'s own word pairs (the label said again), else ``None``.

    >>> _said_again("du 1 au 4 avril le motif".split(), "du 1 au 4 avril au 4 avril le motif".split()) is None
    True
    >>> _said_again("du 1 avril 2025 au 4 avril 2025 le motif".split(),
    ...             "du 1 avril 2025 au 4 avril 2025 au 4 avril 2025 le motif".split())
    'au 4 avril 2025'
    """
    pairs = set(zip(ref, ref[1:]))
    ops = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes()
    for i1, i2, j1, j2 in _merged_diffs(ops):
        if (j2 - j1) - (i2 - i1) < MIN_REPEATED_WORDS:
            continue
        extra = hyp[j1:j2]
        known = sum(p in pairs for p in zip(extra, extra[1:]))
        if known >= REPEATED_PAIR_SHARE * (len(extra) - 1):
            return " ".join(extra)
    return None


def _insertions(ref: list[str], hyp: list[str]) -> list[tuple[int, str, set[str]]]:
    """Stretches `hyp` says beyond `ref`: ``(label index, words, content words)``.

    >>> [(i, text, sorted(c)) for i, text, c in _insertions(
    ...     words("la calcémie est normale, la magnésémie aussi"),
    ...     words("la calcémie est normale la palcémie est normale la magnésémie aussi"))]
    [(5, 'palcémie est normale la', ['est', 'normale', 'palcémie'])]
    >>> _insertions(words("tomodensitométrique cervico thoracique"),
    ...             words("tomo dencytométrique cervico thoracique"))
    []
    """
    ops = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes()
    out = []
    for i1, i2, j1, j2 in _merged_diffs(ops):
        extra = hyp[j1:j2]
        if ((j2 - j1) - (i2 - i1) < MIN_INSERTED_WORDS
                or len("".join(extra)) - len("".join(ref[i1:i2])) < MIN_INSERTED_LETTERS):
            continue
        content = {w for w in extra if _is_content(w)}
        if len(content) >= 2:
            out.append((i1, " ".join(extra), content))
    return out


def _inserted_by_both(ref: list[str], hyp: list[str], stt: list[str]) -> str | None:
    """An insertion both recognizers hear at the same label position (two words of
    slack), sharing a content word, so it is the same extra audio and not two unrelated
    recognition errors."""
    theirs = _insertions(ref, stt)
    for i, text, content in _insertions(ref, hyp):
        if any(abs(i - k) <= 2 and content & c for k, _t, c in theirs):
            return text
    return None


def find_defect(label: str, hyp: str, stt: str | None,
                reasons: Collection[str] = REASONS) -> tuple[str, str] | None:
    """Return ``(reason, offending words)`` for a defective clip, else ``None``.

    `reasons` selects which rules run; the doctests below exercise all three, while the
    CLI defaults to `DEFAULT_REASONS`.

    >>> find_defect("Courrier de suivi.", "Jackie Orbad dit. Courrier de suivi.", None)
    ('preamble', 'jackie orbad dit')
    >>> find_defect("le patient est sorti hier soir", "le patient", "le patient")[0]
    'skipped'
    >>> find_defect("le patient est sorti hier soir", "le patient", "le patient est sorti hier soir") is None
    True
    >>> find_defect("marqueurs cd 11 cd 18 positifs", "marqueurs cd11 cd18 positifs", "marqueurs positifs") is None
    True
    >>> find_defect("cellules cd20 plus cd5 moins bcl2 plus cd10 plus et", "cellules cd20 cd5 bcl2 cd10 et", "cellules et") is None
    True
    >>> find_defect("une uvulo palato pharyngo plastie", "une uvulopalatopharyngoplastie", "une") is None
    True
    >>> find_defect("anticorps et typage hpa maternel et paternel ok", "anticorps étypage hpa maternelle et paternelle ok", "anticorps ok") is None
    True
    >>> find_defect("du 14 août 2024 au 16 août 2024. Le motif", "du 14 août 2024 au 16 août 2024 au 16 août 2024 le motif", "du 14 août 2024 au 16 août 2024, au 16 août 2024. Le motif")
    ('repeated', 'au 16 août 2024')
    >>> find_defect("du 14 août 2024 au 16 août 2024. Le motif", "du 14 août 2024 au 16 août 2024 au 16 août 2024 le motif", "du 14 août 2024 au 16 août 2024. Le motif") is None
    True
    >>> find_defect("Patiente de 75 ans.", "Patiente de 65 ans.", "patiente de 65 ans")
    ('number', '75 -> 65')
    >>> find_defect("Patiente de 75 ans.", "Patiente de 65 ans.", "patiente de 75 ans") is None
    True
    >>> calc = "la calcémie est normale, la magnésémie est normale"
    >>> looped = "la calcémie est normale la falcémie est normale la magnésémie est normale"
    >>> find_defect(calc, looped, looped.replace("falcémie", "palcémie"))
    ('inserted', 'est normale la falcémie')
    """
    ref, h = words(label), words(hyp)
    ops = difflib.SequenceMatcher(a=ref, b=h, autojunk=False).get_opcodes()
    tag, i1, i2, j1, j2 = ops[0]
    # A pure insertion only: a "replace" at the start is usually the model splitting the
    # label's first word ("Suvreza" -> "Essus UV réza"), not extra audio.
    if PREAMBLE in reasons and tag == "insert" and j2 - j1 >= MIN_PREAMBLE_WORDS:
        return PREAMBLE, " ".join(h[j1:j2])
    if not stt:
        return None
    # Whisper must hear the repeat too: one recognizer alone can be a recognition error.
    if REPEATED in reasons and (again := _said_again(ref, h)) and _said_again(ref, words(stt)):
        return REPEATED, again
    if NUMBER in reasons and (swapped := agreed_swaps(label, hyp, stt)):
        return NUMBER, ", ".join(f"{a} -> {b}" for a, b in swapped)
    if INSERTED in reasons and (extra := _inserted_by_both(ref, h, words(stt))):
        return INSERTED, extra
    if SKIPPED not in reasons:
        return None
    # Label word indices Whisper's transcript does match.
    s_ops = difflib.SequenceMatcher(a=ref, b=words(stt), autojunk=False).get_opcodes()
    heard_by_whisper = {i for t, a1, a2, _, _ in s_ops if t == "equal" for i in range(a1, a2)}
    for i1, i2, j1, j2 in _merged_diffs(ops):
        if (i2 - i1) - (j2 - j1) < MIN_SKIPPED_WORDS:
            continue
        # Count only real words: a spelled code ("cd 11 cd 18", "p t 4 n 0") or number
        # ("deux mille vingt quatre") loses words just because a recognizer glues it
        # into one token ("CD11", "pT4N0", "2024"), with nothing missing from the audio.
        span = [i for i in range(i1, i2) if _is_content(ref[i])]
        # And most of the stretch's LETTERS must be gone too: "uvulo palato pharyngo
        # plastie" -> "uvulopalatopharyngoplastie" drops 3 words but no sound.
        glued = len("".join(h[j1:j2])) > len("".join(ref[i1:i2])) / 2
        if len(span) >= MIN_SKIPPED_WORDS and not glued:
            if sum(i in heard_by_whisper for i in span) < len(span) / 2:
                return SKIPPED, " ".join(ref[i1:i2])
    return None


def load_hyps(path: Path, judge_key: str = DEFAULT_JUDGE_KEY) -> dict[str, str]:
    """``{clip_key: hypothesis}`` from either shape of input file.

    Two shapes, one meaning, so the caller does not have to care which it was handed:

    - ``{"audio": ..., "hyp": ...}``, what ``perso/transcribe_manifests.py`` writes;
    - a stage 06 state row, ``{"audio_filepath": ..., "transcriptions": {<judge>:
      {"text": ...}}}``, where a sweep's readings already live.

    ``hyp`` present decides, rather than the file name, so a hand-made file works too. A
    state row without `judge_key` is skipped silently: the sweeps are resumable, so a
    partial one is a normal state, and `main` reports the count it ended up with.

    Parameters
    ----------
    path : Path
        The hypotheses file or state file.
    judge_key : str
        Which reading to take out of a state row's ``transcriptions``.

    Returns
    -------
    dict[str, str]
        Hypothesis text by `nemo_manifest.clip_key`.
    """
    hyp_by_key: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "hyp" in row:
                hyp_by_key[clip_key(row["audio"])] = row["hyp"]
                continue
            reading = (row.get("transcriptions") or {}).get(judge_key)
            if isinstance(reading, dict) and reading.get("text"):
                hyp_by_key[clip_key(row["audio_filepath"])] = reading["text"]
    return hyp_by_key


@click.command()
@click.argument("hyps", type=click.Path(exists=True, path_type=Path))
@click.argument("manifests", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", default=str(DEFAULT_OUT), show_default="99_hf_release/asr_flagged_clips.jsonl",
              type=click.Path(path_type=Path))
@click.option("--judge-key", default=DEFAULT_JUDGE_KEY, show_default=True,
              help="Reading to use when HYPS is a stage 06 state file.")
@click.option("--reason", "reasons", multiple=True, type=click.Choice(REASONS),
              default=DEFAULT_REASONS, show_default=True,
              help="Which rules to apply. `skipped` is off by default, see the module "
                   "docstring's 2026-10-26 measurement.")
@click.option("--extra-hyps", "extra", multiple=True, metavar="FILE",
              type=click.Path(exists=True, path_type=Path),
              help="More hypothesis or state files, merged into HYPS. The corpus is held "
                   "in two state files (improved/ and improved_parrot/), and the second "
                   "one is how the separately licensed PARROT clips get checked too.")
def main(hyps: Path, manifests: tuple[Path, ...], out: Path, judge_key: str,
         reasons: tuple[str, ...], extra: tuple[Path, ...]) -> None:
    """Flag TTS preambles, repeats and confirmed skips from a non-Whisper recognizer's output."""
    hyp_by_key: dict[str, str] = {}
    for source in (hyps, *extra):
        hyp_by_key.update(load_hyps(source, judge_key))
    if not hyp_by_key:
        raise SystemExit(f"no hypotheses in {hyps} (wrong --judge-key for a state file?)")
    flagged, n_checked = {}, 0
    for m in manifests:
        for row in read_jsonl(m):
            key = clip_key(resolve_audio(m.parent, row["audio_filepath"]))
            if key in flagged or key not in hyp_by_key:
                continue
            n_checked += 1
            defect = find_defect(row["text"], hyp_by_key[key], row.get("stt_transcript"),
                                 reasons)
            if defect:
                flagged[key] = {"clip": key, "reason": defect[0], "words": defect[1],
                                "text": row["text"], "hyp": hyp_by_key[key]}
    logger.info(f"rules: {', '.join(reasons)}")
    logger.info(f"checked {n_checked} clips ({len(hyp_by_key)} hypotheses), flagged {len(flagged)}: "
                + ", ".join(f"{sum(f['reason'] == r for f in flagged.values())} {r}"
                            for r in REASONS))
    with out.open("w") as f:
        for key in sorted(flagged):
            f.write(json.dumps(flagged[key], ensure_ascii=False) + "\n")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
