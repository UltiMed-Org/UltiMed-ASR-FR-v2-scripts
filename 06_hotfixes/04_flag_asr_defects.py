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
gated at 30 (~5 French words). The two are not merged on purpose. This file's output
drops clips from the release, so it waits for two recognizers to agree; the live one
only asks for a listen, so it can afford one and can therefore run during generation.
Neither is a superset of the other (a character run catches a skipped phrase the word
rule rejects as a compound; the word rule catches a skip the run splits in two), and
the measurement that would settle whether one can replace the other is to run both over
the same hypotheses file and diff their catches. Until then, do not add a third.

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
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils"))
from nemo_manifest import clip_key, read_jsonl, resolve_audio  # noqa: E402

DEFAULT_OUT = _HERE.parent / "99_hf_release" / "asr_flagged_clips.jsonl"
#: The judge whose readings `--judge-key` picks out of a state file: the parakeet sweep,
#: because the defect this script exists for is the one Whisper is blind to.
DEFAULT_JUDGE_KEY = "onnx-asr/parakeet-tdt-0.6b-v3-ultra"
PREAMBLE = "preamble"
SKIPPED = "skipped"
MIN_PREAMBLE_WORDS = 2
MIN_SKIPPED_WORDS = 3
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


def find_defect(label: str, hyp: str, stt: str | None) -> tuple[str, str] | None:
    """Return ``(reason, offending words)`` for a defective clip, else ``None``.

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
    """
    ref, h = words(label), words(hyp)
    ops = difflib.SequenceMatcher(a=ref, b=h, autojunk=False).get_opcodes()
    tag, i1, i2, j1, j2 = ops[0]
    # A pure insertion only: a "replace" at the start is usually the model splitting the
    # label's first word ("Suvreza" -> "Essus UV réza"), not extra audio.
    if tag == "insert" and j2 - j1 >= MIN_PREAMBLE_WORDS:
        return PREAMBLE, " ".join(h[j1:j2])
    if not stt:
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
def main(hyps: Path, manifests: tuple[Path, ...], out: Path, judge_key: str) -> None:
    """Flag TTS preambles and confirmed skips from a non-Whisper recognizer's output."""
    hyp_by_key = load_hyps(hyps, judge_key)
    if not hyp_by_key:
        raise SystemExit(f"no hypotheses in {hyps} (wrong --judge-key for a state file?)")
    flagged, n_checked = {}, 0
    for m in manifests:
        for row in read_jsonl(m):
            key = clip_key(resolve_audio(m.parent, row["audio_filepath"]))
            if key in flagged or key not in hyp_by_key:
                continue
            n_checked += 1
            defect = find_defect(row["text"], hyp_by_key[key], row.get("stt_transcript"))
            if defect:
                flagged[key] = {"clip": key, "reason": defect[0], "words": defect[1],
                                "text": row["text"], "hyp": hyp_by_key[key]}
    logger.info(f"checked {n_checked} clips ({len(hyp_by_key)} hypotheses), flagged {len(flagged)}: "
                f"{sum(f['reason'] == PREAMBLE for f in flagged.values())} {PREAMBLE}, "
                f"{sum(f['reason'] == SKIPPED for f in flagged.values())} {SKIPPED}")
    with out.open("w") as f:
        for key in sorted(flagged):
            f.write(json.dumps(flagged[key], ensure_ascii=False) + "\n")
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
