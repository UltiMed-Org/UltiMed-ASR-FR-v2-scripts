#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1", "loguru>=0.7", "jiwer>=3.0", "requests>=2.31", "tqdm>=4.66", "soundfile>=0.12", "numpy"]
# ///
"""Which reference does 08_validate_manifests.py's `scores` check recompute against?

It must be `text`, the label. It used to be `asr_training_source`, and that made the
check fire on 3,678 rows of the v2 full manifest, every one of them a false alarm:
05_normalize_text.py rewrites the SOURCE with pronunciation respellings meant for the
TTS (`oestrogènes` -> `éstrogènes`, `foetal` -> `fétal`), which `normalize_for_scoring`
does not fold away, so a recompute against it charged the respelling as a recognition
error. Measured over those 3,686 rows: 0 matched a recompute against the source, 3,684
matched one against `text`, and all 318 tail scores matched `text`.

A release gate that always fails is a gate nobody reads, so what is pinned here is both
halves: the respelled row passes, and a genuinely stale score is still caught.

Run: uv run tests/test_validate_score_reference.py
This file was written with Claude Code.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PATH = HERE.parent / "99_hf_release" / "08_validate_manifests.py"
_spec = importlib.util.spec_from_file_location("validate_scores_mod", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_stt = _mod._load_stt()

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


#: The real divergence, taken from ../acronyms and ../PARHAF rows of the v2 release: the
#: label keeps the French spelling, the shipped source carries the TTS respelling, and
#: the ASR judge writes the label's spelling because that is what the audio sounds like.
LABEL = ("Les récepteurs aux oestrogènes sont positifs et le bilan foetal est normal.")
SOURCE = ("Les récepteurs aux éstrogènes sont positifs et le bilan fétal est normal.")
HEARD = ("Les récepteurs aux oestrogènes sont positifs et le bilan foetal est normal.")

#: The same respelling story, but loud enough to trip a gate rather than just move the
#: score: against the label this pair reads cer 0.0833, over the 0.080 gate, and against
#: the shipped source 0.0000. The LABEL/SOURCE trio above cannot stand in for it (0.0444,
#: under every gate), which is exactly why the gate half of the check went unnoticed.
GATE_LABEL = "Le foetus et l'oesophage."
GATE_SOURCE = "Le f\u00e9tus et l'\u00e9sophage."


def gate_row(**over) -> dict:
    """A row whose transcript matches its SOURCE, so only the label's spelling trips."""
    base = {"text": GATE_LABEL, "asr_training_source": GATE_SOURCE,
            "stt_transcript": GATE_SOURCE, "cer": 0.0}
    base.update(over)
    return row(**base)


def gate_warnings(r: dict) -> list[str]:
    return [k for k in score_warnings(r) if "QC gate" in k]


def row(**over) -> dict:
    stored = _stt.compute_metrics(LABEL, HEARD)[1]
    base = {"audio_filepath": "../PARHAF/000013_0000_x.flac", "duration": 20.3,
            "text": LABEL, "asr_training_source": SOURCE, "stt_transcript": HEARD,
            "stt_model": "whisper.cpp/ggml-large-v3-turbo", "qc_status": "regenerated",
            "cer": stored, "cer_tail": None}
    base.update(over)
    return base


def score_errors(r: dict) -> list[str]:
    f = _mod.Findings()
    _mod.check_scores([r], "unit", f, sample_percent=100.0, seed=0)
    return [kind for _check, kind in f.errors]


def score_warnings(r: dict) -> list[str]:
    f = _mod.Findings()
    _mod.check_scores([r], "unit", f, sample_percent=100.0, seed=0)
    return [kind for _check, kind in f.warnings]


#: The same read with one character wrong, so a PRISTINE row can carry a nonzero CER that
#: is nevertheless correct. That combination is the normal state of every row in this
#: release (median CER 0.0115) and it is what the todo list used to mistake for evidence
#: that a redraw had touched the row.
HEARD_TYPO = HEARD.replace("normal.", "normale.")


class CountingStt:
    """The scorer, wrapped to count how many rows actually get recomputed.

    The cost of this check is one `compute_metrics` per row in the todo list, so counting
    those calls is the only way to pin the SIZE of the list: a pristine row whose stored
    score is correct produces no finding whether it was recomputed or skipped, which is
    exactly why the 100-percent-coverage bug could sit here unnoticed through every run.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.calls = 0

    def compute_metrics(self, ref, hyp):
        self.calls += 1
        return self._real.compute_metrics(ref, hyp)

    def __getattr__(self, name):
        # Everything else (tail_cer, window_metrics, the gate constants) goes to the real
        # module untouched: only the per-row recompute is being counted, and a hand-listed
        # forwarder would break the next time this check reads one more thing from it.
        return getattr(self._real, name)


def run_counted(rows: list[dict], *, sample_percent: float) -> tuple[int, int]:
    """(recomputes, errors) for one check_scores run over `rows`."""
    counter = CountingStt(_mod._load_stt())
    saved = _mod._load_stt
    _mod._load_stt = lambda: counter
    try:
        f = _mod.Findings()
        _mod.check_scores(rows, "unit", f, sample_percent=sample_percent, seed=0)
        return counter.calls, len(f.errors)
    finally:
        _mod._load_stt = saved


def test_only_touched_rows_are_recomputed() -> None:
    """--sample-percent has to actually govern the pristine rows.

    Before this was fixed the todo list also admitted every row with a nonzero CER, so a
    602,710-row manifest recomputed 530,557 rows however small the sample was, at ~26 ms
    each: 3 h 42 min for ONE of the release's 43 manifests, and the whole gate over ten
    hours (measured 2026-10-26, which is when the chain was interrupted to fix this).
    """
    print("\nthe sample governs the pristine rows")
    pristine_cer = _stt.compute_metrics(LABEL, HEARD_TYPO)[1]

    # Guard the fixture: a pristine row has to carry a NONZERO but correct score, or the
    # bug this pins would not have been reachable.
    check("a correct pristine score is nonzero", pristine_cer > 0, str(pristine_cer))

    pristine = [row(audio_filepath=f"../dictionary/{i:06d}_0000_x.flac",
                    qc_status="original", stt_transcript=HEARD_TYPO, cer=pristine_cer)
                for i in range(200)]
    # One redrawn row with a genuinely stale score: the case the check exists for.
    stale = row(audio_filepath="../PARHAF/000013_0000_x.flac", qc_status="regenerated",
                cer=0.42)

    calls, errors = run_counted(pristine + [stale], sample_percent=0.0)
    # TWO alignments, not one, and only because this row's score matches neither
    # reference: the label is tried first and the shipped source only when that
    # disagrees, so an honest row still costs exactly one. That is what keeps the
    # either-reference rule from doubling the gate's runtime.
    check("at 0 percent only the redrawn row is recomputed", calls == 2, str(calls))
    check("and its stale score is still caught", errors == 1, str(errors))

    # The other half: the sample must not be dead the other way either. 200 pristine rows
    # at one alignment each, plus the stale row's two.
    calls_all, errors_all = run_counted(pristine + [stale], sample_percent=100.0)
    check("at 100 percent every row is recomputed", calls_all == 202, str(calls_all))
    check("and the pristine rows still report nothing", errors_all == 1, str(errors_all))


def main() -> int:
    print("the reference the scores check recomputes against")

    # Guard the fixture: if the scorer's own folds erased the respelling there would be
    # nothing to test, and a later change to normalize_for_scoring could make that true.
    cer_label = _stt.compute_metrics(LABEL, HEARD)[1]
    cer_source = _stt.compute_metrics(SOURCE, HEARD)[1]
    check("the respelling really moves the CER", cer_label != cer_source,
          f"both {cer_label}")
    check("and against the label the judge is nearly perfect", cer_label == 0.0,
          str(cer_label))

    check("a row whose score matches its LABEL passes", score_errors(row()) == [],
          str(score_errors(row())))

    # The same row scored against the source is what used to fail. Stating it as a value
    # pins the direction: the source inflates the CER, it does not merely differ.
    check("the source would have read worse", cer_source > cer_label,
          f"{cer_source} vs {cer_label}")

    # The check must still earn its keep: a genuinely stale score is caught.
    stale = score_errors(row(cer=0.42))
    check("a stale cer is still reported", len(stale) == 1, str(stale))
    check("and it is reported as a cer mismatch",
          stale and "cer matches neither reference" in stale[0], str(stale))

    # The population the 2026-10-26 release gate flagged: 1,271 cer plus 469 cer_tail
    # findings over 1,278 distinct clips, every one of which reproduces EXACTLY from the
    # shipped `asr_training_source` rather than from the label. Those rows were scored
    # after 05_normalize_text.py had rewritten the source, so the source is the honest
    # reference for them, while rows scored before it reproduce from the label. Accepting
    # only one of the two fails the other ordering on perfectly good rows, so the check
    # takes whichever agrees.
    src_scored = score_errors(row(cer=cer_source))
    check("a row scored against the SOURCE passes too", src_scored == [],
          str(src_scored))
    tail_source = _stt.tail_cer(SOURCE, HEARD, 42.1)
    src_tail = [k for k in score_errors(row(duration=42.1, cer_tail=tail_source))
                if "cer_tail" in k]
    check("and so does a cer_tail taken against the SOURCE", src_tail == [],
          str(src_tail))
    # A value that is neither reference is still caught, which is the whole point: the
    # rule widens what counts as honest, it does not switch the gate off.
    check("a score matching neither reference is still reported",
          len(score_errors(row(cer=0.42))) == 1, str(score_errors(row(cer=0.42))))

    # Third class from the same 2026-10-26 report: 7 rows "trip a QC gate on their own
    # stored transcript", and all 7 pass every gate against the source (cer 0.049-0.079
    # against 0.080, window 0.1455-0.1502 against 0.150). Stage 06 gates on `best_cer`,
    # the MIN over the references, so the recomputation has to as well or the release gate
    # accuses a respelling of being a misread.
    kept = gate_warnings(gate_row())
    check("a gate tripped only by the label's spelling is not reported", kept == [],
          str(kept))
    # And the gate must still fire when the BEST reference is the one failing: same row,
    # no source to fall back on.
    alone = gate_warnings(gate_row(asr_training_source=None))
    check("a row with no kinder reference still trips the gate", len(alone) == 1,
          str(score_warnings(gate_row(asr_training_source=None))))

    # A stale tail is its own finding, on a clip long enough to have a tail (over 30 s).
    long_row = row(duration=42.1, cer_tail=0.42)
    tail = [k for k in score_errors(long_row) if "cer_tail" in k]
    check("a stale cer_tail is reported too", len(tail) == 1, str(tail))

    # A MISSING tail on a long clip: the shape 5,250 v2 clips were in, and the one this
    # check used to read as "fine" because a null stored_tail skipped the comparison
    # entirely. Both directions have to be reported or the stale-duration bug of
    # 2026-10-26 can recur silently: a clip whose row says 42 s has an ending, and
    # nothing scored it.
    missing = [k for k in score_warnings(row(duration=42.1, cer_tail=None))
               if "no cer_tail" in k]
    check("a long clip with no cer_tail at all is reported", len(missing) == 1,
          str(score_warnings(row(duration=42.1, cer_tail=None))))
    # ...and it is a warning, not an error: a v1 row predating the tail gate has none
    # legitimately, and a gate that always fails is a gate nobody reads.
    check("and only as a warning, so the gate still passes",
          score_errors(row(duration=42.1, cer_tail=None)) == [],
          str(score_errors(row(duration=42.1, cer_tail=None))))
    # The symmetric case must keep working, and a SHORT clip with no tail is correct.
    check("a short clip with no cer_tail is not reported",
          [k for k in score_warnings(row()) if "cer_tail" in k] == [],
          str(score_warnings(row())))

    # A row with no transcript is the structure check's business, not this one's.
    check("a row with no transcript is skipped here",
          score_errors(row(stt_transcript=None)) == [])
    # And a row with no label cannot be scored at all (the label is REQUIRED elsewhere).
    check("a row with no label is skipped here", score_errors(row(text=None)) == [])

    test_only_touched_rows_are_recomputed()

    if failures:
        print(f"\nFAILED: {len(failures)}")
        for item in failures:
            print("  -", item)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
