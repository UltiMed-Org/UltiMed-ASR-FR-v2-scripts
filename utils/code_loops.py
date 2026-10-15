"""Find a derailed Voxtral take in the codes it emitted, not in the audio it became.

Stage 05 accepts or rejects a take on its DURATION (`utils/frame_budget.py`), which
cannot see a take whose length is plausible but whose words are wrong. Measured on
2026-10-15, 25.4 pct of the takes the cfg_alpha ladder "rescued" were still over stage
06's CER gate, and of the severe ones half were a `loop` or an `overrun` (see
`05_generate_audio/03_qc_cpu_stt.py::classify_defect`). This module detects that half
at essentially no cost, without transcribing and without even reading the audio.

**Why the codes and not the waveform.** An acoustic detector has to infer repetition
through a mel spectrogram, where silence and drones are also self-similar and a loop is
a matter of degree; a first attempt scored 0.89-0.93 on clips whose CER was 0.000. The
codes make it exact instead. Voxtral stage 0 emits one row of `N_CODEBOOKS` int16 per
12.5 Hz frame: column 0 is the **semantic** code, chosen by a hard `argmax`
(`voxtral_tts_audio_generation.py`, `semantic_logit.argmax`), and columns 1.. are the
**acoustic** codes, produced by a flow-matching ODE that starts from fresh Gaussian
noise every frame. So only column 0 is a deterministic function of the text and the
context: a loop is literally the same integers coming back, while the acoustic columns
would differ even between two identical-sounding frames. Everything here therefore
reads column 0 and ignores the other 36.

**What the defect actually is, measured.** The obvious detector, a repeating unit of
codes, finds NOTHING: on 12 takes captured 2026-10-15, three of them audibly nothing
but mumbling, `detect_loop` fired on zero. A derailed take does not loop a syllable at
a fixed period. It **collapses its vocabulary**: the model falls onto a handful of
codes, mostly quiet ones, and drones. Per 8 s window, the share of frames carrying a
distinct code runs 0.43-0.65 on clean takes, 0.18 on a partial derailment, and 0.03 on
takes generated at the known-bad cfg_alpha 0.3, which is 8 s of audio built from three
codes. So `collapse_scan` is the detector and `detect_loop` is kept only as a cheap
secondary verdict for a defect this model has not yet been caught producing.

Two measurement choices that the data forced, both easy to get wrong:

* **Per window, not per take.** The whole-take variety fraction is confounded by
  length, since a longer text inevitably reuses codes: clean takes slide from 0.61 at
  94 frames to 0.36 at 418 frames, which drags the clean floor down onto genuinely bad
  takes sitting at 0.24. The worst 8 s window keeps every clean take above 0.43
  whatever its length, which is what leaves the gate somewhere to sit. It also catches
  a PARTIAL derailment, which is the common shape: 40 s of clean speech and then 40 s
  of rambling.
* **A fraction, not a count.** A window shorter than `DEFAULT_WINDOW_FRAMES` (every
  take has one at the end) can hold no more distinct codes than it has frames, so a
  raw count fires on a clean final second purely for being short.

**The trap, which is the same one that sank the acoustic version.** A long pause emits
the same "quiet" semantic code over and over, so a run of identical frames is NOT by
itself evidence of a derailment, and the held code really is quiet: measured at
0.13-0.27x the clip's own RMS, so a stall is the model stuck in a pause rather than
emitting a drone. That is why `detect_loop` requires `min_distinct` distinct codes
inside the repeating span, and why a `held` run is thresholded at 1.6 s, well past the
2-6 frames a clean take spends on one pause or one long vowel, rather than flagged on
sight.

How the dumps come to exist: the Voxtral fork writes
`$VOXTRAL_CODE_DUMP_DIR/<tag>.i16` for any request carrying `voxtral_dump_codes`, as
raw little-endian int16, APPENDED frame by frame so a cancelled or preempted request
still leaves usable frames. Appending is also why a tag must not be reused across two
attempts at the same clip: the second take would be concatenated onto the first.

Pure stdlib on purpose, so any stage can import it without growing a dependency, and
cheap enough to run on every take: one pass per sliding window, a few ms for the
longest take Voxtral can produce (4096 frames).

`CodeTake` keeps every measured number, not just the verdict, and stage 05 records them
in its sidecar. The thresholds here are calibrated on 24 takes and will want retuning,
so the numbers have to survive in a form that lets a later pass re-decide what counts
as bad WITHOUT regenerating any audio.
"""

from __future__ import annotations

import os
from array import array
from dataclasses import dataclass
from pathlib import Path

#: Values per frame in a dump row: 1 semantic + 36 acoustic codebooks. Fixed by the
#: checkpoint (`params.json`, `acoustic_transformer_args`), not configurable.
N_CODEBOOKS = 37

#: Frames per second of audio, from the checkpoint's
#: `multimodal.audio_model_args.audio_encoding_args.frame_rate`. The same 12.5 Hz that
#: makes `max_tokens / 12.5` the length cap in seconds.
FRAME_RATE_HZ = 12.5

#: Semantic ids reserved by `AudioSpecialTokens`, which real codes are offset past.
#: `END_AUDIO` is what the model emits when it decides the utterance is finished, so
#: its ABSENCE means the take was cut off by the token cap rather than ended.
SEMANTIC_EMPTY_AUDIO = 0
SEMANTIC_END_AUDIO = 1

#: Length of the unit whose recurrences are counted, in frames (0.64 s of audio).
#: Detection needs only ONE short unit rather than a scan over unit lengths: any
#: longer repeating unit contains this one, so it recurs at least as often, which is
#: what keeps the whole detector a single pass. Measured on a captured derailment, the
#: clean part of a take has 988 distinct 8-grams in 992 positions, i.e. natural speech
#: essentially never says the same 0.64 s twice, while the derailed part repeated one
#: 8-gram 8 times.
DEFAULT_UNIT_FRAMES = 8

#: How many times the unit must come back, counting the first occurrence.
DEFAULT_MIN_REPEATS = 3

#: A loop must cover at least this much audio. The user's description of the defect
#: was "repetitive mumbling for at least 3 s", and nothing shorter is distinguishable
#: from an ordinary repeated word.
DEFAULT_MIN_SPAN_S = 3.0

#: Largest gap between two consecutive recurrences that still counts as the same run,
#: in seconds. This is what makes three recurrences scattered over a 60 s take NOT a
#: loop, while tolerating the drift that is actually observed: a production derailment
#: repeated its unit at +12 then +13 frames, so a FIXED period finds nothing and only
#: a bound on the gap works.
DEFAULT_MAX_GAP_S = 2.0

#: Distinct semantic codes required inside the unit. See the module docstring: this is
#: what separates a repeated syllable from a held pause, whose unit is one code over
#: and over and which recurs constantly in healthy audio.
DEFAULT_MIN_DISTINCT = 3

#: Window the collapse features below are measured over, in frames (8 s), and
#: how far it slides each step (2 s). Windowing is what makes a PARTIAL
#: derailment visible: measured on 2026-10-15, a take that read 40 s of clean
#: speech and then rambled for 40 s has 158 distinct semantic codes overall,
#: squarely in the healthy range, because the clean half dilutes the collapsed
#: half. Per 8 s window its worst window drops to 18, far below any clean
#: window. The window therefore has to be short enough to sit INSIDE a
#: derailment (the observed ones run 10 s and up) and long enough that ordinary
#: speech fills it with variety.
DEFAULT_WINDOW_FRAMES = 100
DEFAULT_WINDOW_STEP = 25

#: Smallest share of a window's frames that may carry a DISTINCT code. THE
#: primary signal. Healthy speech never says the same 0.08 s twice for long, so
#: a window is mostly distinct codes; a derailed one collapses onto a handful.
#: Measured over 12 captured takes (2026-10-15), per 8 s window: clean floors
#: at 0.37 and runs to 0.56, a partial derailment hits 0.18, and three takes
#: generated at the known-bad cfg_alpha 0.3 hit 0.03, i.e. 8 s of audio built
#: from three codes. The gate sits at 0.25, inside a gap with nothing in it,
#: nearer the bad side because a false rejection costs one redraw while a false
#: accept ships a bad clip.
#:
#: A FRACTION rather than a count on purpose: a window shorter than
#: `DEFAULT_WINDOW_FRAMES` (the take's last one) can never hold more distinct
#: codes than it has frames, so a raw count would fire on a perfectly clean
#: final second purely because it is short.
DEFAULT_MIN_DISTINCT_FRAC = 0.25

#: Largest share of an 8 s window the 3 commonest codes may take. Same signal
#: seen from the other end, and it catches a window that is varied but
#: dominated by one repeated pause. Clean 0.25 to 0.39, partial derailment
#: 0.71, cfg_alpha 0.3 takes 1.00 (three codes, nothing else at all).
DEFAULT_MAX_TOP3_FRAC = 0.60

#: Longest a single semantic code may be held unbroken, in frames (1.6 s).
#: Redundant with the two gates above for a long stall, which also collapses
#: its window, but it catches a shorter one and names the defect precisely.
#: Clean takes hold a code 2 to 6 frames (one pause or one long vowel);
#: the collapsed takes hold code 10 for 54 to 58 frames, i.e. 4.6 s of a
#: single code. Note the held code is QUIET, not a drone: measured at 0.13 to
#: 0.27x the clip's own RMS, so a stall is the model stuck in a pause, which is
#: exactly why it wastes frame budget and then truncates the text.
DEFAULT_MAX_HELD_FRAMES = 20


@dataclass(frozen=True)
class LoopReport:
    """One detected repetition, the strongest in the take.

    Attributes
    ----------
    unit_frames
        Length of the repeating unit that was counted, in frames.
    period_frames, period_s
        Median gap between consecutive recurrences. Reported as a median rather than
        a fixed value because the gap drifts; see `DEFAULT_MAX_GAP_S`.
    start_frame, start_s
        Where the repeating region begins, in the generated audio.
    span_frames, span_s
        How long it goes on, from the first recurrence to the end of the last. This
        is the number to threshold on.
    repeats
        How many times the unit occurs in the run, counting the first.
    distinct
        Distinct semantic codes inside the unit; see `DEFAULT_MIN_DISTINCT`.
    unit
        The repeating codes themselves, which is what makes a flagged take
        inspectable afterwards.
    """

    unit_frames: int
    period_frames: int
    period_s: float
    start_frame: int
    start_s: float
    span_frames: int
    span_s: float
    repeats: int
    distinct: int
    unit: tuple[int, ...]


@dataclass(frozen=True)
class CodeTake:
    """What one dump says about one take.

    Every measured number is kept as a field, not folded into the verdict,
    because the thresholds are calibrated on 12 takes and will want retuning.
    Stage 05 records these in its sidecar, so a later pass can re-decide what
    counts as bad over the whole run WITHOUT regenerating any audio.

    Attributes
    ----------
    frames
        Audio frames the take actually produced (trailing rows after `END_AUDIO`
        excluded), so `frames / FRAME_RATE_HZ` is duration in seconds.
    ended
        True when the model emitted `END_AUDIO`, i.e. it decided to stop. False
        means the take was cut off by the token cap, which is the `overrun`
        defect and is always bad.
    min_distinct_frac
        Smallest share of distinct codes any window held; see `collapse_scan`.
    max_top3_frac
        Largest share any window's 3 commonest codes took.
    held_frames, held_code
        Longest unbroken run of a single code, and that code.
    loop
        The strongest repeating unit found, or None. Kept because it is the
        textbook defect and costs nothing to look for, but note it fired on
        none of the 12 captured takes: `min_distinct_frac` is what catches this
        model's derailments.
    """

    frames: int
    ended: bool
    loop: LoopReport | None
    min_distinct_frac: float = 1.0
    max_top3_frac: float = 0.0
    held_frames: int = 0
    held_code: int = 0

    @property
    def duration_s(self) -> float:
        """Seconds of audio, at the checkpoint's fixed 12.5 Hz frame rate."""
        return self.frames / FRAME_RATE_HZ

    @property
    def verdicts(self) -> tuple[str, ...]:
        """Every defect this take trips, worst first, empty when it is clean.

        Several can fire at once and they are kept separate on purpose: a
        sidecar that says "collapse+held" is a stall, while "truncated" alone
        just means the text needed more frames than the budget allowed.
        """
        # No frames means no dump, which must read as "nothing to say" rather
        # than as every defect at once. Without this, a server that is not the
        # patched fork (no dump written at all) has ended=False on every take
        # and would get its whole run rejected as truncated.
        if self.frames == 0:
            return ()
        out = []
        if not self.ended:
            out.append("truncated")
        if self.min_distinct_frac < DEFAULT_MIN_DISTINCT_FRAC:
            out.append("collapse")
        if self.max_top3_frac > DEFAULT_MAX_TOP3_FRAC:
            out.append("monotonous")
        if self.held_frames > DEFAULT_MAX_HELD_FRAMES:
            out.append("held")
        if self.loop is not None:
            out.append("loop")
        return tuple(out)

    @property
    def is_bad(self) -> bool:
        """True when the take should be redrawn rather than kept."""
        return bool(self.verdicts)


def load_dump(path: str | os.PathLike) -> list[list[int]]:
    """Read a `.i16` dump into frame rows.

    A torn tail (a request killed between two of the 74 bytes of a row) is dropped
    rather than raising, because the dump is appended live and may be read while the
    last row is still being written.

    Parameters
    ----------
    path
        The `<tag>.i16` file.

    Returns
    -------
    list of list of int
        One row of `N_CODEBOOKS` values per frame.
    """
    raw = Path(path).read_bytes()
    values = array("h")
    usable = len(raw) - (len(raw) % (2 * N_CODEBOOKS))
    if usable <= 0:
        return []
    values.frombytes(raw[:usable])
    return [values[i:i + N_CODEBOOKS].tolist()
            for i in range(0, len(values), N_CODEBOOKS)]


def semantic_track(rows: list[list[int]]) -> tuple[list[int], bool]:
    """Pull out the semantic column, truncated where the model said it was done.

    Parameters
    ----------
    rows
        From `load_dump`.

    Returns
    -------
    list of int
        Column 0 up to (not including) the first `END_AUDIO`.
    bool
        Whether that `END_AUDIO` was found. False means the take hit the token cap.

    Examples
    --------
    >>> semantic_track([[10] + [0] * 36, [4242] + [0] * 36, [1] + [0] * 36,
    ...                 [855] + [0] * 36])
    ([10, 4242], True)
    >>> semantic_track([[10] + [0] * 36])
    ([10], False)
    """
    track: list[int] = []
    for row in rows:
        if not row:
            continue
        if row[0] == SEMANTIC_END_AUDIO:
            return track, True
        track.append(row[0])
    return track, False


def collapse_scan(track: list[int], *,
                  window_frames: int = DEFAULT_WINDOW_FRAMES,
                  window_step: int = DEFAULT_WINDOW_STEP,
                  ) -> tuple[float, float, int, int]:
    """Find the worst stretch of a take by how FEW distinct codes it uses.

    This is the detector that actually fires on the observed defect. The one it
    replaces looked for a repeating unit and found nothing on any of the 12
    captured takes, including three that are audibly nothing but mumbling: a
    derailed Voxtral take does not loop a syllable at a fixed period, it
    collapses its vocabulary onto a few codes (mostly quiet ones) and drones.
    Counting distinct codes needs no period, so drift cannot hide from it.

    Returns the worst window rather than the whole-take average because a
    derailment is usually PARTIAL; see `DEFAULT_WINDOW_FRAMES`.

    Parameters
    ----------
    track
        Semantic codes, from `semantic_track`.
    window_frames, window_step
        Window length and stride, in frames. See the module constants.

    Returns
    -------
    tuple of (float, float, int, int)
        `(min_distinct_frac, max_top3_frac, held_frames, held_code)`: the
        smallest share of distinct codes any window held, the largest share any
        window's three commonest codes took, and the longest unbroken run of one
        code in the whole track with the code itself. A track shorter than half a
        window is reported as `(1.0, 0.0, run, code)`, i.e. nothing to say about
        its variety, because a 2 s clip legitimately holds few codes.

    Examples
    --------
    >>> # 8 s built from three codes: a collapse.
    >>> d, top3, held, _ = collapse_scan([10, 855, 4007] * 40)
    >>> round(d, 3), round(top3, 2)
    (0.03, 1.0)
    >>> # Ordinary speech, a fresh code almost every frame.
    >>> d, top3, held, _ = collapse_scan(list(range(200, 500)))
    >>> d, round(top3, 2), held
    (1.0, 0.06, 1)
    >>> # One code held for 4.6 s, as the collapsed takes do.
    >>> collapse_scan([10] * 58 + list(range(300, 400)))[2:]
    (58, 10)
    """
    held, held_code, i = 0, 0, 0
    while i < len(track):
        j = i
        while j + 1 < len(track) and track[j + 1] == track[i]:
            j += 1
        if j - i + 1 > held:
            held, held_code = j - i + 1, track[i]
        i = j + 1

    min_distinct, max_top3 = 1.0, 0.0
    if len(track) >= window_frames // 2:
        step = max(1, window_step)
        # The last window is allowed to be short (down to half) so a take whose
        # derailment sits in its final seconds is still measured; anything
        # shorter is dropped, since a 1 s tail of a pause is not a verdict.
        for start in range(0, max(1, len(track) - window_frames // 2 + 1), step):
            window = track[start:start + window_frames]
            if len(window) < window_frames // 2:
                break
            counts: dict[int, int] = {}
            for code in window:
                counts[code] = counts.get(code, 0) + 1
            min_distinct = min(min_distinct, len(counts) / len(window))
            top3 = sorted(counts.values(), reverse=True)[:3]
            max_top3 = max(max_top3, sum(top3) / len(window))
    return min_distinct, max_top3, held, held_code


def detect_loop(track: list[int], *, unit_frames: int = DEFAULT_UNIT_FRAMES,
                min_repeats: int = DEFAULT_MIN_REPEATS,
                min_span_s: float = DEFAULT_MIN_SPAN_S,
                max_gap_s: float = DEFAULT_MAX_GAP_S,
                min_distinct: int = DEFAULT_MIN_DISTINCT) -> LoopReport | None:
    """Find the strongest recurring unit in the semantic track.

    Counts how often each window of `unit_frames` codes occurs, keeps the unit that
    occurs most, and then asks whether its occurrences form a dense enough run to be
    a loop rather than a coincidence.

    **Why recurrence of a unit and not correlation at a fixed lag.** The obvious
    detector, matching frame `i` against frame `i - p` for a fixed period `p`, FAILS
    on the real defect: a derailment captured on 2026-10-15 repeated its unit at +12
    frames and then at +13, so no single `p` holds and a fixed-lag scan reports
    nothing on a take that is audibly looping. Counting recurrences of the unit is
    invariant to that drift, which is also exactly how the transcript-side
    `classify_defect` finds the same defect in words.

    Parameters
    ----------
    track
        Semantic codes, from `semantic_track`.
    unit_frames
        Length of the counted unit. See `DEFAULT_UNIT_FRAMES`.
    min_repeats
        Occurrences required, counting the first.
    min_span_s
        Shortest loop that counts, in seconds.
    max_gap_s
        Largest gap between consecutive occurrences that keeps them in one run, which
        is what stops three scattered recurrences from reading as a loop.
    min_distinct
        Distinct codes required inside the unit, which keeps a held pause from
        counting. See the module docstring.

    Returns
    -------
    LoopReport or None
        The longest qualifying run, or None when nothing qualifies.

    Examples
    --------
    >>> # Four distinct codes repeating for 10 s: a loop.
    >>> r = detect_loop([100, 200, 300, 400] * 32)
    >>> r.repeats >= 3, round(r.span_s, 2), r.distinct
    (True, 10.24, 4)
    >>> # The same length of silence, i.e. one code held: not a loop, because the
    >>> # unit carries a single distinct code.
    >>> detect_loop([77] * 128) is None
    True
    >>> # Ordinary speech, nothing repeating.
    >>> detect_loop(list(range(200, 400))) is None
    True
    >>> # A unit that comes back three times but spread over a long take is not a
    >>> # loop either: the gaps break the run.
    >>> filler = list(range(1000, 1400))
    >>> unit = [7, 8, 9, 10, 11, 12, 13, 14]
    >>> detect_loop(unit + filler[:200] + unit + filler[200:] + unit) is None
    True
    """
    n = len(track)
    if unit_frames < 1 or n < unit_frames * max(min_repeats, 1):
        return None
    min_span = max(2, int(round(min_span_s * FRAME_RATE_HZ)))
    max_gap = max(1, int(round(max_gap_s * FRAME_RATE_HZ)))

    # One pass: every position's unit, keeping only units that carry enough distinct
    # codes, so a silent stretch never becomes the winner and cannot mask a real loop.
    positions: dict[tuple[int, ...], list[int]] = {}
    for i in range(n - unit_frames + 1):
        unit = tuple(track[i:i + unit_frames])
        if len(set(unit)) < min_distinct:
            continue
        positions.setdefault(unit, []).append(i)

    best: LoopReport | None = None
    for unit, where in positions.items():
        if len(where) < min_repeats:
            continue
        # Occurrences are in increasing order, so one sweep finds the runs of
        # consecutive occurrences whose gaps stay inside max_gap. Overlapping
        # occurrences are KEPT rather than skipped: when the loop's period is shorter
        # than the unit (a two-frame syllable inside an eight-frame window, which is
        # the common case) every position matches and the occurrences necessarily
        # overlap, so requiring a gap of at least unit_frames would miss exactly the
        # tightest loops. Over-counting is harmless here because the span, not the
        # count, is what gets thresholded, and `min_distinct` already rules out the
        # held pause that would otherwise exploit this.
        run = [where[0]]
        runs = [run]
        for previous, current in zip(where, where[1:]):
            if current - previous <= max_gap:
                run.append(current)
            else:
                run = [current]
                runs.append(run)
        for run in runs:
            if len(run) < min_repeats:
                continue
            span = run[-1] + unit_frames - run[0]
            if span < min_span:
                continue
            gaps = sorted(b - a for a, b in zip(run, run[1:]))
            period = gaps[len(gaps) // 2]
            report = LoopReport(
                unit_frames=unit_frames, period_frames=period,
                period_s=period / FRAME_RATE_HZ,
                start_frame=run[0], start_s=run[0] / FRAME_RATE_HZ,
                span_frames=span, span_s=span / FRAME_RATE_HZ,
                repeats=len(run), distinct=len(set(unit)), unit=unit)
            if best is None or (report.span_frames, report.repeats) > (
                    best.span_frames, best.repeats):
                best = report
    return best


def inspect_dump(path: str | os.PathLike, **kwargs) -> CodeTake:
    """`load_dump` + `semantic_track` + `detect_loop`, as one call.

    Parameters
    ----------
    path
        The `<tag>.i16` file.
    **kwargs
        Passed to `detect_loop`.

    Returns
    -------
    CodeTake
        Empty and `ended=False` when the file is missing or holds no whole frame,
        which a caller should treat as "no verdict" rather than as a failure: a dump
        can be absent because the server is not the patched fork.
    """
    try:
        rows = load_dump(path)
    except OSError:
        return CodeTake(frames=0, ended=False, loop=None)
    track, ended = semantic_track(rows)
    if not track:
        return CodeTake(frames=0, ended=ended, loop=None)
    # An absent dump must read as "no verdict", so frames=0 keeps the default
    # min_distinct_frac of 1.0 (maximally varied) rather than 0.0, which
    # would make every missing dump look like a total collapse and reject the
    # whole run on a server that is not the patched fork.
    min_distinct, max_top3, held, held_code = collapse_scan(track)
    return CodeTake(frames=len(track), ended=ended,
                    loop=detect_loop(track, **kwargs),
                    min_distinct_frac=min_distinct, max_top3_frac=max_top3,
                    held_frames=held, held_code=held_code)