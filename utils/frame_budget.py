"""How long a clip may plausibly be for its text, in seconds and in 12.5 Hz frames.

One place for the duration sanity rules, because three callers need the SAME numbers:
`05_generate_audio/01_generate_audio.py` (sends the budget as `max_new_tokens` and
rejects a take that comes back outside it), `05_generate_audio/02_audit_durations.py`
(re-checks clips already on disk) and, eventually, `06_hotfixes` (whose retry path
builds its own request body today).

Why a budget at all: Voxtral's stage 0 emits one audio frame per decode step at
12.5 Hz, so `tokens / 12.5 = seconds` and the server's global cap (VOXTRAL_MAX_TOKENS
4096) is 327.68 s for every request, however short the text. A take that starts
looping syllables therefore runs for ~6 minutes of a request slot, and if it happens
to stop just under the cap the rambling audio is written and nothing notices: the
label says six words, the audio says them twenty times. Asking for only as many
frames as the text can need turns that into a cheap, detected failure.

**The band must cover the SLOWEST voice at one end and the FASTEST at the other**,
because the ten v2 voices do not read at the same speed at all. Measured per voice
over 28,749 v2 dictionary clips (2026-10-14), characters per second of audio:

    voice        median   legit p0.1   share rejected by 3+chars/8
    fr_female     16.36      11.10        0.00%
    h_itf_def     15.65      10.13        0.00%
    fr_male       15.23      10.47        0.00%
    h_frm_nlf     15.22       9.03        0.14%
    h_frf_itm     14.36       8.41        0.14%
    h_frf_esm     14.08       8.67        0.11%
    h_frm_ptf     13.65       7.52        0.27%
    h_itf_arm     13.30       8.14        0.04%
    h_frf_ptm     13.03       7.35        0.31%
    h_frm_chf     10.74       5.58        3.05%   <- the slowest, 1.5x slower than the fastest

("legit p0.1" excludes the clips reading below median/2.5, i.e. the actual runaways,
which are 1.08% of h_frm_chf and under 0.2% of every other voice.) A bound set on the
average voice is therefore a bound that rejects the slow one for being slow: at
`3 + chars/8`, h_frm_chf lost ~2% of its clips to false positives, each after three
attempts. Hence `SLOW_CHARS_PER_S` sits below the slowest voice's slow tail (5.0 vs
its 5.58 at p0.1) and `FAST_CHARS_PER_S` above the fastest clip ever observed (30.0
vs 21.3 here and 28.5 over the 602,813 v1 clips, which were single-voice).

For reference, those v1 clips read at p0.1 11.6 / p1 13.2 / median 16.6 / p99 19.5 /
max 28.5 characters per second, and none was shorter than 1.0 s.

This asymmetry of consequences is deliberate: a false positive here costs a real clip
(retried, then missing), while a false negative only survives until stage 06's
STT-based QC, which compares a transcript to the label and so catches a clip that
rambled or dropped half its text anyway. The duration check is the cheap gross-error
net, not the quality gate. What it must catch is the measured disasters: the worst
runaway at cfg_alpha 2.0 was 202 s of audio for 317 characters (1.6 chars/s, 4.7x the
old bound and 7.6x this one) and 12 of 5,335 clips ran into the server's 327.68 s
frame cap.

Written with Claude Code.
"""

from __future__ import annotations

import math

#: Head and tail room, plus the slowest plausible reading speed (characters per
#: second of audio). Together they are the upper bound: a clip longer than this for
#: its text did not stop at the end of the sentence. 5.0 is the slowest VOICE's slow
#: tail rounded down (h_frm_chf, p0.1 5.58 chars/s), not the corpus average, see the
#: table above. The fixed term dominates on short text (a 20-character dictionary term
#: gets 7 s, i.e. 2.9 chars/s) and fades on long text (a 300-character document gets
#: 63 s, i.e. 4.8 chars/s), which matches where runaways actually appear.
BUDGET_FIXED_S = 3.0
SLOW_CHARS_PER_S = 5.0

#: The fastest plausible reading speed, i.e. the lower bound: a clip shorter than
#: `chars / FAST_CHARS_PER_S` dropped part of its text. 30.0 is above the fastest clip
#: ever measured (21.3 chars/s over 28,749 v2 clips, 28.5 over 602,813 v1 clips), so
#: it flags none of them: a clip reading at 30+ chars/s has to have skipped text, since
#: even the fastest voice would need twice the duration.
FAST_CHARS_PER_S = 30.0

#: Below this, a clip is near-silence whatever the text (no v1 clip is under 1.0 s).
MIN_SECONDS = 0.5

#: Per-voice bounds: voice -> (slowest, fastest) plausible reading speed in characters
#: per second of audio. A voice that is not listed (a single-voice run, a roster the
#: table has not been measured on) falls back to the global SLOW/FAST above, which is
#: the widest band, so an unknown voice is never rejected for being unknown.
#:
#: Derivation, from the same 32,189 v2 dictionary clips (2026-10-14):
#:   slowest = 0.75 x the voice's legit p1 rate (p1 after excluding the clips below
#:             median/2.5, i.e. its actual runaways). 0.75 is where every voice's
#:             rejection share stops exceeding its genuine-runaway share: measured
#:             0.00% to 0.13% rejected per voice, of which 0.00-0.03% genuine.
#:   fastest = 1.45 x the voice's fastest observed clip. The margin is wide on purpose:
#:             each voice has only ~3.2k clips here, and the single-voice v1 corpus of
#:             602,813 clips reached 28.5 chars/s against a p99.9 of 20.5, so the true
#:             tail is wider than a 3.2k sample shows.
#:
#: Being per voice is worth much more on the slow side than on the fast one: the slow
#: tails span 7.00 to 13.22 chars/s (1.9x), while every voice's fastest clip sits in
#: 18.2-21.3 (1.2x), so the floors below are all near the global 30. The gain is on the
#: upper bound, where a runaway by fr_female is now caught at 9.9 chars/s instead of
#: 5.0, i.e. in half the GPU time.
VOICE_RATES: dict[str, tuple[float, float]] = {
    "fr_female": (9.9, 29.7),   # median 16.38, legit p1 13.22, fastest 20.49
    "fr_male": (8.9, 28.2),     # median 15.27, legit p1 11.87, fastest 19.42
    "h_itf_arm": (6.9, 30.9),   # median 13.30, legit p1  9.22, fastest 21.28
    "h_frf_esm": (8.3, 28.5),   # median 14.09, legit p1 11.07, fastest 19.66
    "h_frm_nlf": (8.5, 28.4),   # median 15.20, legit p1 11.34, fastest 19.62
    "h_itf_def": (8.6, 29.7),   # median 15.64, legit p1 11.44, fastest 20.50
    "h_frf_itm": (7.7, 28.6),   # median 14.36, legit p1 10.28, fastest 19.75
    "h_frm_ptf": (7.3, 27.2),   # median 13.67, legit p1  9.67, fastest 18.75
    "h_frf_ptm": (6.6, 27.1),   # median 13.03, legit p1  8.84, fastest 18.72
    "h_frm_chf": (5.3, 26.4),   # median 10.75, legit p1  7.00, fastest 18.22
}

#: Stage 0 emits one frame per decode step at this rate, so seconds * rate = frames.
FRAME_RATE_HZ = 12.5

#: vllm-omni refuses a per-request `max_new_tokens` above this (_TTS_MAX_NEW_TOKENS_MAX).
MAX_NEW_TOKENS = 4096

#: Floor for the requested frame count: 64 frames is 5.12 s, below which a short text
#: plus an unlucky lead-in silence could be cut legitimately.
MIN_NEW_TOKENS = 64


def rates_for_voice(voice: str | None = None, slow: float | None = None,
                    fast: float | None = None) -> tuple[float, float]:
    """The (slowest, fastest) plausible reading speed to judge one clip by.

    Parameters
    ----------
    voice
        The voice that read the clip, as named in VOICE_RATES. None or unknown falls
        back to the global band, which is the widest one.
    slow, fast
        Explicit overrides (a CLI flag the user actually passed). They win over the
        per-voice table, because a human asking for a specific rate means it; this is
        why the corresponding flags default to None rather than to a number.

    Returns
    -------
    tuple of float
        E.g. `(9.9, 29.7)` for fr_female, `(5.3, 26.4)` for h_frm_chf, `(5.0, 30.0)`
        for an unlisted voice.

    Examples
    --------
    >>> rates_for_voice("h_frm_chf")
    (5.3, 26.4)
    >>> rates_for_voice("h_frm_chf", slow=8.0)   # an explicit flag wins
    (8.0, 26.4)
    >>> rates_for_voice("a_voice_we_never_measured")
    (5.0, 30.0)
    """
    table_slow, table_fast = VOICE_RATES.get(voice or "", (SLOW_CHARS_PER_S,
                                                           FAST_CHARS_PER_S))
    return (table_slow if slow is None else slow,
            table_fast if fast is None else fast)


def budget_seconds(text: str, fixed: float = BUDGET_FIXED_S,
                   chars_per_second: float = SLOW_CHARS_PER_S) -> float:
    """Longest duration `text` can plausibly take, in seconds.

    Parameters
    ----------
    text
        The exact string sent as `input`, not the ASR target: the TTS respellings
        change the length (IgM -> I-gé-M).
    fixed, chars_per_second
        The two terms of the bound, overridable for a stage whose prosody differs.

    Returns
    -------
    float
        `fixed + len(text) / chars_per_second`, e.g. 10.5 s for a 60-character term.
    """
    return fixed + len(text) / chars_per_second


def floor_seconds(text: str, chars_per_second: float = FAST_CHARS_PER_S,
                  minimum: float = MIN_SECONDS) -> float:
    """Shortest duration `text` can plausibly take, in seconds.

    Returns
    -------
    float
        `max(minimum, len(text) / chars_per_second)`, e.g. 2.4 s for a 60-character
        term. Anything below means part of the text was not spoken.
    """
    return max(minimum, len(text) / chars_per_second)


def budget_tokens(text: str, fixed: float = BUDGET_FIXED_S,
                  chars_per_second: float = SLOW_CHARS_PER_S) -> int:
    """`budget_seconds` as a frame count for a request's `max_new_tokens`.

    Clamped to [MIN_NEW_TOKENS, MAX_NEW_TOKENS]: the server refuses more than the
    latter, and asking for less than the former would risk cutting a legitimate
    short clip.
    """
    frames = math.ceil(budget_seconds(text, fixed, chars_per_second) * FRAME_RATE_HZ)
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS, frames))


def classify_duration(text: str, duration: float | None, *,
                      budget: float | None = None,
                      fixed: float = BUDGET_FIXED_S,
                      slow_chars_per_second: float = SLOW_CHARS_PER_S,
                      fast_chars_per_second: float = FAST_CHARS_PER_S,
                      tolerance: float = 0.5) -> str:
    """Verdict on one clip: ``"ok"``, ``"over"`` or ``"short"``.

    Parameters
    ----------
    duration
        Clip length in seconds, or None when it could not be parsed (treated as ok,
        since an unreadable container is a different problem).
    budget
        Upper bound in seconds, overriding the one computed from the text. This is
        how a caller that did NOT send a per-clip budget still judges "over": its
        bound is the server's global frame cap, the same for every clip. A
        non-positive value disables the upper check entirely; the floor is always
        checked, since it needs no cooperation from the server.
    tolerance
        Slack on the upper bound only, in seconds. A take that stopped because it ran
        out of frames lands AT its budget, so without slack every budgeted clip that
        ends naturally near its bound would be rejected.

    Returns
    -------
    str
        ``"over"`` when the clip ran past its budget (a runaway), ``"short"`` when it
        is below the floor (text dropped or near silence), ``"ok"`` otherwise.
    """
    if duration is None:
        return "ok"
    cap = budget_seconds(text, fixed, slow_chars_per_second) if budget is None else budget
    if cap > 0 and duration >= cap - tolerance:
        return "over"
    if duration < floor_seconds(text, fast_chars_per_second):
        return "short"
    return "ok"
