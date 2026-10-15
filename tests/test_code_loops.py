"""Stdlib tests for utils/code_loops: catching a derailed take in its codes.

Run: `python tests/test_code_loops.py`

The module decides whether stage 05 keeps a take or redraws it, so a regression
here either ships mumbling into the dataset or burns GPU time redrawing good
clips. Four properties are worth pinning:

1. **The dump parser survives a torn tail.** Dumps are appended live, one row per
   frame, so a read can land mid-row. A dropped partial row is correct; raising
   or returning a short row is not.
2. **The thresholds still separate measured audio.** `tests/data/code_tracks.json`
   holds four REAL semantic tracks captured on 2026-10-15 (two collapsed, one
   partially derailed, two clean controls at the production cfg_alpha). Synthetic
   cases cannot pin a threshold; these can, and they are why the gate sits at
   0.25 rather than anywhere else.
3. **A partial derailment is still caught.** The 999-frame take reads 40 s of
   clean speech before it derails, so its whole-take statistics look healthy. It
   is the reason the features are measured per window and the worst window wins.
4. **Silence does not read as a defect, and a short clip has no opinion.** A held
   pause is the trap that sank the acoustic version of this detector: a long
   quiet stretch is self-similar and low-variety without being wrong. Dictionary
   clips can also be 2 s long, and a clip shorter than half a window must return
   no verdict rather than "collapse" just for being short.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import code_loops as cl  # noqa: E402

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one check, print it, and keep going so one run reports everything."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else f'  <- {detail}'}")
    if not ok:
        failures.append(label)


def write_dump(rows: list[list[int]], *, torn: int = 0) -> Path:
    """Write rows as a `.i16` dump, optionally truncating `torn` bytes off the end."""
    import array
    flat = array.array("h", [v for row in rows for v in row])
    raw = flat.tobytes()
    path = Path(tempfile.mkdtemp()) / "take.i16"
    path.write_bytes(raw[:len(raw) - torn] if torn else raw)
    return path


def frame(semantic: int) -> list[int]:
    """One dump row: the semantic code plus 36 acoustic codes we never read."""
    return [semantic] + [0] * (cl.N_CODEBOOKS - 1)


print("\n-- the dump parser")

rows = [frame(c) for c in (100, 200, 300)]
check("a whole dump reads back exactly",
      cl.load_dump(write_dump(rows)) == rows)
# A read that lands mid-row must drop that row, not return a short one: a
# partial row would put acoustic codes in the semantic column.
check("a torn final row is dropped, not returned short",
      cl.load_dump(write_dump(rows, torn=30)) == rows[:2],
      str(len(cl.load_dump(write_dump(rows, torn=30)))))
check("an empty dump reads as no frames", cl.load_dump(write_dump([])) == [])
check("a missing dump is 'no verdict', not a failure",
      cl.inspect_dump("/nonexistent/take.i16").verdicts == ()
      and cl.inspect_dump("/nonexistent/take.i16").frames == 0)

# END_AUDIO terminates the track; its absence is the overrun defect.
track, ended = cl.semantic_track([frame(10), frame(4242), frame(1), frame(855)])
check("END_AUDIO ends the track and drops what follows",
      (track, ended) == ([10, 4242], True), f"{track} {ended}")
check("no END_AUDIO reads as truncated",
      cl.semantic_track([frame(10), frame(20)]) == ([10, 20], False))

print("\n-- measured tracks: the thresholds separate real audio")

DATA = json.loads((Path(__file__).resolve().parent / "data" / "code_tracks.json").read_text())


def take(name: str) -> cl.CodeTake:
    """Rebuild a CodeTake from a stored real semantic track."""
    t = DATA[name]["track"]
    d, top3, held, held_code = cl.collapse_scan(t)
    return cl.CodeTake(frames=len(t), ended=DATA[name]["ended"],
                       loop=cl.detect_loop(t), min_distinct_frac=d,
                       max_top3_frac=top3, held_frames=held, held_code=held_code)


for name in ("clean_fr_female", "clean_h_frf_esm"):
    t = take(name)
    check(f"{name} ({DATA[name]['source_tag']}) is accepted",
          not t.is_bad,
          f"dFrac {t.min_distinct_frac:.2f} top3 {t.max_top3_frac:.2f} "
          f"held {t.held_frames} -> {t.verdicts}")

collapsed = take("collapse_cfg03")
check("the cfg_alpha 0.3 collapse is rejected", collapsed.is_bad)
check("...and is named a collapse, with the stall and the monotony",
      set(collapsed.verdicts) >= {"collapse", "monotonous", "held"},
      str(collapsed.verdicts))

partial = take("partial_derail")
check("the PARTIAL derailment is rejected (worst window, not whole take)",
      partial.is_bad, str(partial.verdicts))
# Why the features are measured per window rather than per take, stated as a
# number: the whole-take fraction is confounded by LENGTH, because a longer text
# inevitably reuses codes. Measured over 24 takes on 2026-10-15, clean takes
# slide from 0.61 (94 frames) down to 0.36 (418 frames) whole-take, which puts
# the clean floor almost on top of genuinely bad takes at 0.24 and 0.25. Per
# window, every clean take stays at 0.43 or above whatever its length, so the
# gate has somewhere to sit. This check pins the confound itself: a clean LONG
# take must score better per window than it does whole-take.
long_clean = DATA["clean_h_frf_esm"]["track"]
whole_clean = len(set(long_clean)) / len(long_clean)
check("windowing lifts a clean long take away from the gate",
      take("clean_h_frf_esm").min_distinct_frac > whole_clean,
      f"whole-take {whole_clean:.2f} vs worst window "
      f"{take('clean_h_frf_esm').min_distinct_frac:.2f}")

# The clean controls must not sit right on the gate, or voice-to-voice variation
# would start rejecting good clips. Measured margin was 0.50 against a 0.25 gate.
margin = min(take(n).min_distinct_frac for n in ("clean_fr_female", "clean_h_frf_esm"))
check("clean takes clear the gate by at least 1.5x",
      margin >= cl.DEFAULT_MIN_DISTINCT_FRAC * 1.5,
      f"worst clean window {margin:.2f} vs gate {cl.DEFAULT_MIN_DISTINCT_FRAC}")

print("\n-- silence, short clips, and the loop detector")

# A long held pause: low variety by construction, and the reason the loop
# detector requires several distinct codes inside its unit.
check("a held pause is not reported as a loop",
      cl.detect_loop([77] * 128) is None)
check("...but a 4.6 s stall is still caught as 'held'",
      "held" in cl.CodeTake(frames=58, ended=True, loop=None,
                            **dict(zip(("min_distinct_frac", "max_top3_frac",
                                        "held_frames", "held_code"),
                                       cl.collapse_scan([10] * 58)))).verdicts)
# A genuine loop of a varied unit: the textbook defect, kept for completeness
# even though none of the 12 captured takes showed it.
loop = cl.detect_loop([100, 200, 300, 400] * 32)
check("a varied unit repeating for 10 s is a loop",
      loop is not None and loop.repeats >= 3 and loop.distinct == 4)
check("ordinary speech is not a loop", cl.detect_loop(list(range(200, 400))) is None)

# A 2 s dictionary clip is shorter than half a window: no variety verdict.
short = cl.inspect_dump(write_dump([frame(c) for c in range(400, 420)]))
check("a clip shorter than half a window has no variety verdict",
      "collapse" not in short.verdicts and "monotonous" not in short.verdicts,
      str(short.verdicts))
check("...and 20 frames of ONE code is still flagged, being pure drone",
      "held" in cl.inspect_dump(write_dump([frame(10)] * 30)).verdicts)

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
