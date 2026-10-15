#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "soundfile>=0.12",
# ]
# ///
"""Stage 05 must be safe to stop and cheap to resume, on a run that lasts days.

Three properties, all of them things a 602,885-clip run depends on and none of them
visible in the output when they break:

1. **Ctrl-C stops in seconds.** Every row is submitted to the thread pool up front, so
   a `finally: pool.shutdown(wait=True)` without `cancel_futures` keeps the pool
   working through the whole remaining queue while the terminal looks hung. Measured on
   a 2000-task stand-in before the fix: one interrupt still executed 2000/2000 tasks.
   On the real dictionary stage that is days of "stopping", and the operator's natural
   reaction (a second Ctrl-C, or kill -9) is what then risks a half-written clip.
2. **An interrupted run reports 130**, not 0 and not 1, because `generate_v2.sh` has to
   tell "the user stopped this" (stop the whole run) from "a few clips failed"
   (continue to the next stage; a re-run retries exactly those).
3. **A resume generates exactly what is missing**, matched by stem, and re-runs the
   clips the interrupted pass never reached.

The TTS server is stubbed: `synthesize` is replaced by a slow writer, so this is about
the scheduling and bookkeeping around it, not about audio.

Run: `uv run tests/test_generation_resume.py`
"""
from __future__ import annotations

import importlib.util
import json
import signal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

STAGE05 = Path(__file__).resolve().parent.parent / "05_generate_audio" / "01_generate_audio.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(out_dir: Path, *, concurrency: int = 8) -> SimpleNamespace:
    """Namespace with the knobs run_batch reads (the roster is off: voices=None)."""
    return SimpleNamespace(
        output=str(out_dir), format="flac", timeout=30.0, concurrency=concurrency,
        overwrite=False, voice=None, voice_mode="single", voices_file=None,
        model=None, instruction=None, seed=None, speed=None, cfg_alpha=None,
        temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        do_sample=None, max_audio_seconds=327.0,
    )


def _corpus(path: Path, n: int) -> None:
    """n rows shaped like 07_acronyms: one term, n variants, no source_id."""
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"term_index": 1, "variant_index": i, "term": "VIH",
                                "asr_training_source": f"phrase {i}"}) + "\n")


def _slow_writer(delay: float):
    """Stand-in for synthesize(): writes the clip through the same .tmp rename."""
    # `cfg_alpha` is keyword-only here on purpose: synthesize_with_retries passes the
    # ladder rung it drew, and a stand-in that cannot accept it would pass this test
    # while the real call signature had drifted.
    def synthesize(base_url, text, out, args, voice=None, cfg_alpha=None, **kwargs):
        time.sleep(delay)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(b"audio")
        tmp.replace(out)
        return 0.01, 2.0
    return synthesize


def test_interrupt_stops_promptly_and_resume_finishes() -> None:
    mod = load_module(STAGE05, "stage05_resume")
    n, delay = 400, 0.05
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = tmp / "generated_dataset.jsonl"
        out_dir = tmp / "audio"
        out_dir.mkdir()
        _corpus(corpus, n)
        mod.synthesize = _slow_writer(delay)

        # Fire one SIGINT (what Ctrl-C delivers) a moment into the run. The whole queue
        # would take n*delay/concurrency = 2.5 s to drain, so a run that stops "late"
        # and one that stops promptly are separated by seconds, not milliseconds.
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        signal.setitimer(signal.ITIMER_REAL, 0.30)
        t0 = time.monotonic()
        try:
            rc = mod.run_batch("http://stub", corpus, _args(out_dir))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - t0

        written = sorted(p.stem for p in out_dir.glob("*.flac"))
        assert rc == 130, f"an interrupted run must exit 130, got {rc}"
        assert elapsed < 1.5, (
            f"stopping took {elapsed:.2f}s: the queued clips were drained instead of "
            "cancelled (this is the regression the fix is about)")
        assert 0 < len(written) < n, f"expected a partial run, got {len(written)}/{n}"
        # Nothing half-written: a .tmp left behind has stem "X.flac" and so cannot be
        # mistaken for the finished "X" by the resume scan.
        for leftover in out_dir.glob("*.tmp"):
            assert leftover.stem.endswith(".flac"), leftover.name
        print(f"OK interrupt: exit 130 after {elapsed:.2f}s, {len(written)}/{n} clips written")

        # Resume: no interrupt this time. It must generate only what is missing and end
        # with the full set, which is what makes stopping the run cheap.
        before = len(written)
        rc = mod.run_batch("http://stub", corpus, _args(out_dir))
        after = sorted(p.stem for p in out_dir.glob("*.flac"))
        assert rc == 0, f"the resume must succeed, got {rc}"
        assert len(after) == n, f"resume left {len(after)}/{n} clips"
        assert set(written) <= set(after), "a resume must not lose an earlier clip"
        print(f"OK resume: {before} skipped as existing, {n - before} generated, {n}/{n} total")


def test_a_failed_clip_does_not_look_like_an_interrupt() -> None:
    """Exit 1 (some clips failed) must stay distinct from 130 (the user stopped it).

    generate_v2.sh continues to the next stage on 1 and stops on 130, so collapsing the
    two would either abandon four stages because of one transient 500, or keep
    generating after a Ctrl-C.
    """
    mod = load_module(STAGE05, "stage05_resume_fail")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = tmp / "generated_dataset.jsonl"
        out_dir = tmp / "audio"
        out_dir.mkdir()
        _corpus(corpus, 6)

        def flaky(base_url, text, out, args, voice=None, cfg_alpha=None, **kwargs):
            if text.endswith("3"):
                raise RuntimeError("server returned 500")
            tmp_f = out.with_suffix(out.suffix + ".tmp")
            tmp_f.write_bytes(b"audio")
            tmp_f.replace(out)
            return 0.01, 2.0

        mod.synthesize = flaky
        rc = mod.run_batch("http://stub", corpus, _args(out_dir, concurrency=2))
        assert rc == 1, f"a failed clip must exit 1, got {rc}"
        assert len(list(out_dir.glob("*.flac"))) == 5, "the other five must be written"
        # And the failed one is simply missing, so the next run retries just it.
        mod.synthesize = _slow_writer(0.0)
        rc = mod.run_batch("http://stub", corpus, _args(out_dir, concurrency=2))
        assert rc == 0 and len(list(out_dir.glob("*.flac"))) == 6
        print("OK a failed clip exits 1, stays missing, and is retried by the next run")


def test_a_timeout_is_one_failed_clip_not_a_dead_run() -> None:
    """A read timeout must be counted and stepped over, not allowed to escape.

    Regression, observed on the real v2 run on 2026-10-14: TimeoutError is not a
    URLError, so it propagated out of the progress loop, which killed the bar and
    every log line while `finally: shutdown(wait=True)` kept the pool generating.
    The run looked dead and was in fact 22k clips further along six hours later.
    One slow clip must cost one clip.
    """
    mod = load_module(STAGE05, "stage05_timeout")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = tmp / "generated_dataset.jsonl"
        out_dir = tmp / "audio"
        out_dir.mkdir()
        _corpus(corpus, 6)

        def times_out(base_url, text, out, args, voice=None, cfg_alpha=None, **kwargs):
            if text.endswith("1"):          # an early row, so the rest follows it
                raise TimeoutError("the read operation timed out")
            tmp_f = out.with_suffix(out.suffix + ".tmp")
            tmp_f.write_bytes(b"audio")
            tmp_f.replace(out)
            return 0.01, 2.0

        mod.synthesize = times_out
        rc = mod.run_batch("http://stub", corpus, _args(out_dir, concurrency=2))
        assert rc == 1, f"a timed-out clip must exit 1 like any failure, got {rc}"
        assert len(list(out_dir.glob("*.flac"))) == 5, (
            "the five other clips must still be generated: the loop has to survive "
            "the timeout instead of propagating out of it")
        mod.synthesize = _slow_writer(0.0)
        rc = mod.run_batch("http://stub", corpus, _args(out_dir, concurrency=2))
        assert rc == 0 and len(list(out_dir.glob("*.flac"))) == 6
        print("OK a timeout costs one clip and the resume retries just it")


def test_an_unexpected_exception_stops_the_run_instead_of_silencing_it() -> None:
    """Whatever the exception, the queued work is dropped, not executed in silence.

    Same root cause as the test above, but for the exceptions nobody enumerated: the
    loop now has a catch-all that sets `aborted`, which is what makes the shutdown
    cancel the queued futures. With 400 rows and a worker raising on row 2, a run
    that kept going would leave hundreds of clips on disk; the fix leaves a handful.
    """
    mod = load_module(STAGE05, "stage05_unexpected")
    n = 400
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = tmp / "generated_dataset.jsonl"
        out_dir = tmp / "audio"
        out_dir.mkdir()
        _corpus(corpus, n)

        def boom(base_url, text, out, args, voice=None, cfg_alpha=None, **kwargs):
            if text.endswith("2"):
                raise ValueError("something nobody enumerated")
            time.sleep(0.02)
            tmp_f = out.with_suffix(out.suffix + ".tmp")
            tmp_f.write_bytes(b"audio")
            tmp_f.replace(out)
            return 0.01, 2.0

        mod.synthesize = boom
        rc = mod.run_batch("http://stub", corpus, _args(out_dir, concurrency=2))
        written = len(list(out_dir.glob("*.flac")))
        assert rc == 1, f"an aborted run must exit non-zero, got {rc}"
        assert written < n // 4, (
            f"{written}/{n} clips written: the queue was drained after the exception "
            "instead of being cancelled (the regression this test exists for)")
        print(f"OK an unexpected exception aborts after {written}/{n} clips, queue cancelled")


if __name__ == "__main__":
    test_interrupt_stops_promptly_and_resume_finishes()
    test_a_failed_clip_does_not_look_like_an_interrupt()
    test_a_timeout_is_one_failed_clip_not_a_dead_run()
    test_an_unexpected_exception_stops_the_run_instead_of_silencing_it()
    print("all OK")
