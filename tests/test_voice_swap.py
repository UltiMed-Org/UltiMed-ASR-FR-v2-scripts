#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.31",
#   "jiwer>=3.0",
#   "tqdm>=4.66",
#   "loguru>=0.7",
#   "click>=8.1",
#   "soundfile>=0.12",
# ]
# ///
"""The voice swap in 01_recursive_improvement.py: another reader for a clip no draw fixed.

Plan item 4 of docs/V2_QC_PHASE_PLAN.md. A clip whose every draw fails is a text its own
voice cannot read, so the repair pass hands it to ANOTHER voice from the roster its audio
was generated with and gives it one more round, instead of shipping the known-bad
original. This drives that through the real hooks over the full stt -> tts -> stt
alternation and pins the parts that are easy to get silently wrong:

  * the swap is decided where the clip exhausts, and it goes back to the tts QUEUE
    (pending_tts) rather than being resolved;
  * the marker survives the pending_stt hop in between, so the second round actually
    draws with the new voice instead of the one that already failed;
  * one swap per clip, so a clip cannot wander the roster for ever;
  * the original audio is kept untouched until a draw wins, and `voice` is rewritten
    only then, since that column has to describe the audio that ships;
  * document stages (one voice per document, `mode: group`) never swap, and neither
    does a v1 row or a run with --no-voice-swap;
  * the voices already reading the same term are avoided, and that index is harvested
    by the needs_work hook from rows this pass does not even touch;
  * the whole chain exists outside the pass too: driver.sh can turn the swap off and
    passes the flag on the command line, and 99_hf_release/03_sync_hotfix_results.py
    carries the won voice back into the release manifests, which re-derive that column
    from the assignment hash and would otherwise relabel a swapped clip.

The harness (module loader, cfg, one-row pass runner) is imported from
tests/test_recursive_improvement_modes.py rather than copied. Run it with uv, like that
one:

    uv run tests/test_voice_swap.py

This file was written with Claude Code.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "utils"))

import test_recursive_improvement_modes as harness  # noqa: E402
import voice_assignment as va  # noqa: E402

TARGET = "bonjour le patient prend du paracetamol"
BAD = "zzzz zzzz zzzz zzzz"
GROUP = "dictionary:1234"
STEM = "000001_0000_paracetamol"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def install_fakes(mod, good_voice: str | None, calls: list[tuple[str | None, int]]):
    """TTS that records the voice it was asked for, STT that only likes `good_voice`.

    `good_voice=None` makes every draw read badly, which is how a clip exhausts. The
    bytes carry the voice so the STT side can judge a draw by its reader, which is the
    whole point here (the modes test judges by seed instead).
    """

    def fake_tts(url, text, *, response_format, seed, voice, speed, model,
                 timeout, max_retries, cfg_alpha,
                 max_new_tokens=None, dump_tag=None):
        calls.append((voice, seed))
        return f"AUD:{voice}:{seed}".encode()

    def fake_transcribe(path, *args, **kwargs):
        data = Path(path).read_bytes().decode(errors="replace")
        good = good_voice is not None and data.startswith(f"AUD:{good_voice}:")
        return (TARGET if good else BAD), 0.1

    mod.tts_synthesize = fake_tts
    mod._stt.transcribe_timed = fake_transcribe


def make_fixture(tmp: Path, *, mode: str = "row", voices=va.VOICES_FR_V2,
                 voice: str | None = "fr_female") -> tuple[Path, dict]:
    """An audio directory with the provenance stage 05 leaves beside its clips, one
    clip in it, and the manifest row 06 would read for that clip."""
    audio = tmp / "data" / "dictionary"
    audio.mkdir(parents=True)
    if mode != "none":
        va.write_provenance(audio, tmp / "01_dictionnary" / "generated_dataset.jsonl",
                            voices=voices, mode=mode,
                            field="term_index" if mode != "single" else None,
                            counts=Counter({v: 1 for v in voices}))
    (audio / f"{STEM}.flac").write_bytes(b"ORIG")
    row = {
        "audio_filepath": f"{STEM}.flac",
        "asr_training_source": TARGET,
        "asr_training_target": TARGET,
        "category": "dictionary",
        "group_id": GROUP,
        "item_index": 0,
        "index": 7,
        "term": "paracetamol",
        "transcriptions": {},
    }
    if voice is not None:
        row["voice"] = voice
    return audio, row


def make_cfg(mod, mode: str, audio_root: Path, *, voice_swap: bool = True,
             term_voices: dict | None = None):
    cfg = harness.make_cfg(mode, audio_root)
    # The shipped default (driver.sh's MIN_IMPROVEMENT), not the harness's 0.0: at 0.0 a
    # draw that merely TIES the original counts as an improvement, so a clip could never
    # exhaust and there would be nothing here to swap.
    cfg.min_improvement = 0.02
    cfg.voice_swap = voice_swap
    cfg.term_voices = {} if term_voices is None else term_voices
    # One lru_cache for the whole process, keyed by directory: a fixture in a fresh
    # temp dir is a fresh key, but clearing keeps a reused path from reading a stale
    # roster.
    mod._swap_roster.cache_clear()
    return cfg


def run(mod, cfg, row, improved_records, audio_root):
    return harness.run_pass(mod, cfg, row, improved_records,
                            audio_root / "manifest.jsonl",
                            audio_root / "manifest.improved.jsonl")


def test_full_alternation(mod):
    """stt -> tts -> stt (exhausts, swaps) -> tts -> stt (the swap wins)."""
    print("\nthe swap over a full alternation")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp)
        clip = audio / f"{STEM}.flac"
        # Two other variants of the same term are already read by these two voices, so
        # the swap must avoid them if the roster can afford it.
        term_voices = {GROUP: {"fr_female", "fr_male", "h_itf_arm"}}
        expected = va.swap_voice(STEM, "fr_female", voices=va.VOICES_FR_V2,
                                 avoid={"fr_male", "h_itf_arm"})
        calls: list[tuple[str | None, int]] = []
        install_fakes(mod, expected, calls)
        improved: dict = {}

        cfg = make_cfg(mod, "stt", audio, term_voices=term_voices)
        assert run(mod, cfg, row, improved, audio), "a fresh row needs work"
        check("round 1 flags the clip", row["improvement"]["status"] == "pending_tts",
              json.dumps(row["improvement"]))

        cfg = make_cfg(mod, "tts", audio, term_voices=term_voices)
        assert run(mod, cfg, row, improved, audio), "a pending_tts row needs the tts pass"
        check("the first round draws with the clip's own voice",
              {v for v, _ in calls} == {"fr_female"}, str(calls))
        check("the first round is waiting to be scored",
              row["improvement"]["status"] == "pending_stt")

        calls.clear()
        cfg = make_cfg(mod, "stt", audio, term_voices=term_voices)
        assert run(mod, cfg, row, improved, audio), "pending_stt needs the stt pass"
        imp = row["improvement"]
        check("no draw of its own voice wins, so the clip is swapped",
              imp["status"] == "pending_tts" and imp.get("swap_voice") == expected,
              json.dumps(imp))
        check("the swap avoids the voices already reading the term",
              imp.get("swap_voice") not in {"fr_male", "h_itf_arm", "fr_female"},
              str(imp.get("swap_voice")))
        check("a swapped clip is back in the tts queue, not resolved",
              harness.count_pending_rows(row) == (1, 0),
              str(harness.count_pending_rows(row)))
        check("the original audio is untouched until a draw wins",
              clip.read_bytes() == b"ORIG")
        check("the row still says who actually read the shipped audio",
              row["voice"] == "fr_female", str(row.get("voice")))
        check("the candidate files of the failed round are gone",
              not list(audio.glob(f"{STEM}.flac.draft*")),
              str(list(audio.glob(f"{STEM}.flac.draft*"))))

        cfg = make_cfg(mod, "tts", audio, term_voices=term_voices)
        assert run(mod, cfg, row, improved, audio), "a swapped row needs the tts pass"
        check("the second round draws with the SWAP voice",
              {v for v, _ in calls} == {expected}, str(calls))
        check("the swap marker survived the pending_stt hop",
              row["improvement"].get("swap_voice") == expected,
              json.dumps(row["improvement"]))

        cfg = make_cfg(mod, "stt", audio, term_voices=term_voices)
        assert run(mod, cfg, row, improved, audio), "the swap's drafts need scoring"
        check("the swapped draw is promoted",
              row["improvement"]["status"] == "improved", json.dumps(row["improvement"]))
        check("the promoted audio is the swap voice's",
              clip.read_bytes().decode().startswith(f"AUD:{expected}:"),
              clip.read_bytes().decode())
        check("the row now names the voice that read the shipped audio",
              row["voice"] == expected, str(row.get("voice")))
        check("the pristine original is still in .bak",
              (audio / f"{STEM}.flac.bak").read_bytes() == b"ORIG")
        rec = next(iter(improved.values()), {})
        check("the audit record carries the winning voice",
              rec.get("voice") == expected, json.dumps(rec.get("voice")))


def test_one_swap_per_clip(mod):
    """A clip that fails with its swap voice too keeps its original, as in v1."""
    print("\none swap per clip")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp)
        swap = va.swap_voice(STEM, "fr_female")
        calls: list[tuple[str | None, int]] = []
        install_fakes(mod, None, calls)   # nothing ever reads it well
        improved: dict = {}

        for mode in ("stt", "tts", "stt", "tts", "stt"):
            cfg = make_cfg(mod, mode, audio)
            run(mod, cfg, row, improved, audio)
        check("the clip was swapped exactly once",
              {v for v, _ in calls} == {"fr_female", swap}, str(sorted({v for v, _ in calls})))
        check("the second failure exhausts it",
              row["improvement"]["status"] == "exhausted", json.dumps(row["improvement"]))
        check("its exhausted marker still says which voice it was given",
              row["improvement"].get("swap_voice") == swap)
        check("the original audio ships",
              (audio / f"{STEM}.flac").read_bytes() == b"ORIG")
        check("and the row still names the voice that read it",
              row["voice"] == "fr_female")
        # count_pending must see a converged dataset, or the driver would loop for ever
        # on a clip that has nothing left to try.
        check("a twice-failed clip is converged", harness.count_pending_rows(row) == (0, 0),
              str(harness.count_pending_rows(row)))


def exhausts_without_swapping(mod, label: str, **fixture) -> None:
    """One stt -> tts -> stt alternation that must end `exhausted` with no swap."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp, **{k: v for k, v in fixture.items()
                                          if k != "voice_swap"})
        calls: list[tuple[str | None, int]] = []
        install_fakes(mod, None, calls)
        improved: dict = {}
        for mode in ("stt", "tts", "stt"):
            cfg = make_cfg(mod, mode, audio,
                           voice_swap=fixture.get("voice_swap", True))
            run(mod, cfg, row, improved, audio)
        imp = row["improvement"]
        check(label, imp["status"] == "exhausted" and "swap_voice" not in imp,
              json.dumps(imp))


def test_what_never_swaps(mod):
    print("\nwhat must never be swapped")
    # One voice per DOCUMENT: swapping one chunk would change reader mid-report.
    exhausts_without_swapping(mod, "group mode (03_PARHAF / 04_PARROT) never swaps",
                              mode="group")
    # v1 audio: nothing recorded a roster, and the row carries no voice.
    exhausts_without_swapping(mod, "a v1 row with no voice never swaps", voice=None)
    # No provenance beside the clips: the roster is unknown, so a swap would be a guess.
    exhausts_without_swapping(mod, "audio with no voice_assignment.json never swaps",
                              mode="none")
    # A single-voice run recorded as such.
    exhausts_without_swapping(mod, "a single-voice run never swaps",
                              mode="single", voices=("fr_female",))
    # The operator said no.
    exhausts_without_swapping(mod, "--no-voice-swap never swaps", voice_swap=False)
    # A one-voice roster has nothing to swap TO.
    exhausts_without_swapping(mod, "a one-voice roster has no swap to offer",
                              voices=("fr_female",))


def test_the_chain_outside_the_pass() -> None:
    """driver.sh and the release sync, the two places the swap leaves this script."""
    print("\nthe chain outside the pass")
    root = Path(__file__).resolve().parent.parent
    driver = (root / "06_hotfixes" / "driver.sh").read_text(encoding="utf-8")
    for flag in ("--voice-swap", "--no-voice-swap"):
        check(f"driver.sh can pass {flag}", flag in driver, "absent")
    check("VOICE_SWAP is hardened like AUTO_TOKEN_BUDGET, not read as a bare arithmetic "
          "name", 'case "${VOICE_SWAP:-1}" in' in driver, "no case guard")
    check("and the flag rides GUARD_ARGS onto the command line",
          'GUARD_ARGS[@]+"${GUARD_ARGS[@]}"' in driver, "absent")
    # The manifest's voice column is RE-DERIVED from the assignment hash on every rebuild
    # (99_hf_release/01_build_nemo_manifest.py), so the sync is the only thing that can
    # keep a swapped clip honest. Without it the release would name the voice that failed.
    sync = (root / "99_hf_release" / "03_sync_hotfix_results.py").read_text(encoding="utf-8")
    check("the sync reads stage 06's voice", '"voice": row.get("voice")' in sync, "absent")
    check("and writes it only when it has one, so a v1 row cannot blank a manifest voice",
          'if qc["voice"] and row.get("voice") != qc["voice"]:' in sync, "absent")
    check("while staying out of QC_FIELDS, which are written even when null",
          '"voice"' not in sync.split("QC_FIELDS = (")[1].split(")")[0], "it is in there")


def test_term_voices_are_harvested(mod):
    """The accent-spread index is built by the needs_work hook, from EVERY row."""
    print("\nthe term-voice index")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp)
        improved: dict = {}
        cfg = make_cfg(mod, "stt", audio)
        _w, _a, needs_work, _f, _p, _s, bar = mod.make_hooks(
            cfg, "default", audio / "manifest.jsonl",
            audio / "manifest.improved.jsonl", improved)
        # A sibling variant of the same term, already resolved so the pass would skip
        # it: its voice still has to land in the index, since it is exactly the voice a
        # swap of its sibling must avoid.
        sibling = dict(row, audio_filepath="other.flac", voice="h_itf_def",
                       improvement={"status": "improved"})
        other_term = dict(row, audio_filepath="x.flac", voice="h_frm_nlf",
                          group_id="dictionary:9999")
        needs_work(row)
        skipped = needs_work(sibling)
        needs_work(other_term)
        if bar is not None:
            bar.close()
        check("a resolved row is still skipped", skipped is False, str(skipped))
        check("both voices of the term are indexed",
              cfg.term_voices.get(GROUP) == {"fr_female", "h_itf_def"},
              str(cfg.term_voices))
        check("another term keeps its own entry",
              cfg.term_voices.get("dictionary:9999") == {"h_frm_nlf"},
              str(cfg.term_voices))
        check("nothing else was indexed", len(cfg.term_voices) == 2, str(cfg.term_voices))


if __name__ == "__main__":
    module = harness.load_module()
    test_full_alternation(module)
    test_one_swap_per_clip(module)
    test_what_never_swaps(module)
    test_term_voices_are_harvested(module)
    test_the_chain_outside_the_pass()
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)
