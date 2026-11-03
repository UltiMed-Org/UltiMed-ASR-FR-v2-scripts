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
  * the ladder is walked one rung per exhausted round (own voice, then h_itf_*, then
    fr_*), and a clip out of rungs exhausts rather than wandering the roster for ever;
  * the original audio is kept untouched until a draw wins, and `voice` is rewritten
    only then, since that column has to describe the audio that ships;
  * document stages (`mode: group`) DO swap, but every repaired chunk of one document
    lands on the same new reader, so the document gains one reader per rung rather than
    one per chunk; a v1 row and a run with --no-voice-swap still never swap;
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
                 voice: str | None = "fr_female", stem: str = STEM,
                 group_id: str = GROUP) -> tuple[Path, dict]:
    """An audio directory with the provenance stage 05 leaves beside its clips, one
    clip in it, and the manifest row 06 would read for that clip."""
    audio = tmp / "data" / "dictionary"
    audio.mkdir(parents=True, exist_ok=True)
    if mode != "none" and not (audio / va.PROVENANCE_FILENAME).exists():
        va.write_provenance(audio, tmp / "01_dictionnary" / "generated_dataset.jsonl",
                            voices=voices, mode=mode,
                            field="term_index" if mode != "single" else None,
                            counts=Counter({v: 1 for v in voices}))
    (audio / f"{stem}.flac").write_bytes(b"ORIG")
    row = {
        "audio_filepath": f"{stem}.flac",
        "asr_training_source": TARGET,
        "asr_training_target": TARGET,
        "category": "dictionary",
        "group_id": group_id,
        "item_index": 0,
        "index": 7,
        "term": "paracetamol",
        "transcriptions": {},
    }
    if voice is not None:
        row["voice"] = voice
    return audio, row


def make_cfg(mod, mode: str, audio_root: Path, *, voice_swap: bool = True,
             term_voices: dict | None = None, retry_improved: bool = False):
    cfg = harness.make_cfg(mode, audio_root, retry_improved=retry_improved)
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
        expected = va.ladder_voice(STEM, 0, roster=va.VOICES_FR_V2,
                                   tried=("fr_female",), avoid=term_voices[GROUP])
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


def test_the_whole_ladder(mod):
    """Every rung, in order, and then exhaustion: a clip no voice can read.

    This is the shape the operator asked for on 2026-10-20: --n-improv draws at the
    clip's own voice, then that many at h_itf_arm or h_itf_def, then that many at
    fr_male or fr_female, then keep the original. Each rung costs one more
    stt -> tts -> stt alternation, so the whole ladder is seven passes.
    """
    print("\nthe whole ladder, rung by rung")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp)
        rung1 = va.ladder_voice(STEM, 0, roster=va.VOICES_FR_V2, tried=("fr_female",))
        rung2 = va.ladder_voice(STEM, 1, roster=va.VOICES_FR_V2,
                                tried=("fr_female", rung1))
        calls: list[tuple[str | None, int]] = []
        install_fakes(mod, None, calls)   # nothing ever reads it well
        improved: dict = {}

        seen: list[tuple[str, int, str]] = []
        for mode in ("stt", "tts", "stt", "tts", "stt", "tts", "stt"):
            cfg = make_cfg(mod, mode, audio)
            run(mod, cfg, row, improved, audio)
            imp = row["improvement"]
            seen.append((mode, int(imp.get("voice_tier") or 0), imp["status"]))

        # h_itf_def, not h_itf_arm, since 2026-11-03: h_itf_arm is retired, and
        # ladder_voice skips retired names, leaving h_itf_def alone on rung 1.
        check("the rungs are the measured voices, in order",
              (rung1, rung2) == ("h_itf_def", "fr_male"), f"{rung1} then {rung2}")
        # Three draws per rung (the harness cfg's --n-improv), and the ORDER matters:
        # a rung must be finished before the next is started, or the ladder would be
        # three voices racing rather than an escalation.
        tried = [v for v, _ in calls]
        check("each rung gets a full round of draws, own voice first",
              tried == ["fr_female"] * 3 + [rung1] * 3 + [rung2] * 3, str(tried))
        check("the rung counter climbs once per exhausted round",
              [t for _, t, _ in seen] == [0, 0, 1, 1, 2, 2, 2], str(seen))
        check("the clip exhausts only once the ladder is used up",
              [st for _, _, st in seen] == ["pending_tts", "pending_stt", "pending_tts",
                                            "pending_stt", "pending_tts", "pending_stt",
                                            "exhausted"], str(seen))
        check("its exhausted marker still says which voice it was last given",
              row["improvement"].get("swap_voice") == rung2,
              json.dumps(row["improvement"]))
        check("the original audio ships",
              (audio / f"{STEM}.flac").read_bytes() == b"ORIG")
        check("and the row still names the voice that read it",
              row["voice"] == "fr_female")
        # count_pending must see a converged dataset, or the driver would loop for ever
        # on a clip that has nothing left to try.
        check("a clip off the end of the ladder is converged",
              harness.count_pending_rows(row) == (0, 0),
              str(harness.count_pending_rows(row)))


def test_a_win_keeps_the_ladder_rung(mod):
    """A clean win must keep the rungs the clip has already burned.

    `_apply_improvement` used to write the winning marker as a literal dict, which
    dropped CARRIED_FIELDS. Nothing reads `voice_tier` while the clip stays improved,
    which is what kept it quiet, but a win is not the end of the clip's story:
    --retry-improved and a later round's re-score both REOPEN one. A reopened winner
    with no `voice_tier` restarts the ladder at rung 0, so `swap_voice_for` hands it a
    voice from the rung it has already spent five draws failing at, and the SWAP log
    line misreports the rung. 1,320 wins were reopened that way on 2026-10-24.
    """
    print("\na win keeps the rungs it burned")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio, row = make_fixture(tmp)
        rung1 = va.ladder_voice(STEM, 0, roster=va.VOICES_FR_V2, tried=("fr_female",))
        calls: list[tuple[str | None, int]] = []
        install_fakes(mod, rung1, calls)   # only the rung-1 voice reads it well
        improved: dict = {}

        # stt (flag) -> tts (own voice) -> stt (exhausts it, swaps) -> tts -> stt (wins).
        for mode in ("stt", "tts", "stt", "tts", "stt"):
            cfg = make_cfg(mod, mode, audio)
            run(mod, cfg, row, improved, audio)
        imp = row["improvement"]
        check("the swapped draw is promoted", imp["status"] == "improved",
              json.dumps(imp))
        check("and the win records the rung it was won at",
              imp.get("voice_tier") == 1 and imp.get("swap_voice") == rung1,
              json.dumps(imp))

        # Now reopen it the way the 2026-10-24 migration reopened 1,320 wins: --rescore
        # (a scoring-rule change made every stored number stale) plus --retry-improved
        # (a resolved row is settled otherwise). Standing in for "promoted under looser
        # rules", the stored reading is staled to one that no longer clears the gates,
        # which is the state an older run left on disk. Nothing reads well any more
        # either, so the reopened clip has to escalate rather than win again.
        entry = row["transcriptions"]["default"]
        entry["text"] = BAD
        for field in ("cer", "cer_tail", "cer_window", "bad_run_chars"):
            entry.pop(field, None)
        install_fakes(mod, None, calls)
        cfg = make_cfg(mod, "stt", audio, retry_improved=True)
        cfg.rescore = True
        check("--rescore --retry-improved reopens the win",
              run(mod, cfg, row, improved, audio) is True)
        check("the reopened row is back in the tts queue at the rung it reached",
              row["improvement"]["status"] == "pending_tts"
              and row["improvement"].get("voice_tier") == 1,
              json.dumps(row["improvement"]))

        calls.clear()
        for mode in ("tts", "stt"):
            cfg = make_cfg(mod, mode, audio)
            run(mod, cfg, row, improved, audio)
        imp = row["improvement"]
        check("the redraw stays on the voice that won, not the one that failed",
              {v for v, _ in calls} == {rung1}, str(calls))
        check("the ladder resumes at the next rung instead of restarting",
              imp.get("voice_tier") == 2, json.dumps(imp))
        check("so the clip gets a rung-2 voice, not the rung-1 one it never tried",
              imp.get("swap_voice") in va.VOICE_LADDER_FR_V2[1]
              and imp.get("swap_voice") not in va.VOICE_LADDER_FR_V2[0],
              str(imp.get("swap_voice")))


def test_a_document_shares_its_new_reader(mod):
    """`group` mode swaps now, but one new reader for the whole document.

    It used to refuse outright, to keep one reader per document. The first full QC
    sweep then measured 22.6 pct of 03_PARHAF clips over the gates against ~1 pct
    elsewhere, so refusing meant shipping the defect. Keying the rung on the GROUP id
    instead of the clip stem is what keeps the cost to one extra reader per rung.
    """
    print("\na document keeps one new reader")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        doc = "PARHAF:ANATOMOPATHOLOGIE-00003"
        audio, chunk0 = make_fixture(tmp, mode="group", stem="parhaf_0000",
                                     group_id=doc)
        _, chunk7 = make_fixture(tmp, mode="group", stem="parhaf_0007", group_id=doc)
        other = make_fixture(tmp, mode="group", stem="parhaf_0000",
                             group_id="PARHAF:CARDIOLOGIE-00001")[1]
        cfg = make_cfg(mod, "stt", audio)
        picks = [mod.swap_voice_for(r, audio / f"{r['audio_filepath']}", cfg)
                 for r in (chunk0, chunk7, other)]
        check("a group-mode chunk does get a new voice", picks[0] is not None,
              str(picks[0]))
        check("both chunks of one document get the SAME new reader",
              picks[0] == picks[1], str(picks[:2]))
        check("and it is a rung of the ladder, not any roster voice",
              picks[0][0] in va.VOICE_LADDER_FR_V2[0], str(picks[0]))
        check("another document draws independently",
              picks[2] is not None and picks[2][0] in va.VOICE_LADDER_FR_V2[0],
              str(picks[2]))
        # The whole point of keying on the group: a row-mode clip of the same document
        # would scatter, which is what the old refusal was protecting against.
        row_mode = make_fixture(tmp / "row", stem="parhaf_0007", group_id=doc)
        cfg_row = make_cfg(mod, "stt", row_mode[0])
        scattered = mod.swap_voice_for(row_mode[1],
                                       row_mode[0] / "parhaf_0007.flac", cfg_row)
        check("row mode still keys on the clip, so chunk 7 can differ there",
              scattered is not None, str(scattered))


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
    # `(( VOICE_SWAP ))` on a value like `true` reads `true` as an unset NAME, evaluates
    # to 0, and turns the swap OFF when the operator just turned it on. Every 0/1 flag of
    # the driver goes through the `flag` helper for that reason, which also refuses a
    # value it does not recognise instead of guessing.
    check("VOICE_SWAP is normalised, not read as a bare arithmetic name",
          "flag VOICE_SWAP 1" in driver, "not routed through flag()")
    check("and the helper refuses anything that is not a 0/1 spelling",
          'must be 0 or 1' in driver and "1|true|TRUE|yes|YES|y|Y) _dst=1" in driver,
          "the helper accepts junk")
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
    test_the_whole_ladder(module)
    test_a_win_keeps_the_ladder_rung(module)
    test_a_document_shares_its_new_reader(module)
    test_what_never_swaps(module)
    test_term_voices_are_harvested(module)
    test_the_chain_outside_the_pass()
    print(f"\n{len(failures)} failed: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)
