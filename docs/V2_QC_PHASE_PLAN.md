# v2 QC phase: one command, two judges, a measured funnel

What runs after `05_generate_audio/generate_v2.sh` finishes, and what has to change in the
v1 engine before it can run. Agreed with the operator on 2026-10-19. Written with Claude
Code.

The engine already exists: `06_hotfixes/driver.sh` (root, flips the containers itself) plus
`06_hotfixes/switch_server.sh` (stop one stack, start the other, block until healthy) plus
`06_hotfixes/01_recursive_improvement.py` (the two passes). This document is the delta
between what that engine did for v1 and what v2 needs, with the decisions taken rather
than the options.

## The funnel, and what each tier costs

| tier | what | where | cost |
|---|---|---|---|
| T0 | frame budget, code gate, acoustic loop scan, 5% UltiMed CPU sample | during generation, already running | free (CPU, beside the GPU) |
| T1 | full sweep, whisper `ggml-large-v3-turbo`, every clip scored on cer + tail + window + bad_run | `switch_server.sh stt`, 6 replicas on :8002 | ~31 h |
| T2 | second opinion on the T1 flags only, whisper `ggml-large-v3` (not turbo) | same endpoint, model flipped, scale 4 | ~1 h |
| T3 | repair: 5 draws per flagged clip, cfg_alpha ladder, re-judged | alternation, `voxtral-tts` | ~13 h at 12k flags |

T1's 31 h is not an estimate from a benchmark, it is v1's own throughput: `improved/full.stt.jsonl`
holds 601,338 scored clips over 3,115 h of audio, 614.3 h of summed request time, median
RTF 0.223 and p90 0.326, which at `STT_PARALLEL=20` is about 31 h of wall clock. Re-measure
it on 500 clips against the local endpoint before trusting the schedule, because v1's number
came from a remote server whose hardware is not this one.

T2 exists because the judge is not infallible and a false flag costs five TTS draws. The
one Whisper flag found in the 2,000-clip audit of 2026-10-19 was `colobome` spelled
`colobôme` and `électrorétinogramme` hyphenated, which is orthography, not a defect. No clip
is redrawn on turbo's word alone.

T0 is why T1 is a sweep and not a search: the generation-time guards already reject a
derailed take and redraw it before it ever lands on disk, so T1 is measuring what survived
them, not finding everything from scratch.

## Why UltiMed is not in the gating path

`parakeet-tdt-0.6b-v3-UltiMed` is finetuned on v1 of this very corpus, so it snaps a clip
back onto the French it expects and cannot be trusted to report a mispronunciation. The
2,000-clip audit measured what that costs: on 2,000 clips UltiMed scored clean (`cer` and
`window_cer` both under 0.15, every one carrying a number, voice `h_frf_esm`), an
independent Whisper large-v3 pass flagged exactly 1, and that one was orthographic. So
UltiMed's false-negative rate as a router is about 0.05%, which would have made it a usable
T1 at ~8 h instead of 31 h. It is still not worth it: the saving is 21 h of GPU once,
against writing and validating a batched-GPU runner and putting a model trained on the
answers into the position of judging them. UltiMed keeps its T0 job, where it is free.

## Gate values

All four gates live in `06_hotfixes/01_compute_stt.py`, next to the scores they judge,
and every stage reads them from there:

```
DEFAULT_CER_GATE        = 0.08   # whole clip
DEFAULT_TAIL_CER_GATE   = 0.12   # last ~30 s, looser because that window cuts mid-sentence
DEFAULT_WINDOW_CER_GATE = 0.15   # worst 15 s window (213 normalized chars)
DEFAULT_BAD_RUN_GATE    = 30     # longest contiguous dropped reference span, in chars
MIN_IMPROVEMENT=0.02             # a 0.09 clip is not fixed by a 0.088 draw (driver.sh)
```

The two window numbers are measured rather than chosen, on the 19,958 v1 clips of
`improved/full.stt.jsonl` scored against their labels (2026-10-16): `bad_run_chars` runs
p50 1, p99 5, p99.9 9, max 26, so a gate at 30 sits above every clean clip in 20k while
still catching a five-word drop (~30 normalized characters), and `window_cer` runs p99
0.061, max 0.211, so 0.15 costs 1 false positive in 20k. v1 shipped and its transcripts
come from whisper-1, which is worse on this vocabulary than the judges v2 uses, so both
are upper bounds on the false-positive rate.

The window gate needs no special case for short clips. `window_metrics` on a reference
shorter than one window returns the whole-clip score, and 0.15 is looser than the 0.08
whole-clip gate, so on a short clip the window gate can never fire where `cer` has not
already fired.

## The eight work items

Each is one commit. Order is dependency order. All eight items are done. Items 7 and 8 were both added on 2026-10-20, after the first six: 7 so that the phase could be launched before generation ended, 8 once the first full sweep had measured WHICH voices fail and the one-swap rule of item 4 turned out to be aimed at the wrong clips.

1. **Window gates in the repair pass.** DONE (9a639de, 231003c). `is_bad()` tested `cer`
   and `tail` only, so a 180 s document that read 170 s correctly and dropped a phrase in
   the middle (`cer` 0.04, `tail` 0.03) passed. It now tests all four scores, read from
   `01_compute_stt.py` through the existing `_stt` handle, with `WINDOW_CER_THRESHOLD` and
   `BAD_RUN_THRESHOLD` in `driver.sh` (which no longer carries a default for any of the
   four: unset means the script's own, so the numbers keep one home).
   `is_bad` was not the only call site after all: the four scores travel through ~40 of
   them, so they now move as one `Scores` NamedTuple. `needs_work` derives the two window
   scores only when the stored keys are missing AND the reference is long enough for
   either gate to be able to fire, which keeps ~1.8 h of scoring off the front of every
   pass; the bound is exact, see the comment there. Covered by
   `test_window_gates_catch_a_dropped_phrase` in `tests/test_recursive_improvement
   _modes.py` (end to end through `combined_worker` and `needs_work`, at the shipped gate
   values), with `tests/test_window_cer.py` still pinning the metric itself.

2. **`--stems-from`.** DONE (61d9aae). A pass could be restricted by `--category` but not
   to a list of clips, which is what T2 needs (re-judge these 12,000 and nothing else).
   The 2,000-clip audit needed the same thing and got it from a hacked environment
   variable. The list loader is shared with `03_qc_cpu_stt.py` (`utils/stem_requests.py`),
   and a missing list file is fatal rather than an empty allow-list, since a QC stage that
   reports a clean corpus because it could not find its input is the worst failure
   available here.

3. **The v2 generation guards on a redraw.** DONE. A redo sent `cfg_alpha` and
   `voxtral_noise_seed` but no frame budget, and checked `max_audio_seconds` only after the
   fact, so repairing a looping clip could spend 327 s of GPU producing the exact failure the
   generator would have killed at 50 s. Every draw now carries the per-clip
   `max_new_tokens` (`--auto-token-budget`, on by default) and is judged by
   `_tts.classify_duration` against that budget and against its floor, plus the code gate
   when `--code-dump-dir` is given; all three call stage 05's own functions, so
   `utils/frame_budget.py` and `utils/code_loops.py` keep one home each.
   Three things it turned out to need beyond the plan: the frame budget rides FLAT
   while the dump tag rides nested (`max_new_tokens` is read by a generic vllm-omni
   override, `voxtral_dump_codes` only by the fork), `record_code_take` gained optional
   `stage` / `seed` fields because both stages append to the same
   `code_qc_sidecar.jsonl` and attempt 1 of a redo was otherwise indistinguishable from
   attempt 1 of the original generation, and `_synthesize_draft`'s `truncated` flag
   became a `reject` / `reject_detail` pair, which also removed the two copies of the
   frame-cap warning its callers carried. Covered by `tests/test_tts_request_knobs.py`
   (wire format for both knobs, all three rejections, the sidecar row, and that
   `driver.sh` actually passes the flags).
   One consequence worth knowing: a clip whose text is longer than its own budget now
   loses every draw to the `over` guard and stays `exhausted` with its original audio.
   That is the same bound stage 05 generated the clip under, so it can only happen to
   the 0.52% tail stage 05 also rejects, and the log says so per draw and once per clip.
   `AUTO_TOKEN_BUDGET=0` restores the old "only the 327.68 s cap" behaviour.

4. **Voice swap on exhausted.** DONE. A clip that failed all 5 draws kept its original
   audio; it now gets one more round of draws read by another voice of its roster
   (`VOICE_SWAP=1`, the default). The new voice comes from
   `utils/voice_assignment.py::swap_voice`, which re-draws from the same assignment hash
   with the current voice added to the retired list for that one call, so there is no
   second algorithm to keep in step and a re-run gives the same answer. A voice already
   reading the same term is avoided when there is room (that index is harvested by the
   `needs_work` hook, which already visits every row before any worker starts, so it is
   free), and the preference is dropped rather than refusing to swap.
   Three things beyond the plan. The eligibility rule reads `mode` from the
   `voice_assignment.json` beside the audio instead of a hardcoded stage list, so it
   follows the audio rather than the directory name (`group` stages, i.e. `03_PARHAF` and
   `04_PARROT`, did not swap at all until item 8). The decision is carried as `improvement.swap_voice` on
   the row rather than as extra draws inside the TTS pass, which is what makes one
   mechanism serve `both` mode and the split halves, costs nothing on clips the first
   round fixes, and capped it at one swap per clip (item 8 made that a three-rung ladder); that also meant `set_improvement` had to
   be introduced, since the `pending_stt` hop rewrites `improvement` wholesale and would
   have silently redrawn with the voice that already failed (six literal marker dicts
   collapsed into it). And `99_hf_release/03_sync_hotfix_results.py` now carries `voice`
   back into the manifests, because `01_build_nemo_manifest.py` RE-DERIVES that column
   from the hash and a rebuild would relabel a swapped clip with the voice that never
   read it; it is written only when stage 06 has one, so a v1 row cannot blank it. The
   distinct-voices-per-term statistics `99_hf_release` reports therefore change once
   swaps have happened. If the swap also fails, the original ships `exhausted`, as in v1.
   Covered by `tests/test_voice_swap.py` (full stt -> tts -> stt alternation, what never
   swaps, the term index, and the driver + sync chain) and `tests/test_voice_assignment.py`
   sections 7 and 8.

5. **`rounds.jsonl` and a report.** DONE. Each pass appends one line per dataset to
   `rounds.jsonl` beside its output: counts by state (including `bad`, i.e. how many
   clips still fail a gate), the p50 and p90 of all four scores, audio hours, wall
   seconds, what that pass itself did, and the gates it judged by (`bad` is
   uninterpretable without them). A round is an stt pass plus the tts pass it feeds,
   numbered per dataset. `--report` renders the history with the delta against the first
   round on record; it is eager, so it needs none of the run options and no server and
   can be read from another tab mid-run, and `driver.sh` prints it per dataset when that
   dataset's loop ends.
   Two things beyond the plan. The line comes from one parse of the written output
   (`scan_output`), which now also feeds the convergence check the driver loops on, so
   the two no longer read the file separately and `both` mode gets the numbers it had no
   count for at all; the scores are read back rather than re-derived, since the pass
   already stored all four and re-deriving 602k rows costs ~1.8 h of jiwer. And the
   format, reader and renderer live in `utils/rounds_ledger.py` rather than in the pass,
   so `--report` cannot drift from what the passes append and a later stage can append to
   the same ledger. Covered by `tests/test_rounds_ledger.py` (the snapshot's states,
   distributions and filters, `count_pending` unchanged, round numbering across datasets,
   the report and the driver's call).

6. **One entry point.** DONE. `sudo -E ./driver.sh` is now the whole phase:
   `REFRESH_MANIFESTS` rebuilds the manifests, `ACOUSTIC_SCAN` runs the free CPU loop
   scan, `SYNC_RESULTS` carries the QC columns back into the release manifests after the
   last dataset converges, and all three default ON (set one to 0 to skip it). The
   rebuild is fatal on failure, the scan is not, the sync warns. `PIN_SPLITS_FROM` is
   derived from the `99_hf_release/data` symlink (its `NeMO_files` sibling), so a plain
   run pins its splits without a machine path in the repo; there is no unpinned fallback,
   and when nothing can be derived an explicit `REFRESH_MANIFESTS=1` stops the run while
   the default ON skips the refresh. v2 defaults landed too: the local judge
   (`http://127.0.0.1:8002`, with `WDOC_WHISPER_API_KEY` falling back to
   `$CRISPASR_API_KEY`), `STT_PARALLEL` 10 -> 20 (six serialising replicas need a queue),
   `TTS_PARALLEL` 2 -> 32 (the measurement below; 32 is the ceiling, the largest captured
   CUDA graph).

   Two things it turned out to need. `STT_MODEL` was **documented but never read**: the
   header's own example passed it (`sudo -E MODE=stt STT_MODEL=large-v3 ...`) while
   `run_pass` sent `--stt-model "${WDOC_WHISPER_MODEL:-}"`, so the variable the docs told
   you to use did nothing and the judge was whatever `.env` happened to hold (here
   `whisper-1`, which is why every v1 reading sits under that key). Setting it per run,
   which is what the two-judge plan needs, silently had no effect. It is now one variable
   used in both places (the name sent to the server and the key the reading lands under),
   defaulting to `whisper.cpp/ggml-large-v3-turbo`, with `WDOC_WHISPER_MODEL` kept as its
   older spelling. Note the key is also the resume key: changing it makes readings stored
   under the old one invisible, which on a 602k-clip corpus means re-transcribing. For v2
   that is moot (new audio, nothing to reuse), but it is the reason to decide the judge's
   name once and leave it alone. A second trap on the same variable surfaced when T2 was
   actually launched on 2026-10-25: `.env` is sourced with `set -a` BEFORE the
   `${VAR:-default}` lines, so its `STT_MODEL=...-turbo` pin beat the launch line and the
   run announced the turbo judge while being told to use large-v3. The driver now hands
   the launch line its value back after sourcing (the WDOC_WHISPER_* names and the
   credentials stay .env-wins, since for those an inherited value is pollution rather
   than intent), and says which knob the launch line took back. And `03_sync_hotfix_results.py` gained `--stt-key`, which the driver
   passes as `$STT_MODEL`: that script's `STT_KEY` default is v1's `whisper-1` alias,
   which against a v2 row matches nothing and would have written every QC column null, so
   it now also warns and lists the keys it did find.

   Side effect worth noting: there were five near-identical `case` blocks normalising 0/1
   env flags, and these three steps would have made eight. They are now one `flag` helper
   (nameref, so `flag RESCORE 0 RESCORE_ON` keeps the operator's own value readable in
   messages), and the runuser boilerplate is one `as_user` helper. That also fixed the
   per-dataset report call, which was `uv script.py` rather than `uv run script.py` and so
   printed a usage error swallowed by its own `|| true`. Covered by three new cases in
   `tests/test_driver_alternate_mode.py` (nothing extra runs by default, the unpinned
   refusal, and the full order: rebuild, combine, scan, pass, sync with the right
   `--stt-key`).

## 7. The gate, so the phase starts itself (DONE)

`driver.sh` is launched while stage 05 is still generating and starts on its own when the generator exits, so the GPU goes from the last clip of generation into the QC phase with no idle hours and no decision to make in the middle of the night. Two knobs, both patterns for `pgrep -f`, both disabled by an empty value:

* `WAIT_FOR_GENERATOR`, default `generate_v2\.sh|01_generate_audio\.py`: polled every `WAIT_POLL` (60 s), requiring `WAIT_SETTLE` (120 s) of continued absence so a stage boundary or a resumed run does not read as the end of generation, and bounded by `WAIT_TIMEOUT` (0 = forever), which stops the run rather than starting it.
* `KILL_QC`, default `03_qc_cpu_stt\.py|04_qc_acoustic_loops\.py`: SIGTERM, which `03_qc_cpu_stt.py` traps to finish the clip in flight and print its per-voice CER table, then SIGKILL after `KILL_QC_GRACE` (60 s). Its sidecar is append-only and resumes by stem, so this costs at most the clips in flight.

The signal is the generator's PROCESS on purpose. Not the container, since the alternation's own tts passes need it up, so its state says nothing about whether generation is over; and not a lock file, since a crashed run would never release one. What makes the wait necessary rather than merely tidy: the first `switch_server.sh stt` runs `docker compose stop voxtral-tts`, so starting early takes the TTS server out from under a multi-day run, and the manifest refresh would describe a half-written corpus.

Ordering, all of it load-bearing: config errors are reported BEFORE the wait (so the gate only blocks a run that is otherwise ready), and the QC is stopped BEFORE the acoustic scan (they want the same cores, and the stage 05 QC is redundant once this script reads every clip with a stronger judge). Six new cases in `tests/test_driver_alternate_mode.py`, which now stubs `pgrep`/`pkill` so every other case also proves the gate waves through an idle host.

## 8. The voice ladder, because the swap was aimed at the wrong clips (DONE)

Item 4 gave an exhausted clip ONE redraw at any other voice of its roster, and refused to swap at all in `group` mode so that a document would keep one reader start to end. The first full QC sweep measured both decisions wrong, in the same week they were written.

What the sweep says, at 143,093 clips scored (2026-10-20, whisper `ggml-large-v3-turbo`, the four gates of this plan):

| stage | scored | over a gate |
|---|---|---|
| `01_dictionnary` | 134,801 | 0.99 pct |
| `02_drugs` | 5,563 | 1.15 pct |
| `07_acronyms` | 404 | 1.73 pct |
| `03_PARHAF` | 2,325 | **22.62 pct** |

479 of PARHAF's 526 failures are shaped `loop` (the model starts the sentence, then repeats syllables until the token ceiling). Two causes compound. A length cliff that every stage shares, measured on the dictionary alone so the stages cannot confound it: 0.20 pct under 20 s, 0.82 pct at 20-30 s, 8.78 pct at 30-40 s, 55.89 pct at 40-60 s. And a voice effect that survives controlling for duration, measured inside the 40-60 s band only, where the roster splits in two with nothing in between: `fr_female` 0.0 pct (n=34), `h_itf_arm` 1.4 (69), `h_itf_def` 2.2 (46), `fr_male` 5.8 (52), then `h_frf_esm` 41.5 (94), `h_frm_ptf` 44.2 (104), `h_frf_itm` 48.3 (89), `h_frm_nlf` 60.2 (88), `h_frf_ptm` 77.4 (115).

So the old rule spent its one swap on a coin flip (five of nine roster voices fail 40 to 77 pct of long clips), and it spent it nowhere near the problem: PARHAF holds roughly 10,000 of the ~15,400 clips the sweep predicts will need repair, and PARHAF was the half that never swapped. Redrawing a 50 s clip at `h_frf_ptm` is not a repair attempt.

The ladder, `utils/voice_assignment.py::VOICE_LADDER_FR_V2` plus `ladder_voice`, walked one rung per exhausted round by `06_hotfixes/01_recursive_improvement.py::swap_voice_for`:

1. `--n-improv` draws at the clip's own voice (unchanged, 5 by default);
2. then that many at `h_itf_arm` or `h_itf_def`;
3. then that many at `fr_male` or `fr_female`;
4. then the clip exhausts and ships its original, as in v1.

Only the four measured survivors are on it, hybrids before bases: the hybrids are why the roster has nine voices rather than two, so the repair keeps the corpus accent spread as long as it can and falls back to the two shipped voices only once `h_itf_*` has failed too. Which of a rung's two voices a clip gets is the same assignment hash as everywhere else, so it is deterministic and resume stable, and `cfg.term_voices` is still honoured as a preference inside the rung.

`group` mode now swaps, with the rung keyed on the GROUP id rather than the clip stem, so every repaired chunk of one document lands on the SAME new reader: a document gains at most one reader per rung instead of one per repaired chunk. That is the honest trade, stated plainly: a PARHAF document can now change reader mid-way, which item 4 refused. Against that, audio that does not say its text is worse label noise for an ASR corpus than a document changing reader once, each clip is an independent training sample anyway, and the alternative measured at 22.62 pct of the stage. `VOICE_SWAP=0` still turns the whole thing off.

The marker carries `improvement.voice_tier` beside `improvement.swap_voice`, both in `CARRIED_FIELDS` so they survive the `pending_stt` hop (that list replaced the single `swap_voice` special case). The ledger gains `work.swapped_last`, the swaps that went to the LAST rung, which is the number that says whether the ladder is long enough. Cost: nothing for clips an earlier rung fixes, and at most two extra stt+tts rounds for the clips that keep failing, which is why the rungs are ordered best-measured-first.

`tests/test_voice_swap.py` walks the whole ladder rung by rung (the draws per rung, in order, the rung counter climbing once per exhausted round, exhaustion only off the end) and pins the document rule from both sides: two chunks of one document get the same new reader, a row-mode clip still keys on itself.

## TTS_PARALLEL is the cheapest win available

v1 redrew at `TTS_PARALLEL=2`. Stage 05 measured on 2026-10-14, same server and same clips:
16 in flight is 3,860 clips/h, 32 in flight is 4,610 to 4,960 clips/h. At 5 draws per clip a
12,000-clip queue is 60,000 synthesis calls, so **13 h at 32 in flight against roughly 10
days at 2**. Raise it and measure; 32 is the largest captured CUDA graph, so do not go above.

## Flipping the ASR model between T1 and T2

The CrispASR compose pins the model inside the ASR service's extra args:

```
- CRISPASR_EXTRA_ARGS=--verbose -fa --hf-repo ggerganov/whisper.cpp:ggml-large-v3-turbo.bin
```

Parameterize that filename so `switch_server.sh` can export it instead of the operator
editing the compose between tiers, and document it end to end (compose reads it, it reaches
the container inside EXTRA_ARGS, it is written down). Non-turbo large-v3 is roughly 3.1 GB
against turbo's 1.6 GB, so T2 runs at `CRISPASR_STT_SCALE=4`, not 6, on a 24 GB card that
also drives a display.

## How it gets launched

```bash
# from 06_hotfixes/, inside the zellij run tab. Launch it whenever: it waits for stage
# 05 to finish, stops the CPU QC itself, and runs the whole phase from there.
sudo -E ./driver.sh                      # T1, then T3, alternating to convergence
./01_recursive_improvement.py --report   # any time, from another tab

# T2, the second opinion: non-turbo over T1's flags only, no state touched
# (N_IMPROV=0 means transcribe-only, so no clip can be re-flagged and no provenance
# marker can be stripped; RETRY_* only get the reopened rows past needs_work's early
# return; SYNC_RESULTS=0 keeps the second judge's CERs out of the released manifests).
sudo -E CRISPASR_ASR_GGML=ggml-large-v3.bin CRISPASR_STT_SCALE=4 MODE=stt N_IMPROV=0 \
  STT_MODEL=whisper.cpp/ggml-large-v3 STEMS_FROM="$PWD/flags.txt" \
  RETRY_IMPROVED=1 RETRY_EXHAUSTED=1 REFRESH_MANIFESTS=0 ACOUSTIC_SCAN=0 \
  SYNC_RESULTS=0 ./driver.sh 2>&1 | tee t2_judge.log
uv run 10_compare_judges.py              # then read the second opinion out
```

`flags.txt` is generated, not tracked: it is the stems of every clip T1 flagged, i.e.
`Path(row["audio_filepath"]).stem` for each row of `improved/full.stt.jsonl` and
`improved_parrot/full.stt.jsonl` carrying an `improvement` marker (12,678 clips on
2026-10-25). `10_compare_judges.py` then writes `second_opinion_flags.txt`, which is the
input to a repair round: `sudo -E MODE=alternate STEMS_FROM=second_opinion_flags.txt
./driver.sh`.

Check the `driver.sh: judge ... at ...` line before walking away: it names the model AND
the key the readings land under, and filing T2 under T1's key would overwrite 602k
readings with no way back.

## Measured: the flag rate is not uniform across voices

The full Parakeet sweep prints a per-voice CER table every 10k clips, and at 124,495
dictionary rows (2026-10-26 01:37, nine voices each holding ~10k of that slice) two
hybrids own most of the flags:

| voice | n | CER median | CER p90 | over gate | rate |
| --- | --- | --- | --- | --- | --- |
| `fr_female` | 10,274 | 0.0090 | 0.0253 | 16 | 0.16% |
| `fr_male` | 10,058 | 0.0100 | 0.0274 | 16 | 0.16% |
| `h_frm_ptf` | 9,790 | 0.0137 | 0.0346 | 39 | 0.40% |
| `h_frf_esm` | 10,092 | 0.0108 | 0.0300 | 44 | 0.44% |
| `h_frf_ptm` | 9,728 | 0.0131 | 0.0341 | 50 | 0.51% |
| `h_frm_nlf` | 9,927 | 0.0114 | 0.0312 | 64 | 0.64% |
| `h_itf_arm` | 10,114 | 0.0165 | 0.0415 | 103 | 1.02% |
| `h_frf_itm` | 9,869 | 0.0147 | 0.0444 | 293 | 2.97% |
| `h_itf_def` | 10,148 | 0.0170 | 0.0537 | 374 | 3.69% |

So `h_itf_def` and `h_frf_itm` carry 667 of the 999 flags between them, at 19x and 23x
the two shipped voices' rate, while their medians sit only 0.006-0.008 above. A median
that close with a p90 twice as high is the shape of a voice that is usually fine and
occasionally derails, not one that is uniformly mushy, which is what the redraw ladder is
for. Both have the same `it` (Italian) ingredient in the hybrid, so the suspicion is that
ingredient rather than either recipe.

Two consequences for the repair round, neither applied yet:

* The ladder now floors hybrids at 1.5 instead of 1.2 (`utils/voice_assignment.py`), so a
  redraw of these clips cannot land below the band their audition passed at.
* A clip still bad after the ladder gets a different voice, and these two are the ones the
  swap should move AWAY from, not toward. Worth checking the swap's target choice before
  the T3 round, since a naive round-robin would feed `h_itf_def` as often as `fr_female`.

## The tail gate was blind on 5,250 clips, because the state rows' durations were stale

Found on 2026-10-26 with Claude Code, while checking whether the two `../acronyms/000400_000{1,2}_sa.flac` rows the verification pass flagged were really stale. They were, and so were 208,233 others: every sweep rewrites a row's `transcriptions` but none of them rewrites the row's own `duration`, which only a redraw does (`01_recursive_improvement.py`). A clip regenerated by the v2 nine-voice pass and then re-read by the whisper and parakeet sweeps therefore ends up with current transcripts beside a v1-era length. 208,235 of 604,337 rows disagree with the manifest, 208,153 of those also carry no `voice`, and the gaps are not rounding (`015709_0000_gynecologie_00047_t1_c0` stored 34.0 s against an 86.96 s file, `000000_0000_parrot_50_c0` stored 63.28 s against 12.00 s).

The readings themselves are fine, which is what makes this cheap. On the mismatching rows the stored transcript runs at 14.69 characters per second of the MANIFEST duration against 16.67 per second of the stored one, and the rows that agree run at 14.64, so the transcripts describe the audio that is on disk. Their turbo CER distribution matches too (median 0.0109 against 0.0118). Nothing needs re-transcribing.

What the stale number cost is one gate. `_stt.tail_cer` returns None at or under `TAIL_SECONDS` (30 s), so the tail CER, the only gate that asks "did this clip END the way it was supposed to", silently skipped **5,250 clips that really run past 30 s** and scored a tail on 229 that do not. That is precisely the truncation and derail-at-the-end shape the whole-clip average dilutes on a long PARHAF chunk. Closing it needs **no GPU**: `tail_cer` is a function of the stored reference, the stored transcript and the duration, all three already on disk.

`06_hotfixes/13_refresh_stale_durations.py` does it (dry run by default, `--apply` to write, `tests/test_refresh_stale_durations.py` pins it). It refreshes `duration` and `voice` from the manifest, recomputes `cer_tail` and `stt_rtf` per reading, deliberately leaves the length-independent metrics (`cer`, `cer_window`, `bad_run_chars`, `repeat_excess`) alone because they already have an owner, and reports how many readings gained a tail score at or over the 0.12 gate.

Its place in the export chain, which matters in both directions: **after** the manifests are rebuilt from the audio (`REFRESH_MANIFESTS=1`, since it reads durations from them) and **before** `99_hf_release/03_sync_hotfix_results.py --apply`, which copies `cer_tail` into the release columns. Any clip it newly flags has to go through `03_collect_suspicious.py` to join the T3 repair round.

That "before" turned out to be stronger than a preference, which is why the step is now wired into `driver.sh` (commit `3f71869`) rather than left to whoever runs the export. `03_sync_hotfix_results.py:341` copies the state row's `duration` INTO the manifest, on the premise its docstring spelled out: that stage 06's output "already carries the re-probed duration". It does not, for exactly the reason above. So the driver's own post-step sync, running last, was overwriting the rebuild's true durations with the 208,235 stale ones, which is how a blind spot opened by the generator kept being re-sealed by the release step meant to fix it. The refresh now runs immediately before the sync under the same step header (two scripts, one step, like the manifest rebuild), so the copy is a no-op; `tests/test_driver_alternate_mode.py::test_the_duration_refresh_runs_before_the_sync` pins the adjacency rather than the membership, because a refresh *after* the sync repairs the state file for a release that already shipped wrong numbers.

One-time consequence for the T2 pass of 2026-10-26: it was already running when the driver was edited, and an edit replaces the file by rename, so the live `bash` kept the old inode and ran the old tail, i.e. the sync **without** the refresh. The export chain for this release therefore has to rebuild the manifests from the audio first, to undo that, before the refresh and a second sync. The canonical order, which is what the chain script runs:

1. `99_hf_release/01_build_nemo_manifest.py --all --pin-splits-from <NeMO_files>`, durations back from the audio
2. `99_hf_release/02_combine_nemo_manifests.py --pin-splits-from <NeMO_files>`, splits pinned so the old test set cannot leak into train
3. `06_hotfixes/13_refresh_stale_durations.py` (dry run, read the `tail_added_OVER_GATE` count), then `--apply`
4. `99_hf_release/03_sync_hotfix_results.py --apply --stt-key whisper.cpp/ggml-large-v3-turbo --stt-model whisper.cpp/ggml-large-v3-turbo`, turbo because it is the only judge that read all 602,789 clips (T2's own sync named the non-turbo key, which is right for T2 and wrong for the release's primary column)
5. `06_hotfixes/04_flag_asr_defects.py improved/full.stt.jsonl --extra-hyps improved_parrot/full.stt.jsonl <full.jsonl>`, rebuilding `asr_flagged_clips.jsonl`. After the sync, because the rules read the manifest's `stt_transcript`; before the drop, because the drop is what reads the list
6. `99_hf_release/04_drop_bad_rows.py --apply`
7. `99_hf_release/05_normalize_text.py --apply`
8. `99_hf_release/08_validate_manifests.py`, the release gate
9. the training repo's `python perso/build_ultimed_val_slices.py`, last because the nine per-voice slices are sampled from the manifests step 6 just shortened

## The ASR flag list is keyed by clip NAME, so v2 inherited v1's verdicts

Step 5 above was added on 2026-10-26, after noticing that `asr_flagged_clips.jsonl`, which `04_drop_bad_rows.py` drops on, holds 63 clip keys measured on **v1** audio. v2 regenerated the corpus under v1's file names, so 60 of those 63 clips now have audio newer than the list and 3 have no audio at all: the drop would have removed 60 good clips for defects that no longer exist. `04_drop_bad_rows.py::asr_flagged_stale` was already warning about precisely this, loudly and non-fatally, and the warning had simply never been acted on.

The rebuild costs no GPU pass. The detector needs a recognizer that is not Whisper, and the parakeet-v3-ultra sweep already read the whole corpus; its readings sit in the state files under `transcriptions["onnx-asr/parakeet-tdt-0.6b-v3-ultra"]`, and no `.flac` under the release root is newer than the last redraw round (2026-10-25 20:45), so every one of those readings describes the audio that ships. `load_hyps` therefore accepts a state file as well as a `transcribe_manifests.py` hypotheses file.

Two measurements came out of wiring it up, both recorded in full in that script's docstring:

* **`preamble` earns its keep, `skipped` does not.** Over the first 19,729 clips: 7 preamble flags, all 7 real on inspection (four are babble before the sentence, three are the opening phrase spoken twice), and every one passed the existing gates, whisper CER 0.0095 to 0.057 against 0.08 and `bad_run_chars` 1 to 7 against 30. An *insertion* at the start is the one defect shape nothing else measures. Meanwhile `skipped` produced 17 flags whose spans are dosage and unit enumerations, where both recognizers write digits for spelled numbers and `_merged_diffs` chains a dozen one-word diffs into one long "missing" span; a clip that truly lost ten words cannot score CER 0.0049 with `bad_run_chars` 1, which is what `006151_0000_cardiologie_00001_t0_c16` scores. So `--reason` now defaults to `preamble` alone, and `skipped` would have dropped ~520 long dosage-dense PARHAF clips, the long-form data this dataset exists for.
* **The judge has to be parakeet.** The same rule with the turbo reading as hypothesis flags 9 clips, of which only 3 overlap parakeet's 7 and **none of the 6 turbo-only flags is real**: they are whisper's LM pulling a filler forward ("Il y a compte rendu de l'hospitalisation" for a label starting "Compte rendu") or re-spelling the first word. That is the same prior that makes whisper blind to the real babble, and it also rules out making this a gate in `is_bad`, since the alternating loop judges with whisper.

These ~210 clips are dropped rather than redrawn, against the usual preference for repair, and the reason is structural: the pass queues a clip from `is_bad` and promotes a draw only when it beats the stored score by `--min-improvement`, so a babbling clip is never queued (it scores clean) and a flawless new draw would not be promoted either. Repair would mean carrying the metric through scoring, queueing and promotion, and no post-redraw check could confirm the fix without a second parakeet sweep. The audio stays on disk.

## Open TODOs

* Re-measure STT throughput on 500 clips against the local endpoint before trusting the
  31 h.
* `99_hf_release/README.md` still carries four `TODO (v2)` cells, including the distinct
  voices per term numbers that work item 4 changes.
* TODO: run `06_hotfixes/13_refresh_stale_durations.py --apply` in the export chain (see the
  section above), then act on whatever the newly scored tails flag: 5,250 clips have never had
  their ending checked, so the count of real end-of-clip defects in the corpus is still unknown.
* TODO: decide whether the voice swap should avoid `h_itf_def` and `h_frf_itm` as swap
  TARGETS, given the table above. Needs the full sweep's final per-voice numbers, not the
  dictionary-only slice.
