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

```
CER_THRESHOLD=0.08           # whole clip, v1's second sweep
TAIL_CER_THRESHOLD=0.12      # last ~30 s
WINDOW_CER_THRESHOLD=0.15    # worst 15 s window (213 normalized chars)
BAD_RUN_CHARS_THRESHOLD=50   # longest contiguous dropped reference span. TODO: set from the T1 distribution
MIN_IMPROVEMENT=0.02         # a 0.09 clip is not fixed by a 0.088 draw
```

0.15 on the window is one dropped phrase: five words is roughly 30 normalized characters,
which is 0.14 of a 213-character window. It is the same value the 2,000-clip audit used.

The window gate needs no special case for short clips. `window_metrics` on a reference
shorter than one window returns the whole-clip score, and 0.15 is looser than the 0.08
whole-clip gate, so on a short clip the window gate can never fire where `cer` has not
already fired.

## The six work items

Each is one commit. Order is dependency order.

1. **Window gates in the repair pass.** `01_recursive_improvement.py:340` `is_bad()` tests
   `cer` and `tail` only, so a 180 s document that reads 170 s correctly and loops 10 s in
   the middle (`cer` 0.04, `tail` 0.03) passes today. Add `window_cer` and `bad_run_chars`,
   imported from `01_compute_stt.py` through the existing `_stt` module handle so nothing is
   duplicated, plus `WINDOW_CER_THRESHOLD` and `BAD_RUN_CHARS_THRESHOLD` in `driver.sh`.
   `is_bad` is the only call site. Covered by a new test next to `tests/test_window_cer.py`,
   which already pins the metric itself.

2. **`--stems-file`.** A pass can be restricted by `--category` but not to a list of clips,
   which is what T2 needs (re-judge these 12,000 and nothing else). The 2,000-clip audit
   needed the same thing and got it from a hacked environment variable.

3. **The v2 generation guards on a redraw.** A redo sends `cfg_alpha` and
   `voxtral_noise_seed` but no frame budget, and checks `max_audio_seconds` only after the
   fact, so repairing a looping clip can spend 327 s of GPU producing the exact failure the
   generator would have killed at 50 s. Reuse `utils/frame_budget.py` (same function stage
   05 calls, not a second copy) and honour `CODE_DUMP_DIR` for the code gate.

4. **Voice swap on exhausted.** A clip that fails all 5 draws keeps its original audio
   today. Instead, redraw it with another of the nine voices, which is legitimate only for
   the three term stages (`01_dictionnary`, `02_drugs`, `07_acronyms`), where
   `utils/voice_assignment.py` assigns a voice per clip. `03_PARHAF` and `04_PARROT` assign
   one voice per document and a swap would break the single-reader property, so they are
   excluded. Prefer a voice not already reading that term, so the accent spread is kept.
   Note this changes the distinct-voices-per-term statistics that `99_hf_release` reports.
   If the swap also fails, the original is kept and listed in the report, as in v1.

5. **`rounds.jsonl` and a report.** Nothing writes a per-round history, so "how much did it
   improve" is answered by diffing two Markdown reports by eye. Each pass appends one JSON
   line per dataset and round: counts by state, median and p90 of each of the four scores,
   audio hours scored, wall seconds. A `--report` flag prints the table with the delta
   against round 0 and is readable from another tab mid-run.

6. **One entry point.** `driver.sh` gains two optional pre-steps before the dataset loop,
   each a single call to a script that already exists: refresh the v2 manifests (stage 99)
   and run the free CPU acoustic loop scan. It also needs its v2 defaults: the local judge
   (`http://127.0.0.1:8002` with the bearer key, since the v1 remote endpoint returns 502),
   `STT_MODEL` so T1 and T2 land in different `transcriptions` keys on the same row, and
   `TTS_PARALLEL` raised off 2.

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
# from 06_hotfixes/, inside the zellij run tab, once generation has finished:
sudo -E ./driver.sh                      # T1, then T3, alternating to convergence
sudo -E MODE=stt STT_MODEL=large-v3 STEMS_FILE=flags.txt ./driver.sh   # T2
./01_recursive_improvement.py --report   # any time, from another tab
```

## Open TODOs

* `BAD_RUN_CHARS_THRESHOLD=50` is a starting value. Measure the distribution on T1 output
  and set it before T3 spends GPU on it.
* Re-measure STT throughput on 500 clips against the local endpoint before trusting the
  31 h.
* `99_hf_release/README.md` still carries four `TODO (v2)` cells, including the distinct
  voices per term numbers that work item 4 changes.
