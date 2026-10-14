# v2 regeneration runbook

Written with Claude Code on 2026-10-14, for the run that re-synthesizes the whole UltiMed corpus with the ten French Voxtral voices.

v2 keeps v1's texts, file names, group keys and splits-by-intent, and replaces every `.flac`. That one sentence is the source of every trap below: a QC value, a duration or a drop list keyed on a **file name** still "matches" after the audio under that name has been replaced, which makes stale v1 judgements look like fresh v2 ones. The checks in this document exist because none of them are visible in the output: the manifest just carries a plausible number.

Scope: this is the audio run only. The texts were finalized on 2026-10-13 by a separate session (`HANDOFF_v2_texts.md`); nothing here touches them.


## 0. Preflight (done 2026-10-13, re-check cheaply)

Inputs, all five `generated_dataset.jsonl`, verified row by row:

| input | rows | group field | groups | clips per voice (spread) |
|---|---:|---|---:|---|
| `01_dictionnary/generated_dataset.jsonl` | 534,384 | `term_index` | 62,448 | 52,564..54,045 (2.8%) |
| `03_PARHAF/generated_dataset.jsonl` | 43,455 | `source_id` | 4,254 | 4,045..4,850 (18.5%) |
| `02_drugs/generated_dataset.jsonl` | 21,901 | `term_index` | 3,252 | 2,051..2,383 (15.2%) |
| `07_acronyms/generated_dataset.jsonl` | 1,596 | `term_index` | 532 | 123..195 (45.1%) |
| `04_PARROT/generated_dataset.jsonl` | 1,549 | `source_id` | 475 | 116..194 (50.4%) |
| **total** | **602,885** | | **70,961** | ~60k per voice |

- Every row carries `term_index`, `variant_index`, `term` and a non-empty `asr_training_source` (stage 05 hard-requires them at `05_generate_audio/01_generate_audio.py:374`), and no two rows of a stage produce the same file name stem.
- All ten roster voices are used by all five stages. The roster is `utils/voice_assignment.py::VOICES_FR_V2`, byte-identical to `CrispASR/voxtral/cloning/voices_fr_production.txt`, and each name resolves to an embedding file in the checkpoint's `voice_embedding/`.
- The five `generated_audios` symlinks point at **empty** folders under `/media/.../audio_datasets/v2/`, and `99_hf_release/data` points at that same v2 root, so `data/NeMO_files` does not exist yet (scripts defaulting to it will say "no manifest found" until step 3).

The server is the only thing that has to be brought up by hand, and it is not part of this repo:

```bash
# in the CrispASR repo
sudo docker compose up -d --build voxtral-tts
./voxtral/test_seed_determinism.sh doctor     # per-request knobs really reach the model
./tts.py --check                              # health + one real synthesis
```

Nothing listens on `:8003` until that is done, and stage 05 aborts on an unreachable server rather than writing partial work.

**How long the run takes.** Measured from v1's own `05_generate_audio/tts_generate.log` (623,944 per-clip lines), in audio-hours produced per wall-clock hour: the PARHAF stretch of 2026-07-29/30 held **15.6-15.9x** realtime for 20 hours straight (~1,750-2,100 clips/h at 28-32 s per clip), and the dictionary run of 2026-07-14/15 peaked at **18.3-20.5x** (~3,700-4,100 clips/h at 17.9 s per clip), median 14.9x. v2 is the same 3,115 hours of audio, so at a sustained 15x it is **~208 wall hours, i.e. 8.7 days of uninterrupted generation**; at the peak rate ~156 h. All of v1 was generated on the 3090 Ti power-capped to 100-200 W during the heatwave, so uncapped should land at the fast end, and the v2 run does: **28-30x realtime, 4,610-4,960 clips/h** measured over the dictionary stage on 2026-10-14 at the default 32 requests in flight (uncapped, 310-325 W of 350 W), which is **~110 wall hours, 4.6 days** for the 3,115 hours (23.1x and 5.6 days at 16 in flight, see step 1). Read the rate off the `progress:` lines stage 05 logs every 500 clips (cumulative clips/h, realtime factor and an absolute ETA); the tqdm bar's own rate is the same cumulative figure, not an instantaneous one, because clip durations span 12.5-24.1 s and an EMA over them cannot be read. Stage 06's QC is serial wall time on top (the GPU cannot host the TTS and the STT server at once) and is **unmeasured**: time it on acronyms first, 1,596 clips and 4.1 h of audio, then multiply by 760.


## 1. Generate (stage 05)

One run per stage, each resumable (existing files are skipped, so a crash costs nothing):

The two loops below are also `05_generate_audio/generate_v2.sh`, which is what to run
in practice (`./generate_v2.sh`, or `./generate_v2.sh 07_acronyms` for one stage; it
checks the server is up first and logs to `tts_generate_v2.log` so v1's throughput log
stays intact). Spelled out here because the flags are the decisions:

```bash
cd 05_generate_audio

# Term stages: one voice PER CLIP, so a term is heard in several accents.
for s in 01_dictionnary 02_drugs 07_acronyms; do
  uv run 01_generate_audio.py \
    --input  "../$s/generated_dataset.jsonl" \
    --output "../$s/generated_audios" \
    --voice-mode row \
    --cfg-alpha 2.0 \
    --concurrency 32
done

# Document stages: one voice PER DOCUMENT, so a report keeps one reader.
for s in 03_PARHAF 04_PARROT; do
  uv run 01_generate_audio.py \
    --input  "../$s/generated_dataset.jsonl" \
    --output "../$s/generated_audios" \
    --voice-mode group \
    --cfg-alpha 2.0 \
    --concurrency 32
done
```

- **The voice mode is per stage, decided 2026-10-14**, and it has to be right before launching because the voice is baked into the audio. Both modes hash (never count), so edit and resume stability is the same either way: adding or removing a row never moves another clip to another voice.
- `cfg_alpha` **2.0** is the top of the band measured on 2026-10-13 (16 voices x 10 values, 160 takes): usable 1.2-2.0, and 1.5-2.0 for the hybrid voices. It is deliberately the top, not the middle: below 1.2 the flow-matching decoder stops following the text and degrades fast, while above 2.0 distortion grows and takes increasingly fail to terminate at all. Stage 06's retry ladder therefore walks **down** from it (step -0.2, floor 1.2) rather than up.
- `--concurrency 32`, which is what `generate_v2.sh` now defaults to. The older claim that "10 is what v1 ran" was true only of v1's first day (`docs/BUILD_LOG.md` line 55): v1's own log shows it at 15 in flight from 2026-07-07 20:00 onwards and there for 8 days. The mechanism is the server's CUDA-graph capture sizes, `[1, 2, 4, 8, 16, 32]` in `cuda_graph_acoustic_transformer_wrapper.py`, with the batch padded **up** to the next one: 10 in flight runs the 16-row graph with 6 empty rows, so those 6 slots are paid for and unused. Measured, same server, same clips: **2,070 clips/h at 10 in flight, 3,042 at 15 with an unchanged 17.4 s per clip** (v1's log), and **3,860 clips/h = 23.1x realtime at 16** (v2, 2026-10-14). and **4,610-4,960 clips/h = 28-30x realtime at 32**, the top bucket, which is the default: +22% over 16 for ~70% more latency per clip (14.9 s -> 25.1 s, i.e. 48 -> 89 ms per audio frame), which costs a batch run nothing but does mean a clip running to the frame cap takes ~365 s, hence the 600 s `--timeout`. Not more than 32, since a batch bigger than the largest captured graph loses the graph. CrispASR's own notes say the acoustic stage is `codec-bs=1` and saturates around 4-8 concurrent clients; 10 did not hurt, so there is no reason to lower it.
- **Runaway takes, and how to bound them (measured 2026-10-14).** At cfg_alpha 2.0 a share of takes do not terminate: the model reads the sentence then loops syllables until it runs out of frames. On the dictionary stage, **1.68% of clips came back longer than 41.2 s**, which was v1's maximum on the same texts at cfg_alpha 1.3 (where the rate was 4 clips in 1,068,676), and 12 of 5,335 ran to the server's 327.68 s cap. Joined to their own input text, **0.52% exceeded `3 s + chars/8`**, the worst being 202 s of audio for a 317-character term. Two costs: ~6 minutes of a request slot per capped clip, and (worse) a take that stops just under the cap is *written*, so rambling audio enters the dataset under a short label and only stage 06's STT check can catch it.

  Three things are true about stopping them. (1) **Cancellation works**: `/v1/audio/speech` is wrapped in vLLM's `@with_cancellation`, which polls `raw_request.is_disconnected()`, and `vllm_omni/entrypoints/async_omni.py::generate` catches `CancelledError`/`GeneratorExit` and calls `_abort_internal_requests`, so closing the socket (what the client's `--timeout` does) really frees the GPU. Measured against the live run: 32 requests dropped after 3 s left no mark on its throughput, while the same 32 left to finish cut it from ~4,800 clips/h to 600-1,800 clips/h for ~90 s. (2) But a **timeout is the wrong detector**, because it is time-based and cannot tell a long clip from a loop without knowing the text. (3) The right one is the per-clip frame budget, **on by default** since 2026-10-14 (`--no-auto-token-budget`, or `TOKEN_BUDGET=0 ./generate_v2.sh`, to disable), which asks for `(3 s + chars/5) x 12.5` frames and rejects a clip that comes back pinned at its own budget. `max_new_tokens` is honored per request (25 frames returns exactly 2.00 s of audio in 2.4 s of wall), so the loop dies in seconds.

  **The bound is set by the slowest VOICE, not by the corpus average**, and that distinction is worth 2% of a voice's clips. Measured over the 28,749 v2 dictionary clips on disk (`05_generate_audio/02_audit_durations.py` prints this table), the ten voices read at very different speeds, in characters per second of audio:

  | voice | median | legit p1 | fastest clip | **its bounds** | rejected by `3 + chars/8` |
  |---|---:|---:|---:|---:|---:|
  | fr_female | 16.38 | 13.22 | 20.49 | **9.9 - 29.7** | 0.00% |
  | h_itf_def | 15.64 | 11.44 | 20.50 | **8.6 - 29.7** | 0.00% |
  | fr_male | 15.27 | 11.87 | 19.42 | **8.9 - 28.2** | 0.00% |
  | h_frm_nlf | 15.20 | 11.34 | 19.62 | **8.5 - 28.4** | 0.14% |
  | h_frf_itm | 14.36 | 10.28 | 19.75 | **7.7 - 28.6** | 0.14% |
  | h_frf_esm | 14.09 | 11.07 | 19.66 | **8.3 - 28.5** | 0.11% |
  | h_frm_ptf | 13.67 | 9.67 | 18.75 | **7.3 - 27.2** | 0.27% |
  | h_itf_arm | 13.30 | 9.22 | 21.28 | **6.9 - 30.9** | 0.04% |
  | h_frf_ptm | 13.03 | 8.84 | 18.72 | **6.6 - 27.1** | 0.31% |
  | **h_frm_chf** | **10.75** | **7.00** | 18.22 | **5.3 - 26.4** | **3.05%** |

  ("legit p1" excludes the clips reading below that voice's median/2.5, i.e. its actual runaways, 1.08% for h_frm_chf and under 0.2% everywhere else.) So the first bound was rejecting the slow voice *for being slow*, three attempts at a time, while barely bounding the fast ones at all.

  **The bounds are therefore per voice** (`VOICE_RATES` in `utils/frame_budget.py`, resolved by `rates_for_voice`): upper = `0.75 x` that voice's legit p1, the knee where each voice's rejection share stops exceeding its genuine-runaway share; lower = `1.45 x` its fastest observed clip, a wide margin because each voice has only ~3.2k clips here while the 602,813-clip v1 corpus reached 28.5 chars/s against a p99.9 of 20.5. A voice the table does not list falls back to the global `5.0 / 30.0`, the widest band, so an unlisted voice is never rejected for being unlisted; `--budget-chars-per-second` / `--floor-chars-per-second` override the table for every voice.

  What that buys: a runaway by fr_female is now caught at 9.9 chars/s instead of 5.0, i.e. in half the GPU time (37 s instead of 71 s for a 339-character text). Re-audited over 32,605 clips, the rejections spread to 0.00-0.13% per voice instead of 3.05% on one, and 12 clips the global band had let through turned out to read at 5.2-5.9 chars/s against their own voices' medians of 13-14. The floor is nearly voice-independent (every voice's fastest clip is 18.2-21.3 chars/s), so those bounds all land near the old global 30.

  (4) **A clip can also be too SHORT**, and that failure is just as silent: the backend emits a stop token early and answers 200 with audio that sounds complete on its own. `--floor-chars-per-second` (default 30, i.e. `chars/30` seconds) rejects it, with the budget on or off. It flags none of the 602,813 v1 clips nor any of the 29,141 v2 ones, so it only ever fires on a gross failure; stage 06's STT QC remains the real quality gate.

  (5) **Rejections are retried in place**, `--retries` (default 2, so three attempts), and each retry goes **one rung down the cfg_alpha ladder**: 2.0, 1.8, 1.6, clamped at `--cfg-alpha-floor` 1.2 (`--cfg-alpha-step 0` pins them all at `--cfg-alpha`, which was the behaviour before 2026-10-14). Two things are being exploited. The only randomness in Voxtral's pipeline is the fresh Gaussian each frame's flow-matching ODE starts from, so re-sending the same text is already a genuinely new take, and a rambling one is usually a sampling accident: 13 rejected attempts in a live run became 10 clips saved by a retry. But a text can derail **reproducibly**, and a new sample of the same draw never saves it: `000863_0005_acineux` (339 chars, h_frm_chf) ran over budget on all three attempts of two separate runs. 2.0 is the **top** of the measured band (1.2-2.0), so a clip that rambled there is over-guided and the room is below it.

  Each attempt logs `attempt N/3 rejected, retrying at cfg_alpha 1.8: over budget: ...` (or `implausibly short: ...`), a clip that exhausts its attempts logs `giving up after 3 attempts`, and the run ends with `duration rejections: N over budget, M implausibly short, K saved by a retry` plus, when any clip descended, `cfg_alpha ladder: N clip(s) accepted below 2.0 (3 at 1.8, 1 at 1.6), recorded in .../cfg_alpha_sidecar.jsonl`, both beside the `done:` line.

  (6) **A clip drawn below the run's cfg_alpha is recorded per clip**, in `cfg_alpha_sidecar.jsonl` **beside the audio** (`utils/cfg_sidecar.py`, shared with stage 06). This is what unblocked the ladder: the dataset publishes one `cfg_alpha` per row, so a clip quietly drawn at 1.6 would ship as a 2.0 clip, and a provenance lie is worse than a missing clip. One JSON line per such take (stem, cfg_alpha, voice, attempt, reason), append-only, **last row for a stem wins**; stage 06's `original_cfg_alpha_for` reads it and falls back to `--original-cfg-alpha` for the >99% of clips with no row. A clip that already has a row gets a fresh one even when it is re-drawn at the run's value, so a clip deleted by `02_audit_durations.py` (that is how the audit repairs one) and regenerated at 2.0 cannot keep publishing 1.6. Check it with:

  ```bash
  wc -l /media/.../v2/dictionary/cfg_alpha_sidecar.jsonl        # how many clips descended
  jq -r '.cfg_alpha' .../cfg_alpha_sidecar.jsonl | sort | uniq -c   # and onto which rungs
  ```

  Hundreds of clips sitting on the floor (1.2) is the signal to look at the **texts** (respelling, `respell_for_tts`) rather than to draw again.

  Clips that fail every attempt stay missing, so **finish every stage with a plain re-run of the same command** to fill the gaps.

- **Clips generated before 2026-10-14 need `02_audit_durations.py`.** The budget and the floor did not exist then, so those clips were accepted whatever their duration, and ~0.2% of them hold a runaway take that stopped just under the server's global cap, i.e. minutes of looping audio under a label that reads in seconds. The audit applies the same bounds (same `utils/frame_budget.py`) to clips already on disk and deletes the offenders, after which stage 05's ordinary resume regenerates them:

  ```bash
  uv run 05_generate_audio/02_audit_durations.py                      # dry run, all stages
  uv run 05_generate_audio/02_audit_durations.py --report bad.jsonl   # keep the list
  uv run 05_generate_audio/02_audit_durations.py --apply              # delete, then re-run stage 05
  ```

  First run, 2026-10-14 over the 29,141 clips then on disk: **51 over budget (0.175%), 0 implausibly short, 0 unreadable**, 33 of the 51 on h_frm_chf. All 51 deleted. Worth re-running after each stage finishes, as a cheap check that the in-run bounds did their job.

- **While a stage runs, listen to a sample of it on the CPU**, with `05_generate_audio/03_qc_cpu_stt.py`. The duration bounds above are the only thing checking the audio during generation, and they are blind to a clip whose length is plausible but whose text is wrong. This script fills that gap early: it tails the audio directory and the cfg_alpha sidecar and transcribes **every retried clip plus a deterministic 5% sample** with the int8 Parakeet ONNX export on the host's idle cores, scoring them with stage 06's own `normalize_for_scoring` / `compute_metrics` and the same gates (CER 0.08, tail CER 0.12).

  ```bash
  cd 05_generate_audio
  ./03_qc_cpu_stt.py                                            # tail the live run, all five stages
  ./03_qc_cpu_stt.py --backfill                                 # also work through what is already on disk
  ./03_qc_cpu_stt.py --sample-percent 0 --backfill --once        # one shot: just the retried clips
  grep "over gate" ../05_generate_audio/qc_cpu_stt.log           # or jq over qc_stt_sidecar.jsonl
  ```

  Why a sample and not everything: stage 05 writes audio at ~28.9x realtime and the CPU reads it at 7-9x (measured 2026-10-14 under the live run: 7.7x on 4 threads, 9.4x on 3 threads x 2 workers, worse above that, memory-bandwidth bound), so a full inline check would fall ~1,900 hours behind over a 485k-clip stage. The ~5.3% it does check needs ~1.5x realtime, which one worker at `--threads 3 --nice 10` delivers while costing the generator ~1-2%. The four benchmark runs that produced those figures cost 7-10% of generation throughput while they ran (the 500-clip interval containing them did 4,380 clips/h against ~4,860 for the three before it), which is why the defaults are one nice'd worker and not the fastest configuration.

  Output is one JSON line per clip in `qc_stt_sidecar.jsonl` beside the audio, append-only and resumable, carrying `reason` (retry or sample), `voice`, `cfg_alpha`, `cer`, `wer`, `tail_cer`, `duration`, `stt_s`, and the transcript for flagged clips. The per-voice CER table in the log is the point of the 5% sample: it answers "is one voice drifting" within hours rather than at stage 06.

  **It is not a quality gate.** The default model (`parakeet-tdt-0.6b-v3-UltiMed-onnx`) was fine-tuned on **v1 of this dataset**, so it is circular: it knows this corpus's vocabulary, v1's mistakes included, and will agree with audio a listener would reject. Treat a high CER as a gross error (text dropped, syllables looped, wrong text, truncation) and a low one as saying nothing. Point `--model-dir` at `parakeet-tdt-0.6b-v3-redux-onnx` for an impartial read.

- **Why `row` on the term stages** (dictionary, drugs, acronyms: 92.5% of the clips). A group there is one TERM and its ~8 written variants, which are independent sentences, so the default `group` mode would make a given drug name exist in the corpus in exactly ONE accent, and a bad voice/term pairing would cost all of that term's clips at once. Row mode hashes `group:index` instead: measured on the real inputs it gives 5.74 distinct voices per term on the dictionary, 4.99 on drugs, 2.68 on acronyms, and a flatter per-voice balance as a side effect (dictionary spread 2.8% -> 1.2%, acronyms 45% -> 16%).
- **Why `group` on the document stages** (PARHAF, PARROT). A group there is one DOCUMENT and the clips are consecutive chunks of it, so one reader per report is the only natural choice: a clinical report that changes speaker mid-sentence is an artifact no real dictation has.
- Row mode on a document stage was a silent no-op until commit `98f79b3` (those rows carry `variant_index: 0` with the real position in `chunk_index`, and the key preferred `variant_index`, collapsing to the group key). Fixed, with tests, so the two modes are now genuinely different there. It does not change what v2 runs, since the document stages use `group` anyway.
- **No `--seed`.** Omitted on purpose, so the run matches v1's behaviour. The flat `seed` field cannot reach this model at all, and the nested `voxtral_noise_seed` that can would pin every clip in the corpus to the SAME noise draw, which works against the acoustic variety v2 exists for. The dataset card says so under "On seeds". A per-clip seed derived from the clip key (the way the voice is) would be the way to get reproducibility without that, but it is a code change plus a new column, so it is not part of this run.
- Each run writes `voice_assignment.json` beside the audio. **Stage 99 labels the `voice` column from that file**, so do not re-order `utils/voice_assignment.py`'s roster afterwards: the file order is part of the mapping and every clip already on disk would become mislabelled.


## 2. QC loop (stage 06)

```bash
cd 06_hotfixes
CFG_ALPHA_START=2.0 CFG_ALPHA_STEP=-0.2 CFG_ALPHA_FLOOR=1.2 ./driver.sh
```

Those three are already the defaults; they are spelled out because the ladder's direction is the thing to not get wrong (see `README_alternating_improvement.md` for the mode alternation, which switches the docker server between the STT and TTS passes).

The clips that keep their original audio get `--original-cfg-alpha`, which is **2.0** since commit `8faef7d`. That value is written into the `cfg_alpha` column of every non-regenerated row, i.e. the large majority of 600k rows, so running over **v1** audio needs `--original-cfg-alpha 1.3` passed explicitly. The exception is a clip stage 05 drew lower: `original_cfg_alpha_for` resolves the clip, reads `cfg_alpha_sidecar.jsonl` beside it (step 1 item 6) and publishes that rung instead, so the column stays true for the handful of clips that descended the ladder.


## 3. Manifests (stage 99), and the four stale-QC rules

```bash
cd 99_hf_release
uv run 01_build_nemo_manifest.py --all       # per dataset, into data/NeMO_files/<name>/
uv run 02_combine_nemo_manifests.py          # release-wide split + per-source sync
uv run 03_sync_hotfix_results.py --apply     # QC columns from stage 06
uv run 04_drop_bad_rows.py --no-asr-flagged  # dry run first, then --apply
uv run 05_normalize_text.py                  # dry run must report 0 changes
uv run 07_write_back_texts.py                # dry run must report 0 changes
uv run scripts/get_statistics.py             # includes the per-voice sections of 76a6b72
uv run scripts/build_parquet.py              # last, once the manifests are final
```

**(a) No QC column is inherited.** The five `generated_dataset.jsonl` carry no QC fields at all, and `01_build_nemo_manifest.py` builds from them plus the audio folder, so `stt_transcript`, `cer`, `cer_tail`, `n_stt_check`, `stt_model`, `qc_status`, `regenerated` and `cfg_alpha` start **absent** in v2 and are only ever written by stage 06 through `03_sync_hotfix_results.py`, over the audio that is on disk. Nothing to clean, as long as the v2 manifests are built in a **new** directory rather than patched from a copy of v1's.

**(b) Durations.** `probe_durations` caches `{name: [duration, size, mtime]}` in `<outdir>/<dataset>/.duration_cache.json` and only reuses an entry while the size and mtime still match the clip on disk (commit `f09620a`); stale and legacy duration-only entries are re-probed with a warning. Before that fix, a cache copied or inherited from v1 answered every v2 file name with a v1 duration, which lands in the manifest and **decides the train/val/test split**. The v1 caches still exist at `/media/.../audio_datasets/NeMO_files/*/.duration_cache.json`; they are now harmless, but there is no reason to copy them over either.

**(c) The ASR-flagged list.** `99_hf_release/asr_flagged_clips.jsonl` holds 63 clip keys (`<source>/<file>`) that an UltiMed-finetuned Parakeet found defective **in v1 audio**. The key is a name, so all 63 would still match and drop freshly generated clips. Pass `--no-asr-flagged` for v2 until `06_hotfixes/04_flag_asr_defects.py` has re-run on the v2 audio (and then drop the flag). `04_drop_bad_rows.py` also warns on its own when the list is older than the clips it flags (commit `bbe6751`).

**(d) `.size_cache.json`.** `scripts/get_statistics.py` keys its size cache on **absolute** paths, which for v2 live under the new v2 root, so it cannot serve a v1 value. Nothing to do; noted because the handoff flagged it as a risk.


## 4. Pin the splits to v1 (required for any v1-vs-v2 comparison)

`assign_splits` (`utils/nemo_manifest.py`) is duration-greedy, so v2's new durations would produce a **different** train/val/test than v1's. Every v1-vs-v2 WER comparison would then leak: a sentence that was in v1's test set can land in v2's train set, and the v2 model is evaluated on text it was trained on.

Both stage-99 split sites now take `--pin-splits-from DIR` (`load_pinned_splits` + `assign_splits_pinned`):

```bash
V1=/media/$USER/ssd1TO/audio_datasets/NeMO_files
uv run 01_build_nemo_manifest.py --all --pin-splits-from "$V1"
uv run 02_combine_nemo_manifests.py  --pin-splits-from "$V1"
```

Each clip keeps the split it had in v1, matched on `clip_key` (`<source folder>/<file>`, which survives the audio moving to the v2 root). **Only the split label is read**, never any other column, so no v1 duration, CER, transcript or `qc_status` rides along. A clip v1 did not have follows its group's pinned split (so an atomic group cannot straddle two splits) and, failing that, is split normally; the run logs the three counts.

Measured on the real v1 manifests: 602,813 pins read in 17.3 s, no disagreement between the per-dataset and release-wide levels, 80.6 / 9.6 / 9.8 shares. Against v2's 602,885 input rows that leaves exactly the 72 restored rows to split fresh.

`--pin-splits-from` is refused in the combiner's `--no-stratify-duration` mode, which only concatenates the per-dataset splits: pin those in `01_build_nemo_manifest.py` instead.

A second confound, independent of the splits: v2 has slightly more rows than v1 (the 72 restored, minus 2 dropped for text). A parakeet v2-vs-v1 comparison that wants to isolate *voice diversity* has to match the row count, otherwise dataset size is confounded with it.


## 5. Release obligations when v2 uploads

v2 is pushed **over** the v1 Hub repo id (`Olicorne/UltiMed-ASR-FR-v1`, rename later, user's call of 2026-10-13), so the card and the audio land in one operation or the card lies about what is published. `99_hf_release/README.md` is that published card and still describes v1: one `fr_female` voice, `CFG alpha | 1.3`. It must gain the `voice` column, carry `CFG alpha | 2.0`, and describe `cfg_alpha` as 2.0 for the first pass and **1.2-2.0** for regenerated clips (the descending ladder of commit `bd062a0`, not the old 1.5-2.5).

See the `TODO (v2)` block under "Release obligations" in `CLAUDE.md` for the full list.
