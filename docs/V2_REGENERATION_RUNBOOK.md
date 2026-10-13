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


## 1. Generate (stage 05)

One run per stage, each resumable (existing files are skipped, so a crash costs nothing):

```bash
cd 05_generate_audio
for s in 01_dictionnary 02_drugs 03_PARHAF 04_PARROT 07_acronyms; do
  uv run 01_generate_audio.py \
    --input  "../$s/generated_dataset.jsonl" \
    --output "../$s/generated_audios" \
    --cfg-alpha 2.0 \
    --concurrency 4
done
```

- `--voice-mode group` is the default and is what v2 wants: one speaker per term (or per document), assigned by hashing the group key so adding or removing a row never moves another group to another voice.
- `cfg_alpha` **2.0** is the top of the band measured on 2026-10-13 (16 voices x 10 values, 160 takes): usable 1.2-2.0, and 1.5-2.0 for the hybrid voices. It is deliberately the top, not the middle: below 1.2 the flow-matching decoder stops following the text and degrades fast, while above 2.0 distortion grows and takes increasingly fail to terminate at all. Stage 06's retry ladder therefore walks **down** from it (step -0.2, floor 1.2) rather than up.
- `--concurrency 4` because the acoustic stage runs `codec-bs=1`, so Voxtral's audio throughput saturates around client concurrency 4-8.
- Each run writes `voice_assignment.json` beside the audio. **Stage 99 labels the `voice` column from that file**, so do not re-order `utils/voice_assignment.py`'s roster afterwards: the file order is part of the mapping and every clip already on disk would become mislabelled.


## 2. QC loop (stage 06)

```bash
cd 06_hotfixes
CFG_ALPHA_START=2.0 CFG_ALPHA_STEP=-0.2 CFG_ALPHA_FLOOR=1.2 ./driver.sh
```

Those three are already the defaults; they are spelled out because the ladder's direction is the thing to not get wrong (see `README_alternating_improvement.md` for the mode alternation, which switches the docker server between the STT and TTS passes).

The clips that keep their original audio get `--original-cfg-alpha`, which is **2.0** since commit `8faef7d`. That value is written into the `cfg_alpha` column of every non-regenerated row, i.e. the large majority of 600k rows, so running over **v1** audio needs `--original-cfg-alpha 1.3` passed explicitly.


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


## 4. Still open: pin the splits (blocks the v1-vs-v2 comparison, not the generation)

`assign_splits` (`utils/nemo_manifest.py`) is duration-greedy, so v2's new durations produce a **different** train/val/test than v1's. Every v1-vs-v2 WER comparison then leaks: a sentence that was in v1's test set can land in v2's train set, and the model trained on v2 is evaluated on text it saw.

Recommendation (agreed with the handoff, not implemented): at manifest-build time, pin each v2 clip to its v1 split by clip key (audio folder, `term_index`, `variant_index`), read from `/media/.../audio_datasets/NeMO_files/*/{train,val,test}.jsonl`. Take **only** the split label from those rows, never any other field, so no v1 QC value rides along. The ~72 rows that v1 had dropped for audio reasons have no v1 split and need a fresh one (train, or `assign_splits` with their group kept together).

This is needed after generation, so it blocks nothing today. It does have to exist before any v2 model is compared with the published v1 numbers.

A second confound, independent of the splits: v2 has slightly more rows than v1 (the 72 restored, minus 2 dropped for text). A parakeet v2-vs-v1 comparison that wants to isolate *voice diversity* has to match the row count, otherwise dataset size is confounded with it.


## 5. Release obligations when v2 uploads

v2 is pushed **over** the v1 Hub repo id (`Olicorne/UltiMed-ASR-FR-v1`, rename later, user's call of 2026-10-13), so the card and the audio land in one operation or the card lies about what is published. `99_hf_release/README.md` is that published card and still describes v1: one `fr_female` voice, `CFG alpha | 1.3`. It must gain the `voice` column, carry `CFG alpha | 2.0`, and describe `cfg_alpha` as 2.0 for the first pass and **1.2-2.0** for regenerated clips (the descending ladder of commit `bd062a0`, not the old 1.5-2.5).

See the `TODO (v2)` block under "Release obligations" in `CLAUDE.md` for the full list.
