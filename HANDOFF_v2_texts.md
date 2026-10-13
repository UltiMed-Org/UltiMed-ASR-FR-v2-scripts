# Handoff: v2 texts are ready, notes before regenerating the audio

Written 2026-10-13 by a Claude Code session (the label-audit session) for the Claude Code session building the 10-voice TTS engine. Read it before running stage 05 for v2.

## What changed in the inputs

Every stage's `generated_dataset.jsonl` (`01_dictionnary`, `02_drugs`, `03_PARHAF`, `04_PARROT`, `07_acronyms`) now carries the final v2 texts:

- `asr_training_target` = the reviewed manifest `text` (label conventions of `utils/label_conventions.py`: v1.3 labels, spelling rules, drug casing, names and ages spread, English drug names respelled with their French final e, Elodie -> Hélodie...).
- `asr_training_source` = the manifest source, with the TTS respellings already applied (for example `IgM` -> `I-gé-M`, `oestrogène` -> `éstrogène`). The respellings live in `respell_for_tts` (`utils/label_conventions.py`) and only ever touch the source, never the target.
- Every other field (`term`, `term_index`, `variant_index`, `source_id`, ...) is untouched, so the stage 05 file names and the voice assignment group keys do not move.

This was done by `99_hf_release/07_write_back_texts.py` (matched on audio folder, `term_index`, `variant_index`). A re-run reports 0 changes. The pre-write-back files are backed up in `/media/$USER/ssd1TO/audio_datasets/generated_datasets_v1_bak-ignore_backups/`.

Row counts versus v1:

- 72 rows that `04_drop_bad_rows.py` had dropped for AUDIO reasons (bad v1 clip, fine text) were KEPT: their texts went through 06 then 05 in `/media/$USER/ssd1TO/audio_datasets/v2_text_prep-ignore_backups/dropped_rows.jsonl`. They have no v1 split (see below).
- 2 rows were dropped for their text (ellipsis): dictionary `3407_2`, PARHAF `10662_0`.

## Paths

- The `generated_audios` symlinks of the five stages now point to EMPTY folders `/media/$USER/ssd1TO/audio_datasets/v2/<dictionary|drugs|PARHAF|PARROT|acronyms>`. `07_acronyms/generated_audios` is new.
- `99_hf_release/data` -> `/media/$USER/ssd1TO/audio_datasets/v2`, so `99_hf_release/data/NeMO_files` does not exist until the v2 manifests are built. Scripts defaulting to `data/NeMO_files` will say "no manifest found" until then.
- v1 audio and the v1 (now relabelled) manifests stay in `/media/$USER/ssd1TO/audio_datasets/` (`NeMO_files/`, `dictionary/`, ...). Backup of the manifests before the TTS respell: `NeMO_files_bak-2026-10-13-pre-tts-respell-ignore_backups`.

## Things to handle before or while building the v2 manifests

1. **Split reshuffle (most important).** `assign_splits` (`utils/nemo_manifest.py`, called by `01_build_nemo_manifest.py` and `02_combine_nemo_manifests.py`) is duration-greedy. New durations give a different train/val/test, so v1 test clips would land in v2 train and every v1-vs-v2 comparison on test would leak. Recommendation: pin each v2 clip to its v1 split by clip key (audio folder, `term_index`, `variant_index`), read from the v1 `NeMO_files/*/{train,val,test}.jsonl`. Only the 72 kept rows need a fresh split (put them in train, or assign them with `assign_splits` keeping their group together).
2. **`99_hf_release/asr_flagged_clips.jsonl`** (63 rows) lists v1 clips with ASR-detected defects. `04_drop_bad_rows.py` reads it by clip key, so it would wrongly drop the v2 versions of those 63 clips. Empty it for v2, or rebuild it from v2 audio with `06_hotfixes/04_flag_asr_defects.py`.
3. **QC columns and durations are stale.** Any QC column, duration or STT transcript (`stt_transcript` is Whisper's v1 output, e.g. "Simvastatin Teva", not a label) comes from v1 audio. The duration probe in `99_hf_release/scripts/get_statistics.py` caches by file path in `<root>/.size_cache.json`: since v2 reuses the same file names, delete that cache.
4. **QC must be redone** on the v2 audio anyway.

## TTS notes

- `cfg_alpha` only changes the acoustic flow-matching step. The semantic tokens are an argmax taken before CFG, so `cfg_alpha` cannot change the pronunciation. Pronunciation fixes have to go through `respell_for_tts`.
- Seeds are not byte-reproducible when the server batches requests: the same seed can give a slightly different clip.
- Duplication: the request bodies of stage 05 (`05_generate_audio/`) and stage 06 (`06_hotfixes/`) are built separately and kept in sync only by `tests/test_tts_wire_format.py`. If you change the wire format (voice field...), change both and run that test.

## Checking the texts later

- `uv run 99_hf_release/05_normalize_text.py <manifests>` (dry run) must show 0 changes on the v2 manifests once built.
- `uv run 99_hf_release/07_write_back_texts.py <manifests>` (dry run) shows whether the generated datasets and the manifests still agree.
