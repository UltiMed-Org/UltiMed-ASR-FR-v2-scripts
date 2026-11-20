
*Here, I intend to keep track of how I'm creating and curating the dataset for ASR finetuning*
# Dataset creation logs

---

# 01_dictionnary

## filtering

Generated using:

```bash
uv run 01_llm_scoring.py original_dictionnary.jsonl --model "openrouter/openai/gpt-oss-120b" --resume --limit 1000
```

- **Entries:** 62601
- **Estimated input tokens:** ~47,924,801 (62601 entries × ~628 system tokens + entry text)

### Score distribution (after scoring 62569 entries)

| Score | Count | % |
|------:|------:|------:|
| 0 | 2494 | 3.99 |
| 1 | 794 | 1.27 |
| 2 | 4601 | 7.35 |
| 3 | 1332 | 2.13 |
| 4 | 1459 | 2.33 |
| 5 | 4233 | 6.77 |
| 6 | 7916 | 12.65 |
| 7 | 6714 | 10.73 |
| 8 | 11362 | 18.16 |
| 9 | 21652 | 34.60 |
| 10 | 12 | 0.02 |
| **Total** | **62569** | **100.00** |

We ended up *not* filtering out low-score entries. Instead, the score is fed
into `03_generate_texts.py` to decide how many text samples to generate per
term (via the `--n-variants` range-policy), so lower-quality entries simply
contribute fewer variants rather than being dropped entirely.

## Text generation

Generated the texts with:
```
uv run 03_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" --n-variants '{"0-1":2,"2-3":4,"4":5,"5":6,"6":7,"7":8,"8-10":11}' -v
```

With a cache hit rate of about 95%, the total cost to transform the 62 461 terms into 534 441 text samples cost between $50 and $60.

# Audio generation

I decided to only generate audios using the female voice as it's very high quality, subjectively much higher than `fr_male`.
```
uv run ../05_generate_audio/01_generate_audio.py --input generated_dataset.jsonl --output voxtral_audios_euler_and_cfg_max --voice "fr_female" --seed 42 --concurrency=10
```

---

# 02_DRUGS

## Text generation

```bash
uv run 01_generate_drug_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" --n-variants '{"0-1":2,"2-3":4,"4":5,"5":6,"6":7,"7":8,"8-10":11}' -v
```

Same engine and score policy as the dictionary stage: reads `drugs_freq_dosages.jsonl`, emits N `asr_training_target` variants per drug (each must contain the drug name) plus the deterministic `asr_training_source`, into `generated_dataset.jsonl`.

---

# 03_PARHAF

Source: [HealthDataHub/PARHAF on Hugging Face](https://huggingface.co/datasets/HealthDataHub/PARHAF).

Dual-licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and [Etalab Open License 2.0](https://github.com/etalab/licence-ouverte/blob/master/LO.md). Both allow commercial use, modification, and redistribution of derivative works (text-to-speech audio counts as a derivative). The only real obligation is attribution.

## What I have to do when releasing the derived ASR dataset

- **Credit the source.** Name the licensor (HealthDataHub / Plateforme des Données de Santé), link back to the [HF dataset page](https://huggingface.co/datasets/HealthDataHub/PARHAF), and mention the date / version of PARHAF used.
- **Cite the paper.** PARHAF, arXiv [2603.20494](https://arxiv.org/abs/2603.20494) (BibTeX key `parhaf2025` on the dataset card).
- **State both licenses** of the upstream data (CC BY 4.0 + Etalab 2.0), keep them in a NOTICE / README shipped with the derived dataset, and indicate that the data has been modified (text synthesized to audio), as CC BY 4.0 requires.
- **No endorsement.** Do not suggest HealthDataHub endorses the derived dataset or any model trained on it (explicit requirement of Etalab 2.0).
- **No clinical claims.** PARHAF's dataset card lists clinical decision-making, clinical validation, performance claims and clinical deployment as out-of-scope uses. The release must carry a "research use only, not for clinical deployment" disclaimer.
- **Do not redistribute the test set.** Only the training set is publicly released; test sets are under embargo.

## parquet to jsonl

```bash
uv run 01_parquet_to_jsonl.py
```

Reads `train-00000-of-00001.parquet` and writes `01_parhaf_documents.jsonl`, keeping only `id`, `local_id` and `documents` per row (4254 entries).

Then `02_clean_split_texts.py` turns that into `02_parhaf_texts.jsonl`: one coherent longish French text per entry, line-filtered (keeps lines ending in `.`/`?`, no colon, at least 15 chars, not all-caps, starting with a capital then a non-capital), parentheses and `°` normalized away, contiguous survivors joined and blocks under 80 chars dropped. Each entry gets `id = {orig_id}-t{text_index}-s{split_index}` and `category = "parhaf"`. This file is now audit-only; the dataset text comes from the rewrite path below.

## Text generation

Chunk the raw documents, then rewrite each chunk into one faithful French paragraph (fixed one paragraph per chunk, so no `--n-variants`):

```bash
uv run 03_chunk_for_rewrite.py
uv run 04_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

The rewrite prompt is the shared base rules plus a PARHAF clinical-document example (`PROMPT_REWRITE_PARHAF_EXAMPLES.md`), written into `generated_dataset.jsonl`.

### The 40 chunks the length cap rejected: `--no-max-len-ratio`

That run left 40 chunks out of 43456 unwritten, all logged to `term_missing_skips.jsonl` as `validation_after_retries`. Every one of them failed the same validator, the rewrite length upper bound (`MAX_LEN_RATIO = 2.0` in `utils/text_rewrite.py`), at 201% to 318% of the source length.

They are not bad rewrites. They are the chunks that are almost pure shorthand, lab panels and biology tables (half of them are over 10% digit characters), where the mandatory spell-out mechanically triples the character count: `Na+ 142meq/L; K+ 3,6meq/L` becomes "le sodium est à cent quarante-deux milliéquivalents par litre, le potassium à trois virgule six milliéquivalents par litre". No faithful rewrite of those fits under any cap worth keeping for the other 99.9% of chunks.

So the cap is turned off for a targeted re-run instead of being raised. `--no-max-len-ratio` drops the upper bound only, the lower bound (too much content dropped) still applies. The run resumes from `generated_dataset.jsonl`, so it picks up exactly those 40 pending chunks and nothing else:

```bash
uv run 04_generate_texts.py --no-max-len-ratio --n-jobs 8 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

`03_PARHAF/skipped_chunks_inspect.jsonl` (uncommitted, rebuildable) holds those 40 source chunks with their failure ratios, for eyeballing what they look like. The same flag exists on `04_PARROT/04_generate_texts.py`, which shares the rewrite engine.

---

# 04_PARROT

Source: [PARROT radiology reports](https://doi.org/10.1016/j.ejrai.2025.100066), French rows only. Source link and license note are done, in `99_hf_release/NOTICE.md`: CC BY-NC-SA 4.0, shipped as a separate non-commercial eval-only subset.

## Text generation

`01_filter_french.py` keeps the French rows, then the same chunk + rewrite path as PARHAF (one paragraph per chunk):

```bash
uv run 03_chunk_for_rewrite.py
uv run 04_generate_texts.py --n-jobs 32 --model="openrouter/deepseek/deepseek-v4-pro" --provider="deepseek" -v
```

Same shared base rules, but with a PARROT radiology example (`PROMPT_REWRITE_PARROT_EXAMPLES.md`), written into `generated_dataset.jsonl`.

---

# 99_hf_release

Final pipeline stage: assemble the generated audio + transcripts into NeMo ASR
manifests, then publish to Hugging Face. Both the manifest builders below and the
Parquet packaging / upload are now done. Written with Claude Code.

## NeMo-format manifests

Stage `05_generate_audio` only writes the audio clips (one `.flac` per row, named
`{term_index:06d}_{variant_index:04d}_{normalized_term}.flac`); it emits no
manifest. Two scripts here turn each stage's `generated_dataset.jsonl` + its clips
into the **NeMo-format JSONL** the fine-tuning pipeline in `../NeMo/` consumes
(one object per line, `{"audio_filepath": "...", "duration": 3.21, "text": "..."}`,
plus a few provenance fields NeMo ignores). The shared split logic lives in
`utils/nemo_manifest.py` so neither script forks it; it is covered by
`tests/test_nemo_manifest.py`.

### Per dataset: `01_build_nemo_manifest.py`

Writes `data/NeMO_files/<name>/{full,train,val,test}.jsonl`.

```bash
uv run 01_build_nemo_manifest.py --all           # all four known stages
# or one explicitly:
uv run 01_build_nemo_manifest.py \
    --input ../01_dictionnary/generated_dataset.jsonl \
    --audio-dir data/dictionary --name dictionary
```

- `text` is `asr_training_target` (the written label), not the TTS source.
- Clips are matched to rows by the stage-05
  `{term_index:06d}_{variant_index:04d}` filename prefix (reusing that naming, not
  re-deriving the slug); rows whose audio is missing are dropped.
- `duration` is read from the FLAC header (`soundfile`) and cached per dataset in
  `.duration_cache.json`, so re-runs are cheap. Each entry stores the clip's size and
  mtime next to the duration and is only reused while both still match, because the
  same file name carries different audio twice over: stage 06 overwrites a bad clip in
  place, and a new generation of the dataset (v2) re-synthesizes every name. Entries
  that fail that check, and legacy duration-only ones, are re-probed with a warning.
- `audio_filepath` is stored **relative** to the manifest, never absolute: an
  absolute path under the audio SSD would bake in the home/username.
- `--split` (default `80/10/10`) sets the ratio; a `0` disables that split.
- `--stratify` (default on) keeps groups sensible. dictionary/drugs term variants
  are *distributed* (>=1 variant in train; >=1 in every split when the term has
  >=3 variants); PARHAF/PARROT document chunks are *atomic* (all chunks of one
  `source_id` land in the same split, so no document leaks train<->test).
  Auto-detected: rows with a `source_id` are atomic-by-document, the rest
  distribute-by-term.
- `--stratify-duration` (default on) balances the split by summed clip duration
  instead of row count.
- `--strict-coverage` / `--best-effort-coverage` (default best-effort) trades term
  coverage against ratio accuracy. Best-effort guarantees only >=1 variant in
  train, so val/test hit 80/10/10 (dictionary and drugs both land at 10% by
  duration). Strict additionally forces every >=3-variant term into all three
  splits: each split then sees the term, but that floors val/test above 10%
  (drugs ~15%). The atomic stages (PARHAF/PARROT) are unaffected either way.

### Combined: `02_combine_nemo_manifests.py`

Reads every per-dataset `full.jsonl` and writes the release-wide
`data/NeMO_files/{full,train,val,test}.jsonl`, re-basing each `audio_filepath` for
the shallower parent location (a plain concat would point one `../` too far).

```bash
uv run 02_combine_nemo_manifests.py                   # global duration re-split
uv run 02_combine_nemo_manifests.py --no-stratify-duration   # plain concat
uv run 02_combine_nemo_manifests.py --exclude PARROT  # leave a subset out of the combine
```

`--stratify-duration` (default on) re-derives the split over the pooled rows so the
combined manifest hits the target ratio by **total audio duration** (the proxy for
how much speech each split holds), staying group-aware (atomic documents whole,
term guarantees preserved). Off just concatenates the per-dataset splits and
reports the duration skew. The same `--strict-coverage` / `--best-effort-coverage`
(default best-effort) applies to the global re-split. `--exclude NAME` (repeatable)
leaves a named subdataset out of the combined manifests, so a differently-licensed
subset (the release ships PARROT as a separate, test-only CC BY-NC-SA 4.0 subset,
built with `--split 0/0/100`) can be kept separable while the rest stays under one
licence.

### Statistics: `scripts/get_statistics.py`

Prints a Markdown report over the manifests: character / word / duration
min/mean/median/max, file and duration counts, and a duration histogram, broken
down per split, per dataset, and per dataset x split.

When the rows carry a `voice` (v2 audio), three more sections appear: **By voice**
(clips, hours and text stats per voice, with the spread as `max - min` over the mean),
**Voice balance per dataset** (groups, voices used and the spread of groups / clips /
hours per voice) and **By voice x split** (each voice's share inside train, val and
test). They are the measured answer to a question the hash assignment leaves open: it
trades exact balance for edit and resume stability, so "how unbalanced, and where" is
not predictable from the code, only countable afterwards. Expect the small datasets to
be the skewed ones: a few hundred groups cannot divide evenly over ten voices. A v1
report (no `voice` field anywhere) omits all three rather than printing a table about
one unnamed voice.

```bash
uv run scripts/get_statistics.py                    # per-dataset splits, to stdout
uv run scripts/get_statistics.py --source combined  # the release-wide re-split
uv run scripts/get_statistics.py --output stats.md
```

## Release obligations

This stage also carries the attribution / license / disclaimer requirements
already documented in the **03_PARHAF** section above (credit the source, cite
the paper, state both licenses in a NOTICE, mark the data as modified, no
endorsement, research-use-only disclaimer, do **not** redistribute the PARHAF
test set). PARROT's source link and license note are done, see `99_hf_release/NOTICE.md`.

## Label spelling fixes after the v2 model (2026-11-15)

The v2 model was trained **before** these fixes: its training labels still carry the old spellings. They live in the convention code and tables only, so the next `05_normalize_text.py` export applies them to `text` and `asr_training_source`; no exported file was rewritten when they were made. Made with Claude Code.

- **Source-sheet misspellings**: `nalbufine` -> `nalbuphine`, `polyomielite` -> `poliomyélite` (and the DTP compound), `glycopirronium` -> `glycopyrronium`. The first two (and `TIOGUANINE`) come from the CNAM Open Medic `L_ATC5` column read by `02_drugs/sources/drugs_frequency/02_combine_datasets.py` (a TODO there says to fix them at the source); `glycopirronium` is an LLM typo.
- **Typos**: the owner's list (`perindoprill`, `azythromycine`, `gentamycine`, `cloxaciline`, `entreesto`, `luspathercept`, `affinitor`, `ellebrato`, `clotafact`, `hydrochlorotiazide`, `tazociline`, `aktiskenan` and its variants) plus a systematic search of the drug names in the labels (`amitryptiline`, `doxycyline`, `néphopam`, `ferrinject`, `vycril`, English INN forms such as `empagliflozin`, ...). The ones that change the sound (most of these) are held back until re-synthesis, see below.
- **Spelling winners**, two right spellings and one taught: `aciclovir`, `cyclosporine`, `thioguanine`, `cefalexine` (no accent, per the INN convention; it used to be kept as an exception), `alfacalcidol`, and `estr-` over `oestr-` (`estrogènes`, `estradiol`, `phytoestrogènes`, `diéthylstilbestrol`; `oestrus`, `oestral` and the `Oestrogel` brand keep the `oe`).
- **INN accents**: the INN-without-accents convention now also covers the accented INNs the audit skipped (`ertapenem`, `lanadelumab`, `olodaterol`, `dexamethasone`), including the ones whose accented form still had the majority (`levetiracetam`, `levothyroxine`, `methylprednisolone`, `meticilline`, ...), kept in their own dict and commit. `protéase`, `paracétamol` and `méthotrexate` stay accented. The TTS text keeps the accents voxtral needs.
- An all-caps word now hits the accent table too (`PERINDIPRIL` -> `perindopril`), and a typo entry can never be inverted into the TTS respellings.
- A rewrite that turns a consonant-initial word into a vowel-initial one elides the particle before it (2026-11-16): `le cestradiol` -> `l'estradiol`, `de` -> `d'`, `que` -> `qu'`, `du` -> `de l'`, `au` -> `à l'`, `ce` -> `cet` (`_elide` in `utils/label_conventions.py`, pinned by `test_accent_rewrite_elides`). In the whole table only `cestradiol` -> `estradiol` changes the initial letter class, and its one clip (`...par litre, cestradiol à 200...`) has no particle before it, so nothing elides in practice yet.

The rewrites are `HAND_REWRITES` in `06_hotfixes/07_build_accent_spellings.py`, synced by hand into `utils/accent_spellings.json` (since 2026-11-19 by running `07_build_accent_spellings.py` with no TSV argument; the audit TSV the builder reads is not committed; `tests/test_label_variant_audit.py` checks the two match); the `estr-` rule is in `_SPELLING_RULES` of `utils/label_conventions.py`. Dry run of the new conventions against the exported labels (602,776 clips, all splits, token changes / clips touched):

Glued compounds (2026-11-16): `normalize_compounds` runs before the accent table, so a hyphenated key like `crânio-caudale` reached it already glued (`crâniocaudale`) and never matched. Seven words kept two label spellings, and `convention_fixes.tsv` (UltiMed-ASR-FR-v2-regex-fixes) mapped their accent variants to both. `label_conventions._accent_table` now also carries the glued form of every hyphenated key the compound rule glues (an explicit key wins), which gives exactly those seven, all on the owner's choice (dictionary spelling): `craniocaudale`, `craniocervicale`, `craniofacial`, `craniorachidienne` (no circumflex on `cranio`), `antithyroperoxydase` (no accent), `sternocléidomastoïdien` and `sternocléidomastoïdiens`. Label only: the TTS source skips the accent table and keeps `crâniocaudale` (`tests/test_label_conventions.py::test_glued_compounds_take_the_accent_choice`).

Colliding spellings, decided by the owner (2026-11-16), counts as measured in the export's labels before the decision:

- `écho-endoscopie` over `échoendoscopie` (942 against 92 for the `-endoscopi*` family, the hyphen wins). The compound table only had the singular forms, so `échoendoscopies`, `échoendoscopiques`, `échoendoscope(s)` and the spaced forms stayed as written; the unaccented `echoendoscopie` (no clip in the labels) was mapped by the accent table to the glued `échoendoscopie` and now goes to `écho-endoscopie`.
- `pipéracilline-tazobactam` over `pipéracilline tazobactam` (300 against 62, the hyphen wins), every casing (`Pipéracilline Tazobactam` -> `Piperacilline-tazobactam`). The accent still goes, by the INN convention already in the accent table, so the label reads `piperacilline-tazobactam`. An all-caps `PIPERACILLINE TAZOBACTAM` gets the hyphen but keeps its capitals (no clip has it).
- `cranio-encéphalique`, hyphenated and without circumflex, over `crânio-encéphalique` (101), `cranioencéphalique` (90) and `crânioencéphalique` (21): the owner chose this form although it is not the majority (35), to match `cranio-caudale` and the other `cranio` words above. Before, the compound table glued `cranio-encéphalique` while the accent table produced it from `crânio-encéphalique`, so two label spellings coexisted; the plural `crânio-encéphaliques` was missing from the accent table.
- `antépartum`, fused and accented (the dictionary form), over `ante-partum` (38), `antépartum` (36), `anté-partum` (17) and `antepartum` (12). Before, the compound table hyphenated `antepartum` and the accent table removed the accent, giving `ante-partum`.

As for every compound choice, the separator lives in `utils/compound_spellings.json` (`OVERRIDES` of `06_hotfixes/08_build_compound_spellings.py`) and also reaches the TTS source, since a hyphen is silent; the letters (`â` -> `a`, `ante` -> `anté`) live in `utils/accent_spellings.json` (`OVERRIDES` of `06_hotfixes/07_build_accent_spellings.py`) and stay label-only. Pinned by `tests/test_label_conventions.py::test_colliding_spellings_owner_choices`. Dry run against the exported `full.jsonl` labels (tokens rewritten): `écho-endoscopi*` 3 (`échoendoscope` 2, `écho endoscopiques` 1), `piperacilline-tazobactam` 61, `cranio-encéphalique(s)` 112 (`cranioencéphalique(s)` 90, `crânio-encéphaliques` 22), `antépartum` 38 (`ante-partum`). The decision counts above do not match the current `text` field (which already holds only these variants), so the dry run is what the next export actually changes. Made with Claude Code.

| Fix | Tokens | Clips |
|---|---:|---:|
| Source-sheet misspellings | 15 | 15 |
| Typos | 117 | 116 |
| Spelling winners | 584 | 564 |
| `oestr-` -> `estr-` | 1,190 | 1,093 |
| INN accents, unaccented majority | 187 | 185 |
| INN accents, accented majority | 5,220 | 5,150 |

These counts predate the owner's hand review (`OWNER_REVIEW`, about 30 more clips) and the held-back sound-changing typos below (59 fewer clips); not re-run.

Every one of these fixes rewrites the label (`text`) only. The `asr_training_source`, which `07_write_back_texts.py` later feeds back to the TTS stage, keeps what voxtral actually read: `apply_label_conventions(..., tts_source=True)` skips the spelling fixes, and `tests/test_label_conventions.py::test_spelling_fixes_never_reach_tts_source` sweeps the whole table to pin it (before that fix, 326 of the 630 entries leaked into the source).

Some typo rewrites change the sound, not only the spelling, so the fixed label is a near-miss of what the audio says. Owner's rule (2026-11-16): **every** sound-changing fix is held back, and when in doubt it is held; pure accent, case, hyphen or silent-letter fixes (`azythromycine`, `gentamycine`, `aktiskenan`, `ferrinject`, `bêtadine`, `mesalamine`, ...) are not. The held ones live in `RESYNTH_TYPOS` (`06_hotfixes/07_build_accent_spellings.py`), outside `HAND_REWRITES` and the JSON, until their clips are re-synthesized (`tests/test_label_variant_audit.py::test_resynth_typos_held_back` keeps them out). That is 47 table keys and **59 clips** in PARHAF, dictionary and drugs (none in PARROT or acronyms). Two keys sit in no released clip: `actisenan` (the clip spells it `Actisénan`) and `actiksenan`, which therefore stays in `TYPOS` (nothing to re-synthesize). Borderline cases held for safety: `tocoferol` (accent plus `ph`, same call as `theralithe`), `perindoprill`, `entreesto`, `teimocilline`, `bénzatine` (may stay nasal), `desmoprésine` (s/z) and the English INN forms (`empagliflozin`, `midostaurin`, `darbepoetin`, whose final `-in` voxtral may read as a nasal).

| Typo | Fix | Clips (release subdir) |
|---|---|---|
| `PERINDIPRIL` | perindopril | `PARHAF/028064_0000_medecine_inter_specialites_00014_t0_c1` |
| `Bisorolol` | bisoprolol | `PARHAF/028405_0000_medecine_inter_specialites_00158_t0_c7` |
| `KAREGIC` | Kardégic | `PARHAF/028507_0000_medecine_inter_specialites_00209_t0_c2` |
| `Cetriaxone` | ceftriaxone | `PARHAF/031867_0000_nephrologie_00012_t0_c4` |
| `Actisénan` | Actiskenan | `PARHAF/003372_0000_cancero_adulte_00155_t0_c1` |
| `perindoprill` | perindopril | `PARHAF/030951_0000_medecine_interne_00299_t0_c8` |
| `ENTREESTO` | Entresto | `PARHAF/018312_0000_hematologie_clinique_00093_t0_c1` |
| `hydrochlorothéiazide` | hydrochlorothiazide | `PARHAF/028473_0000_medecine_inter_specialites_00203_t0_c1` |
| `Ackiskenan` | Actiskenan | `PARHAF/002303_0000_cancero_adulte_00015_t0_c1` |
| `Actikskenan` | Actiskenan | `PARHAF/007902_0000_chir_ortho_et_traumato_00055_t2_c5` |
| `Actikenan` | Actiskenan | `PARHAF/034245_0000_neurologie_00142_t0_c6`, `PARHAF/034246_0000_neurologie_00142_t0_c7` |
| `DOXYCYLINE` | doxycycline | `PARHAF/004741_0000_cancero_adulte_00372_t0_c4` |
| `CEFRIAXONE` | ceftriaxone | `PARHAF/020956_0000_maladies_infectieuses_00059_t0_c5` |
| `amplodipine` | amlodipine | `PARHAF/017751_0000_hematologie_clinique_00013_t0_c1` |
| `Pantopazole` | pantoprazole | `PARHAF/037375_0000_pneumologie_00065_t0_c1` |
| `desmoprésine` | desmopressine | `dictionary/056108_0003_toxicite` |
| `candersartan` | candesartan | `PARHAF/033087_0000_neurologie_00009_t0_c7` |
| `aprazolam` | alprazolam | `PARHAF/038269_0000_pneumologie_00168_t0_c3` |
| `Irbersartan` | irbésartan | `PARHAF/021317_0000_maladies_infectieuses_00084_t0_c1` |
| `adalmumab` | adalimumab | `dictionary/019398_0003_enthesite` |
| `zolendronique` | zolédronique | `PARHAF/040870_0000_pneumologie_00556_t0_c0` |
| `FORMETEROL` | formotérol | `PARHAF/034554_0000_neurologie_00201_t0_c1` |
| `Formétérol` | formotérol | `PARHAF/035198_0000_neurologie_00319_t0_c1` |
| `imbrutinib` | ibrutinib | `PARHAF/023398_0000_maladies_infectieuses_00679_t0_c15` |
| `glicazide` | gliclazide | `PARHAF/006752_0000_cardiologie_00052_t0_c0` |
| `lanceotide` | lanréotide | `dictionary/023654_0003_gigantisme` |
| `selpercatnib` | selpercatinib | `PARHAF/004664_0000_cancero_adulte_00350_t0_c13` |
| `tezepilumab` | tezepelumab | `PARHAF/037057_0000_pneumologie_00026_t0_c3` |
| `RILMENILDINE` | rilmenidine | `PARHAF/004638_0000_cancero_adulte_00348_t0_c0` |
| `teimocilline` | temocilline | `PARHAF/040196_0000_pneumologie_00474_t0_c18` |
| `bénzatine` | benzathine | `dictionary/010297_0003_cardiopathie_rhumatismale` |
| `métrondiazole` | métronidazole | `dictionary/024922_0001_ver_de_guinee` |
| `Lévifloxacine` | levofloxacine | `dictionary/009977_0003_capnophile` |
| `Acétrétine` | acitrétine | `dictionary/027940_0003_ichtyose_congenitale_lamelleuse`, `dictionary/027940_0008_ichtyose_congenitale_lamelleuse`, `dictionary/054319_0003_syndrome_kid` |
| `Eczétimibe` | ézétimibe | `dictionary/027306_0003_hyperlipemie` |
| `Tocoferol` | tocophérol | `dictionary/006387_0003_ataxie_de_type_friedreich_avec_retinite_pigmentaire`, `_0008_...`, `_0010_...` |
| `Theralithe` | Téralithe | `PARHAF/018935_0000_hepato_gastro_entero_00039_t0_c1` |
| `BACRIM` | Bactrim | `PARHAF/024948_0000_medecine_geriatrique_00023_t0_c2` |
| `FERINJEC` | Ferinject | `PARHAF/018715_0000_hematologie_clinique_00160_t0_c5` |
| `empagliflozin` | empagliflozine | `PARHAF/029503_0000_medecine_interne_00081_t0_c1`, `PARHAF/029511_0000_medecine_interne_00081_t0_c9` |
| `midostaurin` | midostaurine | `dictionary/034185_0001_mastocytose_systemique`, `_0003_...`, `_0007_...` |
| `darbepoetin` | darbepoetine | `dictionary/020273_0010_facteur_induit_par_l_hypoxie`, `drugs/001732_0004_mircera` |
| `diphtérie-polyomielite-tétanos` | diphtérie-poliomyélite-tétanos | `drugs/002555_0000_diphterie_polyomielite_tetanos` to `_0004_...` (5 clips)` |
| `alfacalcidiol` | alfacalcidol | `PARHAF/002770_0000_cancero_adulte_00071_t0_c1` |
| `Laxilix` | Lasilix | `PARHAF/034637_0000_neurologie_00217_t0_c10` |
| `cestradiol` | estradiol | `dictionary/060916_0009_technique_de_wood` |

One more fix is held the same way but spans several words, so it lives in `RESYNTH_PHRASES` (same file) instead of the word-level table: the dictionary term of group `023937` was scrambled as `capsule de réfléchie Glisson`; the real term is `capsule de Glisson réfléchie` (owner, 2026-11-17; it also exists correctly as group `010000`). All 11 clips `dictionary/023937_0000` to `023937_0010` (`..._capsule_de_reflechie_glisson`) say the scrambled order, so they are re-synthesized with the others. The audio file names keep the old slug, and `01_dictionnary/original_dictionnary*.jsonl` still holds the scrambled term (not edited).

Where the scrambled term came from, and what else was fixed for it (2026-11-17, Claude Code). The dictionary lists some headwords in index form, the qualifier in parentheses after the head: `Glisson réfléchie (capsule de)`, `moniteur (d'un appareil de radiothérapie)`, `hypersignal (en IRM)`. The extraction moved the parenthesis to the front without checking what it was, which gives a real term for an inverted name and a non-term for a domain qualifier (`d'un appareil de radiothérapie moniteur`, `en IRM hypersignal`), and it misplaced the head for a few (`capsule de réfléchie Glisson`, `Codman triangle de`). For index 23937 the `term` field was corrected by hand in the four `01_dictionnary/original_dictionnary*.jsonl` files and in the 11 stage rows of `01_dictionnary/generated_dataset.jsonl` (the reopen chain only rewrites the two text fields); it now duplicates index 10000, which is harmless (two groups of clips for one term). The clip stems keep `capsule_de_reflechie_glisson`: the driver redraws by stem, and no manifest carries the term apart from that name. The phrase was also removed from `parakeet_web/phrase_boosting/french_medical.txt` and the `.pwc` recompiled. The v2 regex rules and lookups never contained it. A hunt over all 62,448 dataset terms for the same defect (same words as another term in another order, leading function word, dangling end, unbalanced bracket) flagged 149 terms, 71 of which appear verbatim in at least one clip (472 rows); the owner chose to rewrite and re-synthesize them, trailing punctuation included (2026-11-17, Claude Code). Result: 60 `term` fields renamed by index in the four `original_dictionnary*.jsonl` files and in 427 rows of `generated_dataset.jsonl` (39 scrambled terms whose clips say the scrambled phrase, 18 terms with a trailing `,` or ` :`, 3 reordered terms whose clips were already right); 10 flagged terms were false alarms. The 39 text fixes are 77 context phrases in `RESYNTH_PHRASES` (258 clips), written per sentence because a bare swap is often not French (`en IRM hypersignal` becomes `un hypersignal`, `en IRM un hypersignal`, ...). Each key was checked to hit only its own term's clips and to be stable on a second pass (a first draft `de Hutchinson melanotic freckle` was not, and was split into four context keys; `tests/test_reopen_resynth_typos.py` covers both properties). Rewrites the owner may want to re-read: 20322 (`facteurs temps d'une irradiation`), 34660 (English term kept), 35432 (word order), 20868 and 26359 (linking words added), 36343 (the four rewrites drop the term: `des unités multiples de la normale` became `plusieurs fois la normale`). For 51878 the owner chose `syndrome de skin peeling` over a French rendering (2026-11-17). The boost list of `parakeet_web` got the same 58 lines replaced. The v2 regex rules and lookups contain none of the old phrases.

Found by scanning the `text` of `99_hf_release/data/NeMO_files/full.jsonl` (602,776 rows) for any `RESYNTH_TYPOS` key as a whole word, case-insensitive (Python `re`, `(?<!\w)(key1|key2|...)(?!\w)`; `grep -iE` also hits non-text fields). They also sit in the split files (`train.jsonl`, `test.jsonl`, ...). The `Actisénan` clip is accented, so the old `actisenan` key never matched it anyway. The "TTS typo guard" (`_tts_accent_table`) only stops a typo entry from being inverted into a TTS respelling; it does not protect these clips, which is why they are held back.

To re-synthesize them with the existing TTS stage (done, see the 2026-11-19 note after the steps; the steps stay as the recipe for a future hold):

1. Rewrite each clip's text with `label_conventions.normalize_accents(text, RESYNTH_TYPOS)` (it also elides the article, `le cestradiol` -> `l'estradiol`) in its source `generated_dataset.jsonl`, in every release manifest (the two `full.jsonl` the rounds read and, since 2026-11-17, the splits, down-samples and per-source files too, see "Split manifests kept the old text" below) and in the stage-06 state file, marking the row `pending_tts`: `06_hotfixes/21_reopen_resynth_typos.py` does this (17's chain, now taking the rewrite as an argument, so no copy). From `06_hotfixes/`: `uv run 21_reopen_resynth_typos.py --out-stems resynth_stems.txt` (census), then the same with `--apply`. It strips the generic lab names too (`--generic-labs`, default on), which queues the `Simvastatine Téva` clip below with them.
   Dry run 2026-11-17 17:12 (Claude Code), with the 78 phrases of `RESYNTH_PHRASES`: 329 clips (the 71 of the run below plus 258), no unmatched row, `actisenan` still the only key without a clip; about 16 min.
   Dry run 2026-11-17 (Claude Code), with `RESYNTH_PHRASES`: 71 clips, the 60 below plus the 11 of group `023937` (all in `01_dictionnary`); about 14 min, the three big manifests are streamed.
   Dry run 2026-11-16 (Claude Code): 60 clips, i.e. 18 rows in `01_dictionnary`, 6 in `02_drugs`, 36 in `03_PARHAF`, 60 in `NeMO_files/full.jsonl`, 60 state rows marked in `improved/full.stt.jsonl` (7 of them currently `improved`, 53 with no marker), 0 in PARROT; per-key counts equal the table above, and `actisenan` is the only key that matches no clip.
2. Nothing to queue by hand: the marking in step 1 is the queue (`19_queue_stems.py` is not needed), and `resynth_stems.txt` is the round's `STEMS_FROM`.
3. Redraw them, from `06_hotfixes/`: `sudo -E MODE=alternate STEMS_FROM=resynth_stems.txt REFRESH_MANIFESTS=0 ACOUSTIC_SCAN=0 ./driver.sh`. Full launch notes (env vars, GPU, how to tell it converged) in `06_hotfixes/README_alternating_improvement.md`, "Re-synthesizing the held-back spelling fixes".
4. Move the `RESYNTH_TYPOS` entries into a dict of the `HAND_REWRITES` union and leave `RESYNTH_TYPOS` as an empty hook, then sync `utils/accent_spellings.json` with `uv run 06_hotfixes/07_build_accent_spellings.py` (no TSV argument: it merges `HAND_REWRITES` over the committed table). Keep `tests/test_label_variant_audit.py::test_resynth_typos_held_back` as the guard for the next hold. `RESYNTH_PHRASES` is NOT merged: once its clips are redrawn no label carries the old phrase, so empty it.
5. Export as below.

Done (2026-11-19, Claude Code). The clips of both batches were redrawn, and a read-only census (`uv run 21_reopen_resynth_typos.py --out-stems ...` in `06_hotfixes/`, about 35 min) found 0 clips carrying any held key (spelling or phrase) across every stage file and all 42 release manifests. Step 4 is applied: the 47 spellings are now `RESYNTHESIZED_TYPOS` in the `HAND_REWRITES` union, a permanent guard against the typos coming back; `utils/accent_spellings.json` gained exactly those 47 entries (604 -> 651, no other line changed, still no chained canonical form); `RESYNTH_TYPOS` and `RESYNTH_PHRASES` are empty hooks (the 78 phrases are in git history), and `21_reopen_resynth_typos.py` now logs that nothing is held and exits at once when both are empty. Step 5 (the export) is not run here.

Generic lab names (`strip_generic_labs`, 2026-11-16 check): **0 clips** have audio saying a lab name that the export would drop from the label. `strip_generic_labs(x) != x` holds for no `text` or `asr_training_source` in `99_hf_release/data/NeMO_files/full.jsonl`, nor for any `asr_training_target` or `asr_training_source` in the `01_dictionnary`, `02_drugs` and `03_PARHAF` `generated_dataset.jsonl` (`04_PARROT` and `07_acronyms` carry no lab string), most likely because those sources were redrawn after the lab rule. Query: `grep -E "Arrow|Biogaran|Cristers|Evolugen|Mylan|Sandoz|Substipharm|Teva|Viatris|Zentiva|Zydus|Almus|Ratiopharm|Qualimed|Krka|Ranbaxy" full.jsonl` gives 27 lines, then keep those where `strip_generic_labs` changes the text: none. The 27 are person names (`Mylan X`, `Monsieur Teva`, `Mylane`), the `Tevagrastim` brand, and one edge case, `PARHAF/011560_0000_chirurgie_viscerale_00018_t0_c1`, whose text and source both say `Simvastatine Téva`: the accented `Téva` was not in `_LAB_RE`. Fixed the same day (`T[eé]va`, case in `tests/test_label_conventions.py::test_generic_labs`), which makes this the **one** clip whose label the export now changes against its audio (the export drops `Téva`, the audio still says it; same row in `03_PARHAF/generated_dataset.jsonl`, `asr_training_target` and `asr_training_source`). Re-synthesize it with the held-back typo clips above (`21_reopen_resynth_typos.py` includes it by default, 60 stems).

Eponym inner capitals (2026-11-17, Claude Code; owner decision 2026-11-17: "fix Mounier-Kuhn in the labels too"). Labels carried `Mounier-kuhn` (11 rows in `text`) next to the correct `Mounier-Kuhn` (3), and the TTS source `Mounier-kühn`. Root cause: `utils/label_conventions.py::normalize_accents` replaced a table hit by its lowercase canonical and only re-capitalised the first letter, so every inner capital of a hyphenated or elided word was lost (`Mounier-Kühn` -> `Mounier-kuhn`, `béta-hCG` -> `Bêta-hcg`); `restore_tts_accents` reuses that function, hence the lowercase `k` in the source too. Fix: `_case_like` keeps acronym-like parts (`hCG`, `HCG`) letter by letter while ordinary parts keep the convention lowercase (`Piperacilline-tazobactam` unchanged), a canonical carrying its own capitals is written as is, and eponym capitals now live in the table: `EPONYM_CASING` in `06_hotfixes/07_build_accent_spellings.py` maps `mounier-kühn` and the umlaut-less `mounier-kuhn` to `Mounier-Kuhn`, the second key being what lets a plain `05_normalize_text.py` rerun repair labels already built. `_tts_accent_table` is keyed lowercase so `Mounier-Kuhn` still restores to `Mounier-Kühn` for the TTS text. Covered by `tests/test_label_conventions.py::test_eponym_inner_capital_kept` (fails on the previous code). Casing is silent, so **no re-synthesis**: the label alone changes. Dry run 2026-11-17 20:51 (Claude Code), `05_normalize_text.py data/NeMO_files/full.jsonl`: 602,776 rows, `text` changed in 11 (all `Mounier-kuhn` -> `Mounier-Kuhn`), `asr_training_source` in 0 (the source keeps what voxtral read). Same mechanism, not fixed yet (owner to decide whether to add them to `EPONYM_CASING`): 10 other eponyms with a lowercased inner part in `text`, e.g. `Gomori-grocott` 22, `Malpighi-gartner` 22, `Lich-gregoir` 12, `Gerstmann-sträussler-scheinker` 12, `Hermansky-pudlak` 11, `Chédiak-higashi` 11, `Boucher-neuhäuser` 11, `Rokitansky-küster-hauser` 4, `Klippel-trénaunay` 4, `Roch-léri` 4 (113 occurrences); `Bêta-hcg` 4 is fixed for future builds but not for built labels, since the built `hcg` no longer says it was `hCG`. `Mounier Kuhn` with a space never occurs in `text`. The stage file `01_dictionnary/generated_dataset.jsonl` still holds 11 `Mounier-kuhn` in `asr_training_target`, and the driver rebuilds the manifests from it every round: apply `05_normalize_text.py --stages --apply` first, then the manifest pass, else the next rebuild brings the bad casing back. Its dry run (2026-11-17 21:00, Claude Code) is much wider than Mounier: `asr_training_target` would change in 5,894 rows of `01_dictionnary`, 779 of `02_drugs`, 672 of `03_PARHAF`, 16 of `04_PARROT`, 7 of `07_acronyms` (7,368 total) while the manifest pass changes only 11, so the stage files lag earlier convention passes; review that diff before `--apply` (owner).

To apply, from `99_hf_release/`: `uv run 05_normalize_text.py` (dry run), then `uv run 05_normalize_text.py --apply`, then `uv run scripts/build_parquet.py`.

Split manifests kept the old text (2026-11-17, found and repaired with Claude Code). After the resynth round, the re-synthesised clips' labels and audio disagreed in every manifest except the two `full.jsonl`: the audio said the corrected word while `test.jsonl`, `train.jsonl`, `val.jsonl`, the `*.down-N` fixtures, the per-source `dictionary/`, `drugs/`, `PARHAF/` files and the derived `train.holdout*` / `train.keep5` / `train.voice.*` / `train.drugs.*` files still carried the typo (`text` and `asr_training_source`). The benchmark fixtures are built from those splits, so a corrected hyp was scored against the stale label. Root cause: `17_reopen_hgvs.py::reopen_chain`, which 21 reuses, rewrote only `full.jsonl` and `PARROT/full.jsonl`, and `REFRESH_MANIFESTS=0` (the right setting for a 60-clip round) skipped the rebuild that would have propagated them. `03_sync_hotfix_results.py` does visit every manifest at the end of the round, but it syncs audio, durations and QC columns, not the label. Fix: the chain now rewrites every file matched by `nemo_manifest.MANIFEST_GLOBS` (all 67 `*.jsonl` under `NeMO_files/`), and `21_reopen_resynth_typos.py --manifests-only [--apply]` rewrites the release manifests alone, leaving the stage `generated_dataset.jsonl` files and the stage-06 state untouched and queueing nothing (test: `tests/test_reopen_resynth_typos.py`). Repair: a dry run changed exactly the rows an independent comparison against `full.jsonl` (by audio basename) flagged, 1,768 rows in 38 files (e.g. `train.jsonl` 259, `test.jsonl` 34, `val.jsonl` 35, `dictionary/full.jsonl` 287, `PARHAF/full.jsonl` 35, `drugs/full.jsonl` 6, `test.dictionary.down-5290` 4, `test.parhaf.down-2500` 6, `test.drugs.down-2059` 1), each giving exactly the `full.jsonl` text; the old lines are kept in `label_fix_backup_ignore_backups/release_20261117_splitsync/` on the experiments drive, then `--manifests-only --apply`; an independent check afterwards found no re-synthesised clip still disagreeing. Not repaired, because they predate this bug and are not label typos: 32 rows in the derived subsets (8 distinct clips) differ from `full.jsonl` in `asr_training_source` only (e.g. `faricimab`, `grazax`, `increlex`), and 11 rows of `train.holdout_arm_ptm` / `train.voice.h_*` / `val.voice.h_frf_itm.down-75` no longer exist in `full.jsonl` (clips dropped after those subsets were cut). No tracked script builds the `train.holdout*` / `train.keep5` / `train.voice.*` subsets; the `*.down-N` and `val.voice.*` ones come from `perso/build_ultimed_val_slices.py` in the NeMo training-scripts repo.

Owner spellings (2026-11-18, Claude Code; owner's list of the same day). Casing, hyphen and silent-letter changes only, so the audio stays and the label alone changes (no re-synthesis, `asr_training_source` untouched). Rules in `utils/label_conventions.py::_SPELLING_RULES` (hormone and virus names) and in the accent table (`OWNER_SPELLING_20261118` in `06_hotfixes/07_build_accent_spellings.py`, mirrored in `utils/accent_spellings.json`), covered by `tests/test_label_conventions.py::test_owner_spellings_20261118`:

- `bêta-hCG` / `Bêta-hCG` for every `bêta-HCG`, `béta-hcg`, `beta HCG`, `bétaHCG` ... (470 occurrences in the labels before), and a bare `HCG` / `hcg` becomes `hCG` (99). A prefixed `B-HCG` is left alone (one PARHAF row, in the release: `full`, `train`, `PARHAF/full`, `PARHAF/train`).
- `SARS-CoV`, `SARS-CoV-1`, `SARS-CoV-2` for the hyphenated and spaced spellings (`Sars-Cov 2`, `SARS-CoV2`, `SARS CoV2`, `sars-cov` ..., 49). `SARS COVID` stays as written (one PARHAF row, in `full`, `val`, `PARHAF/full`, `PARHAF/val`): it names the disease, not the virus.
- `Teralithe` (19, was `Téralithe`), `Sjögren` (2, was `Sjogren`), `Hem-o-lok` (23 hyphenated forms). The unhyphenated `hemoloc`, `Hemolock`, `hémolocks` are left alone; `utils/drug_casing.json` keeps `Téralithe` for the TTS side.
- `méti-S` / `méti-R` mid-sentence (`Meti-S`, `meti-s` ..., 21, decided clip by clip), capitalised only at a sentence start. `métis` and the long forms (`méticilline-sensible`) are untouched.
- Echo compounds: no label change (the labels already follow the hyphen convention, `écho-endoscopie` since 2026-11-16).

Backup before applying: `label_fix_backup_ignore_backups/release_20261118_spelling/` (all 67 manifests, same layout as `NeMO_files/`) and `label_fix_backup_ignore_backups/stages_20261118/` (the five stage `generated_dataset.jsonl`). Applied 2026-11-18 from `99_hf_release/` in two passes (06:59, then 08:00 once verification showed the spaced `SARS CoV2` slipping through and the rule was widened): `uv run 05_normalize_text.py --stages --apply` (`asr_training_target` changed in 01_dictionnary 515, 02_drugs 18, 03_PARHAF 138, PARROT and acronyms 0) then `uv run 05_normalize_text.py --apply`: `text` changed in 671 of the 602,776 rows of `full.jsonl`, 4,388 rows over 44 of the 67 manifests (e.g. `train.jsonl` 542, `test.jsonl` 74, `val.jsonl` 55, `test.dictionary.down-5290` 9, `test.parhaf.down-2500` 6, `val.dictionary.down-600` 2, `test.drugs.down-2059` 1; 1,686 of the rows were in the derived `train.*` subsets removed right after, see below), `asr_training_source` in 0. Checks: a backup-vs-live comparison found equal line counts, valid JSON and no key but `text` changed in any manifest (`asr_training_target` alone in the stage files), and no old form left in the release but the two kept above; `06_hotfixes/21_reopen_resynth_typos.py --manifests-only` (dry run, no `--apply`, 85 min on CPU) rewrote 0 rows in 0 of 67 files, so every split agrees with `full.jsonl`. The benchmark fixtures rebuilt from these manifests (`wer_fixtures_v2_fixed3`) differ from `wer_fixtures_v2_fixed2` on 18 labels (`med_dictionary` 2, `test_dictionary` 9, `test_parhaf` 6, `test_drugs` 1), all `text`. `scripts/build_parquet.py` not run: no v2 parquet exists yet, it is part of the release export.

Derived training subsets removed (2026-11-18, Claude Code; owner approval same day, after the label check above). The 25 `NeMO_files/` subsets no release script builds or reads were deleted: `train.keep5`, `train.holdout4`, `train.holdout_arm_ptm`, the 9 `train.voice.*`, `train.drugs`, `train.drugs.{keep5,holdout4,holdout_arm_ptm}` and the 9 `train.drugs.voice.*` (1,481,656 lines in all; they were cut for the v2 training sweeps and carried copies of `train.jsonl` rows, so every label fix had to be repeated in them). `NeMO_files/` now holds 42 manifests, which is what `nemo_manifest.MANIFEST_GLOBS` matches from here on (the "67" above is the count at the time). `train.drugs.jsonl` was never an official split: the drugs training split is `drugs/train.jsonl`, and `train.drugs.jsonl` (every drugs row of `train.jsonl`) is regenerated by `perso/build_ultimed_val_slices.py` in the NeMo training-scripts repo, whose `perso/training_config.yaml` still names it for drug upsampling, as do historical sweep scripts there. Pre-pass copies of all 25 remain in `label_fix_backup_ignore_backups/release_20261118_spelling/`.

Clip brand `Hem-o-lok` in every spelling (2026-11-18, Claude Code; owner decision of the same day, which overrides the "left alone" above). Every spelling of the surgical clip becomes `Hem-o-lok`, a plural keeps its s (`hémolocks` becomes `Hem-o-loks`), whole words only. Label only: no re-synthesis, `asr_training_source` keeps what the TTS read. It is a rule in `utils/label_conventions.py::_SPELLING_RULES` rather than accent-table rows, since the table keys one word and `hémo lock` is two; the six `hem-o-lok` / `hémoloc` ... rows left `utils/accent_spellings.json`, and `07_build_accent_spellings.py` now keeps those families out of the table (`KEEP`). Covered by `tests/test_label_conventions.py::test_hem_o_lok_every_spelling` (variants, plurals, elisions, the TTS side, and near words that must not move: `hémolyse`, `hémoglobine`, `Hemoleven`, `hémo locale`). Before, `full.jsonl` held 123 occurrences, 26 already `Hem-o-lok` and 97 to change: `hem-o-lock` 39, `Hemoloc` 10, `Hémolocs` 7, `hémolocks` 6, `Hemolock` 5, `Hemolok` 4, `hemolock` 4, `hemolok` 4, `Hem-O-Lock` 4, `Hémolocks` 4, `hemoloc` 3, `Hem-o-Lock` 2, `hémo lock` 2, `hemoloK` 1, `hémolocs` 1, `Hémoloks` 1. Every context is the clip (cystic duct, arteries, ureter), nothing was doubtful. `trois hémo lock violets` has no s in the label, so it stays singular. After: `Hem-o-lok` 104, `Hem-o-loks` 19, no other spelling in any of the 42 manifests. All in PARHAF. `B-HCG` untouched.

Backup before applying: `label_fix_backup_ignore_backups/release_20261118_hemolok/` (the 10 changed manifests, `NeMO_files/` layout) and `stages_20261118_hemolok/03_PARHAF/generated_dataset.jsonl`. Applied from `99_hf_release/`: `uv run 05_normalize_text.py --stages --apply` changed `asr_training_target` in 61 rows of 03_PARHAF (0 elsewhere), then `uv run 05_normalize_text.py --apply` changed `text` in 61 rows of `full.jsonl`, `PARHAF/full.jsonl`, 41 of `train.jsonl` and `PARHAF/train.jsonl`, 12 of `val.jsonl` and `PARHAF/val.jsonl`, 8 of `test.jsonl` and `PARHAF/test.jsonl`, 5 of `test.parhaf.down-2500` and 1 of `val.parhaf.down-300`. Backup-vs-live check: line counts equal, JSON valid, only `text` (manifests) or `asr_training_target` (stage) differs, the 97 word changes are all the clip, 0 old forms left. Split sync dry run (`06_hotfixes/21_reopen_resynth_typos.py --manifests-only`, no `--apply`): 0 manifest rows to rewrite in 0 of 42 files, so the splits agree. The benchmark fixtures rebuilt from these manifests (`wer_fixtures_v2_fixed4`) differ from `wer_fixtures_v2_fixed3` on 6 labels (`med_parhaf` 1, `test_parhaf` 5), all `text`. `scripts/build_parquet.py` not run, as above.

Eponym casing `Rio Branco` / `Martins Branco` (2026-11-19, Claude Code; owner decision of the same day). Two rules in `utils/label_conventions.py::_SPELLING_RULES` (the Hem-o-lok route: the names are two words and `branco` alone must not move), whole two-word names only, casing only, so no re-synthesis and no new STT; covered by `tests/test_label_conventions.py::test_rio_branco_martins_branco_casing`. The survey found the brief's 4 `rio branco` and 1 `Rio branco` only in STT output (`parakeet_stt` 2 + 1, `whisper_stt` / `stt_transcript` 1), never in a label: `full.jsonl` labels already held 33 `Rio Branco` (29 + 3 with a comma + 1 with a period) and nothing to fix, so that rule is a guard for future generations. The one label that changed is the PARHAF signatory `Gerard Rufo martins branco` -> `Martins Branco` (clip `027129_0000_medecine_geriatrique_00163_t0_c10.flac`). Left alone and reported to the owner: `selon Rio-Branco.` (1 PARHAF label, hyphenated). Every other `branco` string (`branco-alvéolaire`, `brancopulmonaire`, `Brancodual`, `Brancotual`, `Martens Branco` ...) sits only in STT transcript fields, not in labels.

Backup before applying: `label_fix_backup_ignore_backups/release_20261119_branco/` (`full`, `train`, `PARHAF/full`, `PARHAF/train`) and `stages_20261119_branco/` (`03_PARHAF/generated_dataset.jsonl`, plus `06_hotfixes/improved/full.stt.jsonl`). Applied from `99_hf_release/` after clean dry runs: `uv run 05_normalize_text.py --stages --apply` changed `asr_training_target` in 1 row of 03_PARHAF (0 elsewhere), then `uv run 05_normalize_text.py --apply` changed `text` in 1 row each of `full.jsonl`, `train.jsonl`, `PARHAF/full.jsonl`, `PARHAF/train.jsonl` (`asr_training_source` 0, no other manifest written). Backup-vs-live check: line counts equal (602,776 / 485,946 / 43,384 / 32,543 / 43,455), only `text` (manifests) or `asr_training_target` (stage) differs, on that one clip; `full.jsonl` labels hold 0 `rio branco`, 0 `Rio branco`, 0 `martins branco`. `transcriptions`, `cer`, `stt_transcript` untouched, no sync run. Stage-06 state NOT updated: running 05 on `improved/full.stt.jsonl` as an explicit manifest (dry run) would fix the one `text` but also rewrite `asr_training_source` (the CER reference) in 13,592 state rows (9 in `improved_parrot`), so it was not applied. Consequence: that state row still says `martins branco`, and `06_hotfixes/01_compute_stt.py` keys a row on audio path AND `text`, so the next driver round will drop this one row's stored readings and re-run STT on it (1 clip) unless its state `text` is aligned first. The state still has 0 rows with empty `transcriptions`.

Misspelled dictionary headwords unified (2026-11-20, Claude Code; owner decision of the same day). The source dictionary carried misspelled headwords, and every clip generated from one repeated the misspelling. The silent ones (same sound, so label only, `asr_training_source` keeps what the TTS read) are the table `OWNER_SPELLING_20261120` in `06_hotfixes/07_build_accent_spellings.py`: `Cokett` to `Cockett`, `Abrikossof` to `Abrikossoff`, `Rosenmüler` to `Rosenmüller`, `Grynfelt` to `Grynfeltt`, `Lachapèle` to `Lachapelle`, `Schwartzman` to `Shwartzman` (also `Sanarelli-Shwartzman`), `Kußmaul` to `Kussmaul` (also `Kussmaul-Maier`), `Thornwaldt` to `Tornwaldt`, `idradénome` to `hidradénome`, `hydrocystome` to `hidrocystome`, `médulaire` to `médullaire`, `souscutanée` to `sous-cutanée`, `falsiforme` to `falciforme`, `correctopie` to `corectopie`, `callidine` to `kallidine`, `karyophérine(s)` to `caryophérine(s)`, `cholestyramine` to `colestyramine`. One rule in `utils/label_conventions.py::_SPELLING_RULES` writes `méthyl aminolévulinate` without the e. Covered by `tests/test_label_conventions.py::test_owner_spellings_20261120`. Left as they are: `peroxisome` (all 11 hits are the English `peroxisome proliferator-activated receptor`), `Herying` and `dyscentromérique` (open questions to the owner).

The sound-changing ones are held in `RESYNTH_TYPOS` until their clips are redrawn, per the 2026-11-16 rule: `bégnine(s)` to `bénigne(s)`, `pléiomorphe(s)` to `pléomorphe(s)`, `lipocalicine` / `lipocalycine` to `lipocaline`, `damètre` to `diamètre`, `dystropnie` to `dystrophie`, `déferrioxamine` to `desferrioxamine`, `Castelman` to `Castleman`, `jiroveci` to `jirovecii` (held as doubtful), `Soemmering` / `Soemmerring` to `Sömmerring`, `Jadassohn-Levandovsky` to `Jadassohn-Lewandowsky`, `gangraenosum` to `gangrenosum`.

The headwords were also fixed at the source, in the four `01_dictionnary/original_dictionnary*.jsonl` files (44 / 43 / 44 / 43 lines, `term` only, plus `definition` and `examples` for lipocaline; line counts unchanged, backup in `label_fix_backup_ignore_backups/dico_20261120/`). `01_dictionnary/generated_dataset.jsonl` keeps its `term` as it was: that field drives the stage-05 file names and the voice-group keys, so a clip of `Cockett` is still named `..._cokett.flac`.

Backup before applying: `label_fix_backup_ignore_backups/release_20261120_unify/` (the 16 changed manifests, `NeMO_files/` layout) and `stages_20261120_unify/` (`01_dictionnary` and `02_drugs` `generated_dataset.jsonl`, plus `06_hotfixes/improved/full.stt.jsonl` and `full.improved.jsonl`). Applied from `99_hf_release/` after clean dry runs: `uv run 05_normalize_text.py --stages --apply` changed `asr_training_target` in 274 rows of 01_dictionnary and 12 of 02_drugs (0 elsewhere), then `uv run 05_normalize_text.py --apply` changed `text` in 286 rows of `full.jsonl` (`train` 230, `test` 30, `val` 26, `dictionary/full` 274, `dictionary/train` 222, `dictionary/test` 28, `dictionary/val` 24, `drugs/full` 12, `drugs/train` 8, `drugs/test` 2, `drugs/val` 2, `train.drugs` 8, `test.drugs.down-2059` 2, `test.dictionary.down-5290` 1, `val.drugs.down-300` 1), `asr_training_source` in 0. Backup-vs-live check: line counts equal in all 16 manifests and both stage files, only `text` (manifests) or `asr_training_target` (stages) differs, 0 old forms left in the `full.jsonl` labels.

Stage-06 state aligned this time. `06_hotfixes/01_compute_stt.py` keys a state row on its audio path AND its `text`: a row whose `text` no longer equals the manifest's loses its stored readings, and under `STEMS_FROM` it is not transcribed again, so the turbo-keyed sync then writes a null `cer` and `stt_transcript`. That is how the PARHAF row `027129_0000_medecine_geriatrique_00163_t0_c10` (the 2026-11-19 Branco fix above) ended up null. So after this apply the `text` of the 286 rows was copied from `full.jsonl` into `06_hotfixes/improved/full.stt.jsonl` (0 mismatches before the apply, 286 after it, 0 after the alignment; nothing else in the rows touched, which is safe because the readings are scored against `asr_training_source`). Any later silent label fix must do the same before the next driver round.

Held half redrawn the same day (2026-11-20, Claude Code). `06_hotfixes/21_reopen_resynth_typos.py --apply` rewrote the 15 held spellings in 182 `01_dictionnary` clips (stage file, 729 rows in 9 release manifests, stage-06 state marked `pending_tts`), then one driver round (`MODE=alternate`, `STEMS_FROM` those 182 stems plus the PARHAF clip `027129_0000_medecine_geriatrique_00163_t0_c10` that had lost its reading, `REFRESH_MANIFESTS=0 ACOUSTIC_SCAN=0`, 08:05 to 08:34) redrew them from the fixed text. Result: 182 of 182 closed as `improved`, turbo CER median 0.012, max 0.058; the PARHAF row has `cer` 0.019 again; 0 rows dropped, `full.jsonl` still 602,776 rows; state and manifest `text` agree on all 183; no old form is left in those labels. The 15 entries then moved from `RESYNTH_TYPOS` (empty again) to `RESYNTHESIZED_TYPOS`, so `utils/accent_spellings.json` holds them (686 rewrites) as a guard against the typo coming back.

Owner brands and casing (2026-11-20, Claude Code). Silent label fixes (casing, hyphens, gluing; the audio is unchanged), encoded in `utils/label_conventions.py::_SPELLING_RULES` and covered by `tests/test_label_conventions.py::test_owner_brands_and_casing_20261120`: `Diffu-K` in every spelling (`DiffuK`, `Diffuk`, `diffu-K`, `diffU-K`, `diffuK`, `diffuk`), `Gonal-F` (`gonal-F`, `Gonal-f`), `Di-Hydan`, `Co-Renitec`, `QuantiFERON` to `Quantiferon`, mid-sentence `Ostéogenèse Imparfaite` lowercased, `cranio caudale` to `craniocaudale`, `re-ascension` to `réascension`, `re-épithélialisation` to `réépithélialisation`, mid-sentence `Esomeprazole` lowercased, `Rio-Branco` to `Rio Branco` (reverses the 2026-11-19 exception above), `anti-Thyroglobuline` to `antithyroglobuline`. Left alone on purpose: `Herpes gestationis` / `zoster` / `simplex`, `nitro-imidazole`, `Herying`, `dyscentromérique`, `NEO-Mercazole`, `Klean-PREP`, `CHIBROPROSCAR`, `Co-Aprovel`, `Zyma-D`, `Co-Doliprane`, `X-Geva`, `nivolumab-IPILUMUMAB`, `GAZYVARO-vénétoclax`. Separately, the drug sources spelled NALBUFINE (`02_drugs/drugs_freq_dosages.jsonl`, `drugs_frequency_2025.jsonl`, the TODO in `sources/drugs_frequency/02_combine_datasets.py`); no label carried it. Backup before applying: `label_fix_backup_ignore_backups/release_2026-11-20_brands/` (the 21 changed manifests, `NeMO_files/` layout) and `stages_2026-11-20_brands/` (`01_dictionnary`, `02_drugs`, `03_PARHAF` `generated_dataset.jsonl`, plus `06_hotfixes/improved/full.stt.jsonl` and `full.improved.jsonl`). Applied from `99_hf_release/` after clean dry runs (identical counts): `uv run 05_normalize_text.py --stages --apply` changed `asr_training_target` in 66 rows (01_dictionnary 8, 02_drugs 19, 03_PARHAF 39, 0 in PARROT and acronyms), then `uv run 05_normalize_text.py --apply` changed `text` in 66 rows of `full.jsonl` (`asr_training_source` 0), and in `train` 47, `val` 9, `test` 10, `train.drugs` 15, `test.drugs.down-2059` 3, `test.dictionary.down-5290` 1, `test.parhaf.down-2500` 1, `val.parhaf.down-300` 1, `PARHAF/{full,train,val,test}` 39 / 26 / 7 / 6, `dictionary/{full,train,val,test}` 8 / 6 / 1 / 1, `drugs/{full,train,val,test}` 19 / 15 / 1 / 3. Per form: Diffu forms 28, `gonal-F` 7 + `Gonal-f` 1, `DI-Hydan` 6, `CO-Renitec` 6, `QuantiFERON` 3, `d'Ostéogenèse Imparfaite` 2, `Esomeprazole` 8 (the plan counted only the 2 elided ones; the other 6 are bare mid-sentence occurrences such as `Il prend Esomeprazole 30 milligrammes`), and one each for `cranio caudale`, `re-ascension`, `re-épithélialisation`, `Rio-Branco`, `anti-Thyroglobuline`. Backup-vs-live check: line counts equal, valid JSON, only `text` (manifests) or `asr_training_target` (stages) differs, 0 old forms left in the `full.jsonl` labels. A second dry run of both modes changes 0 rows, and `06_hotfixes/21_reopen_resynth_typos.py --manifests-only` (dry run) has nothing held, so nothing to resync. Stage-06 state: `uv run 06_hotfixes/22_align_state_text.py 99_hf_release/data/NeMO_files/full.jsonl 06_hotfixes/improved/full.stt.jsonl` found 66 `text` mismatches, 51 alignable; `--apply` copied those 51 (re-run: 0 alignable, 51 lines differ from the backup, line count unchanged). The other 15 are left as they are: their state `asr_training_source` already predates the manifest's (`un` where the manifest writes `1`, like 18,164 state rows in all, none of them with a `cer` or `stt_transcript`), so the script skips them by design. `improved_parrot` had 0 mismatches. Still pending: the benchmark fixtures are not rebuilt and `scripts/build_parquet.py` has not been run (both deferred to the release export).

---

# 07_acronyms

`wikipedia_acronyms.csv` (columns `ACRONYM,MEANING`) is scraped from the French
Wikipedia article
[Liste d'abréviations en santé](https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9),
one row per `* SIGLE : signification` entry across its A-Z (plus Symboles /
Chiffres / Lettres grecques) sections. The wikitext was fetched via the MediaWiki
`action=raw` API and its markup (links, templates, `<ref>` tags, italics) stripped
with `mwparserfromhell`. Wikipedia text is licensed CC BY-SA; credit the article
if this list ships in the release. Built with Claude Code.
