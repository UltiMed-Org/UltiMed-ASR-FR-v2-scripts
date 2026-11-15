
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

The rewrites are `HAND_REWRITES` in `06_hotfixes/07_build_accent_spellings.py`, synced by hand into `utils/accent_spellings.json` (the audit TSV the builder reads is not committed; `tests/test_label_variant_audit.py` checks the two match); the `estr-` rule is in `_SPELLING_RULES` of `utils/label_conventions.py`. Dry run of the new conventions against the exported labels (602,776 clips, all splits, token changes / clips touched):

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

Found by scanning the `text` of `99_hf_release/data/NeMO_files/full.jsonl` (602,776 rows) for any `RESYNTH_TYPOS` key as a whole word, case-insensitive (Python `re`, `(?<!\w)(key1|key2|...)(?!\w)`; `grep -iE` also hits non-text fields). They also sit in the split files (`train.jsonl`, `test.jsonl`, ...). The `Actisénan` clip is accented, so the old `actisenan` key never matched it anyway. The "TTS typo guard" (`_tts_accent_table`) only stops a typo entry from being inverted into a TTS respelling; it does not protect these clips, which is why they are held back.

To re-synthesize them with the existing TTS stage (not done yet):

1. Rewrite each clip's text with `label_conventions.normalize_accents(text, RESYNTH_TYPOS)` (it also elides the article, `le cestradiol` -> `l'estradiol`) in its source `generated_dataset.jsonl` (`01_dictionnary/`, `02_drugs/` or `03_PARHAF/`; `asr_training_target` and `asr_training_source`), in the release manifests the rounds read (the `HOTFIX_SOURCES` targets' `full.jsonl`, see `99_hf_release/03_sync_hotfix_results.py`) and in the stage-06 state file, marking the row `pending_tts`. `06_hotfixes/17_reopen_hgvs.py` does exactly this chain for HGVS; TODO: generalize it to take the rewrite (e.g. `RESYNTH_TYPOS`) rather than copying it.
2. Queue the 59 stems: `uv run 06_hotfixes/19_queue_stems.py STEMS.txt --apply`.
3. Redraw them: `sudo -E STEMS_FROM=STEMS.txt ./driver.sh` (from `06_hotfixes/`).
4. Merge `RESYNTH_TYPOS` into `HAND_REWRITES`, add the entries to `utils/accent_spellings.json`, and drop the hold-back test.
5. Export as below.

Generic lab names (`strip_generic_labs`, 2026-11-16 check): **0 clips** have audio saying a lab name that the export would drop from the label. `strip_generic_labs(x) != x` holds for no `text` or `asr_training_source` in `99_hf_release/data/NeMO_files/full.jsonl`, nor for any `asr_training_target` or `asr_training_source` in the `01_dictionnary`, `02_drugs` and `03_PARHAF` `generated_dataset.jsonl` (`04_PARROT` and `07_acronyms` carry no lab string), most likely because those sources were redrawn after the lab rule. Query: `grep -E "Arrow|Biogaran|Cristers|Evolugen|Mylan|Sandoz|Substipharm|Teva|Viatris|Zentiva|Zydus|Almus|Ratiopharm|Qualimed|Krka|Ranbaxy" full.jsonl` gives 27 lines, then keep those where `strip_generic_labs` changes the text: none. The 27 are person names (`Mylan X`, `Monsieur Teva`, `Mylane`), the `Tevagrastim` brand, and one edge case, `PARHAF/011560_0000_chirurgie_viscerale_00018_t0_c1`, whose text and source both say `Simvastatine Téva`: the accented `Téva` is not in `_LAB_RE`, so label and audio still agree, but the rule would miss that spelling in a new source.

To apply, from `99_hf_release/`: `uv run 05_normalize_text.py` (dry run), then `uv run 05_normalize_text.py --apply`, then `uv run scripts/build_parquet.py`.

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
