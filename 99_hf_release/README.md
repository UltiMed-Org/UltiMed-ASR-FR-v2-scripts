---
license:
    - cc-by-4.0
    - cc-by-nc-sa-4.0
    # A YAML list of two licences is intentional and renders fine on the HF Hub
    # (verified live): cc-by-4.0 covers the main corpus (dictionary + drugs + PARHAF + acronyms),
    # cc-by-nc-sa-4.0 covers the separate PARROT radiology subset. Keep both entries;
    # do NOT collapse to a single value or to `license: other`. See "Licensing" below.
language:
  - fr
task_categories:
  - automatic-speech-recognition
pretty_name: "UltiMed-ASR-FR-v2"
size_categories:
  - 100K<n<1M
source_datasets:
  # PARHAF is the only upstream corpus that lives on the Hub, so it is the only one
  # worth a Hub id here. The other sources (Wiktionnaire, BDPM / Open Medic /
  # Retrocedam, Wikipedia, PARROT) are off-Hub and stay linked in "Sources" below.
  - HealthDataHub/PARHAF
tags:
  - medical
  - french
  - asr
  - speech
  - synthetic-speech
  - text-to-speech
  - voxtral
  - parakeet
  - nemo
  - parrot
  - parhaf
  - healthdatahub
  - radiology
  - surgery
  - anatomy
  - drugs
  - acronyms
  - technical
# The whole corpus ships as sharded Parquet with the FLAC bytes embedded, built by
# scripts/build_parquet.py, one subset per source. The per-subset licence is encoded
# in the config (subset) name with underscores (e.g. dictionary_CC_BY_4.0 and
# parrot_CC_BY-NC-SA_4.0), so it shows in the viewer's subset dropdown. Config names
# must be valid identifiers (letters / digits / _ / - / .): spaces and parentheses
# make the HF dataset viewer error out, but underscores, hyphens and dots are fine.
# The HF Data Viewer and
# load_dataset(...) work on the ENTIRE dataset, not a sample. Declaring configs
# also disables HF's automatic data-file detection, so the split globs below are
# authoritative. To rebuild the NeMo training layout (loose FLAC or tarred shards)
# from these Parquets, see scripts/parquet_to_nemo.py and "Loading and training".
configs:
  - config_name: dictionary_CC_BY_4.0
    default: true
    data_files:
      - split: train
        path: dictionary/train-*.parquet
      - split: val
        path: dictionary/val-*.parquet
      - split: test
        path: dictionary/test-*.parquet
  - config_name: drugs_CC_BY_4.0
    data_files:
      - split: train
        path: drugs/train-*.parquet
      - split: val
        path: drugs/val-*.parquet
      - split: test
        path: drugs/test-*.parquet
  - config_name: parhaf_CC_BY_4.0
    data_files:
      - split: train
        path: parhaf/train-*.parquet
      - split: val
        path: parhaf/val-*.parquet
      - split: test
        path: parhaf/test-*.parquet
  - config_name: acronyms_CC_BY_4.0
    data_files:
      - split: train
        path: acronyms/train-*.parquet
      - split: val
        path: acronyms/val-*.parquet
      - split: test
        path: acronyms/test-*.parquet
  - config_name: parrot_CC_BY-NC-SA_4.0   # non-commercial, eval-only, separate subset
    data_files:
      - split: test
        path: parrot/test-*.parquet
---

# UltiMed-ASR-FR-v2

> [!IMPORTANT]
> **Want faster improvements?** This project is entirely self-funded on my minimum-wage salary, and every training run competes for a single consumer GPU. If you or your organisation can donate an **RTX 5090**, or the money to buy one second-hand, it would directly speed up the next versions of the dataset and the fine-tuned models. Reach out via [olicorne.org](https://olicorne.org) or open a discussion on this page.

> [!NOTE]
> **The UltiMed effort has three parts:**
> - the **dataset** (this repository): [Olicorne/UltiMed-ASR-FR-v2](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2): 3,300+ hours of French medical dictation, to train or evaluate ASR models;
> - the **finetuned model**: [Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx): parakeet-ultra finetuned on that dataset;
> - the **regex fixes**: [UltiMed-ASR-FR-v2-regex-fixes](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-regex-fixes): drug-name and medical-term fix rules that should catch many of the remaining errors at almost no cost (text substitution after decoding), and also help models that were not finetuned.
>
> Every UltiMed repository, dataset and model is listed on [UltiMed-Org](https://github.com/UltiMed-Org).

*A large, fully documented French medical speech dataset for evaluating or training models, plus an open recipe to rebuild it in any language or topic.*

## Contents

- [Changelog](#changelog)
- [What is UltiMed](#what-is-ultimed)
- [Why I made UltiMed](#why-i-made-ultimed)
- [Who made UltiMed](#who-made-ultimed)
- [Quick start (TL;DR)](#quick-start-tldr)
- [How I made UltiMed](#how-i-made-ultimed)
  - [Sources](#sources)
  - [Breakdown by source](#breakdown-by-source)
  - [Text generation (LLM)](#text-generation-llm)
  - [Audio synthesis (TTS)](#audio-synthesis-tts)
  - [Hardware and conditions](#hardware-and-conditions)
  - [Repository layout on the Hub](#repository-layout-on-the-hub)
  - [Row format and the two text fields](#row-format-and-the-two-text-fields)
  - [Loading and training](#loading-and-training)
  - [Splits (combined release-wide re-split)](#splits-combined-release-wide-re-split)
- [Evaluation as a standardized benchmark](#evaluation-as-a-standardized-benchmark)
  - [Results](#results)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Audio quality](#audio-quality)
  - [The five voices, and what synthetic audio still costs](#the-five-voices-and-what-synthetic-audio-still-costs)
  - [Known limitations](#known-limitations)
  - [Privacy and PII](#privacy-and-pii)
- [Licensing](#licensing)
  - [Main corpus: CC BY 4.0](#main-corpus-cc-by-40)
  - [PARROT subset: CC BY-NC-SA 4.0](#parrot-subset-cc-by-nc-sa-40)
- [Citations](#citations)
  - [UltiMed-ASR-FR-v2](#ultimed-asr-fr-v2)
  - [PARHAF (source, please cite)](#parhaf-source-please-cite)
  - [PARROT (source, please cite)](#parrot-source-please-cite)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Changelog

<details>
<summary><b>Expand</b>: every release, from v1.0 to the latest.</summary>

- **v2** (2026-11-21): **all the audio is new**, read by **five French voices** instead of one (two shipped Voxtral voices plus three accented hybrids, 3 female and 2 male, see [the five voices](#the-five-voices-and-what-synthetic-audio-still-costs)). New `voice` column saying which of the five read each clip. The splits follow v1.3's group by group, but **the labels moved on too**, so WER on v2 `test` is **not** directly comparable with v1.3 numbers: on top of v1.3, numbers said as words are written in digits (`deux mille vingt-quatre` -> `2024`, `18 000` glued to `18000`, Roman numerals to digits outside staging words), HGVS variants read `c.1163C vers T` instead of `c.1163C>T` (`>` is not in the Parakeet vocabulary; 446 labels), drug INNs follow one accent convention, compounds are glued (`postopératoire`), generic lab boilerplate was stripped and redrawn, and PARHAF/PARROT surnames and ages were diversified. The authoritative list of every label convention is the docstring of [`utils/label_conventions.py`](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts/blob/main/utils/label_conventions.py) in the scripts repo. Synthesis moved to `cfg_alpha` **2.0** (measured band 1.2 to 2.0, see [Audio synthesis](#audio-synthesis-tts)) partway through the run, so the first-pass clips drawn earlier sit at 1.3. Every quality-control column was recomputed on the v2 audio, and two more ASR judges were added (see [Row format](#row-format-and-the-two-text-fields)). The quality pass replaced **16,593** clips (16,458 in the main corpus, 135 in PARROT), and 512 clips that no redraw could fix were dropped (295 flagged by ASR, 163 exhausted, 54 drawn with guidance off). 72 clips v1 had dropped for bad **audio** (their text was fine) are back. Main corpus now **601,228 clips / 3,347.6 h / 266.02 GB** (train 485,946, val 57,692, test 57,590), PARROT **1,548 clips / 11.1 h / 900.1 MB**. v1's single-voice audio is **not** part of this release, which is published as `Olicorne/UltiMed-ASR-FR-v2`; v1 stays up, unchanged, at [Olicorne/UltiMed-ASR-FR-v1](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v1).
- **v1.3** (2026-09-28): more label conventions, no new audio, 64 clips dropped (63,219 then 48,041 clip labels changed). A spelled number before a unit of measure or a duration is now written in digits, as the corpus majority already did (62k `milligrammes` after digits vs 1.9k after words): `quatre milligrammes` -> `4 milligrammes`, `zéro virgule vingt-cinq milligramme` -> `0,25 milligramme`, `deux cent cinquante millilitres` -> `250 millilitres`, `cinq jours sur sept` -> `5 jours sur 7`. Most counts stay spelled (`deux comprimés`), and so does a lone `un` / `une` (`une heure`, `une fois par jour`). `petit déjeuner` is hyphenated (`petit-déjeuner`, 213 vs 35). Words spelled two ways that sound the same now have one spelling, found by transcribing val/test with a fine-tuned Parakeet and ranking the diffs: the `œ` ligature is written `oe` everywhere (`coeur`, `oedème`, `oeil`, `oesophage`; `oeil` was already the majority, 10.7k vs 1.5k), `aiguë` (28.4k) over `aigüe` (1.4k), `compte rendu` (81k) over `compte-rendu` (2k), `bétabloquant` over `bêtabloquant` / `bêta-bloquant`. Suture gauges read `Vicryl 3-0` (was also `Vicryl trois zéro` and `Vicryl 3 zéro`), in the label only: the TTS source keeps the words the voice said. A decimal half in digits becomes a decimal (`3 virgule 5 mégahertz` -> `3,5 mégahertz`), and `grays`, `hertz` and `teslas` join the units written after digits. A second pass (48,041 clip labels changed), from a Parakeet sweep of val/test/PARROT and half of train, ranking the diffs where Parakeet and the stage-06 Whisper transcript agree against the label: `fois` and `séances` counts, the head of a range and a score out of 10 are written in digits (`deux fois par jour` -> `2 fois par jour`, 31.7k labels spelled it and 8.9k did not; `deux à trois jours` -> `2 à 3 jours`; `deux sur dix` -> `2 sur 10`); a staging number is written in digits after `stade`, `grade`, `type`, `classe`, `palier`, `niveau` and `NYHA` (`stade IIIb` -> `stade 3b`, `palier deux` -> `palier 2`; the corpus mixed Roman numerals, digits and words), while Roman numerals elsewhere stay (`angiotensine II`); `anévrisme` (2.7k) over `anévrysme` (1.0k), `urètre` / `urétral` over `urèthre` / `uréthral`; a split prefix is glued (`extra hépatiques` -> `extra-hépatiques`, `multi lithiasique` -> `multilithiasique`); and a sentence-initial `A` before an infinitive gets its accent (`A surveiller` -> `À surveiller`, 8.9k labels already wrote `À`). The pipeline also learnt to repair cp1252 mojibake (a C1 control character such as U+009C standing for `œ`), found in a hand-made evaluation set but in no UltiMed clip. Dropped 63 clips whose audio does not say their label, found by that Parakeet sweep (it covered val, test, PARROT and a third of train) and invisible to the stage-06 Whisper QC: 49 where the voice babbles a few invented words BEFORE the sentence (`John Ritteny Dudd, transmission IDE...`), which Whisper silently skips, and 14 where it skipped a phrase (mostly inside long lab-value lists), confirmed by the Whisper transcript missing it too; the list ships in the GitHub repo history as `99_hf_release/asr_flagged_clips.jsonl` (the file now holds v2's own, shorter list). One more test clip was dropped because its relabelled text became identical to a training text. Main corpus now 601,264 clips / 3,104.4 h / 254.61 GB (train 485,962, val 57,700, test 57,602). val/test labels changed, so WER on v1.3 is not directly comparable to v1.2 numbers.
- **v1.2** (2026-09-27): label conventions, no new audio. Every `text` label and its `asr_training_source` now follow one written form per spoken thing, chosen as the corpus majority, applied to both so audio and label still say the same words (18,777 clips changed). Drug names get one spelling each: the ALLCAPS copies of the drug database are recased (`DOLIPRANE` -> `Doliprane`, `PARACETAMOL` -> `paracétamol`, `UVEDOSE` -> `Uvedose`) and accent variants harmonized (`Primperan` -> `Primpéran`); substances take the ANSM accents, brands the spelling the corpus uses most (a brand never written other than ALLCAPS gets Title case with no accent, since nothing tells its accents), and acronyms stay ALLCAPS (`LP`, `AINS`). Titles before a name are written out (`M.` -> `Monsieur`, `Mme` -> `Madame`, `Mlle` -> `Mademoiselle`, `Dr` / `Pr` -> `docteur` / `professeur`, capitalized at a sentence start), with `Monsieur` / `Madame` capitalized before a name and left lowercase otherwise (`le monsieur du lit 4`). Dates use digits (`le quinze mars deux mille vingt` -> `le 15 mars 2020`, `premier janvier` -> `1er janvier`), and clock times read `14 heures 30` (v1.1 mixed `14h30`, `14 h 30` and the spoken form). Dropped the 2 clips whose label held an ellipsis (truncated or anonymized text). Main corpus now 601,328 clips / 3,105.0 h / 254.65 GB. The per-source `<source>/{train,val,test}` NeMo manifests now match the release-wide split (the Parquet always used the release-wide one). val/test labels changed, so WER on v1.2 is not directly comparable to v1.1 numbers.
- **v1.1** (2026-09-25): data fixes, no new audio. Dropped the 6 `qc_status: exhausted` clips (they failed QC and no redraw fixed them, so their audio does not match their label; v1.0 shipped them by mistake) and 2 PARHAF val/test clips whose sentence also appears in a training clip. Main corpus is now 601,330 clips / 3,105.0 h / 254.65 GB. Normalized the `text` labels: every percentage is written `%` (v1.0 mixed `pour cent`, `pourcent` and `%`; a "per hundred X" such as `pour cent millilitres` is kept), whitespace runs including 6 raw newlines collapse to one space, and the 4 characters the Parakeet tokenizer cannot represent as written are replaced by their NFKC form (`CO₂` -> `CO2`) or dropped (`™`). `asr_training_source` uses the one-word `pourcent` everywhere (those clips were read from `pour cent` or `%`, spoken identically). The pipeline's generators now write `%` in the label directly, so a rebuild reproduces this. The val/test labels changed with it, so WER on v1.1 is not directly comparable with v1.0 numbers.
- **v1.0** (2026-08-19): initial release. Main corpus 601,338 clips / 3,105.0 h / 254.66 GB (dictionary + drugs + PARHAF + acronyms, CC BY 4.0), plus a separate **test-only** 1,549-clip / 10.1 h PARROT radiology subset (CC BY-NC-SA 4.0). [Voxtral][voxtral] `fr_female`.

</details>

## What is UltiMed

A 3,300+ hours corpus of **dictation-style French medical sentences** spoken by a high-quality open-weights TTS model. As far as I know, the largest French medical speech dataset of its kind. Each clip pairs a written French medical transcript with its synthesized speech, so the audio trains and evaluates ASR on the technical vocabulary (anatomy, pathology, drug names, clinical phrasing) that general-purpose ASR handles badly. A small radiology subset from [PARROT][parrot-paper] ships separately under a different licence (see [Licensing](#licensing)).

| | |
|---|---|
| Language | French (`fr`) |
| Task | Automatic Speech Recognition (training **and** evaluation) |
| Audio | **24 kHz, mono, FLAC (PCM 16-bit), straight out of [Voxtral][voxtral]**, unprocessed (see [Audio synthesis](#audio-synthesis-tts)) |
| Packaging | **Sharded Parquet with embedded FLAC** (one subset per source, split into `<source>/<split>-*.parquet`); rebuild NeMo loose/tarred data with `scripts/parquet_to_nemo.py`; see [Repository layout](#repository-layout-on-the-hub) |
| Clips | **601,228** (main corpus) + **1,548** (PARROT subset) |
| Duration | **3,347.6 h** (main corpus) + **11.1 h** (PARROT subset) |
| On-disk size | **266.02 GB** (main corpus) + **900.1 MB** (PARROT subset), FLAC |
| Splits | train / val / test, target 80 / 10 / 10, balanced by **audio duration** |
| Sources | public French medical dictionaries (Wiktionary + others), French drug names, PARHAF, common medical acronyms (Wikipedia-sourced, hand-filtered) (main corpus, CC BY 4.0); PARROT radiology (separate CC BY-NC-SA 4.0, test-only subset) |
| Voices | **five French voices**, 3 female and 2 male (2 shipped [Voxtral][voxtral] voices + 3 accented hybrids I built), picked by hash, per clip on the term sources and per document on PARHAF / PARROT, see [the five voices](#the-five-voices-and-what-synthetic-audio-still-costs) |

Numbers regenerate from the source NeMo manifests (the working master the Parquet is built from) with:

```bash
uv run scripts/get_statistics.py --source combined   # the actual train/val/test
```

## Why I made UltiMed

As a resident psychiatrist and developer, I think medicine vastly underuses large-scale data, and a big part of the blame is **input**: getting information in and out of a computer is painfully slow for clinicians. Speech is high-bandwidth and already how clinicians pass much information to each other, yet ASR is barely used in medicine because general-purpose models mis-transcribe exactly the technical vocabulary clinical work is made of. A good medical ASR model is a concrete first step, and nobody seemed to be building it, so I did.

It is also a **general recipe**: a Github repository contains all the useful code needed to easily replicate this on new domain or languages: [github repo](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts).

A finetuned release of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for French medical ASR is available: [Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx)

## Who made UltiMed

Made entirely by **Olivier Cornelis** ([olicorne.org](https://olicorne.org), Hugging Face [@Olicorne](https://huggingface.co/Olicorne)), French resident psychiatrist and developer. Fully self-funded, no conflicts of interest to declare.

- Version: **v2** (v1.3's splits, further label normalization, every clip re-synthesized with five French voices). The single-voice v1 audio is not part of it; the [Changelog](#changelog) lists what changed since v1.3.
- Fine-tuned model: [Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx) (a [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) fine-tune, released separately, trained with NeMo)
- Multilingual baseline it was fine-tuned from: [Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-optimized-onnx) (the upstream ONNX rebuilt for int8 accuracy and browser speed; use it when you do not need French medical vocabulary)
- In-browser app that runs either model: [Parakeet Web](https://github.com/thiswillbeyourgithub/parakeet_web)
- Reproduction scripts: [github repo](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts) (the full pipeline, released to make it easier to adapt to another domain or language)

## Quick start (TL;DR)

If you just want the data and an opinion on whether it fits, this is the whole thing in one screen. Everything after it is detail (how it was built, exact row format, splits, licensing).

<details>
<summary><b>Expand</b>: a runnable <code>load_dataset</code> snippet, what you get, the subsets, and the one caveat.</summary>

```python
from datasets import load_dataset

# main corpus (dictionary + drugs + PARHAF + acronyms), CC BY 4.0
ds = load_dataset("Olicorne/UltiMed-ASR-FR-v2", "dictionary_CC_BY_4.0", split="train")
row = ds[0]
row["audio"]  # 24 kHz mono waveform, decoded from the embedded FLAC
row["text"]   # the transcript, and the field you train an ASR model on
```

- **What you get**: ~3,350 h (about 600k clips) of clean, studio-quality **synthetic** French medical speech. Each clip is a short dictation-style sentence (anatomy, pathology, drug names, clinical phrasing) read by one of **five** voices (3 female, 2 male, two native French and three accented) and paired with its exact transcript. Built to train **and** evaluate medical ASR.
- **Train on the `text` field.** Everything else (`asr_training_source`, provenance columns) is there for transparency, not as a label.
- **Subsets, picked by config name**: `dictionary_CC_BY_4.0`, `drugs_CC_BY_4.0`, `parhaf_CC_BY_4.0`, `acronyms_CC_BY_4.0` are the main corpus (**CC BY 4.0**); `parrot_CC_BY-NC-SA_4.0` is a small, separate, **evaluation-only** radiology set under a **non-commercial** licence. Download or skip each independently.
- **`val` / `test` are ready-made in-domain benchmarks**: held out group-disjoint from `train`, so report WER on `test` to benchmark technical French medical speech.
- **The one caveat to know up front**: the audio is **synthetic**, all of it out of one TTS model. The five voices give you some speaker, accent and gender diversity, but there is still no real **recording condition**: no room, no microphone, no background noise, no telephone channel. See [the five voices](#the-five-voices-and-what-synthetic-audio-still-costs) and [Known limitations](#known-limitations).
- **Training with NeMo?** NeMo does not read Parquet directly; rebuild the loose/tarred NeMo layout with `scripts/parquet_to_nemo.py`. See [Loading and training](#loading-and-training).

</details>

## How I made UltiMed

Three-stage pipeline (source vocabulary -> written transcript -> synthesized audio), packaged as sharded Parquet with the FLAC bytes embedded (one subset per source). Scripts are released ([github repo](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts)). Each stage is detailed below.

<details>
<summary><b>Expand</b>: the full build detail, stage by stage (sources, LLM text generation, TTS, hardware, repo layout, row format, loading, splits), plus the standardized benchmark protocol.</summary>

### Sources

<details>
<summary><b>Expand</b>: the five source corpora, and how the drug vocabulary was built and ranked.</summary>

| Source | What it contributes | Link |
|--------|--------------------|------|
| Medical and french dictionaries | ~60k French medical terms (eg: `blépharospasme`) -> dictation sentences | various public sources ([Wiktionnaire](https://fr.wiktionary.org/) and assorted medical dictionaries / glossaries) |
| Drug names | the exhaustive 2025 list of French drug names, i.e. the DCI (INN, active-substance names) plus the commercial brand names -> LLM-written sentences that name a medication and often a real dosage | built by the author from French open data (below) |
| [PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) | French clinical documents -> faithful LLM rewrites, one spoken-style paragraph per chunk | [HealthDataHub/PARHAF](https://huggingface.co/datasets/HealthDataHub/PARHAF) |
| Medical acronyms | 511 common medical acronyms (eg: `BPCO`, `CPAP`), hand-filtered from French Wikipedia's list of health abbreviations -> dictation sentences, spoken with their real French pronunciation | [Wikipedia][wiki-abbrev] |
| [PARROT][parrot-paper] (French rows only) | radiology reports, given the same rewrite treatment | [PARROT-reports/PARROT_v1.0](https://github.com/PARROT-reports/PARROT_v1.0) |

**The drug vocabulary is original work.** It was built from **official 2025 France
data** and ranked by real prescription frequency:

- **[OPEN_MEDIC 2025][openmedic]** (CNAM / Assurance Maladie): ambulatory reimbursement volumes.
- **[RETROCEDAM 2025][retrocedam]** (CNAM): hospital retrocession volumes (in UCD units).
- **BDPM** (Base de donnees publique des medicaments) via [data.gouv open data][bdpm-opendata] and the [medicaments-api](https://github.com/Giygas/medicaments-api) project (its live [API](https://medicaments-api.giygas.dev/), snapshot 2025-12-31): drug names, pharmaceutical forms, active substances and dosages.

Joined on drug codes, aggregated per substance and per brand, then scored 0-10 by combined sales (log-normalized): the most prescribed drugs (paracetamol, DOLIPRANE) get up to 11 sentences, the long tail as few as 2.

</details>

### Breakdown by source

<details>
<summary><b>Expand</b>: per-source clip counts, hours, size and mean clip length.</summary>

| Source | Clips | % clips | Hours | % duration | Size | Mean clip |
|--------|------:|------:|------:|-----------:|-----:|----------:|
| dictionary | 534,349 | 88.9% | 2,866.5 | 85.6% | 227.32 GB | 19.3 s |
| drugs | 21,900 | 3.6% | 83.4 | 2.5% | 6.54 GB | 13.7 s |
| PARHAF | 43,384 | 7.2% | 393.0 | 11.7% | 31.82 GB | 32.6 s |
| acronyms | 1,595 | 0.3% | 4.7 | 0.1% | 371.9 MB | 10.5 s |
| **main corpus** | **601,228** | 100% | **3,347.6** | 100% | **266.02 GB** | 20.0 s |

Shipped separately as the `parrot` subset under CC BY-NC-SA 4.0, evaluation-only:

| Source | Clips | Hours | Size | Mean clip | Subset |
|--------|------:|------:|-----:|----------:|--------|
| PARROT | 1,548 | 11.1 | 900.1 MB | 25.8 s | `parrot` (eval-only, test split) |

The dictionary dominates by count; PARHAF clips are fewer but longer (whole rewritten clinical paragraphs, median 32.1 s against the dictionary's 19.0 s), so it carries 11.7% of the duration on 7.2% of the clips. The acronyms subset is deliberately tiny: a targeted patch of short sentences (mean 10.5 s) teaching the sigles the other sources miss. Full per-split tables, a duration histogram and the truncation audit come from `scripts/get_statistics.py`.

That audit is worth reading before training: a TTS generation stopped by its token limit is cut off mid-sentence at exactly the cap, so the transcript no longer matches the audio, and the only observable is a pile-up of clips on one duration. The current corpus is clean (every subset has a single clip at its maximum, no pile-up). An earlier build was not: PARHAF was generated against a 2048-frame cap and 42% of its clips were pinned at 163.84 s, which is why the chunking was tightened and the audio regenerated.

</details>

### Text generation (LLM)

<details>
<summary><b>Expand</b>: the models used, how many sentences each term gets, and how documents are rewritten.</summary>

The written transcripts were generated with **[DeepSeek V4 Pro][deepseek]** (`deepseek/deepseek-v4-pro`) via **[OpenRouter][openrouter]**. Three shapes were used: `Term`, `Acronym` and `Document`

- **Term -> N sentences** (dictionary, drugs): one call per term produces N short, natural, dictation-style sentences, each of which must actually contain the term.

  Varying N spends more audio on the terms that matter and less on the long tail. Each term gets an **importance signal** (`0-10`) that sets its N:

  | Importance (0-10) | Number of spoken samples |
  |------:|---------:|
  | 0-1 | 2 |
  | 2-3 | 4 |
  | 4 | 5 |
  | 5 | 6 |
  | 6 | 7 |
  | 7 | 8 |
  | 8-10 | 11 |

  The two sources compute that signal differently, because only one has real usage data:

  - **Drugs have sales volumes**, so importance is just the **official 2025 sales figure** (log-normalized to `0-10`): common drugs are heard up to 11 times, rare ones only 2. No LLM needed. Drug sentences often also mention a **real dosage** (eg: a sentence naming both `Doliprane` and `comprimés de 1000 mg`).
  - **Dictionary terms have no such data** (a word like `blépharospasme` has no sales count). There is nothing to rank them by, so the **LLM stands in for the missing frequency data**: it scores each term `0-10` for how useful it is to teach the model (roughly, how likely a clinician is to dictate it). A proxy for the signal drugs get for free, not an extra filter on top.

- **Acronym -> 3 sentences** (acronyms): 511 common medical acronyms missing from the other sources, hand-filtered from French Wikipedia's [list of health abbreviations][wiki-abbrev], expanded to 532 (acronym, pronunciation) pairs because some have several accepted French readings (`gamma-GT` letter-spelled or read as a word). Each pair gets 3 sentences containing the acronym verbatim, exactly one of which also glosses its meaning in French. The TTS text swaps the written acronym for its pronunciation (`CAMSP` -> `came-S-P`) so the voice reads it the intended way while the written label keeps the acronym.

- **Document -> one faithful paragraph** (PARHAF, PARROT): each document is split into <= 500-char chunks, each rewritten into one faithful, dictation-ready French paragraph (the originals are written-style: typos, headers, newlines). The chunk cap is a clip-length cap: one chunk becomes one spoken clip.

  Measured on the built corpus, the cap lands PARHAF at a **median clip of 32.1 s** (p90 46.9, p95 52.5, p99 64.6, max 121.4) and PARROT at 25.5 s. Note the cap bounds the **source chunk**, not the spoken text: rewriting spells numbers and units out, so the paragraph actually read aloud averages 503 characters and reaches 1,423, at roughly 0.065 s per character. That is why **12.7% of PARHAF clips (5,529 clips, 20.6% of its duration) run past 45 s**, the ceiling ASR fine-tuning configs commonly impose on `max_duration`: those clips are silently skipped by such a trainer unless the ceiling is raised. Lower the chunk cap if you need all of PARHAF to fit under 45 s.

  The longest paragraphs all come from the same kind of chunk: near-pure shorthand (lab panels, biology tables), where a faithful spell-out of every value legitimately runs to 3x the source length (`Na+ 142meq/L` becomes "le sodium est à cent quarante-deux milliéquivalents par litre"). A length sanity check on the rewrite initially rejected those, leaving 41 of PARHAF's 43,456 chunks unwritten; they were regenerated with the check's upper bound lifted, so **every chunk of every source document is present**.

Resulting volumes:

| Source | Terms / chunks in | Generated texts out |
|--------|------------------:|--------------------:|
| dictionary | 62,448 terms | 534,385 (88.6%) |
| drugs | 3,252 terms | 21,901 (3.6%) |
| PARHAF | 43,456 chunks | 43,456 (7.2%) |
| acronyms | 532 (acronym, pronunciation) pairs | 1,596 (0.3%) |
| PARROT | 1,549 chunks | 1,549 (0.3%) |

**The text is deliberately tailored to Parakeet TDT v3.** Every source term and every generated transcript is checked against the [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) vocabulary, and anything its tokenizer would turn into `<unk>` is fixed before the text is kept. Characters the model does not know are rewritten to the form it does (en-dash and Unicode hyphen to the ASCII `-`, `ɶ` to `œ`, `Ō` to `O`, `ğ` to `g`), symbols that have no spoken form are excluded (`[`, `]`, `→`, `;`), and the two symbols that do are written as the words they are read as (`CD4+` to `CD4 plus`, `T2*` to `T2 étoile`). The check is done on the text as stored, not only as SentencePiece would see it after its own NFKC pass: v1.0 checked after NFKC only, so four characters the tokenizer silently folds (`CO₂`, `Brª`, `CUBE™` and a decomposed `ç`) stayed in the labels even though the model learns `CO2`, `Bra`, `CUBETM`, `ç`. Since v1.1 the labels carry the folded form (`™` dropped), and the generators do the same fold when they parse the LLM output (`ParakeetTokenizer.clean_label` in the pipeline repo). The result is still ordinary French, but it is normalized for one tokenizer's quirks rather than being source-faithful punctuation: if you train a model with a different vocabulary, expect the label set to be narrower than yours (no bracket, no arrow, no semicolon, no `+`).

Total LLM cost was **under $100**, thanks to a **95%+ prompt-cache hit rate** on the shared system prompt. Each written `target` then gets a deterministically derived TTS `source` (Roman numerals after a staging word, plus the rarer `/ µ °` units and `ARNm`, spelled out for the voice, no second LLM call); the `target` is the shipped label. Both are kept (see [Row format](#row-format-and-the-two-text-fields)).

</details>

### Audio synthesis (TTS)

<details>
<summary><b>Expand</b>: the TTS model, its exact settings, and what was (not) done to the audio.</summary>

Every `asr_training_source` was spoken locally by **[Voxtral][voxtral]** (`mistralai/Voxtral-4B-TTS-2603`), served with **[vLLM](https://vllm.ai/) 0.22.0** in a docker container with a single RTX 3090 Ti.

TTS settings:

| Setting | Value |
|---|---|
| Voices | five, one per clip (roster and assignment rule below) |
| Euler steps | 16 |
| CFG alpha | **2.0** (the default since 2026-10-13; earlier first-pass clips are at 1.3); a redraw walks **down** 2.0, 1.8, 1.6, 1.4, 1.2 |
| Acoustic temperature | 0.7 |
| Seed | none sent (see [On seeds](#row-format-and-the-two-text-fields)) |
| Output | 24 kHz mono FLAC (PCM 16-bit) |

`cfg_alpha` **2.0** is not inherited, it was measured: 16 voices x 10 values from 0.1 to 5.0 on one dense medical sentence, with the noise seed pinned, listened to by ear. The usable band is 1.2 to 2.0 (1.5 to 2.0 for the hybrid voices, which need more guidance to hold their articulation), and it is asymmetric: below 1.2 the flow-matching decoder stops following the text and degrades fast (at 0.5 and below every voice rambled to ~2.7x the sentence's natural length instead of stopping), while above 3.0 distortion grows and takes increasingly fail to terminate at all. 2.0 is therefore the **top** of the band, which is why the quality pass redraws downwards from it. v1 ran the whole corpus at 1.3, inside the same band.

The measurement landed partway through the v2 run, so the shipped `cfg_alpha` column is not uniform (all rows, PARROT included):

| `cfg_alpha` | Clips | Why |
|---:|---:|---|
| 2.0 | 478,168 | the default since 2026-10-13 |
| 1.3 | 114,115 | first-pass clips drawn before the default moved |
| 1.8 / 1.6 / 1.4 / 1.2 | 4,153 / 4,152 / 627 / 422 | quality-pass redraws, plus 972 duration redraws |
| 1.5 | 1,005 | redraws at the hybrid voices' floor |
| 2.5 / 3.0 | 13 / 10 | redraws above the band |
| 1.0 | 111 | guidance off; these passed QC and ship |

67,427 hybrid-voice clips sit at 1.3, below the 1.5 floor measured for the hybrids (see [Known limitations](#known-limitations)).

**The clips are the raw FLAC the model emitted**: no trimming, silence removal, loudness normalization, or resampling.

</details>

### Hardware and conditions

<details>
<summary><b>Expand</b>: the GPU, its power cap, and the room it all ran in.</summary>

v1 (~3,115 hours of single-voice audio) was generated on **a single NVIDIA RTX 3090 Ti, power-capped to 100-200 W with its memory downclocked by 100 MHz**, running in a **Parisian apartment with no air conditioning during the 2026 French heatwave**. v2's audio, all ~3,359 hours of it, was generated on **the same single NVIDIA RTX 3090 Ti**. v2 power cap: 350 W.

</details>

### Repository layout on the Hub

<details>
<summary><b>Expand</b>: the file tree on the Hub and what each folder and column holds.</summary>

The whole corpus ships as **sharded Parquet with the original FLAC bytes embedded** (one subset per source), so the HF Data Viewer and `load_dataset(...)` work on the **entire** dataset with no manifest to pair, and it is a few hundred Parquet files instead of ~603k loose FLACs (which would throttle any transfer to a crawl):

```
UltiMed-ASR-FR-v2/
  README.md
  NOTICE.md                                                                       # per-source attributions the licences require
  dictionary/  train-00000-of-00174.parquet .. val-*.parquet .. test-*.parquet   # config "dictionary_CC_BY_4.0"
  drugs/       train-*.parquet   val-*.parquet   test-*.parquet                    # config "drugs_CC_BY_4.0"
  parhaf/      train-*.parquet   val-*.parquet   test-*.parquet                    # config "parhaf_CC_BY_4.0"
  acronyms/    train-*.parquet   val-*.parquet   test-*.parquet                    # config "acronyms_CC_BY_4.0"
  parrot/      test-*.parquet                                                      # config "parrot_CC_BY-NC-SA_4.0", eval-only
  scripts/     build_parquet.py   parquet_to_nemo.py
  preview_samples/  dictionary/  drugs/  parhaf/  acronyms/  parrot/                # 5 clips + manifest.jsonl each
  regex-fixes/ MOVED.txt                                                          # points to the fix rules repository        
```

- **`<source>/<split>-NNNNN-of-NNNNN.parquet`**: ~2,500 clips each. Every row embeds the **original FLAC bytes verbatim** (byte-identical round-trip) in the `datasets` `Audio` feature, plus `text`, `duration`, `category`, provenance (`group_id`, `group_mode`, `item_index`, `voice`), `filename` and the quality-control columns (`cer`, `cer_tail`, `stt_transcript`, `stt_model`, `n_stt_check`, `cfg_alpha`, `regenerated`, `qc_status`, plus the per-judge `whisper_*`, `whisper_v3_*`, `parakeet_*` columns and `cer_max` / `cer_window_max`, see [Row format](#row-format-and-the-two-text-fields)). The steps that turn the pipeline's manifests into this Parquet (syncing QC results, dropping bad rows, label normalization, validation, `build_parquet.py`) are documented in the scripts repo's README, section [`99_hf_release/` is the dataset repo](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts#99_hf_release-is-the-dataset-repo).
- **One subset per source**, each a declared config, browsable and downloadable independently. The licence is encoded in the config name (`dictionary_CC_BY_4.0`, `drugs_CC_BY_4.0`, `parhaf_CC_BY_4.0`, `acronyms_CC_BY_4.0`, `parrot_CC_BY-NC-SA_4.0`) so it shows in the viewer's subset dropdown; the Parquet still lives under the plain `dictionary/`, `drugs/`, `parhaf/`, `acronyms/`, `parrot/` folders.
- **`dictionary` + `drugs` + `parhaf` + `acronyms`** are the main corpus (CC BY 4.0); **`parrot`** is separate (CC BY-NC-SA 4.0, **test-only**), download or skip independently.
- **`NOTICE.md`**: the per-source attributions (see [Licensing](#licensing)).
- **`scripts/`**: `build_parquet.py` (how the Parquet was built) and `parquet_to_nemo.py` (rebuilds the NeMo layout, loose or tarred; see [Loading and training](#loading-and-training)).
- **`preview_samples/`**: a few-MB offline preview, one folder per subset with 5 random `.flac` clips and a 5-line NeMo `manifest.jsonl`. Grab just this to listen and read transcripts without pulling the full corpus (or if the viewer is down). It is not a split; do not train on it.

</details>

### Row format and the two text fields

<details>
<summary><b>Expand</b>: the columns, why there are two text fields, and a worked example.</summary>

Each Parquet row is one clip, with these columns:

- `audio`: original FLAC bytes (24 kHz mono) via the `datasets` `Audio` feature; decoded by `load_dataset` or read raw from `row["audio"]["bytes"]`. Byte-identical to the synthesized FLAC.
- `text`: the written ASR label (`asr_training_target`), what the model should output, and **the field to train on.**
- `asr_training_source`: the exact text spoken to the TTS engine, derived **deterministically** from `text` (no second LLM call): the same sentence with a few tokens respelled for the voice. Shipped for transparency, **not** the training label.
- `duration`: clip length in seconds (from the FLAC header).
- `category`: the source (`dictionary`, `drugs`, `parhaf`, `acronyms`, or `parrot`).
- `group_id` / `group_mode` / `item_index`: provenance used by the split logic (`distribute` for dictionary / drug / acronym term variants, `atomic` for PARHAF / PARROT document chunks); ignored during training.
- `voice`: which of the five voices read this clip (`fr_female`, `fr_male`, `h_frf_esm`, `h_frm_nlf` or `h_itf_def`). Always set in v2; it is an empty string only on v1-era manifests, which were single-voice and carried no such field. Use it to score a model per voice, to hold a voice out, or to train on a subset of the roster.
- `filename`: the clip's source basename, e.g. `000014_0001_abaissement_de_la_cataracte.flac`.

Every clip was also transcribed back and scored, and those results ship with it:

- `cer`: character error rate of `stt_transcript` against `text`, after folding away differences that are only notation (a unit spelled out on one side and abbreviated on the other, `1er` against `premier`, and so on). **This is a quality signal, not a defect list**: the main-corpus median is 0.010 (mean 0.013), and most of what remains at 0.05 to 0.10 is the STT model's spelling of rare medical terms or French agreement it cannot hear, not audio that is wrong. Filter on it if you want a stricter subset.
- `cer_tail`: the same score over the last ~30 seconds only, `null` on clips shorter than that. It catches endings a whole-clip average dilutes (a chunk truncated at the TTS output cap, a voice that derails or loops on an enumeration).
- `stt_transcript`: what the STT model actually heard, so every `cer` above can be checked rather than trusted.
- `stt_model`: which model produced them, `whisper.cpp/ggml-large-v3-turbo` (Whisper large-v3-turbo served by whisper.cpp through an OpenAI-compatible endpoint). Spelled out on purpose: the API alias the client sends is `whisper-1`, which says nothing about what actually ran.
- `n_stt_check`: how many readings a clip needed when a bad score triggered a re-transcription at a higher temperature. `null` on every v2 row (no such re-reading ran on the v2 audio); kept so the schema matches v1.
- `cfg_alpha`: the classifier-free guidance the audio was drawn at (see the distribution in [Audio synthesis](#audio-synthesis-tts): mostly 2.0, with an earlier first-pass cohort at 1.3).
- `regenerated` / `qc_status`: whether the quality pass replaced this clip. `qc_status` is `improved` (a better draw replaced the original), `exhausted` (the clip tripped a gate but no redraw was good enough, so **the original audio ships**: v1.0 shipped 6 of them by mistake and v1.1 dropped them, so the value no longer appears), or `original` (never tripped a gate). Across the release 16,593 clips were replaced (16,458 main corpus, 135 PARROT) and the mean CER of the shipped main-corpus audio against its label is **0.013**.

**Two judges, one column group each (v2).** Every clip was read by two different ASR models, and both readings ship, because they fail in opposite directions and a single blended number hides that. Whisper's decoder suppresses repeated n-grams, so a clip that says the same clause three times comes back transcribed **once** and scores clean: on 269 clips an acoustic scan had flagged, whisper read CER 0.0167 where parakeet read 0.1796 on the same audio, a 10.7x gap. Parakeet has the opposite bias: it is a general French model with no exposure to this corpus's drug brand names, so it mishears clean audio at CER 0.08-0.14. Neither is the truth alone.

- `whisper_stt` / `whisper_cer` / `whisper_cer_window` / `whisper_repeat_excess`: the reading from `whisper.cpp/ggml-large-v3-turbo`, which read all of the corpus. `whisper_repeat_excess` is `null` on every row (the whisper scorer does not compute it, since its decoder suppresses the repeats it would count); it ships anyway so every judge has the same four columns. `whisper_stt` and `whisper_cer` are the same values as the flat `stt_transcript` and `cer` columns above, kept there unchanged for v1 compatibility. `whisper_cer_window` is a different measure from `cer_tail`: the CER of the worst fixed-length window anywhere in the clip, the only score that sees a phrase dropped in the **middle** of a long clip.
- `whisper_v3_*`: the same four from non-turbo `ggml-large-v3`, a second opinion run only over the clips turbo flagged (6,717 main-corpus clips, 54 PARROT), so `null` on most rows. `whisper_v3_repeat_excess` is `null` on every row, for the same reason as turbo's.
- `parakeet_*`: the same four from `onnx-asr/parakeet-tdt-0.6b-v3-ultra`, on 329,148 main-corpus clips and 891 PARROT clips (`null` elsewhere). Its `parakeet_repeat_excess` is the only loop detector in the release: it counts word n-grams the hypothesis repeats more often than the label does, and whisper structurally cannot report that.
- `cer_max` / `cer_window_max`: the worst CER and worst window CER any judge gave the clip. Filter on these for a pessimistic subset; filter on one judge's own column when you know which bias you can live with.

**On seeds.** The per-clip TTS seed is deliberately **not** shipped, because there is none to ship: the clips were drawn unseeded. A seed would not have bought much anyway. The flat `seed` field of the API does nothing on this model (it lands in a sampler that is handed a vocab-wide `-inf` row with one finite entry, so the token is forced whatever the seed says), the audio's only randomness is the Gaussian each frame's flow-matching ODE starts from, and reaching it takes a per-request route this model's stock server does not expose. Pinning one seed for the whole corpus would also have made every clip start from the same noise draw, which works against the acoustic variety this release exists for. So regenerating a byte-identical clip from the released text is not possible; the audio itself is the artifact.

**Why two fields.** `text` is the clean transcript the model must produce; `asr_training_source` is the same sentence adjusted only where the voice would mispronounce a token: Roman numerals after a staging word (`type II`, `grade I`, `stade II`) written out, and the rarer `/`, `µ`, `°` units and `ARNm` spelled out. Everything else, plain digits included (`16 mg`, `4 semaines`), stays as written. Both are French, restricted to characters the [Parakeet TDT v3][parakeet-v3] tokenizer can represent (see [Text generation](#text-generation-llm)), and free of banned non-spoken symbols.

Worked example (a real `drugs` clip):

| field | value |
|-------|-------|
| `text` (the label you train on) | Instauration de KENZEN 16 milligrammes le matin pour une hypertension artérielle de grade **I**, contrôle tensionnel à 4 semaines. |
| `asr_training_source` (what was spoken) | Instauration de KENZEN 16 milligrammes le matin pour une hypertension artérielle de grade **un**, contrôle tensionnel à 4 semaines. |

Only `grade I` became `grade un`; `16 milligrammes` and `4 semaines` stayed as written. You train the model on `text`; the audio in `audio` is what `asr_training_source` produced.

</details>

### Loading and training

<details>
<summary><b>Expand</b>: loading with <code>datasets</code>, rebuilding the NeMo layout, and reading the raw FLAC bytes.</summary>

**Browsing / general use** (`datasets`): load any subset by config name; audio and transcript come back together, no manifest to pair.

```python
from datasets import load_dataset
ds = load_dataset("Olicorne/UltiMed-ASR-FR-v2", "dictionary_CC_BY_4.0", split="train")
ds[0]["audio"]  # {'array': ..., 'sampling_rate': 24000}, decoded from the embedded FLAC
ds[0]["text"]   # the transcript label
```

**With NeMo** (the intended training path): NeMo does not read Parquet directly, so rebuild the training layout from it with the shipped `scripts/parquet_to_nemo.py`, which decodes the embedded FLAC verbatim and writes a NeMo manifest.

```bash
# loose FLAC + manifest.jsonl per split (drop-in for a manifest_filepath config)
uv run scripts/parquet_to_nemo.py --format loose

# or NeMo tarred (WebDataset) shards, for an is_tarred config
uv run scripts/parquet_to_nemo.py --format tarred
```

By default it merges `dictionary` + `drugs` + `parhaf` + `acronyms` into `train` / `val` / `test` and keeps `parrot` as its own eval-only group (`--per-source` keeps every subset separate). Loose manifests plug into `model.train_ds.manifest_filepath=...`; tarred output into `is_tarred: true` + `tarred_audio_filepaths=<group>/tarred/'audio__OP_0..N_CL_.tar'`.

**Without NeMo**: read the Parquet with any Arrow / `datasets` reader and pull the raw bytes from `row["audio"]["bytes"]` (a complete FLAC file) if you want the original encoded clip rather than a decoded waveform.

</details>

### Splits (combined release-wide re-split)

<details>
<summary><b>Expand</b>: the 80 / 10 / 10 duration-balanced tables and the grouping rules that prevent leakage.</summary>

**For both training and evaluation.** `val` and `test` are held out group-disjoint from `train` (no term or document leaks), so they are ready-made **in-domain benchmarks**, not just training monitors: report WER on `test` for technical French medical speech. The separate `parrot` subset is an **out-of-domain radiology** benchmark (below).

Main corpus balanced 80 / 10 / 10 by **audio duration**, group-aware so nothing leaks:

| Split | Clips | % clips | Hours | % duration | Size |
|-------|------:|------:|------:|-----------:|-----:|
| train | 485,946 | 80.8% | 2,681.8 | 80.1% | 213.06 GB |
| val | 57,692 | 9.6% | 332.8 | 9.9% | 26.48 GB |
| test | 57,590 | 9.6% | 333.0 | 9.9% | 26.48 GB |
| **all** | **601,228** | 100% | **3,347.6** | 100% | **266.02 GB** |

These splits are the **main corpus** (dictionary + drugs + PARHAF + acronyms). PARROT is **not** mixed in: it ships as its own eval-only `parrot` subset (all 1,548 clips as one `test` split) under its own licence, **test-only on purpose** because the source authors ask that it not be trained on, and one untouched `test` split keeps it a clean out-of-domain benchmark (not split into val/test: the main corpus already has a large `val`, and halving 11.1 h would only add noise). Counts run slightly above 80% while duration lands at about 80% because the dictionary's short clips are the most numerous. Grouping rules:

- **Dictionary / drug / acronym terms**: the variants of one term are *distributed* across splits (at least one variant in train; one per split when a term has >= 3 variants), so no term is memorized in train and only tested in test.
- **PARHAF documents** (and, in its own subset, **PARROT** documents): all chunks of one source document stay in the **same** split (*atomic*), so no document appears in both train and test.

</details>

---

## Evaluation as a standardized benchmark

<details>
<summary><b>Expand</b>: what to report, the fixed normalization recipe, and a runnable scoring snippet.</summary>

`val` and `test` are held out group-disjoint from `train` (no term or document leaks), so they are ready-made in-domain benchmarks. Please report them with the recipe below rather than an ad-hoc setup, so numbers stay comparable across models.

**What to report**

- **Metrics**: WER (primary) and CER (secondary).
- **Reference text**: the `text` field (the training label), never `asr_training_source`.
- **Splits**: the main-corpus `test` split (in-domain) and the `parrot` `test` split (out-of-domain radiology). Report them separately; do not average or mix them.
- **Per subset**: report `dictionary`, `drugs`, `parhaf` and `acronyms` on their own as well as combined, since their difficulty and clip length differ a lot (see [Breakdown](#breakdown-by-source)).

**Normalization**, applied identically to reference and hypothesis, in this order:

1. lowercase
2. French number words to digits, so the spoken "seize" scores against the written "16" (e.g. `text_to_num.alpha2digit(s, lang="fr")`)
3. strip punctuation
4. collapse whitespace

Step 2 is not optional here: [Voxtral][voxtral] speaks a written "16 mg" as "seize milligrammes", so without it every digit in the label counts as an error. It does not canonicalize spelled-out unit words ("milligrammes" vs "mg"); that is a known residual mismatch, so keep the four steps fixed as above rather than adding per-model tweaks.

```python
# uv run --with datasets --with jiwer --with text_to_num eval.py
import re
import jiwer
from datasets import load_dataset
from text_to_num import alpha2digit

def normalize(s: str) -> str:
    s = alpha2digit(s.lower(), lang="fr", relaxed=True)  # "seize" -> "16"
    s = re.sub(r"[^\w\s]", " ", s)                        # strip punctuation
    return re.sub(r"\s+", " ", s).strip()                 # collapse whitespace

ds = load_dataset("Olicorne/UltiMed-ASR-FR-v2", "dictionary_CC_BY_4.0", split="test")
refs = [normalize(r["text"]) for r in ds]
hyps = [normalize(transcribe(r["audio"])) for r in ds]   # plug in your ASR model

print(f"WER {jiwer.wer(refs, hyps):.4f}  CER {jiwer.cer(refs, hyps):.4f}")
```

**With NeMo**: rebuild the layout with `scripts/parquet_to_nemo.py` (see [Loading and training](#loading-and-training)) and score with NeMo's built-in WER, applying the **same** four normalization steps to hypotheses and references so the numbers line up with the snippet above.

When you publish results, state the model, the exact subset and split, and that you followed this recipe.

### Results

The v2 model's WER and CER on each subset's `test` split, with and without the regex fixes, are in the [benchmarks of its model card](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx#benchmarks), kept there so the numbers live in one place.

</details>

</details>

---

## Considerations for Using the Data

<details>
<summary><b>Expand</b>: audio quality, the five voices, the known limitations, and privacy.</summary>

### Audio quality

<details>
<summary><b>Expand</b>: what the clips actually sound like.</summary>

High-fidelity 24 kHz mono: clean, studio-style speech, no background noise, channel effects, or clipping. All five voices read technical vocabulary (drug names, anatomy, pathology) accurately, which is the single criterion they were selected on, and that is exactly what an in-domain medical set needs. The flip side (synthetic speech, no real recording condition) is covered below.

</details>

### The five voices, and what synthetic audio still costs

<details>
<summary><b>Expand</b>: the roster, how a clip gets its voice, and the diversity that is still missing.</summary>

v1 read all 600k clips with **one** voice (`fr_female`), the limitation everyone pointed at first: no speaker, accent or gender diversity, so a model could learn one speaker's acoustics instead of French medical speech. v2 re-synthesized the entire corpus with **five** voices, 3 female and 2 male:

| `voice` | gender | accent | origin | clips |
|---|---|---|---|---:|
| `fr_female` | F | native French | shipped Voxtral voice | 121,206 |
| `fr_male` | M | native French | shipped Voxtral voice | 122,697 |
| `h_frf_esm` | M | Spanish-accented French | hybrid | 120,764 |
| `h_frm_nlf` | F | Dutch-accented French | hybrid | 119,899 |
| `h_itf_def` | F | German-accented French | hybrid | 118,210 |

**Where the three hybrids come from.** Voxtral ships 20 voices but only two meant for French, and its open checkpoint ships **no voice encoder**, so a real voice clone is not possible: `ref_audio` cloning hits exactly the missing weights. A Voxtral voice is however just a small `[N, 3072]` bfloat16 prefix in the LLM's embedding space, so new voices can be made by blending and splicing the shipped ones, no training and no encoder involved. Each `h_<a>_<b>` name is such a blend: the voice you hear is the **acoustic** donor `<b>` (its timbre and accent), while the **semantic** donor `<a>` only moves pronunciation stability. That is how Voxtral's non-French voices end up reading French with their own accent, which is precisely the diversity v1 lacked.

**They were auditioned, then measured.** Blends can land off the manifold real prefixes live on and come out as mush, so the roster is the survivor of two listening rounds: 16 candidates cleared on FLEURS sentences, then **all 16 re-auditioned on hard medical text**, which disagreed about 6 of them (FLEURS clearance does not predict medical clearance: a voice that reads the news cleanly can still mangle a drug name). Ten distinct identities survived, all judged by ear on jargon accuracy first. One then failed on audio: a CPU ASR pass over a random sample of the first v2 clips flagged 14.9% of that voice's clips against 0.0-4.6% for the others, so it was dropped and its clips regenerated. Four more were retired on 2026-11-03 by a **training** measurement rather than a listening one: nine 1500-step Parakeet fine-tunes, each on one voice's slice of the corpus, ranked by mean WER on two **real-speech** validation sets. The ranking split in two with nothing in between (the five kept voices at 0.0638 to 0.0648, the four retired at 0.0667 to 0.0687), and both shipped voices sat in the good tier, so the gap is damage done by the hybrid surgery rather than ordinary spread between voices. Their clips were re-drawn onto the survivors. Five remain, hence 3F/2M.

**How a clip gets its voice.** A blake2b hash of the clip's group key indexes the original roster, and a key that lands on a retired voice is re-drawn until it hits a survivor (uniform over the five). Two regimes, on purpose:

- **dictionary, drugs, acronyms** (92.8% of the clips) hash **per clip**, so the ~8 written variants of one term are spread over about 4 of the 5 voices (4.06 distinct voices per term in the dictionary, 3.81 in drugs, 2.44 in acronyms). You hear any given drug name in several accents, and a voice that happens to read one term badly cannot take all of that term's clips down with it.
- **PARHAF and PARROT** hash **per document**, so every chunk of one clinical report keeps the same reader. A report switching speaker mid-sentence would be an artifact no real dictation has.

It is a hash and not a counter, which is what makes the assignment stable: adding, dropping or re-generating a clip never moves another clip to another voice, so the corpus can be rebuilt incrementally and a resumed run assigns exactly what the crashed one would have.

The balance is close to even but not exact, for an arithmetic reason rather than a bug: a dataset with only a few hundred groups (acronyms, PARROT) cannot divide evenly over five voices. `scripts/get_statistics.py` prints the per-voice counts, including the per-split ones, so you can check no voice is over-represented in `val` or `test` before benchmarking on them.

**What is still missing, and it matters.** Five voices are five voices, not a speaker population: every clip still comes out of **one TTS model**, at 24 kHz, in studio conditions. There is no room acoustics, no microphone colour, no codec, no background noise, no overlapping speech, no disfluency, and the prosody is uniformly dictation-style. The accents are embedding surgery on a French-capable model, not recordings of actual Spanish, Dutch or German speakers of French, so treat them as **acoustic variety**, not as a representative sample of accented French. For deployment, augment (noise, impulse responses, codecs, speed perturbation) and, where you can, fine-tune on real in-domain recordings; this corpus is the vocabulary and the benchmark, not the channel.

The voices were built, blended and auditioned in my own Voxtral serving harness, which is where the recipe and the selection scripts live: [UltiMed-ASR-FR-v2-Voxtral][voxtral-harness].

</details>

### Known limitations

<details>
<summary><b>Expand</b>: synthetic audio, LLM-written text, tokenizer-normalized punctuation, France-only drugs, French only, accent spellings, known unknowns.</summary>

- **All audio is synthetic**: every clip comes from one TTS model (Voxtral), so there is no background noise, microphone variation, room acoustics, or spontaneous disfluency, and five voices are not a speaker population.
- **LLM-generated text**: sentences are model-written and validated by automatic gates, but may still contain occasional artifacts.
- **Text normalized for one tokenizer**: the labels were gated on the [Parakeet TDT v3][parakeet-v3] vocabulary, so a handful of characters were rewritten or excluded to avoid `<unk>` (details in [Text generation](#text-generation-llm)). Harmless for most models, but the punctuation is not source-faithful.
- **Drug coverage is France-only**: the drug vocabulary is the 2025 French prescription base (OPEN_MEDIC, RETROCEDAM, BDPM). Drugs not marketed in France, new molecules, withdrawn drugs and research chemicals are simply absent, so neither the dataset nor a model trained on it covers them. v3 is meant to widen the list.
- **French only**: no medical English and no code-switching, although French clinicians routinely drop English terms into dictation. A model fine-tuned only on this corpus may lose some of its English.
- **Uneven repetition per drug**: each drug name is heard in 2 to 11 sentences depending on its sales, so rare drugs are seen only a couple of times.
- **Guidance below the hybrid floor**: 67,427 hybrid-voice clips were drawn at `cfg_alpha` 1.3, below the 1.5 floor measured for the hybrids. They passed QC, but filter on `cfg_alpha` and `voice` if you want only clips inside the measured band.
- **Known unknowns**: the automatic gates cannot catch everything, and some blind spots are certainly still in there. If you hit a wrong transcript, a mispronounced term, a systematic artifact, or anything else worth fixing, please report it (see [Contact](#contact)). Feedback is what v3 will be built on.

</details>

### Privacy and PII

<details>
<summary><b>Expand</b>: why every source is PII-free.</summary>

**Only PII-free sources were used, and no personal data was added.** Dictionary terms are from public vocabularies, the drug data is public aggregate statistics (no patient-level data), PARHAF is public de-identified clinical data, the acronyms come from a public Wikipedia list, and PARROT reports are **completely fictional** per their authors. Transcripts are LLM-generated, so any string resembling a real person is **coincidental** or already present in DeepSeek's training data or a public source. None was introduced by me.

</details>

</details>

---

## Licensing

<details>
<summary><b>Expand</b>: the two licence tiers and the per-source attribution to ship.</summary>

**This release ships in two licence tiers, so the bulk stays permissive while the one non-commercial source is kept separable:**

- **Main corpus** -- dictionary + drugs + PARHAF + acronyms, i.e. the `dictionary`, `drugs`, `parhaf` and `acronyms` subsets: **[CC BY 4.0][ccby]**.
- **PARROT radiology subset** -- the `parrot` subset, shipped on its own and **not** mixed into the main-corpus subsets: **[CC BY-NC-SA 4.0][ccbyncsa]**.

Ship a `NOTICE` with the per-source attributions below, **mark the data as modified** (text synthesized to audio), and keep the no-endorsement statement.

### Main corpus: [CC BY 4.0][ccby]

- **Medical terms** -- drawn from various public French sources (Wiktionary and assorted medical dictionaries / glossaries). Only individual terms are used; no definitions or article text are redistributed.
- **PARHAF** -- [CC BY 4.0][ccby] + [Etalab Open Licence 2.0][etalab]. Credit HealthDataHub / Plateforme des Donnees de Sante, link the [dataset][parhaf-ds] and version, cite the paper (below), note the data was **modified** (synthesized to audio), no endorsement, **training set only** (test set embargoed, not redistributed).
- **Drug names** -- French **open data** under the [Licence Ouverte / Etalab 2.0][etalab]: [OPEN_MEDIC][openmedic] and [RETROCEDAM][retrocedam] (CNAM / Assurance Maladie) and the [BDPM public medicines database][bdpm-opendata]. Only names and public dosages are used; credit CNAM / Assurance Maladie and the BDPM.
- **Acronyms** -- hand-filtered from French Wikipedia's [list of health abbreviations][wiki-abbrev]. Only the acronyms and their expansions (uncopyrightable facts) are used; no article text is redistributed. The sentences containing them are LLM-generated.

### PARROT subset: [CC BY-NC-SA 4.0][ccbyncsa]

The PARROT clips are **LLM-based rewrites** of PARROT's fictional radiology reports, so this subset carries PARROT's own licence, [CC BY-NC-SA 4.0][ccbyncsa]: credit the [PARROT authors][parrot-paper] (Le Guellec et al., *European Journal of Radiology Artificial Intelligence*, 2026; [dataset repo][parrot-repo]), non-commercial use only, and any adaptation stays under the same licence. Please [cite the paper](#parrot-source-please-cite). It ships as a **test-only** evaluation subset, separate from the training data. The reports are fictional.

</details>

## Citations

### UltiMed-ASR-FR-v2

```bibtex
@misc{cornelis_ultimed_asr_fr_2026,
  title        = {UltiMed-ASR-FR-v2: a large synthetic French medical speech dataset for ASR},
  author       = {Cornelis, Olivier},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2}
}
```

### PARHAF (source, please cite)

<!-- The arXiv id is intentionally omitted from this citation. Any contiguous form
of it (an arxiv.org URL, the DOI, or a bibtex eprint field) makes the HF Hub
auto-derive an arxiv:<id> tag that makes PARHAF's paper look like THIS dataset's
own paper. Readers reach the paper through the PARHAF dataset page linked below. -->

```bibtex
@misc{tannier2026parhaf,
  title        = {PARHAF, a human-authored corpus of clinical reports for fictitious patients in French},
  author       = {Tannier, Xavier and Abbara, Salam and Flicoteaux, R{\'e}mi and Khalil, Youness and N{\'e}v{\'e}ol, Aur{\'e}lie and Zweigenbaum, Pierre and Bacry, Emmanuel},
  year         = {2026},
  howpublished = {arXiv preprint; see the PARHAF dataset page},
  url          = {https://huggingface.co/datasets/HealthDataHub/PARHAF}
}
```

### PARROT (source, please cite)

```bibtex
@article{leguellec2026parrot,
  title   = {PARROT, an open multilingual radiology reports dataset},
  author  = {Le Guellec, Bastien and Adambounou, Kokou and Adams, Lisa C. and Agripnidis, Thibault and Ahn, Sung Soo and Ait Chalal, Radhia and Akinci D'Antonoli, Tugba and Amouyel, Philippe and Andersson, Henrik and Bentegeac, Rapha{\"e}l and Benzoni, Claudio and Blandino, Antonino Andrea and Busch, Felix and Can, Elif and Cau, Riccardo and Cavallo, Armando Ugo and Chavihot, Christelle and Chiquete, Erwin and Cuocolo, Renato and Divjak, Eugen and Dziadkowiec-Macek, Barbara and Elogne, Armel and Fanni, Salvatore Claudio and Ferrarotti, Carlos and Fossataro, Claudia and Fossataro, Federica and Fu{\l}ek, Katarzyna and Fu{\l}ek, Micha{\l} and Ga{\'c}, Pawe{\l} and Gachowska, Martyna and Garc{\'i}a-Ju{\'a}rez, Ignacio and Gatti, Marco and Gorelik, Natalia and Goulianou, Alexia Maria and Hamroun, Aghiles and Herinirina, Nicolas and Holay, Quentin and Ivanac, Gordana and Kitamura, Felipe and Klontzas, Michail E. and Kompanowska, Anna and Kompanowski, Rafa{\l} and Kraik, Krzysztof and Krupka, Dominik and Lef{\`e}vre, Alexandre and Lemke, Tristan and Lindholz, Maximilian and Macek, Piotr and Makowski, Marcus and Mannacio, Luigi and Meddeb, Aymen and M{\"u}ller, Lukas and Natale, Antonio and Nguema Edzang, B{\'e}atrice and Ojeda, Adriana and Park, Yae Won and Piccione, Federica and Ponsiglione, Andrea and Por{\k{e}}ba, Ma{\l}gorzata and Por{\k{e}}ba, Rafa{\l} and Prucker, Philipp and Pruvo, Jean-Pierre and Pugliesi, Rosa alba and Rabemanorintsoa, Feno Hasina and Rafailidis, Vasileios and Resler, Katarzyna and Rotkegel, Jan and Saba, Luca and Siebert, Ezann and Stanzione, Arnaldo and Tekin, Ali Fuat and Toapanta-Yanchapaxi, Liz and Triantafyllou, Matthaios and Tsaoulia, Ekaterini and Urban, Szymon and Vassalou, Evangelia and Vernuccio, Federica and Wang, Weilang and Wass{\'e}lius, Johan and W{\l}odarczak, Adrian and W{\l}odarczak, Szymon and Wysocki, Andrzej and Xu, Lina and Zato{\'n}ski, Tomasz and Zhang, Shuhang and Ziegelmayer, Sebastian and Kuchcinski, Gr{\'e}gory and Bressem, Keno K.},
  journal = {European Journal of Radiology Artificial Intelligence},
  year    = {2026},
  volume  = {5},
  pages   = {100066},
  doi     = {10.1016/j.ejrai.2025.100066},
  url     = {https://doi.org/10.1016/j.ejrai.2025.100066}
}
```

## Acknowledgements

*In no particular order*

- [Corentin Sautier](https://csautier.github.io/), PhD (Hugging Face [@csautier](https://huggingface.co/csautier)).
- [Adrien Parrot](https://www.linkedin.com/in/adrien-parrot-doc/), MD, at [InterHop](https://interhop.org).
- [Bastien Leguellec](https://bleguellec.org/), MD and currently pursuing a PhD.
- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock), which substantially improved my GPU performance.

## Contact

Olivier Cornelis, feel free to reach me at [olicorne.org](https://olicorne.org)

**Please report errors and blind spots.** Bad clips, mispronunciations, missing vocabulary, anything that looks off: open a discussion on this dataset page or reach out directly. Corrections are welcome and will feed a future version.

This card and the pipeline scripts were written with the help of Claude Code (Anthropic's coding harness).

[parakeet-v3]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
[ccby]: https://creativecommons.org/licenses/by/4.0/
[ccbyncsa]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[etalab]: https://github.com/etalab/licence-ouverte/blob/master/LO.md
[openmedic]: https://www.assurance-maladie.ameli.fr/etudes-et-donnees/open-medic-base-complete-depenses-medicaments
[retrocedam]: https://www.assurance-maladie.ameli.fr/etudes-et-donnees/medicaments-retrocession-hospitaliere-retrocedam
[bdpm-opendata]: https://www.data.gouv.fr/dataservices/api-medicaments-fr
[parhaf-ds]: https://huggingface.co/datasets/HealthDataHub/PARHAF
[parrot-repo]: https://github.com/PARROT-reports/PARROT_v1.0
[parrot-paper]: https://doi.org/10.1016/j.ejrai.2025.100066
[wiki-abbrev]: https://fr.wikipedia.org/wiki/Liste_d%27abr%C3%A9viations_en_sant%C3%A9
[voxtral]: https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
[voxtral-harness]: https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-Voxtral
[openrouter]: https://openrouter.ai/
[deepseek]: https://openrouter.ai/deepseek/deepseek-v4-pro

