# 05_generate_audio

Stage 05 of the pipeline: speak the text produced by stages 01 to 04 with the
local TTS server. Both candidate services bind `:8003` (only one up at a
time) and the client covers whichever answers: `crispasr-tts` (HumeAI TADA 3B
multilingual), `voxtral-tts` (vLLM serving mistralai/Voxtral-4B-TTS-2603), or
a qwen3-tts VoiceDesign deployment. The final backend choice is still under
investigation. Written with Claude Code.

`CrispASR`, referenced here and in stage 06, is a private unpublished project of
mine: a high-performance self-hosted Whisper server whose compose project also
hosts these TTS services. You do not need it. Point the client at any server that
answers on `:8003`.

The `voxtral-tts` container that ended up speaking the dataset is published at
[UltiMed-ASR-FR-v1-Voxtral](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-Voxtral).

## `01_generate_audio.py`

A minimal client for `POST /v1/audio/speech`, simplified from the full
`CrispASR/tts.py` (which absorbed the old voxtral query client). Run it with
`uv run 01_generate_audio.py` (PEP 723 script).

Two modes, chosen by `--input`:

- **Text mode**: `--input "some sentence"` synthesizes once and writes the
  audio to `--output` (a file, default `tts_out.wav`).
- **Batch mode**: `--input path/to/file.jsonl` reads one JSON object per line,
  synthesizes each row's `asr_training_source`, and writes into the
  `--output` folder as `{term_index}_{variant_index}_{normalized_term}.wav`
  (term slug: accents stripped, lowercased, non-alphanumerics to `_`).

Batch runs are resumable: existing files are skipped unless `--overwrite` is
given, a tqdm bar tracks only the remaining work, per-row server errors are
logged and skipped (exit code 1 if any row failed), and an unreachable server
aborts the run so you can re-run to resume. Rows missing the required keys
(`term_index`, `variant_index`, `term`, `asr_training_source`) are logged and
counted as bad.

Sanity check the server first:

```bash
uv run 01_generate_audio.py --check
```

Typical batch run over the stage 01 output:

```bash
uv run 01_generate_audio.py \
  --input ../01_dictionnary/generated_dataset.jsonl \
  --output ./audio \
  --seed 42
```

## Ten voices (v2)

v1 read the whole corpus with a single voice (`fr_female`). v2 spreads it over
the ten French voices in `utils/voice_assignment.py`, which were picked by ear
in the CrispASR harness repo (`voxtral/cloning/voices_fr_production.txt`). The
voice is a per-request field, so **one run covers all ten**, no restart and no
second pass:

```bash
uv run 01_generate_audio.py \
  --input ../01_dictionnary/generated_dataset.jsonl \
  --output ./audio          # --voice-mode group is the default
```

- `--voice-mode group` (default): the voice is a hash of the row's group key,
  so all variants of a term, and all chunks of a PARHAF/PARROT document, are
  read by one speaker. A hash rather than a round-robin because it is stable
  under editing: adding or deleting a row does not move any other group to
  another voice, so clips already on disk stay correctly labelled (a resume
  matches filenames, and a filename says nothing about the voice).
- `--voice-mode row`: every clip draws its own voice, spreading a term's
  variants over speakers.
- `--voice X` or `--voice-mode single`: one voice for everything, i.e. v1.
  `--voice` wins over the roster, so the old invocations still behave.
- `--voices-file FILE`: another roster, one name per line, `#` comments
  ignored. It reads the CrispASR file above as is. **File order is part of the
  mapping**, so re-ordering it re-assigns every voice.

The run logs the spread (`voices: 10 of them, by term_index (group mode):
fr_female=52564, ...`) and writes `voice_assignment.json` into the output
folder: the roster, the mode, the group field and the row count per voice.
Stage 99 reads that file to put the right name in the manifest's `voice`
column, so do not delete it, and do not edit the roster between generating and
releasing. The balance is deliberately approximate, measured at 2.8 pct spread
over `01_dictionnary`'s 62,448 term groups and up to 50 pct over `04_PARROT`'s
475 document groups.

v2 is generated at Voxtral `cfg_alpha` **2.0**, retrying a bad clip at 1.5:
below about 1.2 the flow-matching decoder stops following the text, and from
3.0 up takes distort and increasingly fail to terminate at all (measured in
the CrispASR repo, 160-take sweep, 2026-10-13).

Per-request knobs (`--instruction`, `--voice`, `--seed`, `--speed`,
`--model`, and the talker sampling flags `--temperature`, `--top-p`,
`--top-k`, `--repetition-penalty`, `--do-sample`) are sent only when set;
anything unset keeps the server's startup default. That contract is also what
keeps TADA-only fields out of Voxtral requests: Voxtral honours only
`input` / `voice` / `response_format` / `speed` per request (its sampling and
seed are startup env vars). Backend cheat sheet:

- **TADA (crispasr-tts)**: all knobs work; `--voice` takes a
  `tada-ref-*.gguf` path or registry name.
- **Voxtral (voxtral-tts)**: pass `--voice` one of the 20 built-in embedding
  names (`fr_male`, `fr_female`, `neutral_female`, ...; see `VOXTRAL_VOICES`
  in the script, a deliberate copy from `CrispASR/tts.py`), leave the
  sampling flags unset, and add `--model mistralai/Voxtral-4B-TTS-2603` if
  vLLM rejects a modelless request. Formats: `wav/flac/mp3/aac/opus/pcm`.
- **Qwen3 VoiceDesign**: `--instruction` carries the voice-direction prose.

Clip durations (stdlib `wave`, wav responses only) are logged per file at
DEBUG level and totalled in the batch summary. Logs go to stderr and are
appended to `tts_generate.log` next to the script (override with
`--log-file`).
