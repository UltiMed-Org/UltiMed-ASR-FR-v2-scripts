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

### The code gate (`--code-dump-dir`)

A duration check cannot see a take that is a plausible **length** for its
text and still wrong, and the CPU QC tail says that is about half of the
severe failures. So every take is also checked against the codes it emitted,
which is free: the patched Voxtral fork already dumps them, and column 0 of
a dump is the semantic code, a hard `argmax` and therefore a deterministic
function of text and context.

```bash
uv run 01_generate_audio.py ... \
  --code-dump-dir "$HOME/path/to/CrispASR/cache/cloning/dumps"
```

The argument is the **host** path of the directory the server knows as
`$VOXTRAL_CODE_DUMP_DIR` (the container sees `/cache/cloning/dumps`), so
only the operator can say what it is. A derailed take raises
`DerailedTakeError` and walks the same cfg_alpha ladder as a duration
rejection.

What "derailed" means is measured, not assumed, and it is **not** a loop: a
repeating-unit scan fires on none of 12 captured takes, three of which are
audibly nothing but mumbling. A derailed take collapses its vocabulary onto
a handful of codes, mostly quiet ones. Per 8 s window the share of frames
carrying a distinct code runs 0.45-0.63 on 18 control takes (the nine voices
x two real texts at the production cfg_alpha), 0.18 on a partial
derailment, and 0.03 at the known-bad cfg_alpha 0.3. The gate sits at 0.25.
See `utils/code_loops.py` for the thresholds and the measurement behind
each.

Two properties worth knowing before turning it on:

* **A missing dump rejects nothing.** A stock, unpatched server merges
  `voxtral_dump_codes` and ignores it, so the absence of a dump reads as "no
  opinion". Pointing the gate at the wrong server costs nothing rather than
  failing a whole run.
* **The raw dumps are deleted as they are read.** They are 925 B/s of audio,
  so ~11.7 GB and 600k files over the corpus if they were kept.

Each attempt appends one row to `code_qc_sidecar.jsonl` beside the audio,
accepted takes included, carrying the measured features **and** the semantic
track (base64 int16, ~500 bytes for a 15 s clip, ~480 MB over the corpus).
Accepted takes are recorded because the thresholds are calibrated on 30
takes and a file holding only the rejects cannot say what a good clip looks
like; the stored codes are what let any feature be re-derived later and
compared against stage 06's CER gate, which is the real label.
`--no-store-semantic-codes` keeps the features and drops the track.

## `03_qc_cpu_stt.py`

A sidecar process that tails a running generation and STT-checks a sample of
its clips on the host's **spare CPU cores**, so a systematic defect shows up
within hours instead of at stage 06, after the whole corpus is on disk.

It cannot check everything, and the arithmetic is the whole design. Measured
on this host (i7-7800X, 6 cores) under the live v2 run, the int8 UltiMed
Parakeet ONNX export reads audio at 7.7x realtime on 4 threads and 9.4x on
3 threads x 2 workers, while stage 05 *writes* audio at ~28.9x realtime. So a
full inline check would fall ~1,900 hours of audio behind over a 485k-clip
stage. Instead it checks two populations:

- **every clip that needed a retry**, i.e. every clip with a row in
  `cfg_alpha_sidecar.jsonl` (~0.3% of the stream). Those takes were rejected
  once for their duration and re-drawn one rung down the cfg_alpha ladder, so
  they are the only clips nothing has looked at twice.
- **a deterministic ~5% sample** of the rest (`--sample-percent`), drawn by
  hashing the clip stem with its own salt so the sample is stable across
  restarts and uncorrelated with the voice assignment, which is what makes the
  per-voice CER table it prints comparable.

Together ~5.3% of the stream, about 1.5x realtime of STT, which one worker at
3 threads delivers with most of its duty cycle to spare and which costs the
generator ~1-2% of its throughput (the four benchmark runs that measured the
numbers above cost 7-10% while they ran, so the niceness and the single worker
are not decoration).

```bash
./03_qc_cpu_stt.py                                  # tail the live run
./03_qc_cpu_stt.py --backfill                       # and work through what is on disk
./03_qc_cpu_stt.py --stage 01_dictionnary \
    --sample-percent 0 --backfill --once            # one shot, retried clips only
```

Output is one JSON line per clip in `qc_stt_sidecar.jsonl` beside the audio
(`reason`, `voice`, `cfg_alpha`, `cer`, `wer`, `tail_cer`, `duration`,
`stt_s`), append-only, and a restart resumes from it. A clip over either gate
(`--cer-threshold` 0.08, `--tail-cer-threshold` 0.12, stage 06's own numbers)
is logged at WARNING and keeps its transcript in the row.

The scoring is stage 06's (`06_hotfixes/01_compute_stt.py::normalize_for_scoring`
and `compute_metrics`), imported rather than reimplemented, so "over the gate"
means the same thing here and in the real QC loop.

A flagged row also carries `defect`, the SHAPE of the failure, from the two
texts alone and at no extra cost: `loop` (a word n-gram repeats far more than
the label's own repetition does, i.e. the model derailed into saying the same
thing over and over), `overrun` (materially more words, new material rather
than a repeat), `truncated` (materially fewer), `wrong_words` (the same amount
of speech at a normal rate, saying something else). `--defect-mix` reports the
mix over the rows already on disk without transcribing anything, so it needs
neither the model nor the audio and can run while the real QC holds the CPU:

```bash
./03_qc_cpu_stt.py --stage 01_dictionnary --defect-mix --log-file ""
```

The split matters because it bounds what a cheaper detector could ever replace.
Of the first 51 flagged dictionary clips (2026-10-15), 73% were `wrong_words`,
which is fluent speech with no acoustic signature at all, so only STT can see
it. Among the severe ones (CER >= 0.25) it becomes 42% `wrong_words` against
50% `loop` or `overrun`, which a waveform-only detector can find. Every loop
and overrun in that batch belonged to one voice, `h_frm_chf`.

**This is not a quality gate.** The default model is fine-tuned on v1 of this
very dataset, so it has been taught this corpus's vocabulary and will agree
with audio a neutral listener would reject. Read it as a gross-error detector
(text dropped, syllables looped, the wrong text spoken, the clip truncated).
For an impartial read, point `--model-dir` at the base export
(`parakeet-tdt-0.6b-v3-redux-onnx`) or score with Whisper through stage 06.

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
  variants over speakers. The key is `group:index`, where the index is
  `chunk_index` when the row has one (PARHAF/PARROT, whose `variant_index` is
  always 0) and `variant_index` otherwise, so the mode is not silently the same
  as `group` on the document stages.
- **What v2 actually runs** (decided 2026-10-14): `row` for `01_dictionnary`,
  `02_drugs` and `07_acronyms`, `group` for `03_PARHAF` and `04_PARROT`. A term's
  variants are independent sentences, so hearing them in several accents is worth
  more than one voice per term; a document's chunks are one report and keep one
  reader. The mode is recorded per audio folder in `voice_assignment.json`, so
  mixing them across stages needs nothing special.
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
