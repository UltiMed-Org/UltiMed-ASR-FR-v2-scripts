# How UltiMed v2's audio was made as clean as possible

A plain-language account of every safeguard between "an LLM wrote a sentence" and "that sentence's audio ships in the training set", written for a software engineer who has never heard of this project. Written with Claude Code on 2026-10-27, from the measurements recorded in [`V2_QC_PHASE_PLAN.md`](V2_QC_PHASE_PLAN.md), [`V2_REGENERATION_RUNBOOK.md`](V2_REGENERATION_RUNBOOK.md) and [`../05_generate_audio/README.md`](../05_generate_audio/README.md), which hold the full detail and the commits. Where this page and those disagree, they win.

## 1. What UltiMed is, in one minute

UltiMed is a French **medical speech-recognition** project. The goal is a speech-to-text model (NVIDIA's `parakeet-tdt-0.6b-v3`, fine-tuned) that writes "amoxicilline 500 mg trois fois par jour" correctly when a doctor dictates it, where an off-the-shelf model writes "ami au si il line".

Fine-tuning needs pairs of `(audio, exact transcript)`, tens of thousands of hours of them, and real French medical dictation with exact transcripts is not available at that scale. So the dataset is **synthetic**:

1. Take text rich in medical vocabulary: a ~62.6k-term medical dictionary, drug databases, real clinical documents (PARHAF), radiology reports (PARROT), and 511 medical acronyms.
2. Have an LLM turn it into natural dictation-style sentences. That sentence is the **label**.
3. Have a text-to-speech (TTS) model read the sentence aloud. That is the **audio**.
4. Train the speech recognizer on `(audio, label)`.

One row of the dataset looks like this (simplified):

```python
{
    "audio_filepath": "dictionary/012345_0002_amoxicilline.flac",
    "text": "Prescription d'amoxicilline 500 mg trois fois par jour pendant sept jours.",
    "duration": 5.84,                 # seconds
    "voice": "h_frf_ptm",             # which synthetic speaker read it
    "cfg_alpha": 2.0,                 # a TTS generation setting, see section 4
    "stt_transcript": "Prescription d'amoxicilline 500 mg 3 fois par jour pendant 7 jours.",
    "cer": 0.004,                     # how far that transcript is from the label, see section 5
    "qc_status": "original",          # or "improved" if the clip was regenerated
}
```

v1 of the dataset (601k clips, 3,104 h) was read by a single voice. **v2 regenerates the entire corpus with nine voices**, so the recognizer does not learn one speaker's quirks. About 602,700 clips were regenerated, and every one of them had to be checked again.

## 2. Why synthetic audio cannot be trusted as-is

The TTS model is [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603), served locally with vLLM. It is good, but it fails in specific, recurring ways. And a training clip whose audio does not say its label teaches the recognizer to *hallucinate*: to write words it did not hear. For an ASR dataset that is the worst kind of noise.

The failure modes we hunted:

| failure | what you hear | why it is dangerous |
| --- | --- | --- |
| **loop** | the sentence starts fine, then a clause repeats ("une ARM veineuse 3D T1 écho de gradient, une ARM veineuse 3D T1 écho de gradient") or syllables repeat until a length cap | the model learns to skip audio it heard |
| **runaway** | the take never stops: 200+ s of babble for a 20 s sentence | same, plus it wastes GPU time |
| **derail / mumbling** | plausible-sounding French gibberish, often quiet, after a correct start | labelled words that were never spoken |
| **truncation / fade** | the take stops, or fades out, before the end of the sentence | the label's last words have no audio |
| **dropped phrase** | five words in the middle are skipped | invisible in a whole-clip average |
| **preamble babble** | noise or the opening words twice before the real sentence | an insertion nothing else measures |
| **misreading** | "500 mg" read as "cinq zéro zéro m g" | the recognizer learns a wrong mapping |

No single check catches all of these, which is why the pipeline below has several layers, each aimed at a different shape.

## 3. Layer 0: the text the TTS reads is not the text the model learns

Every row carries two texts:

```python
asr_training_target = "Le patient reçoit 2 mg/kg de kétamine à J3."            # the label
asr_training_source = "Le patient reçoit deux milligrammes par kilo de kétamine à jour trois."  # what the TTS reads
```

The source is derived from the target **by code, not by a second LLM call** (`utils/voxtral_normalize.py`): it spells out the units, digits and Roman numerals that the TTS mispronounces, and it sends anything it does not recognize to a review queue instead of guessing. Because the transform is deterministic, the two texts cannot drift apart in meaning, which is how synthetic corpora usually get poisoned.

A few related fixes on the text side:

* **Exotic Unicode spaces** (` `, ` `, ... common in French typography before `:` or `%`) make Voxtral loop. Measured: median WER 47.7 pct down to 10.5 pct, and unusable takes 16 of 48 down to 3 of 48, once they are replaced by ordinary spaces before synthesis.
* **Pronunciation respellings** (`oestrogène` read as `éstrogène`) go into the source only, so the label keeps the medical spelling.

## 4. Layer 1: guards at generation time

### Choosing the voices by ear, then by measurement

Voxtral ships 20 voices. We also built "hybrid" voices by mixing two voices' internal codes (one voice sets the timbre, the other shifts pronunciation and stability). Hybrids read a typical sentence *better* than their donors, but **degenerate far more often**: on 332 French sentences, the 20 shipped voices produced 1 runaway in 6,640 takes, the 18 hybrids produced 142 in 5,976.

The nine voices in v2 were selected in two listening rounds, the second on hard medical text, because passing on general French did not predict passing on medical French (the second round reversed 6 of 16 verdicts). A tenth voice, `h_frm_chf`, was cleared by ear and then **dropped on data**: in the first 5 pct sample of v2 it put 14 of 180 clips over the error gate against at most 4 of 165 for every other voice.

### The guidance setting (`cfg_alpha`)

Voxtral's classifier-free guidance controls how hard the model sticks to the text. It was swept: 16 voices x 10 values from 0.1 to 5.0, all on one dense medical sentence, all listened to.

* Below ~1.2 the model stops following the text: at 0.5 and below, every voice rambled to ~2.7x the sentence's natural length.
* From 3.0 up the audio distorts and takes increasingly never stop (they ran to the 327.68 s cap).
* So v2 generates at **2.0**, the top of the usable band, and a retry walks **down** from there (2.0, 1.8, 1.6, ...), with a floor of 1.6 for hybrids in the late repair rounds since their usable band is 1.5 to 2.0.

### A per-clip length budget

A looping take keeps generating audio. So every request carries a **frame budget** computed from its text: roughly `(3 s + characters / slow_rate) x 12.5 frames per second`, where `slow_rate` is a deliberately pessimistic speaking rate measured **per voice** (the slowest voice reads 13.0 characters per second where the fastest reads 16.4, so a single global rate would reject the slow voice for being slow). A take that comes back pinned at its budget is a runaway and is redrawn. Example: a 317-character term once produced 202 s of audio; with the budget, that take is cut off at roughly 35 to 50 s depending on the voice, instead of running for minutes, and is redrawn.

### The code gate: catching mumbling from the model's own internals

A mumbling take can have a perfectly plausible length. But we patched the TTS server to dump the discrete codes it emits, and a derailed take **collapses its vocabulary**: per 8 s window, the share of frames carrying a distinct code is 0.45 to 0.63 on good takes, 0.18 on a partial derailment, 0.03 on a fully broken one. Takes under 0.25 are rejected at generation time and redrawn. This costs nothing: the codes already exist.

### An acoustic loop scan, without any recognizer

`04_qc_acoustic_loops.py` reads the waveform only. When a clause is said twice, the clip's self-similarity matrix grows a stripe parallel to its diagonal (frame `t` sounds like frame `t + lag`). Scoring that stripe flags 92 pct of loops at a cost of 0.9 pct false alarms, and it runs on CPU over 100 pct of the corpus (~7 h on 4 cores) while the GPU generates.

## 5. Layer 2: two independent listeners score every clip

Every clip is transcribed by a speech recognizer (the "judge"), and the transcript is compared to the label.

### CER, and why it needs folding

**CER** (character error rate) is the edit distance between the two strings divided by the label's length:

```python
label      = "500 mg trois fois par jour"
transcript = "500 milligrammes 3 fois par jour"
# Raw CER would charge "mg" vs "milligrammes" and "trois" vs "3" as errors.
# They are not: the audio is correct, the judge just writes it differently.
```

So before comparing, both sides go through `01_compute_stt.py`'s folds (units, digits, ordinals, Roman numerals, spelled acronyms, ...), using the same tables `voxtral_normalize` uses in the other direction. Each fold was adopted only after replaying it over already-scored clips and counting how many scores it improved against how many it made worse.

### Five gates, because an average hides things

A clip is **bad** if any of these trips:

| gate | threshold | catches |
| --- | --- | --- |
| `cer`, whole clip | >= 0.08 | general misreading |
| `cer_tail`, last ~30 s | >= 0.12 | truncation, fade, derail near the end |
| `cer_window`, worst 15 s window | >= 0.15 | a broken passage inside a long clip |
| `bad_run_chars`, longest dropped span | >= 30 characters | about five consecutive dropped words |
| `repeat_excess`, extra repeated 4-grams (parakeet only) | >= 4 | loops (see below) |

A concrete case of why the window matters: a 180 s document read correctly except for one dropped phrase scores `cer` 0.04 overall, comfortably "clean". Its worst 15 s window does not.

The window thresholds are measured, not guessed: on 19,958 clean clips, `bad_run_chars` maxed out at 26 (gate 30), and `window_cer`'s p99 was 0.061 (gate 0.15 costs 1 false positive in 20k).

### Two judges, because one of them cannot hear loops

The first judge is OpenAI's Whisper (`large-v3-turbo`, run locally through whisper.cpp). It read all 602,789 clips (estimated at about 31 hours of GPU from v1's throughput).

Then something surprising showed up: **Whisper's decoder collapses repeated speech.** Hand it audio that says a clause four times, and it writes the clause once. The transcript then matches the label almost perfectly, so every Whisper-based gate calls a looping clip clean.

The second judge, NVIDIA's `parakeet-tdt-0.6b-v3` in its "ultra" variant, does not collapse: it writes the repetition out. It also read the whole corpus, and it is what feeds the `repeat_excess` gate. It is not used alone either, because it misreads medical vocabulary more often than Whisper. The loop rule is therefore a **conjunction**: parakeet rejects the clip *and* what is wrong is a repeat.

How we know that conjunction is the real defect, with no judge at all: a clip that says something twice takes longer than its text needs. Clips meeting both conditions read at **-14 to -20 pct characters per second** below length-matched clean clips; clips with repeats but a clean CER sit at about -1 to -4 pct, i.e. normal.

A third judge, non-turbo Whisper `large-v3`, served as a second opinion on the ~12.7k clips the first sweep flagged.

## 6. Layer 3: repair instead of discard

A bad clip is not deleted. It is **redrawn**:

1. The TTS synthesizes 5 new takes of the same sentence (different random seeds, `cfg_alpha` walking down the ladder).
2. The judge scores all 5.
3. The best take replaces the original only if it beats it by at least 0.02 CER (so a 0.090 clip is not "fixed" by a 0.088 take).
4. Still bad? It goes back for another round. When a whole round of draws fails, it climbs a **voice ladder**: the same sentence, read by one of the voices measured to fail least (two sturdy hybrids, then the two shipped French voices).
5. Only a clip that fails the entire ladder is marked `exhausted`, and exhausted clips are dropped from the release.

The GPU cannot hold the TTS server and the recognizers at once, so a driver script alternates them: one TTS pass draws takes for every queued clip, then one STT pass scores them all, and the loop repeats until nothing is pending.

### Picking among good takes: prefer the one that is not suspiciously long

Since all 5 takes say the same sentence, their lengths should be close: speaking-rate jitter is small and symmetric, while a repeat only adds time. So among the takes that pass every gate, one more than 15 pct longer than the **low median** of the passing takes loses to one that is not, before CER is compared. Real example from today: for `033348_0000_neurologie_00037_t0_c1`, a 58.2 s take was kept over a 72.3 s one that scored a slightly lower CER.

It is a preference, never a veto. On 1,703 real same-sentence take pairs, length ratios form one smooth bump (median 1.12, p90 1.32) with no gap to cut at, so any hard threshold would throw away good takes. The anchor is the low median rather than the shortest take because anchoring on the fastest take demoted 33 pct of all takes (it was rewarding speed, not catching loops); the median demotes 9 pct.

## 7. The rounds, with numbers

| tier | what it did | outcome |
| --- | --- | --- |
| T0 | during generation: frame budget, code gate, acoustic scan, CPU spot checks | runaways and derails redrawn before they reached disk |
| T1 | Whisper turbo read all 602,789 clips | early flag rates: 0.99 pct of dictionary clips, but **22.6 pct of PARHAF** (long documents, mostly loops, mostly 5 of the 9 voices) |
| T2 | non-turbo Whisper re-judged the ~12.7k flagged clips | a second opinion, no audio touched |
| T3 | repair round on the clips the second opinion disputed | 1,624 clips promoted |
| T4 | loop round: 1,288 looping clips found with parakeet (1,287 of them hybrid voices) | most repaired; a duration audit then showed ~22 pct of promotions still carried a loop, because the round's judge (Whisper) cannot see loops |
| T5 | residue round: the 341 clips still looping after T4, plus 63 resurrected clips no judge had read (see section 8) | 391 promotions, 1 clip exhausted, and the final looping PARROT clip went from CER 0.090 to 0.002 |

After T5: **602,715 clips scored, 14,077 of them regenerated at least once, 1 exhausted.**

## 8. Bugs that would have shipped bad data silently

Most of the effort went into these. Each passed every test that existed when it was found, and each is now fixed and covered by a test.

* **Stale durations blinded the tail gate.** The score files kept v1-era clip durations for 208,235 rows (one stored 34.0 s for an 86.96 s file). The tail gate only runs on clips over 30 s, so it silently skipped **5,250 clips** that really were long, exactly the long PARHAF chunks most likely to derail at the end. Fixed by re-reading durations from the audio files before every release.
* **The release step was undoing that fix.** The final sync copied the stale durations *back* into the manifests, on a docstring's (wrong) promise that they were fresh. The refresh now runs immediately before the sync, and a test pins that order.
* **v2 inherited v1's verdicts.** A drop list of 63 "defective" clips was keyed by file name and measured on v1 audio. v2 reused the names, so the release would have dropped 60 good clips for defects that no longer existed. Rebuilt from v2 readings.
* **The judge setting was documented but never read.** `STT_MODEL` was in the docs, and the script used another variable, so "judge with model X" silently judged with whatever a config file held.
* **The judge that flags must be the judge that promotes.** A clip flagged by the strict judge but rescored by the lenient one can never be promoted (its lenient score is already low), so a round judged by the wrong model would burn hours and fix nothing. Each round now names its judge explicitly.
* **Dropped clips came back.** Some drop reasons (exhausted, flagged by the second recognizer, generated with guidance off) live in the scoring state, not in the row. The state follows the manifest it is pointed at, so the QC round after a drop forgot the dropped clips, and the next manifest rebuild found their audio on disk and re-added them with no record. The release gate caught 71 of them by their empty columns; 63 earlier ones had slipped back in and been re-read by Whisper, which passed 61 of them. Drops are now kept in a committed ledger (`99_hf_release/dropped_clips.jsonl`) and re-applied for as long as the audio is the same bytes, which put 134 clips back out (one had since been redrawn).
* **A "finished" run skipped a new queue.** The driver remembered completed runs by a fingerprint that ignored the list of clips to process, so T5's first launch saw "same run as T4, already done" and did nothing. The fingerprint now includes the list's contents.

## 9. What is dropped rather than repaired

* **Exhausted clips**: failed every draw on every rung of the voice ladder.
* **Preamble babble**: ~210 clips where parakeet hears an insertion before the sentence. They score clean under Whisper, so the repair loop would never pick them up, and no repaired take could be confirmed without another full sweep. The audio stays on disk.
* **Repeats both judges hear**: 206 clips found by the final audit on 2026-10-28, where turbo AND parakeet transcribe a clause the label says once as said twice. 172 of them are a date range (`009599_0000_chir_ortho_et_traumato_00263_t0_c0`: "hospitalisé du 14 août 2024 au 16 août 2024 au 16 août 2024"). On a 40 to 90 s clip one repeated clause costs only ~0.05 of CER, under the 0.08 gate, which is how they got through every round. The rule (`repeated` in `06_hotfixes/04_flag_asr_defects.py`, on by default) flags a stretch of 4 or more extra words built from the label's own word pairs, in both transcripts. Dropped for the same reason as the preamble babble: the repair loop's promotion compares CER, which barely sees the defect. The audio stays on disk.
* **55 takes made at `cfg_alpha` 1.0** (guidance off, the documented failure mode), from before the ladder had a floor. Measurably fine, dropped anyway because 0.009 pct of the corpus is cheaper than loosening the validator.

## 10. The release gate

`99_hf_release/08_validate_manifests.py` runs last and fails the export on any error. It checks that every row has every expected column filled (transcript, judge name, scores, voice, `cfg_alpha`, QC status, ...). It also checks that each stored score actually reproduces from the stored transcript and the label, that no row trips a gate on its own stored numbers, and that train, validation and test splits stay pinned to v1's so the old test set cannot leak into training.

## 11. What is still not covered

Honest limits, so nobody mistakes "passed QC" for "perfect":

* **Both judges are recognizers.** A mispronunciation that both happen to "correct" back to the label passes. The fine-tuned UltiMed model was deliberately kept *out* of judging for this reason: trained on the labels, it would snap a mispronounced word back to them.
* **No clipping/saturation detector** and no detector for a voice changing mid-clip.
* **Clips no parakeet reading covers**: 608 shipped clips (447 PARHAF, 154 dictionary) were judged by Whisper alone, which is blind to loops.
* **Number misreadings**: a wrong digit costs about one character of CER, so a take saying "145 sur 85" for "145 sur 95" passes. `037990_0000_pneumologie_00139_t0_c2` is a likely case (several numbers misread, all judges agree), kept because the second Whisper judge scored it 0.047.
* **A clip where most of the 5 draws loop** defeats the length preference (the median itself is a loop). The parakeet sweep after each round is what catches that residue.
* **35 pct of hybrid-voice clips were generated at `cfg_alpha` 1.3**, below the hybrids' measured band. On every metric they read as well as the rest, but that comparison is biased (bad 1.3 clips were redrawn at higher values and left the group), so this is a documented property, not a proven non-issue.
* **Trailing silence is left alone on purpose.** It also teaches the recognizer not to hallucinate on silence.

## 12. Release gate result

The last export chain finished on 2026-10-28 at 08:48. The gate (`08_validate_manifests.py`) checked 602,445 rows in `full.jsonl` and reported **0 error rows** and 54 warning rows. That is after the drop step removed the 206 repeated clips of section 9 and re-applied the drop ledger (section 8), which now remembers 341 clips; none of them is in any manifest. A separate run of the score check over 100 pct of the rows (instead of the gate's 2 pct sample) on the 2026-10-27 export found 2 acronym rows (`000400_000{1,2}_sa`) whose stored CER does not reproduce (0.0146 stored, 0.0098 recomputed, a bookkeeping mismatch on a clean clip) and 4 rows with a turbo CER of 0.08 or more, all cleared by the second Whisper judge.

The warnings, none of which blocks training:

- 43 rows were generated at a guidance scale above the 1.2 to 2.0 band (22 at 3.0, 21 at 2.5). These came from the cfg sweep of section 4: the lower values failed on those sentences, these passed both judges, so they ship.
- 59,687 source documents have clips in more than one split (for example `acronyms:1` has clips in train, val and test). The gate counts this as one summary row. It follows from the `distribute` grouping, which spreads a document's sentences across splits on purpose; only `atomic` documents (44,880 rows, PARHAF-style long texts) are kept whole. Whether val/test should be document-disjoint is an open decision, not a defect of the audio.
- Housekeeping, one summary row per category (5 categories, so 10 rows): stray files next to the audio (`.bak` copies of redrawn takes, for example 8,203 in PARHAF and 293 in drugs) and audio files on disk that no manifest row mentions (dropped takes, for example 34 in PARHAF and 87 in dictionary). They are invisible to training, which reads only the manifests.

The val/test slices of the training repo were rebuilt from these manifests in the same run.
