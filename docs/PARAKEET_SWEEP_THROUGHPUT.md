# Parakeet sweep throughput: what was measured and why the sweep is configured this way

The T2 judge sweep transcribes the whole v2 corpus (~602k clips) with
`parakeet-tdt-0.6b-v3-ultra` so that loops and repetitive gibberish can be caught, which
the Whisper judge structurally cannot see (it deduplicates repeated speech). At the rate
the sweep first launched at, that pass was a full day of GPU time. This file records what
was measured on 2026-10-25/26 to get it to a few hours, so the next person does not
re-run the grid.

Hardware: RTX 3090 Ti (24,564 MiB, also driving the display), 12 cores, 123 GB RAM.
Corpus sample: 300 real `01_dictionnary` clips, 24 kHz FLAC, p50 duration 20.3 s, max 59 s.
Written with Claude Code.

## The configuration the sweep runs

```
./03_qc_cpu_stt.py --providers CUDAExecutionProvider \
  --quantization fp16 --decoder-quantization fp32 \
  --batch 64 --batch-sec 600 --workers 6 --threads 2
```

with `LD_LIBRARY_PATH=/tmp/ort-cuda-shim:/usr/local/cuda-12.8/lib64` so onnxruntime-gpu
finds the uv-cached cuBLAS/cuDNN libraries.

Measured 25.1 clips/s end to end against 4.3 clips/s for the configuration the sweep
first launched with (fp32 encoder, one clip per call, `cudnn_conv_algo_search=EXHAUSTIVE`),
i.e. 5.8x, which turns a 36 h pass into ~6 h.

## The two things that actually mattered

### 1. onnx-asr's batching is encoder-only

This is the single biggest lever and it is invisible from the API. `model.recognize([w1,
w2, ...])` looks like a batched call and the encoder genuinely is batched, but
`NemoConformerTdt._decode` (onnx-asr 0.12) runs `decoder_joint` with
`encoder_outputs[None, :, None]` and `targets [[tok]]`: batch size 1, one encoder frame
at a time, per clip.

Concretely, a 64-clip batch of 20 s clips is 64 x 250 frames at 12.5 fps, so ~16,000
sequential ONNX calls, each a few hundred microseconds of launch latency. The GPU sits
~74% idle waiting for Python. Measured 10.0 clips/s, with `nvidia-smi` reading 26%
utilization.

`batched_tdt_decoding` steps the whole batch in lockstep instead, one call per step with
every still-running clip as a row (8 clips -> `encoder_outputs` (8, 1024, 1), `targets`
(8, 1), states (2, 8, 640)), so the same batch costs ~250 calls rather than ~16,000.
Measured 25.1 clips/s in the sweep, 38.5 clips/s in the bench harness that did no scoring.

It is **copied** into `05_generate_audio/03_qc_cpu_stt.py` from
`parakeet_web/scripts/wer-quants.py`, by explicit decision on 2026-10-26, after the
alternative (importing it across repos by relative path) was rejected as fragile: that
path breaks as soon as `--model-dir` points at a bare HF download. **It is therefore the
same function in two repos and a fix to either belongs in the other.** Both copies carry
a comment saying so.

One correctness condition: batching reproduces the per-clip result only if the decoder's
rows cannot interact. True for the fp32 decoder, NOT for the dynamic-int8 one, which
derives a single activation scale per call, so a clip could flip a near-tie word
depending on its batch mates. Hence `--decoder-quantization fp32`.

### 2. The batch cap must be in padded seconds, not clips

A batch costs what it allocates, which is `clips x the longest clip in the batch`, not
`clips`. With p50 20.3 s clips a 400 s cap binds at ~14 clips, which is why `--batch 16`,
`24`, `32` and `48` all produced the same 21 batches over 300 clips and measured the same
rate. `--batch` is a ceiling the seconds cap almost always reaches first.

Length-sorting over a window of several batches is what makes the cap pay: a batch then
pads to a length close to its own. Measured 39.5 clips/s sorted against 31.5 unsorted
(-20%), and offline, sorted wastes 23.4% of its padded samples against 41.7% unsorted.
(An earlier claim that unsorted batches OOM was wrong: it was two bench processes sharing
the card.)

## Peak VRAM is a floor, not a budget

onnxruntime's BFC arena never returns GPU memory to the driver, and a sweep hands it
thousands of distinct padded shapes, so its high-water mark climbs for hours. The bench
cells were 300 clips long and too short to see it:

| cap   | peak VRAM over 300 clips | after ~2,500 sweep clips |
| ----- | ------------------------ | ------------------------ |
| 600 s | 8.6 GB                   | ~15-23 GB                |
| 900 s | 13.0 GB                  | 23.4 GB, then failures   |

At a 900 s cap the sweep reached 23.4 GB of a 24.6 GB card within 2,500 clips and the
widest batches then died on a single 1.47 GB pre-encode conv buffer. Two fixes, both in
place: the cap is 600 s (10% slower than 900 in the bench, 35.7 against 39.9 clips/s), and
`run_group` retries a failing batch **in halves** down to a single clip. Before that
retry existed, the handler sat around the whole 192-clip sort window and one failure
charged 192 clips to `failed`, unchecked: the first full-corpus launch lost 576 clips that
way in three minutes.

`arena_extend_strategy=kSameAsRequested` slows the climb (the default doubles on a miss,
and one configuration touched 24.0 GB where the same work fits in 13 GB) but does not stop
it. A 1300 s cap fails outright, asking for one 4.24 GB buffer.

## Precision: fp16 encoder, fp32 decoder

Throughput, 300 clips, fp32 decoder throughout, HEURISTIC conv search:

| encoder | one clip per call | cap 600 s | cap 900 s | peak VRAM at cap 900 |
| ------- | ----------------- | --------- | --------- | -------------------- |
| fp32    | 4.71              | 29.85     | -         | 17.4 GB              |
| fp16    | 3.65              | 35.71     | 39.86     | 12.8 GB              |
| int8    | 8.31              | 30.19     | -         | 15.7 GB              |
| w4a8    | 7.93              | -         | -         | -                    |

The ranking **inverts** with batch size, which is why the model repo's own README picks
differently: at one clip per call fp16 is the slowest (launch-latency and cast bound),
while batched the work is dense GEMMs where fp16 hits tensor cores and int8's
`MatMulNBits` weights must first be dequantized.

Accuracy decided it, not speed. Drift against an fp32 / one-clip-per-call reference on
the same 300 clips:

| encoder          | identical transcripts | mean CER | max CER |
| ---------------- | --------------------- | -------- | ------- |
| fp32, batched    | 299/300               | ~0       | 0.003   |
| fp16, batched    | 282/300               | 0.0005   | 0.017   |
| int8, batched    | 274/300               | 0.0007   | 0.047   |
| w4a8, batched    | 113/300               | 0.0069   | 0.081   |

Batching itself is lossless (299/300). w4a8 is disqualified: its max CER straddles this
sweep's own 0.08 gate, so it would manufacture flags. FLEURS agrees int8 is free
(10.95% macro WER, same as fp32) and w4a8 costs +0.7 points, +2.6 on long audio.

End to end, the batched path was checked against the one-clip-at-a-time path on 318 real
`07_acronyms` clips: 317 rows matched on every field, `over_gate` matched on all 318, and
the one difference was a hyphen (`pré-gabaline` for `prégabaline`).

## Things that did not help, so do not re-try them

- **`cudnn_conv_algo_search`**: the model repo's note that `EXHAUSTIVE` "left the GPU ~idle
  at ~12 s/clip" does not reproduce here. `HEURISTIC` measured 0.2293 against 0.2347 s per
  clip, ~2%, with bit-identical transcripts over 60 clips. It is set because there is
  nothing to lose, not because it was the bottleneck.
- **More loader threads**: 4 / 6 / 8 / 12 gave 39.5 / 41.8 / 39.9 / 40.1 clips/s. The
  loaders are not the constraint; `wall_s_per_clip` was within 1% of `gpu_s_per_clip` in
  every cell.
- **A second process**: two streams of the batched decoder measured 21.72 + 21.60 = 43.3
  clips/s combined, +8% over one stream for 4 GB more VRAM, because without CUDA MPS the
  kernels of two contexts time-slice rather than overlap. With the *unbatched* decoder
  (GPU 26% busy) a second process was worse than useless: the first stream dropped from
  10.0 to 8.4 clips/s and the card filled to 24.1 of 24.6 GB.
- **`--batch` above ~32**: the seconds cap has already cut the batch (see above).

## Known headroom, not taken

At 25.1 clips/s the sweep is no longer GPU bound: `nvidia-smi` reads 0% between batches
because scoring (jiwer WER/CER, the tail and sliding-window alignments, the loop count)
runs single-threaded in the one batch worker, serially with the GPU. Solving
`1/25.1 = 1/38.5 + 1/x` puts scoring at ~72 clips/s, so overlapping the two should
recover most of the gap to the bench's 38.5. The work is independent per clip and the
loader pool is already there; it was left alone rather than risk a concurrency change on a
running overnight pass.
