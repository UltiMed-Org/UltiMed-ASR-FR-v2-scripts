#!/usr/bin/env bash
# Run 03_qc_cpu_stt.py on the GPU. Wrapper rather than a note in a docstring, because
# the invocation is not discoverable: onnxruntime's CUDA provider needs three things
# that the script's own pinned deps deliberately do not supply.
#
# WHY the GPU at all. The CPU judge is the only instrument in this project that can see
# a loop, since whisper's decoder suppresses repeated n-grams and scores a doubled
# clause as a perfect read. A whole-corpus parakeet pass is therefore worth wanting, and
# on the CPU it needs ~230 h against ~10-20 h here (measured 2026-10-25: 0.230 s/clip
# wall at 2 workers, 0.444 s of STT for 38.4 s of audio, i.e. ~86x realtime per worker,
# at 35% GPU utilisation, so it scales with --workers).
#
# WHAT it adds, and why each piece:
#  1. onnxruntime-gpu. The script pins onnx-asr[cpu] on purpose: it is built to run
#     BESIDE a GPU job without stealing the card. The GPU wheel is opt-in, here.
#  2. nvidia-cudnn-cu12. onnxruntime-gpu ships the CUDA kernels but not cuDNN.
#  3. A shim directory of UNVERSIONED symlinks. The nvidia wheels install
#     libcudnn.so.9, while onnxruntime dlopens the bare "libcudnn.so" and fails with
#     NOT_IMPLEMENTED / "cuDNN is unavailable" when it is missing. The shim is rebuilt
#     from whatever the uv cache currently holds, so it survives a dependency bump.
#
# Usage: every argument is passed through to the judge. --providers and the precision
# are set here; pass --quantization to override.
#
#   ./run_gpu_judge.sh --backfill --once --sample-percent 100 --no-retries \
#       --keep-transcripts --model-dir ../../parakeet-tdt-0.6b-v3-ultra-onnx \
#       --out-name qc_parakeet_ultra.jsonl --workers 6 --threads 2
#
# NOTE the precision: pass an EMPTY --quantization (the default below) for the ultra
# export. Its fp16/ folder ships an encoder only, so fp16 cannot resolve a decoder and
# the unsuffixed pair under fp32/ is the only complete one.
#
# NOTE on VRAM: one session takes ~5GB, and it is shared across --workers, so raising
# the worker count costs almost nothing. Stop anything else holding the card first
# (`sudo docker compose stop crispasr crispasr-lb` frees ~18GB of idle whisper
# replicas), or the session fails to initialise.
#
# This file was written by Claude Code.
set -euo pipefail

cd "$(dirname "$0")"

WITH=(--with onnxruntime-gpu --with nvidia-cudnn-cu12)

# Resolve the nvidia wheels' lib directories inside an env built from the SAME
# requirements, so the shim points at the copies the judge will actually load.
NV_LIBS=$(uv run --quiet "${WITH[@]}" python -c '
import nvidia, os
base = list(nvidia.__path__)[0]
dirs = [os.path.join(base, d, "lib") for d in sorted(os.listdir(base))]
print(":".join(d for d in dirs if os.path.isdir(d)))
')
if [[ -z "$NV_LIBS" ]]; then
    echo "run_gpu_judge: found no nvidia/*/lib directories; is nvidia-cudnn-cu12 installed?" >&2
    exit 1
fi

SHIM="${TMPDIR:-/tmp}/ort-cuda-shim"
rm -rf "$SHIM"; mkdir -p "$SHIM"
shimmed=0
IFS=':' read -r -a libdirs <<< "$NV_LIBS"
for d in "${libdirs[@]}"; do
    for so in "$d"/lib*.so.*; do
        [[ -e "$so" ]] || continue
        base=$(basename "$so")
        ln -sf "$so" "$SHIM/$base"
        ln -sf "$so" "$SHIM/${base%%.so.*}.so"   # the unversioned name ORT dlopens
        shimmed=$((shimmed + 1))
    done
done
echo "run_gpu_judge: shimmed $shimmed library/libraries into $SHIM"

export LD_LIBRARY_PATH="$SHIM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# __pycache__ beside the scripts is read-only on this host.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${TMPDIR:-/tmp}/pyc}"

exec uv run "${WITH[@]}" ./03_qc_cpu_stt.py \
    --providers CUDAExecutionProvider --quantization "" "$@"
