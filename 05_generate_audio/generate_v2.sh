#!/usr/bin/env bash
# The v2 audio run, in one command: re-synthesize all five stages with the ten
# French voices of utils/voice_assignment.py.
#
#     ./generate_v2.sh                       # all five stages, in this order
#     ./generate_v2.sh 07_acronyms           # just one (or several) stages
#     CONCURRENCY=6 ./generate_v2.sh         # override a knob, see below
#
# Run it under tmux / screen: the full corpus is ~3,115 h of audio, i.e. about
# 8.7 days at the 15x realtime v1 sustained. Every stage is resumable (clips
# already on disk are skipped by filename), so a crash, a Ctrl-C or a reboot
# costs only the clip in flight: re-run the same command.
#
# THE VOICE MODE IS PER STAGE, and that is the one thing not to get wrong here,
# because the voice is baked into the audio:
#   row   for 01_dictionnary, 02_drugs, 07_acronyms -- a group there is one TERM
#         and its ~8 written variants are independent sentences, so each one draws
#         its own voice (the term is then heard in ~5 accents, and a bad
#         voice/term pairing costs one clip instead of all of them).
#   group for 03_PARHAF, 04_PARROT -- a group there is one DOCUMENT and the clips
#         are consecutive chunks of it, so they must share a reader.
# See docs/V2_REGENERATION_RUNBOOK.md step 1.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Knobs, all overridable from the environment. cfg_alpha 2.0 is the measured top
# of the usable band (1.2 to 2.0); stage 06's retry ladder walks DOWN from it.
CFG_ALPHA="${CFG_ALPHA:-2.0}"
CONCURRENCY="${CONCURRENCY:-10}"          # what v1 ran, and what its throughput was measured at
URL="${URL:-http://127.0.0.1:8003}"
# A separate log from v1's tts_generate.log on purpose: that file is the only
# record of v1's throughput and is what the runbook's ETA was measured from.
LOG_FILE="${LOG_FILE:-$PWD/tts_generate_v2.log}"

declare -A MODE=(
  [01_dictionnary]=row
  [02_drugs]=row
  [07_acronyms]=row
  [03_PARHAF]=group
  [04_PARROT]=group
)
# Default order: the three term stages first (the bulk, and the one whose voice
# spread is worth eyeballing early), then the two document stages.
STAGES=(01_dictionnary 02_drugs 07_acronyms 03_PARHAF 04_PARROT)
if [[ $# -gt 0 ]]; then
  STAGES=("$@")
  for s in "${STAGES[@]}"; do
    [[ -n "${MODE[$s]:-}" ]] || { echo "unknown stage: $s (known: ${!MODE[*]})" >&2; exit 2; }
  done
fi

# Fail in one second rather than at the first clip: the server is not part of this
# repo and has to be started by hand (see the runbook preflight).
curl -fsS -m 5 "$URL/health" >/dev/null \
  || { echo "no TTS server on $URL -- start it first: sudo docker compose up -d --build voxtral-tts" >&2; exit 1; }

echo "v2 run: ${#STAGES[@]} stage(s), cfg_alpha $CFG_ALPHA, concurrency $CONCURRENCY, log $LOG_FILE"
run_start=$SECONDS
for s in "${STAGES[@]}"; do
  m="${MODE[$s]}"
  echo
  echo "=== $s (--voice-mode $m) $(date '+%Y-%m-%d %H:%M:%S') ==="
  stage_start=$SECONDS
  uv run 01_generate_audio.py \
    --input  "../$s/generated_dataset.jsonl" \
    --output "../$s/generated_audios" \
    --url "$URL" \
    --voice-mode "$m" \
    --cfg-alpha "$CFG_ALPHA" \
    --concurrency "$CONCURRENCY" \
    --log-file "$LOG_FILE"
  printf '=== %s done in %s ===\n' "$s" "$(date -u -d "@$((SECONDS - stage_start))" '+%H:%M:%S')"
done
printf '\nall stages done in %s. Next: the QC loop, 06_hotfixes/driver.sh (see the runbook).\n' \
  "$(date -u -d "@$((SECONDS - run_start))" '+%H:%M:%S')"
