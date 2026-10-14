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
# 32 requests in flight, the top of the server's CUDA-graph capture sizes
# ([1,2,4,8,16,32], with the batch padded UP to the next one, so 17..32 in flight all
# execute the same 32-row graph and the Euler loop runs once over the whole batch).
# NOT the 10 v1 started with (v1 itself moved to 15 after a day and stayed 8 days).
# Measured on 2026-10-14, same server, same clips:
#   10 in flight  2,070 clips/h  17.4 s per clip   (v1's log)
#   15 in flight  3,042 clips/h  17.4 s per clip   (v1's log, same evening)
#   16 in flight  3,860 clips/h  14.9 s per clip   23.1x realtime
#   32 in flight  4,610-4,960    25.1 s per clip   28-30x realtime
# So +50% from 10 to 15 at identical latency, then +22% more at 32 for ~70% more
# latency per clip (48 -> 89 ms per audio frame), which a batch run does not care
# about. Do NOT go above 32: a batch larger than the biggest captured graph has no
# graph to pad into. The longer per-clip latency is also why --timeout defaults to
# 600 s: a clip running to the frame cap takes ~365 s at this concurrency.
CONCURRENCY="${CONCURRENCY:-32}"
URL="${URL:-http://127.0.0.1:8003}"
# The per-clip frame budget, ON by default: the server is asked for only as many
# frames as each text can plausibly need (3 s + chars/5 at 12.5 Hz), so a take that
# starts looping syllables is stopped in seconds instead of running to the 327.68 s
# cap, and is retried instead of landing rambling audio in the dataset under a short
# label. The rate is the SLOWEST voice's slow tail, not the average one: the ten v2
# voices read at medians 10.7 to 16.4 chars/s, and a bound fitted on the average
# rejected h_frm_chf at 3.05% against 0.00-0.31% for the other nine (see
# utils/frame_budget.py and the runbook, step 1). The same run also rejects clips that
# are implausibly SHORT for their text (below chars/30). TOKEN_BUDGET=0 turns the
# upper bound off for a run at a lower cfg_alpha, where rambling is rare; the floor
# stays, since it needs nothing from the server.
TOKEN_BUDGET="${TOKEN_BUDGET:-1}"
# Extra attempts for a clip whose duration is rejected. Each attempt is a genuinely
# new take (Voxtral's only randomness is the fresh Gaussian each frame's flow-matching
# ODE starts from) AND one rung lower on the cfg_alpha ladder (2.0, 1.8, 1.6..., floor
# 1.2), so a rambling take is usually fixed by one retry and a reproducibly bad text
# gets a genuinely different draw. Any clip that lands below $CFG_ALPHA is recorded in
# cfg_alpha_sidecar.jsonl beside the audio, which is what stage 06 publishes as that
# row's cfg_alpha. Pass --cfg-alpha-step 0 to the script to pin every retry at
# $CFG_ALPHA instead. RETRIES=0 reverts to "one attempt, then leave it for the next
# re-run".
RETRIES="${RETRIES:-2}"
# An `if/else` rather than `[ ... ] && ...`: the script runs under `set -e`, where a
# trailing test that FAILS is the last command of the line and kills the run. The
# flag is passed EXPLICITLY either way, so the log shows which regime a run used
# instead of leaving it to the client's default of the day.
if [ "$TOKEN_BUDGET" = "1" ]; then
  budget_args=(--auto-token-budget)
else
  budget_args=(--no-auto-token-budget)
fi
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

echo "v2 run: ${#STAGES[@]} stage(s), cfg_alpha $CFG_ALPHA, concurrency $CONCURRENCY, token budget $TOKEN_BUDGET, retries $RETRIES, log $LOG_FILE"
echo "stop it any time with Ctrl-C: in-flight clips finish, queued ones are dropped, and"
echo "re-running this same command resumes on exactly the clips that are missing."
# The duration bounds are the only check the audio gets in here, and they cannot
# see a clip of plausible LENGTH whose text is wrong. 03_qc_cpu_stt.py transcribes
# the retried clips and a 5% sample on the idle CPU cores while this runs, for ~1-2%
# of throughput; it is a separate process so stopping either leaves the other alone.
echo "optional, in another tab: ./03_qc_cpu_stt.py tails this run and STT-checks the retried clips plus 5% of the rest on the spare CPU cores."
run_start=$SECONDS
with_failures=()
for s in "${STAGES[@]}"; do
  m="${MODE[$s]}"
  echo
  echo "=== $s (--voice-mode $m) $(date '+%Y-%m-%d %H:%M:%S') ==="
  stage_start=$SECONDS
  # `|| rc=$?` instead of letting `set -e` act, because the two non-zero exits mean
  # opposite things: 130 is "the operator stopped this" and must stop everything, while
  # 1 is "some clips failed" and must NOT abandon the remaining stages over a handful of
  # transient 500s out of 600k requests (a later re-run retries exactly those, since
  # resume matches clips by filename).
  rc=0
  uv run 01_generate_audio.py \
    --input  "../$s/generated_dataset.jsonl" \
    --output "../$s/generated_audios" \
    --url "$URL" \
    --voice-mode "$m" \
    --cfg-alpha "$CFG_ALPHA" \
    --concurrency "$CONCURRENCY" \
    --retries "$RETRIES" \
    "${budget_args[@]}" \
    --log-file "$LOG_FILE" || rc=$?
  elapsed="$(date -u -d "@$((SECONDS - stage_start))" '+%H:%M:%S')"
  case "$rc" in
    0) printf '=== %s done in %s ===\n' "$s" "$elapsed" ;;
    130)
      printf '=== %s interrupted after %s: re-run to resume ===\n' "$s" "$elapsed"
      exit 130 ;;
    *)
      # A dead server reports the same exit 1 as a few bad clips, and continuing would
      # then burn through the remaining stages failing every row. One health probe tells
      # the two apart for free.
      if ! curl -fsS -m 5 "$URL/health" >/dev/null; then
        echo "=== $s exited $rc after $elapsed and $URL is not answering: stopping ===" >&2
        exit 1
      fi
      echo "=== $s exited $rc after $elapsed (clips failed, server still up); continuing ===" >&2
      with_failures+=("$s") ;;
  esac
done
printf '\nall stages done in %s.\n' "$(date -u -d "@$((SECONDS - run_start))" '+%H:%M:%S')"
if [[ ${#with_failures[@]} -gt 0 ]]; then
  printf 'clips failed in: %s. Re-run this script to retry just those, then go to the QC loop.\n' \
    "${with_failures[*]}" >&2
  exit 1
fi
printf 'Next: the QC loop, 06_hotfixes/driver.sh (see docs/V2_REGENERATION_RUNBOOK.md).\n'
