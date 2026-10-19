#!/usr/bin/env bash
# Unattended driver for the alternating single-GPU improvement flow. Loops:
#
#   switch to STT server -> run --mode stt (transcribe + score + promote candidates)
#   switch to TTS server -> run --mode tts (generate candidates for flagged clips)
#
# repeating until the --mode stt pass reports the dataset has converged (exit code 10:
# no clip anywhere still needs either pass), then moving on to the next manifest in INPUT
# (by default the main corpus first, then the separately licensed PARROT manifest).
#
# NO sudoers edit needed: launch this ONCE with sudo so it can flip docker without a
# per-command password, and it drops back to your normal user for each Python pass (so
# every output / backup / draft file stays user-owned, matching the UID the TTS/STT
# containers write as). Run under tmux / a systemd unit so it survives a disconnect; a
# single long-lived loop (not cron) so two passes can never overlap on the GPU.
#
#     sudo -E ./driver.sh
#
# -E carries your session env (WDOC_WHISPER_*, CRISPASR_COMPOSE) into the root process;
# alternatively (or in addition), put those in 06_hotfixes/.env (KEY=value lines), which
# is sourced below, and run plain `sudo ./driver.sh`. .env holds a token: gitignore it.
#
# Required env. The WDOC_WHISPER_* names are not arbitrary: I ran this against my
# own self-hosted Whisper server rather than a public API, so the endpoint, the
# bearer token and the model name all had to be supplied explicitly, and I reused
# the variable names my other tooling already had for that server. Point them at
# any OpenAI-compatible /v1/audio/transcriptions and it works the same.
#   CRISPASR_COMPOSE       your CrispASR docker-compose.yml
#   WDOC_WHISPER_ENDPOINT  STT base URL (/v1/audio/transcriptions is appended). Defaults
#                          to http://127.0.0.1:8002, the local CrispASR ASR stack
#                          switch_server.sh brings up, because the remote endpoint v1 ran
#                          against now answers 502.
#   WDOC_WHISPER_API_KEY   STT bearer token (falls back to $CRISPASR_API_KEY). The ASR
#                          service sets CRISPASR_API_KEYS, so a local run needs one too:
#                          only /health is public.
#   STT_MODEL              STT model name, and the KEY the readings are stored under on
#                          each row, which is what keeps the v2 plan's two judges apart:
#                          T1 (turbo) and T2 (large-v3) must land in different keys or the
#                          second opinion overwrites the first. Defaults to
#                          whisper.cpp/ggml-large-v3-turbo, what the local endpoint
#                          reports loading. WDOC_WHISPER_MODEL is the older name for the
#                          same thing and still works. Match how a dataset was already
#                          transcribed or resume re-transcribes it from scratch.
# Optional overrides (defaults match the manual runs):
#   TARGET_USER (defaults to whoever ran sudo) OUTPUT AUDIO_ROOT N_IMPROV
#   CER_THRESHOLD STT_PARALLEL TTS_PARALLEL TTS_VOICE GPU_POWER_LIMIT
#   CFG_ALPHA_START / CFG_ALPHA_STEP / CFG_ALPHA_FLOOR  TTS guidance ladder across
#                       the draws of one redo: draw N uses START + STEP * N clamped at
#                       FLOOR, so the defaults (2.0 / -0.2 / 1.2) walk DOWN 2, 1.8, 1.6,
#                       1.4, 1.2 over five draws, which is the measured usable band (see
#                       the note at the assignment). A negative START omits the knob,
#                       leaving the server's startup value every draw.
#   TAIL_CER_THRESHOLD  CER gate on the last ~30s of clips longer than 30s (looser than
#                       CER_THRESHOLD because the tail is a short window; negative
#                       disables that one gate)
#   WINDOW_CER_THRESHOLD  CER gate on the WORST ~15s window of a clip: the only one of
#                       the four that sees a phrase dropped in the MIDDLE of a long clip
#                       (five lost words are 0.14 of a 213-char window but 0.04 of a 48s
#                       PARHAF chunk, so no whole-clip gate catches them without flagging
#                       every clean long clip too)
#   BAD_RUN_THRESHOLD   gate on the longest CONTIGUOUS run of reference characters the
#                       recognizer never said, in characters. A count and not a rate,
#                       because a skipped phrase has a size and dividing it by the clip's
#                       length is exactly what hides it
#   None of those four carries a default HERE: unset means "whatever the scripts ship"
#                       (01_compute_stt.py's DEFAULT_*_GATE, which
#                       01_recursive_improvement.py reads for its own click defaults), so
#                       this file never becomes a second copy of a number that then drifts
#   MIN_IMPROVEMENT     how much better a draw must be to replace the original when it
#                       does not clear the gates (default 0.02, see the note below)
#   RESCORE=1           re-derive every stored CER under the CURRENT scoring rules before
#                       deciding what is bad, for after a normalization change (costs no
#                       STT call: stored transcripts are reused). One-time migration, so
#                       only the FIRST stt pass of each dataset gets it, see below.
#   RESCORE_ONLY=1      do ONLY that migration and stop: one pass per dataset that reads
#                       stored transcripts, re-derives their scores and flags what now
#                       reads bad, without transcribing anything. Never-transcribed clips
#                       are left for a normal pass. Makes no server call at all, so the
#                       GPU is left exactly as it is (no docker switch) and it is safe to
#                       run while the TTS server is loaded. Implies RESCORE.
#   RETRY_EXHAUSTED=1   give the clips a previous run gave up on (`exhausted`) another
#                       chance: they are re-judged, cleared if they now read clean, and
#                       put back in the tts queue if they do not. Also one-shot, and for
#                       a stronger reason than RESCORE: see the block below. The clips
#                       that passed and the ones already improved are left alone, unlike
#                       --force. Pair it with START_SEED past the previous run's range,
#                       or the redraws reproduce the draws that already lost.
#   START_SEED          seed of the first draw of a redo, +1 per draw (default 43, so a
#                       five-draw redo uses 43..47). The only reason to move it is a
#                       retry over clips that were already drawn for.
#   MAX_AUDIO_SECONDS   TTS frame cap in seconds (default 327.68 = the served 4096 frames
#                       at 12.5 Hz). A draw landing on the cap stopped for LENGTH, not at
#                       the end of the text, so it is discarded instead of promoted. Must
#                       match the server's VOXTRAL_MAX_TOKENS; 0 disables the check. With
#                       AUTO_TOKEN_BUDGET=1 (the default) this is only the fallback bound:
#                       each draw is held to its own per-clip budget instead.
#   AUTO_TOKEN_BUDGET   1 (default) sends every draw a per-clip frame budget sized from
#                       its text and its VOICE's measured reading speed, and discards a
#                       draw that comes back past it (looping) or below its floor (text
#                       not spoken). Same guard, same numbers, as the generation pass:
#                       utils/frame_budget.py. 0 falls back to MAX_AUDIO_SECONDS alone,
#                       which only catches the worst runaways, and only after paying ~6
#                       minutes of GPU for each.
#   VOICE_SWAP          1 (default) gives a clip that lost all MAX_ATTEMPTS draws one more
#                       round of draws read by ANOTHER voice of its roster, instead of
#                       giving up on it. Only term stages qualify (one voice per clip);
#                       a document stage assigns one reader per document, so swapping
#                       there would break the single-reader property and the pass skips
#                       it on its own, reading the mode from the voice_assignment.json
#                       beside the audio. One swap per clip, ever. 0 keeps v1 behaviour
#                       (the original audio ships, labelled `exhausted`).
#   CODE_DUMP_DIR       HOST path of the directory the TTS server writes code dumps into
#                       (the host side of $VOXTRAL_CODE_DUMP_DIR in the CrispASR compose,
#                       container /cache/cloning/dumps). Empty by default. Setting it
#                       turns on the code gate: a draw whose own semantic codes say it
#                       derailed is discarded before it costs an STT call, which is the
#                       half of the defects a duration check cannot see. Harmless against
#                       a stock server, which writes no dump (that reads as no opinion,
#                       never as a rejection). Same name and meaning as in
#                       05_generate_audio/generate_v2.sh.
#   INPUT     space-separated list of NeMo manifests, processed one after the other
#             (each to convergence). Defaults to the main corpus manifest followed by
#             the PARROT one, which is a separate file because PARROT ships under its
#             own licence (CC BY-NC-SA 4.0, eval-only) as its own HF subset. The first
#             manifest writes to ${OUTPUT}, each later one to ${OUTPUT}_<parent dir>
#             (e.g. improved_parrot), since both are named full.jsonl and the output
#             file is named after the manifest stem. Set INPUT to a single path to work
#             on just one dataset.
#   MODE      alternate|stt|tts (default alternate). `alternate` is the normal flow
#             above. `stt` runs ONLY the scoring passes and never touches the TTS
#             server, which is what you want when the TTS side is down or being
#             reconfigured: clips found bad are flagged pending_tts and left for a
#             later alternate run, and the loop stops once scoring has run dry
#             (pass exit 11) instead of spinning on work only a tts pass can clear.
#             `tts` is the mirror image. Example, an STT-only night over the two
#             finished subsets:
#                 sudo -E MODE=stt CATEGORY="dictionary drugs" ./driver.sh
#   CATEGORY  space-separated list of categories to restrict work to (e.g.
#             CATEGORY=parhaf, or CATEGORY="parhaf parrot"). Rows in other
#             categories are passed through unchanged and convergence is judged
#             over the selected categories only, so the loop stops when just
#             those subsets are done. Empty (default) = the whole dataset.
#   Optional pre- and post-steps, so one command can do the whole phase (all default OFF;
#   each is a single call to a script that already exists, run as ${TARGET_USER} at the
#   same nice level as a pass):
#   REFRESH_MANIFESTS=1  rebuild the NeMo manifests from the generated audio before any
#                       pass, which is what picks up the durations of regenerated clips
#                       (99_hf_release/01_build_nemo_manifest.py --all, then
#                       02_combine_nemo_manifests.py). Needs PIN_SPLITS_FROM.
#   PIN_SPLITS_FROM     manifest root of the PREVIOUS generation (v1's NeMO_files/),
#                       passed to both builders as --pin-splits-from. REQUIRED by
#                       REFRESH_MANIFESTS and refused when empty, not defaulted: the
#                       splitter is duration-greedy, so an unpinned rebuild reshuffles
#                       train/val/test and leaks the old test set into the new train set,
#                       which silently invalidates every v1-vs-v2 comparison. The path is
#                       machine-specific, so only you can supply it.
#   ACOUSTIC_SCAN=1     run the CPU-only acoustic loop scan first
#                       (05_generate_audio/04_qc_acoustic_loops.py), which scores every
#                       clip for self-similarity and writes qc_acoustic_sidecar.jsonl
#                       beside the audio. Free (no GPU, no server), and its sidecar is a
#                       STEMS_FROM file, so a cheap scan can aim an expensive pass.
#   ACOUSTIC_WORKERS    worker processes for that scan (default 4).
#   SYNC_RESULTS=1      after every dataset has converged, carry stage 06's QC columns
#                       (and a swapped clip's voice) back into the release manifests
#                       (99_hf_release/03_sync_hotfix_results.py --apply --stt-key
#                       $STT_MODEL). Without it the manifests still describe the audio as
#                       it was before this run.
#   STEMS_FROM space-separated list of files naming the only CLIPS to work on, each
#             either a QC sidecar jsonl (its flagged rows) or a plain list of stems,
#             one per line. Same pass-through and convergence semantics as CATEGORY,
#             one clip at a time instead of one subset at a time. This is the v2
#             plan's second judge: the fast model sweeps all 602k clips, then the
#             slower, stronger one is pointed at just the ones it flagged, e.g.
#                 sudo -E MODE=stt STT_MODEL=large-v3 STEMS_FROM=flags.txt ./driver.sh
#
# This file was written with Claude Code.
set -uo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "driver.sh runs as root so it can flip docker without per-command sudo." >&2
  echo "launch it with:  sudo -E ./driver.sh   (from 06_hotfixes/)" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"   # so the default ../99_hf_release/... relative paths resolve as they do manually
SWITCH="${SWITCH_SERVER:-${HERE}/switch_server.sh}"

# Local config: load 06_hotfixes/.env if present (KEY=value lines, e.g. CRISPASR_COMPOSE
# and the WDOC_WHISPER_* paths/token). Auto-exported (set -a) so child processes
# (switch_server.sh) inherit them too. This lets you run plain `sudo ./driver.sh`
# without -E. .env holds a token, so it MUST be gitignored (see README).
if [[ -f "${HERE}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${HERE}/.env"
  set +a
fi

# Non-root user that runs the Python passes. Defaults to whoever ran sudo, so nothing
# is hardcoded; override with TARGET_USER=<name> for a dedicated account.
TARGET_USER="${TARGET_USER:-${SUDO_USER:-}}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == root ]]; then
  echo "driver.sh: cannot tell which non-root user to run the Python as." >&2
  echo "launch via 'sudo -E ./driver.sh' (sets SUDO_USER) or export TARGET_USER=<you>." >&2
  exit 1
fi

: "${CRISPASR_COMPOSE:?set CRISPASR_COMPOSE (sudo -E to carry it, or put it in improve.env)}"
# v2 default: the local CrispASR ASR stack switch_server.sh brings up on :8002 (replicas
# behind the Caddy load balancer). v1 ran against a remote server, which now answers 502,
# so the local one is the judge and the variable only has to be set to use another.
WDOC_WHISPER_ENDPOINT="${WDOC_WHISPER_ENDPOINT:-http://127.0.0.1:8002}"
# That service sets CRISPASR_API_KEYS, so /v1/audio/transcriptions needs a bearer token
# (only /health is public). $CRISPASR_API_KEY is the name the CrispASR clients already
# use, so accept it rather than making you set the same secret twice.
WDOC_WHISPER_API_KEY="${WDOC_WHISPER_API_KEY:-${CRISPASR_API_KEY:-}}"
if [[ -z "${WDOC_WHISPER_API_KEY}" ]]; then
  echo "driver: no STT token (WDOC_WHISPER_API_KEY / CRISPASR_API_KEY). If the ASR" >&2
  echo "  service has CRISPASR_API_KEYS set, every transcription will 401." >&2
fi
# Both the model name sent to the server and the key the readings are stored under, which
# is what keeps the plan's two judges on one row: see the header note.
STT_MODEL="${STT_MODEL:-${WDOC_WHISPER_MODEL:-whisper.cpp/ggml-large-v3-turbo}}"
# Say which judge is about to score the corpus, because the defaults above are easy to
# lose: `.env` is sourced with `set -a` BEFORE them, so a stale line there wins silently.
# A v1 `.env` pins the old remote endpoint (now 502) and WDOC_WHISPER_MODEL=whisper-1,
# which would file every v2 reading under the v1 key. One line beats a 401 or a 502 per
# clip, or a key nobody notices until the sync writes nulls.
echo "driver.sh: judge ${STT_MODEL} at ${WDOC_WHISPER_ENDPOINT}" \
     "(readings stored under the key '${STT_MODEL}')"
case "${WDOC_WHISPER_ENDPOINT}" in
  *127.0.0.1*|*localhost*) ;;
  *) echo "driver: that endpoint is not local. Deliberate? The v2 judge is the CrispASR" >&2
     echo "  stack switch_server.sh starts on :8002; a WDOC_WHISPER_ENDPOINT left over in" >&2
     echo "  06_hotfixes/.env overrides the default and v1's remote host now answers 502." >&2 ;;
esac

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
# Resolve uv. It lives on ${TARGET_USER}'s PATH (~/.local/bin), not root's, and a
# NON-interactive login shell may not source it (under zsh the PATH line is in
# ~/.zshrc, which `runuser -l` does not read), so probe the home dir directly first.
# Override by setting UV_BIN in .env.
resolve_uv() {
  local c
  if [[ -n "${UV_BIN:-}" && -x "${UV_BIN}" ]]; then printf '%s\n' "${UV_BIN}"; return 0; fi
  for c in "${TARGET_HOME}/.local/bin/uv" "${TARGET_HOME}/.cargo/bin/uv" \
           /usr/local/bin/uv /usr/bin/uv; do
    [[ -x "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  # Last resort: ask the user's login shell (works if uv is on a login-sourced PATH).
  c="$(runuser -l "${TARGET_USER}" -c 'command -v uv' 2>/dev/null || true)"
  [[ -n "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  return 1
}
UV_BIN="$(resolve_uv || true)"
if [[ -z "${UV_BIN}" ]]; then
  echo "driver.sh: could not find 'uv'. As ${TARGET_USER}, run 'command -v uv' and put" >&2
  echo "  UV_BIN=<that absolute path> in 06_hotfixes/.env" >&2
  exit 1
fi
# PATH forwarded to the passes so uv (and anything it shells out to) resolves.
USER_PATH="${TARGET_HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Everything this script runs that is not a container is Python owned by ${TARGET_USER},
# so it all needs the same four things: drop root, their HOME and PATH, and nice 19. One
# helper rather than one copy per call site (the passes, the per-dataset report, the pre-
# and post-steps). `nice -n 40` because nice takes an INCREMENT and sudo may hand us a
# negative priority; the result clamps at 19. See the long note in run_pass.
as_user() {
  runuser -u "${TARGET_USER}" -- \
    env HOME="${TARGET_HOME}" PATH="${USER_PATH}" nice -n 40 "$@"
}

# Normalise one 0/1 env flag in place. There are eight of them, hence a helper rather
# than eight `case` blocks: under `set -u`, `(( VOICE_SWAP ))` on a value like `true`
# reads `true` as an unset NAME, evaluates to 0, and turns OFF the very thing the
# operator just turned on. So every flag goes through here, which accepts the usual
# spellings, treats empty as the default, and refuses anything else outright.
#   flag VOICE_SWAP 1          -> $VOICE_SWAP becomes 0 or 1
#   flag RESCORE 0 RESCORE_ON  -> reads $RESCORE, writes $RESCORE_ON, so the operator's
#                                own value stays readable in the logs and messages
flag() {
  local name="$1" default="$2"
  local -n _src="${name}" _dst="${3:-$1}"
  case "${_src:-${default}}" in
    1|true|TRUE|yes|YES|y|Y) _dst=1 ;;
    0|false|FALSE|no|NO|n|N) _dst=0 ;;
    *) echo "driver.sh: ${name} must be 0 or 1 (got '${_src}')" >&2
       exit 1 ;;
  esac
}

echo "driver.sh: uv=${UV_BIN}, running passes as ${TARGET_USER}"

# Raise the GPU power limit once at start (root-only; best-effort, never fatal).
GPU_POWER_LIMIT="${GPU_POWER_LIMIT:-350}"
nvidia-smi -pl "${GPU_POWER_LIMIT}" \
  || echo "driver: could not set GPU power limit to ${GPU_POWER_LIMIT}W (continuing)" >&2

# INPUT may list SEVERAL manifests (space separated). They are processed one after the
# other: the alternating loop runs to convergence on the first, then moves to the next.
# The default chains the main corpus (dictionary + drugs + PARHAF) and then PARROT, which
# lives in its own manifest because it ships under a different licence (CC BY-NC-SA 4.0,
# eval-only) as a separate HF subset.
INPUT="${INPUT:-../99_hf_release/data/NeMO_files/full.jsonl ../99_hf_release/data/NeMO_files/PARROT/full.jsonl}"
OUTPUT="${OUTPUT:-improved}"
# Empty (the default) = derive each manifest's audio root from its own directory, which is
# what every manifest here needs (their audio_filepath values are relative to themselves:
# ../PARHAF/x.flac from NeMO_files, ../../PARROT/x.flac from NeMO_files/PARROT). Set it to
# pin ONE root for every manifest instead.
AUDIO_ROOT="${AUDIO_ROOT:-}"
N_IMPROV="${N_IMPROV:-5}"
# Sweep depth, not a permanent verdict. The corpus converged at 0.15 / 0.20 (no clip
# left above either gate), so these are the SECOND sweep: they queue the next ~1550
# clips, about 10.5 h of audio, which is roughly one night of alternating passes. Raise
# them back if you only want the worst clips.
# One flag per gate, and only when its env var is set, so an unset gate keeps whatever
# 01_recursive_improvement.py ships (see the header note on why no gate value is
# defaulted in this file). A loop over name:flag pairs rather than four if blocks, so a
# fifth gate is one line and cannot be half-wired.
GATE_ARGS=()
for _gate in "CER_THRESHOLD:--cer-threshold" \
             "TAIL_CER_THRESHOLD:--tail-cer-threshold" \
             "WINDOW_CER_THRESHOLD:--window-cer-threshold" \
             "BAD_RUN_THRESHOLD:--bad-run-threshold"; do
  _gate_var="${_gate%%:*}"
  _gate_val="${!_gate_var:-}"
  if [[ -n "${_gate_val}" ]]; then
    GATE_ARGS+=("${_gate#*:}" "${_gate_val}")
  fi
done
if (( ${#GATE_ARGS[@]} )); then
  echo "driver: gate overrides: ${GATE_ARGS[*]}"
fi
# How much better a draw must be to REPLACE the original when it does not clear the
# gates. It matters at a low gate: a clip flagged at 0.09 whose best draw comes in at
# 0.088 has not been fixed, it has been fitted to the judge, and the CER of a clean clip
# moves by that much on Whisper's own spelling choices. 0 (the script's default) accepts
# any gain at all, which is the right call only when the gate is high enough that
# everything flagged is genuinely broken.
#
# The v2 generation guards, applied to every draw a redo makes (the pass's step 2a).
# AUTO_TOKEN_BUDGET=1 (the default) sends each draw a per-clip frame budget and rejects
# one that comes back past it or below its floor; set it to 0 to go back to the server's
# global 327.68s cap, which only catches the worst runaways and only after paying ~6
# minutes of GPU for them. CODE_DUMP_DIR is the HOST path of the directory the TTS
# server writes code dumps into, i.e. the host side of $VOXTRAL_CODE_DUMP_DIR in the
# CrispASR compose (container /cache/cloning/dumps): nothing can derive it, which is why
# it has to be given. Setting it turns on the code gate (a draw whose own semantic codes
# say it derailed is discarded before it costs an STT call) and appends one row per draw
# to the code_qc_sidecar.jsonl beside the audio. Empty by default, and harmless against
# a stock server, which writes no dump: that reads as no opinion, never as a rejection.
# Same variable name and meaning as 05_generate_audio/generate_v2.sh, on purpose.
flag AUTO_TOKEN_BUDGET 1
flag VOICE_SWAP 1
CODE_DUMP_DIR="${CODE_DUMP_DIR:-}"
GUARD_ARGS=()
if (( AUTO_TOKEN_BUDGET )); then
  GUARD_ARGS+=(--auto-token-budget)
else
  GUARD_ARGS+=(--no-auto-token-budget)
  echo "driver: per-clip frame budget OFF, draws are only bounded by the server's cap"
fi
if (( VOICE_SWAP )); then
  GUARD_ARGS+=(--voice-swap)
else
  GUARD_ARGS+=(--no-voice-swap)
  echo "driver: voice swap OFF, a clip that loses every draw keeps its original audio"
fi
if [[ -n "${CODE_DUMP_DIR}" ]]; then
  GUARD_ARGS+=(--code-dump-dir "${CODE_DUMP_DIR}")
  echo "driver: code gate ON, reading dumps from ${CODE_DUMP_DIR}"
fi

MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.02}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-327.68}"
# v2 defaults, both raised: the judge is now LOCAL (:8002, several CrispASR replicas
# behind the Caddy load balancer, see switch_server.sh CRISPASR_STT_SCALE) instead of one
# remote server, and each replica serialises its own requests, so the only way to keep
# them all busy is to have more requests than replicas in flight. 20 against 6 replicas
# leaves a queue of ~3 per replica, enough that no replica idles between clips without
# making any single clip wait minutes.
STT_PARALLEL="${STT_PARALLEL:-20}"
# Measured, not guessed: stage 05 timed the same server on the same clips on 2026-10-14
# and got 3,860 clips/h at 16 in flight against 4,610 to 4,960 at 32. On a 12,000-clip
# queue at 5 draws each (60,000 synthesis calls) that is ~13 h at 32 against roughly 10
# days at v1's 2, which only existed because that server was shared with interactive
# work. Do NOT raise this above 32: it is the largest captured CUDA graph, past which
# vLLM falls back to the eager path.
TTS_PARALLEL="${TTS_PARALLEL:-32}"
# FALLBACK voice only: a v2 manifest row carries its own `voice` (one of the ten),
# and the clip is regenerated with that one, because the voice column is re-derived
# from the group key and cannot follow a regeneration. This value is used for rows
# without the field, i.e. v1 audio, which fr_female read entirely.
TTS_VOICE="${TTS_VOICE:-fr_female}"
# Classifier-free guidance per draw: draw N gets START + STEP * N (N from 0), clamped
# at FLOOR, so the defaults walk 2.0, 1.8, 1.6, 1.4, 1.2 across the five draws of a
# redo. The point is spread: a clip the TTS derailed on at its tuned setting is unlikely
# to be rescued by another draw at the same setting, and the winner is picked on CER
# anyway, so a draw at the wrong guidance simply loses.
#
# The ladder DESCENDS because the window is measured and 2.0 sits at its top: 16 French
# voices x 10 values from 0.1 to 5.0 on one dense medical sentence, listened to on
# 2026-10-13 (CrispASR voxtral/cloning/cfg_sweep.sh). Usable band 1.2-2.0 for the
# shipped voices, 1.5-2.0 for the hybrids. The generation pass runs at 2.0, so every
# retry worth making is below it: from 3.0 up, distortion grows with the value and
# takes increasingly fail to terminate at all, while 1.2-2.0 is all usable. A clip that
# derailed at 2.0 is over-guided, not under-guided.
#
# FLOOR is a hard clamp and not decoration: N_IMPROV is settable, and ten draws at
# -0.2 would reach 0.2. Below ~1.2 quality falls off far faster than the same distance
# above 2.0 (at 0.5 and under, every voice rambled to ~2.7x the sentence's natural
# length instead of stopping), and 1.0 is CFG off, which makes the flow-matching
# decoder stop following the text and invent phonetically plausible words: exactly the
# failure this stage exists to repair. The server's entrypoint refuses 1.0 at startup
# for the same reason. Needs the forked vllm-omni to be useful (upstream drops a flat
# cfg_alpha silently); the pass sends it nested in extra_params, see tts_synthesize.
CFG_ALPHA_START="${CFG_ALPHA_START:-2.0}"
CFG_ALPHA_STEP="${CFG_ALPHA_STEP:--0.2}"
CFG_ALPHA_FLOOR="${CFG_ALPHA_FLOOR:-1.2}"
# The other half of what makes one draw differ from the next: draw N uses seed START + N.
# A clip's attempt count restarts when it is requeued, so a retry over already-drawn clips
# (RETRY_EXHAUSTED=1) with this left at its default asks the server for the SAME five
# draws it already rejected. Move it past the range the previous run used (43..47 for a
# five-draw redo) or the retry only spends GPU time reproducing known losers.
START_SEED="${START_SEED:-43}"

# Which passes to run. `alternate` is the stt <-> tts flow; `stt` / `tts` run that one
# pass in a loop and never bring the other server up, so a half-configured GPU (or a
# TTS backend mid-rebuild) does not block the half that works.
MODE="${MODE:-alternate}"
case "${MODE}" in
  alternate|stt|tts) ;;
  *) echo "driver.sh: MODE must be alternate, stt or tts (got '${MODE}')" >&2; exit 1 ;;
esac
if [[ "${MODE}" != alternate ]]; then
  echo "driver: ${MODE}-only mode, the ${MODE} server is the only one this run touches"
fi

# RESCORE=1 re-derives every stored score from the transcript already on the row, so a
# clip whose CER was computed under older normalization rules is judged by the current
# ones. It is a one-time migration and is applied to the FIRST stt pass of each dataset
# only, for two reasons: the second pass would recompute scores that are already current,
# and a pass that re-examines every row always reports work done, which would keep a
# single-mode (MODE=stt) loop from ever reaching its "ran dry" stop (exit 11).
flag RESCORE 0 RESCORE_ON
if (( RESCORE_ON )); then
  echo "driver: RESCORE=1, the first stt pass of each dataset re-derives every stored score"
  if [[ "${MODE}" == tts ]]; then
    echo "driver: ...but MODE=tts runs no stt pass, so RESCORE will not apply" >&2
  fi
fi

# RESCORE_ONLY=1 does that migration and NOTHING else: one offline pass per dataset, then
# the next dataset. It cannot loop (every pass would re-examine the same rows and always
# report work done), and it makes no server call at all, so the docker switch is skipped
# and the pass can run while the TTS server is up.
flag RESCORE_ONLY 0 RESCORE_ONLY_ON
NO_SWITCH="${RESCORE_ONLY_ON}"
if (( RESCORE_ONLY_ON )); then
  if [[ "${MODE}" == tts ]]; then
    echo "driver.sh: RESCORE_ONLY=1 scores stored transcripts, MODE=tts scores nothing." >&2
    exit 1
  fi
  RESCORE_ON=1
  echo "driver: RESCORE_ONLY=1, one offline scoring pass per dataset then done"
  echo "driver: no STT/TTS call is made, the GPU and the running container are left alone"
fi

# RETRY_EXHAUSTED=1 reopens the clips a previous run gave up on (`exhausted`: flagged
# bad, no draw good enough). They are the only resolved rows worth revisiting, since they
# still ship their ORIGINAL, known-weak audio, and nothing else would ever look at them
# again. Like RESCORE it rides on the FIRST stt pass of each dataset only, and for a
# sharper reason than saving work: the retry cycle ends in a fresh `exhausted` marker for
# whatever still fails, so a flag left on for every pass would reopen its own output and
# the alternating loop would never converge.
#
# Pairs naturally with RESCORE_ONLY=1: the offline pass re-judges those clips under the
# current rules, drops the marker on the ones that now read clean, and puts the rest back
# in the tts queue for a plain `./driver.sh` run to redraw.
flag RETRY_EXHAUSTED 0 RETRY_EXHAUSTED_ON
if (( RETRY_EXHAUSTED_ON )); then
  echo "driver: RETRY_EXHAUSTED=1, the first stt pass of each dataset reopens the clips that gave up"
  if [[ "${MODE}" == tts ]]; then
    echo "driver: ...but MODE=tts runs no stt pass, so RETRY_EXHAUSTED will not apply" >&2
  fi
fi

# Optional category restriction (space-separated, e.g. CATEGORY="parhaf parrot").
# Expanded once into repeated --category flags so both passes share the same filter;
# empty (default) processes the whole dataset. Unquoted expansion below is intentional
# word-splitting to turn the list into separate flags.
CATEGORY="${CATEGORY:-}"
CATEGORY_ARGS=()
for _cat in ${CATEGORY}; do
  CATEGORY_ARGS+=(--category "${_cat}")
done
if (( ${#CATEGORY_ARGS[@]} )); then
  echo "driver: restricting work to categories: ${CATEGORY}"
fi

# Optional CLIP restriction (space-separated list of files, each either a QC sidecar
# jsonl or a plain list of stems). Same mechanism as CATEGORY one clip at a time, and
# the reason the v2 plan has a second judge at all: the full sweep scores 602k clips with
# the fast model, and this is how the slower, stronger one is pointed at just the ~12k it
# flagged. Each file must exist (the pass refuses a missing one rather than checking
# nothing, which would read as a clean corpus). Unquoted expansion is intentional
# word-splitting.
STEMS_FROM="${STEMS_FROM:-}"
STEMS_ARGS=()
for _f in ${STEMS_FROM}; do
  if [[ ! -f "${_f}" ]]; then
    echo "driver.sh: STEMS_FROM file not found: ${_f}" >&2
    exit 1
  fi
  STEMS_ARGS+=(--stems-from "${_f}")
done
if (( ${#STEMS_ARGS[@]} )); then
  echo "driver: restricting work to the clips listed in: ${STEMS_FROM}"
fi

# Optional pre-steps, in the only order that makes sense: refresh the manifests from the
# audio on disk FIRST (so the durations the passes read describe the clips as they are
# now, including anything regenerated since), then scan that audio for acoustic loops
# (its sidecar is a STEMS_FROM file, so it can aim the expensive passes that follow).
# Both call a script that already exists, as ${TARGET_USER} and at nice 19 like a pass.
flag REFRESH_MANIFESTS 0
flag ACOUSTIC_SCAN 0
flag SYNC_RESULTS 0
PIN_SPLITS_FROM="${PIN_SPLITS_FROM:-}"
ACOUSTIC_WORKERS="${ACOUSTIC_WORKERS:-4}"
# The repo root, so the pre- and post-steps can reach the other stages. HERE is
# <root>/06_hotfixes, and the two are kept apart because every path a pass gets is
# relative to HERE.
ROOT="$(cd "${HERE}/.." && pwd)"

if (( REFRESH_MANIFESTS )); then
  if [[ -z "${PIN_SPLITS_FROM}" ]]; then
    echo "driver.sh: REFRESH_MANIFESTS=1 needs PIN_SPLITS_FROM=<the previous generation's" >&2
    echo "  manifest root, i.e. v1's NeMO_files/>. Not defaulted on purpose: the splitter" >&2
    echo "  is duration-greedy, so an unpinned rebuild reshuffles train/val/test and leaks" >&2
    echo "  the old test set into the new train set, which invalidates every v1-vs-v2" >&2
    echo "  comparison without any error. See docs/V2_REGENERATION_RUNBOOK.md." >&2
    exit 1
  fi
  if [[ ! -d "${PIN_SPLITS_FROM}" ]]; then
    echo "driver.sh: PIN_SPLITS_FROM is not a directory: ${PIN_SPLITS_FROM}" >&2
    exit 1
  fi
  echo "=== $(date '+%F %T') pre-step: rebuilding the NeMo manifests," \
       "splits pinned to ${PIN_SPLITS_FROM} ==="
  # Fatal if either fails, unlike the scan below: a half-written manifest set is what the
  # passes then read as the truth about the corpus.
  as_user "${UV_BIN}" run "${ROOT}/99_hf_release/01_build_nemo_manifest.py" \
    --all --pin-splits-from "${PIN_SPLITS_FROM}" \
    || { echo "!!! the per-source manifest rebuild failed, stopping" >&2; exit 1; }
  as_user "${UV_BIN}" run "${ROOT}/99_hf_release/02_combine_nemo_manifests.py" \
    --pin-splits-from "${PIN_SPLITS_FROM}" \
    || { echo "!!! the combined manifest rebuild failed, stopping" >&2; exit 1; }
fi

if (( ACOUSTIC_SCAN )); then
  echo "=== $(date '+%F %T') pre-step: acoustic loop scan," \
       "${ACOUSTIC_WORKERS} CPU worker(s), no GPU and no server ==="
  # Never fatal: the sidecar is advisory (it says which clips repeat themselves, which
  # the passes do not need to run), and this scan is minutes to hours of CPU that must
  # not be able to stop a GPU run that would otherwise work.
  as_user "${UV_BIN}" run "${ROOT}/05_generate_audio/04_qc_acoustic_loops.py" \
    --workers "${ACOUSTIC_WORKERS}" \
    || echo "driver: the acoustic scan failed; continuing without its sidecar" >&2
fi

# Expand the INPUT list into one (manifest, output dir, audio root) triple per dataset.
# The output FILE is named after the manifest stem (<stem>.stt.jsonl), so two manifests
# both called full.jsonl would fight over one resume file: the first dataset keeps
# ${OUTPUT} as-is (so an in-flight run is untouched) and every later one gets
# ${OUTPUT}_<its parent directory, lowercased>, e.g. improved_parrot. Unquoted expansion
# below is intentional word-splitting.
MANIFESTS=(); OUT_DIRS=(); AUDIO_ROOTS=()
for _m in ${INPUT}; do
  if [[ ! -f "${_m}" ]]; then
    echo "driver.sh: manifest not found: ${_m} (run from 06_hotfixes/, or fix INPUT)" >&2
    exit 1
  fi
  MANIFESTS+=("${_m}")
  if (( ${#MANIFESTS[@]} == 1 )); then
    OUT_DIRS+=("${OUTPUT}")
  else
    _label="$(basename "$(dirname "${_m}")" | tr 'A-Z' 'a-z')"
    OUT_DIRS+=("${OUTPUT}_${_label}")
  fi
  AUDIO_ROOTS+=("${AUDIO_ROOT:-$(dirname "${_m}")}")
done

# Guard: two datasets writing the same <output>/<stem>.stt.jsonl would corrupt each
# other's resume state, so refuse up front instead of discovering it mid-run.
for _i in "${!MANIFESTS[@]}"; do
  for _j in "${!MANIFESTS[@]}"; do
    (( _j <= _i )) && continue
    _si="$(basename "${MANIFESTS[$_i]}" .jsonl)"; _sj="$(basename "${MANIFESTS[$_j]}" .jsonl)"
    if [[ "${OUT_DIRS[$_i]}/${_si}" == "${OUT_DIRS[$_j]}/${_sj}" ]]; then
      echo "driver.sh: ${MANIFESTS[$_i]} and ${MANIFESTS[$_j]} both write" >&2
      echo "  ${OUT_DIRS[$_i]}/${_si}.stt.jsonl -- give one of them a different OUTPUT." >&2
      exit 1
    fi
  done
done

echo "driver: ${#MANIFESTS[@]} dataset(s) queued:"
for _i in "${!MANIFESTS[@]}"; do
  echo "  $((_i + 1)). ${MANIFESTS[$_i]} -> ${OUT_DIRS[$_i]}/ (audio root ${AUDIO_ROOTS[$_i]})"
done

run_pass() {  # $1 = stt|tts ; runs on the CUR_* dataset; returns the pass's exit code
  local mode="$1"
  if (( NO_SWITCH )); then
    # This pass talks to no server, so flipping the GPU would only cost a container restart.
    echo "=== $(date '+%F %T') ${mode} pass needs no server, leaving docker as it is ==="
  else
    echo "=== $(date '+%F %T') switching to ${mode} server (root) ==="
    if ! "${SWITCH}" "${mode}" "${CRISPASR_COMPOSE}"; then
      echo "!!! switch to ${mode} failed" >&2
      return 70
    fi
  fi
  echo "=== $(date '+%F %T') running --mode ${mode} pass as ${TARGET_USER} ==="
  # --rescore only ever rides on an stt pass (a tts pass scores nothing), and only on the
  # first one of this dataset: consumed here so a retry loop does not repeat it.
  local extra=()
  if [[ "${mode}" == stt ]] && (( RESCORE_PENDING )); then
    if (( RESCORE_ONLY_ON )); then
      extra+=(--rescore-only)
      echo "driver: offline pass, stored transcripts rescored, nothing transcribed (--rescore-only)"
    else
      extra+=(--rescore)
      echo "driver: re-deriving every stored score under the current rules (--rescore)"
    fi
    RESCORE_PENDING=0
  fi
  # Same shape, same reason, plus one of its own: reopening the exhausted clips on every
  # pass would reopen the markers this very retry writes (see the RETRY_EXHAUSTED block).
  if [[ "${mode}" == stt ]] && (( RETRY_EXHAUSTED_PENDING )); then
    extra+=(--retry-exhausted)
    echo "driver: reopening the clips a previous run gave up on (--retry-exhausted)"
    RETRY_EXHAUSTED_PENDING=0
  fi
  # Drop to the normal user for the Python (files stay user-owned). runuser without -l
  # keeps CWD (=HERE), so the relative --input/--audio-root resolve as they do manually;
  # we forward the STT config explicitly and call uv by absolute path.
  # `nice -n 40` puts the pass, and everything it spawns, at 19: the lowest priority.
  # This is an unattended job that runs for days and must never be the reason an
  # interactive session stutters, and it costs nothing, a pass waits on the GPU server
  # far more than on the CPU. 40 rather than 19 because nice takes an INCREMENT, not a
  # level: sudo can hand the driver a negative priority (-5 observed here), where +19
  # would land at 14. The result is clamped to 19, so an oversized step pins the floor
  # wherever it starts from. Placed after `env` so it resolves on USER_PATH.
  as_user env WDOC_WHISPER_ENDPOINT="${WDOC_WHISPER_ENDPOINT}" \
        WDOC_WHISPER_API_KEY="${WDOC_WHISPER_API_KEY:-}" \
        WDOC_WHISPER_MODEL="${STT_MODEL}" \
        "${UV_BIN}" run "${HERE}/01_recursive_improvement.py" \
        --mode "${mode}" \
        --input "${CUR_INPUT}" --output "${CUR_OUTPUT}" --audio-root "${CUR_AUDIO_ROOT}" \
        --stt-endpoint "${WDOC_WHISPER_ENDPOINT%/}/v1/audio/transcriptions" \
        --stt-api-token "${WDOC_WHISPER_API_KEY:-}" --stt-model "${STT_MODEL}" \
        --tts-voice "${TTS_VOICE}" \
        --cfg-alpha-start "${CFG_ALPHA_START}" --cfg-alpha-step="${CFG_ALPHA_STEP}" \
        --cfg-alpha-floor "${CFG_ALPHA_FLOOR}" \
        --start-seed "${START_SEED}" \
        --n-improv "${N_IMPROV}" \
        ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
        --min-improvement "${MIN_IMPROVEMENT}" \
        ${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"} \
        --max-audio-seconds "${MAX_AUDIO_SECONDS}" \
        --stt-parallel "${STT_PARALLEL}" --tts-parallel "${TTS_PARALLEL}" \
        ${CATEGORY_ARGS[@]+"${CATEGORY_ARGS[@]}"} \
      ${STEMS_ARGS[@]+"${STEMS_ARGS[@]}"} \
        ${extra[@]+"${extra[@]}"} \
        --shuffle
}

# One dataset at a time, each run to convergence before the next one starts, so the GPU
# only ever hosts the server the current pass needs.
for i in "${!MANIFESTS[@]}"; do
  CUR_INPUT="${MANIFESTS[$i]}"
  CUR_OUTPUT="${OUT_DIRS[$i]}"
  CUR_AUDIO_ROOT="${AUDIO_ROOTS[$i]}"
  # Per dataset: each manifest carries its own stored scores, so each one needs the
  # rescore pass once.
  RESCORE_PENDING="${RESCORE_ON}"
  # Likewise: each manifest holds its own exhausted clips, so each gets one reopening.
  RETRY_EXHAUSTED_PENDING="${RETRY_EXHAUSTED_ON}"
  # Rounds in a row whose tts pass found nothing to draft. Bounded below, see the loop.
  DRY_TTS_ROUNDS=0
  echo "=== $(date '+%F %T') dataset $((i + 1))/${#MANIFESTS[@]}: ${CUR_INPUT} -> ${CUR_OUTPUT}/ ==="
  # RESCORE_ONLY is a migration, not a loop: one pass, then the next dataset. Looping would
  # never end, since re-examining the same stored transcripts always counts as work done.
  if (( RESCORE_ONLY_ON )); then
    run_pass stt
    rc=$?
    if (( rc != 0 && rc != 10 && rc != 11 )); then
      echo "!!! rescore pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    echo "=== $(date '+%F %T') rescored ${CUR_INPUT}; nothing was transcribed. ==="
    continue
  fi
  while :; do
    # Single-mode runs loop on that one pass; the alternating run leads with stt.
    if [[ "${MODE}" == tts ]]; then FIRST=tts; else FIRST=stt; fi
    run_pass "${FIRST}"
    rc=$?
    if (( rc == 10 )); then
      echo "=== $(date '+%F %T') converged: no improvement work left on ${CUR_INPUT}. ==="
      break
    fi
    # 11 = this mode has nothing left to do, but the OTHER one still has work.
    # In a single-mode run that is the stop condition. In an alternating run it is
    # NOT: it is exactly the state a run left behind when it stopped (or was killed)
    # after an stt pass, so every flagged clip is waiting on a tts pass and there is
    # nothing for stt to do until that pass has run. Breaking here would skip the
    # dataset entirely, so we carry on to the tts pass instead.
    FIRST_DRY=0
    if (( rc == 11 )); then
      if [[ "${MODE}" != alternate ]]; then
        echo "=== $(date '+%F %T') ${FIRST} pass ran dry on ${CUR_INPUT}; the rest needs the other pass. ==="
        break
      fi
      FIRST_DRY=1
      echo "=== $(date '+%F %T') stt pass had nothing to do on ${CUR_INPUT}; its clips are waiting on the tts pass. ==="
    elif (( rc != 0 )); then
      echo "!!! ${FIRST} pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    [[ "${MODE}" == alternate ]] || continue
    run_pass tts
    rc=$?
    # 11 from the tts pass is "no clip is waiting for candidates", not a failure:
    # the stt pass owns convergence (exit 10), so let the loop go back to it.
    if (( rc != 0 && rc != 11 )); then
      echo "!!! tts pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    if (( rc == 11 )); then DRY_TTS_ROUNDS=$((DRY_TTS_ROUNDS + 1)); else DRY_TTS_ROUNDS=0; fi
    # Neither pass could do anything, yet stt did not report convergence: whatever is
    # left is stuck (e.g. flagged clips whose audio has since gone missing). Stop
    # rather than swap the two servers forever over work nobody can clear.
    if (( FIRST_DRY && rc == 11 )); then
      echo "=== $(date '+%F %T') both passes ran dry on ${CUR_INPUT} without converging;" \
           "stopping, the leftovers need a look (see the pending counts above). ==="
      break
    fi
    # Same dead end, one step less obvious: a healthy round either converges (stt exit
    # 10) or leaves clips for the tts pass to draft, so an stt pass that keeps reporting
    # work done while the tts pass keeps finding nothing to draft is re-running rows it
    # can never resolve (clips whose audio has gone missing, say). Bound the alternation
    # instead of swapping the two servers over them forever.
    if (( DRY_TTS_ROUNDS >= 2 )); then
      echo "=== $(date '+%F %T') the tts pass found nothing to draft in ${DRY_TTS_ROUNDS} rounds" \
           "while ${CUR_INPUT} still did not converge: the rest is stuck (missing audio?)," \
           "stopping this dataset. ==="
      break
    fi
  done
  # Where this dataset stands now, from the per-round ledger every pass appends to
  # (improved/rounds.jsonl). Read-only and instant, and it is what puts the before /
  # after of the whole run in the driver's own log instead of only in the live chatter
  # of each pass. Same table as `./01_recursive_improvement.py --report` from another
  # tab. Never fatal: a missing or unreadable ledger must not fail a finished dataset.
  as_user "${UV_BIN}" run "${HERE}/01_recursive_improvement.py" --report "${CUR_OUTPUT}" || true
done

# Post-step: carry what stage 06 learned and changed back into the release manifests,
# which is what makes the run visible to anything downstream of this stage (the QC
# columns, and the voice of a clip whose reader was swapped). Last, not first: it reads
# the <output>/*.stt.jsonl every pass above has been writing. --stt-key is ${STT_MODEL}
# because that is the key the readings were stored under, and the default is v1's
# ("whisper-1"), which would match nothing and write nulls.
if (( SYNC_RESULTS )); then
  echo "=== $(date '+%F %T') post-step: syncing stage 06 results into the manifests ==="
  as_user "${UV_BIN}" run "${ROOT}/99_hf_release/03_sync_hotfix_results.py" \
    --apply --stt-key "${STT_MODEL}" \
    || echo "driver: the manifest sync failed; the audio is fine, re-run" \
            "99_hf_release/03_sync_hotfix_results.py --apply by hand" >&2
fi

echo "=== $(date '+%F %T') every dataset converged. Done. ==="
