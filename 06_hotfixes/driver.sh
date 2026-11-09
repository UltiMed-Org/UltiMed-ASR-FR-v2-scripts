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
#                          against now answers 502. A non-local value is REFUSED at
#                          startup (see ALLOW_REMOTE_JUDGE), because `sudo -E` imports
#                          whatever your interactive shell exports for its own wdoc use
#                          and that silently replaces the judge for all 602k clips.
#   ALLOW_REMOTE_JUDGE     0 (default). Set to 1 to accept a non-local
#                          WDOC_WHISPER_ENDPOINT on purpose, e.g. to score against a
#                          second machine. The refusal it lifts is a startup check only.
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
#   LENGTH_TOL          among one clip's candidate draws, how much longer than the low
#                       median of the passing draws a draw may be before a shorter one is
#                       preferred ahead of CER (a repeat only ever ADDS time, and whisper
#                       cannot hear one). Negative disables it. Not a gate: it never
#                       rejects a draw, only orders acceptable ones
#   NUMBER_THRESHOLD    gate on how many label numbers the judge heard as ANOTHER number
#                       ("75 ans" read "65 ans", utils/number_swaps.py). OFF unless set
#                       (the script's default is 0): set 1 for a round that redraws the
#                       wrong-number clips, so a draw that repeats the error is not
#                       promoted. One judge alone mishears a number on ~1.5% of clean
#                       clips, cheap inside such a round, too many for a whole-corpus one
#   None of those six carries a default HERE: unset means "whatever the scripts ship"
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
#                       --force. The redraws no longer need START_SEED moved: a reopened
#                       clip walks the seed ladder on from where it stopped.
#   RETRY_IMPROVED=1    give the clips a previous run closed as `improved` WHILE THEY WERE
#                       STILL over a gate another chance (promoted on --min-improvement
#                       alone: better audio, not good enough). Runs built before that
#                       requeue existed left ~1.4k of them resolved, and nothing else
#                       would ever look at them again. One-shot like the two above, and
#                       harmless on a clip that now reads clean: it keeps its marker and
#                       stays improved.
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
#   Pre- and post-steps, so `sudo -E ./driver.sh` alone is the whole phase. All three
#   default ON; set one to 0 to skip it. Each is a single call to a script that already
#   exists, run as ${TARGET_USER} at the same nice level as a pass:
#   REFRESH_MANIFESTS=1  (default) rebuild the NeMo manifests from the generated audio
#                       before any pass, which is what picks up the durations of
#                       regenerated clips (99_hf_release/01_build_nemo_manifest.py --all,
#                       then 02_combine_nemo_manifests.py).
#   PIN_SPLITS_FROM     manifest root of the PREVIOUS generation (v1's NeMO_files/),
#                       passed to both builders as --pin-splits-from. Derived when unset,
#                       as the `NeMO_files` sibling of whatever 99_hf_release/data points
#                       at, so no machine-specific path is written down here. There is no
#                       unpinned fallback: the splitter is duration-greedy, so an
#                       unpinned rebuild reshuffles train/val/test and leaks the old test
#                       set into the new train set, which silently invalidates every
#                       v1-vs-v2 comparison. When no pin can be derived, an explicit
#                       REFRESH_MANIFESTS=1 stops the run and a defaulted one skips the
#                       refresh.
#   WAIT_FOR_GENERATOR  wait for stage 05 to finish before doing ANYTHING, by polling
#                       for a process whose command line matches this pattern (pgrep
#                       -f). Default "generate_v2\.sh|01_generate_audio\.py", i.e. the
#                       v2 generation run; "" starts immediately. This is what lets you
#                       launch the whole phase at the tail end of a generation run and
#                       walk away: the first stt switch STOPS the TTS container, so
#                       starting early takes the server out from under stage 05, and the
#                       manifest refresh would describe a half-written corpus.
#   WAIT_POLL           seconds between those polls (default 60).
#   WAIT_SETTLE         seconds the generator must stay gone before the phase starts
#                       (default 120), so a stage boundary, a respawn or a resumed run
#                       does not read as the end of generation.
#   WAIT_TIMEOUT        give up and exit 1 after this many seconds of waiting (default
#                       0, i.e. wait as long as it takes).
#   WAIT_LOG_EVERY      seconds between "still waiting" lines (default 600).
#   KILL_QC             stop the live CPU QC before the pre-steps, matching this pattern
#                       (pgrep -f). Default "03_qc_cpu_stt\.py|04_qc_acoustic_loops\.py",
#                       "" leaves it running. SIGTERM first (03_qc_cpu_stt.py traps it,
#                       finishes the clip in flight and prints its per-voice CER table),
#                       SIGKILL after KILL_QC_GRACE. Its sidecar is append-only and
#                       resumes by stem, so stopping it costs at most the clips in
#                       flight, its 25 pct sample is subsumed by the full sweep this
#                       script runs with a stronger judge, and its cores are exactly
#                       what the acoustic scan below wants.
#   KILL_QC_GRACE       seconds to wait for that SIGTERM before SIGKILL (default 60).
#   ACOUSTIC_SCAN=1     (default) run the CPU-only acoustic loop scan first
#                       (05_generate_audio/04_qc_acoustic_loops.py), which scores every
#                       clip for self-similarity and writes qc_acoustic_sidecar.jsonl
#                       beside the audio. Free (no GPU, no server), and its sidecar is a
#                       STEMS_FROM file, so a cheap scan can aim an expensive pass.
#   ACOUSTIC_WORKERS    worker processes for that scan (default: the host's cores
#                       minus 2, floor 2). One core each, BLAS threads pinned to 1.
#                       The script's own default of 4 is sized for sharing the box
#                       with the GPU generator, which the gate below has already
#                       waited out, so here the box is otherwise idle.
#   ACOUSTIC_STAGES     generator stages to scan, space-separated (e.g. "01_dictionnary
#                       04_PARROT"). Empty (default) = all of them.
#   SYNC_RESULTS=1      (default) after every dataset has converged, carry stage 06's QC
#                       columns (and a swapped clip's voice) back into the release
#                       manifests (99_hf_release/03_sync_hotfix_results.py --apply
#                       --stt-key $STT_MODEL --stt-model $STT_MODEL), then drop the rows
#                       that status rules out (99_hf_release/04_drop_bad_rows.py --apply).
#                       Without it the manifests still describe the audio as it was
#                       before this run.
#   STEMS_FROM space-separated list of files naming the only CLIPS to work on, each
#             either a QC sidecar jsonl (its flagged rows) or a plain list of stems,
#             one per line. Same pass-through and convergence semantics as CATEGORY,
#             one clip at a time instead of one subset at a time. This is the v2
#             plan's second judge: the fast model sweeps all 602k clips, then the
#             slower, stronger one is pointed at just the ones it flagged, e.g.
#                 sudo -E MODE=stt STT_MODEL=large-v3 STEMS_FROM=flags.txt ./driver.sh
#
# Resume. A phase is a day or more of work, so a stopped run picks up where it was:
# every step records itself in an append-only log when it SUCCEEDS, and a relaunch of
# the same command skips what is already in there, printing `[step N/M]` headers so the
# position is visible either way. What counts as finished is narrow on purpose: a
# dataset only when the stt pass reported convergence (never when it stopped on stuck
# leftovers, which is what a human usually fixes between two runs), the manifest
# rebuild only when BOTH builders landed, the acoustic scan only when it ran to the end
# (a partial scan resumes from its own sidecar, so re-entering it is nearly free).
#   STEPS_LOG   where that record lives (default <OUTPUT>/driver_steps.log, which the
#               repo's `**/*.log` rule already ignores). It also doubles as the phase's
#               history: `cat improved/driver_steps.log`. A log whose first line
#               describes a different run (other corpus, OUTPUT, MODE, judge or
#               STEMS_FROM queue) is moved aside rather than trusted.
#   RESET_STATE=1  move it aside anyway, i.e. re-run every step from scratch.
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
  # Precedence: a knob you set on the launch line (`sudo -E STT_MODEL=... ./driver.sh`)
  # has to BEAT the .env line, or the documented one-off override is silently a no-op.
  # bash cannot tell a launch-line assignment from any other exported variable, so the
  # exception is spelled out the other way round: the names .env BORROWS from another
  # tool stay .env-wins, because for those the inherited value is pollution rather than
  # intent (~/.scripts_oli/ubiquitous_scripts/shared_env/70_wdoc.sh exports
  # WDOC_WHISPER_ENDPOINT and WDOC_WHISPER_MODEL for wdoc's own sake; see the STT_MODEL
  # note below), as are credentials, which a stale shell export must not replace and
  # which must never be echoed. Everything else .env assigns is only a default.
  # This cost a sweep on 2026-10-25: `sudo -E MODE=stt STT_MODEL=whisper.cpp/ggml-large-v3`
  # announced `judge whisper.cpp/ggml-large-v3-turbo` because .env pinned the turbo key,
  # and the T2 second opinion would have been filed on top of T1's 602k readings.
  declare -A _env_file_override=()
  while IFS= read -r _name; do
    case "${_name}" in WDOC_*|*API_KEY*|*TOKEN*) continue ;; esac
    [[ -v ${_name} ]] && _env_file_override["${_name}"]="${!_name}"
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*$/\2/p' \
             "${HERE}/.env")
  set -a
  # shellcheck disable=SC1091
  source "${HERE}/.env"
  set +a
  for _name in "${!_env_file_override[@]}"; do
    if [[ "${!_name}" != "${_env_file_override[${_name}]}" ]]; then
      echo "driver.sh: ${_name} comes from the launch line, not from the .env line"
    fi
    export "${_name}=${_env_file_override[${_name}]}"
  done
fi

# Normalise one 0/1 env flag in place. There are nine of them, hence a helper rather
# than nine `case` blocks: under `set -u`, `(( VOICE_SWAP ))` on a value like `true`
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
# lose from TWO directions. `.env` is sourced with `set -a` BEFORE them, so a stale line
# there wins silently; and the documented launch is `sudo -E`, which carries the whole
# interactive environment in, so a WDOC_WHISPER_* exported by your shell for wdoc's own
# sake fills the `${VAR:-default}` slot and COMMENTING THE .env LINE OUT CHANGES NOTHING.
# That is what happened on 2026-10-20: ~/.scripts_oli/ubiquitous_scripts/shared_env/
# 70_wdoc.sh exports the v1 remote endpoint and WDOC_WHISPER_MODEL=whisper-1, so two
# launches with a clean .env still announced the remote judge. To pin the v2 judge, SET
# the value in .env rather than removing it (an assignment beats the inherited one).
# That holds for the WDOC_WHISPER_* names only. STT_MODEL is the driver's own knob, so
# the .env loader above hands the launch line back its value after sourcing: a one-off
# second opinion is `sudo -E STT_MODEL=whisper.cpp/ggml-large-v3 ... ./driver.sh` and the
# .env pin stays where it is. Read the judge line below to confirm which one won.
echo "driver.sh: judge ${STT_MODEL} at ${WDOC_WHISPER_ENDPOINT}" \
     "(readings stored under the key '${STT_MODEL}')"
# A refusal, not a warning. A warning scrolls past in a pane that then prints one wait
# line a minute for a day, and the cost of missing it is the whole sweep: v1's remote host
# answers 502, so every clip would fail its reading, and a judge that DOES answer would
# file 602k readings under a key the rest of the phase does not look up.
flag ALLOW_REMOTE_JUDGE 0 ALLOW_REMOTE_JUDGE_OK
case "${WDOC_WHISPER_ENDPOINT}" in
  *127.0.0.1*|*localhost*) ;;
  *)
    if (( ALLOW_REMOTE_JUDGE_OK )); then
      echo "driver: judging against the non-local ${WDOC_WHISPER_ENDPOINT}," \
           "ALLOW_REMOTE_JUDGE=1 says that is deliberate." >&2
    else
      echo "driver.sh: refusing to start: the judge endpoint is not local." >&2
      echo "  WDOC_WHISPER_ENDPOINT=${WDOC_WHISPER_ENDPOINT}" >&2
      echo "  The v2 judge is the CrispASR stack switch_server.sh starts on :8002, and" >&2
      echo "  v1's remote host now answers 502. This almost always comes from your shell" >&2
      echo "  through 'sudo -E' (70_wdoc.sh in shared_env/), NOT from .env, so commenting" >&2
      echo "  the .env line out does not help: put the real value in 06_hotfixes/.env," >&2
      echo "    WDOC_WHISPER_ENDPOINT=http://127.0.0.1:8002" >&2
      echo "  or set ALLOW_REMOTE_JUDGE=1 to judge against that host on purpose." >&2
      exit 1
    fi ;;
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
             "BAD_RUN_THRESHOLD:--bad-run-threshold" \
             "LENGTH_TOL:--length-tol" \
             "NUMBER_THRESHOLD:--number-threshold"; do
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
# The other half of what makes one draw differ from the next: draw N of a clip's FIRST
# round uses seed START + N, and every later round for that clip walks on from where the
# previous one stopped (the marker carries the count, see `seeds_burned` in the pass), so
# round 2 of a five-draw redo draws 48..52. That is not a nicety: the TTS is deterministic
# per voice, seed and cfg_alpha, so a clip that redrew the same seeds would reproduce the
# audio it is already carrying, which the pass treats as a dead seed and ABORTS the run
# on (exit 2). It is also why a retry over already-drawn clips (RETRY_EXHAUSTED=1,
# RETRY_IMPROVED=1) no longer needs this moved by hand.
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

# RETRY_IMPROVED=1 reopens the clips a previous run closed over a gate. A draw is promoted
# either because it clears every gate or because it merely beats the original by
# --min-improvement, and the second kind leaves better-but-still-bad audio on disk; the
# pass now sends those back to the tts queue itself (`pending_tts`), so this flag is the
# MIGRATION for the rows written before it did. Same one-shot shape and the same reason as
# RETRY_EXHAUSTED: the cycle ends in a fresh marker, so a flag left on every pass would
# reopen its own output and the loop would never converge.
flag RETRY_IMPROVED 0 RETRY_IMPROVED_ON
if (( RETRY_IMPROVED_ON )); then
  echo "driver: RETRY_IMPROVED=1, the first stt pass of each dataset reopens the wins that still read bad"
  if [[ "${MODE}" == tts ]]; then
    echo "driver: ...but MODE=tts runs no stt pass, so RETRY_IMPROVED will not apply" >&2
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

# The pre- and post-steps, in the only order that makes sense: refresh the manifests from
# the audio on disk FIRST (so the durations the passes read describe the clips as they are
# now, including anything regenerated since), then scan that audio for acoustic loops (its
# sidecar is a STEMS_FROM file, so it can aim the expensive passes that follow), and sync
# the results back after the last dataset converges. Each is a single call to a script
# that already exists, as ${TARGET_USER} and at nice 19 like a pass.
#
# All three default ON, so `sudo -E ./driver.sh` is the whole phase. Remember whether the
# refresh was ASKED for, because that changes what an undecidable PIN_SPLITS_FROM means:
# explicit means stop, defaulted means skip the refresh and carry on.
REFRESH_EXPLICIT="${REFRESH_MANIFESTS+1}"
flag REFRESH_MANIFESTS 1
flag ACOUSTIC_SCAN 1
flag SYNC_RESULTS 1
# Worker processes for the scan. The script's own default is 4, which is sized for
# running BESIDE the GPU generator (that is what it does during stage 05, as the live
# QC). Here it cannot be: the gate above waited for generation to leave, so the box is
# otherwise idle and the scan is pure CPU with its BLAS threads pinned to 1, i.e. one
# core per worker and nothing shared. Measured on this corpus at ~15 clips/s per worker,
# so 4 workers left ~8 of 12 cores unused and turned a 30-minute sweep into 100 minutes.
# Leave 2 cores for the rest of the machine (the TTS container idling, syncthing, a
# session watching the log); set ACOUSTIC_WORKERS explicitly when something else heavy
# shares the box.
if [[ -z "${ACOUSTIC_WORKERS:-}" ]]; then
  _cores="$(nproc 2>/dev/null || echo 4)"
  ACOUSTIC_WORKERS=$(( _cores > 4 ? _cores - 2 : 2 ))
fi
# Which generator stages the acoustic scan looks at (space-separated, e.g.
# "01_dictionnary 04_PARROT"). Empty = all of them, which is the script's own default.
ACOUSTIC_STAGES="${ACOUSTIC_STAGES:-}"
ACOUSTIC_STAGE_ARGS=()
for _st in ${ACOUSTIC_STAGES}; do
  ACOUSTIC_STAGE_ARGS+=(--stage "${_st}")
done
# The repo root, so the pre- and post-steps can reach the other stages. HERE is
# <root>/06_hotfixes, and the two are kept apart because every path a pass gets is
# relative to HERE.
ROOT="$(cd "${HERE}/.." && pwd)"

# --- resume: what this phase has already finished ----------------------------------
# A phase is a day or more of work, and a restart used to mean redoing it: an 18-minute
# manifest rebuild, then a multi-hour acoustic scan over 600k clips, before the first
# pass could run again. Each step now records itself here when it SUCCEEDS, and a
# relaunch skips what is already recorded. The passes themselves were always resumable
# (each reads its own <output>/<stem>.stt.jsonl back), and so is the scan (it reads its
# sidecar), so this only has to decide which STEPS to re-enter.
#
# Append-only and plain text, so it doubles as the phase's own history, and named .log
# because the repo already ignores `**/*.log`.
STEPS_LOG="${STEPS_LOG:-${OUTPUT}/driver_steps.log}"
flag RESET_STATE 0 RESET_STATE_ON
# What the record describes. A log written for another corpus, output directory, mode,
# judge or clip queue says nothing about this run, so it is moved aside rather than
# trusted. The queue enters by CONTENT, not by path: a repair round that rewrites its
# queue file, or a new one launched with the same knobs as the last (T5 after T4, both
# on turbo), is new work, and with the queue left out it found every dataset already
# "finished" and ran nothing. Same content is still the same run, so an interrupted
# round relaunched with its own queue resumes as before. Note
# RESCORE_ONLY is deliberately NOT in here: it gets its own step keys below, so a
# migration run does not throw away the pre-step markers the next full run can reuse.
_stems_fp=""
if [[ -n "${STEMS_FROM}" ]]; then
  # shellcheck disable=SC2086  # word-split on purpose, the same list as STEMS_ARGS
  _stems_fp="$(cat -- ${STEMS_FROM} | md5sum | cut -d" " -f1)"
fi
STATE_FP="$(printf '%s|%s|%s|%s|%s' "${INPUT}" "${OUTPUT}" "${MODE}" "${STT_MODEL}" \
            "${_stems_fp}" | md5sum | cut -d" " -f1)"
# Created as ${TARGET_USER} (root appends to it afterwards) so the phase leaves no
# root-owned file in a directory the user owns.
as_user mkdir -p "${OUTPUT}"
if (( RESET_STATE_ON )) && [[ -f "${STEPS_LOG}" ]]; then
  mv -- "${STEPS_LOG}" "${STEPS_LOG}.$(date '+%Y%m%d-%H%M%S')"
  echo "driver: RESET_STATE=1, the step log was moved aside; every step runs again"
elif [[ -f "${STEPS_LOG}" ]]; then
  _prev_fp="$(grep -a " start " -- "${STEPS_LOG}" | tail -1 | awk "{print \$NF}")"
  if [[ -n "${_prev_fp}" && "${_prev_fp}" != "${STATE_FP}" ]]; then
    mv -- "${STEPS_LOG}" "${STEPS_LOG}.$(date '+%Y%m%d-%H%M%S')"
    echo "driver: the step log describes a different run (corpus, output, mode or" \
         "judge or clip queue changed), moved aside; every step runs again"
  fi
fi
as_user touch "${STEPS_LOG}"

step_log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "${STEPS_LOG}"; }
# Step keys are plain words (manifests, acoustic, dataset-2, sync): matched anchored, so
# `dataset-1` cannot be answered by `dataset-10`.
step_done() { grep -qaE "^[0-9-]+ [0-9:]+ done ${1}$" -- "${STEPS_LOG}"; }
step_when() { grep -aE "^[0-9-]+ [0-9:]+ done ${1}$" -- "${STEPS_LOG}" | tail -1 | cut -d" " -f1-2; }
step_log "start ${STATE_FP}"

# Where the PREVIOUS generation's manifests live, which the rebuild must pin its splits
# to. Derived rather than hardcoded (no machine path or username in a tracked file): the
# release stage's `data` symlink points at THIS generation's audio root, and the previous
# one's manifest root is its sibling `NeMO_files/`, e.g. data -> .../audio_datasets/v2 so
# the pin is .../audio_datasets/NeMO_files. Set PIN_SPLITS_FROM to override.
PIN_SPLITS_FROM="${PIN_SPLITS_FROM:-}"
if [[ -z "${PIN_SPLITS_FROM}" && -e "${ROOT}/99_hf_release/data" ]]; then
  _audio_root="$(readlink -f "${ROOT}/99_hf_release/data" 2>/dev/null || true)"
  if [[ -n "${_audio_root}" && -d "$(dirname "${_audio_root}")/NeMO_files" ]]; then
    PIN_SPLITS_FROM="$(dirname "${_audio_root}")/NeMO_files"
    # Refuse the degenerate case where that resolves to the manifests we are ABOUT to
    # rewrite: pinning a rebuild to its own output pins nothing.
    if [[ "${PIN_SPLITS_FROM}" == "$(readlink -f "${ROOT}/99_hf_release/data" )/NeMO_files" ]]; then
      echo "driver: the derived pin (${PIN_SPLITS_FROM}) is this generation's own" >&2
      echo "  manifest root, which pins nothing. Set PIN_SPLITS_FROM explicitly." >&2
      PIN_SPLITS_FROM=""
    else
      echo "driver: splits pinned to ${PIN_SPLITS_FROM} (derived from 99_hf_release/data;" \
           "set PIN_SPLITS_FROM to override)"
    fi
  fi
fi

if (( REFRESH_MANIFESTS )) && [[ -z "${PIN_SPLITS_FROM}" ]]; then
  # No pin, and none can be derived. An UNPINNED rebuild is never an option: the splitter
  # is duration-greedy, so it reshuffles train/val/test and leaks the old test set into
  # the new train set, which invalidates every old-vs-new comparison with no error
  # anywhere. See docs/V2_REGENERATION_RUNBOOK.md.
  echo "driver.sh: cannot pin the splits: no PIN_SPLITS_FROM, and none derivable from" >&2
  echo "  99_hf_release/data (expected <that symlink's parent>/NeMO_files)." >&2
  if [[ -n "${REFRESH_EXPLICIT}" ]]; then
    echo "  REFRESH_MANIFESTS was asked for explicitly, so stopping rather than" >&2
    echo "  rebuilding unpinned. Pass PIN_SPLITS_FROM=<previous NeMO_files dir>." >&2
    exit 1
  fi
  echo "  Skipping the manifest refresh (it is only ON by default). The passes will use" >&2
  echo "  the manifests as they are; pass PIN_SPLITS_FROM=<dir> to refresh them." >&2
  REFRESH_MANIFESTS=0
fi
if (( REFRESH_MANIFESTS )) && [[ ! -d "${PIN_SPLITS_FROM}" ]]; then
  echo "driver.sh: PIN_SPLITS_FROM is not a directory: ${PIN_SPLITS_FROM}" >&2
  exit 1
fi

# How many steps this phase has, so every header can say which one it is on. The dataset
# count comes from INPUT rather than the validated MANIFESTS array, which is built later
# (it cannot be built here: the manifest rebuild below is what CREATES those files on a
# first run).
DATASET_COUNT="$(set -- ${INPUT}; echo $#)"
STEP_TOTAL=$(( REFRESH_MANIFESTS + ACOUSTIC_SCAN + DATASET_COUNT + 2 * SYNC_RESULTS ))  # sync, then drop
STEP_NOW=0
step_header() {   # $* = what this step is
  STEP_NOW=$(( STEP_NOW + 1 ))
  echo "=== $(date '+%F %T') [step ${STEP_NOW}/${STEP_TOTAL}] $* ==="
}
step_skip() {     # $1 = step key, $2.. = what it was
  STEP_NOW=$(( STEP_NOW + 1 ))
  echo "=== $(date '+%F %T') [step ${STEP_NOW}/${STEP_TOTAL}] skipping ${*:2}:" \
       "finished $(step_when "$1") in an earlier run (RESET_STATE=1 to redo it) ==="
}
# ----------------------------------------------------------------------------------
# The gate, so that `sudo -E ./driver.sh` can be launched NOW and start by itself the
# moment stage 05 is done. Two things have to happen before anything else:
#
#   1. wait for the generator to leave. The signal is its PROCESS, deliberately: not
#      the container (the tts passes below need that container up, so its state says
#      nothing about whether generation is over) and not a lock file (a crashed run
#      would never release one, and this must not need a human at 3 in the morning).
#   2. stop the live CPU QC, which has been the early warning DURING generation and is
#      redundant the moment this script starts reading every clip with a stronger
#      model. It also wants the same cores as the acoustic scan.
#
# Order matters and is the whole point: config errors above are already reported, so
# the wait only ever happens on a run that is otherwise ready to go.
WAIT_FOR_GENERATOR="${WAIT_FOR_GENERATOR-generate_v2\.sh|01_generate_audio\.py}"
WAIT_POLL="${WAIT_POLL:-60}"
WAIT_SETTLE="${WAIT_SETTLE:-120}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-0}"
WAIT_LOG_EVERY="${WAIT_LOG_EVERY:-600}"
KILL_QC="${KILL_QC-03_qc_cpu_stt\.py|04_qc_acoustic_loops\.py}"
KILL_QC_GRACE="${KILL_QC_GRACE:-60}"

# PIDs whose full command line matches $1, one per line, empty when none. Never this
# script and never its parent: `pgrep -f` matches the WHOLE command line, so a pattern
# handed around as an argument can match the process doing the asking.
procs_matching() {
  pgrep -f "$1" 2>/dev/null | grep -vx -e "$$" -e "${PPID}" || true
}

# Block until no process matches $1, then return 0. Returns 1 only on WAIT_TIMEOUT.
wait_for_generator() {
  local pattern="$1" start=${SECONDS} absent_since="" last_log=${SECONDS} pids elapsed
  pids="$(procs_matching "${pattern}" | tr '\n' ' ')"
  if [[ -z "${pids// /}" ]]; then
    echo "driver: nothing matches '${pattern}', so stage 05 is not running: starting now."
    echo "  (if it IS running, stop this: the first stt switch stops the TTS container.)"
    return 0
  fi
  echo "=== $(date '+%F %T') waiting for stage 05 to finish before touching anything ==="
  local listed="${pids% }"
  echo "driver: generation is still alive, pid(s) ${listed}. Polling every ${WAIT_POLL}s;"
  echo "  'ps -fp ${listed// /,}' says what they are. Nothing is touched until they exit."
  while :; do
    sleep "${WAIT_POLL}"
    pids="$(procs_matching "${pattern}" | tr '\n' ' ')"
    elapsed=$(( SECONDS - start ))
    if [[ -z "${pids// /}" ]]; then
      if [[ -z "${absent_since}" ]]; then
        absent_since=${SECONDS}
        echo "driver: $(date '+%F %T') the generator is gone after ${elapsed}s;" \
             "confirming it stays gone for ${WAIT_SETTLE}s."
        # The one case this gate cannot tell apart: a generation INTERRUPTED on purpose
        # looks exactly like a finished one. Say so while there is still time to act,
        # because the next thing that happens is the TTS container being stopped.
        echo "  (if you interrupted stage 05 and mean to resume it, Ctrl-C now:" \
             "the phase stops the TTS container.)"
      fi
      if (( SECONDS - absent_since >= WAIT_SETTLE )); then
        echo "=== $(date '+%F %T') stage 05 is done (waited ${elapsed}s)," \
             "the QC phase starts now ==="
        return 0
      fi
    else
      if [[ -n "${absent_since}" ]]; then
        echo "driver: a generator process is back (pid(s) ${pids% }), still waiting." >&2
        absent_since=""
      fi
      if (( WAIT_LOG_EVERY > 0 && SECONDS - last_log >= WAIT_LOG_EVERY )); then
        last_log=${SECONDS}
        echo "driver: $(date '+%F %T') still waiting on pid(s) ${pids% }," \
             "${elapsed}s so far."
      fi
    fi
    if (( WAIT_TIMEOUT > 0 && elapsed >= WAIT_TIMEOUT )); then
      echo "driver.sh: WAIT_TIMEOUT=${WAIT_TIMEOUT}s elapsed and a generator process is" >&2
      echo "  still alive (pid(s) ${pids% }). Refusing to stop its TTS server from under" >&2
      echo "  it. Raise WAIT_TIMEOUT, or set WAIT_FOR_GENERATOR= to start anyway." >&2
      return 1
    fi
  done
}

# SIGTERM everything matching $1, SIGKILL whatever is left after KILL_QC_GRACE.
kill_qc() {
  local pattern="$1" pids waited=0
  pids="$(procs_matching "${pattern}" | tr '\n' ' ')"
  if [[ -z "${pids// /}" ]]; then
    echo "driver: no live QC process matches '${pattern}', nothing to stop."
    return 0
  fi
  echo "=== $(date '+%F %T') stopping the live QC, pid(s) ${pids% } ==="
  # pkill rather than kill on the PIDs just listed, because `uv run` and the python it
  # exec'd are two processes and a new one can appear between the two calls. Safe to
  # re-match here and nowhere later: this runs BEFORE the acoustic pre-step starts, so
  # the pattern cannot match a scan of ours.
  pkill -TERM -f "${pattern}" 2>/dev/null || true
  while (( waited < KILL_QC_GRACE )); do
    sleep 2
    waited=$(( waited + 2 ))
    pids="$(procs_matching "${pattern}" | tr '\n' ' ')"
    if [[ -z "${pids// /}" ]]; then
      echo "driver: the QC stopped after ${waited}s (its sidecar is complete up to the"
      echo "  last clip it finished; it resumes by stem if you ever run it again)."
      return 0
    fi
  done
  echo "driver: pid(s) ${pids% } ignored SIGTERM for ${KILL_QC_GRACE}s, sending SIGKILL." >&2
  pkill -KILL -f "${pattern}" 2>/dev/null || true
  return 0
}

if [[ -n "${WAIT_FOR_GENERATOR}" ]]; then
  wait_for_generator "${WAIT_FOR_GENERATOR}" || exit 1
else
  echo "driver: WAIT_FOR_GENERATOR is empty, not waiting for stage 05."
fi
if [[ -n "${KILL_QC}" ]]; then
  kill_qc "${KILL_QC}"
fi


if (( REFRESH_MANIFESTS )) && step_done manifests; then
  step_skip manifests "the NeMo manifest rebuild"
elif (( REFRESH_MANIFESTS )); then
  step_header "pre-step: rebuilding the NeMo manifests," \
       "splits pinned to ${PIN_SPLITS_FROM}"
  # Fatal if either fails, unlike the scan below: a half-written manifest set is what the
  # passes then read as the truth about the corpus.
  as_user "${UV_BIN}" run "${ROOT}/99_hf_release/01_build_nemo_manifest.py" \
    --all --pin-splits-from "${PIN_SPLITS_FROM}" \
    || { echo "!!! the per-source manifest rebuild failed, stopping" >&2; exit 1; }
  as_user "${UV_BIN}" run "${ROOT}/99_hf_release/02_combine_nemo_manifests.py" \
    --pin-splits-from "${PIN_SPLITS_FROM}" \
    || { echo "!!! the combined manifest rebuild failed, stopping" >&2; exit 1; }
  # Only now: both halves of the rebuild have to land for the manifests to be the truth.
  step_log "done manifests"
fi

if (( ACOUSTIC_SCAN )) && step_done acoustic; then
  step_skip acoustic "the acoustic loop scan"
elif (( ACOUSTIC_SCAN )); then
  step_header "pre-step: acoustic loop scan," \
       "${ACOUSTIC_WORKERS} CPU worker(s), no GPU and no server"
  # Never fatal: the sidecar is advisory (it says which clips repeat themselves, which
  # the passes do not need to run), and this scan is minutes to hours of CPU that must
  # not be able to stop a GPU run that would otherwise work.
  # A partial scan is NOT recorded: it resumes from its own sidecar, so re-entering it
  # costs only the clips it had not reached, and skipping it would lose them for good.
  if as_user "${UV_BIN}" run "${ROOT}/05_generate_audio/04_qc_acoustic_loops.py" \
      --workers "${ACOUSTIC_WORKERS}" \
      ${ACOUSTIC_STAGE_ARGS[@]+"${ACOUSTIC_STAGE_ARGS[@]}"}; then
    step_log "done acoustic"
  else
    echo "driver: the acoustic scan failed; continuing without its sidecar" >&2
  fi
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
  # Same shape and the same self-reopening hazard, for the wins that shipped over a gate.
  if [[ "${mode}" == stt ]] && (( RETRY_IMPROVED_PENDING )); then
    extra+=(--retry-improved)
    echo "driver: reopening the wins that still read bad (--retry-improved)"
    RETRY_IMPROVED_PENDING=0
  fi
  # `as_user` drops root and pins nice 19 (see its definition); the inner `env` adds the
  # STT config on top, because that is the only thing a pass needs beyond HOME and PATH.
  # runuser without -l keeps CWD (=HERE), so the relative --input/--audio-root resolve as
  # they do when run by hand, and uv is called by absolute path.
  #
  # nice 19 matters here in particular: this is an unattended job that runs for days and
  # must never be the reason an interactive session stutters, and it costs nothing, since
  # a pass waits on the GPU server far more than on the CPU.
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
  # And its own wins closed over a gate, so each gets one reopening of those too.
  RETRY_IMPROVED_PENDING="${RETRY_IMPROVED_ON}"
  # Rounds in a row whose tts pass found nothing to draft. Bounded below, see the loop.
  DRY_TTS_ROUNDS=0
  # A RESCORE_ONLY run is a one-pass migration, not convergence, so it records itself
  # under its own key: a later full run must not read it as "this dataset is done".
  if (( RESCORE_ONLY_ON )); then STEP_KEY="rescore-$((i + 1))"; else STEP_KEY="dataset-$((i + 1))"; fi
  if step_done "${STEP_KEY}"; then
    step_skip "${STEP_KEY}" "dataset $((i + 1))/${#MANIFESTS[@]} (${CUR_INPUT})"
    continue
  fi
  step_header "dataset $((i + 1))/${#MANIFESTS[@]}: ${CUR_INPUT} -> ${CUR_OUTPUT}/"
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
    step_log "done ${STEP_KEY}"
    continue
  fi
  while :; do
    # Single-mode runs loop on that one pass; the alternating run leads with stt.
    if [[ "${MODE}" == tts ]]; then FIRST=tts; else FIRST=stt; fi
    run_pass "${FIRST}"
    rc=$?
    if (( rc == 10 )); then
      echo "=== $(date '+%F %T') converged: no improvement work left on ${CUR_INPUT}. ==="
      # The ONLY place a dataset is recorded as finished. Every other break below is a
      # dead end rather than a convergence (clips nobody can clear, missing audio), and
      # those are exactly the ones a relaunch should re-enter, since the thing blocking
      # them is usually what the human fixed in between.
      step_log "done ${STEP_KEY}"
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
# ("whisper-1"), which would match nothing and write nulls. --stt-model goes with it so
# the `stt_model` column names the judge whose reading is in the row next to it.
if (( SYNC_RESULTS )) && step_done sync; then
  step_skip sync "the manifest sync"
elif (( SYNC_RESULTS )); then
  # That script reads a fixed pair of outputs (improved/, improved_parrot/), so a run
  # under a different OUTPUT has nothing for it to find. It says so per source, but by
  # then the run is over, so say it here first.
  case "${OUTPUT}" in
    improved|improved_*) ;;
    *) echo "driver: OUTPUT is '${OUTPUT}', while 03_sync_hotfix_results.py reads" >&2
       echo "  06_hotfixes/improved{,_parrot}/full.stt.jsonl. Expect it to find nothing." >&2 ;;
  esac
  step_header "post-step: refreshing durations, then syncing stage 06 into the manifests"
  # Two scripts, one step, like the manifest rebuild above, because the first exists only
  # to make the second correct. The sync copies the STATE row's duration INTO the
  # manifest on the premise that stage 06 re-probed it; only a redraw does, so a clip the
  # generator rewrote keeps the length it had in v1 (208,235 of 604,337 rows on
  # 2026-10-26). Without the refresh the sync overwrites the rebuild's true durations with
  # stale ones, and `tail_cer`, which returns null at or under 30 s, stays blind on every
  # clip whose real length crosses that window. Non-fatal: it reads 99_hf_release/data,
  # which a run from a copied driver (the tests) or an unmounted release symlink does not
  # have, and that must not take a finished QC pass down.
  if ! as_user "${UV_BIN}" run "${HERE}/13_refresh_stale_durations.py" --apply; then
    echo "driver: the duration refresh failed, so the sync below pushes stale durations" >&2
    echo "  into the manifests. Re-run 06_hotfixes/13_refresh_stale_durations.py --apply," >&2
    echo "  then 99_hf_release/03_sync_hotfix_results.py --apply, by hand." >&2
  fi
  if as_user "${UV_BIN}" run "${ROOT}/99_hf_release/03_sync_hotfix_results.py" \
      --apply --stt-key "${STT_MODEL}" --stt-model "${STT_MODEL}"; then
    step_log "done sync"
  else
    echo "driver: the manifest sync failed; the audio is fine, re-run" \
         "99_hf_release/03_sync_hotfix_results.py --apply, then 04_drop_bad_rows.py" \
         "--apply, by hand" >&2
  fi
fi

# The sync writes qc_status into the manifests but leaves the rows it marks bad
# (exhausted clips, eval texts that also sit in train, ...) in place; dropping them is
# 04's job, and it must follow every sync, or the next training run eats those rows and
# dies on its data-leak check (2026-11-07: two PARHAF test texts found in train). Its own
# step, so a resumed run skips it like the others, gated on the sync having finished,
# since it reads the status the sync wrote.
if (( SYNC_RESULTS )) && step_done drop; then
  step_skip drop "the bad-row drop"
elif (( SYNC_RESULTS )) && step_done sync; then
  step_header "post-step: dropping the rows that must not ship"
  if as_user "${UV_BIN}" run "${ROOT}/99_hf_release/04_drop_bad_rows.py" --apply; then
    step_log "done drop"
  else
    echo "driver: the drop failed; the manifests carry stage 06's status but still hold" \
         "the bad rows, re-run 99_hf_release/04_drop_bad_rows.py --apply by hand" >&2
  fi
elif (( SYNC_RESULTS )); then
  step_header "post-step: NOT dropping the bad rows, the sync above did not finish"
fi

echo "=== $(date '+%F %T') every dataset converged. Done. ==="
