# Alternating single-GPU improvement (STT and TTS cannot share VRAM)

`01_recursive_improvement.py` normally runs STT and TTS at once (`--mode both`).
When the two servers cannot fit on the GPU together, split the work into alternating
passes with `--mode stt` / `--mode tts`, driven unattended by `driver.sh`.

This setup was written with Claude Code.

## Why a split is needed

Scoring a regenerated candidate requires transcribing it, so `--mode both`'s repair
loop needs both servers up at the same time. The split breaks that coupling:

- `--mode tts` (only the TTS server up): for every clip a prior `stt` pass flagged
  bad, synthesize all `--n-improv` candidates and persist each as a
  `<clip>.draft<seed>.<fmt>` sidecar. No scoring.
- `--mode stt` (only the STT server up): transcribe and score the originals (flagging
  bad ones), and score the candidates a prior `tts` pass left, promoting the best draw
  (or exhausting the clip) and deleting the candidate files.

State lives in each row's `improvement.status` (`pending_tts` -> `pending_stt` ->
`improved` / `exhausted`), so every pass resumes where the last stopped. The only thing
lost versus `--mode both` is per-clip early stop: the whole candidate set is generated
before any is scored. That is the right trade when minimizing server swaps dominates.

The `stt` pass only picks from a **complete** candidate set. The pick is final (the clip
becomes `improved` / `exhausted` and is never revisited), so judging 3 candidates when
`--n-improv` asks for 5 would lock in a draw that may not be the best available: such a
clip goes back to `pending_tts` for the `tts` pass to make the set in full. "Complete"
means the `tts` pass delivered every draw it was asked for, not one file per draw: a
draw it skipped as a duplicate or as frame-cap truncated is skipped deterministically
(the seed of each attempt is fixed), so demanding a file for it would bounce the clip
between the two passes forever. In practice this triggers when `--n-improv` is raised
between runs, or when a sidecar has been deleted.

## How the alternation converges

`driver.sh` loops: switch to the STT server, run `--mode stt`; switch to the TTS
server, run `--mode tts`; repeat. The `--mode stt` pass exits **10** when no clip
anywhere still needs either pass, which is the driver's stop signal. A typical run is
`stt` (find bad clips) -> `tts` (draft) -> `stt` (score + promote) -> exit 10.

It is fine to start an alternating run on a dataset with **no scoring work left**: that
is what a run stopped (or killed) right after an `stt` pass leaves behind, every flagged
clip waiting on a `tts` pass. The `stt` pass then exits **11** ("this mode has run dry",
see below), which the alternating driver reads as "the work is on the other side" and
follows with the `tts` pass rather than stopping. It gives up on the dataset only when
both passes run dry in the same round, i.e. the leftovers are stuck (a flagged clip whose
audio has since gone missing, say), since swapping the two servers again would not move
them. Same stop for the less obvious shape of that dead end: two rounds in a row where
the `tts` pass has nothing to draft while the `stt` pass keeps reporting work done and
never converges, which means it is re-running rows it can never resolve.

### Running one half on its own

`MODE=stt` (or `MODE=tts`) runs only that pass and never brings the other server up,
for when the other half is down, unconfigured, or mid-rebuild. Scoring is useful on
its own: bad clips are flagged `pending_tts` and simply wait for a later run, so an
STT-only night loses nothing.

```bash
sudo -E MODE=stt CATEGORY="dictionary drugs" ./driver.sh
```

A single-mode loop stops on exit **11**, "this mode has run dry": the pass completed
nothing, so repeating it changes nothing, but clips are still waiting on the other
mode (which is why it is not 10). The alternating driver does see 11 and does not stop
on it: it just runs the other pass (see the paragraph above).

## One-time setup

**What CrispASR is.** A private, unpublished project of mine: a high-performance
self-hosted Whisper server (several GPU replicas behind a Caddy load balancer,
OpenAI-compatible `/v1/audio/transcriptions`), with the TTS service bolted into the
same compose project so both can take turns on the one GPU. Nothing in this stage
needs *that* server specifically. Any OpenAI-compatible transcription endpoint works,
and if your STT and TTS do not have to share a GPU you can skip `switch_server.sh`
and the alternating driver entirely and just run the passes. The references to it
throughout stage 05 and 06 are there because it is what actually built the dataset.

Both servers are services in the same CrispASR `docker-compose.yml`:

- STT: `crispasr` (whisper replicas) + `crispasr-lb` (Caddy, host `:8002`)
- TTS: `voxtral-tts` (host `:8003`)

1. **Config** (nothing is hardcoded, so no path or username is committed). Either keep
   these in your shell env (and launch with `sudo -E`), or put them in a local
   `06_hotfixes/.env` file (`KEY=value` lines) that `driver.sh` sources:

   ```sh
   CRISPASR_COMPOSE=/abs/path/to/CrispASR/docker-compose.yml   # TODO: your path
   WDOC_WHISPER_API_KEY=...    # token -> gitignore .env (or set CRISPASR_API_KEY)
   # Both optional now, and both default to the local v2 stack:
   # WDOC_WHISPER_ENDPOINT=http://127.0.0.1:8002
   # STT_MODEL=whisper.cpp/ggml-large-v3-turbo   # must match how the dataset was
   #                                             # transcribed, and is the key the
   #                                             # readings are stored under
   ```

   `.env` is sourced with `set -a` **before** those defaults, so a line there always
   wins. A v1 `.env` therefore keeps pointing at the old remote endpoint (which now
   answers 502) and keeps `WDOC_WHISPER_MODEL=whisper-1`, which would file every v2
   reading under the v1 key. **Set them to the values above rather than commenting them
   out**, because the documented launch is `sudo -E`, which imports the whole
   interactive environment: if your shell exports `WDOC_WHISPER_ENDPOINT` or
   `WDOC_WHISPER_MODEL` for its own tooling, those fill the `${VAR:-default}` slots and
   an absent `.env` line changes nothing. That is exactly what happened on 2026-10-20:
   two launches with a clean `.env` both announced the v1 remote judge, because
   `shared_env/70_wdoc.sh` exports it. An assignment in `.env` beats the inherited
   value; a comment does not.

   The driver prints the judge and endpoint it is about to use at startup and
   **refuses to start** when the endpoint is not local, because a warning scrolls past
   in a pane that then prints one wait line a minute for a day, and the cost of missing
   it is the whole sweep. Set `ALLOW_REMOTE_JUDGE=1` to score against a non-local host
   on purpose (a second machine doing the T2 opinion, say).

2. **Make the scripts executable:** `chmod +x switch_server.sh driver.sh`.

That is the whole setup: **no sudoers edit**. `driver.sh` runs as root (one password at
launch) so it can flip docker without a per-command sudo, and drops back to your normal
user (`$SUDO_USER`, or `TARGET_USER=<name>`) via `runuser` for each Python pass, so
every output / backup / draft file stays user-owned (matching the UID the containers
write as). It also sets the GPU power limit once at start (`nvidia-smi -pl`,
`GPU_POWER_LIMIT` default 350W).

## Run it

```sh
# from 06_hotfixes/, ideally inside tmux so it survives a disconnect:
sudo -E ./driver.sh
```

`-E` carries your session env (`WDOC_WHISPER_*`, `CRISPASR_COMPOSE`) into the root
process. Alternatively, put those vars in `06_hotfixes/.env` (sourced by the driver) and
just run `sudo ./driver.sh`. `.env` holds a token, so it must be gitignored.

Override any default via env, e.g. `N_IMPROV=8 CER_THRESHOLD=0.25 sudo -E ./driver.sh`.
The defaults mirror the manual runs (`--tts-voice fr_female`, now only a FALLBACK
for clips with no `voice` field in their manifest row: a v2 clip is regenerated with
the voice that read it originally, since the `voice` column is re-derived from the
group key and so cannot follow a regeneration; `--stt-parallel 10`,
`--tts-parallel 2`, `--shuffle`, output dir `improved/`, bad-clip threshold
`CER_THRESHOLD=0.08`, tail threshold `TAIL_CER_THRESHOLD=0.12`, minimum gain to replace a
clip `MIN_IMPROVEMENT=0.02`).

Two defaults changed for v2, both about concurrency. `STT_PARALLEL` is 20 (was 10): the
judge is now the LOCAL CrispASR stack on `:8002`, which `switch_server.sh` brings up as
several replicas behind the Caddy load balancer, and each replica serialises its own
requests, so the only way to keep six of them busy is to have more than six requests in
flight. `TTS_PARALLEL` is 32 (was 2), which stage 05 measured on 2026-10-14 on the same
server and clips: 3,860 clips/h at 16 in flight against 4,610 to 4,960 at 32, i.e. ~13 h
instead of roughly 10 days for a 12,000-clip queue at 5 draws each. 32 is the ceiling,
not a starting point: it is the largest captured CUDA graph, and past it vLLM drops to
the eager path. v1's 2 existed because that server was shared with interactive work.

### Launch it before generation ends (the gate)

`driver.sh` can be started while stage 05 is still generating. It does nothing until the generator is gone, which means the GPU goes from the last clip of generation straight into the QC phase with no idle night in between and no decision to make at 3 in the morning:

```sh
sudo -E ./driver.sh            # launch now, it starts itself when stage 05 exits
```

Two things happen before anything else touches the disk, the GPU or docker:

1. **Wait for the generator.** `WAIT_FOR_GENERATOR` is the `pgrep -f` pattern it polls for, default `generate_v2\.sh|01_generate_audio\.py`, i.e. the stage 05 run and its per-stage worker. Polled every `WAIT_POLL` seconds (default 60), with a `WAIT_SETTLE` window (default 120) of continued absence required before the phase starts, so a stage boundary or a resumed run does not read as the end of generation. `WAIT_TIMEOUT` (default 0, meaning wait as long as it takes) stops the run instead of starting. `WAIT_FOR_GENERATOR=` (empty) disables the gate. The one case it cannot tell apart is a generation **interrupted on purpose**, which looks exactly like a finished one, so it says so during the settle window: Ctrl-C there if you meant to resume stage 05 instead.
2. **Stop the live CPU QC.** `KILL_QC` is the pattern for that, default `03_qc_cpu_stt\.py|04_qc_acoustic_loops\.py`. SIGTERM first, because `03_qc_cpu_stt.py` traps it, finishes the clip in flight and prints its per-voice CER table on the way out; SIGKILL after `KILL_QC_GRACE` (default 60) for anything that ignored it. Its sidecar is append-only and resumes by stem, so stopping it costs at most the clips in flight. `KILL_QC=` (empty) leaves it running.

Why the process and not the container: the alternation's own tts passes need the TTS container **up**, so the container's state says nothing about whether generation is over. Why not a lock file: a crashed generation would never release one. And why it must wait at all, rather than just starting: the first `switch_server.sh stt` runs `docker compose stop voxtral-tts`, which would take the TTS server out from under a run that has been going for days, and the manifest refresh would describe a corpus that is still half-written.

The order is load-bearing in the other direction too. The QC is stopped **before** the acoustic scan starts, because the two want the same cores, and the stage 05 QC is redundant the moment this script begins: its 25 pct sample is subsumed by a sweep that reads every clip with a stronger judge. Config errors (an undecidable `PIN_SPLITS_FROM`, a missing compose file, no `uv`) are all reported **before** the wait, so the gate only ever blocks a run that is otherwise ready to go.

### The whole phase in one command

The three things that surround the alternation are steps of the driver, and all three are
**ON by default**, so the QC phase is one invocation rather than a checklist:

```sh
sudo -E ./driver.sh
```

In order: rebuild the NeMo manifests from the audio on disk (`99_hf_release/01_build_nemo_manifest.py --all`
then `02_combine_nemo_manifests.py`), scan that audio for acoustic loops
(`05_generate_audio/04_qc_acoustic_loops.py`, CPU only, `ACOUSTIC_WORKERS` processes,
default **the host's cores minus 2** (floor 2), restrict with
`ACOUSTIC_STAGES="01_dictionnary 04_PARROT"`), run the
alternation to convergence for every dataset, then carry stage 06's QC columns back into
the release manifests (`99_hf_release/03_sync_hotfix_results.py --apply --stt-key
$STT_MODEL --stt-model $STT_MODEL`) and drop the rows that status rules out
(`99_hf_release/04_drop_bad_rows.py --apply`; without it the bad rows stay in train and
the next training run dies on its data-leak check). Set any of `REFRESH_MANIFESTS`, `ACOUSTIC_SCAN`,
`SYNC_RESULTS` to 0 to skip that one, e.g. `sudo -E REFRESH_MANIFESTS=0 ./driver.sh` when
the manifests are already current.

The rebuild needs its splits pinned to the PREVIOUS generation's manifest root, and
`PIN_SPLITS_FROM` is **derived** when you do not set it: whatever `99_hf_release/data`
points at, with `NeMO_files` as its sibling (`data -> .../audio_datasets/v2` gives
`.../audio_datasets/NeMO_files`). That keeps a machine-specific path out of the repo while
still letting a plain run work. There is no unpinned fallback, ever: the splitter is
duration-greedy, so an unpinned rebuild redraws train/val/test and leaks v1's test set
into v2's train set with no error anywhere, and the model would then be evaluated on clips
it trained on (see `docs/V2_REGENERATION_RUNBOOK.md`). When no pin can be derived, the two
cases differ on purpose: `REFRESH_MANIFESTS=1` asked for explicitly **stops the run**,
while the default ON **skips the refresh** and says so, since a missing v1 root must not
block a corpus that only needs the passes.

The manifest rebuild is fatal if it fails (every pass after it reads those files as the
truth about the corpus); the acoustic scan is not (its sidecar is advisory, and hours of
CPU must not be able to stop a GPU run that would otherwise work); the sync warns rather
than failing, because by then the audio is already fixed and the sync can be re-run by
hand. The sync reads `improved/` and `improved_parrot/`, so the driver also warns up front
when `OUTPUT` is something else and it will find nothing.

Why the driver overrides the scan's own `--workers` default of 4: that default is sized
for the scan's *other* job, running beside the GPU generator as stage 05's live QC. Here
it cannot be beside anything, because the gate waited for generation to leave first. The
work is pure CPU with every BLAS/OMP thread pinned to 1 before numpy loads (measured 122
ms unpinned vs 42 ms pinned on one 20 s clip: the matrices are small, so the thread pool
costs more than it buys), so it is one core per worker with nothing shared, and 4 workers
on a 12-core box left two thirds of the machine idle: ~15 clips/s per worker, i.e. 60
clips/s instead of 150, which is the difference between half an hour and an hour and a
half over 533k clips. Set `ACOUSTIC_WORKERS` by hand when something else heavy shares the
machine.

### Stopping it, and picking up where it was

The phase is a day or more of work, so the same command can be relaunched after a stop
(Ctrl-C, a reboot, a machine that had to be used for something else) and it continues
instead of starting over:

```sh
sudo -E ./driver.sh            # run 1: rebuild, scan, passes, sync
# Ctrl-C somewhere in the middle
sudo -E ./driver.sh            # run 2: skips what run 1 finished, carries on
```

Every step appends a line to `improved/driver_steps.log` (or `<OUTPUT>/driver_steps.log`,
override with `STEPS_LOG`) when it **succeeds**, and the relaunch skips what is in there,
saying when it was done:

```
=== 2026-10-21 16:04:11 [step 1/4] skipping the NeMo manifest rebuild: finished 2026-10-21 14:01:04 in an earlier run (RESET_STATE=1 to redo it) ===
```

Those `[step N/M]` headers are on every step either way, so the position in the phase is
readable from one line of the log instead of counted by hand. The log is also the phase's
own history: `cat improved/driver_steps.log`.

What counts as finished is deliberately narrow, because the cost of wrongly skipping a
step is a corpus that silently never got the work:

- the **manifest rebuild** only when both builders landed (a half-rebuilt manifest set is
  what every pass after it reads as the truth about the corpus);
- the **acoustic scan** only when it ran to the end. A partial scan is not recorded on
  purpose: it resumes from its own append-only sidecar, so re-entering it costs only the
  clips it had not reached, while skipping it would lose them for good;
- a **dataset** only when the `stt` pass reported convergence (exit 10). Every other way
  the per-dataset loop can end is a dead end, not a result: clips nobody can clear,
  missing audio, both passes dry. Those are exactly what a relaunch should re-enter,
  since whatever was blocking them is usually what got fixed in between. A `RESCORE_ONLY`
  migration records itself under its own key for the same reason, so it cannot be read as
  a converged dataset later;
- the **sync** only when it applied.

Two escape hatches. `RESET_STATE=1` moves the log aside and re-runs everything, and a log
whose first line describes a **different** run (another corpus, `OUTPUT`, `MODE` or judge)
is moved aside by itself rather than trusted, since none of its markers would mean
anything here. The passes themselves were always resumable (each rebuilds its
`<output>/<stem>.stt.jsonl` from the manifest and keeps the stored readings, see "When the
manifest is rebuilt under a running output"); this only decides which **steps** to
re-enter.

One caveat that has nothing to do with the log: do not edit `driver.sh` while a run is in
flight. Bash reads a script in chunks as it goes, so inserting lines above the point it
has reached shifts every byte after it and the running shell resumes parsing in the wrong
place. Edit it, then relaunch, which is now cheap.

### The candidate sidecars on disk, and `09_prune_drafts.py`

A `tts` pass persists every draw it keeps as `<clip>.draft<seed>.<fmt>` next to the clip,
so a flagged clip costs up to `--n-improv` extra files. They are not small: measured on
the first v2 round, a candidate is **0.84 MiB** on average, so the 12.4k clips the judge
flagged come to about **49 GiB** of sidecars while the round is in flight. Worth knowing
before starting a round on a disk with 96 GiB free.

Nothing needs doing in the normal cycle: the `stt` pass deletes a clip's whole candidate
set the moment it judges it (win or lose), and a `tts` pass clears what it finds before
redrawing, so the sidecars are transient by construction. `09_prune_drafts.py` is for the
two cases that cleanup cannot reach, where files are left with nobody who will ever look
at them again: a round that was **interrupted** (or whose dataset later converged by
another route), and a queue **dropped after a scoring-rule change** (see `RESCORE=1`).

```bash
# always a dry run first: it prints each bucket with its size and deletes nothing
uv run 06_hotfixes/09_prune_drafts.py \
    --state 06_hotfixes/improved/full.stt.jsonl \
    --audio-root 99_hf_release/data/NeMO_files
# then the same line with --apply
```

Pass one `--state` per dataset (with its `--audio-root`, the manifest's own directory,
the same value the driver passes; a single root covers every state file). Doing all of
them at once is what makes the orphan count meaningful, since the datasets share audio
directories.

The rule, which is what the tool is really about: a clip whose row says **`pending_stt`
keeps every candidate beside it**, because those are exactly the files the next `stt`
pass is about to judge, including one the row does not list (a pass that died between
writing a sidecar and recording it). `pending_tts`, `improved`, `exhausted` and "no
marker at all" are all dead. A candidate whose clip is in none of the `--state` files is
an **orphan**: counted and reported, deleted only with `--delete-orphans`, since the
usual cause is a dataset that was not passed in. `<clip>.bak` backups are never touched:
they hold the only remaining copy of the audio an improvement overwrote.

Two guards, because what it deletes costs GPU hours to make. It is a dry run unless
`--apply`, and it **refuses to delete while a pass is live**: the state file lags the
filesystem by one flush (adaptive, a few minutes on a 900 MB file), so a clip can own
fresh candidates while its row still reads `pending_tts`. Deleting those would not
corrupt anything, the clip would simply be redrawn, but it would throw away real work.
`--allow-running` overrides that, and `--min-age-min` (60 by default) stays in force
either way. The live-pass check matches the script as its own argv element under a real
interpreter rather than anywhere in the command line, so the shell you launch it from
cannot look like a running pass (the `pkill -f` trap).

Pinned by `tests/test_prune_drafts.py` (`uv run tests/...`), including the round trip
against the pass's own `draft_path_for`, so the two halves of the naming convention
cannot drift.

### Which judge, and where its reading is stored (`STT_MODEL`)

`STT_MODEL` is both the model name sent to the STT server and the key the reading is
stored under on each row (`transcriptions[<model>]`). That one variable is what lets the
v2 plan use two judges on the same corpus: `whisper.cpp/ggml-large-v3-turbo` (the
default) sweeps all 602k clips, then the slower `whisper.cpp/ggml-large-v3` re-reads only
what turbo flagged, with `STEMS_FROM` pointing at the flag file. Different key, same row,
so neither overwrites the other and the two can be compared:

```sh
sudo -E MODE=stt STT_MODEL=whisper.cpp/ggml-large-v3 STEMS_FROM=flags.txt ./driver.sh
```

`WDOC_WHISPER_MODEL` is the older name for the same thing and still works, and it is
what the existing v1 outputs were judged under (`whisper-1`, from `.env`). Mind that the
key is also the **resume** key: a pass only sees readings stored under its own
`STT_MODEL`, so renaming the judge makes every earlier reading invisible and the corpus
is transcribed again (~31 h for 602k clips). For v2 that costs nothing, since the audio
is new, but it is a reason to settle on the judge's name before the first full sweep.
`SYNC_RESULTS=1` passes `--stt-key "$STT_MODEL"` for the same reason: that script's own
default is v1's `whisper-1` alias, which against a v2 row matches nothing and would write
every QC column empty (it now warns and lists the keys it did find).

The endpoint and token follow from that. `WDOC_WHISPER_ENDPOINT` defaults to
`http://127.0.0.1:8002`, the local stack, because the remote endpoint v1 ran against now
answers 502, and a non-local value is refused at startup unless `ALLOW_REMOTE_JUDGE=1`
says it is deliberate (see the config step above for why a warning was not enough). That service sets `CRISPASR_API_KEYS`, so transcription needs a bearer
token: `WDOC_WHISPER_API_KEY` falls back to `$CRISPASR_API_KEY`, the name the CrispASR
clients already use, and the driver warns at startup when neither is set rather than
letting every clip 401.

### What differs between the draws of one redo

Two things, and only two. Draw N (from 0) gets seed `--start-seed + N` (43, 44, 45, ...)
and classifier-free guidance `CFG_ALPHA_START + CFG_ALPHA_STEP * N`, which the driver
defaults to 1, 1.5, 2, 2.5, 3 across five draws. The spread is the point: a clip the TTS
derailed on is unlikely to be rescued by four more draws at the setting that derailed
it, and since the winner is picked on CER, a draw that guidance pushed too hard simply
loses. It stays near the server's tuned value (it boots at 1.3) instead of sweeping far
out, so the draws land where the model is actually usable. `--speed-jitter` is NOT a
third lever (it is off by default and should stay there): vllm-omni applies `speed` as a
phase vocoder over the finished waveform, so it
time-stretches identical audio instead of drawing it differently.

Both knobs are sent as flat request fields, and both of them require the **forked**
vllm-omni that the CrispASR `voxtral-tts` image builds from
(`CrispASR/voxtral/vllm-omni`). Against a stock server they are silently inert: upstream
has no `cfg_alpha` field at all and drops it (pydantic `extra='ignore'`), and its `seed`
only reaches a sampler that Voxtral's stage 0 feeds forced logits, so the audio's
randomness (the flow-matching Gaussian) comes off the global torch RNG regardless. That
was the state of this stage for a while: five "identical" requests whose audio differed
only by RNG drift, no error anywhere. `tests/test_tts_request_knobs.py` pins the request
body so it cannot regress quietly. One server-side requirement comes with it: stage 0's
deploy YAML must keep a default `cfg_alpha` under `default_sampling_params.extra_args`,
or vllm-omni sets `has_sampling_extra_args=False` and drops every per-request extra.

The clips regenerated during that inert period had already been recorded with the knobs
the client ASKED for (`cfg_alpha` 1.3 to 1.7, `seed` 43 to 47), which is not what made
the audio. `oneoff_fix_preforked_knobs.py` is the one-shot migration that corrects them:
for every win drawn before the fork went live (`--cutoff`, which sits between the last
promotion of a pre-fork draw and the first post-fork one, NOT at the image build, because
the tts pass persists candidates that a later stt pass promotes) it sets `cfg_alpha` to
the server's startup value
(`--server-cfg-alpha`, 1.3 from `VOXTRAL_CFG_ALPHA`), sets `seed` to `null` (the draw was
unseeded, and since every draw of one redo went out with identical effective parameters,
which one won carries no information), and stamps `knobs_effective: false` on the audit
record so it still says why. It touches the audit file, the row's `cfg_alpha` column and
the row's `improvement` marker; it is a dry run unless given `--apply`, refuses to run
while a pass is alive, keeps a `.prefork.bak` copy, re-reads what it wrote, and is
idempotent. It has been run over 289 wins (264 pre-fork draws, then the 25 candidates the
interrupted run had already synthesized and the restarted driver only scored afterwards);
it should not be needed again.

### Two thresholds: whole clip and tail

A clip is flagged for regeneration when EITHER gate fires:

| Gate | Env | Default | Applies to |
|------|-----|---------|-----------|
| whole-clip CER | `CER_THRESHOLD` | 0.08 | every scored clip |
| tail CER | `TAIL_CER_THRESHOLD` | 0.12 | clips longer than 30 s only |

Both defaults are a SWEEP DEPTH, not a claim about where good audio ends. The corpus is
taken to convergence at one pair (no clip left above either gate), then the gates are
lowered and it is swept again, worst first. It has converged at 0.15 / 0.20 (505 clips
regenerated, 500 of them kept); 0.08 / 0.12 is the next sweep, ~1,550 clips and ~10.5 h of
audio, of which the first ~395 are what 0.10 / 0.15 alone would have queued.

The tail CER scores only the last ~30 seconds (the last `TAIL_CHARS` normalized characters
of each side, extrapolated from the corpus median pace of seconds per character). A long
clip that loops, derails or is cut short at the end can still pass the whole-clip gate
because its correct beginning outweighs the broken ending, and the tail gate is what
catches it. Its threshold is looser than the whole-clip one because the window boundary
cuts mid-sentence: the window is ~500 characters, the size of a whole PARHAF chunk rather
than a fifth of one. Measured over 23,639 long clips of the finished corpus, the tail
distribution is P95 0.060, P99 0.161, P99.5 0.191: 0.20 sat just past P99.5, and the
current 0.12 sits around P98 (390 long clips).

The two gates are NOT scaled versions of each other, and moving one is not a reason to
move the other. Counting the clips the tail gate catches on its own (whole-clip CER under
the gate, tail over it) and detecting the TTS derailments among them mechanically (a
25-character window the transcript repeats three or more times):

| tail band | clips caught by the tail gate alone | of them looping |
|---|---|---|
| 0.20 to 0.25 | 56 | 18 |
| 0.25 to 0.30 | 24 | 5 |
| over 0.30 | 2 | 0 |

That table was measured when the gate sat at 0.20, and it answers the question of whether
to RAISE it: raising does not buy precision, it discards the band where the real
derailments are. 0.25 would drop 18 of the 23 confirmed loops, and 0.30 turns the gate off
(2 clips, none of them looping). Set `TAIL_CER_THRESHOLD=-1` to disable it entirely.

Going the other way is a different trade. The whole-clip band below 0.15 is genuinely
mixed: sampling the diffs, ~43% of clips at 0.10-0.12 and ~57% at 0.12-0.15 carry a
dropped or invented run of words, but the rest is French agreement homophony
(`adenopathies axillaires reactionnelles` heard as the singular, same audio), Whisper's
spelling of rare medical terms, and notation neither side folds yet. So a lower gate
spends GPU time on clips that were never broken, and it also degrades the PICK: the loop
promotes the draw whose transcript best matches the label, so below the noise floor it
selects whichever draw best fits Whisper's spelling habits rather than the best audio.

`MIN_IMPROVEMENT` is what makes a low gate safe. A draw that clears both gates is always
promoted, but a draw that only comes closer has to beat the original by that margin
(0.02) to replace it. Without it (the script's own default is 0.0) any gain at all wins,
which at a gate of 0.08 means swapping audio on a 0.002 CER difference, i.e. on nothing.
Clips whose best draw does not clear that bar are marked `exhausted` and keep their
original audio, so a sweep that finds nothing real costs time and no quality.

### What the CER ignores on purpose

The audio is what is on trial, but the two sides of the comparison write it down
differently: the label spells out what a French dictation spells out, and Whisper
abbreviates whatever it hears. Before scoring, both sides are therefore folded to one
written form per spoken thing (`normalize_for_scoring`), on top of the older case /
accent / ligature / punctuation folding:

| Class | Same thing, both spellings | Canonical |
|---|---|---|
| units | `500 mg` = `500 milligrammes`, `4,2 mmol/L` = `4,2 millimoles par litre` | spoken |
| units, squared / cubed | `750 cm3` = `750 centimetres cubes`, `2 mm2` = `2 millimetres carres` | spoken |
| unit inflection | `par kilogramme` = `par kilogrammes` (every word, either side) | the table's form |
| rate tails | `15 milligrammes par kg` = `... par kilogramme` = `... par kilo`, `15 mg/kg` | spoken |
| `G/L` vs `g/L` | `189 G/L` = `189 giga par litre`, kept apart from `grammes par litre` | spoken, case-sensitive |
| titles | `M. Dupont` = `Monsieur Dupont`, `Dr` = `docteur` | spoken |
| percent | `96 %` = `96 pour cent` | one word |
| numbers | `trois` = `3`, `quatre-vingt-seize` = `96`, `stade IV` = `stade quatre` | digits |
| decimals | `0,5` = `zero virgule cinq` | `0 5`, separator dropped |
| thousands / leading zeros | `11 500` = `11500`, `05 janvier` = `5 janvier` | bare digits |
| ordinals | `premier` = `1er` = `1re`, `troisieme` = `3eme` = `3e`, plural too (`quatriemes` = `4emes`) | `<n>e` |
| spoken symbols | `1+` = `1 plus`, `T2*` = `T2 etoile` | spoken |
| ratios | `3/0` = `3 barre 0` = `3 0`, `120/80` = `120 sur 80` | `<n> <n>`, separator dropped |
| day offsets | `J2` = `J+2` = `J plus 2` | `j plus <n>` |
| compact times | `2h05` = `2 heures 5` = `2 heures 5 min` = `2 heures 5 minutes` | `2 heures 5` |

The unit table is not a second copy: it is `utils/voxtral_normalize.py`'s `UNIT_SPOKEN`,
read the other way round (that module spells the unit out for the TTS, the scorer folds
Whisper's abbreviation back onto it), plus the bare units voxtral pronounces correctly
and so does not list. Numbers canonicalize to digits because that direction needs only a
reader, not a French number speller.

Measured on 12,000 real scored clips when the first table landed: mean CER 0.031 to
0.013, clips over the then-0.15 gate 63 to 11, clips over the tail gate 15 to 1. A third
of clips score slightly WORSE, by a median of 0.0001: folding shortens the reference, so
the errors that are real weigh a little more, which is the point.

The later rows of the table were picked off a cost ranking rather than guessed: every
surviving diff over 60,000 scored clips, each pair weighted by the characters CER charges
for it. The top of that list was not exotic, it was `par kilogramme` against `par
kilogrammes` (7,128 characters), `cm3` against `centimetres cubes` (6,585), the decimal
comma (5,422), the accented `zero` that could never match the accent-folded text it was
compared against (3,832), and `G/L` read as grammes (3,718).

Every rule was then replayed over a 30,000-clip sample, comparing the stored CER with the
re-derived one and printing the clips that got WORSE, which is how the number separators
ended up being DROPPED rather than spelled. Spelling the decimal out (`3,0` to `3 virgule
0`) looked right and was the single worst rule in the batch: surgical reports are full of
suture gauges the label writes `3 0` and Whisper writes `3,0` or `3/0`, and only deletion
puts all three spellings on one string. The same replay demoted `J+7` to its short form
(Whisper mishears the letter as `G` often enough that the long form charged 7 characters
for a 1-character error) and put the letter guard on the unit rules (`M1M2`, the
metatarsals, was being read as square metres). Final effect: mean CER 0.01337 to 0.01272,
clips over the 0.10 gate 110 to 92, and the clips that score worse do so by a median of
0.00012, only 13 in 30,000 by more than 0.01. A fold that fires on one side but not the
other is worse than no fold at all, so a new rule is worth the corpus replay every time.

The rate tail came later, off the same kind of ranking run over the clips no redraw could
fix: the unit rules are anchored on a NUMBER, so `15 mg/kg` folds whole while the form the
two sides actually disagree on (`15 milligrammes par kg` against `par kilogramme`) only
folds at the head. `kg` alone was the most frequent surviving diff in those clips, and
`par kg` appears in 8,015 clips of the corpus. Folding a unit that sits after `par` is
restricted to multi-letter symbols on purpose: `par l` is the elided article far more often
than it is litres, and the punctuation strip turns `par l'aorte` into `par l aorte`.

That rule is also the one that most needed the corpus replay, and needed it over the WHOLE
corpus rather than a sample: at 30,000 clips it looked perfect (214 better, 0 worse), and
at 601,289 it was not. Two of its entries had to go:

- `kilo` fires only after `par`, never after a number. Number-anchored it ate the head of
  `87,5 kilo-unites par litre` (the IgE assay unit, kU/L) and stretched Whisper's
  mishearing of `3 culots globulaires` as `3 kilos` into a longer error.
- bare `mm` is barred entirely. It earns 4 clips in 600k (a rate tail is a dose
  denominator, not a length, and `par mm3` has its own entry) and it collides with a real
  one: Whisper writes the M-M-RVAXPRO vaccine as `MM-VAX-PRO`.

Final effect over all 601,289 scored clips: 8,707 better, 5 worse, 174 clips crossing under
the 0.08 gate, tails 384 better and 6 worse. The 5 survivors share one benign shape, a real
STT error sitting inside a rate the label states twice and the transcript once, so folding
both sides makes the difference between them 7 characters longer.

### Count charged characters, not clips whose CER moved

The last batch of rules (six imaging / physiology units, the clock time, plural ordinals)
was measured the usual way first, and the usual way said to throw it out: replayed over the
whole corpus it left 1432 clips with a lower CER and **1688 with a higher one**. Reading the
regressions clip by clip showed why that number is not what it looks like. CER is edit
distance over the length of the normalized reference, and a fold moves BOTH. Folding
`dixiemes` to `10e` shortens the reference by five characters, so every clip carrying an
unrelated error elsewhere scores a higher CER on exactly the same errors, and a fold that
lengthens the reference (`ms` to `millisecondes`) quietly flatters itself the same way.

So the question a replay must ask is not "did CER move" but **"did the fold stop the clip
being charged for something that was never an error"**, which is the raw edit distance on
the normalized strings, plus how many clips cross the gate. Measured that way, over the
601,289 scored clips:

| rule | charged less | charged more | net chars | crosses under 0.08 | crosses over |
|------|--------------|--------------|-----------|--------------------|--------------|
| the six units | 714 | 6 | -6,395 | 14 | 0 |
| plural ordinals | 141 | 2 | -1,533 | 3 | 1 |
| clock time | 570 | 50 | -4,900 | 6 | 0 |
| all three | 1,423 | 58 | -12,828 | 23 | 1 |

Same corpus, same rules, opposite conclusion: 1688 clips "worse" is 58 clips actually
charged more. All six unit regressions are one family, the label writing the symbol while
Whisper mangles it (`662 keV` heard as `kV`, `465 ms` as `mSq`, `0,2 milligray` as `mY`),
together about 25 characters against 6,395 saved, so they stay.

One rejected candidate is worth recording, because it looked obligatory: the label often
writes the `et` (`1 heure et 10 minutes`) where Whisper writes `1h10`, and the clock rule
does not match across it, so it strips the word from one side only. Adding an `et` rule
does fix those clips, and over the corpus it buys 35 and breaks 24 for no gate movement at
all. Not worth the rule. `1 heures 10 minutes` in a label whose transcript has no clock
form at all is the same shape and is the bulk of the clock rule's own 50.

What is deliberately NOT folded: French plural and gender endings (`antalgiques` vs
`antalgique`, `suivie` vs `suivi`). They are inaudible and they do show up in the diffs,
but folding them would also hide a TTS that dropped a word, and at 1-2 characters they
cost far less than the classes above. Bare `m` only counts as a title in front of a real
word, so a spelled-out `L.M.B.R.1` is not read as `L. Monsieur B. R. 1`.

Changing any of this changes every stored CER, so follow it with a `RESCORE_ONLY=1` pass
(see below): it re-derives the scores from the transcripts already on disk, without
transcribing anything again.

### The generation guards on a draw (`AUTO_TOKEN_BUDGET`, `CODE_DUMP_DIR`)

Before a draw is worth an STT call it has to survive the same three guards the
**generation** pass generates under, with the same numbers (they live in
`utils/frame_budget.py` and `utils/code_loops.py`, and stage 06 calls them rather than
carrying a copy, so re-measuring a voice moves both stages at once):

* **a per-clip frame budget**, on by default (`AUTO_TOKEN_BUDGET=1`). Each draw is asked
  for only the frames its text can plausibly need, from that text's length and the
  **voice's** measured slow rate (`fr_female` 9.9 chars/s, `h_frm_bre` 8.6, and so on;
  5.0 for a voice not in the table), plus 3 s of head and tail. A draw that comes back
  past that budget was looping rather than reading, and is discarded. This is the guard
  that matters most for cost: without it a looping draw runs to the server's 327.68 s
  cap, which is about six minutes of GPU per draw and five draws per clip, to produce a
  take that is then thrown away anyway. `AUTO_TOKEN_BUDGET=0` falls back to
  `MAX_AUDIO_SECONDS` alone (next section).
* **a floor**, always on, budget or no budget: it needs nothing from the server. A draw
  far shorter than its text can be read in did not speak all of it, which is precisely
  the defect this stage exists to repair, so keeping it would be the worst outcome
  available. Rejecting it here also saves the transcription that would have scored it.
* **the code gate**, off unless `CODE_DUMP_DIR` is set. That is the HOST path of the
  directory the TTS server writes code dumps into, i.e. the host side of
  `$VOXTRAL_CODE_DUMP_DIR` in the CrispASR compose (container `/cache/cloning/dumps`);
  only the operator knows where their checkout sits, so nothing can derive it. With it
  set, each draw's own semantic codes are read back and a draw that collapsed onto a
  handful of codes, held one for over a second or looped a unit is discarded. That is the
  half of the severe defects a duration check cannot see: a plausible length for the
  text, and still not a reading of it. Pointing it at a stock, unpatched server is
  harmless, since no dump is written and an absent dump reads as "no opinion", never as a
  rejection. One row per draw is appended to the `code_qc_sidecar.jsonl` beside the
  audio, the same file stage 05 writes, tagged `"stage": "redo"` with the draw's seed so
  the two are distinguishable; `--no-store-semantic-codes` keeps the measured features
  and drops the codes themselves.

Each rejected draw logs its own line with the numbers behind the verdict (how far over or
under, against which bound, for how much text), and a clip that lost draws to the guards
gets one summary line: five "over" rejections out of five is not bad luck, it is a clip
whose text no seed can fit.

### The TTS frame cap (a "stopped for length" guard)

`/v1/audio/speech` returns raw audio with no `finish_reason`, so a generation that ran out
of tokens instead of reaching the end of the text still answers HTTP 200, with a clip that
just stops mid-sentence. The only observable is a duration pinned at the backend's frame
cap, so any draw landing within half a second of `MAX_AUDIO_SECONDS` (default 327.68 s, the
served 4096 frames at 12.5 Hz) is discarded rather than scored or promoted, and the clip
keeps its original. With `AUTO_TOKEN_BUDGET=1` (the default) this is only the fallback
bound: each draw is held to its own per-clip budget instead, which is tighter by one to
two orders of magnitude on a dictionary term, and `MAX_AUDIO_SECONDS` then only matters
for a run with the budget turned off. When every draw for a clip hits the cap the log says so explicitly: no
seed can fix it, the source text is simply too long to be spoken in full, and the fix is
re-chunking that text. Keep `MAX_AUDIO_SECONDS` in step with the server's actual
`VOXTRAL_MAX_TOKENS` (163.84 for the packaged 2048 frames, 327.68 for 4096);
`MAX_AUDIO_SECONDS=0` disables the check.

Note this deliberately does NOT re-flag the old clips stuck at 163.84 s, which were
generated back when the server capped at 2048 frames: they are truncated, but under the
current cap they are simply bad audio, and the CER / tail gates already catch them and can
now regenerate them in full. Stage 05 applies the same guard when it generates a clip in
the first place
(`--max-audio-seconds`), where a truncated clip is not written at all so a resume retries it.

### The per-round history: `rounds.jsonl` and `--report`

Each pass logs plenty about what it is doing right now and nothing that survives it, so
"how much did this improve the corpus" used to be a question you answered by diffing two
Markdown reports by eye. Every pass now appends **one line per dataset** to
`rounds.jsonl` beside its output (`improved/rounds.jsonl`,
`improved_parrot/rounds.jsonl`), describing the dataset as it stands when that pass ends:

* `states`: rows, scored, `bad` (fails any of the four gates), `pending_tts`,
  `pending_stt`, `improved`, `exhausted`, `clean` (scored, passes every gate, no marker);
* `scores`: the median and p90 of `cer`, `cer_tail`, `cer_window` and `bad_run_chars`,
  with the count that carried each one;
* `work`: what this pass itself did (queued, done, transcribed, rescored, improved,
  exhausted, swapped, requeued, unflagged, ...), which is a different question from where
  the dataset stands;
* `gates` and `n_improv`: the numbers the pass judged by, because `bad` means nothing
  without them. A run that lowers a gate mid-sweep would otherwise read as a corpus that
  got worse;
* `audio_hours`, `wall_s`, `ts`, `mode`, and the `round` / `pass` counters.

A **round** is an stt pass plus the tts pass it feeds, which is what the driver
alternates and what "one more round of draws" means: an stt (or `both`) pass opens a
round, the tts pass joins it. Numbering is per dataset, so round 3 of the main corpus
says nothing about PARROT.

The whole line comes from one extra parse of the written output (`scan_output`), which
also feeds the convergence check the driver loops on, so the ledger costs one scan of the
file per pass and no second pass over the corpus. The scores are **read back, not
re-derived**: the pass already stored all four on every row it scored, and re-deriving
602k rows would cost ~1.8 h of jiwer for numbers that are already on disk.

Read the history with:

```bash
./01_recursive_improvement.py --report              # ledgers here and one level down
./01_recursive_improvement.py --report improved     # or one specific output dir
```

`--report` is eager, so it needs none of the run options (no `--input`, no
`--output`, no `--stt-endpoint`) and no server: it is meant to be run from another tab
while a pass is working. It prints one table per dataset, one line per pass, then the
delta against the first round on record, the total wall time, and whether anything is
still pending. `driver.sh` prints the same table for a dataset when that dataset's loop
ends, so the driver log carries the before / after too.

```
improved/full.stt.jsonl: 3 pass(es), 2 round(s)
  rnd pass mode      rows     bad pend_tts pend_stt   impr    exh  swap           cer  ...
    1    1 stt    601,338  12,044   12,044        0      0      0     0   0.012/0.031  ...
    1    2 tts    601,338  12,044        0   12,044      0      0     0   0.012/0.031  ...
    2    3 stt    601,338   1,943    1,788        0 10,101  1,943 1,788   0.012/0.029  ...
  since round 1 (2026-10-21T10:00:00+02:00): bad -10,101 (-83.9 pct), cer p50 -0.0004 ...
```

The ledger is a snapshot, deliberately, and it is useless as a source of truth about any
individual clip: `full.stt.jsonl` owns that, and the ledger must never become a second
copy of it. Appending is best effort, so a ledger that cannot be written warns and lets
the pass finish, and an unparseable line is skipped rather than fatal.

### Another reader for a clip no draw could fix (`VOICE_SWAP`)

A clip that loses all `N_IMPROV` draws is usually not an unlucky seed, it is a text its own voice cannot read: a number its accent says in Spanish, a Latin term it stumbles on every time. v1 shipped the original audio and labelled it `exhausted`. With `VOICE_SWAP=1` (the default) the pass instead walks that clip up an escalation **ladder** of voices, one rung per exhausted round, each rung getting a full `N_IMPROV` draws with the cfg_alpha ladder starting over from the top:

| rung | voice | why |
|---|---|---|
| 0 | the clip's own | unchanged from v1 |
| 1 | `h_itf_arm` or `h_itf_def` | the two best hybrids, 1.4 and 2.2 pct gate-fail on long clips |
| 2 | `fr_male` or `fr_female` | the shipped bases, 5.8 and 0.0 pct, last resort |
| off the end | none | `exhausted`, ships the **original** audio, as in v1 |

The ladder is `utils/voice_assignment.py::VOICE_LADDER_FR_V2` and only those four voices are on it, because the first full QC sweep (2026-10-20, 143,093 clips) measured the roster splitting in two with nothing in between: inside the 40-60 s band, `fr_female` 0.0 pct over a gate, `h_itf_arm` 1.4, `h_itf_def` 2.2, `fr_male` 5.8, then `h_frf_esm` 41.5, `h_frm_ptf` 44.2, `h_frf_itm` 48.3, `h_frm_nlf` 60.2 and `h_frf_ptm` 77.4. Redrawing a 50 s clip at one of the five is not a repair attempt, which is what the old any-other-roster-voice swap was doing. Hybrids come before the bases on purpose: the hybrids are why the roster has nine voices rather than two, so the repair keeps the accent spread as long as it can. `docs/V2_QC_PHASE_PLAN.md` item 8 has the full measurement.

Which of a rung's two voices a clip gets is the same hash that assigned the old one, so there is no second algorithm to keep in step and a re-run gives the same answer (`ladder_voice`, which is `voice_for_key` over the rung). It still prefers a voice **not already reading that term**, so a term that lands twice in the corpus keeps two accents; that preference is dropped rather than refusing to swap when both voices of the rung are already on the term. The index of voices-per-term costs nothing: the `needs_work` hook already visits every row of the manifest before any worker starts, including the rows this pass will not touch, so it is harvested there.

Three properties worth knowing:

* **Document stages swap too, but a document gains one reader per rung, not one per chunk.** `01_dictionnary`, `02_drugs` and `07_acronyms` assign a voice per clip, so another reader is just another reader, and the rung is keyed on the clip stem. `03_PARHAF` and `04_PARROT` assign one reader per *document*, so there the rung is keyed on the **group id**: every chunk of that document which needs repairing lands on the same new reader. These stages refused to swap at all until 2026-10-20, to keep the single-reader property. Then the sweep measured 22.62 pct of PARHAF clips over a gate against ~1 pct everywhere else, so refusing meant shipping the defect, and audio that does not say its text is worse label noise for an ASR corpus than a document changing reader once. The pass carries no stage list for this: it reads `mode` out of the `voice_assignment.json` beside the audio, so the rule follows the audio even if a stage is renamed or a directory is moved.
* **Two rungs per clip at most, so it cannot wander the roster.** The state is recorded on the row as `improvement.swap_voice` (the voice to draw with) and `improvement.voice_tier` (how many rungs are gone), and a clip that already failed a rung's voice is never sent back to it. Worst case is 3 x `N_IMPROV` draws spread over three rounds, after which the clip is `exhausted` with its **original** audio, exactly as in v1.
* **The original audio is untouched until a draw wins.** A swap only queues the clip back into the TTS half (`pending_tts`), so it travels the normal `pending_tts -> pending_stt` alternation and works the same whether the pass runs in `both` mode or as two separate halves. `voice` on the row is rewritten only when a draw actually ships, because that column has to describe the audio on disk.

One consequence outside this stage: `99_hf_release/01_build_nemo_manifest.py` **re-derives** the `voice` column from the assignment hash, which cannot know about a swap, so `99_hf_release/03_sync_hotfix_results.py` copies the won voice back over that re-derivation (exactly as it already does for `duration` and `cfg_alpha`). Run the sync after the improvement run and before building the parquet, or the release will name the voice that failed. It also means the distinct-voices-per-term numbers the release README reports change once swaps have happened.

In the ledger, `swap` counts every rung handed out and `swapped_last` the ones that went to rung 2: if `swapped_last` is a large share of `swap`, the ladder is too short and the next rung has to be invented (or those clips dropped).

`VOICE_SWAP=0` restores the v1 behaviour (no swap at all, straight to `exhausted`). The ladder is covered by `tests/test_voice_swap.py` (the full alternation, every rung in order, the document rule from both sides, what never swaps, and the driver + sync chain) and `tests/test_voice_assignment.py` section 7 (the draw itself: never the current voice, deterministic, spread within 20 pct over the roster).

### After a scoring-rule change: `RESCORE=1`

A clip that already carries a transcript is normally judged on its **stored** CER, which
was computed by whatever scoring rules were in force at the time. Change the scoring
normalization (fold ligatures, fold accents, ...) and those stored numbers go stale, so a
clip that is bad under the new rules can sit there looking fine and never be queued.

`RESCORE=1` re-derives every score from the transcript already on the row, then judges it
under the current rules:

```bash
sudo -E RESCORE=1 MODE=stt CATEGORY="dictionary drugs" ./driver.sh
```

It costs **no STT call**: the pass reuses stored transcripts (only never-transcribed clips
go to the server), so this is a file read plus a CER computation per row. Clips whose audio
file is missing are left alone, so a good stored transcript is never replaced by nothing.

It is a one-time migration, and the driver treats it as one: only the **first `stt` pass of
each dataset** gets the flag. Repeating it would recompute already-current scores, and a
pass that re-examines every row always reports work done, which would stop a single-mode
`MODE=stt` loop from ever reaching its "ran dry" exit 11. `MODE=tts` runs no `stt` pass, so
`RESCORE` does not apply there and the driver says so.

Every row the pass reports carries a marker saying which half it took, so a run that is
quietly transcribing when you expected it to reuse stored text is visible at a glance:

```
000123_0001_pneumopathie  RESCORED cer=0.041                     <- stored transcript, no call
000124_0000_bronchiolite           cer=0.062 stt=1.8s rtf=0.11   <- transcribed now
```

and the end-of-run summary counts both (`N rescored from stored transcripts (no STT call),
M transcribed`), plus how many of the rescored rows now read bad under the current rules.

**It also re-judges the regeneration queue**, which is the half that is easy to forget: a
row already flagged `pending_tts` carries the `orig_cer` a candidate will later have to
beat, and a normal `stt` pass hands those rows straight to the `tts` pass without looking
at them, so nothing else ever revisits them. Under a rescore they are re-derived from the
stored transcript: one that now reads clean drops its marker and leaves the queue (no
regeneration spent on a clip that was never broken), one that stays bad keeps it with a
refreshed score, so the comparison a candidate has to win is against today's number and not
a stale inflated one that any draw would beat. The summary reports it:

```
full.jsonl: regeneration queue re-judged, 812 clip(s) left it (they read clean now, no
regeneration needed), 305 stayed with a refreshed score
```

Rows already **resolved** (`improved` / `exhausted`) are the exception: they keep the score
they were resolved with, since nothing will regenerate them again. Re-deriving the
`improved` ones means calling the script directly with `--force` (the driver does not
expose it as an env var); for the `exhausted` ones, see the next section.

### Giving the clips that gave up another go: `RETRY_EXHAUSTED=1`

`exhausted` means the clip was flagged bad and **no draw was good enough**, so its ORIGINAL
audio is what ships. Those are the known-weak clips of the release, and the marker is
final: nothing revisits them, which also freezes their score under the rules of the run
that gave up. A scoring change therefore cannot reach the rows most likely to benefit from
it, and neither can a new seed or `cfg_alpha` ladder.

`RETRY_EXHAUSTED=1` reopens exactly those rows, as if they carried no marker:

```bash
# Re-judge them offline first (no GPU, no server call), then redraw for what is left.
sudo -E RESCORE_ONLY=1 RETRY_EXHAUSTED=1 ./driver.sh
sudo -E ./driver.sh
```

`START_SEED` used to be mandatory here and no longer is. Draw N of a clip's first round is
seed `START_SEED + N`, and the clip's marker now carries how many rungs of that ladder it
has burned, so a second round draws 48..52 where the first drew 43..47. Without it a retry
would ask the server for the exact draws it already rejected (same seed, same `cfg_alpha`,
same audio), which is worse than wasteful once a clip has had a draw promoted: the redraw
would reproduce the audio sitting on its disk, and the byte-identical guard reads that as a
dead seed and aborts the whole run (exit 2).

The first pass re-derives each one's score from its stored transcript. A clip that now
reads clean **loses its marker** (nothing left to fix, no regeneration spent), and one that
still reads bad goes back to `pending_tts` with a refreshed `orig_cer`, which is the number
a new candidate then has to beat. The second command is a plain alternating run: it drafts
for that queue and judges the results, ending in `improved` or a fresh `exhausted`.

A reopened clip resumes the voice ladder where it stopped: both markers travel with the row, so the retry draws with the voice of the rung it had reached and climbs from there rather than starting the ladder over. One that was exhausted before `VOICE_SWAP` existed carries no marker, so it draws with its original voice and walks the whole ladder like any other clip.

It is deliberately narrower than `--force`: the clips that passed and the ones already
improved are left alone, so a retry costs the weak tail (173 clips after the 0.08 / 0.12
sweep) instead of re-reading the corpus. Like `RESCORE` it rides the **first `stt` pass of
each dataset only**, and here that is not just about saving work: the retry ends by writing
a fresh `exhausted` marker for whatever still fails, so a flag left on for every pass would
reopen its own output and the alternating loop would never converge.

### Giving the wins that still read bad another go: `RETRY_IMPROVED=1`

A draw is promoted either because it clears every gate or because it merely beats the
original by `--min-improvement`, and the second kind leaves **better audio that is still
bad** on disk. The pass sends those back to the tts queue itself now (`REQUEUE` in the log,
`rq` in `--report`), judged from then on against the draw that just won rather than against
the original, whose audio is gone. Runs built before that existed marked them `improved`
and nothing reopens an improved row, so round 1 of the v2 corpus closed **1,387** clips
over a gate (13.5% of its 10,356 promotions).

`RETRY_IMPROVED=1` is the migration for exactly those rows:

```bash
sudo -E RETRY_IMPROVED=1 ./driver.sh
```

Same one-shot shape as `RETRY_EXHAUSTED` (first stt pass of each dataset only, for the same
self-reopening reason), and it is safe to pass over a corpus where most wins are clean: a
reopened win that reads clean **keeps its marker** and stays `improved`. That is the one
place where it differs from a reopened `exhausted` clip, which loses its marker instead:
the win's marker is the only row-level record of which draw is on disk (its seed, speed and
`cfg_alpha`), so dropping it would erase the win from the state file and from every tally
built on it.

### Rescoring on its own first: `RESCORE_ONLY=1`

`RESCORE=1` mixes the two: it rescores what has a transcript and transcribes what does not,
in one long pass. To separate them, and see the damage a rule change did before spending
any GPU time on it:

```bash
sudo -E RESCORE_ONLY=1 CATEGORY="dictionary drugs" ./driver.sh
```

That pass reads stored transcripts, re-derives their scores and stops. It flags nothing:
clips that now read bad are reported (`N clip(s) now read bad under the current rules`) and
left for the next normal `stt` pass, which triple-checks them before queueing.
Never-transcribed clips are left untouched for a normal run too. It makes **no server call
at all**, which is why the driver skips the docker switch entirely: the GPU is left as it
is, so this can run while the TTS server is loaded.

It cannot loop (re-examining the same rows always counts as work done, so the loop would
never end): the driver runs exactly **one** pass per dataset, then moves to the next and
exits. `MODE=tts` is rejected outright, since a `tts` pass scores nothing.

### Several datasets in a row

`INPUT` is a **space-separated list of manifests**, processed one after the other: the
alternating loop runs to convergence on the first, then starts over on the next. The
default chains both manifests the release ships:

| # | Manifest | Output dir | Audio root |
|---|---|---|---|
| 1 | `../99_hf_release/data/NeMO_files/full.jsonl` (dictionary + drugs + PARHAF) | `improved/` | `.../NeMO_files` |
| 2 | `../99_hf_release/data/NeMO_files/PARROT/full.jsonl` (1,549 radiology clips) | `improved_parrot/` | `.../NeMO_files/PARROT` |

PARROT lives in its own manifest because it ships under its own licence (CC BY-NC-SA 4.0,
evaluation-only) as a separate HF subset, so it is never merged into the main corpus.

Both files are named `full.jsonl` and the pass names its output after the manifest stem
(`<stem>.stt.jsonl`), so the first dataset keeps `${OUTPUT}` unchanged (an in-flight run is
untouched) and every later one gets `${OUTPUT}_<parent directory, lowercased>`. The driver
refuses to start if two manifests would still land on the same output file, and each
dataset's audio root defaults to the manifest's own directory (set `AUDIO_ROOT` to pin one
root for all of them). Pass a single path in `INPUT` to work on just one dataset.

Note that `CATEGORY` applies to every dataset in the list: with `CATEGORY=parhaf` the
PARROT pass converges immediately, since none of its rows match.

### When the manifest is rebuilt under a running output

The `.stt.jsonl` is a resume file, not a second source of truth: each pass rebuilds it
from the manifest it is given, so a row the new manifest no longer lists is simply gone
from the next write. Re-chunking a source (which renumbers the clips) therefore drops the
old rows on its own, with no cleanup step.

A row resumes on its audio path **and** its text. Chunk filenames are only unique within
one chunking, so a re-chunk can reuse a filename for different text; the stored
transcription, CER and improvement state then describe audio that no longer exists.
Inheriting them would score the new clip against the old label, and an inherited
`exhausted` status would freeze it for good, so a changed text drops the stored work and
the pass logs how many rows that hit.

Two things do NOT follow the manifest, by design: the `.improved.jsonl` audit keeps every
recorded win (it tracks audio that really was overwritten, whatever the manifest says
today), and a `<clip>.flac.bak` stays next to its clip as the pristine original. Both
survive a re-chunk on purpose.

### Why there is no NOPASSWD sudoers alternative

An earlier revision of this repo shipped a `switch_server.sudoers.example` offering a
"narrow" NOPASSWD entry for `switch_server.sh`, so the driver could run as your user
and still flip servers without a password. It has been removed, because the entry was
not narrow at all: `switch_server.sh` takes the compose file path from its caller, and
`docker compose up` on a caller-chosen compose file is arbitrary code as root (a
compose file can bind-mount `/` and run any image). Granting NOPASSWD on it is
equivalent to granting full passwordless root, which is not what the file said.

The `sudo -E ./driver.sh` flow above is one password prompt for an unattended run that
lasts hours, and it drops back to your user for every Python pass, so use that.

### Re-synthesizing the held-back spelling fixes (`21_reopen_resynth_typos.py`)

Some label typos change how the word sounds (`PERINDIPRIL`, `Cetriaxone`, `KAREGIC`, `empagliflozin`): voxtral read the typo, so the audio says it, and correcting only the label would make the label disagree with the audio. Those fixes are held back in `RESYNTH_TYPOS` (`07_build_accent_spellings.py`) until their clips are redrawn; the clip list and the reason are in `docs/BUILD_LOG.md`, "Label spelling fixes after the v2 model". This is the one round that redraws them, written with Claude Code.

`21_reopen_resynth_typos.py` is step 1. It reuses `17_reopen_hgvs.py`'s chain with the spelling table as the rewrite, so it touches exactly the files 17 does: each stage's `generated_dataset.jsonl` (label and TTS source), the release manifests the rounds read (`99_hf_release/data/NeMO_files/full.jsonl` and `PARROT/full.jsonl`), and the stage-06 state files, where every row whose TTS source changed is marked `pending_tts` with no reference score. It also strips the generic lab names the export drops (`--generic-labs`, on by default) for the one clip whose audio says `Simvastatine Téva`. Its census logs, per held spelling, how many clips it changes and their stems, then the spellings that matched nothing.

The full sequence, from `06_hotfixes/`:

```sh
# 1. census only: prints the per-spelling table and writes the stem list, changes nothing
uv run 21_reopen_resynth_typos.py --out-stems resynth_stems.txt

# 2. rewrite and queue. Not while anything else rewrites the release manifests
#    (99_hf_release/05_normalize_text.py --apply, another driver round): both replace
#    the same full.jsonl and the last writer wins.
uv run 21_reopen_resynth_typos.py --apply --out-stems resynth_stems.txt

# 3. redraw only those clips (inside tmux, the GPU must be free, see below)
sudo -E MODE=alternate STEMS_FROM=resynth_stems.txt REFRESH_MANIFESTS=0 ACOUSTIC_SCAN=0 \
    ./driver.sh 2>&1 | tee resynth_driver.out
```

Env vars step 3 needs, by name (from your shell with `sudo -E`, or from `06_hotfixes/.env`): `CRISPASR_COMPOSE` (path to the CrispASR `docker-compose.yml`), `WDOC_WHISPER_API_KEY` or `CRISPASR_API_KEY` (the ASR judge's key), optionally `WDOC_WHISPER_ENDPOINT` and `STT_MODEL` (default `whisper.cpp/ggml-large-v3-turbo`, the same judge as every other round).

Why these settings:

- `STEMS_FROM=resynth_stems.txt` is what limits the round to these clips: every pass skips any row whose stem is not in the file, so the round costs about 60 clips times `N_IMPROV` (5) draws, minutes of TTS rather than hours. The step log fingerprint includes the file's content, so a new list reruns the pre-steps rather than being skipped as done.
- `REFRESH_MANIFESTS=0`: step 2 already wrote the new text into the release manifests, and a whole-corpus rebuild (about 18 min) buys nothing for 60 clips. `ACOUSTIC_SCAN=0`: the scan is whole-corpus CPU (30 to 100 min) and not restricted by `STEMS_FROM`. `SYNC_RESULTS` stays at its default 1 so the new audio and QC columns reach the release manifests at the end.
- `MODE=alternate` (the default, spelled out): the clips start `pending_tts`, so the first useful pass is TTS, then STT scores the draws against the NEW label and promotes a draw only once it clears every gate. A clip no draw fixes ends `exhausted` and the release drops it, which is the right outcome, since its audio still says the typo.

GPU: the round needs the card to itself, as every driver round does (that is why it alternates, see the top of this file). The STT half runs `CRISPASR_STT_SCALE` (default 6) turbo replicas of about 1.6 GB each behind the load balancer on a 24,564 MiB card (`docs/V2_QC_PHASE_PLAN.md`; a `ggml-large-v3` judge is 4,534 MiB per replica, so 4 replicas there). The TTS half is Voxtral, whose two vLLM stages each reserve a fixed fraction of the card: the CrispASR compose (2026-11-16) sets `VOXTRAL_GPU_MEM_UTIL=0.7` for stage 0, plus the packaged 0.1 for stage 1, so about 0.8 x 24 GB = 19.6 GB plus two CUDA contexts. So nothing else can be on the GPU (a training run included) while the driver runs.

How to tell it converged:

- the log (`resynth_driver.out`) prints `=== ... converged: no improvement work left on ...` for the dataset (the stt pass exited 10), then `=== ... every dataset converged. Done. ===`;
- `improved/driver_steps.log` gains a `done dataset-...` line per dataset and `done sync`;
- `./01_recursive_improvement.py --report` shows the round's ledger; every stem of `resynth_stems.txt` should end `improved` or `exhausted`, none `pending_tts`.

Afterwards, the label side (`docs/BUILD_LOG.md`, same section, steps 4 and 5): merge `RESYNTH_TYPOS` into `HAND_REWRITES`, add the entries to `utils/accent_spellings.json`, drop `tests/test_label_variant_audit.py::test_resynth_typos_held_back`, then export from `99_hf_release/` (`uv run 05_normalize_text.py`, `--apply`, `uv run scripts/build_parquet.py`).

## The second opinion: `10_compare_judges.py`

One judge cannot catch its own blind spots. So the phase scores every clip with
`ggml-large-v3-turbo` (fast enough for 602k clips) and then re-reads only the clips that
pass flagged with non-turbo `ggml-large-v3`, filed under its own key:

```python
row["transcriptions"]["whisper.cpp/ggml-large-v3-turbo"] = {"cer": 0.01, ...}  # T1
row["transcriptions"]["whisper.cpp/ggml-large-v3"]       = {"cer": 0.19, ...}  # T2
```

That second pass is transcribe-only (`N_IMPROV=0`), so it cannot re-flag a clip or strip a
provenance marker; it only adds a reading. `10_compare_judges.py` puts both readings
through the same four gates (it imports `is_bad`, it does not re-spell it) and keeps the
two directions of disagreement apart, because they mean opposite things:

```bash
uv run 10_compare_judges.py --dry-run --samples 20   # just look
uv run 10_compare_judges.py                          # + write the two artefacts
```

A clip the better judge rejects while turbo waved it through is a **false negative that
ships**, which is the entire reason for the pass. The reverse, turbo flagging what the
better judge forgives, is cheap: that clip was redrawn for nothing. The tool writes
`judge_disagreements.jsonl` (both transcripts beside the source text, so the call gets
made by reading rather than by trusting a number) and `second_opinion_flags.txt`, the stem
list of what the second judge rejects, worst CER first, which the repair round takes as
it stands: `sudo -E MODE=alternate STEMS_FROM=second_opinion_flags.txt ./driver.sh`.

## Listening to what the numbers flagged: `03_collect_suspicious.py`

Every gate here is a proxy. A high CER usually means bad audio, but it can also mean
Whisper mishearing a rare drug name, and a clip nobody ever plays can sit in the release
either way. This script gathers the audio behind each reason into one local folder, with
the text beside it:

```bash
uv run 03_collect_suspicious.py --dry-run          # how many, how big, copies nothing
uv run 03_collect_suspicious.py                    # fill ./suspicious_audio/
uv run 03_collect_suspicious.py --reason exhausted --limit 50
uv run 03_collect_suspicious.py --match 'facteur V'   # hear one normalization rule
```

The reasons are the pipeline's own: `worst-cer`, `worst-tail`, `pending-tts`,
`exhausted` (regeneration gave up, so this audio ships as it is: the pile that most
deserves ears), `ceiling` (durations piled at the top of the distribution, the shape a TTS
that stopped on its token limit leaves behind, using the release statistics' own
`utils/nemo_manifest.duration_ceiling`), `stt-error`, and `match` (a plain listening
sample, not a defect). A clip is copied once, under the first reason that claimed it, and
named `<reason>_<cer>_<category>_<clip>.flac` so `ls` sorts by badness. `index.md` lists
each one with what it should say, what the TTS was given, and what Whisper heard;
`index.jsonl` is the same, machine-readable. `--symlink` skips the disk cost (but then the
audio drive must stay mounted to play them).

**Rescore before you listen.** The scores come from the jsonl as it stands, so a stale
list is mostly false positives: after the unit / title / number folding landed, 855 of the
1,152 clips over the 0.15 gate scored clean, and 32 of the 35 tail flags did too. Run
`RESCORE_ONLY=1` first, then collect, or you will spend an evening listening to clips that
were never broken.

## Manual equivalents

```sh
sudo ./switch_server.sh stt        # bring up whisper + LB, wait until healthy
uv run 01_recursive_improvement.py --mode stt  ...   # score + promote; exits 10 when done
sudo ./switch_server.sh tts        # bring up Voxtral, wait until healthy
uv run 01_recursive_improvement.py --mode tts  ...   # generate candidates
```

## TODOs to fill in

- `CRISPASR_COMPOSE`: absolute path to your CrispASR `docker-compose.yml` (env or `.env`).
- `.env`: gitignore it (it holds `WDOC_WHISPER_API_KEY`).
- `CRISPASR_STT_SCALE` (optional, default 6) if you want a different whisper replica count.
- `GPU_POWER_LIMIT` (optional, default 350) watts for `nvidia-smi -pl` at start.
