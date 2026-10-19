#!/usr/bin/env python
"""`06_hotfixes/driver.sh` must keep alternating when one half has nothing to do.

The alternating loop leads with the `stt` pass, and that pass legitimately reports
"I have run dry" (exit 11) whenever a previous run stopped right after flagging clips
`pending_tts`: there is nothing left to score until a `tts` pass has drafted
candidates. Treating that as a stop condition (as the driver used to) skips the whole
dataset and the flagged clips are never regenerated, which is exactly the state
`improved_parrot/full.stt.jsonl` was left in (4 pending_tts, 0 pending_stt).

This runs the REAL driver.sh with the passes stubbed out: a fake `uv` records the
`--mode` it was called with and exits with the next code from a plan file, so a whole
alternation is a plan like `11, 0, 10`. driver.sh insists on being root (it flips
docker), so the test re-execs itself under `unshare -r`, which maps the current user to
uid 0 without granting anything; `runuser` and `nvidia-smi` are stubbed on PATH. If
unprivileged user namespaces are disabled, the test SKIPS loudly rather than lying.

    python tests/test_driver_alternate_mode.py

This file was written with Claude Code.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "06_hotfixes" / "driver.sh"

RUNUSER_STUB = """#!/usr/bin/env bash
# Stand-in for `runuser -u USER -- env ... uv run ...`: drop everything up to the
# `--` separator and exec the rest, so the stubbed uv runs in this same shell.
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift
exec "$@"
"""

NVIDIA_SMI_STUB = "#!/usr/bin/env bash\nexit 0\n"

UV_STUB = """#!/usr/bin/env bash
# Stand-in for the improvement pass: record the --mode it was asked for and the
# scheduling priority it inherited, then exit with the next code from the plan. Running
# past the end of the plan means the driver looped more times than the test expected, so
# answer 99 (an abort) instead of spinning forever.
# The driver calls uv for the pre- and post-steps too (the manifest rebuild, the
# acoustic scan, the result sync), which are other scripts entirely. Record the script
# name and its arguments, then succeed without touching the pass log or the exit plan.
args="$*"
script=""
for arg in "$@"; do
  if [[ "${arg}" == *.py ]]; then script="${arg##*/}"; break; fi
done
if [[ "${script}" != 01_recursive_improvement.py ]]; then
  echo "${script}|${args}" >> "%(steps)s"
  exit 0
fi
mode=""
retry=0
report=0
while [[ $# -gt 0 ]]; do
  [[ "$1" == --mode ]] && mode="$2"
  [[ "$1" == --retry-exhausted ]] && retry=1
  [[ "$1" == --report ]] && report=1
  shift
done
# The driver also calls the script once per dataset with --report (print the per-round
# ledger and exit), which is not a pass: recording it would make every test see one
# extra "pass" with no mode, and consume a line of the exit-code plan.
if (( report )); then
  exit 0
fi
echo "${mode}|$(nice)|${retry}" >> "%(log)s"
n=$(wc -l < "%(log)s")
code=$(sed -n "${n}p" "%(plan)s")
exit "${code:-99}"
"""

PGREP_STUB = """#!/usr/bin/env bash
# Stand-in for `pgrep -f PATTERN`: answer from a per-pattern plan file, one call per
# line, popped as it is read. An exhausted plan therefore means "nothing matches" for
# every later call, which is what lets one stub serve both the wait gate (a generator
# that goes away) and the kill (a QC that dies on SIGTERM).
key=other
case "$*" in
  *generate*) key=gen ;;
  *qc_*)      key=qc ;;
esac
plan="%(plans)s/plan.${key}"
line=""
if [[ -s "${plan}" ]]; then
  line="$(head -n 1 "${plan}")"
  tail -n +2 "${plan}" > "${plan}.rest" && mv "${plan}.rest" "${plan}"
fi
echo "pgrep|${key}|${line}" >> "%(events)s"
[[ -z "${line}" ]] && exit 1
printf '%%s\n' ${line}
exit 0
"""

PKILL_STUB = """#!/usr/bin/env bash
# Stand-in for `pkill -SIGNAL -f PATTERN`: record which signal went to which pattern
# and claim success. The pgrep plan above is what decides whether it "worked".
echo "pkill|$1|${*: -1}" >> "%(events)s"
exit 0
"""

SWITCH_STUB = """#!/usr/bin/env bash
echo "switch $1" >> "%(log)s"
exit 0
"""


def _write_exec(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def run_driver(tmp: Path, plan: list[int], env_extra: dict[str, str],
               n_manifests: int = 1,
               record: list | None = None,
               steps: list[str] | None = None,
               gen_plan: list[str] | None = None,
               qc_plan: list[str] | None = None,
               signals: list[str] | None = None,
               events: list[str] | None = None) -> tuple[int, list[str], list[str], str]:
    """Run driver.sh once with the passes stubbed. Returns (exit code, the --mode of
    each pass in order, the server switches in order, combined output). `record`, if
    given, also receives one [mode, niceness, retry-exhausted] entry per pass, for the
    cases that care about which flags a pass was handed. `steps`, if given, receives one
    "<script>|<args>" line per pre- or post-step, in the order they ran.

    `gen_plan` / `qc_plan` drive the stubbed `pgrep -f`: one line per call, each a
    space-separated list of PIDs, "" for "nothing matches". They default to empty, so
    every other test sees a host where stage 05 is already finished and no QC is
    running, which is the shape the gate must wave through untouched. `signals`, if
    given, receives the pgrep/pkill calls. It and `steps` come from ONE log, so the
    order of a kill against a pre-step is observable."""
    work = Path(tempfile.mkdtemp(dir=tmp))
    # Run a COPY of driver.sh: it sources `<its dir>/.env`, and the real one holds the
    # user's STT token and would override the test's env.
    driver = work / "driver.sh"
    shutil.copy(DRIVER, driver)

    pass_log, switch_log = work / "passes.log", work / "switches.log"
    # One log for the pre-/post-steps AND the process signals, so their relative order
    # is a fact the tests can assert on (the QC must stop BEFORE the acoustic scan
    # starts competing with it for the same cores).
    steps_log = work / "steps.log"
    plans = work / "plans"
    plans.mkdir()
    (plans / "plan.gen").write_text("".join(f"{l}\n" for l in (gen_plan or [])),
                                    encoding="utf-8")
    (plans / "plan.qc").write_text("".join(f"{l}\n" for l in (qc_plan or [])),
                                   encoding="utf-8")
    pass_log.touch()
    switch_log.touch()
    steps_log.touch()
    (work / "plan.txt").write_text("".join(f"{c}\n" for c in plan), encoding="utf-8")

    stub_bin = work / "bin"
    stub_bin.mkdir()
    _write_exec(stub_bin / "runuser", RUNUSER_STUB)
    _write_exec(stub_bin / "nvidia-smi", NVIDIA_SMI_STUB)
    _write_exec(stub_bin / "pgrep", PGREP_STUB % {"plans": plans, "events": steps_log})
    _write_exec(stub_bin / "pkill", PKILL_STUB % {"events": steps_log})
    uv = _write_exec(work / "uv-stub", UV_STUB % {"log": pass_log, "plan": work / "plan.txt",
                                                  "steps": steps_log})
    switch = _write_exec(work / "switch-stub", SWITCH_STUB % {"log": switch_log})

    # driver.sh only checks the manifests exist (and names their outputs after their
    # parent dir), so empty files in distinct dirs are enough.
    manifests = []
    for i in range(n_manifests):
        d = work / f"ds{i}"
        d.mkdir()
        (d / "full.jsonl").touch()
        manifests.append(str(d / "full.jsonl"))
    (work / "compose.yml").touch()

    env = dict(os.environ)
    env.update({
        "PATH": f"{stub_bin}:{env['PATH']}",
        "TARGET_USER": getpass.getuser(),
        "UV_BIN": str(uv),
        "SWITCH_SERVER": str(switch),
        "CRISPASR_COMPOSE": str(work / "compose.yml"),
        "WDOC_WHISPER_ENDPOINT": "http://127.0.0.1:1",
        # Keep the gate's own clock out of the suite's runtime. The DEFAULTS (60s poll,
        # 120s settle, 60s grace) are asserted as text below instead, since a test that
        # actually sat through them would take minutes.
        "WAIT_POLL": "1",
        "WAIT_SETTLE": "0",
        "KILL_QC_GRACE": "2",
        "INPUT": " ".join(manifests),
        "OUTPUT": str(work / "out"),
    })
    env.update(env_extra)

    cmd = ["bash", str(driver)]
    if os.geteuid() != 0:
        cmd = ["unshare", "-r", *cmd]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    entries = [e.split("|") for e in pass_log.read_text(encoding="utf-8").split()]
    modes = [e[0] for e in entries]
    switches = switch_log.read_text(encoding="utf-8").split()[1::2]  # "switch stt" pairs
    # Invariant of every case below, not of one of them: this is a multi-day unattended
    # job and no pass may ever compete with an interactive session for the CPU.
    niceness = [e[1] for e in entries]
    assert all(n == "19" for n in niceness), (
        f"every pass must run at nice 19, got {niceness}\n{proc.stdout}{proc.stderr}")
    if record is not None:
        record.extend(entries)
    all_events = steps_log.read_text(encoding="utf-8").splitlines()
    if steps is not None:
        steps.extend(e for e in all_events
                     if not e.startswith("pgrep|") and not e.startswith("pkill|"))
    if signals is not None:
        signals.extend(e for e in all_events
                       if e.startswith("pgrep|") or e.startswith("pkill|"))
    if events is not None:
        events.extend(all_events)
    return proc.returncode, modes, switches, proc.stdout + proc.stderr


def test_alternate_survives_a_dry_stt_start(tmp: Path) -> None:
    """The regression: nothing to score, everything waiting on tts. The stt pass exits
    11 and the driver must run the tts pass anyway, not give up on the dataset."""
    rc, modes, switches, out = run_driver(tmp, [11, 0, 10], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert switches == ["stt", "tts", "stt"], f"servers switched: {switches}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_alternate_survives_a_dry_stt_start: OK")


def test_alternate_stops_when_both_halves_are_dry(tmp: Path) -> None:
    """...but it must not spin: if neither pass can do anything and stt never reported
    convergence, the leftovers are stuck, so stop after the round."""
    rc, modes, _switches, out = run_driver(tmp, [11, 11], {"MODE": "alternate"})
    assert modes == ["stt", "tts"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "both passes ran dry" in out, out
    print("test_alternate_stops_when_both_halves_are_dry: OK")


def test_alternate_tolerates_a_dry_tts_pass(tmp: Path) -> None:
    """A tts pass with no clip to draft for exits 11 too. That is not a failure: the
    stt pass owns convergence, so the loop goes back to it."""
    rc, modes, _switches, out = run_driver(tmp, [0, 11, 10], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_alternate_tolerates_a_dry_tts_pass: OK")


def test_alternate_gives_up_on_a_stuck_dataset(tmp: Path) -> None:
    """The other dead end: the stt pass keeps reporting work done (it re-runs rows it
    can never resolve, e.g. clips whose audio is gone) while the tts pass has nothing
    to draft. Bounded at two such rounds, or the driver swaps servers forever. Note the
    stub exits 99 past the end of the plan, so a loop that did not stop aborts here."""
    rc, modes, _switches, out = run_driver(tmp, [0, 11, 0, 11], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt", "tts"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "found nothing to draft in 2 rounds" in out, out
    print("test_alternate_gives_up_on_a_stuck_dataset: OK")


def test_alternate_still_aborts_on_a_real_failure(tmp: Path) -> None:
    """Only 10 and 11 are outcomes; any other non-zero code still aborts the driver."""
    rc, modes, _switches, out = run_driver(tmp, [7], {"MODE": "alternate"})
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    assert rc == 7, f"driver exited {rc}, expected the pass's 7\n{out}"
    print("test_alternate_still_aborts_on_a_real_failure: OK")


def test_single_mode_still_stops_when_dry(tmp: Path) -> None:
    """Unchanged for MODE=stt: 11 is the stop condition there, since no tts pass is
    coming to clear the rest."""
    rc, modes, _switches, out = run_driver(tmp, [11], {"MODE": "stt"})
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "ran dry" in out, out
    print("test_single_mode_still_stops_when_dry: OK")


def test_dry_start_on_the_second_dataset(tmp: Path) -> None:
    """The live shape: the main corpus converges, then PARROT starts with only
    pending_tts clips. The second dataset must still be worked, not skipped."""
    rc, modes, _switches, out = run_driver(tmp, [10, 11, 0, 10], {"MODE": "alternate"},
                                           n_manifests=2)
    assert modes == ["stt", "stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_dry_start_on_the_second_dataset: OK")


def test_retry_exhausted_is_one_shot_per_dataset(tmp: Path) -> None:
    """RETRY_EXHAUSTED=1 reopens the clips that gave up, and must do so exactly ONCE per
    dataset: the retry ends by writing a fresh `exhausted` marker for whatever still
    fails, so a flag left on for every stt pass would reopen its own output and the
    alternating loop would never converge. Each dataset gets its own reopening, though,
    since each manifest holds its own."""
    entries: list = []
    rc, modes, _switches, out = run_driver(
        tmp, [0, 0, 10, 0, 0, 10], {"MODE": "alternate", "RETRY_EXHAUSTED": "1"},
        n_manifests=2, record=entries)
    assert modes == ["stt", "tts", "stt", "stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    retried = [e[2] for e in entries]
    assert retried == ["1", "0", "0", "1", "0", "0"], (
        "--retry-exhausted should ride the first stt pass of each dataset only, got "
        f"{list(zip(modes, retried))}\n{out}")
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_retry_exhausted_is_one_shot_per_dataset: OK")


def test_retry_exhausted_defaults_off(tmp: Path) -> None:
    """Nothing changes for a plain run: the resolved clips stay resolved."""
    entries: list = []
    rc, _modes, _switches, out = run_driver(tmp, [10], {"MODE": "alternate"}, record=entries)
    assert [e[2] for e in entries] == ["0"], f"unasked-for retry\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_retry_exhausted_defaults_off: OK")


def test_it_waits_for_the_generator_then_stops_the_qc(tmp: Path) -> None:
    """The reason this gate exists: `sudo -E ./driver.sh` is launched while stage 05 is
    still generating, so the driver must do NOTHING until the generator leaves, then
    stop the live CPU QC, and only then start its own pre-steps. Getting the order
    wrong is not cosmetic: the first stt switch stops the TTS container, and the
    acoustic scan wants the cores the QC is holding."""
    signals: list[str] = []
    events: list[str] = []
    rc, modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "stt"},
        gen_plan=["4242 4243", ""], qc_plan=["5150 5151"],
        signals=signals, events=events)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "waiting for stage 05 to finish" in out, out
    assert "stage 05 is done" in out, out
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    termed = [s for s in signals if s.startswith("pkill|-TERM|")]
    assert termed, f"the QC was never SIGTERMed: {signals}"
    assert "qc_cpu_stt" in termed[0], f"SIGTERM went to the wrong pattern: {termed}"
    assert not [s for s in signals if s.startswith("pkill|-KILL|")], (
        f"SIGKILLed a QC that had already stopped: {signals}")
    kill_at = next(i for i, e in enumerate(events) if e.startswith("pkill|-TERM|"))
    scan_at = next(i for i, e in enumerate(events) if e.startswith("04_qc_acoustic_loops"))
    assert kill_at < scan_at, f"the acoustic scan started before the QC stopped: {events}"
    print("test_it_waits_for_the_generator_then_stops_the_qc: OK")


def test_a_qc_that_ignores_sigterm_gets_sigkilled(tmp: Path) -> None:
    """A QC still alive after KILL_QC_GRACE is killed outright. Safe by construction:
    its sidecar is one appended JSON line per clip and resumes by stem, so the worst
    case is losing the clips in flight."""
    signals: list[str] = []
    rc, _modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "stt"},
        qc_plan=["5150", "5150", "5150"], signals=signals)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert [s for s in signals if s.startswith("pkill|-TERM|")], signals
    assert [s for s in signals if s.startswith("pkill|-KILL|")], (
        f"a QC that ignored SIGTERM was left running: {signals}")
    assert "ignored SIGTERM" in out, out
    print("test_a_qc_that_ignores_sigterm_gets_sigkilled: OK")


def test_the_gate_waves_through_a_host_with_nothing_running(tmp: Path) -> None:
    """The common case once generation is over: no generator, no QC, so the phase must
    start at once rather than poll for something that will never appear."""
    signals: list[str] = []
    rc, modes, _switches, out = run_driver(tmp, [10], {"MODE": "stt"}, signals=signals)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    assert "so stage 05 is not running: starting now" in out, out
    assert "nothing to stop" in out, out
    assert not [s for s in signals if s.startswith("pkill|")], (
        f"signalled something on an idle host: {signals}")
    print("test_the_gate_waves_through_a_host_with_nothing_running: OK")


def test_both_halves_of_the_gate_can_be_turned_off(tmp: Path) -> None:
    """An empty pattern disables each half, for the run that must start NOW and the one
    that must leave a QC alone. Empty, not 0: the value IS the pattern."""
    signals: list[str] = []
    rc, _modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "stt", "WAIT_FOR_GENERATOR": "", "KILL_QC": ""},
        gen_plan=["4242"], qc_plan=["5150"], signals=signals)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "not waiting for stage 05" in out, out
    assert not [s for s in signals if s.startswith("pkill|")], (
        f"killed a QC with KILL_QC empty: {signals}")
    assert not [s for s in signals if s.startswith("pgrep|")], (
        f"polled with both patterns empty: {signals}")
    print("test_both_halves_of_the_gate_can_be_turned_off: OK")


def test_wait_timeout_refuses_to_start_on_top_of_a_generator(tmp: Path) -> None:
    """With WAIT_TIMEOUT set, a generator that outlasts it stops the run instead of
    starting the phase anyway. The failure mode it rules out is the expensive one:
    stopping the TTS container from under a multi-day generation run."""
    rc, modes, switches, out = run_driver(
        tmp, [10], {"MODE": "stt", "WAIT_TIMEOUT": "2"},
        gen_plan=["4242", "4242", "4242", "4242"])
    assert rc == 1, f"driver exited {rc}, expected 1\n{out}"
    assert modes == [], f"ran a pass anyway: {modes}\n{out}"
    assert switches == [], f"touched docker anyway: {switches}\n{out}"
    assert "WAIT_TIMEOUT=2s elapsed" in out, out
    print("test_wait_timeout_refuses_to_start_on_top_of_a_generator: OK")


def test_the_gate_defaults_are_the_documented_ones(tmp: Path) -> None:
    """The suite runs with a 1s poll so it stays fast, so the shipped defaults are
    pinned as text: 60s between polls, 120s of confirmed absence, 60s of grace, and the
    two patterns that actually match the v2 generator and the stage 05 QC scripts."""
    driver = DRIVER.read_text(encoding="utf-8")
    for want in ('WAIT_FOR_GENERATOR="${WAIT_FOR_GENERATOR-generate_v2\\.sh'
                 '|01_generate_audio\\.py}"',
                 'WAIT_POLL="${WAIT_POLL:-60}"',
                 'WAIT_SETTLE="${WAIT_SETTLE:-120}"',
                 'KILL_QC="${KILL_QC-03_qc_cpu_stt\\.py|04_qc_acoustic_loops\\.py}"',
                 'KILL_QC_GRACE="${KILL_QC_GRACE:-60}"'):
        assert want in driver, f"gate default changed or moved: {want}"
    # `-` and not `:-`, or an empty pattern would silently fall back to the default and
    # there would be no way to disable either half.
    assert 'WAIT_FOR_GENERATOR:-' not in driver, "an empty WAIT_FOR_GENERATOR cannot disable"
    assert 'KILL_QC:-' not in driver, "an empty KILL_QC cannot disable"
    print("test_the_gate_defaults_are_the_documented_ones: OK")


def test_the_steps_are_on_by_default(tmp: Path) -> None:
    """`sudo -E ./driver.sh` alone is meant to be the whole phase, so the scan and the
    sync run with no env at all. The refresh is the one that cannot: these tests run
    against a copy of the driver in a tmpdir, where there is no 99_hf_release/data to
    derive the split pin from. A DEFAULTED refresh in that position must skip itself and
    let the run continue, not take the run down."""
    steps: list[str] = []
    rc, modes, _switches, out = run_driver(tmp, [10], {"MODE": "alternate"}, steps=steps)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    scripts = [line.split("|", 1)[0] for line in steps]
    assert scripts == ["04_qc_acoustic_loops.py", "03_sync_hotfix_results.py"], \
        f"steps ran: {scripts}\n{out}"
    assert "Skipping the manifest refresh" in out, out
    print("test_the_steps_are_on_by_default: OK")


def test_every_step_can_be_turned_off(tmp: Path) -> None:
    """...and a run that wants only the passes can still have exactly that: the steps
    rebuild manifests and rewrite the release files, which is not always wanted."""
    steps: list[str] = []
    rc, modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "alternate", "REFRESH_MANIFESTS": "0",
                    "ACOUSTIC_SCAN": "no", "SYNC_RESULTS": "0"}, steps=steps)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert modes == ["stt"] and steps == [], f"{modes} {steps}\n{out}"
    print("test_every_step_can_be_turned_off: OK")


def test_an_explicit_refresh_refuses_unpinned_splits(tmp: Path) -> None:
    """PIN_SPLITS_FROM has no safe fallback: an unpinned rebuild redraws train/val/test
    and leaks the old test set into the new train set, silently. Asked for explicitly
    with no pin available, the driver stops BEFORE any pass rather than rebuilding and
    hoping someone notices. (Defaulted, it skips: the test above.)"""
    steps: list[str] = []
    rc, modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "alternate", "REFRESH_MANIFESTS": "1"}, steps=steps)
    assert rc == 1, f"driver exited {rc}, expected a refusal\n{out}"
    assert "PIN_SPLITS_FROM" in out, out
    assert modes == [] and steps == [], f"it did work anyway: {modes} {steps}\n{out}"

    # A path that is not a directory is the same mistake one step later.
    rc, _modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "alternate", "REFRESH_MANIFESTS": "1",
                    "PIN_SPLITS_FROM": str(tmp / "nope")})
    assert rc == 1 and "not a directory" in out, f"exited {rc}\n{out}"
    print("test_an_explicit_refresh_refuses_unpinned_splits: OK")


def test_the_whole_phase_in_one_command(tmp: Path) -> None:
    """The point of the item: pre-steps, then the passes, then the sync, in that order.
    Order is the assertion that matters. The manifests must be rebuilt before anything
    reads a duration, the scan must see the audio the rebuild described, and the sync
    must run last because it reads what the passes wrote."""
    pinned = tmp / "v1_NeMO_files"
    pinned.mkdir(exist_ok=True)
    steps: list[str] = []
    rc, modes, _switches, out = run_driver(
        tmp, [10], {"MODE": "alternate", "REFRESH_MANIFESTS": "1",
                    "PIN_SPLITS_FROM": str(pinned), "ACOUSTIC_SCAN": "1",
                    "ACOUSTIC_WORKERS": "3", "ACOUSTIC_STAGES": "01_dictionnary 04_PARROT",
                    "SYNC_RESULTS": "1", "STT_MODEL": "whisper.cpp/ggml-large-v3"},
        steps=steps)
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    scripts = [line.split("|", 1)[0] for line in steps]
    assert scripts == ["01_build_nemo_manifest.py", "02_combine_nemo_manifests.py",
                       "04_qc_acoustic_loops.py", "03_sync_hotfix_results.py"], \
        f"steps ran: {scripts}\n{out}"
    assert f"--pin-splits-from {pinned}" in steps[0], steps[0]
    assert f"--pin-splits-from {pinned}" in steps[1], steps[1]
    assert "--workers 3" in steps[2], steps[2]
    assert "--stage 01_dictionnary --stage 04_PARROT" in steps[2], steps[2]
    # The sync must read the key the passes WROTE under, not v1's `whisper-1` default,
    # or it matches nothing and writes every QC column empty.
    assert "--apply" in steps[3] and "--stt-key whisper.cpp/ggml-large-v3" in steps[3], steps[3]
    # And the model NAME it writes into the manifest must be that same judge, or the row
    # would credit turbo (the script's own default) with large-v3's reading.
    assert "--stt-model whisper.cpp/ggml-large-v3" in steps[3], steps[3]
    print("test_the_whole_phase_in_one_command: OK")


def main() -> None:
    if shutil.which("unshare") is None and os.geteuid() != 0:
        print("SKIP test_driver_alternate_mode: no `unshare`, cannot fake the root check")
        return
    if os.geteuid() != 0:
        probe = subprocess.run(["unshare", "-r", "true"], capture_output=True)
        if probe.returncode != 0:
            print("SKIP test_driver_alternate_mode: unprivileged user namespaces are "
                  "disabled, cannot fake driver.sh's root check")
            return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_alternate_survives_a_dry_stt_start(tmp)
        test_alternate_stops_when_both_halves_are_dry(tmp)
        test_alternate_tolerates_a_dry_tts_pass(tmp)
        test_alternate_gives_up_on_a_stuck_dataset(tmp)
        test_alternate_still_aborts_on_a_real_failure(tmp)
        test_single_mode_still_stops_when_dry(tmp)
        test_dry_start_on_the_second_dataset(tmp)
        test_retry_exhausted_is_one_shot_per_dataset(tmp)
        test_retry_exhausted_defaults_off(tmp)
        test_the_steps_are_on_by_default(tmp)
        test_every_step_can_be_turned_off(tmp)
        test_an_explicit_refresh_refuses_unpinned_splits(tmp)
        test_the_whole_phase_in_one_command(tmp)
        test_it_waits_for_the_generator_then_stops_the_qc(tmp)
        test_a_qc_that_ignores_sigterm_gets_sigkilled(tmp)
        test_the_gate_waves_through_a_host_with_nothing_running(tmp)
        test_both_halves_of_the_gate_can_be_turned_off(tmp)
        test_wait_timeout_refuses_to_start_on_top_of_a_generator(tmp)
        test_the_gate_defaults_are_the_documented_ones(tmp)
    print("test_driver_alternate_mode: OK")


if __name__ == "__main__":
    sys.exit(main())
