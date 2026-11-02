"""Where stage 06 keeps its state files, and what their audio paths are relative to.

Each state file (``full.stt.jsonl``) was produced from one release manifest, and its
rows' ``audio_filepath`` values are spelled relative to THAT manifest's directory, which
differs per state: the PARROT state writes ``../../PARROT/x.flac`` (relative to
``NeMO_files/PARROT``), the root state ``../PARROT/x.flac`` (relative to ``NeMO_files``).
Joining a state's path onto the wrong base still lands on a real file, through the
``data`` symlink onto the v1 tree's file of the same name, a different take: that is
how 89 PARROT rows got v1 lengths on 2026-11-02. So the pairing lives here, once, and
every script that walks the states imports it instead of carrying its own tuple.

Written with Claude Code.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOTFIXES = REPO / "06_hotfixes"
RELEASE = REPO / "99_hf_release"

#: State file (relative to ``06_hotfixes/``) -> the directory its rows' audio paths
#: resolve against (relative to ``99_hf_release/``). Order matters: 03_sync applies the
#: root state first and lets the per-dataset one complete it (see its overlap rule).
STATE_AUDIO_BASES: dict[str, str] = {
    "improved/full.stt.jsonl": "data/NeMO_files",
    "improved_parrot/full.stt.jsonl": "data/NeMO_files/PARROT",
}

#: The state files, relative to ``06_hotfixes/``, for ``--state`` style defaults.
DEFAULT_STATES: tuple[str, ...] = tuple(STATE_AUDIO_BASES)


def default_state_paths() -> list[Path]:
    """The state files as absolute paths under ``06_hotfixes/``."""
    return [HOTFIXES / rel for rel in DEFAULT_STATES]


def audio_base(state: Path) -> Path | None:
    """The directory ``state``'s audio paths resolve against, or None for a state file
    that is not one of `STATE_AUDIO_BASES` (one given by hand, a test's temp file)."""
    resolved = Path(state).resolve()
    for rel, base in STATE_AUDIO_BASES.items():
        if resolved == (HOTFIXES / rel).resolve():
            return RELEASE / base
    return None
