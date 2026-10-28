"""Import a numbered pipeline script (`14_reopen_loop_clips.py`) as a module.

Those filenames start with a digit, so `import` cannot name them; every script that
reuses another one's functions loads it by path instead. This is that loader, once:

    >>> from pathlib import Path
    >>> mod = load_script(Path(__file__).with_name("number_swaps.py"), "swaps_demo")
    >>> mod.agreed_swaps("75 ans", "65 ans", "65 ans")
    [('75', '65')]

The module is registered in `sys.modules` under `name` before it runs, which is what
dataclasses and pickling inside the loaded script need. Pick a `name` unique to the
caller (`recimp_for_reopen`, `sync_for_ledger`): two callers loading the same file under
one name would share, and re-execute, one module object.

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_script(path: Path, name: str):
    """Execute the script at `path` as module `name` and return it."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
