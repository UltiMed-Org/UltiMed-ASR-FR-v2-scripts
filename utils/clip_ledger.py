"""One line per clip: everything every QC stage recorded about the audio on disk NOW.

Why this exists. Each QC stage writes its own append-only sidecar beside the audio
(`qc_acoustic_sidecar.jsonl`, `qc_stt_sidecar.jsonl`, `qc_parakeet_ultra.jsonl`, ...),
stage 06 keeps its judges' readings in `06_hotfixes/improved*/full.stt.jsonl`, and the
release keeps its drop ledger and ASR flags in `99_hf_release/`. Every one of those
answers a different question about the same clip, several hold more than one row per
clip (one per take), and only some rows still describe the take on disk. Reading them
one by one is how two clips the acoustic scan had flagged `looped: true` (and the
parakeet sweep `over_gate: true`) were shipped and then handed to a human to listen to
(2026-10-28): the keep/redraw decision only ever read the whisper verdict, and nothing
put the other verdicts next to it.

The ledger is a DERIVED view, rebuilt from scratch by `06_hotfixes/15_build_clip_ledger.py`.
The producers keep appending to their own sidecars (resumable, append-only, unchanged),
and anything that has to decide about a clip reads this one file instead. It is
`clip_ledger.jsonl` in each category directory, one JSON object per clip, sorted by stem:

    {"stem": "005258_0000_cancero_adulte_00760_t0_c5",
     "clip": "PARHAF/005258_0000_cancero_adulte_00760_t0_c5.flac",   # nemo_manifest.clip_key
     "text": "...",                                  # the label stage 06 scores against
     "audio": {"exists": true, "mtime": 1792521050.8, "duration": 41.2},
     "release": {"splits": ["train"], "dropped": null, "asr_flag": null},
     "generation": {"voice": "h_frf_itm", "cfg_alpha": 1.3, "cfg_attempts": [...],
                    "code_qc": null},
     "repair": {"status": null, "attempts": null, "history": []},
     "judges": {"turbo": {"text": "...", "cer": 0.0595, "tail": 0.0616, "window": 0.1315,
                          "bad_run": 3, "repeat": null, "gates": []},
                "large_v3": {...}, "parakeet": {..., "repeat": 3, "gates": ["window"]}},
     "qc": {"acoustic": {"similarity": 0.8585, "looped": true, ..., "stale_takes": 0},
            "live_stt": {"cer": 0.115, "over_gate": true, "defect": "loop", ...},
            "parakeet_ultra": {...}},
     "text_defects": {"number": "75 -> 65"},         # 04_flag_asr_defects.defects
     "flags": ["acoustic:loop", "live_stt:over_gate", "parakeet:window",
               "parakeet_ultra:over_gate", "text:number"]}

`flags` is the point: every detector that fired on the CURRENT take, as flat strings, so
a consumer is a set test (`"acoustic:loop" in row["flags"]`) and a census is a Counter.

Rules, all in one place:

* **Freshness.** A sidecar row only counts for the take it heard. `clip_freshness.is_stale`
  decides, from the row's `ts` (STT sidecars, generation records) or `mtime` (the
  acoustic scan), falling back to the sidecar file's own mtime for legacy rows that carry
  neither, exactly as `05_generate_audio/04_qc_acoustic_loops.py` does. Generation rows
  are written ~0.5 s before their clip (measured 2026-10-28: p99 1.0 s on code_qc), so
  the same 2 s slack fits them. The latest fresh row per source wins; older takes are
  only counted (`stale_takes`), so "this clip was looked at before and redrawn" stays
  visible without carrying its outdated numbers.
* **Stage-06 overlap.** The PARROT clips sit in both state files under two relative
  spellings. As in `99_hf_release/03_sync_hotfix_results.py::merge_hotfix`, the later
  source wins per judge reading when it has one, and a present repair marker beats an
  absent one. When both carry a turbo transcript and they differ, `state_conflict` is
  flagged rather than silently resolved.
* **Gates.** Judge gates are computed by `10_compare_judges.which_gates` over
  `01_recursive_improvement.scores_from_reading`, at stage 06's default thresholds
  (`01_compute_stt.DEFAULT_*_GATE`) plus the parakeet loop gate
  `14_reopen_loop_clips.DEFAULT_REPEAT_GATE`. Nothing here re-spells a threshold.
* **Text rules.** `text_defects` holds every `06_hotfixes/04_flag_asr_defects.py` rule
  (preamble, repeated, number, inserted, skipped) that fires on the stage-06 parakeet
  reading confirmed by the turbo one, as `text:<reason>` flags. Those are the defects
  the CER gates are blind to by construction (a wrong number is two characters, a repeat
  adds no missing ones), so a redraw queue must see them next to the gates.
* **Unknown sidecars refuse.** A `*.jsonl` in a category directory that this module does
  not know is an error, not a skip: a new QC stage whose verdicts the ledger silently
  ignored is the exact failure this file exists to stop. Teach `SOURCES` about it.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from clip_freshness import is_stale, parse_row_ts
from nemo_manifest import clip_key, read_jsonl, resolve_audio

LEDGER_NAME = "clip_ledger.jsonl"

#: Per-take QC verdict sidecars: filename -> ledger key under `qc`.
SOURCES = {
    "qc_acoustic_sidecar.jsonl": "acoustic",
    "qc_acoustic_confirm.jsonl": "acoustic_confirm",
    "qc_stt_sidecar.jsonl": "live_stt",
    "qc_gpu_probe.jsonl": "gpu_probe",
    "qc_third_judge.jsonl": "third_judge",
    "qc_parakeet_ultra.jsonl": "parakeet_ultra",
    "qc_parakeet_refill.jsonl": "parakeet_refill",
}
#: Generation records: every attempt is kept, not just the latest.
CFG_SIDECAR = "cfg_alpha_sidecar.jsonl"
CODE_SIDECAR = "code_qc_sidecar.jsonl"
KNOWN_JSONL = frozenset(SOURCES) | {CFG_SIDECAR, CODE_SIDECAR, LEDGER_NAME}

#: Fields dropped from sidecar rows: identity (already the ledger's key), bulky payloads
#: nothing decides on, and `reference`, replaced by `reference_differs` (see below).
_DROP = frozenset({"stem", "stage", "reference", "semantic"})

#: Stage-06 `transcriptions` keys -> ledger judge names.
JUDGES = {
    "whisper.cpp/ggml-large-v3-turbo": "turbo",
    "whisper.cpp/ggml-large-v3": "large_v3",
    "onnx-asr/parakeet-tdt-0.6b-v3-ultra": "parakeet",
    "whisper-1": "whisper_1",
}


def check_known(cat_dir: Path) -> list[str]:
    """The `*.jsonl` files in `cat_dir` this module does not know how to read."""
    return sorted(p.name for p in cat_dir.glob("*.jsonl") if p.name not in KNOWN_JSONL)


def _row_ts(row: dict, legacy: float) -> float:
    """When the row was written: its own stamp, else the sidecar file's mtime."""
    for key in ("ts", "mtime"):
        stamp = parse_row_ts(row.get(key))
        if stamp is not None:
            return stamp
    return legacy


def _iter_rows(path: Path):
    """Sidecar rows in file order; a torn line (killed mid-write) is skipped."""
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("stem"):
                yield row


def read_verdicts(path: Path, audio_dir: Path) -> dict[str, dict]:
    """stem -> latest row that still describes the clip on disk, plus `stale_takes`.

    A stem whose every row is stale maps to `{"stale_takes": n}` alone: it WAS looked at,
    but nothing it said is about the audio there now.

    >>> import tempfile
    >>> d = Path(tempfile.mkdtemp()); clip = d / "a.flac"; _ = clip.write_bytes(b"")
    >>> os.utime(clip, (1000.0, 1000.0))
    >>> side = d / "qc.jsonl"
    >>> _ = side.write_text('{"stem": "a", "ts": 500.0, "cer": 0.9}\\n'
    ...                     '{"stem": "a", "ts": 1001.0, "cer": 0.1}\\n')
    >>> read_verdicts(side, d)["a"]
    {'ts': 1001.0, 'cer': 0.1, 'stale_takes': 1}
    """
    legacy = path.stat().st_mtime
    fresh: dict[str, dict] = {}
    stale: Counter = Counter()
    for row in _iter_rows(path):
        stem = row["stem"]
        if is_stale(audio_dir / f"{stem}.flac", _row_ts(row, legacy)):
            stale[stem] += 1
        else:
            fresh[stem] = row
    out: dict[str, dict] = {}
    for stem in set(fresh) | set(stale):
        rec = {k: v for k, v in fresh.get(stem, {}).items() if k not in _DROP}
        if stem in fresh and "reference" in fresh[stem]:
            rec["_reference"] = fresh[stem]["reference"]
        rec["stale_takes"] = stale[stem]
        out[stem] = rec
    return out


def read_generation(cat_dir: Path) -> dict[str, dict]:
    """stem -> generation record: every cfg attempt, and the last code-QC verdict
    that still describes the clip."""
    out: dict[str, dict] = {}
    cfg = cat_dir / CFG_SIDECAR
    if cfg.is_file():
        for row in _iter_rows(cfg):
            out.setdefault(row["stem"], {"cfg_attempts": []})["cfg_attempts"].append(
                {k: row.get(k) for k in ("cfg_alpha", "voice", "attempt", "attempts",
                                         "reason", "ts")})
    code = cat_dir / CODE_SIDECAR
    if code.is_file():
        for stem, rec in read_verdicts(code, cat_dir).items():
            out.setdefault(stem, {"cfg_attempts": []})["code_qc"] = rec
    return out


def load_state(states: list[tuple[Path, Path]]) -> dict[str, dict]:
    """clip key -> {text, duration, voice, cfg_alpha, improvement, readings, conflict}.

    `states` is `(state file, directory its relative audio paths resolve against)`, in
    `HOTFIX_SOURCES` order; see the module docstring for the overlap rule.
    """
    out: dict[str, dict] = {}
    for path, base in states:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            key = clip_key(resolve_audio(base, row["audio_filepath"]))
            readings = {JUDGES.get(k, k): v for k, v in (row.get("transcriptions") or {}).items()
                        if isinstance(v, dict)}
            prior = out.get(key)
            if prior is None:
                out[key] = {"text": row.get("text"), "source": row.get("asr_training_source"),
                            "duration": row.get("duration"),
                            "voice": row.get("voice"), "cfg_alpha": row.get("cfg_alpha"),
                            "improvement": row.get("improvement"), "readings": readings,
                            "conflict": False}
                continue
            a = (prior["readings"].get("turbo") or {}).get("text")
            b = (readings.get("turbo") or {}).get("text")
            prior["conflict"] |= bool(a and b and a.strip() != b.strip())
            prior["readings"].update(readings)
            for field, src in (("text", "text"), ("source", "asr_training_source"),
                               ("duration", "duration"), ("voice", "voice"),
                               ("cfg_alpha", "cfg_alpha")):
                if row.get(src) is not None:
                    prior[field] = row[src]
            if row.get("improvement"):
                prior["improvement"] = row["improvement"]
    return out


def load_repairs(paths: list[Path]) -> dict[str, list[dict]]:
    """audio stem -> every repair stage 06 applied (`full.improved.jsonl` rows), oldest first."""
    keep = ("ts_iso", "orig_cer", "new_cer", "orig_window", "new_window", "orig_duration",
            "new_duration", "attempts", "cfg_alpha", "seed", "voice")
    out: dict[str, list[dict]] = {}
    for path in paths:
        if path.is_file():
            for row in read_jsonl(path):
                stem = Path(row["audio_filepath"]).stem
                out.setdefault(stem, []).append({k: row.get(k) for k in keep})
    for hist in out.values():
        hist.sort(key=lambda h: h.get("ts_iso") or "")
    return out


def load_release(nemo_dir: Path, drop_ledger: Path, asr_flags: Path, still_dropped) -> dict:
    """clip key -> {splits, dropped, asr_flag}. `still_dropped(entry, audio)` is
    `04_drop_bad_rows.still_dropped`, passed in so the one rule stays where it lives."""
    out: dict[str, dict] = {}

    def slot(key: str) -> dict:
        return out.setdefault(key, {"splits": [], "dropped": None, "asr_flag": None})

    for split in ("train", "val", "test"):
        path = nemo_dir / f"{split}.jsonl"
        if path.is_file():
            for row in read_jsonl(path):
                slot(clip_key(resolve_audio(nemo_dir, row["audio_filepath"])))["splits"].append(split)
    data_dir = nemo_dir.parent
    if drop_ledger.is_file():
        for row in read_jsonl(drop_ledger):
            audio = str(data_dir / row["clip"])
            slot(row["clip"])["dropped"] = {"reason": row["reason"],
                                            "applies": bool(still_dropped(row, audio))}
    if asr_flags.is_file():
        for row in read_jsonl(asr_flags):
            slot(row["clip"])["asr_flag"] = {"reason": row["reason"], "words": row.get("words")}
    return out


def judge_block(readings: dict, scores_from_reading, which_gates, gates) -> dict:
    """The stage-06 readings, trimmed to what a decision needs, each with its gates."""
    out = {}
    for name, r in readings.items():
        if r.get("error") or not (r.get("text") or "").strip():
            out[name] = {"text": None, "error": r.get("error")}
            continue
        out[name] = {"text": r["text"].strip(), "cer": r.get("cer"), "tail": r.get("cer_tail"),
                     "window": r.get("cer_window"), "bad_run": r.get("bad_run_chars"),
                     "repeat": r.get("repeat_excess"),
                     "gates": which_gates(scores_from_reading(r), gates)}
    return out


def flags_of(rec: dict) -> list[str]:
    """Every detector that fired on the clip's current take, plus its release fate."""
    flags = set()
    audio, rel = rec["audio"], rec["release"]
    if not audio["exists"]:
        flags.add("missing_audio")
    if rec["text"] is None:
        flags.add("no_state")
    if rec.get("_conflict"):
        flags.add("state_conflict")
    if rel["dropped"]:
        flags.add(f"dropped:{rel['dropped']['reason']}" if rel["dropped"]["applies"]
                  else "dropped_stale")
    if rel["asr_flag"]:
        flags.add(f"asr_flag:{rel['asr_flag']['reason']}")
    if audio["exists"] and not rel["splits"] and not rel["dropped"]:
        flags.add("unreleased")
    for name, j in rec["judges"].items():
        flags.update(f"{name}:{g}" for g in j.get("gates") or ())
    flags.update(f"text:{reason}" for reason in rec.get("text_defects") or ())
    for src, q in rec["qc"].items():
        if q.get("looped"):
            flags.add(f"{src}:loop")
        if q.get("over_gate"):
            flags.add(f"{src}:over_gate")
        if q.get("reference_differs"):
            flags.add(f"{src}:reference_differs")
    code = (rec["generation"] or {}).get("code_qc") or {}
    if code.get("verdicts"):
        flags.add("code_qc:" + ",".join(sorted(map(str, code["verdicts"]))))
    return sorted(flags)


def text_block(label: str | None, judges: dict, defects) -> dict[str, str]:
    """``{reason: offending words}`` for every text rule `defects` fires, on parakeet's
    reading confirmed by turbo's (the pairing `04_flag_asr_defects.py` was measured on).

    >>> fake = lambda label, hyp, stt: iter([("number", "75 -> 65")] if stt else [])
    >>> text_block("75 ans", {"parakeet": {"text": "65 ans"}, "turbo": {"text": "65 ans"}}, fake)
    {'number': '75 -> 65'}
    >>> text_block("75 ans", {"turbo": {"text": "65 ans"}}, fake)
    {}
    """
    hyp = (judges.get("parakeet") or {}).get("text")
    if not (label and hyp):
        return {}
    return dict(defects(label, hyp, (judges.get("turbo") or {}).get("text")))


def build_category(cat_dir: Path, state: dict, repairs: dict, release: dict,
                   judge_fn, normalize=str.strip, text_fn=None) -> list[dict]:
    """Every clip of one category directory, as ledger records sorted by stem.

    The clip set is the union of what is on disk, in stage 06's state, in the release and
    in any sidecar, so an orphan flac, a manifest row whose audio vanished and a sidecar
    row about a deleted clip all surface instead of falling between two files.

    `normalize` is what two references are compared through before `reference_differs`
    fires: stage 06's `normalize_for_scoring` in production, because the label was
    respelled after several QC passes (`œ` -> `oe`, `pourcent` -> `%`, accent fixes: 97%
    of the raw mismatches on 2026-10-28) and a respelling does not move a verdict.
    """
    cat = cat_dir.name
    qc = {SOURCES[p.name]: read_verdicts(p, cat_dir)
          for p in sorted(cat_dir.glob("*.jsonl")) if p.name in SOURCES}
    gen = read_generation(cat_dir)
    on_disk = {p.name[:-5] for p in os.scandir(cat_dir) if p.name.endswith(".flac")}
    prefix = f"{cat}/"
    stems = on_disk | set(gen) | {s for side in qc.values() for s in side}
    stems |= {k[len(prefix):-5] for k in (*state, *release) if k.startswith(prefix)}
    rows = []
    for stem in sorted(stems):
        key = f"{prefix}{stem}.flac"
        st = state.get(key) or {}
        text = st.get("text")
        path = cat_dir / f"{stem}.flac"
        exists = stem in on_disk
        qc_rec = {}
        for src, side in qc.items():
            if stem not in side:
                continue
            q = dict(side[stem])
            ref = q.pop("_reference", None)
            if ref is not None and text is not None:
                # QC stages score against `text` or `asr_training_source` (the PARROT
                # sweep used the latter); either is current. Almost every sidecar
                # reference IS the current label byte for byte, and normalizing costs a
                # few hundred regexes, so only the ~3% that differ pay for it.
                current = [t.strip() for t in (text, st.get("source")) if t]
                q["reference_differs"] = ref.strip() not in current and \
                    normalize(ref) not in {normalize(t) for t in current}
            qc_rec[src] = q
        imp = st.get("improvement") or {}
        g = gen.get(stem) or {"cfg_attempts": []}
        rec = {
            "stem": stem, "clip": key, "text": text,
            "audio": {"exists": exists, "mtime": path.stat().st_mtime if exists else None,
                      "duration": st.get("duration")},
            "release": release.get(key) or {"splits": [], "dropped": None, "asr_flag": None},
            "generation": {"voice": st.get("voice"), "cfg_alpha": st.get("cfg_alpha"),
                           "cfg_attempts": g["cfg_attempts"], "code_qc": g.get("code_qc")},
            "repair": {"status": imp.get("status"), "attempts": imp.get("attempts"),
                       "history": repairs.get(stem, [])},
            "judges": judge_fn(st.get("readings") or {}),
            "qc": qc_rec,
            "_conflict": st.get("conflict", False),
        }
        rec["text_defects"] = text_fn(text, rec["judges"]) if text_fn else {}
        rec["flags"] = flags_of(rec)
        del rec["_conflict"]
        rows.append(rec)
    return rows


def write_ledger(cat_dir: Path, rows: list[dict]) -> Path:
    """Atomically replace `cat_dir/clip_ledger.jsonl` (a reader never sees half of it)."""
    out = cat_dir / LEDGER_NAME
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return out


def load_ledger(cat_dir: Path) -> dict[str, dict]:
    """stem -> ledger record, for consumers. Raises if the ledger was never built."""
    return {r["stem"]: r for r in read_jsonl(cat_dir / LEDGER_NAME)}
