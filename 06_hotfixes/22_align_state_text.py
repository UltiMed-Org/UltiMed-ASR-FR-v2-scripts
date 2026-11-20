#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Copy a release manifest's label ``text`` into a stage-06 state file after a label fix.

Usage (dry run, then the same line with ``--apply``):

    uv run 06_hotfixes/22_align_state_text.py 99_hf_release/data/NeMO_files/full.jsonl \
        06_hotfixes/improved/full.stt.jsonl [--apply] [--ignore-source]

Step 7 of the label-fix procedure (``docs/BUILD_LOG.md``, 2026-11-20). ``01_compute_stt.py``
keys a state row on its audio path AND its ``text``, so a row whose label was respelled by
``99_hf_release/05_normalize_text.py --apply`` no longer matches, and the next driver round
nulls its ``cer`` / ``stt_transcript`` and re-runs STT on it. This counts (and with
``--apply`` rewrites) the state rows whose ``text`` lags the manifest. Only rows with the
same audio path AND the same ``asr_training_source`` are aligned: a different source means
the clip was redrawn, which is not a label fix (``--ignore-source`` overrides that).

Do NOT run ``05_normalize_text.py`` on the state file instead: it would also rewrite
``asr_training_source`` (the CER reference) in about 13.6k state rows.

Written with Claude Code (2026-11-20; promoted from a scratch script).
"""
import json, os, sys
manifest, state = sys.argv[1], sys.argv[2]
apply = "--apply" in sys.argv
man = {}
for l in open(manifest, encoding="utf-8"):
    if l.strip():
        r = json.loads(l); man[r["audio_filepath"]] = (r.get("text"), r.get("asr_training_source"))
n = diff = alignable = src_diff = missing = 0
ex = []
tmp = state + ".aligntmp"
out = open(tmp, "w", encoding="utf-8") if apply else None
for l in open(state, encoding="utf-8"):
    if not l.strip():
        continue
    n += 1
    r = json.loads(l)
    m = man.get(r["audio_filepath"])
    changed = False
    if m is None:
        missing += 1
    elif m[0] != r.get("text"):
        diff += 1
        if m[1] == r.get("asr_training_source") or "--ignore-source" in sys.argv:
            alignable += 1
            if len(ex) < 3: ex.append((os.path.basename(r["audio_filepath"]), r.get("text")[:70], m[0][:70]))
            if apply:
                r["text"] = m[0]; changed = True
        else:
            src_diff += 1
    if out:
        out.write(json.dumps(r, ensure_ascii=False) + "\n" if changed else l)
if out:
    out.close(); os.replace(tmp, state)
print(f"state rows {n} manifest rows {len(man)} missing_in_manifest {missing} text_diff {diff} alignable {alignable} source_also_differs {src_diff} applied {apply}")
for e in ex: print(e)
