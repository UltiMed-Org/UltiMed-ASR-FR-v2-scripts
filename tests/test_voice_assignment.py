"""Stdlib tests for utils/voice_assignment: the v2 voice column.

Run: `python tests/test_voice_assignment.py`

Four properties, each of which would otherwise fail silently and produce a `voice`
column that does not describe the audio:

1. **Cross-process determinism.** The generator picks the voice at synthesis time and
   the manifest builder re-derives it later, in another process. Python randomises
   string hashing per process, so a builtin `hash()` here would mislabel every clip.
   The test re-derives in subprocesses under two different `PYTHONHASHSEED` values.
2. **Group cohesion and the key field.** All clips of a group must share a voice, and
   the group field must be the same one `nemo_manifest.detect_grouping` picks
   (`source_id` when present, else `term_index`), or a PARHAF document could be split
   across speakers while still claiming to be one group.
3. **Balance.** Not exact (the assignment is a hash, see the module docstring), but
   it must not collapse: over many groups every voice has to appear, within a loose
   band. A mapping that silently used one voice would still pass tests 1 and 2.
4. **Roster order is part of the mapping**, and `load_voices` round-trips the roster
   file format shared with the CrispASR repo.
"""
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import voice_assignment as va  # noqa: E402

failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record one check, print it, and keep going so one run reports everything."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'' if ok else f'  <- {detail}'}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- 1. determinism
KEYS = ["0", "42", "534384", "PARHAF_0001", "PARROT:12", "accentué"]
local = [va.voice_for_key(k) for k in KEYS]

_utils = str(Path(__file__).resolve().parent.parent / "utils")
snippet = (
    f"import sys; sys.path.insert(0, {_utils!r}); import voice_assignment as va; "
    f"print(chr(10).join(va.voice_for_key(k) for k in {KEYS!r}))"
)
remote = {}
for seed in ("0", "1", "12345"):
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                          text=True, check=True, env={"PYTHONHASHSEED": seed,
                                                      "PATH": "/usr/bin:/bin"})
    remote[seed] = proc.stdout.split()

check("same voices under every PYTHONHASHSEED",
      all(v == local for v in remote.values()),
      f"local={local} remote={remote}")
# A regression guard with the values written out: if the digest, the encoding or the
# roster order changes, every v2 clip would be relabelled, so that must be a decision
# and not an accident. Update these only together with a regeneration of the audio.
check("mapping is pinned to known values",
      local == ["h_frm_chf", "h_frf_ptm", "fr_female", "h_frm_chf", "h_frf_esm",
                "h_frm_ptf"],
      str(local))

# --------------------------------------------------- 2. group cohesion + key field
dict_rows = [{"term_index": 7, "variant_index": i, "term": "aspirine"} for i in range(9)]
doc_rows = [{"source_id": "PARHAF_0001", "term_index": 3, "chunk_index": i}
            for i in range(5)]

check("group field is term_index without source_id",
      va.group_field(dict_rows[0]) == "term_index", va.group_field(dict_rows[0]))
check("group field is source_id with one",
      va.group_field(doc_rows[0]) == "source_id", va.group_field(doc_rows[0]))
check("a term's variants all get one voice",
      len({va.voice_for_row(r) for r in dict_rows}) == 1,
      str({va.voice_for_row(r) for r in dict_rows}))
check("a document's chunks all get one voice",
      len({va.voice_for_row(r) for r in doc_rows}) == 1,
      str({va.voice_for_row(r) for r in doc_rows}))
check("source_id wins over term_index, so two documents sharing a term_index differ",
      va.voice_for_row({"source_id": "A", "term_index": 3})
      != va.voice_for_row({"source_id": "PARHAF_0001", "term_index": 3}))
# "row" mode is the opposite guarantee: the same group, different variants, several voices.
check("row mode spreads a group over voices",
      len({va.voice_for_row(r, mode="row") for r in dict_rows}) > 1,
      str({va.voice_for_row(r, mode="row") for r in dict_rows}))

# ------------------------------------------------------------------- 3. balance
counts = Counter(va.voice_for_key(str(i)) for i in range(50_000))
even = 50_000 / len(va.VOICES_FR_V2)
check("every voice is used", len(counts) == len(va.VOICES_FR_V2), str(sorted(counts)))
check("no voice is off by more than 5 pct over 50k keys",
      all(abs(n - even) < 0.05 * even for n in counts.values()),
      str(dict(counts)))

# ------------------------------------------------- 4. roster order and load_voices
rotated = va.VOICES_FR_V2[1:] + va.VOICES_FR_V2[:1]
check("roster order is part of the mapping",
      va.voice_for_key("42", rotated) != va.voice_for_key("42", va.VOICES_FR_V2))

with tempfile.TemporaryDirectory() as tmp:
    roster = Path(tmp) / "voices.txt"
    roster.write_text("# a comment\n\nfr_female\nfr_male\n  h_itf_arm  \n")
    check("load_voices reads the shared roster format",
          va.load_voices(roster) == ("fr_female", "fr_male", "h_itf_arm"),
          str(va.load_voices(roster)))
    dupes = Path(tmp) / "dupes.txt"
    dupes.write_text("fr_male\nfr_male\n")
    try:
        va.load_voices(dupes)
        check("load_voices rejects duplicates", False, "no error raised")
    except ValueError:
        check("load_voices rejects duplicates", True)

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
