"""Stdlib tests for utils/voice_assignment: the v2 voice column.

Run: `python tests/test_voice_assignment.py`

Four properties, each of which would otherwise fail silently and produce a `voice`
column that does not describe the audio:

1. **Cross-process determinism.** The generator picks the voice at synthesis time and
   the manifest builder re-derives it later, in another process. Python randomises
   string hashing per process, so a builtin `hash()` here would mislabel every clip.
   The test re-derives in subprocesses under two different `PYTHONHASHSEED` values.
2. **Group cohesion and the key field.** All clips of a group must share a voice, and
   the group field must be the same one `01_build_nemo_manifest.detect_grouping` picks
   (`source_id` when present, else `term_index`), or a PARHAF document could be split
   across speakers while still claiming to be one group.
3. **Balance.** Not exact (the assignment is a hash, see the module docstring), but
   it must not collapse: over many groups every voice has to appear, within a loose
   band. A mapping that silently used one voice would still pass tests 1 and 2.
4. **Roster order is part of the mapping**, and `load_voices` round-trips the roster
   file format shared with the CrispASR repo.
5. **The provenance file round-trips between the two stages.** Stage 05 writes it,
   stage 99 reads it to check that the roster it is about to hash with is the one
   the audio was generated with. Both go through the same two functions, so what is
   tested is that a write is readable, that a second stage written into the same
   directory does not erase the first (all five stages' inputs are called
   `generated_dataset.jsonl`, so the key cannot be the filename), and that a missing
   file reads as None rather than raising.
"""
import json
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
# `retired=()` throughout this block: it checks the HISTORICAL mapping, the one
# the audio on disk was generated under, which must stay reproducible forever.
local = [va.voice_for_key(k, retired=()) for k in KEYS]

_utils = str(Path(__file__).resolve().parent.parent / "utils")
snippet = (
    f"import sys; sys.path.insert(0, {_utils!r}); import voice_assignment as va; "
    f"print(chr(10).join(va.voice_for_key(k, retired=()) for k in {KEYS!r}))"
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
# This pin is the PRE-retirement mapping (`retired=()`), which is the one the audio
# on disk is generated under. It has moved exactly once, on 2026-10-15, when the
# roster went from ten voices to nine (`h_frm_chf` dropped, see
# voice_assignment.RETIRED_VOICES_FR_V2) and the 148,303 clips then on disk were
# deleted in the same operation, which is the only circumstance that permits it.
check("historical mapping is pinned to known values",
      local == ["h_frm_nlf", "h_frf_esm", "h_frm_ptf", "h_frf_itm", "h_frf_esm",
                "h_frm_nlf"],
      str(local))
# No voice is retired at the moment (RETIRED_VOICES_FR_V2 is empty, see the comment
# on it), so the mechanism is exercised with an explicit tuple. Testing it against the
# module default instead would make these checks vanish the moment the default is empty,
# which is exactly when a future retirement needs them most.
RETIRED = ("h_frf_esm",)
# The live mapping differs from the historical one in exactly the positions that drew
# the retired voice: of the six keys above, "42" and "PARROT:12" move and the other
# four do not. That is the whole promise of retiring by re-draw instead of by deletion.
check("retiring only moves the retired voice's keys",
      [va.voice_for_key(k, retired=RETIRED) for k in KEYS]
      == ["h_frm_nlf", "h_frm_nlf", "h_frm_ptf", "h_frf_itm", "h_frm_nlf",
           "h_frm_nlf"],
      str([va.voice_for_key(k, retired=RETIRED) for k in KEYS]))
check("a retired voice is never returned",
      not {va.voice_for_key(str(i), retired=RETIRED) for i in range(5_000)} & set(RETIRED),
      str(sorted({va.voice_for_key(str(i), retired=RETIRED) for i in range(5_000)})))
# Retiring a voice must leave the other eight ninths of the corpus alone, or every
# already-generated clip of a surviving voice would silently become mislabelled.
_moved = sum(va.voice_for_key(str(i), retired=RETIRED)
             != va.voice_for_key(str(i), retired=())
             for i in range(20_000))
check("retiring one of nine voices moves only ~11 pct of keys",
      1_800 < _moved < 2_700, f"{_moved} of 20000 moved")
# `retired=()` must still be able to name the retired voice, since that is how a
# maintenance pass finds the clips it already read and deletes them.
check("the historical mapping still yields the retired voice",
      RETIRED[0] in {va.voice_for_key(str(i), retired=()) for i in range(5_000)})
# With nothing retired, the live mapping must BE the historical one, bit for bit: that
# is what keeps every clip already on disk correctly labelled, and it is the property
# that makes a mid-run retirement cheap and a mid-run roster edit impossible.
check("an empty retirement changes nothing",
      all(va.voice_for_key(str(i)) == va.voice_for_key(str(i), retired=())
          for i in range(5_000)))
# A roster that is entirely retired is a configuration error, not a silent fallback.
try:
    va.voice_for_key("1", voices=("a", "b"), retired=("a", "b"))
    _raised = False
except ValueError:
    _raised = True
check("an entirely retired roster raises", _raised)
# Retiring a second voice must not disturb the keys that never drew either of them.
_two = ("h_frf_esm", "h_itf_def")
_kept = [str(i) for i in range(4_000)
         if va.voice_for_key(str(i), retired=RETIRED) not in _two]
check("a second retirement leaves untouched keys untouched",
      all(va.voice_for_key(k, retired=RETIRED) == va.voice_for_key(k, retired=_two)
          for k in _kept),
      f"{sum(va.voice_for_key(k, retired=RETIRED) != va.voice_for_key(k, retired=_two) for k in _kept)}"
      f" of {len(_kept)} moved")

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
# Over the SURVIVORS: a retired voice takes no share, and its share is spread evenly
# over the rest rather than dumped on one replacement.
_survivors = [v for v in va.VOICES_FR_V2 if v not in va.RETIRED_VOICES_FR_V2]
even = 50_000 / len(_survivors)
check("every surviving voice is used", len(counts) == len(_survivors),
      str(sorted(counts)))
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

# ------------------------------------------------------- 5. provenance round-trip
with tempfile.TemporaryDirectory() as tmp:
    audio = Path(tmp) / "dictionary"
    audio.mkdir()
    dict_in = Path(tmp) / "01_dictionnary" / "generated_dataset.jsonl"
    drugs_in = Path(tmp) / "02_drugs" / "generated_dataset.jsonl"

    check("the key names the stage, not just the shared filename",
          va.provenance_key(dict_in) == "01_dictionnary/generated_dataset.jsonl",
          va.provenance_key(dict_in))
    check("two stages do not collide on their key",
          va.provenance_key(dict_in) != va.provenance_key(drugs_in))

    check("a missing provenance file reads as None",
          va.read_provenance(audio, dict_in) is None)

    va.write_provenance(audio, dict_in, voices=va.VOICES_FR_V2, mode="group",
                        field="term_index", counts=Counter({"fr_male": 3}))
    entry = va.read_provenance(audio, dict_in)
    check("a written entry reads back whole",
          entry == {"input": "01_dictionnary/generated_dataset.jsonl",
                    "voices": list(va.VOICES_FR_V2), "mode": "group",
                    "group_field": "term_index", "rows_per_voice": {"fr_male": 3}},
          json.dumps(entry))

    # The second stage must not erase the first: in some layouts both generate into
    # one audio folder, and stage 99 then reads the wrong roster (or none).
    va.write_provenance(audio, drugs_in, voices=("fr_female",), mode="single",
                        field=None, counts=Counter())
    doc = json.loads((audio / va.PROVENANCE_FILENAME).read_text())
    check("a second stage is merged in, not written over", len(doc) == 2, str(sorted(doc)))
    check("the first stage's entry survived",
          va.read_provenance(audio, dict_in) == entry)
    # An unknown stage in a file that exists is the same answer as no file: None, so
    # stage 99 takes the same "generated before this existed" path either way.
    check("an entry for another stage is not returned",
          va.read_provenance(audio, Path(tmp) / "07_acronyms" / "generated_dataset.jsonl")
          is None)

# --------------------------------------------- 6. "row" mode keys on the index that varies
# v2 runs "row" on the term stages and "group" on the document stages, but the two must
# not be the same thing by accident: PARHAF and PARROT rows carry variant_index == 0 on
# EVERY row with the real position in chunk_index, so a row-mode key built from
# variant_index collapses to the group key and the mode silently does nothing. These
# checks fail if the field preference in assignment_key is flipped back.
doc_rows = [{"source_id": "PARHAF_0001", "term_index": 7, "variant_index": 0,
             "chunk_index": i} for i in range(6)]
doc_group_keys = [va.assignment_key(r, field="source_id", mode="group") for r in doc_rows]
doc_row_keys = [va.assignment_key(r, field="source_id", mode="row") for r in doc_rows]

check("a document's chunks share one group-mode key", len(set(doc_group_keys)) == 1,
      str(doc_group_keys))
check("row mode gives a document's chunks distinct keys", len(set(doc_row_keys)) == 6,
      str(doc_row_keys))
check("row mode keys a chunk on chunk_index, not variant_index",
      doc_row_keys[3] == "PARHAF_0001:3", doc_row_keys[3])
check("row mode is not group mode on a document",
      doc_row_keys[0] != doc_group_keys[0], f"{doc_row_keys[0]} == {doc_group_keys[0]}")
# The point of row mode on a document stage would be to spread chunks over speakers; the
# reason v2 does NOT use it there is editorial (one reader per report), not technical.
check("row mode actually spreads a document over several voices",
      len({va.voice_for_row(r, field="source_id", mode="row") for r in doc_rows}) > 1,
      str([va.voice_for_row(r, field="source_id", mode="row") for r in doc_rows]))

# The term stages have no chunk_index at all, so the fix must leave them exactly as they
# were: dictionary / drugs / acronyms are 92.5 pct of the clips and are what v2 runs in
# row mode, so a shift here would relabel them.
term_rows = [{"term_index": 1234, "variant_index": i} for i in range(8)]
term_row_keys = [va.assignment_key(r, field="source_id", mode="row") for r in term_rows]
check("a term's variants key on variant_index", term_row_keys == [f"1234:{i}" for i in range(8)],
      str(term_row_keys))
check("row mode spreads a term's variants over several voices",
      len({va.voice_for_row(r, field="source_id", mode="row") for r in term_rows}) > 1,
      str([va.voice_for_row(r, field="source_id", mode="row") for r in term_rows]))
# A row with neither index (possible on a hand-made row) must still produce a key rather
# than raise, since the group alone already identifies it.
check("a row with no index field still keys",
      va.assignment_key({"term_index": 9}, field="source_id", mode="row") == "9:0",
      va.assignment_key({"term_index": 9}, field="source_id", mode="row"))


# ------------------------------------------- 7. the swap a repair pass draws instead
# 06_hotfixes gives a clip that failed every redraw to ANOTHER voice (plan item 4), so
# the swap has to be a different voice, the same one on a re-run (or a resumed pass
# would ship audio the manifest does not describe), and uniform over the survivors
# rather than piling the rejects onto whichever name happens to follow in the roster.
SWAP_KEYS = [str(k) for k in range(9000)]
swaps = {k: va.swap_voice(k, va.voice_for_key(k, retired=()), retired=())
         for k in SWAP_KEYS}
check("a swap is never the voice that already read the clip",
      all(swaps[k] != va.voice_for_key(k, retired=()) for k in SWAP_KEYS))
check("a swap is deterministic",
      all(va.swap_voice(k, va.voice_for_key(k, retired=()), retired=()) == swaps[k]
          for k in SWAP_KEYS[:200]))
swap_counts = Counter(swaps.values())
check("every voice takes swapped clips", len(swap_counts) == len(va.VOICES_FR_V2),
      str(sorted(swap_counts)))
# 9000 keys over 9 voices: 1000 land on each, and each one's 1000 rejects spread over
# the 8 others, so ~1000 per voice again. A 20 pct band catches a collapse without
# failing on hash noise (observed spread: 974 to 1068).
lo, hi = min(swap_counts.values()), max(swap_counts.values())
check("swaps are spread, not dumped on one voice", hi <= 1.2 * lo, f"{lo}..{hi}")
check("a retired voice is never swapped TO",
      all(va.swap_voice(k, "fr_female", retired=("h_frf_esm",)) != "h_frf_esm"
          for k in SWAP_KEYS[:500]))

# `avoid` is the accent spread: the voices already reading the same term. Honoured
# when the roster can afford it, dropped when it cannot, because a clip that keeps
# failing needs a different voice more than the term needs its spread.
avoid_all = va.swap_voice("42", "h_frf_esm", avoid=va.VOICES_FR_V2)
check("avoid is honoured when there is room",
      va.swap_voice("42", "h_frf_esm", avoid=("h_frm_nlf",)) != "h_frm_nlf")
check("avoid is dropped rather than giving up",
      avoid_all is not None and avoid_all != "h_frf_esm", str(avoid_all))
check("a one-voice roster has no swap to offer",
      va.swap_voice("42", "solo", voices=("solo",)) is None)
check("a swap with no current voice still answers",
      va.swap_voice("42", None) in va.VOICES_FR_V2)

# --------------------------- 8. read_dir_provenance: the audio dir is all stage 06 has
with tempfile.TemporaryDirectory() as tmp:
    audio = Path(tmp) / "data" / "dictionary"
    audio.mkdir(parents=True)
    dict_in = Path(tmp) / "01_dictionnary" / "generated_dataset.jsonl"
    drugs_in = Path(tmp) / "02_drugs" / "generated_dataset.jsonl"

    check("no provenance file means no answer", va.read_dir_provenance(audio) is None)

    va.write_provenance(audio, dict_in, voices=va.VOICES_FR_V2, mode="row",
                        field="term_index", counts=Counter({"fr_male": 3}))
    entry = va.read_dir_provenance(audio)
    check("the one entry in a directory is returned",
          entry is not None and entry["mode"] == "row"
          and entry["voices"] == list(va.VOICES_FR_V2), json.dumps(entry))

    # Two stages into one folder is a supported layout, so agreeing entries answer.
    va.write_provenance(audio, drugs_in, voices=va.VOICES_FR_V2, mode="row",
                        field="term_index", counts=Counter({"fr_female": 5}))
    check("two stages that agree still answer",
          (va.read_dir_provenance(audio) or {}).get("mode") == "row")

    # ...and disagreeing ones answer None rather than picking a roster at random: a
    # swap drawn over the wrong roster would name a voice that never read anything here.
    va.write_provenance(audio, Path(tmp) / "03_PARHAF" / "generated_dataset.jsonl",
                        voices=va.VOICES_FR_V2, mode="group", field="source_id",
                        counts=Counter({"fr_male": 2}))
    check("stages that disagree answer None", va.read_dir_provenance(audio) is None)


print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
