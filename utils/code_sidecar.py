"""Store what the codes said about each take, beside the audio.

Stage 05 gates takes on `utils/code_loops.py`, whose thresholds are calibrated on
30 takes. That is thin, and the only way to improve it without regenerating audio
is to have kept the numbers. So every attempt, accepted or rejected, appends one
row here, and the row carries the measured features rather than just the verdict:
a later pass can re-decide what counts as derailed over the whole corpus and
compare its answer against stage 06's CER gate, which is the real label.

The row also carries the **semantic track itself**, base64 of little-endian
int16, which is what makes the stored data outlive the current thresholds
entirely. Costs are measured, not guessed: a 15 s clip is 187 frames, so 374
bytes of codes, about 500 base64 characters and ~800 bytes of row including the
JSON. Over 602,885 clips that is roughly 480 MB, against ~120 GB of audio for
the same corpus. The 36 ACOUSTIC columns are deliberately dropped: they would be
925 B/s instead of 25 B/s (~11.7 GB), and they are a flow-matching draw from
fresh noise each frame, so they differ between two identical-sounding takes and
cannot be compared across takes the way the argmax semantic column can.

Mirrors `utils/cfg_sidecar.py` deliberately, down to the module-level append lock
and the "unparseable last line is skipped, not raised" rule: the file is appended
to live by several worker threads, and a run killed mid-write leaves half a line.

Pure stdlib, like `code_loops`, so no stage grows a dependency to write a row.
"""
from __future__ import annotations

import array
import base64
import datetime as dt
import json
import threading
from pathlib import Path

#: Beside the audio, next to cfg_alpha_sidecar.jsonl and voice_assignment.json.
SIDECAR_FILENAME = "code_qc_sidecar.jsonl"

#: Serialises appends from the worker threads. A row carrying a semantic track is
#: several KB, well past the size a single write() is atomic for, so unlike a
#: cfg_alpha row it genuinely cannot be left to O_APPEND. Threads, not processes:
#: the generator's concurrency is a thread pool, same as cfg_sidecar assumes.
_APPEND_LOCK = threading.Lock()


def sidecar_path(audio_dir: Path | str) -> Path:
    """Where the sidecar lives for one audio directory."""
    return Path(audio_dir) / SIDECAR_FILENAME


def encode_track(track: list[int]) -> str:
    """Semantic codes -> base64 of little-endian int16.

    int16 because that is what the dump already is, so this is a copy rather than a
    conversion, and the codes fit: the semantic vocabulary is 8192 wide.

    Examples
    --------
    >>> encode_track([10, 4007, 1])
    'CgCnDwEA'
    >>> decode_track(encode_track([10, 4007, 1]))
    [10, 4007, 1]
    """
    return base64.b64encode(array.array("h", track).tobytes()).decode("ascii")


def decode_track(encoded: str) -> list[int]:
    """Inverse of `encode_track`, for a later analysis pass.

    Examples
    --------
    >>> decode_track('CgCnDwEA')
    [10, 4007, 1]
    >>> decode_track('')
    []
    """
    values = array.array("h")
    raw = base64.b64decode(encoded)
    values.frombytes(raw[:len(raw) - (len(raw) % values.itemsize)])
    return values.tolist()


def record_code_take(audio_dir: Path | str, *, stem: str, take, attempt: int,
                     accepted: bool, track: list[int] | None = None,
                     voice: str | None = None,
                     cfg_alpha: float | None = None) -> Path:
    """Append one row for one attempt at one clip.

    Unlike `cfg_sidecar.record_cfg_alpha`, which only records the exceptions, this
    records EVERY attempt including the accepted ones, because the question it has to
    answer later is "what do the features look like on good clips", and a file holding
    only the rejects cannot answer it.

    Parameters
    ----------
    audio_dir
        The stage's audio directory; the sidecar is written inside it.
    stem
        The clip's stem, the join key to the audio and to the other sidecars.
    take
        A `code_loops.CodeTake`. Every measured field is stored, not just the verdict.
    attempt, accepted
        Which attempt this was, and whether it is the take that landed. A stem with
        several rows is a clip that needed redrawing, which is itself the statistic
        the gate is judged by.
    track
        The semantic codes, stored base64 when given. None stores the features only.
    voice, cfg_alpha
        Provenance, so a defect can be attributed without joining three files.

    Returns
    -------
    Path
        The sidecar path, also when the append failed.

    An OSError is swallowed with no re-raise, exactly as in cfg_sidecar: a QC row is
    not worth failing a clip that is already on disk and valid. The caller logs.
    """
    row = {
        "stem": stem,
        "attempt": attempt,
        "accepted": accepted,
        "voice": voice,
        "cfg_alpha": cfg_alpha,
        "frames": take.frames,
        "dur_s": round(take.duration_s, 3),
        "ended": take.ended,
        "min_distinct_frac": round(take.min_distinct_frac, 4),
        "max_top3_frac": round(take.max_top3_frac, 4),
        "held_frames": take.held_frames,
        "held_code": take.held_code,
        "verdicts": list(take.verdicts),
        "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if track:
        row["semantic"] = encode_track(track)
    # None is dropped (absent voice/cfg_alpha), but False and 0 are kept: "ended":
    # false and "held_frames": 0 are the informative values, so the cfg_sidecar trick
    # of filtering on truthiness would delete exactly the rows that matter.
    line = json.dumps({k: v for k, v in row.items() if v is not None},
                      ensure_ascii=False) + "\n"
    path = sidecar_path(audio_dir)
    try:
        with _APPEND_LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    except OSError:
        pass
    return path


def load_code_takes(audio_dir: Path | str) -> list[dict]:
    """Every row from one directory's sidecar, in append order.

    Returns [] when there is no sidecar. Unparseable lines are skipped rather than
    raising, since the file is appended to live and the last line of a killed run is
    often half written. No caching and no "last row wins" collapse, unlike
    cfg_sidecar: callers here want the whole history, including the rejected attempts.
    """
    out: list[dict] = []
    try:
        text = sidecar_path(audio_dir).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "stem" in row:
            out.append(row)
    return out
