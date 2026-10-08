#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""Rank what a Parakeet transcription of the corpus writes differently from the labels.

Run over a greedy transcription (no phrase boosting, no regex fixes) of every clip, by a
model fine-tuned on UltiMed. Such a model writes the MAJORITY spelling of what it was
taught, so a label it keeps "getting wrong" in the same way often holds a minority
convention (``post opératoire`` where the model writes ``post-opératoire``) rather than a
model error. Made 2026-10-08 to choose label fixes before re-synthesizing the corpus with
several voices; ``05_label_variant_audit.py`` is the text-only side of the same question.

Each label is aligned word by word with its hypothesis (``difflib``); every differing
stretch of at most ``--max-span`` words on each side becomes a ``label -> hyp`` pair:

    {"kind": "ortho", "label": "post opératoire", "hyp": "post-opératoire", "count": 160}
    {"kind": "case", "label": "Doppler", "hyp": "doppler", "count": 95}
    {"kind": "other", "label": "3-0", "hyp": "trois zéro", "count": 40}

- ``case``: the same words but for capital letters.
- ``ortho``: the same letters once lowercased with accents, hyphens, apostrophes and
  spaces removed: a spelling convention, never a hearing error.
- ``other``: everything else: a real recognition error, a TTS slip, or a label notation
  the audio does not say (digits, symbols).

Punctuation is dropped before aligning (the model's punctuation is not scored here).
Output: a TSV sorted by kind then count, with the clip count per split and one example
clip, plus per-subset and per-split WER on stdout.

This file was written by Claude Code.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import click

_spec = importlib.util.spec_from_file_location("variant_audit", Path(__file__).resolve().parent / "05_label_variant_audit.py")
_va = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_va)

TOKEN = re.compile(r"\w+(?:['’-]\w+)*")


def tokens(text: str) -> list[str]:
    """Words of ``text`` with case, accents, hyphens and elisions kept, punctuation dropped.

    >>> tokens("Suites post-opératoires simples, l'IRM est normale.")
    ['Suites', 'post-opératoires', 'simples', "l'IRM", 'est', 'normale']
    """
    return TOKEN.findall(text.replace("’", "'"))


def fold(span: str) -> str:
    """
    >>> fold("Post opératoire"), fold("post-opératoire"), fold("l'IRM")
    ('postoperatoire', 'postoperatoire', 'lirm')
    """
    return "".join(c for c in _va.strip_accents(span.lower()) if c not in "-' ")


def kind_of(label: str, hyp: str) -> str:
    """
    >>> kind_of("Doppler", "doppler"), kind_of("post opératoire", "post-opératoire"), kind_of("3-0", "trois zéro")
    ('case', 'ortho', 'other')
    """
    if label.lower() == hyp.lower():
        return "case"
    return "ortho" if fold(label) == fold(hyp) else "other"


def diff_pairs(label: str, hyp: str, max_span: int = 4) -> list[tuple[str, str]]:
    """The differing ``(label stretch, hyp stretch)`` pairs, each side at most ``max_span`` words.

    >>> diff_pairs("Suites post opératoires simples.", "Suites post-opératoires simples")
    [('post opératoires', 'post-opératoires')]
    >>> diff_pairs("Le Doppler est normal", "le doppler est normal")
    [('Le Doppler', 'le doppler')]
    >>> diff_pairs("fil de Vicryl 3-0", "fil de vicryl trois zéro")
    [('Vicryl 3-0', 'vicryl trois zéro')]
    """
    a, b = tokens(label), tokens(hyp)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal" and i2 - i1 <= max_span and j2 - j1 <= max_span:
            out.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return out


def word_errors(label: str, hyp: str) -> tuple[int, int]:
    """``(word edits, label words)`` on lowercased tokens, for the WER summary.

    >>> word_errors("Le Doppler est normal", "le doppler est anormal")
    (1, 4)
    """
    a, b = [t.lower() for t in tokens(label)], [t.lower() for t in tokens(hyp)]
    edits = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            edits += max(i2 - i1, j2 - j1)
    return edits, len(a)


def split_of(manifest_dir: str) -> dict[str, str]:
    """``audio path -> split`` from ``<subset>/{train,val,test}.jsonl`` under the NeMo manifest dir."""
    out = {}
    for sub in sorted(os.listdir(manifest_dir)):
        for split in ("train", "val", "test"):
            mf = os.path.join(manifest_dir, sub, f"{split}.jsonl")
            if not os.path.exists(mf):
                continue
            base = os.path.dirname(os.path.realpath(mf))
            for line in open(mf, encoding="utf-8"):
                p = json.loads(line)["audio_filepath"]
                out[os.path.normpath(p if os.path.isabs(p) else os.path.join(base, p))] = split
    return out


@click.command()
@click.option("--hyps", "hyps_paths", multiple=True, required=True, type=click.Path(exists=True),
              help='{"audio", "hyp"} jsonl (scripts/drug-rules/transcribe_onnx.py in the ultra model repo).')
@click.option("--manifests", "manifest_paths", multiple=True, required=True, type=click.Path(exists=True),
              help="NeMo manifests with absolute audio_filepath and a subset field, or relative to the manifest.")
@click.option("--splits-dir", type=click.Path(exists=True), help="NeMo manifest dir, to count clips per split.")
@click.option("--out", type=click.Path(), required=True, help="TSV to write.")
@click.option("--max-span", default=4, show_default=True)
@click.option("--min-count", default=3, show_default=True)
def main(hyps_paths, manifest_paths, splits_dir, out, max_span, min_count):
    hyp = {}
    for p in hyps_paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            hyp[r["audio"]] = r["hyp"]
    split = split_of(splits_dir) if splits_dir else {}
    counts: Counter = Counter()
    by_split: dict = defaultdict(Counter)
    example: dict = {}
    wer: dict = defaultdict(lambda: [0, 0])
    for mf in manifest_paths:
        base = os.path.dirname(os.path.realpath(mf))
        for line in open(mf, encoding="utf-8"):
            r = json.loads(line)
            p = os.path.normpath(r["audio_filepath"] if os.path.isabs(r["audio_filepath"]) else os.path.join(base, r["audio_filepath"]))
            if p not in hyp:
                continue
            sp, sub = split.get(p, "?"), r.get("subset", "?")
            e, n = word_errors(r["text"], hyp[p])
            for key in (f"{sub} {sp}", f"all {sp}", "all all"):
                wer[key][0] += e
                wer[key][1] += n
            for pair in set(diff_pairs(r["text"], hyp[p], max_span)):
                counts[pair] += 1
                by_split[pair][sp] += 1
                example.setdefault(pair, (p, r["text"], hyp[p]))
    rows = [(kind_of(*pair), n, pair) for pair, n in counts.items() if n >= min_count]
    order = {"ortho": 0, "case": 1, "other": 2}
    rows.sort(key=lambda r: (order[r[0]], -r[1], r[2]))
    with open(out, "w", encoding="utf-8") as f:
        f.write("kind\tcount\ttrain\tval\ttest\tlabel\thyp\texample_audio\texample_label\texample_hyp\n")
        for kind, n, pair in rows:
            s = by_split[pair]
            ex = example[pair]
            f.write(f"{kind}\t{n}\t{s['train']}\t{s['val']}\t{s['test']}\t{pair[0]}\t{pair[1]}\t{os.path.join(*ex[0].split(os.sep)[-2:])}\t{ex[1]}\t{ex[2]}\n")
    for key in sorted(wer):
        e, n = wer[key]
        click.echo(f"WER {key}: {100 * e / max(n, 1):.2f} % ({e} / {n} words)")
    for k in order:
        sel = [r for r in rows if r[0] == k]
        click.echo(f"{k}: {len(sel)} pairs seen >= {min_count} times, {sum(r[1] for r in sel)} clip occurrences")
    click.echo(f"-> {out}")


if __name__ == "__main__":
    main()
