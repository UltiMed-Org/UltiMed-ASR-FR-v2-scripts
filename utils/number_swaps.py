"""Which of a label's numbers did a recognizer hear as a DIFFERENT number?

A voxtral take sometimes says the wrong number: "75 ans" comes out as "soixante-cinq",
"50 giga par litre" as "cinq", "1 semaine" as "mille semaines". The CER gates barely see
it (one or two characters out of a few hundred), yet for a medical ASR corpus it is the
worst kind of label noise there is: the model learns to write 75 for what it heard as 65.
Measured on 2026-10-28 over the 601,837 released clips both turbo and parakeet read:
5,291 clips where both recognizers replace the same label number by the same other
number, concentrated on the Italian-accented hybrids (h_itf_def 2,643, h_itf_arm 1,404,
against 73 for fr_female) and on the French teen/seventy/ninety compounds (75 -> 65 752
times, 74 -> 64 448, 6 -> 16 264).

The comparison is done on tokens where each number is ONE token, its digits only, so the
recognizers' formatting choices cancel out: ``1,5`` / ``1.5`` / ``15`` are all ``#15``
and ``100 000`` / ``100000`` are ``#100000``. A swap is counted only inside an
equal-length replace block of the word-level alignment, i.e. the words around the number
line up and only the number differs. That is what keeps a transcript that drifted
somewhere else ("dans 3 semaines" against "zone Lailé 1") from pairing two unrelated
numbers, and a ``1,0`` written ``1-0`` (two tokens for one) from counting at all.

Used two ways: ``04_flag_asr_defects.py`` asks for swaps BOTH recognizers agree on (one
recognizer alone mishears numbers about as often as the TTS mis-says them, 9,198 clips for
turbo alone), and the repair pass's opt-in number gate asks the in-loop judge alone, where
a false reject only costs one more draw.

This file was written by Claude Code.
"""

from __future__ import annotations

import difflib
import re

#: Digits, thousands groups of exactly three joined by a (narrow / no-break) space, one
#: decimal part. Lookahead so "12 345 678" is one number but "2 3" stays two.
_NUMBER = re.compile(r"\d+(?:[   ]\d{3}(?!\d))*(?:[.,]\d+)?")
#: "2 milligrammes et demi" is 2,5 milligrammes, and recognizers write it that way.
_AND_A_HALF = re.compile(r"\b(\d+)((?: [^\W\d]+)?) et demie?\b")
_NON_WORD = re.compile(r"[^\w\s'#]")


def tokens(text: str | None) -> list[str]:
    """Lowercased words, each number one ``#digits`` token (leading zeros dropped).

    >>> tokens("Hb à 1,5 g/dL, 100 000 plaquettes")
    ['hb', 'à', '#15', 'g', 'dl', '#100000', 'plaquettes']
    >>> tokens("2 milligrammes et demi") == tokens("2,5 milligrammes")
    True
    """
    text = _AND_A_HALF.sub(r"\1,5\2", (text or "").lower())
    text = _NUMBER.sub(lambda m: " #" + (re.sub(r"\D", "", m.group()).lstrip("0") or "0") + " ",
                       text)
    return _NON_WORD.sub(" ", text).split()


def swaps(label: str | None, hyp: str | None) -> dict[int, tuple[str, str]]:
    """``{label token index: (label number, heard number)}`` for each swapped number.

    >>> swaps("patient de 75 ans, revu à 3 mois", "Patient de 65 ans revu à 3 mois.")
    {2: ('75', '65')}
    >>> swaps("revu dans 3 semaines", "zone Lailé 1")
    {}
    >>> swaps("score à 1,0", "score à 1-0")
    {}
    """
    ref, heard = tokens(label), tokens(hyp)
    out: dict[int, tuple[str, str]] = {}
    ops = difflib.SequenceMatcher(a=ref, b=heard, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for k in range(i2 - i1):
            a, b = ref[i1 + k], heard[j1 + k]
            if a[0] == "#" and b[0] == "#":
                out[i1 + k] = (a[1:], b[1:])
    return out


def agreed_swaps(label: str | None, a: str | None, b: str | None) -> list[tuple[str, str]]:
    """Swaps two recognizers agree on: same label number, heard as the same other number.

    >>> agreed_swaps("âgé de 75 ans", "âgé de 65 ans", "Âgé de 65 ans.")
    [('75', '65')]
    >>> agreed_swaps("âgé de 75 ans", "âgé de 65 ans", "âgé de 75 ans")
    []
    """
    sa, sb = swaps(label, a), swaps(label, b)
    return [sa[i] for i in sorted(sa) if sb.get(i) == sa[i]]
