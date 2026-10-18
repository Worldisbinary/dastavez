"""Unicode helpers: normalisation, script statistics, tokenisation, sentence splitting.

Devanagari notes
----------------
* Block U+0900-U+097F. Dependent vowel signs (matras) U+093E-U+094C and U+0962-U+0963,
  virama (halant) U+094D, nukta U+093C, and the signs U+0900-U+0903 are *combining*
  characters: they must follow a consonant (or, for some signs, a vowel). Python's
  ``\\w`` does not match combining marks, so tokenisation uses explicit ranges.
* Dandas U+0964/U+0965 are sentence punctuation, not letters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEVANAGARI_LETTER = re.compile(r"[ऄ-हऽॐक़-ॡॲ-ॿ]")
DEVANAGARI_ANY = re.compile(r"[ऀ-ॿ]")
LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")

# Dependent signs that cannot start a syllable.
_MATRA = "ा-ौॎॏॕ-ॗॢॣ"
_SIGN = "ऀ-ः"
_VIRAMA = "्"
_NUKTA = "़"
_CONSONANT = "क-हक़-य़ॸ-ॿ"
_INDEPENDENT_VOWEL = "ऄ-औॠॡॲ-ॷ"

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "­​⁠﻿"), None)
_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "•": "*",
        "●": "*",
        "▪": "*",
        "": "*",
        " ": " ",
        " ": " ",
        " ": " ",
    }
)

TOKEN_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*"  # numbers incl. 1,50,000 and 6.5
    r"|[A-Za-zÀ-ɏ]+(?:'[A-Za-z]+)?"  # latin words
    r"|[ऀ-ॣ०-ॿ‌‍]+"  # devanagari runs (no dandas)
)

EN_STOP = frozenset(
    """a an the of to in on for by with from at as is are was were be been being it its this that these
    those and or not no but if then than so such which who whom whose what when where why how all any each
    shall will would should can could may might must do does did has have had under per via into over upon
    about also other their there they them he she his her we our you your i me my up out""".split()
)
HI_STOP = frozenset(
    """का के की को में से पर है हैं था थे थी और या एवं तथा भी यह ये वह वे इस इन उस उन एक लिए द्वारा
    कि जो जिस जिन किया किए की गई गया होगा होगी होंगे हो कर करने करना साथ अपने नहीं""".split()
)
STOPWORDS = EN_STOP | HI_STOP


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Light, lossless-in-meaning cleanup used for stored page text."""
    text = nfc(text).translate(_ZERO_WIDTH).translate(_PUNCT_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_eval(text: str) -> str:
    """Normalisation applied to BOTH hypothesis and reference before scoring extraction.

    NFC, zero-width removal, quote/dash unification, Devanagari digits -> ASCII, casefold,
    all whitespace (including line breaks) collapsed to single spaces. Layout/line breaks are
    deliberately not scored: we measure *text* fidelity, not geometry.
    """
    text = clean_text(text).translate(DEVANAGARI_DIGITS).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    text = nfc(text).translate(DEVANAGARI_DIGITS).casefold()
    toks = [t.replace(",", "") if t[0].isdigit() else t for t in TOKEN_RE.findall(text)]
    toks = [t.replace("‌", "").replace("‍", "") for t in toks]
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [t for t in toks if t]


_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+(?=[\"'(\[]?[A-Z0-9ऀ-ॿ])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Split into sentences on . ! ? and Devanagari danda, also on blank lines."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        # avoid splitting on common abbreviations ("Rs. 10,000", "No. 5", "i.e.")
        if out and re.search(r"\b(?:Rs|No|Nos|Sr|Dr|Mr|Ms|Smt|Shri|i\.e|e\.g|viz|etc|para|Para)\.$", out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


@dataclass(frozen=True)
class ScriptStats:
    n_chars: int  # non-whitespace characters
    printable_ratio: float
    devanagari_letters: int
    latin_letters: int
    devanagari_ratio: float  # devanagari letters / (devanagari + latin letters)
    pua_ratio: float  # private-use + replacement characters / n_chars
    replacement_chars: int


def script_stats(text: str) -> ScriptStats:
    chars = [c for c in text if not c.isspace()]
    n = len(chars)
    if n == 0:
        return ScriptStats(0, 1.0, 0, 0, 0.0, 0.0, 0)
    printable = 0
    dev = lat = pua = repl = 0
    for c in chars:
        cat = unicodedata.category(c)
        cp = ord(c)
        if c == "�":
            repl += 1
        elif 0xE000 <= cp <= 0xF8FF:
            pua += 1
        if cat[0] in "LMNPS" and c != "�" and not (0xE000 <= cp <= 0xF8FF):
            printable += 1
        if DEVANAGARI_LETTER.match(c):
            dev += 1
        elif LATIN_LETTER.match(c):
            lat += 1
    letters = dev + lat
    return ScriptStats(
        n_chars=n,
        printable_ratio=printable / n,
        devanagari_letters=dev,
        latin_letters=lat,
        devanagari_ratio=(dev / letters) if letters else 0.0,
        pua_ratio=(pua + repl) / n,
        replacement_chars=repl,
    )


def contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_ANY.search(text))


_INVALID_DEV = re.compile(
    # a combining sign that does not follow a consonant / nukta / independent vowel
    rf"(?:(?<![{_CONSONANT}{_NUKTA}])[{_MATRA}{_VIRAMA}{_NUKTA}])"
    # two dependent vowel signs in a row
    rf"|(?:[{_MATRA}][{_MATRA}])"
    # a vowel sign right after anusvara/visarga/candrabindu  (e.g. "मंी")
    rf"|(?:[{_SIGN}][{_MATRA}{_VIRAMA}])"
    # a vowel sign after an independent vowel
    rf"|(?:[{_INDEPENDENT_VOWEL}][{_MATRA}{_VIRAMA}])"
)


def invalid_devanagari_sequences(text: str) -> int:
    """Count orthographically impossible Devanagari sequences.

    These appear when a PDF's font lacks a correct ToUnicode map: glyphs for conjuncts and
    reordered matras are dropped or emitted out of logical order (e.g. "धानमंी" for
    "प्रधानमंत्री"), which no human-typed Unicode text contains.
    """
    return len(_INVALID_DEV.findall(nfc(text)))


def devanagari_words(text: str) -> list[str]:
    return re.findall(r"[ऀ-ॣ०-ॿ]+", nfc(text))


_VOWELS = set("aeiouyAEIOUY")


def is_wordlike_latin(token: str) -> bool:
    """Heuristic: does a latin token look like a real (English/romanised) word?"""
    if not token.isalpha() or not token.isascii():
        return False
    if len(token) == 1:
        return token.lower() in {"a", "i"}
    if not any(c in _VOWELS for c in token):
        return token.isupper() and len(token) <= 5  # acronyms like PMAY, ULB, SHG
    if re.search(r"(.)\1\1", token):
        return False
    if re.search(r"[^aeiouyAEIOUY]{5,}", token):
        return False
    # mixed case inside a word ("kkHk", "gfjK") is typical of legacy-font mojibake
    return not re.search(r"[a-z][A-Z]", token)


def latin_wordlike_ratio(text: str) -> tuple[float, int]:
    """Fraction of whitespace-separated latin-ish tokens that look like words.

    Tokens are split on whitespace only so that legacy-font mojibake such as "gfj;k.kk" or
    "jkstxkj" counts as one non-word, instead of being cut into plausible fragments.
    Returns (ratio, number_of_tokens_considered).
    """
    considered = 0
    good = 0
    for raw in text.split():
        tok = raw.strip(".,;:!?()[]{}\"'`-*/|")
        if not tok or not re.search(r"[A-Za-z]", tok):
            continue
        if re.fullmatch(r"[0-9.,/()-]+", tok):
            continue
        considered += 1
        # allow hyphenated / slashed compounds made of words ("PM-KISAN", "EWS/LIG")
        parts = [p for p in re.split(r"[-/]", tok) if p]
        if parts and all(is_wordlike_latin(p) or p.isdigit() for p in parts):
            good += 1
    return ((good / considered) if considered else 1.0), considered
