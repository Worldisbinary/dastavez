"""Per-page routing: keep the PDF text layer, or send the page to OCR.

The cascade is cheap-first: every page gets the layout-aware text layer; only pages whose
text layer is empty, undecodable, orthographically broken (Devanagari), in the wrong script
(legacy 8-bit Hindi fonts such as Kruti Dev render as latin mojibake), garbled, or that are
scans carrying an old OCR layer are rasterised and OCR'd. When OCR runs, its output must beat
the text layer on a reference-free quality score, otherwise the text layer is kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .textutil import (
    devanagari_words,
    invalid_devanagari_sequences,
    latin_wordlike_ratio,
    script_stats,
)

# Thresholds were fixed by inspecting per-page text-layer statistics of the corpus *before*
# the gold sample was drawn; they were never tuned against any reference transcription.
MIN_CHARS = 25
MAX_PUA_RATIO = 0.005
MIN_PRINTABLE = 0.85
MAX_INVALID_DEV_RATE = 0.03
MIN_WORDLIKE = 0.60
MIN_WORDLIKE_HI_DOC = 0.75
MAX_DEV_RATIO_FOR_MISMATCH = 0.15
SCANNED_IMAGE_COVERAGE = 0.80
OCR_MARGIN = 0.02


@dataclass(frozen=True)
class PageQuality:
    n_chars: int
    printable_ratio: float
    pua_ratio: float
    devanagari_ratio: float
    dev_words: int
    invalid_dev_rate: float
    latin_wordlike: float
    latin_tokens: int
    score: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def assess(text: str, expected_lang: str = "en") -> PageQuality:
    st = script_stats(text)
    dwords = devanagari_words(text)
    invalid = invalid_devanagari_sequences(text)
    inv_rate = invalid / len(dwords) if dwords else 0.0
    wordlike, n_lat = latin_wordlike_ratio(text)
    return PageQuality(
        n_chars=st.n_chars,
        printable_ratio=st.printable_ratio,
        pua_ratio=st.pua_ratio,
        devanagari_ratio=st.devanagari_ratio,
        dev_words=len(dwords),
        invalid_dev_rate=inv_rate,
        latin_wordlike=wordlike,
        latin_tokens=n_lat,
        score=quality_score(text, expected_lang),
    )


def quality_score(text: str, expected_lang: str = "en") -> float:
    """Reference-free text quality in [0, 1], used to arbitrate text layer vs. OCR output."""
    st = script_stats(text)
    if st.n_chars < MIN_CHARS:
        return 0.0
    dwords = devanagari_words(text)
    dev_valid = 1.0 - min(1.0, (invalid_devanagari_sequences(text) / len(dwords)) * 3) if dwords else 1.0
    wordlike, n_lat = latin_wordlike_ratio(text)
    letters = st.devanagari_letters + st.latin_letters
    if letters == 0:
        return 0.1 * st.printable_ratio
    dev_share = st.devanagari_letters / letters
    content = dev_share * dev_valid + (1 - dev_share) * (wordlike if n_lat else 1.0)
    score = st.printable_ratio * (1 - min(1.0, st.pua_ratio * 10)) * content
    if "hi" in expected_lang and expected_lang != "hi+en" and dev_share < MAX_DEV_RATIO_FOR_MISMATCH:
        score *= 0.5  # a Hindi document whose text layer has (almost) no Devanagari
    # a little credit for volume: a 30-char fragment should not beat a full page
    volume = min(1.0, st.n_chars / 400)
    return round(score * (0.8 + 0.2 * volume), 4)


@dataclass(frozen=True)
class RouteDecision:
    needs_ocr: bool
    reason: str
    ocr_lang: str


def decide(text: str, expected_lang: str = "en", image_coverage: float = 0.0) -> RouteDecision:
    q = assess(text, expected_lang)
    wants_hindi = "hi" in expected_lang or q.dev_words > 0
    lang = "hin+eng" if wants_hindi else "eng"
    if q.n_chars < MIN_CHARS:
        return RouteDecision(True, "empty_text_layer", lang)
    if q.pua_ratio > MAX_PUA_RATIO:
        return RouteDecision(True, "undecodable_glyphs", lang)
    if q.printable_ratio < MIN_PRINTABLE:
        return RouteDecision(True, "low_printable_ratio", lang)
    if q.dev_words >= 5 and q.invalid_dev_rate > MAX_INVALID_DEV_RATE:
        return RouteDecision(True, "broken_devanagari", "hin+eng")
    if (
        "hi" in expected_lang
        and q.devanagari_ratio < MAX_DEV_RATIO_FOR_MISMATCH
        and q.latin_tokens >= 3
        and q.latin_wordlike < MIN_WORDLIKE_HI_DOC
    ):
        return RouteDecision(True, "script_mismatch_legacy_font", "hin+eng")
    if q.latin_tokens >= 10 and q.latin_wordlike < MIN_WORDLIKE:
        return RouteDecision(True, "garbled_text", lang)
    if image_coverage >= SCANNED_IMAGE_COVERAGE:
        return RouteDecision(True, "scanned_with_text_layer", lang)
    return RouteDecision(False, "text_layer_ok", lang)


def arbitrate(text_layer: str, ocr_text: str, expected_lang: str) -> tuple[str, float, float]:
    """Pick OCR output only if it beats the text layer by a margin. Returns (choice, q_text, q_ocr)."""
    q_text = quality_score(text_layer, expected_lang)
    q_ocr = quality_score(ocr_text, expected_lang)
    choice = "ocr" if q_ocr > q_text + OCR_MARGIN else "text_layer"
    return choice, q_text, q_ocr
