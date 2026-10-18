"""OCR cascade tests. They need the tesseract binary with eng + hin traineddata (installed in CI
via apt: tesseract-ocr tesseract-ocr-hin, fonts-noto-core). Skipped when unavailable."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from dastavez.corpus import Document, sha256_file
from dastavez.ingest import ingest_document
from dastavez.ocr import TesseractOCR, probe

pytestmark = pytest.mark.ocr

INFO = probe()
needs_tesseract = pytest.mark.skipif(not INFO.available, reason=f"tesseract unavailable: {INFO.error}")

DEVANAGARI_FONTS = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/Nirmala.ttf",
    "C:/Windows/Fonts/mangal.ttf",
]
LATIN_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _font(cands, size):
    for c in cands:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return None


def _scan_pdf(tmp_path: Path, lines: list[str], font) -> Path:
    img = Image.new("L", (1700, 900), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((80, 80 + i * 110), line, fill=0, font=font)
    pdf = tmp_path / "scan.pdf"
    img.save(pdf, "PDF", resolution=200)  # image-only PDF: no text layer at all
    return pdf


def _doc(pdf: Path, lang: str) -> Document:
    return Document("scan", "t", "s", "m", lang, "scanned", "file://local", sha256_file(pdf), 1)


@needs_tesseract
def test_english_scan_is_routed_to_ocr(tmp_path):
    font = _font(LATIN_FONTS, 56)
    if font is None:
        pytest.skip("no latin TTF font found")
    folder = tmp_path / "pdfs"
    folder.mkdir()
    pdf = _scan_pdf(folder, ["Pradhan Mantri Awas Yojana", "Housing for All Mission"], font)
    recs = ingest_document(_doc(pdf, "en"), folder, TesseractOCR(dpi=200, cache_dir=tmp_path / "cache"))
    r = recs[0]
    assert r.pypdf_text == ""  # the baseline gets nothing from an image-only page
    assert r.reason == "empty_text_layer" and r.route == "ocr" and r.ocr_lang == "eng"
    assert "awas yojana" in r.final_text.lower()


@needs_tesseract
@pytest.mark.skipif(not INFO.has("hin"), reason="hin traineddata missing")
def test_hindi_scan_ocr(tmp_path):
    font = _font(DEVANAGARI_FONTS, 64)
    if font is None:
        pytest.skip("no Devanagari font found (install fonts-noto-core)")
    folder = tmp_path / "pdfs"
    folder.mkdir()
    _scan_pdf(folder, ["प्रधानमंत्री आवास योजना", "सबके लिए आवास"], font)
    recs = ingest_document(
        _doc(folder / "scan.pdf", "hi"), folder, TesseractOCR(dpi=200, cache_dir=tmp_path / "c")
    )
    r = recs[0]
    assert r.route == "ocr" and r.ocr_lang == "hin+eng"
    assert "आवास" in r.final_text


def test_missing_tesseract_degrades_to_ocr_unavailable(tmp_path):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    img = Image.new("L", (400, 200), 255)
    img.save(folder / "scan.pdf", "PDF")
    recs = ingest_document(_doc(folder / "scan.pdf", "en"), folder, None)
    assert recs[0].route == "ocr_unavailable" and recs[0].reason == "empty_text_layer"
