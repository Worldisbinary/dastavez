"""OCR backend: render a PDF page with pypdfium2 and run Tesseract (LSTM engine).

Tesseract is located via, in order: ``TESSERACT_CMD``, ``PATH``, and the default Windows
install locations. Language packs: ``eng`` and ``hin`` (Devanagari). Results are cached on
disk keyed by (pdf sha256, page, lang, dpi, tesseract version) so re-running ingestion is cheap.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_WINDOWS_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"E:\tools\tesseract\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


@dataclass(frozen=True)
class OcrInfo:
    cmd: str | None
    version: str | None
    languages: tuple[str, ...]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.cmd is not None and self.error is None

    def has(self, lang: str) -> bool:
        return all(part in self.languages for part in lang.split("+"))


def find_tesseract(explicit: str | None = None) -> str | None:
    for cand in [explicit, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"), *_WINDOWS_CANDIDATES]:
        if cand and Path(cand).exists():
            return str(cand)
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


@lru_cache(maxsize=4)
def probe(explicit: str | None = None) -> OcrInfo:
    cmd = find_tesseract(explicit)
    if not cmd:
        return OcrInfo(None, None, (), "tesseract binary not found")
    try:
        ver = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=30)
        version = (ver.stdout or ver.stderr).splitlines()[0].strip()
        langs = subprocess.run([cmd, "--list-langs"], capture_output=True, text=True, timeout=30)
        lines = (langs.stdout or "").splitlines()[1:]
        languages = tuple(sorted(ln.strip().replace("\\", "/") for ln in lines if ln.strip()))
        return OcrInfo(cmd, version, languages)
    except (OSError, subprocess.SubprocessError) as exc:  # e.g. blocked by Smart App Control
        return OcrInfo(cmd, None, (), f"tesseract failed to run: {exc}")


def render_page(pdf_path: str | Path, page_no: int, dpi: int = 300):
    """Render a 1-based page to a greyscale PIL image."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_no - 1]
        image = page.render(scale=dpi / 72.0, grayscale=True).to_pil()
        page.close()
        return image
    finally:
        pdf.close()


class TesseractOCR:
    name = "tesseract"

    def __init__(self, cmd: str | None = None, dpi: int = 300, cache_dir: str | Path | None = None):
        self.info = probe(cmd)
        self.dpi = dpi
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def available(self) -> bool:
        return self.info.available

    def _cache_path(self, pdf_sha: str, page_no: int, lang: str, psm: int) -> Path | None:
        if not self.cache_dir:
            return None
        key = f"{pdf_sha}|{page_no}|{lang}|{self.dpi}|{psm}|{self.info.version}"
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.txt"

    def ocr_page(self, pdf_path: str | Path, page_no: int, lang: str, pdf_sha: str = "", psm: int = 3) -> str:
        if not self.available:
            raise RuntimeError(self.info.error or "tesseract unavailable")
        if not self.info.has(lang):
            raise RuntimeError(f"tesseract language pack missing for {lang!r}; have {self.info.languages}")
        cache = self._cache_path(pdf_sha or str(pdf_path), page_no, lang, psm)
        if cache and cache.exists():
            return cache.read_text(encoding="utf-8")
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = self.info.cmd
        image = render_page(pdf_path, page_no, self.dpi)
        text = pytesseract.image_to_string(image, lang=lang, config=f"--oem 1 --psm {psm}")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text
