"""Runtime settings, read from the environment (and an optional .env file).

Nothing secret is ever written to disk by this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: python-dotenv is a core dependency, but stay importable without it
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    home: Path = field(default_factory=lambda: Path(_env("DASTAVEZ_HOME", os.getcwd())).resolve())
    database_url: str | None = field(default_factory=lambda: _env("DASTAVEZ_DATABASE_URL"))
    gemini_api_key: str | None = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("DASTAVEZ_GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embed_model: str = field(
        default_factory=lambda: _env("DASTAVEZ_GEMINI_EMBED_MODEL", "gemini-embedding-001")
    )
    # auto | fastembed | gemini | hash
    embedder: str = field(default_factory=lambda: _env("DASTAVEZ_EMBEDDER", "auto"))
    embed_model: str = field(
        default_factory=lambda: _env("DASTAVEZ_EMBED_MODEL", "intfloat/multilingual-e5-large")
    )
    # auto | cross-encoder | lexical | none
    reranker: str = field(default_factory=lambda: _env("DASTAVEZ_RERANKER", "auto"))
    rerank_model: str = field(
        default_factory=lambda: _env("DASTAVEZ_RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
    )
    tesseract_cmd: str | None = field(default_factory=lambda: _env("TESSERACT_CMD"))
    ocr_dpi: int = field(default_factory=lambda: int(_env("DASTAVEZ_OCR_DPI", "300")))

    @property
    def corpus_dir(self) -> Path:
        return self.home / "corpus"

    @property
    def pdf_dir(self) -> Path:
        # Downloaded PDFs are untrusted data and live in their own directory.
        return Path(_env("DASTAVEZ_PDF_DIR", str(self.corpus_dir / "pdfs")))

    @property
    def data_dir(self) -> Path:
        return Path(_env("DASTAVEZ_DATA_DIR", str(self.home / "data")))

    @property
    def model_cache(self) -> Path:
        return Path(_env("DASTAVEZ_MODEL_CACHE", str(self.home / ".cache" / "fastembed")))

    @property
    def results_dir(self) -> Path:
        return self.home / "results"

    @property
    def gold_dir(self) -> Path:
        return self.home / "gold"


def get_settings() -> Settings:
    return Settings()
