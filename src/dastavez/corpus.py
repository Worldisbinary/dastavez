"""Corpus manifest, download and SHA-256 verification.

PDFs are third-party, untrusted inputs. They are stored in their own directory
(``corpus/pdfs`` by default), never executed, and every file is verified against the
pinned SHA-256 in ``corpus/manifest.json`` before it is parsed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; dastavez-corpus-fetcher/0.1; +https://github.com/Worldisbinary)"
MAX_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    scheme: str
    ministry: str
    language: str  # en | hi | hi+en
    source_kind: str  # digital | scanned
    source_url: str
    sha256: str
    pages: int
    bytes: int = 0

    @property
    def filename(self) -> str:
        return f"{self.doc_id}.pdf"

    @property
    def stratum(self) -> str:
        """Evaluation stratum for pages of this document."""
        if self.source_kind == "scanned":
            return "scanned"
        return "hindi_digital" if self.language.startswith("hi") else "english_digital"


def load_manifest(path: str | Path) -> list[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = Document.__dataclass_fields__
    return [Document(**{k: v for k, v in d.items() if k in fields}) for d in data["documents"]]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(doc: Document, pdf_dir: str | Path) -> bool:
    path = Path(pdf_dir) / doc.filename
    return path.exists() and sha256_file(path) == doc.sha256


def download(
    docs: list[Document], pdf_dir: str | Path, *, force: bool = False, timeout: float = 120.0
) -> dict:
    """Download every document, verify its hash, and quarantine mismatches.

    Returns a report {doc_id: status}. A file whose hash does not match is kept as
    ``<doc_id>.pdf.mismatch`` for inspection and is never used by ingestion.
    """
    import httpx

    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, str] = {}
    with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        for doc in docs:
            target = pdf_dir / doc.filename
            if not force and verify(doc, pdf_dir):
                report[doc.doc_id] = "ok (cached)"
                continue
            try:
                with client.stream("GET", doc.source_url) as resp:
                    resp.raise_for_status()
                    buf = bytearray()
                    for part in resp.iter_bytes():
                        buf.extend(part)
                        if len(buf) > MAX_BYTES:
                            raise ValueError("file exceeds size cap")
            except Exception as exc:
                report[doc.doc_id] = f"error: {exc}"
                continue
            data = bytes(buf)
            if not data.startswith(b"%PDF"):
                report[doc.doc_id] = "error: response is not a PDF (site may block automated downloads)"
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest != doc.sha256:
                (pdf_dir / f"{doc.filename}.mismatch").write_bytes(data)
                report[doc.doc_id] = f"HASH MISMATCH (got {digest[:12]}..., expected {doc.sha256[:12]}...)"
                continue
            target.write_bytes(data)
            report[doc.doc_id] = "ok"
    return report


def import_local(docs: list[Document], src_dir: str | Path, pdf_dir: str | Path) -> dict:
    """Copy already-downloaded PDFs (any filename) into place, matched by SHA-256."""
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    by_hash = {d.sha256: d for d in docs}
    report = {}
    for p in Path(src_dir).glob("*.pdf"):
        d = by_hash.get(sha256_file(p))
        if d:
            (pdf_dir / d.filename).write_bytes(p.read_bytes())
            report[d.doc_id] = f"imported from {p.name}"
    return report
