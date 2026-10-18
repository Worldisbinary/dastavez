"""Ingestion: per page, extract (pypdf baseline + layout-aware), route, OCR when needed.

Output: ``data/pages/<doc_id>.jsonl`` with one record per page holding every candidate text
(so evaluation can compare systems on identical pages) and the route taken.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .corpus import Document, verify
from .ocr import TesseractOCR
from .parse import blocks_from_plain_text, layout_pages, pypdf_pages
from .routing import arbitrate, assess, decide
from .textutil import clean_text

log = logging.getLogger(__name__)


@dataclass
class PageRecord:
    doc_id: str
    page: int
    page_label: str | None
    stratum: str
    language: str
    pypdf_text: str
    layout_text: str
    ocr_text: str | None
    final_text: str
    blocks: list[dict]
    route: str  # text_layer | ocr | ocr_rejected | ocr_unavailable
    reason: str
    ocr_lang: str | None
    image_coverage: float
    quality_text_layer: dict
    q_ocr: float | None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def ingest_document(
    doc: Document, pdf_dir: Path, ocr: TesseractOCR | None, *, force_ocr: bool = False
) -> list[PageRecord]:
    pdf_path = pdf_dir / doc.filename
    if not verify(doc, pdf_dir):
        raise FileNotFoundError(f"{pdf_path} missing or hash mismatch; run `dastavez corpus download`")
    baseline = pypdf_pages(pdf_path)
    records: list[PageRecord] = []
    for parsed in layout_pages(pdf_path):
        layout_text = parsed.text
        decision = decide(layout_text, doc.language, parsed.image_coverage)
        if force_ocr and not decision.needs_ocr:
            decision = type(decision)(True, "forced", decision.ocr_lang)
        quality = assess(layout_text, doc.language)
        ocr_text = None
        q_ocr = None
        final_text = layout_text
        blocks = [b.__dict__ for b in parsed.blocks]
        route = "text_layer"
        if decision.needs_ocr:
            if ocr is None or not ocr.available or not ocr.info.has(decision.ocr_lang):
                route = "ocr_unavailable"
            else:
                raw = ocr.ocr_page(pdf_path, parsed.page, decision.ocr_lang, pdf_sha=doc.sha256)
                ocr_text = clean_text(raw)
                choice, _, q_ocr = arbitrate(layout_text, ocr_text, doc.language)
                if choice == "ocr":
                    route = "ocr"
                    final_text = ocr_text
                    blocks = [b.__dict__ for b in blocks_from_plain_text(raw)]
                else:
                    route = "ocr_rejected"
        records.append(
            PageRecord(
                doc_id=doc.doc_id,
                page=parsed.page,
                page_label=parsed.page_label,
                stratum=doc.stratum,
                language=doc.language,
                pypdf_text=clean_text(baseline[parsed.page - 1]) if parsed.page - 1 < len(baseline) else "",
                layout_text=layout_text,
                ocr_text=ocr_text,
                final_text=final_text,
                blocks=blocks,
                route=route,
                reason=decision.reason,
                ocr_lang=decision.ocr_lang if decision.needs_ocr else None,
                image_coverage=parsed.image_coverage,
                quality_text_layer=quality.to_dict(),
                q_ocr=q_ocr,
            )
        )
    return records


def pages_path(data_dir: Path, doc_id: str) -> Path:
    return data_dir / "pages" / f"{doc_id}.jsonl"


def write_pages(records: list[PageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_pages(data_dir: Path, doc_ids: list[str] | None = None) -> list[dict]:
    out = []
    folder = data_dir / "pages"
    files = sorted(folder.glob("*.jsonl")) if doc_ids is None else [folder / f"{d}.jsonl" for d in doc_ids]
    for p in files:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def ingest_corpus(docs: list[Document], pdf_dir: Path, data_dir: Path, ocr: TesseractOCR | None) -> dict:
    t0 = time.time()
    routes: Counter = Counter()
    reasons: Counter = Counter()
    per_doc = {}
    for doc in docs:
        t = time.time()
        records = ingest_document(doc, pdf_dir, ocr)
        write_pages(records, pages_path(data_dir, doc.doc_id))
        c = Counter(r.route for r in records)
        routes.update(c)
        reasons.update(r.reason for r in records)
        per_doc[doc.doc_id] = {"pages": len(records), "routes": dict(c), "seconds": round(time.time() - t, 1)}
        log.info("%s: %d pages %s (%.1fs)", doc.doc_id, len(records), dict(c), time.time() - t)
    return {
        "pages": sum(v["pages"] for v in per_doc.values()),
        "routes": dict(routes),
        "reasons": dict(reasons),
        "documents": per_doc,
        "ocr_engine": (ocr.info.version if ocr and ocr.available else None),
        "ocr_languages": list(ocr.info.languages) if ocr and ocr.available else [],
        "seconds": round(time.time() - t0, 1),
    }
