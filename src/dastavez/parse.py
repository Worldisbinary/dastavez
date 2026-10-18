"""PDF text extraction: a pypdf baseline and a layout-aware pdfplumber parser.

The layout-aware parser keeps, for every page:
* reading order (line grouping top-to-bottom, with simple two-column detection),
* headings (font-size / bold heuristics relative to the page's body size),
* tables (pdfplumber table finder, rendered as `cell | cell` rows, kept in place),
* the physical page number and, when found, the printed page label.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .textutil import clean_text

log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class Block:
    kind: str  # heading | paragraph | table
    text: str


@dataclass
class ParsedPage:
    page: int  # 1-based physical page index
    text: str
    blocks: list[Block] = field(default_factory=list)
    page_label: str | None = None
    image_coverage: float = 0.0  # fraction of page area covered by raster images
    n_words: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["blocks"] = [asdict(b) for b in self.blocks]
        return d


# --------------------------------------------------------------------------- pypdf baseline


def pypdf_pages(pdf_path: str | Path) -> list[str]:
    """Baseline: plain pypdf ``extract_text()`` per page (what most RAG tutorials do)."""
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception as exc:  # malformed content streams happen in the wild
            log.warning("pypdf failed on a page of %s: %s", pdf_path, exc)
            out.append("")
    return out


# --------------------------------------------------------------------------- layout-aware

_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?[-–(]?\s*([0-9ivxlcdm०-९]{1,5})\s*[-–)]?\s*$", re.I)


@dataclass
class _Line:
    top: float
    bottom: float
    x0: float
    x1: float
    text: str
    size: float
    bold: bool


def _group_lines(words: list[dict], y_tol: float = 3.0) -> list[_Line]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines: list[list[dict]] = []
    for w in words:
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for ws in lines:
        ws.sort(key=lambda w: w["x0"])
        sizes = [float(w.get("size") or 0) for w in ws]
        bold = sum("bold" in str(w.get("fontname", "")).lower() for w in ws) >= max(1, len(ws) * 0.6)
        out.append(
            _Line(
                top=min(w["top"] for w in ws),
                bottom=max(w["bottom"] for w in ws),
                x0=ws[0]["x0"],
                x1=ws[-1]["x1"],
                text=" ".join(w["text"] for w in ws),
                size=statistics.median(sizes) if sizes else 0.0,
                bold=bold,
            )
        )
    return out


def _find_gutter(words: list[dict], width: float) -> float | None:
    """Return the x of a vertical gutter splitting a two-column page, if any."""
    if len(words) < 80:
        return None
    lo, hi = width * 0.35, width * 0.65
    step = width / 200
    best = None
    x = lo
    while x <= hi:
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        left = sum(1 for w in words if w["x1"] <= x)
        right = sum(1 for w in words if w["x0"] >= x)
        if crossing <= len(words) * 0.02 and left > len(words) * 0.3 and right > len(words) * 0.3:
            best = x if best is None else best
        x += step
    return best


def _lines_to_blocks(lines: list[_Line], body_size: float) -> list[Block]:
    blocks: list[Block] = []
    para: list[str] = []
    prev: _Line | None = None

    def flush() -> None:
        if para:
            blocks.append(Block("paragraph", " ".join(para)))
            para.clear()

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        n_words = len(text.split())
        is_heading = (
            n_words <= 14
            and not text.endswith((",", ";"))
            and (
                (body_size and ln.size >= body_size * 1.15)
                or (ln.bold and n_words <= 12)
                or bool(
                    re.match(r"^(?:\d+(?:\.\d+){0,3}\.?|[A-Z]\.|[IVX]+\.|अध्याय|भाग)\s+\S", text)
                    and n_words <= 10
                    and ln.bold
                )
            )
        )
        if is_heading:
            flush()
            blocks.append(Block("heading", text))
            prev = ln
            continue
        # paragraph break on a vertical gap larger than ~1 line
        if prev is not None and para and (ln.top - prev.bottom) > max(4.0, (ln.bottom - ln.top) * 0.9):
            flush()
        para.append(text)
        prev = ln
    flush()
    return blocks


def _table_text(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [clean_text(c or "").replace("\n", " ") for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def layout_pages(pdf_path: str | Path) -> Iterator[ParsedPage]:
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                yield _parse_page(page, idx)
            except Exception as exc:
                log.warning("layout parse failed on %s p%d: %s", pdf_path, idx, exc)
                yield ParsedPage(page=idx, text="", blocks=[])


def _parse_page(page, idx: int) -> ParsedPage:
    # "fake bold" PDFs draw every glyph twice with a tiny offset ("PPrraaddhhaann")
    page = page.dedupe_chars(tolerance=1)
    width, height = float(page.width), float(page.height)
    area = max(width * height, 1.0)
    img_area = 0.0
    for im in page.images:
        x0, x1 = max(0.0, float(im["x0"])), min(width, float(im["x1"]))
        t, b = max(0.0, float(im["top"])), min(height, float(im["bottom"]))
        if x1 > x0 and b > t:
            img_area += (x1 - x0) * (b - t)
    image_coverage = min(1.0, img_area / area)

    tables = []
    try:
        for tb in page.find_tables():
            data = tb.extract()
            if data and len(data) >= 2 and max(len(r) for r in data) >= 2:
                tables.append((tb.bbox, _table_text(data)))
    except Exception:  # table finder is best-effort
        tables = []

    words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False, use_text_flow=False)

    def in_table(w: dict) -> bool:
        cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b, _ in tables)

    body_words = [w for w in words if not in_table(w)]

    # printed page label: a lone number in the top/bottom 7% band
    page_label = None
    margin_words = []
    for ln in _group_lines(body_words):
        in_margin = ln.top < height * 0.07 or ln.bottom > height * 0.93
        m = _PAGE_NUM_RE.match(ln.text)
        if in_margin and m:
            page_label = m.group(1)
            margin_words.append(ln)
    if margin_words:
        tops = {round(ln.top) for ln in margin_words}
        body_words = [
            w for w in body_words if round(w["top"]) not in tops or not _PAGE_NUM_RE.match(w["text"])
        ]

    sizes = [float(w.get("size") or 0) for w in body_words if w.get("size")]
    body_size = statistics.median(sizes) if sizes else 0.0

    gutter = _find_gutter(body_words, width)
    if gutter is not None:
        left = [w for w in body_words if (w["x0"] + w["x1"]) / 2 < gutter]
        right = [w for w in body_words if (w["x0"] + w["x1"]) / 2 >= gutter]
        line_groups = [_group_lines(left), _group_lines(right)]
    else:
        line_groups = [_group_lines(body_words)]

    blocks: list[Block] = []
    # interleave tables by vertical position within the (first) column stream
    pending_tables = sorted(tables, key=lambda t: t[0][1])
    for group in line_groups:
        stream: list[tuple[float, object]] = [(ln.top, ln) for ln in group]
        if pending_tables and group is line_groups[0]:
            stream += [(bbox[1], txt) for bbox, txt in pending_tables]
            pending_tables = []
        stream.sort(key=lambda t: t[0])
        buf: list[_Line] = []
        for _, item in stream:
            if isinstance(item, str):
                blocks.extend(_lines_to_blocks(buf, body_size))
                buf = []
                if item.strip():
                    blocks.append(Block("table", item))
            else:
                buf.append(item)
        blocks.extend(_lines_to_blocks(buf, body_size))

    blocks = [
        Block(b.kind, clean_text(b.text) if b.kind != "table" else b.text) for b in blocks if b.text.strip()
    ]
    text = "\n\n".join(b.text for b in blocks)
    return ParsedPage(
        page=idx,
        text=text,
        blocks=blocks,
        page_label=page_label,
        image_coverage=round(image_coverage, 4),
        n_words=len(words),
    )


def blocks_from_plain_text(text: str) -> list[Block]:
    """Blocks for OCR output: blank-line separated paragraphs; short title-like lines -> headings."""
    blocks = []
    for para in re.split(r"\n\s*\n", text):
        para = clean_text(para)
        if not para:
            continue
        lines = para.split("\n")
        if len(lines) == 1 and len(para.split()) <= 10 and not para.endswith((".", ",", ";", "।")):
            blocks.append(Block("heading", para))
        else:
            blocks.append(Block("paragraph", " ".join(lines)))
    return blocks
