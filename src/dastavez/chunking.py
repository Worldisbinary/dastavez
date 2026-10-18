"""Heading-aware chunking that never crosses a page boundary.

Every chunk carries (doc_id, page, page_label, heading) so any retrieved text can be cited
at page level. Headings open a new chunk; long paragraphs are split on sentence boundaries;
tables are kept whole when they fit, otherwise split by rows. The nearest heading seen so far
(possibly on an earlier page) is carried as context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .textutil import split_sentences

MAX_CHARS = 900
MIN_CHARS = 120


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    page_label: str | None
    heading: str
    text: str
    title: str = ""

    def embed_text(self) -> str:
        head = f"{self.title} | {self.heading}" if self.heading else self.title
        return f"{head}\n{self.text}" if head else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buf = ""
    for sent in split_sentences(text) or [text]:
        while len(sent) > max_chars:  # pathological run-on text (OCR without punctuation)
            cut = sent.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append((buf + " " + sent[:cut]).strip() if buf else sent[:cut])
            buf = ""
            sent = sent[cut:].strip()
        if buf and len(buf) + 1 + len(sent) > max_chars:
            pieces.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _split_table(text: str, max_chars: int) -> list[str]:
    rows = text.split("\n")
    out, buf = [], []
    for row in rows:
        if buf and sum(len(r) + 1 for r in buf) + len(row) > max_chars:
            out.append("\n".join(buf))
            buf = []
        buf.append(row[: max_chars * 2])
    if buf:
        out.append("\n".join(buf))
    return out


def chunk_pages(
    pages: list[dict],
    titles: dict[str, str] | None = None,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    """pages: records with doc_id, page, page_label, blocks=[{kind,text}] (or final_text)."""
    titles = titles or {}
    chunks: list[Chunk] = []
    last_heading: dict[str, str] = {}
    for rec in sorted(pages, key=lambda r: (r["doc_id"], r["page"])):
        doc_id, page = rec["doc_id"], rec["page"]
        blocks = rec.get("blocks") or [{"kind": "paragraph", "text": rec.get("final_text", "")}]
        heading = last_heading.get(doc_id, "")
        units: list[tuple[str, str]] = []  # (heading, text)
        buf = ""
        for b in blocks:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if b.get("kind") == "heading":
                if buf:
                    units.append((heading, buf))
                    buf = ""
                heading = text[:200]
                continue
            parts = (
                _split_table(text, max_chars) if b.get("kind") == "table" else _split_long(text, max_chars)
            )
            for part in parts:
                if buf and len(buf) + 2 + len(part) > max_chars:
                    units.append((heading, buf))
                    buf = part
                else:
                    buf = f"{buf}\n{part}".strip() if buf else part
        if buf:
            units.append((heading, buf))
        last_heading[doc_id] = heading
        # merge tiny trailing units into the previous unit on the same page
        merged: list[tuple[str, str]] = []
        for h, t in units:
            if merged and len(t) < min_chars and len(merged[-1][1]) + len(t) + 1 <= max_chars * 1.2:
                merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + t)
            else:
                merged.append((h, t))
        for i, (h, t) in enumerate(merged):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:p{page}:c{i}",
                    doc_id=doc_id,
                    page=page,
                    page_label=rec.get("page_label"),
                    heading=h,
                    text=t,
                    title=titles.get(doc_id, ""),
                )
            )
    return chunks
