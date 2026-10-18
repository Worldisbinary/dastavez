"""FastAPI service: POST /ask returns a verified answer, page citations and page snippets."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .answer import Answerer
from .config import get_settings
from .retrieve import MODES

app = FastAPI(
    title="Dastavez", version=__version__, description="Citation-verified RAG over Indian scheme documents"
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: str = Field("hybrid_rerank", description=f"one of {MODES}")
    k: int = Field(5, ge=1, le=20)


class CitationOut(BaseModel):
    doc_id: str
    page: int
    title: str = ""
    source_url: str = ""
    snippet: str = ""


class SentenceOut(BaseModel):
    text: str
    citation: CitationOut | None
    support: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    not_found: bool
    mode: str
    generator: str
    sentences: list[SentenceOut]
    citations: list[CitationOut]
    dropped: list[dict]
    gate: dict


@lru_cache(maxsize=1)
def _doc_meta() -> dict[str, dict]:
    from .corpus import load_manifest

    try:
        docs = load_manifest(get_settings().corpus_dir / "manifest.json")
    except OSError:
        return {}
    return {d.doc_id: {"title": d.title, "source_url": d.source_url} for d in docs}


@lru_cache(maxsize=1)
def _default_answerer() -> Answerer:
    return Answerer.from_settings(get_settings())


def get_answerer() -> Answerer:
    try:
        return _default_answerer()
    except FileNotFoundError as exc:
        raise HTTPException(503, "index not built; run `dastavez ingest`") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, answerer: Answerer = Depends(get_answerer)) -> AskResponse:
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    res = answerer.ask(req.question, mode=req.mode, k=req.k)
    meta = _doc_meta()

    def cite(doc_id: str, page: int, snippet: str = "") -> CitationOut:
        m = meta.get(doc_id, {})
        return CitationOut(
            doc_id=doc_id,
            page=page,
            title=m.get("title", ""),
            source_url=m.get("source_url", ""),
            snippet=snippet,
        )

    sentences = [
        SentenceOut(
            text=s.sentence,
            citation=cite(s.supporting.doc_id, s.supporting.page, s.span) if s.supporting else None,
            support=s.reason,
            score=s.score,
        )
        for s in res.sentences
    ]
    snippets = {(s.citation.doc_id, s.citation.page): s.citation.snippet for s in sentences if s.citation}
    return AskResponse(
        question=res.question,
        answer=res.answer,
        not_found=res.not_found,
        mode=res.mode,
        generator=res.generator,
        sentences=sentences,
        citations=[cite(d, p, snippets.get((d, p), "")) for d, p in res.citations],
        dropped=[{"text": s.sentence, "reason": s.reason} for s in res.dropped],
        gate=res.gate,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return resources.files("dastavez").joinpath("web/index.html").read_text(encoding="utf-8")
