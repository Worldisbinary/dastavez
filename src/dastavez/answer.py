"""Citation-first answering.

Pipeline: retrieve -> gate on evidence strength -> generate (Gemini) or extract (fallback)
-> verify every sentence against the page it cites -> drop unsupported sentences ->
"Not found in the documents." when nothing verifiable is left.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .rerank import CrossEncoderReranker
from .retrieve import Hit, Retriever
from .textutil import split_sentences, tokenize
from .verify import SentenceCheck, verify_sentences

log = logging.getLogger(__name__)

NOT_FOUND = "Not found in the documents."
DEFAULT_GATES = {"cross_encoder": 0.10, "lexical": 0.50}

PROMPT = """You answer questions about Indian government scheme documents using ONLY the sources below.
Rules:
- Every sentence MUST end with one or more citations in the exact form [doc_id p.N] copied from the source headers.
- Only state facts that are explicitly written on the cited page. Prefer the documents' own wording.
- If the sources do not contain the answer, reply with exactly: NOT_FOUND
- At most 4 sentences. Answer in the language of the question.

Question: {question}

Sources:
{sources}
"""


@dataclass
class AnswerResult:
    question: str
    answer: str
    not_found: bool
    mode: str
    generator: str
    sentences: list[SentenceCheck] = field(default_factory=list)
    dropped: list[SentenceCheck] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    gate: dict = field(default_factory=dict)

    @property
    def citations(self) -> list[tuple[str, int]]:
        out = []
        for s in self.sentences:
            c = s.supporting or (s.citations[0] if s.citations else None)
            if c and (c.doc_id, c.page) not in out:
                out.append((c.doc_id, c.page))
        return out

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "not_found": self.not_found,
            "mode": self.mode,
            "generator": self.generator,
            "gate": self.gate,
            "sentences": [
                {
                    "text": s.sentence,
                    "citation": (
                        {"doc_id": s.supporting.doc_id, "page": s.supporting.page} if s.supporting else None
                    ),
                    "support": s.reason,
                    "score": s.score,
                    "snippet": s.span,
                }
                for s in self.sentences
            ],
            "dropped": [{"text": s.sentence, "reason": s.reason} for s in self.dropped],
            "citations": [{"doc_id": d, "page": p} for d, p in self.citations],
            "retrieved": [h.to_dict() for h in self.hits],
        }

    def render(self) -> str:
        lines = [self.answer, ""]
        if self.sentences:
            lines.append("Citations:")
            for d, p in self.citations:
                lines.append(f"  - {d} p.{p}")
        if self.dropped:
            lines.append(f"({len(self.dropped)} unsupported sentence(s) dropped by the verifier)")
        return "\n".join(lines)


def _coverage(question: str, text: str, idf: dict[str, float] | None = None) -> float:
    q = list(dict.fromkeys(tokenize(question)))
    if not q:
        return 0.0
    ts = set(tokenize(text))
    w = {t: (idf or {}).get(t, 1.0) for t in q}
    return sum(w[t] for t in q if t in ts) / (sum(w.values()) or 1.0)


class GeminiGenerator:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model
        self.name = f"gemini:{model}"

    def generate(self, question: str, hits: list[Hit]) -> str:
        import httpx

        sources = "\n\n".join(
            f"[{h.chunk.doc_id} p.{h.chunk.page}] ({h.chunk.title} / {h.chunk.heading})\n{h.chunk.text}"
            for h in hits
        )
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": PROMPT.format(question=question, sources=sources)}]}
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 600},
        }
        r = httpx.post(
            self.URL.format(model=self.model), json=body, headers={"x-goog-api-key": self.api_key}, timeout=90
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()


def extractive_answer(
    question: str, hits: list[Hit], idf: dict[str, float] | None = None, max_sentences: int = 3
) -> list[str]:
    """Quote the best-matching sentences from the top chunks, each with its page citation."""
    scored = []
    for rank, h in enumerate(hits):
        for s in split_sentences(h.chunk.text.replace("\n", " ")):
            if not (25 <= len(s) <= 600):
                continue
            cov = _coverage(question, s, idf)
            scored.append((cov / (1 + 0.15 * rank), rank, s, h))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, seen = [], set()
    for score, _, s, h in scored:
        if score < 0.2 or len(out) >= max_sentences:
            break
        key = s[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{s} [{h.chunk.doc_id} p.{h.chunk.page}]")
    return out


class Answerer:
    def __init__(
        self, retriever: Retriever, page_texts: dict[tuple[str, int], str], generator=None, gates=None
    ):
        self.retriever = retriever
        self.page_texts = page_texts
        self.generator = generator
        self.gates = {**DEFAULT_GATES, **(gates or {})}
        self.idf = retriever.store.idf() if hasattr(retriever.store, "idf") else None

    def gate(self, question: str, hits: list[Hit], mode: str) -> dict:
        if not hits:
            return {"kind": "empty", "value": 0.0, "threshold": 0.0, "pass": False}
        if mode == "hybrid_rerank" and isinstance(self.retriever.reranker, CrossEncoderReranker):
            v = float(hits[0].rerank_score or 0.0)
            t = self.gates["cross_encoder"]
            return {"kind": "cross_encoder", "value": round(v, 4), "threshold": t, "pass": v >= t}
        v = max(_coverage(question, f"{h.chunk.heading} {h.chunk.text}", self.idf) for h in hits)
        t = self.gates["lexical"]
        return {"kind": "lexical", "value": round(v, 4), "threshold": t, "pass": v >= t}

    def ask(
        self, question: str, mode: str = "hybrid_rerank", k: int = 5, hits: list[Hit] | None = None
    ) -> AnswerResult:
        hits = self.retriever.search(question, mode=mode, k=k) if hits is None else hits[:k]
        gate = self.gate(question, hits, mode)
        gen_name = self.generator.name if self.generator else "extractive"
        if not gate["pass"]:
            return AnswerResult(question, NOT_FOUND, True, mode, gen_name, hits=hits, gate=gate)
        sentences: list[str]
        if self.generator is not None:
            try:
                raw = self.generator.generate(question, hits)
            except Exception as exc:  # network / quota: degrade to extractive, never to uncited text
                log.warning("generation failed (%s); using extractive fallback", exc)
                raw, gen_name = "", "extractive(fallback)"
            if raw.strip().upper().startswith("NOT_FOUND"):
                return AnswerResult(question, NOT_FOUND, True, mode, gen_name, hits=hits, gate=gate)
            sentences = _split_cited(raw) if raw else extractive_answer(question, hits, self.idf)
        else:
            sentences = extractive_answer(question, hits, self.idf)
        result = verify_sentences(sentences, self.page_texts)
        kept = result.supported
        if not kept:
            return AnswerResult(question, NOT_FOUND, True, mode, gen_name, [], result.dropped, hits, gate)
        answer = " ".join(f"{s.sentence} [{s.supporting.doc_id} p.{s.supporting.page}]" for s in kept)
        return AnswerResult(question, answer, False, mode, gen_name, kept, result.dropped, hits, gate)

    @classmethod
    def from_settings(cls, settings, text_source: str = "cascade") -> Answerer:
        from .ingest import load_pages

        retriever = Retriever.from_settings(settings, text_source)
        field_name = "final_text" if text_source == "cascade" else "pypdf_text"
        page_texts = {(p["doc_id"], p["page"]): p[field_name] for p in load_pages(settings.data_dir)}
        generator = (
            GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
            if settings.gemini_api_key
            else None
        )
        gates = load_gates(settings.results_dir / "calibration.json")
        return cls(retriever, page_texts, generator, gates)


def _split_cited(raw: str) -> list[str]:
    """Split generated text into sentences, keeping trailing citation markers attached."""
    raw = re.sub(r"\s+", " ", raw).strip()
    parts = re.split(r"(?<=\])\s+(?=[^\[])", raw)
    out = []
    for p in parts:
        out.extend(split_sentences(p) if p.count("[") > 1 and not p.rstrip().endswith("]") else [p])
    return [p for p in out if p.strip()]


def load_gates(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: float(v) for k, v in data.get("gates", {}).items()}
    except (OSError, ValueError):
        return {}
