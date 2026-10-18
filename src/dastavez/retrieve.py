"""Hybrid retrieval: BM25 + dense, fused with RRF, optionally reranked by a cross-encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .chunking import Chunk
from .embeddings import Embedder, HashingEmbedder, make_embedder
from .lexical import rrf
from .rerank import Reranker, make_reranker
from .store import MemoryStore, PgVectorStore

log = logging.getLogger(__name__)

MODES = ("lexical", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "page": self.chunk.page,
            "page_label": self.chunk.page_label,
            "heading": self.chunk.heading,
            "score": round(self.score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "ranks": self.ranks,
        }


class Retriever:
    def __init__(
        self,
        store,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 30,
        rerank_top: int = 20,
        rrf_k: int = 60,
    ):
        self.store, self.embedder, self.reranker = store, embedder, reranker
        self.candidates, self.rerank_top, self.rrf_k = candidates, rerank_top, rrf_k

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = max(k, self.candidates)
        lex = self.store.lexical(query, n) if mode != "dense" else []
        den = self.store.dense(self.embedder.embed_query(query), n) if mode != "lexical" else []
        lex_rank = {cid: r for r, (cid, _) in enumerate(lex, start=1)}
        den_rank = {cid: r for r, (cid, _) in enumerate(den, start=1)}
        if mode == "lexical":
            fused = lex
        elif mode == "dense":
            fused = den
        else:
            fused = rrf([[c for c, _ in lex], [c for c, _ in den]], k=self.rrf_k)
        ids = [c for c, _ in fused]
        scores = dict(fused)
        if mode == "hybrid_rerank" and self.reranker is not None and ids:
            pool = self.store.get(ids[: self.rerank_top])
            rr = self.reranker.score(query, [c.embed_text() for c in pool])
            order = sorted(zip(pool, rr, strict=True), key=lambda x: (-x[1], ids.index(x[0].chunk_id)))
            return [
                Hit(c, s, {"lexical": lex_rank.get(c.chunk_id, 0), "dense": den_rank.get(c.chunk_id, 0)}, s)
                for c, s in order[:k]
            ]
        chunks = self.store.get(ids[:k])
        return [
            Hit(
                c,
                scores[c.chunk_id],
                {"lexical": lex_rank.get(c.chunk_id, 0), "dense": den_rank.get(c.chunk_id, 0)},
            )
            for c in chunks
        ]

    @classmethod
    def from_settings(cls, settings, text_source: str = "cascade") -> Retriever:
        from .index import index_dir

        mem, meta = MemoryStore.load(index_dir(settings, text_source))
        embedder: Embedder
        if meta["embedder"].startswith("hashing-"):
            embedder = HashingEmbedder(int(meta["embedder"].split("-")[1]))
        else:
            embedder = make_embedder(settings)
            if embedder.name != meta["embedder"]:
                raise RuntimeError(
                    f"index was built with {meta['embedder']!r} but the configured embedder is {embedder.name!r}; "
                    "rebuild with `dastavez index`"
                )
        store = mem
        if settings.database_url and text_source == "cascade":
            store = PgVectorStore(settings.database_url, meta["dim"])
        reranker = make_reranker(settings, idf=mem.idf())
        return cls(store, embedder, reranker)
