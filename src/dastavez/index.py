"""Build / load the retrieval index from parsed pages."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from .chunking import chunk_pages
from .corpus import load_manifest
from .embeddings import Embedder, make_embedder
from .ingest import load_pages
from .store import MemoryStore, PgVectorStore

log = logging.getLogger(__name__)


class EmbeddingCache:
    """Disk cache {sha256(model + text) -> vector}, so rebuilding the index is cheap."""

    def __init__(self, folder: Path, model_name: str):
        slug = hashlib.sha1(model_name.encode()).hexdigest()[:10]
        self.keys_path = folder / f"{slug}.keys.json"
        self.vecs_path = folder / f"{slug}.npy"
        self.model_name = model_name
        self.keys: list[str] = []
        self.vecs: np.ndarray | None = None
        if self.keys_path.exists() and self.vecs_path.exists():
            self.keys = json.loads(self.keys_path.read_text())
            self.vecs = np.load(self.vecs_path)
        self.pos = {k: i for i, k in enumerate(self.keys)}

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\x00{text}".encode()).hexdigest()

    def embed(self, embedder: Embedder, texts: list[str], batch: int = 64) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.pos]
        for start in range(0, len(missing), batch):
            idx = missing[start : start + batch]
            new = embedder.embed_documents([texts[i] for i in idx])
            self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])
            for i in idx:
                self.pos[keys[i]] = len(self.keys)
                self.keys.append(keys[i])
            self.save()
            log.info("embedded %d/%d new texts", min(start + batch, len(missing)), len(missing))
        return (
            np.stack([self.vecs[self.pos[k]] for k in keys])
            if keys
            else np.zeros((0, embedder.dim), np.float32)
        )

    def save(self) -> None:
        self.keys_path.parent.mkdir(parents=True, exist_ok=True)
        self.keys_path.write_text(json.dumps(self.keys))
        np.save(self.vecs_path, self.vecs)


def _pages_for_source(pages: list[dict], text_source: str) -> list[dict]:
    if text_source == "cascade":
        return pages
    if text_source == "pypdf":
        return [
            {
                **p,
                "blocks": [{"kind": "paragraph", "text": p.get("pypdf_text", "")}],
                "final_text": p.get("pypdf_text", ""),
            }
            for p in pages
        ]
    raise ValueError(text_source)


def index_dir(settings, text_source: str = "cascade") -> Path:
    return settings.data_dir / "index" / text_source


def build_index(settings, text_source: str = "cascade", embedder: Embedder | None = None) -> dict:
    t0 = time.time()
    docs = load_manifest(settings.corpus_dir / "manifest.json")
    titles = {d.doc_id: d.title for d in docs}
    pages = load_pages(settings.data_dir)
    if not pages:
        raise RuntimeError("no parsed pages; run `dastavez ingest` first")
    chunks = chunk_pages(_pages_for_source(pages, text_source), titles)
    embedder = embedder or make_embedder(settings)
    cache = EmbeddingCache(settings.data_dir / "emb_cache", embedder.name)
    emb = cache.embed(embedder, [c.embed_text() for c in chunks])
    store = MemoryStore(chunks, emb)
    meta = {
        "text_source": text_source,
        "embedder": embedder.name,
        "dim": int(emb.shape[1]) if emb.size else embedder.dim,
        "chunks": len(chunks),
        "pages": len(pages),
        "built_seconds": round(time.time() - t0, 1),
    }
    store.save(index_dir(settings, text_source), meta)
    if settings.database_url and text_source == "cascade":
        pg = PgVectorStore(settings.database_url, meta["dim"])
        pg.create_schema(drop=True)
        pg.upsert(chunks, emb)
        meta["pgvector_rows"] = pg.count()
        pg.close()
    return meta
