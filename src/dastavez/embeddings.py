"""Dense embedders.

* ``FastEmbedEmbedder`` - local ONNX models via fastembed (default:
  ``intfloat/multilingual-e5-large``, 1024-d, Hindi + English). e5 models need the
  ``query: `` / ``passage: `` prefixes, which are added here.
* ``GeminiEmbedder`` - Google ``gemini-embedding-001`` over REST when GEMINI_API_KEY is set.
* ``HashingEmbedder`` - deterministic feature hashing of tokens and character n-grams. No
  semantics beyond surface overlap, but it runs offline in milliseconds, so tests and CI do
  not download a 2 GB model.

All embedders return L2-normalised float32 vectors, so cosine similarity == dot product.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from .textutil import nfc, tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = tokenize(text)
        norm = nfc(text).casefold()
        feats += [norm[i : i + 4] for i in range(0, max(0, len(norm) - 3))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(self, model: str, cache_dir: str | None = None, batch_size: int = 16):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model, cache_dir=cache_dir)
        self.name = model
        self.batch_size = batch_size
        self._e5 = "e5" in model.lower()
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return _normalize(np.stack(list(self._model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        q = f"query: {text}" if self._e5 else text
        return _normalize(np.stack(list(self._model.embed([q]))))[0]


class GeminiEmbedder:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768):
        self.api_key, self.model, self.dim = api_key, model, dim
        self.name = f"gemini:{model}:{dim}"

    def _call(self, texts: list[str], task: str) -> np.ndarray:
        import httpx

        out = []
        for i in range(0, len(texts), 100):
            body = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": self.dim,
                    }
                    for t in texts[i : i + 100]
                ]
            }
            r = httpx.post(
                self.URL.format(model=self.model),
                json=body,
                headers={"x-goog-api-key": self.api_key},
                timeout=120,
            )
            r.raise_for_status()
            out.extend(e["values"] for e in r.json()["embeddings"])
        return _normalize(np.array(out, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._call(texts, "RETRIEVAL_DOCUMENT") if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "RETRIEVAL_QUERY")[0]


def make_embedder(settings) -> Embedder:
    choice = (settings.embedder or "auto").lower()
    if choice in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder(settings.embed_model, cache_dir=str(settings.model_cache))
        except Exception as exc:
            if choice == "fastembed":
                raise
            log.warning("fastembed unavailable (%s); falling back", exc)
    if choice in ("auto", "gemini") and settings.gemini_api_key:
        return GeminiEmbedder(settings.gemini_api_key, settings.gemini_embed_model)
    if choice == "gemini":
        raise RuntimeError("DASTAVEZ_EMBEDDER=gemini but GEMINI_API_KEY is not set")
    log.warning("using the deterministic hashing embedder (no semantic model available)")
    return HashingEmbedder()
