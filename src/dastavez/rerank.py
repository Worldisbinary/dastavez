"""Rerankers applied to the fused candidate list.

* ``CrossEncoderReranker`` - joint (query, passage) scoring with a multilingual cross-encoder
  (default ``BAAI/bge-reranker-v2-m3`` int8 ONNX via fastembed). Scores are logits; we map
  them through a sigmoid so a single "is this relevant at all?" threshold can gate answers.
* ``LexicalReranker`` - light fallback when no cross-encoder can run: query-term coverage
  with IDF weights plus a bigram bonus. Returns scores in [0, 1].
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

from .textutil import tokenize

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


class CrossEncoderReranker:
    def __init__(self, model: str, cache_dir: str | None = None, batch_size: int = 8):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model, cache_dir=cache_dir)
        self.name = model
        self.batch_size = batch_size

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        logits = list(self._model.rerank(query, list(passages), batch_size=self.batch_size))
        return [_sigmoid(float(x)) for x in logits]


class LexicalReranker:
    name = "lexical-coverage"

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf = idf or {}

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = list(dict.fromkeys(tokenize(query)))
        if not q:
            return [0.0] * len(passages)
        w = {t: self.idf.get(t, 1.0) for t in q}
        total = sum(w.values()) or 1.0
        q_bigrams = set(pairwise(q))
        out = []
        for p in passages:
            toks = tokenize(p)
            ts = set(toks)
            cov = sum(w[t] for t in q if t in ts) / total
            bigrams = set(pairwise(toks))
            bonus = (len(q_bigrams & bigrams) / len(q_bigrams)) if q_bigrams else 0.0
            out.append(round(0.85 * cov + 0.15 * bonus, 6))
        return out


def make_reranker(settings, idf: dict[str, float] | None = None) -> Reranker | None:
    choice = (settings.reranker or "auto").lower()
    if choice == "none":
        return None
    if choice in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker(settings.rerank_model, cache_dir=str(settings.model_cache))
        except Exception as exc:
            if choice == "cross-encoder":
                raise
            log.warning("cross-encoder unavailable (%s); using lexical reranker", exc)
    return LexicalReranker(idf)
