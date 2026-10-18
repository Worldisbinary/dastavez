"""BM25 (Okapi) and reciprocal rank fusion, implemented from scratch.

BM25 score of document d for query q:

    score(q, d) = sum_{t in q} IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    IDF(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )        # Lucene's non-negative variant

Reciprocal rank fusion of rankings R_1..R_m:

    RRF(d) = sum_i 1 / (k + rank_i(d))        rank is 1-based; documents absent from R_i add 0
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

from .textutil import tokenize


class BM25:
    def __init__(
        self,
        docs: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ):
        self.k1, self.b, self.tokenizer = k1, b, tokenizer
        self.tfs = [Counter(tokenizer(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.N = len(docs)
        self.avgdl = (sum(self.lengths) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.df = df
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score_doc(self, query_terms: Iterable[str], i: int) -> float:
        tf, dl = self.tfs[i], self.lengths[i]
        s = 0.0
        for t in query_terms:
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            s += self.idf(t) * f * (self.k1 + 1) / denom
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        terms = list(dict.fromkeys(self.tokenizer(query)))  # unique, order-preserving
        candidates: set[int] = set()
        for t in terms:
            candidates.update(self.postings.get(t, ()))
        scored = [(i, self.score_doc(terms, i)) for i in candidates]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def rrf(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Ties are broken by best individual rank, then id (deterministic)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for r, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + r)
            best_rank[doc_id] = min(best_rank.get(doc_id, r), r)
    return sorted(scores.items(), key=lambda x: (-x[1], best_rank[x[0]], x[0]))
