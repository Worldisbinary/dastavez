"""Evaluation metrics (pure functions, unit-tested against hand-computed values).

Text fidelity (both strings pass through ``normalize_for_eval`` first):
* similarity = 1 - Indel(h, r) / (|h| + |r|)        (rapidfuzz ``fuzz.ratio`` / 100)
  where Indel counts insertions + deletions only (a substitution costs 2).
* CER        = Levenshtein(h, r) / |r|              (can exceed 1 when h is much longer)

Retrieval (page level; a chunk "hits" if its (doc_id, page) is a gold page):
* recall@k   = |gold pages within the top-k chunks| / |gold pages|
* hit@k      = 1 if any gold page is within the top-k chunks
* MRR        = 1 / rank of the first chunk on a gold page (0 if none in the list)
* citation precision = |cited pages that are gold| / |cited pages|
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable, Sequence

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .textutil import normalize_for_eval

Page = tuple[str, int]


def similarity(hyp: str, ref: str, normalize: bool = True) -> float:
    if normalize:
        hyp, ref = normalize_for_eval(hyp), normalize_for_eval(ref)
    if not hyp and not ref:
        return 1.0
    return fuzz.ratio(hyp, ref) / 100.0


def cer(hyp: str, ref: str, normalize: bool = True) -> float:
    if normalize:
        hyp, ref = normalize_for_eval(hyp), normalize_for_eval(ref)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(hyp, ref) / len(ref)


def agreement(a: str, b: str) -> dict:
    """Inter-annotator agreement for one page: symmetric similarity plus CER each way."""
    return {
        "similarity": similarity(a, b),
        "cer_a_vs_b": cer(a, b),  # a as hypothesis, b as reference
        "cer_b_vs_a": cer(b, a),
        "mean_cer": (cer(a, b) + cer(b, a)) / 2,
    }


def _dedupe(pages: Iterable[Page]) -> list[Page]:
    seen, out = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def recall_at_k(retrieved: Sequence[Page], gold: set[Page], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: Sequence[Page], gold: set[Page], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def reciprocal_rank(retrieved: Sequence[Page], gold: set[Page]) -> float:
    for i, p in enumerate(retrieved, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


def citation_precision(cited: Sequence[Page], gold: set[Page]) -> float | None:
    cited = _dedupe(cited)
    if not cited:
        return None
    return sum(p in gold for p in cited) / len(cited)


def mean(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def bootstrap_ci(
    xs: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% percentile bootstrap interval of ``stat`` over items (fixed seed => reproducible)."""
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    boots = sorted(stat([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return boots[int(0.025 * n)], boots[int(0.975 * n) - 1]
