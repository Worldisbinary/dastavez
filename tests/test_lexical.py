import math

import pytest

from dastavez.lexical import BM25, rrf


def _tok(s: str) -> list[str]:
    return s.split()


def test_bm25_matches_hand_computation():
    docs = ["a b", "a c c", "d"]
    bm = BM25(docs, k1=1.5, b=0.75, tokenizer=_tok)
    assert bm.N == 3 and bm.avgdl == pytest.approx(2.0)
    # idf(c) = ln((3 - 1 + 0.5) / (1 + 0.5) + 1) = ln(2.6667)
    idf_c = math.log(2.5 / 1.5 + 1)
    assert bm.idf("c") == pytest.approx(idf_c)
    # doc 1: tf(c)=2, |d|=3 -> 2*2.5 / (2 + 1.5*(0.25 + 0.75*1.5))
    expected = idf_c * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
    assert bm.score_doc(["c"], 1) == pytest.approx(expected)
    res = bm.search("c", k=5)
    assert res[0][0] == 1 and res[0][1] == pytest.approx(expected)
    assert all(i != 0 and i != 2 for i, _ in res)  # docs without the term are not returned


def test_bm25_prefers_rarer_terms_and_shorter_docs():
    docs = ["scheme farmer income", "scheme scheme scheme loan vendor street", "scheme"]
    bm = BM25(docs, tokenizer=_tok)
    assert bm.idf("farmer") > bm.idf("scheme")
    assert bm.search("vendor", 1)[0][0] == 1


def test_bm25_default_tokenizer_on_hindi():
    bm = BM25(["आवास योजना शहरी", "किसान सम्मान निधि"])
    assert bm.search("शहरी आवास", 1)[0][0] == 0


def test_rrf_matches_hand_computation():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in fused] == ["b", "c", "a", "d"]


def test_rrf_ties_are_deterministic_and_weights_apply():
    fused = rrf([["x"], ["y"]], k=60)
    assert [d for d, _ in fused] == ["x", "y"]  # equal score and best rank -> id order
    weighted = rrf([["x"], ["y"]], k=60, weights=[1.0, 2.0])
    assert weighted[0][0] == "y"
