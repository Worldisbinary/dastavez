import pytest

from dastavez.eval_retrieval import calibrate, split_of
from dastavez.metrics import (
    agreement,
    bootstrap_ci,
    cer,
    citation_precision,
    hit_at_k,
    mean,
    recall_at_k,
    reciprocal_rank,
    similarity,
)


def test_similarity_hand_values():
    # Indel("abcd", "abce") = 2 (delete d, insert e); 1 - 2/8 = 0.75
    assert similarity("abcd", "abce") == pytest.approx(0.75)
    assert similarity("", "") == 1.0
    assert similarity("abc", "") == 0.0
    # normalisation: case, whitespace and quotes do not count as errors
    assert similarity("Hello   “World”", 'hello "world"') == 1.0


def test_cer_hand_values():
    assert cer("abce", "abcd") == pytest.approx(1 / 4)  # one substitution
    assert cer("abc", "abcd") == pytest.approx(1 / 4)  # one deletion
    assert cer("xxxxxxxx", "ab") == pytest.approx(8 / 2)  # can exceed 1
    assert cer("", "") == 0.0 and cer("a", "") == 1.0
    assert cer("नमस्ते", "नमस्ते") == 0.0


def test_agreement_is_symmetric_in_similarity():
    a, b = "the scheme pays 6000", "the scheme pays 6,000"
    ag = agreement(a, b)
    assert ag["similarity"] == pytest.approx(agreement(b, a)["similarity"])
    assert ag["cer_a_vs_b"] == pytest.approx(1 / 21)  # one deletion relative to b (21 chars)
    assert ag["cer_b_vs_a"] == pytest.approx(1 / 20)
    assert ag["mean_cer"] == pytest.approx((1 / 21 + 1 / 20) / 2)


def test_retrieval_metrics_hand_values():
    retrieved = [("d", 3), ("d", 1), ("e", 2), ("d", 1), ("f", 9)]
    gold = {("d", 1), ("f", 9)}
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert recall_at_k(retrieved, gold, 5) == 1.0
    assert hit_at_k(retrieved, gold, 1) == 0.0 and hit_at_k(retrieved, gold, 2) == 1.0
    assert reciprocal_rank(retrieved, gold) == 0.5
    assert reciprocal_rank(retrieved, {("z", 1)}) == 0.0
    assert recall_at_k(retrieved, set(), 3) == 0.0


def test_citation_precision():
    gold = {("d", 1)}
    assert citation_precision([("d", 1), ("d", 2)], gold) == 0.5
    assert citation_precision([("d", 1), ("d", 1)], gold) == 1.0  # duplicates count once
    assert citation_precision([], gold) is None


def test_mean_skips_none_and_bootstrap_is_reproducible():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None
    xs = [0.1, 0.5, 0.9, 0.7, 0.3]
    ci1, ci2 = bootstrap_ci(xs, seed=1), bootstrap_ci(xs, seed=1)
    assert ci1 == ci2 and ci1[0] <= sum(xs) / len(xs) <= ci1[1]
    assert bootstrap_ci([0.5]) is None


def test_gate_calibration_maximises_balanced_accuracy():
    vals = [(0.9, True), (0.8, True), (0.4, True), (0.3, False), (0.1, False)]
    assert calibrate(vals) == pytest.approx(0.35)  # midpoint of the separating gap
    # ties in balanced accuracy are broken by the widest margin, not the first cut
    vals = [(0.99, True), (0.995, True), (0.15, False)]
    assert calibrate(vals) == pytest.approx((0.15 + 0.99) / 2)
    # overlapping classes: best achievable cut still found
    vals = [(0.9, True), (0.2, True), (0.5, False), (0.1, False)]
    assert calibrate(vals) == pytest.approx(0.7)  # 0.15 and 0.7 tie at 0.75; 0.7 has the wider gap
    assert calibrate([]) == 0.0


def test_dev_test_split_is_deterministic():
    assert split_of("q001") == split_of("q001")
    splits = [split_of(f"q{i:03d}") for i in range(200)]
    assert 0.1 < splits.count("dev") / 200 < 0.3
