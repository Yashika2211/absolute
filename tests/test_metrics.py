import math

import pytest

from streamline.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

RECS = [5, 3, 9, 1, 7]


def test_recall() -> None:
    assert recall_at_k(RECS, {3, 7, 42}, k=5) == pytest.approx(2 / 3)
    assert recall_at_k(RECS, {3, 7, 42}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(RECS, {42}, k=5) == 0.0
    assert recall_at_k(RECS, set(), k=5) == 0.0


def test_ndcg_hand_computed() -> None:
    # hits at ranks 2 and 5 -> DCG = 1/log2(3) + 1/log2(6); ideal = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RECS, {3, 7}, k=5) == pytest.approx(expected)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k(RECS, {5, 3}, k=5) == pytest.approx(1.0)
    assert ndcg_at_k(RECS, {42}, k=5) == 0.0
    # more relevant items than k: ideal is capped at k positions
    assert ndcg_at_k([1, 2], {1, 2, 3, 4}, k=2) == pytest.approx(1.0)


def test_mrr() -> None:
    assert mrr_at_k(RECS, {9, 7}, k=5) == pytest.approx(1 / 3)
    assert mrr_at_k(RECS, {7}, k=4) == 0.0


def test_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RECS, {1}, k=0)
