from collections.abc import Sequence

import polars as pl
import pytest

from conftest import make_events
from streamline.eval.harness import build_eval_set, evaluate, score


class Oracle:
    """Cheats by reading the target; used to check the harness wiring."""

    name = "oracle"

    def __init__(self, answers: dict[int, list[int]]) -> None:
        self.answers = answers

    def fit(self, history: pl.DataFrame) -> None:
        pass

    def recommend(self, user_ids: Sequence[int], k: int) -> list[list[int]]:
        return [self.answers.get(u, [])[:k] for u in user_ids]


HISTORY = make_events([(0, 1, 10, "view"), (1, 2, 11, "view")])
TARGET = make_events(
    [
        (5, 1, 20, "view"),
        (6, 1, 21, "view"),
        (7, 1, 20, "addtocart"),
        (8, 2, 22, "view"),
        (9, 3, 23, "view"),  # cold user: no history
    ]
)


def test_build_eval_set_warm_users_only() -> None:
    es = build_eval_set(HISTORY, TARGET)
    assert es.user_ids == [1, 2]
    assert es.relevant == [{20, 21}, {22}]
    assert es.n_cold_users == 1


def test_evaluate_oracle_is_perfect() -> None:
    m = evaluate(Oracle({1: [20, 21], 2: [22]}), HISTORY, TARGET)
    assert m["recall@50"] == pytest.approx(1.0)
    assert m["ndcg@10"] == pytest.approx(1.0)
    assert m["mrr@50"] == pytest.approx(1.0)
    assert m["eval_users"] == 2


def test_score_partial() -> None:
    es = build_eval_set(HISTORY, TARGET)
    m = score([[99, 20], []], es, ks=[10])
    assert m["recall@10"] == pytest.approx((0.5 + 0.0) / 2)
    assert m["mrr@10"] == pytest.approx((0.5 + 0.0) / 2)


def test_score_length_mismatch() -> None:
    with pytest.raises(ValueError):
        score([[1]], build_eval_set(HISTORY, TARGET), ks=[10])
