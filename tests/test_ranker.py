from pathlib import Path

import numpy as np
import polars as pl

from streamline.training.ranker import Ranker, RankerConfig, with_positives


def _frame(n_requests: int, seed: int) -> pl.DataFrame:
    """Each request has 20 candidates; the relevant ones have high `signal`."""
    rng = np.random.default_rng(seed)
    n = n_requests * 20
    signal = rng.normal(size=n)
    label = (signal + 0.3 * rng.normal(size=n) > 1.2).astype(np.int8)
    return pl.DataFrame(
        {
            "request_id": np.repeat(np.arange(n_requests), 20),
            "item_id": np.tile(np.arange(20), n_requests),
            "signal": signal,
            "noise": rng.normal(size=n),
            "label": label,
        }
    )


CFG = RankerConfig(n_estimators=200, min_data_in_leaf=20, early_stopping_rounds=20)


def test_ranker_learns_signal_and_roundtrips(tmp_path: Path) -> None:
    ranker = Ranker(CFG, features=["signal", "noise"])
    stats = ranker.fit(_frame(300, 0), _frame(100, 1))
    assert stats["valid_ndcg@10"] > 0.9
    imp = ranker.importance()
    assert next(iter(imp)) == "signal"

    test = _frame(50, 2)
    top = ranker.rerank(test, n_requests=50, k=3)
    best = test.sort("signal", descending=True).group_by("request_id", maintain_order=True).first()
    hits = sum(
        int(best.filter(pl.col("request_id") == r)["item_id"][0] in top[r]) for r in range(50)
    )
    assert hits >= 45

    ranker.save(tmp_path / "rk")
    loaded = Ranker.load(tmp_path / "rk")
    np.testing.assert_allclose(loaded.predict(test), ranker.predict(test))


def test_with_positives_drops_unlabeled_requests() -> None:
    frame = pl.DataFrame(
        {"request_id": [0, 0, 1, 1, 2], "item_id": [1, 2, 3, 4, 5], "label": [0, 1, 0, 0, 1]}
    )
    assert with_positives(frame)["request_id"].to_list() == [0, 0, 2]


def test_rerank_handles_requests_without_candidates() -> None:
    ranker = Ranker(CFG, features=["signal", "noise"])
    ranker.fit(_frame(200, 0), _frame(50, 1))
    out = ranker.rerank(_frame(3, 3), n_requests=5, k=2)
    assert [len(x) for x in out] == [2, 2, 2, 0, 0]
