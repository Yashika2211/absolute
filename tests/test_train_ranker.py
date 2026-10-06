import polars as pl

from streamline.training.requests import RequestSet
from streamline.training.train_ranker import recent_then_popular, results_table


def test_recent_then_popular_at_request_time() -> None:
    frame = pl.DataFrame(
        {"request_id": [0, 1], "hist_items": [[5, 6, 5, 7], []]},
        schema={"request_id": pl.UInt32, "hist_items": pl.List(pl.Int64)},
    )
    reqs = RequestSet(frame=frame, relevant=[{1}, {2}], seen=[set(), set()])
    out = recent_then_popular(reqs, popular=[7, 1, 2, 3], k=4)
    assert out[0] == [7, 5, 6, 1]  # most recent distinct first, popularity backfill
    assert out[1] == [7, 1, 2, 3]


def test_results_table() -> None:
    m = {
        "recall@50": 0.5,
        "ndcg@10": 0.25,
        "mrr@50": 0.2,
        "new_recall@50": 0.1,
        "new_ndcg@10": 0.05,
    }
    table = results_table({"A": m})
    assert table.splitlines()[2] == "| A | 0.5000 | 0.2500 | 0.2000 | 0.1000 | 0.0500 |"


def test_pruned_features_only_drops_cheap_to_skip_item_windows() -> None:
    from streamline.features.definitions import ITEM_FEATURES
    from streamline.training.ranker_features import FEATURES
    from streamline.training.train_ranker import pruned_features

    importance = dict.fromkeys(FEATURES, 0.0) | {"item_views_24h": 0.01}
    kept = pruned_features(importance)
    assert [f for f in kept if f in ITEM_FEATURES] == ["item_views_24h"]
    assert set(FEATURES) - set(kept) == set(ITEM_FEATURES) - {"item_views_24h"}
