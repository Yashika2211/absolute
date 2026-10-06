import numpy as np
import polars as pl
import pytest

from conftest import random_events
from streamline.features.definitions import ITEM_FEATURES, USER_FEATURES
from streamline.training.ann import AnnConfig
from streamline.training.candidates import Retriever
from streamline.training.ranker_features import FEATURES, add_labels, build_features
from streamline.training.requests import sample_requests
from streamline.training.two_tower import TwoTowerConfig, TwoTowerRecommender


@pytest.fixture(scope="module")
def setup() -> tuple[pl.DataFrame, Retriever]:
    events = random_events(11, n_users=40, n_items=60, n=4000)
    cut = int(events["ts_ms"].quantile(0.6))  # type: ignore[arg-type]
    model = TwoTowerRecommender(TwoTowerConfig(dim=8, batch_size=64, epochs=2, min_item_count=1))
    model.fit(events.filter(pl.col("ts_ms") < cut))
    return events, Retriever(model, AnnConfig(kind="flat"))


def _requests(events: pl.DataFrame, seed: int = 0):  # type: ignore[no-untyped-def]
    lo = int(events["ts_ms"].quantile(0.6))  # type: ignore[arg-type]
    hi = int(events["ts_ms"].max()) + 1  # type: ignore[arg-type]
    return sample_requests(events, lo, hi, n=25, seed=seed)


def test_candidates_union_ann_and_recent(setup) -> None:  # type: ignore[no-untyped-def]
    events, retriever = setup
    reqs = _requests(events)
    f = reqs.frame
    cands = retriever.candidates(
        f["request_id"].to_list(), f["hist_items"].to_list(), f["hist_events"].to_list(), k=10
    )
    assert cands.select("request_id", "item_id").is_unique().all()
    for rid, items in zip(f["request_id"], f["hist_items"], strict=True):
        c = cands.filter(pl.col("request_id") == rid)
        assert set(items) <= set(c["item_id"])  # every recent item is a candidate
        ann = c.drop_nulls("ann_rank").sort("ann_rank")
        assert ann.height == 10
        assert ann["ann_score"].is_sorted(descending=True)
        recent = c.drop_nulls("recent_rank").sort("recent_rank")
        assert recent["item_id"][0] == items[-1]  # most recent first


def test_scores_nan_outside_vocabulary(setup) -> None:  # type: ignore[no-untyped-def]
    _, retriever = setup
    vecs = retriever.user_vectors([[1, 2]], [["view", "view"]])
    s = retriever.scores(vecs, np.array([0, 0]), np.array([1, 10**9]))
    assert np.isfinite(s[0]) and np.isnan(s[1])


def test_features_and_labels(setup) -> None:  # type: ignore[no-untyped-def]
    events, retriever = setup
    reqs = _requests(events)
    f = reqs.frame
    cands = retriever.candidates(
        f["request_id"].to_list(), f["hist_items"].to_list(), f["hist_events"].to_list(), k=10
    )
    feats = add_labels(build_features(cands, f, events), reqs.relevant)
    assert feats.height == cands.height
    assert set(FEATURES) <= set(feats.columns)
    # labels match the relevant sets exactly
    for rid, rel in enumerate(reqs.relevant):
        pos = feats.filter((pl.col("request_id") == rid) & (pl.col("label") == 1))["item_id"]
        assert set(pos) == rel & set(feats.filter(pl.col("request_id") == rid)["item_id"])
    # user-item features agree with the history lists
    row = feats.filter(pl.col("ui_is_last") == 1).row(0, named=True)
    hist = f.filter(pl.col("request_id") == row["request_id"]).row(0, named=True)
    assert hist["hist_items"][-1] == row["item_id"]
    assert row["ui_events"] == hist["hist_items"].count(row["item_id"])
    assert row["ui_age_ms"] == hist["as_of_ms"] - hist["hist_ts"][-1]


def test_features_ignore_events_at_or_after_request_time(setup) -> None:  # type: ignore[no-untyped-def]
    """Leakage test: perturb everything at/after each request's as_of; features must not move."""
    events, retriever = setup
    reqs = _requests(events)
    f = reqs.frame
    cands = retriever.candidates(
        f["request_id"].to_list(), f["hist_items"].to_list(), f["hist_events"].to_list(), k=10
    )
    base = build_features(cands, f, events)
    t_min = int(f["as_of_ms"].min())  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    noise = (
        pl.DataFrame(
            {
                "ts_ms": rng.integers(t_min, t_min + 10**8, 3000),
                "user_id": rng.integers(0, 40, 3000),
                "item_id": rng.integers(0, 60, 3000),
            }
        )
        .with_columns(
            event=pl.lit("transaction").cast(events.schema["event"]),
            transaction_id=pl.lit(None, pl.Int64),
        )
        .select(events.columns)
    )
    for rid in f["request_id"].to_list():
        t = int(f.filter(pl.col("request_id") == rid)["as_of_ms"][0])
        perturbed = pl.concat([events, noise.filter(pl.col("ts_ms") >= t)])
        one = cands.filter(pl.col("request_id") == rid)
        req = f.filter(pl.col("request_id") == rid)
        got = build_features(one, req, perturbed)
        want = base.filter(pl.col("request_id") == rid)
        assert got.select(ITEM_FEATURES + USER_FEATURES).equals(
            want.select(ITEM_FEATURES + USER_FEATURES)
        ), rid
