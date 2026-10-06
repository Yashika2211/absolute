"""Training/serving skew for the ranker: the serving path (Redis) and the offline
evaluation path (feature engine over the event log) must produce identical ranker
inputs and identical recommendations for the same user and time."""

import fakeredis
import numpy as np
import polars as pl
import pytest

from conftest import random_events
from streamline.features.online import OnlineStore
from streamline.ingest.simulator import iter_events
from streamline.serving.recommender import ServingRecommender
from streamline.training.ann import AnnConfig
from streamline.training.candidates import Retriever
from streamline.training.ranker import Ranker, RankerConfig
from streamline.training.ranker_features import FEATURES, add_labels, build_features
from streamline.training.requests import sample_requests
from streamline.training.two_tower import TwoTowerConfig, TwoTowerRecommender


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    events = random_events(21, n_users=40, n_items=60, n=4000)
    cut = int(events["ts_ms"].quantile(0.5))  # type: ignore[arg-type]
    mid = int(events["ts_ms"].quantile(0.75))  # type: ignore[arg-type]
    end = int(events["ts_ms"].max()) + 1  # type: ignore[arg-type]
    model = TwoTowerRecommender(TwoTowerConfig(dim=8, batch_size=64, epochs=2, min_item_count=1))
    model.fit(events.filter(pl.col("ts_ms") < cut))
    retriever = Retriever(model, AnnConfig(kind="flat"))

    def offline(lo: int, hi: int, seed: int):  # type: ignore[no-untyped-def]
        reqs = sample_requests(events, lo, hi, n=30, seed=seed)
        f = reqs.frame
        cands = retriever.candidates(
            f["request_id"].to_list(), f["hist_items"].to_list(), f["hist_events"].to_list()
        )
        frame = add_labels(build_features(cands, f, events), reqs.relevant)
        return reqs, frame.sort(["request_id", "item_id"])

    _, train = offline(cut, mid, 0)
    ranker = Ranker(RankerConfig(n_estimators=30, min_data_in_leaf=5, early_stopping_rounds=10))
    ranker.fit(train, train)
    test_reqs, test = offline(mid, end, 1)
    return events, retriever, ranker, test_reqs, test


def _serving_for(events: pl.DataFrame, retriever, ranker, as_of: int, prefix: str):  # type: ignore[no-untyped-def]
    store = OnlineStore(fakeredis.FakeRedis(), prefix=prefix)
    store.write_events(iter_events(events.filter(pl.col("ts_ms") < as_of)))
    return ServingRecommender(store, retriever, ranker, popular=[1, 2, 3])


def test_serving_features_match_offline(world) -> None:  # type: ignore[no-untyped-def]
    events, retriever, ranker, reqs, offline = world
    checked = 0
    for row in reqs.frame.iter_rows(named=True):
        rid, user, t = row["request_id"], row["user_id"], row["as_of_ms"]
        serving = _serving_for(events, retriever, ranker, t, prefix=f"p{rid}")
        online = serving.features(user, t).sort("item_id")
        want = offline.filter(pl.col("request_id") == rid).sort("item_id")
        assert online["item_id"].to_list() == want["item_id"].to_list(), rid
        exact = [f for f in FEATURES if f != "ann_score"]
        a = online.select(pl.col(f).cast(pl.Float64) for f in exact).to_numpy()
        b = want.select(pl.col(f).cast(pl.Float64) for f in exact).to_numpy()
        np.testing.assert_array_equal(a, b, err_msg=f"request {rid}")
        # the two-tower score is float32: batched (offline) vs single-row (serving)
        # inference sums in a different order, so allow rounding-level differences only
        np.testing.assert_allclose(
            online["ann_score"].to_numpy(), want["ann_score"].to_numpy(), rtol=1e-5, atol=1e-6
        )

        # identical inputs -> identical ranking
        rec = serving.recommend(user, t, k=10)
        expected = ranker.rerank(want.with_columns(pl.lit(0).alias("request_id")), 1, 10)[0]
        assert rec.items == expected and rec.source == "ranked"
        checked += 1
    assert checked == len(reqs)


def test_cold_user_gets_popular(world) -> None:  # type: ignore[no-untyped-def]
    events, retriever, ranker, _, _ = world
    serving = _serving_for(events, retriever, ranker, 10**15, prefix="cold")
    rec = serving.recommend(user_id=10**6, as_of_ms=10**15, k=2)
    assert rec.source == "popular" and rec.items == [1, 2]
    assert "total" in rec.timings_ms
