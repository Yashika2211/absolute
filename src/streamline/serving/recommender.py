"""Request-time recommendation: Redis features -> user tower -> FAISS -> LightGBM.

This is the online twin of the Phase 3 offline evaluation. Candidates and
user-item features come from the same code (training/candidates.py,
training/ranker_features.py); user/item feature values come from the online
store instead of the offline engine. tests/test_serving_parity.py proves the
ranker sees identical inputs on both paths.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import polars as pl
import torch

from streamline.features.online import OnlineStore
from streamline.training.ann import AnnIndex
from streamline.training.candidates import Retriever
from streamline.training.ranker import Ranker
from streamline.training.ranker_features import user_item_features
from streamline.training.two_tower import TwoTowerRecommender

HISTORY_SCHEMA: dict[str, pl.DataType] = {
    "request_id": pl.UInt32(),
    "user_id": pl.Int64(),
    "as_of_ms": pl.Int64(),
    "hist_ts": pl.List(pl.Int64),
    "hist_items": pl.List(pl.Int64),
    "hist_events": pl.List(pl.Utf8),
}


@dataclass
class Recommendation:
    user_id: int
    as_of_ms: int
    items: list[int]
    scores: list[float]
    source: str  # "ranked" or "popular" (cold start)
    n_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)


class _Timer:
    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, stage: str) -> None:
        now = time.perf_counter()
        self.marks[stage] = (now - self._last) * 1000
        self._last = now


class ServingRecommender:
    def __init__(
        self, store: OnlineStore, retriever: Retriever, ranker: Ranker, popular: list[int]
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.ranker = ranker
        self.popular = popular

    @classmethod
    def from_artifacts(cls, path: Path, store: OnlineStore) -> ServingRecommender:
        # Per-request work is tiny; one thread per worker avoids OpenMP pools from
        # torch, FAISS and LightGBM oversubscribing the cores across workers.
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        model = TwoTowerRecommender.load(path / "two_tower")
        retriever = Retriever.from_index(model, AnnIndex.load(path / "ann"))
        popular = json.loads((path / "popular.json").read_text())
        return cls(store, retriever, Ranker.load(path / "ranker"), popular)

    def features(self, user_id: int, as_of_ms: int, timer: _Timer | None = None) -> pl.DataFrame:
        """Ranker input rows (one per candidate) for this user at as_of_ms.

        Empty frame when the user has no history (cold start).
        """
        timer = timer or _Timer()
        [(ts, items, events)] = self.store.user_histories([user_id], as_of_ms)
        if not items:
            return pl.DataFrame()
        [user_feats] = self.store.user_features([user_id], as_of_ms)
        timer.lap("redis_user")

        cands = self.retriever.candidates([0], [items], [events])
        timer.lap("retrieval")

        item_rows = self.store.item_features(cands["item_id"].to_list(), as_of_ms)
        timer.lap("redis_items")

        request = pl.DataFrame(
            {
                "request_id": [0],
                "user_id": [user_id],
                "as_of_ms": [as_of_ms],
                "hist_ts": [ts],
                "hist_items": [items],
                "hist_events": [events],
            },
            schema=HISTORY_SCHEMA,
        )
        frame = (
            cands.with_columns(
                pl.lit(user_id, pl.Int64).alias("user_id"),
                pl.lit(as_of_ms, pl.Int64).alias("as_of_ms"),
            )
            .hstack(pl.DataFrame(item_rows))
            .with_columns(pl.lit(v, pl.Int64).alias(k) for k, v in user_feats.items())
            .join(
                user_item_features(request),
                on=["request_id", "item_id"],
                how="left",
                maintain_order="left",
            )
            .with_columns(pl.col("ui_is_last").fill_null(0))
        )
        timer.lap("features")
        return frame

    def recommend(self, user_id: int, as_of_ms: int, k: int = 10) -> Recommendation:
        start = time.perf_counter()
        timer = _Timer()
        frame = self.features(user_id, as_of_ms, timer)
        if frame.is_empty():
            timer.lap("redis_user")
            rec = Recommendation(user_id, as_of_ms, self.popular[:k], [], "popular", 0)
        else:
            scores = self.ranker.predict(frame, num_threads=1)
            ranked = (
                frame.select("item_id")
                .with_columns(score=pl.Series(scores))
                .sort(["score", "item_id"], descending=[True, False])
                .head(k)
            )
            timer.lap("rank")
            rec = Recommendation(
                user_id,
                as_of_ms,
                ranked["item_id"].to_list(),
                [round(s, 6) for s in ranked["score"].to_list()],
                "ranked",
                frame.height,
            )
        rec.timings_ms = {**timer.marks, "total": (time.perf_counter() - start) * 1000}
        return rec
