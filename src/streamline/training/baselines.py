"""Non-learned baselines.

Popularity: the same top-K for everyone, scored by event-weighted counts over the
last `window_days` of history (None = all history).

RecentThenPopular: the user's own recently touched items (most recent first),
backfilled with popularity. Repeat interactions are common in e-commerce, so this
is the bar a personalised model has to clear.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from streamline.training.split import DAY_MS

EVENT_WEIGHTS: Mapping[str, float] = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


def popularity_ranking(
    history: pl.DataFrame,
    window_days: int | None = 14,
    weights: Mapping[str, float] = EVENT_WEIGHTS,
    limit: int = 1000,
) -> list[int]:
    if window_days is not None:
        end = int(history["ts_ms"].max()) + 1  # type: ignore[arg-type]
        history = history.filter(pl.col("ts_ms") >= end - window_days * DAY_MS)
    weight = pl.col("event").cast(pl.Utf8).replace_strict(dict(weights), return_dtype=pl.Float64)
    return (
        history.group_by("item_id")
        .agg(weight.sum().alias("score"), pl.col("ts_ms").max().alias("last_ts"))
        .sort(["score", "last_ts", "item_id"], descending=[True, True, False])
        .head(limit)["item_id"]
        .to_list()
    )


class Popularity:
    def __init__(self, window_days: int | None = 14) -> None:
        self.window_days = window_days
        self.name = f"popularity_{window_days or 'all'}d"
        self.ranking: list[int] = []

    def fit(self, history: pl.DataFrame) -> None:
        self.ranking = popularity_ranking(history, self.window_days)

    def recommend(self, user_ids: Sequence[int], k: int) -> list[list[int]]:
        top = self.ranking[:k]
        return [list(top) for _ in user_ids]


class RecentThenPopular:
    def __init__(self, window_days: int | None = 14, max_recent: int = 50) -> None:
        self.popularity = Popularity(window_days)
        self.max_recent = max_recent
        self.name = "recent_then_popular"
        self.recent: dict[int, list[int]] = {}

    def fit(self, history: pl.DataFrame) -> None:
        self.popularity.fit(history)
        per_user = (
            history.group_by("user_id", "item_id")
            .agg(pl.col("ts_ms").max())
            .sort(["user_id", "ts_ms", "item_id"], descending=[False, True, False])
            .group_by("user_id", maintain_order=True)
            .agg(pl.col("item_id").head(self.max_recent))
        )
        self.recent = dict(zip(per_user["user_id"], per_user["item_id"].to_list(), strict=True))

    def recommend(self, user_ids: Sequence[int], k: int) -> list[list[int]]:
        out = []
        for user in user_ids:
            recs = list(self.recent.get(user, []))[:k]
            seen = set(recs)
            for item in self.popularity.ranking:
                if len(recs) >= k:
                    break
                if item not in seen:
                    recs.append(item)
            out.append(recs)
        return out
