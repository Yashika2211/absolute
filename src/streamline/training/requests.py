"""Point-in-time recommendation requests for ranker training and evaluation.

A request is a real moment a returning user shows up: one of their events at
time t, with t later than their first-ever event. Everything the system may use
is computed strictly before t (history, streaming features, candidates); the
relevant set is the distinct items the user touches in [t, t + horizon), clipped
to the window so labels never cross into the next split.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from streamline.features.definitions import MINUTE_MS
from streamline.features.offline import user_histories

HORIZON_MS = 30 * MINUTE_MS


@dataclass
class RequestSet:
    frame: pl.DataFrame  # request_id, user_id, as_of_ms, hist_ts, hist_items, hist_events
    relevant: list[set[int]]
    seen: list[set[int]]  # every item the user touched before as_of (for new-item metrics)

    def __len__(self) -> int:
        return self.frame.height


def sample_requests(
    events: pl.DataFrame,
    start_ms: int,
    end_ms: int,
    n: int,
    seed: int = 0,
    horizon_ms: int = HORIZON_MS,
) -> RequestSet:
    first_seen = events.group_by("user_id").agg(pl.col("ts_ms").min().alias("first_ts"))
    eligible = (
        events.filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms))
        .join(first_seen, on="user_id")
        .filter(pl.col("ts_ms") > pl.col("first_ts"))  # has at least one earlier event
        .select("user_id", pl.col("ts_ms").alias("as_of_ms"))
        .unique()
        .sort(["user_id", "as_of_ms"])  # canonical order so seeded sampling is reproducible
    )
    one_per_user = (
        eligible.sample(fraction=1.0, shuffle=True, seed=seed)
        .unique("user_id", keep="first", maintain_order=True)
        .sort("user_id")
    )
    chosen = (
        one_per_user.sample(min(n, one_per_user.height), seed=seed)
        .sort(["as_of_ms", "user_id"])
        .with_row_index("request_id")
    )

    frame = user_histories(events, chosen.select("request_id", "user_id", "as_of_ms"))
    user_events = events.join(chosen.select("user_id"), on="user_id", how="semi").select(
        "user_id", "ts_ms", "item_id"
    )
    pairs = chosen.join(user_events, on="user_id")
    relevant = _sets(
        pairs.filter(
            (pl.col("ts_ms") >= pl.col("as_of_ms"))
            & (pl.col("ts_ms") < pl.min_horizontal(pl.col("as_of_ms") + horizon_ms, end_ms))
        ),
        chosen.height,
    )
    seen = _sets(pairs.filter(pl.col("ts_ms") < pl.col("as_of_ms")), chosen.height)
    return RequestSet(frame=frame, relevant=relevant, seen=seen)


def _sets(pairs: pl.DataFrame, n: int) -> list[set[int]]:
    grouped = pairs.group_by("request_id").agg(pl.col("item_id").unique())
    out: list[set[int]] = [set() for _ in range(n)]
    for rid, items in grouped.iter_rows():
        out[rid] = set(items)
    return out
