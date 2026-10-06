"""Ranker features for (request, candidate) pairs, all as of the request time.

Groups:
  retrieval  ann_score, ann_rank, recent_rank
  user-item  from the user's last-50-event history (the same history the online store serves)
  item       streaming item features at as_of (features/definitions.py)
  user       streaming user/session features at as_of
Nulls mean "not applicable" (e.g. never interacted) and are left to LightGBM.
"""

from __future__ import annotations

import polars as pl

from streamline.features.definitions import ITEM_FEATURES, USER_FEATURES
from streamline.features.offline import item_features, user_features

RETRIEVAL_FEATURES = ("ann_score", "ann_rank", "recent_rank")
USER_ITEM_FEATURES = ("ui_events", "ui_carts", "ui_age_ms", "ui_last_pos", "ui_is_last")
FEATURES: tuple[str, ...] = RETRIEVAL_FEATURES + USER_ITEM_FEATURES + ITEM_FEATURES + USER_FEATURES


def user_item_features(requests: pl.DataFrame) -> pl.DataFrame:
    """Per (request_id, item_id) aggregates over the request's history lists."""
    exploded = (
        requests.select("request_id", "as_of_ms", "hist_ts", "hist_items", "hist_events")
        .with_columns(n=pl.col("hist_items").list.len())
        .filter(pl.col("n") > 0)
        .explode(["hist_ts", "hist_items", "hist_events"])
        .with_columns(pos=pl.int_range(pl.len()).over("request_id"))
        .with_columns(pos_from_end=pl.col("n") - 1 - pl.col("pos"))
    )
    return exploded.group_by("request_id", pl.col("hist_items").alias("item_id")).agg(
        ui_events=pl.len().cast(pl.Int32),
        ui_carts=(pl.col("hist_events") == "addtocart").sum().cast(pl.Int32),
        ui_age_ms=(pl.col("as_of_ms").first() - pl.col("hist_ts").max()),
        ui_last_pos=pl.col("pos_from_end").min().cast(pl.Int32),
        ui_is_last=(pl.col("pos_from_end").min() == 0).cast(pl.Int8),
    )


def build_features(
    candidates: pl.DataFrame, requests: pl.DataFrame, events: pl.DataFrame
) -> pl.DataFrame:
    """candidates (request_id, item_id, ann_rank, recent_rank, ann_score) + features.

    `events` is the full event log; every feature only reads events before as_of.
    """
    req = requests.select("request_id", "user_id", "as_of_ms")
    pairs = candidates.join(req, on="request_id", how="left", maintain_order="left")

    items = item_features(events, pairs.select("item_id", "as_of_ms")).drop("item_id", "as_of_ms")
    users = user_features(events, req.select("user_id", "as_of_ms")).drop("user_id", "as_of_ms")
    users = pl.concat([req.select("request_id"), users], how="horizontal")

    return (
        pairs.hstack(items)
        .join(users, on="request_id", how="left", maintain_order="left")
        .join(
            user_item_features(requests),
            on=["request_id", "item_id"],
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.col("ui_is_last").fill_null(0))
    )


def add_labels(frame: pl.DataFrame, relevant: list[set[int]]) -> pl.DataFrame:
    pairs = pl.DataFrame(
        {
            "request_id": [rid for rid, items in enumerate(relevant) for _ in items],
            "item_id": [item for items in relevant for item in items],
        },
        schema={"request_id": pl.Int64, "item_id": pl.Int64},
    ).with_columns(label=pl.lit(1, pl.Int8))
    return frame.join(
        pairs.with_columns(pl.col("request_id").cast(frame.schema["request_id"])),
        on=["request_id", "item_id"],
        how="left",
        maintain_order="left",
    ).with_columns(pl.col("label").fill_null(0))
