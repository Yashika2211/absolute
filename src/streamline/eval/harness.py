"""Offline evaluation harness: fit on history, predict the next time window.

Protocol: a recommender is fit on every event before a cutoff, then asked for a
top-K list for each *warm* user (a user with at least one event before the cutoff
who is active in the target window). The relevant set is every distinct item the
user touched in the target window. Cold users (no prior events) are counted but
not scored, since without history every model degrades to popularity.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl

from streamline.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DEFAULT_KS = (10, 50)


class Recommender(Protocol):
    name: str

    def fit(self, history: pl.DataFrame) -> None: ...

    def recommend(self, user_ids: Sequence[int], k: int) -> list[list[int]]: ...


@dataclass(frozen=True)
class EvalSet:
    user_ids: list[int]
    relevant: list[set[int]]
    n_cold_users: int

    def __len__(self) -> int:
        return len(self.user_ids)


def build_eval_set(history: pl.DataFrame, target: pl.DataFrame) -> EvalSet:
    warm = history.select("user_id").unique()
    target_users = target["user_id"].n_unique()
    grouped = (
        target.join(warm, on="user_id", how="semi")
        .group_by("user_id")
        .agg(pl.col("item_id").unique())
        .sort("user_id")
    )
    return EvalSet(
        user_ids=grouped["user_id"].to_list(),
        relevant=[set(items) for items in grouped["item_id"].to_list()],
        n_cold_users=target_users - grouped.height,
    )


def score(recs: Sequence[Sequence[int]], eval_set: EvalSet, ks: Sequence[int]) -> dict[str, float]:
    if len(recs) != len(eval_set):
        raise ValueError("need one recommendation list per eval user")
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(
            np.mean(
                [recall_at_k(r, rel, k) for r, rel in zip(recs, eval_set.relevant, strict=True)]
            )
        )
        out[f"ndcg@{k}"] = float(
            np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(recs, eval_set.relevant, strict=True)])
        )
    kmax = max(ks)
    out[f"mrr@{kmax}"] = float(
        np.mean([mrr_at_k(r, rel, kmax) for r, rel in zip(recs, eval_set.relevant, strict=True)])
    )
    return out


def evaluate(
    model: Recommender,
    history: pl.DataFrame,
    target: pl.DataFrame,
    ks: Sequence[int] = DEFAULT_KS,
    fit: bool = True,
) -> dict[str, float]:
    """Optionally fit `model` on `history`, then score it on `target`."""
    if fit:
        model.fit(history)
    eval_set = build_eval_set(history, target)
    start = time.perf_counter()
    recs = model.recommend(eval_set.user_ids, max(ks))
    elapsed = time.perf_counter() - start
    metrics = score(recs, eval_set, ks)
    metrics["eval_users"] = float(len(eval_set))
    metrics["cold_users_skipped"] = float(eval_set.n_cold_users)
    metrics["recommend_ms_per_user"] = 1000 * elapsed / max(len(eval_set), 1)
    return metrics
