"""Offline evaluation harness: fit on history, predict the next time window.

Protocol: a recommender is fit on every event before a cutoff, then asked for a
top-K list for each *warm* user (a user with at least one event before the cutoff
who is active in the target window). The relevant set is every distinct item the
user touched in the target window. Cold users (no prior events) are counted but
not scored, since without history every model degrades to popularity.

`new_*` metrics isolate discovery: items the user already touched before the
cutoff are removed from both the recommendations and the relevant set, and only
users with at least one new item in the target window are scored.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from streamline.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

DEFAULT_KS = (10, 50)
SEEN_HEADROOM = 50  # extra candidates requested so lists stay full after removing seen items


class Recommender(Protocol):
    name: str

    def fit(self, history: pl.DataFrame) -> None: ...

    def recommend(self, user_ids: Sequence[int], k: int) -> list[list[int]]: ...


@dataclass(frozen=True)
class EvalSet:
    user_ids: list[int]
    relevant: list[set[int]]
    n_cold_users: int
    seen: list[set[int]] = field(default_factory=list)

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
    seen = (
        history.join(grouped.select("user_id"), on="user_id", how="semi")
        .group_by("user_id")
        .agg(pl.col("item_id").unique())
    )
    seen_by_user = dict(zip(seen["user_id"], seen["item_id"].to_list(), strict=True))
    users = grouped["user_id"].to_list()
    return EvalSet(
        user_ids=users,
        relevant=[set(items) for items in grouped["item_id"].to_list()],
        n_cold_users=target_users - grouped.height,
        seen=[set(seen_by_user.get(u, [])) for u in users],
    )


def novel_view(recs: Sequence[Sequence[int]], eval_set: EvalSet) -> tuple[list[list[int]], EvalSet]:
    """Drop already-seen items from recs and targets; keep users with a new target."""
    keep = [
        i
        for i, (rel, seen) in enumerate(zip(eval_set.relevant, eval_set.seen, strict=True))
        if rel - seen
    ]
    new_recs = [[x for x in recs[i] if x not in eval_set.seen[i]] for i in keep]
    return new_recs, EvalSet(
        user_ids=[eval_set.user_ids[i] for i in keep],
        relevant=[eval_set.relevant[i] - eval_set.seen[i] for i in keep],
        n_cold_users=eval_set.n_cold_users,
        seen=[eval_set.seen[i] for i in keep],
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
    recs = model.recommend(eval_set.user_ids, max(ks) + SEEN_HEADROOM)
    elapsed = time.perf_counter() - start
    metrics = score([r[: max(ks)] for r in recs], eval_set, ks)
    new_recs, new_set = novel_view(recs, eval_set)
    metrics.update({f"new_{k}": v for k, v in score(new_recs, new_set, ks).items()})
    metrics["eval_users"] = float(len(eval_set))
    metrics["new_eval_users"] = float(len(new_set))
    metrics["cold_users_skipped"] = float(eval_set.n_cold_users)
    metrics["recommend_ms_per_user"] = 1000 * elapsed / max(len(eval_set), 1)
    return metrics
