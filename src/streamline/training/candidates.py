"""Candidate generation: FAISS top-K from the two-tower model + the user's recent items.

Recent items matter because repeat engagement is common (see Phase 1); ANN alone
does not reliably surface them. Every candidate keeps its retrieval score so the
ranker can use it, including recent-only items.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from streamline.training.ann import AnnConfig, AnnIndex
from streamline.training.two_tower import TwoTowerRecommender

ANN_K = 500
N_RECENT = 50


class Retriever:
    def __init__(self, model: TwoTowerRecommender, config: AnnConfig | None = None) -> None:
        self.model = model
        self.item_ids, self.item_vecs = model.item_embeddings()  # ids are sorted ascending
        self.index = AnnIndex(self.item_ids, self.item_vecs, config)

    @classmethod
    def from_index(cls, model: TwoTowerRecommender, index: AnnIndex) -> Retriever:
        """Wrap a prebuilt (e.g. loaded) index instead of building one."""
        obj = cls.__new__(cls)
        obj.model = model
        obj.item_ids, obj.item_vecs = model.item_embeddings()
        obj.index = index
        return obj

    def user_vectors(
        self, hist_items: Sequence[Sequence[int]], hist_events: Sequence[Sequence[str]]
    ) -> np.ndarray:
        return self.model.user_embeddings(hist_items, hist_events)

    def scores(self, user_vecs: np.ndarray, rows: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Two-tower score of (user row, item) pairs; NaN for items outside the vocabulary."""
        pos = np.searchsorted(self.item_ids, items)
        pos = np.minimum(pos, len(self.item_ids) - 1)
        known = self.item_ids[pos] == items
        out = np.einsum("ij,ij->i", user_vecs[rows], self.item_vecs[pos]).astype(np.float32)
        return np.where(known, out, np.nan).astype(np.float32)

    def candidates(
        self,
        request_ids: Sequence[int],
        hist_items: Sequence[Sequence[int]],
        hist_events: Sequence[Sequence[str]],
        k: int = ANN_K,
        n_recent: int = N_RECENT,
    ) -> pl.DataFrame:
        """Long frame: request_id, item_id, ann_rank, recent_rank, ann_score.

        ann_rank / recent_rank are null when the item did not come from that source.
        """
        user_vecs = self.user_vectors(hist_items, hist_events)
        ann_ids, _ = self.index.search(user_vecs, k)
        n, kk = ann_ids.shape
        ann = pl.DataFrame(
            {
                "row": np.repeat(np.arange(n), kk),
                "item_id": ann_ids.ravel(),
                "ann_rank": np.tile(np.arange(kk, dtype=np.int32), n),
            }
        ).filter(pl.col("item_id") >= 0)

        recent_rows, recent_items, recent_rank = [], [], []
        for row, items in enumerate(hist_items):
            seen: set[int] = set()
            for item in reversed(items):  # most recent first
                if item not in seen:
                    seen.add(item)
                    recent_rows.append(row)
                    recent_items.append(item)
                    recent_rank.append(len(seen) - 1)
                    if len(seen) >= n_recent:
                        break
        recent = pl.DataFrame(
            {"row": recent_rows, "item_id": recent_items, "recent_rank": recent_rank},
            schema={"row": pl.Int64, "item_id": pl.Int64, "recent_rank": pl.Int32},
        )

        merged = ann.with_columns(pl.col("row").cast(pl.Int64)).join(
            recent, on=["row", "item_id"], how="full", coalesce=True
        )
        rows = merged["row"].to_numpy()
        merged = merged.with_columns(
            ann_score=pl.Series(self.scores(user_vecs, rows, merged["item_id"].to_numpy())),
            request_id=pl.Series(np.asarray(request_ids, dtype=np.int64)[rows]),
        )
        return merged.select("request_id", "item_id", "ann_rank", "recent_rank", "ann_score").sort(
            ["request_id", "item_id"]
        )
