"""FAISS index over item-tower embeddings (inner product on L2-normalised vectors).

HNSW is the serving index; Flat (exact) is kept to measure how much recall the
approximation costs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from streamline.openmp import check_single_openmp

check_single_openmp()

import faiss  # noqa: E402

IndexKind = Literal["hnsw", "flat"]


@dataclass(frozen=True)
class AnnConfig:
    kind: IndexKind = "hnsw"
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 800  # must exceed the 500 candidates retrieved per request


class AnnIndex:
    def __init__(self, item_ids: np.ndarray, vectors: np.ndarray, config: AnnConfig | None = None):
        self.config = config or AnnConfig()
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        self.index: Any
        if self.config.kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.config.ef_construction
            self.index.hnsw.efSearch = self.config.ef_search
        start = time.perf_counter()
        self.index.add(vectors)
        self.build_seconds = time.perf_counter() - start

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k raw item ids and scores per query row (ids are -1 when fewer than k)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        scores, idx = self.index.search(queries, min(k, len(self)))
        ids = np.where(idx >= 0, self.item_ids[np.maximum(idx, 0)], -1)
        return ids, scores

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "item_ids.npy", self.item_ids)
        (path / "config.json").write_text(json.dumps(asdict(self.config)))

    @classmethod
    def load(cls, path: Path) -> AnnIndex:
        obj = cls.__new__(cls)
        obj.config = AnnConfig(**json.loads((path / "config.json").read_text()))
        obj.index = faiss.read_index(str(path / "index.faiss"))
        if obj.config.kind == "hnsw":
            obj.index.hnsw.efSearch = obj.config.ef_search
        obj.item_ids = np.load(path / "item_ids.npy")
        obj.build_seconds = 0.0
        return obj


def ann_recall(approx: np.ndarray, exact: np.ndarray) -> float:
    """Mean fraction of the exact top-k that the approximate top-k recovers."""
    hits = [len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact, strict=True) if len(e)]
    return float(np.mean(hits))
