from pathlib import Path

import numpy as np

from streamline.training.ann import AnnConfig, AnnIndex, ann_recall


def _vectors(n: int = 3000, dim: int = 16, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)
    normed: np.ndarray = v / np.linalg.norm(v, axis=1, keepdims=True)
    return normed


def test_flat_index_is_exact_and_maps_ids() -> None:
    vecs = _vectors()
    ids = np.arange(3000) * 7 + 5  # raw ids differ from positions
    index = AnnIndex(ids, vecs, AnnConfig(kind="flat"))
    got, scores = index.search(vecs[:10], k=5)
    brute = np.argsort(-(vecs[:10] @ vecs.T), axis=1)[:, :5]
    assert (got == ids[brute]).all()
    assert (np.diff(scores, axis=1) <= 1e-6).all()


def test_hnsw_recall_close_to_exact() -> None:
    vecs = _vectors()
    queries = _vectors(200, seed=1)
    exact, _ = AnnIndex(np.arange(3000), vecs, AnnConfig(kind="flat")).search(queries, 50)
    approx, _ = AnnIndex(np.arange(3000), vecs).search(queries, 50)
    assert ann_recall(approx, exact) > 0.95


def test_save_load_roundtrip(tmp_path: Path) -> None:
    vecs = _vectors()
    index = AnnIndex(np.arange(3000) + 100, vecs)
    index.save(tmp_path / "ann")
    loaded = AnnIndex.load(tmp_path / "ann")
    a, _ = index.search(vecs[:5], 10)
    b, _ = loaded.search(vecs[:5], 10)
    assert (a == b).all()


def test_k_larger_than_index() -> None:
    index = AnnIndex(np.arange(3), _vectors(3), AnnConfig(kind="flat"))
    ids, _ = index.search(_vectors(1, seed=2), k=10)
    assert sorted(ids[0].tolist()) == [0, 1, 2]
