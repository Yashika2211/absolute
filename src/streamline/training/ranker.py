"""LightGBM LambdaRank reranker over retrieval candidates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from streamline.training.ranker_features import FEATURES


@dataclass(frozen=True)
class RankerConfig:
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 1000
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambdarank_truncation_level: int = 50
    early_stopping_rounds: int = 50
    seed: int = 42


def _groups(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("request_id", maintain_order=True).len()["len"].to_numpy()
    return counts


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(pl.col(f).cast(pl.Float32) for f in features).to_numpy()


def with_positives(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep only requests with at least one relevant candidate (others carry no gradient)."""
    keep = frame.group_by("request_id").agg(pl.col("label").max()).filter(pl.col("label") > 0)
    return frame.join(keep.select("request_id"), on="request_id", how="semi", maintain_order="left")


class Ranker:
    def __init__(self, config: RankerConfig | None = None, features: Sequence[str] = FEATURES):
        self.config = config or RankerConfig()
        self.features = list(features)
        self.booster: lgb.Booster | None = None

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> dict[str, float]:
        """Frames must be sorted by request_id and contain `label`."""
        cfg = self.config
        train, valid = with_positives(train), with_positives(valid)
        dtrain = lgb.Dataset(
            _matrix(train, self.features),
            train["label"].to_numpy(),
            group=_groups(train),
            feature_name=self.features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            _matrix(valid, self.features),
            valid["label"].to_numpy(),
            group=_groups(valid),
            reference=dtrain,
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [10, 50],
            "num_leaves": cfg.num_leaves,
            "learning_rate": cfg.learning_rate,
            "min_data_in_leaf": cfg.min_data_in_leaf,
            "feature_fraction": cfg.feature_fraction,
            "bagging_fraction": cfg.bagging_fraction,
            "bagging_freq": cfg.bagging_freq,
            "lambdarank_truncation_level": cfg.lambdarank_truncation_level,
            "seed": cfg.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }
        evals: dict[str, dict[str, list[float]]] = {}
        self.booster = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg.n_estimators,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals),
            ],
        )
        best = self.booster.best_iteration
        return {
            "best_iteration": float(best),
            "valid_ndcg@10": evals["valid"]["ndcg@10"][best - 1],
            "valid_ndcg@50": evals["valid"]["ndcg@50"][best - 1],
            "train_requests": float(train["request_id"].n_unique()),
            "valid_requests": float(valid["request_id"].n_unique()),
        }

    def predict(self, frame: pl.DataFrame, num_threads: int = 0) -> np.ndarray:
        """num_threads=0 uses LightGBM's default (all cores); serving passes 1."""
        assert self.booster is not None, "fit or load first"
        scores = self.booster.predict(
            _matrix(frame, self.features),
            num_iteration=self.booster.best_iteration,
            num_threads=num_threads,
        )
        return np.asarray(scores, dtype=np.float64)

    def rerank(self, frame: pl.DataFrame, n_requests: int, k: int) -> list[list[int]]:
        """Top-k item ids per request_id in [0, n_requests) (empty list if no candidates)."""
        ranked = (
            frame.select("request_id", "item_id")
            .with_columns(score=pl.Series(self.predict(frame)))
            .sort(["request_id", "score", "item_id"], descending=[False, True, False])
            .group_by("request_id", maintain_order=True)
            .agg(pl.col("item_id").head(k))
        )
        out: list[list[int]] = [[] for _ in range(n_requests)]
        for rid, items in ranked.iter_rows():
            out[rid] = items
        return out

    def importance(self) -> dict[str, float]:
        assert self.booster is not None
        gain = self.booster.feature_importance(importance_type="gain")
        total = float(gain.sum()) or 1.0
        pairs = sorted(zip(self.features, gain, strict=True), key=lambda p: -p[1])
        return {name: round(float(g) / total, 4) for name, g in pairs}

    def save(self, path: Path) -> None:
        assert self.booster is not None
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "ranker.txt"), num_iteration=self.booster.best_iteration)
        (path / "ranker.json").write_text(
            json.dumps({"config": asdict(self.config), "features": self.features}, indent=2)
        )

    @classmethod
    def load(cls, path: Path) -> Ranker:
        meta = json.loads((path / "ranker.json").read_text())
        ranker = cls(RankerConfig(**meta["config"]), meta["features"])
        ranker.booster = lgb.Booster(model_file=str(path / "ranker.txt"))
        return ranker
