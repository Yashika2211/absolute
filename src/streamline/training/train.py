"""Train and compare retrieval models; log everything to MLflow.

Protocol (see eval/harness.py):
  1. Select hyperparameters on validation: fit on train, predict the val window.
  2. Report test: refit on train + val with the selected settings, predict test.

Usage: uv run python -m streamline.training.train [--seeds 42 43 44] [--epochs 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from streamline.config import get_settings
from streamline.eval.harness import build_eval_set, evaluate, score
from streamline.ingest.events import load_events
from streamline.training.baselines import Popularity, RecentThenPopular
from streamline.training.split import TimeSplit, time_split
from streamline.training.two_tower import TwoTowerConfig, TwoTowerRecommender

EXPERIMENT = "streamline-retrieval"
POPULARITY_WINDOWS: tuple[int | None, ...] = (3, 7, 14, 30, None)
SELECTION_METRIC = "recall@50"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _setup_mlflow(uri: str) -> None:
    import urllib.request

    try:
        urllib.request.urlopen(f"{uri}/health", timeout=3)
        mlflow.set_tracking_uri(uri)
    except OSError:
        local = get_settings().data_dir.parent / "mlruns"
        print(f"MLflow server at {uri} unreachable; logging to {local} (run `make up`)")
        mlflow.set_tracking_uri(local.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _log_split(split: TimeSplit) -> None:
    mlflow.log_params(
        {
            "val_start_ms": split.val_start_ms,
            "test_start_ms": split.test_start_ms,
            "train_events": split.train.height,
            "val_events": split.val.height,
            "test_events": split.test.height,
        }
    )
    mlflow.set_tag("git_sha", _git_sha())


def _log_metrics(prefix: str, metrics: dict[str, float], step: int | None = None) -> None:
    clean = {f"{prefix}_{k.replace('@', '_at_')}": v for k, v in metrics.items()}
    mlflow.log_metrics(clean, step=step)


def run_popularity(split: TimeSplit) -> tuple[dict[str, float], dict[str, float], int | None]:
    with mlflow.start_run(run_name="popularity"):
        _log_split(split)
        best_window, best_val, best_score = None, {}, -1.0
        for window in POPULARITY_WINDOWS:
            val = evaluate(Popularity(window), split.train, split.val)
            mlflow.log_metric(f"val_recall_at_50_window_{window or 'all'}", val[SELECTION_METRIC])
            if val[SELECTION_METRIC] > best_score:
                best_window, best_val, best_score = window, val, val[SELECTION_METRIC]
        mlflow.log_param("window_days", best_window or "all")
        test = evaluate(Popularity(best_window), split.train_val, split.test)
        _log_metrics("val", best_val)
        _log_metrics("test", test)
        return best_val, test, best_window


def run_recent(split: TimeSplit, window: int | None) -> tuple[dict[str, float], dict[str, float]]:
    with mlflow.start_run(run_name="recent_then_popular"):
        _log_split(split)
        mlflow.log_param("window_days", window or "all")
        val = evaluate(RecentThenPopular(window), split.train, split.val)
        test = evaluate(RecentThenPopular(window), split.train_val, split.test)
        _log_metrics("val", val)
        _log_metrics("test", test)
        return val, test


def run_two_tower(
    split: TimeSplit, config: TwoTowerConfig
) -> tuple[dict[str, float], dict[str, float], int]:
    with mlflow.start_run(run_name=f"two_tower_seed{config.seed}"):
        _log_split(split)
        mlflow.log_params(config.to_dict())
        val_set = build_eval_set(split.train, split.val)

        def on_epoch(epoch: int, loss: float, model: TwoTowerRecommender) -> float:
            metrics = score(model.recommend(val_set.user_ids, 50), val_set, (10, 50))
            _log_metrics("val", {**metrics, "train_loss": loss}, step=epoch)
            print(
                f"  seed={config.seed} epoch={epoch} loss={loss:.3f} "
                f"val_recall@50={metrics['recall@50']:.4f}",
                flush=True,
            )
            return metrics[SELECTION_METRIC]

        start = time.perf_counter()
        model = TwoTowerRecommender(config)
        model.fit(split.train, on_epoch=on_epoch)
        best_epoch = model.best_epoch
        val = evaluate(model, split.train, split.val, fit=False)
        mlflow.log_param("best_epoch", best_epoch)

        # refit on train + val for exactly the selected number of epochs
        final_cfg = TwoTowerConfig(**{**config.to_dict(), "epochs": best_epoch})  # type: ignore[arg-type]
        final = TwoTowerRecommender(final_cfg)
        test = evaluate(final, split.train_val, split.test)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)
        _log_metrics("val", val)
        _log_metrics("test", test)
        _log_model(final)
        return val, test, best_epoch


def _log_model(model: TwoTowerRecommender) -> None:
    assert model.net is not None and model.vocab is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        torch.save(model.net.state_dict(), path / "state_dict.pt")
        np.save(path / "item_ids.npy", model.vocab.ids)
        (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2))
        mlflow.log_artifacts(str(path), artifact_path="model")


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


COLUMNS = [
    ("test", "recall@50", "Recall@50"),
    ("test", "ndcg@10", "NDCG@10"),
    ("test", "mrr@50", "MRR@50"),
    ("test", "new_recall@50", "New-item Recall@50"),
    ("test", "new_ndcg@10", "New-item NDCG@10"),
    ("val", "recall@50", "Val Recall@50"),
]


def results_table(results: dict[str, dict[str, Any]]) -> str:
    header = "| Model | " + " | ".join(c[2] for c in COLUMNS) + " |"
    lines = [header, "|---" * (len(COLUMNS) + 1) + "|"]
    for name, res in results.items():
        cells = []
        for split_name, metric, _ in COLUMNS:
            values = [run[split_name][metric] for run in res["runs"]]
            mean, std = _mean_std(values)
            cells.append(f"{mean:.4f} ± {std:.4f}" if len(values) > 1 else f"{mean:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    settings = get_settings()
    _setup_mlflow(settings.mlflow_tracking_uri)
    split = time_split(load_events())
    print(f"train={split.train.height:,} val={split.val.height:,} test={split.test.height:,}")

    results: dict[str, dict[str, Any]] = {}
    pop_val, pop_test, window = run_popularity(split)
    results["Popularity"] = {"runs": [{"val": pop_val, "test": pop_test}], "window_days": window}
    print(f"popularity (window={window}) test: {pop_test}")

    rec_val, rec_test = run_recent(split, window)
    results["Recently viewed + popularity"] = {"runs": [{"val": rec_val, "test": rec_test}]}
    print(f"recent_then_popular test: {rec_test}")

    runs = []
    for seed in args.seeds:
        cfg = TwoTowerConfig(epochs=args.epochs, seed=seed, device=args.device)
        val, test, best_epoch = run_two_tower(split, cfg)
        runs.append({"val": val, "test": test, "best_epoch": best_epoch, "seed": seed})
        print(f"two_tower seed={seed} best_epoch={best_epoch} test: {test}")
    results[f"Two-tower ({len(args.seeds)} seeds)"] = {"runs": runs}

    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "val_start_ms": split.val_start_ms,
        "test_start_ms": split.test_start_ms,
        "eval_users_test": int(pop_test["eval_users"]),
        "cold_users_skipped_test": int(pop_test["cold_users_skipped"]),
        "new_eval_users_test": int(pop_test["new_eval_users"]),
        "git_sha": _git_sha(),
    }
    (out / "results.json").write_text(
        json.dumps({"meta": meta, "models": results}, indent=2) + "\n"
    )
    table = results_table(results)
    (out / "results.md").write_text(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
