"""Phase 3: FAISS retrieval + LightGBM reranking, evaluated at request time.

  1. Two-tower A: fit on train (early stopping on val). Two-tower B: refit on
     train + val with the selected epoch count.
  2. Ranker training data: requests sampled from the val window, candidates
     from A's FAISS index, features as of each request time.
  3. Evaluation: requests from the test window, candidates from B's index,
     reranked by the ranker. Nothing at or after a request's time is used.

Exports B + its index + the ranker to artifacts/serving for Phase 4.

Usage: uv run python -m streamline.training.train_ranker [--requests 8000]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

import mlflow
import numpy as np
import polars as pl

from streamline.config import REPO_ROOT, get_settings
from streamline.eval.harness import EvalSet, build_eval_set, novel_view, score
from streamline.ingest.events import load_events
from streamline.training.ann import AnnConfig, AnnIndex, ann_recall
from streamline.training.baselines import Popularity
from streamline.training.candidates import ANN_K, Retriever
from streamline.training.ranker import Ranker
from streamline.training.ranker_features import add_labels, build_features
from streamline.training.requests import HORIZON_MS, RequestSet, sample_requests
from streamline.training.split import TimeSplit, time_split
from streamline.training.train import _git_sha, _setup_mlflow
from streamline.training.two_tower import TwoTowerConfig, TwoTowerRecommender

KS = (10, 50)
ARTIFACTS = REPO_ROOT / "artifacts"


def fit_two_towers(split: TimeSplit, seed: int) -> tuple[TwoTowerRecommender, TwoTowerRecommender]:
    """A on train (early-stopped on val Recall@50), B on train + val for A's best epoch."""
    a_path, b_path = (
        ARTIFACTS / f"two_tower_train_s{seed}",
        ARTIFACTS / f"two_tower_trainval_s{seed}",
    )
    if (a_path / "config.json").exists() and (b_path / "config.json").exists():
        print("reusing cached two-tower models in artifacts/")
        return TwoTowerRecommender.load(a_path), TwoTowerRecommender.load(b_path)

    val_set = build_eval_set(split.train, split.val)

    def on_epoch(epoch: int, loss: float, model: TwoTowerRecommender) -> float:
        recall = score(model.recommend(val_set.user_ids, 50), val_set, (50,))["recall@50"]
        print(f"  two-tower A epoch={epoch} loss={loss:.3f} val_recall@50={recall:.4f}", flush=True)
        return recall

    a = TwoTowerRecommender(TwoTowerConfig(seed=seed))
    a.fit(split.train, on_epoch=on_epoch)
    b = TwoTowerRecommender(TwoTowerConfig(seed=seed, epochs=a.best_epoch))
    b.fit(split.train_val)
    a.save(a_path)
    b.save(b_path)
    return a, b


def featurize(
    retriever: Retriever, reqs: RequestSet, events: pl.DataFrame
) -> tuple[pl.DataFrame, float]:
    f = reqs.frame
    start = time.perf_counter()
    cands = retriever.candidates(
        f["request_id"].to_list(), f["hist_items"].to_list(), f["hist_events"].to_list()
    )
    frame = add_labels(build_features(cands, f, events), reqs.relevant).sort(
        ["request_id", "item_id"]
    )
    return frame, time.perf_counter() - start


def request_eval_set(reqs: RequestSet) -> EvalSet:
    return EvalSet(
        user_ids=reqs.frame["request_id"].to_list(),
        relevant=reqs.relevant,
        n_cold_users=0,
        seen=reqs.seen,
    )


def evaluate_lists(recs: list[list[int]], eval_set: EvalSet) -> dict[str, float]:
    metrics = score([r[: max(KS)] for r in recs], eval_set, KS)
    new_recs, new_set = novel_view(recs, eval_set)
    metrics.update({f"new_{k}": v for k, v in score(new_recs, new_set, KS).items()})
    return metrics


def recent_then_popular(reqs: RequestSet, popular: list[int], k: int) -> list[list[int]]:
    out = []
    for items in reqs.frame["hist_items"].to_list():
        recs = list(dict.fromkeys(reversed(items)))[:k]
        seen = set(recs)
        recs += [i for i in popular if i not in seen][: k - len(recs)]
        out.append(recs)
    return out


def latency_profile(
    retriever: Retriever, ranker: Ranker, reqs: RequestSet, frame: pl.DataFrame, n: int = 300
) -> dict[str, float]:
    """Single-request timings (ms) for the model parts of the serving path."""
    f = reqs.frame.head(n)
    tower, ann, rank = [], [], []
    for i, row in enumerate(f.iter_rows(named=True)):
        t0 = time.perf_counter()
        vec = retriever.user_vectors([row["hist_items"]], [row["hist_events"]])
        t1 = time.perf_counter()
        retriever.index.search(vec, ANN_K)
        t2 = time.perf_counter()
        ranker.predict(frame.filter(pl.col("request_id") == row["request_id"]))
        t3 = time.perf_counter()
        if i >= 10:  # skip warm-up
            tower.append(t1 - t0)
            ann.append(t2 - t1)
            rank.append(t3 - t2)
    out = {}
    for name, xs in (("user_tower", tower), ("ann_search", ann), ("ranker_predict", rank)):
        ms = np.asarray(xs) * 1000
        out[f"{name}_p50_ms"] = float(np.percentile(ms, 50))
        out[f"{name}_p99_ms"] = float(np.percentile(ms, 99))
    return out


def ann_quality(retriever: Retriever, reqs: RequestSet, n: int = 2000) -> dict[str, float]:
    f = reqs.frame.head(n)
    vecs = retriever.user_vectors(f["hist_items"].to_list(), f["hist_events"].to_list())
    flat = AnnIndex(retriever.item_ids, retriever.item_vecs, AnnConfig(kind="flat"))
    approx_ids, _ = retriever.index.search(vecs, ANN_K)
    exact_ids, _ = flat.search(vecs, ANN_K)
    return {
        "ann_recall@500_vs_exact": ann_recall(approx_ids, exact_ids),
        "ann_recall@50_vs_exact": ann_recall(approx_ids[:, :50], exact_ids[:, :50]),
        "hnsw_build_seconds": retriever.index.build_seconds,
    }


def _safe(metrics: dict[str, float], prefix: str = "") -> dict[str, float]:
    """MLflow metric names allow no '@'; map 'recall@50' -> 'recall_at_50'."""
    return {f"{prefix}{k}".replace("@", "_at_"): float(v) for k, v in metrics.items()}


def results_table(rows: dict[str, dict[str, float]]) -> str:
    cols = [
        ("recall@50", "Recall@50"),
        ("ndcg@10", "NDCG@10"),
        ("mrr@50", "MRR@50"),
        ("new_recall@50", "New-item Recall@50"),
        ("new_ndcg@10", "New-item NDCG@10"),
    ]
    lines = [
        "| Method | " + " | ".join(c[1] for c in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.4f}" for c, _ in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    _setup_mlflow(settings.mlflow_tracking_uri)
    events = load_events()
    split = time_split(events)
    end_ms = int(events["ts_ms"].max()) + 1  # type: ignore[arg-type]

    with mlflow.start_run(run_name="retrieval_ranking"):
        mlflow.set_tag("git_sha", _git_sha())
        mlflow.log_params(
            {"requests": args.requests, "seed": args.seed, "ann_k": ANN_K, "horizon_ms": HORIZON_MS}
        )
        tt_a, tt_b = fit_two_towers(split, args.seed)

        train_reqs = sample_requests(
            events, split.val_start_ms, split.test_start_ms, args.requests, seed=args.seed
        )
        test_reqs = sample_requests(
            events, split.test_start_ms, end_ms, args.requests, seed=args.seed + 1
        )
        print(f"requests: ranker-train={len(train_reqs):,} test={len(test_reqs):,}")

        retr_a, retr_b = Retriever(tt_a), Retriever(tt_b)
        train_frame, t_train = featurize(retr_a, train_reqs, events)
        test_frame, t_test = featurize(retr_b, test_reqs, events)
        print(
            f"features: train {train_frame.shape} in {t_train:.0f}s, test {test_frame.shape} "
            f"in {t_test:.0f}s"
        )

        # time-ordered split of the val-window requests for early stopping
        cut = int(len(train_reqs) * 0.8)
        ranker = Ranker()
        fit_stats = ranker.fit(
            train_frame.filter(pl.col("request_id") < cut),
            train_frame.filter(pl.col("request_id") >= cut),
        )
        print(f"ranker: {fit_stats}")

        eval_set = request_eval_set(test_reqs)
        n = len(test_reqs)
        popular = Popularity(14)
        popular.fit(split.train_val)
        hist = test_reqs.frame
        user_vecs = retr_b.user_vectors(hist["hist_items"].to_list(), hist["hist_events"].to_list())
        flat = AnnIndex(retr_b.item_ids, retr_b.item_vecs, AnnConfig(kind="flat"))
        cand_lists = (
            test_frame.group_by("request_id", maintain_order=True)
            .agg("item_id")["item_id"]
            .to_list()
        )
        rows: dict[str, dict[str, float]] = {
            "Popularity (14d)": evaluate_lists(popular.recommend(list(range(n)), 100), eval_set),
            "Recently viewed + popularity": evaluate_lists(
                recent_then_popular(test_reqs, popular.ranking, 100), eval_set
            ),
            "Two-tower, exact": evaluate_lists(flat.search(user_vecs, 100)[0].tolist(), eval_set),
            "Two-tower, FAISS HNSW": evaluate_lists(
                retr_b.index.search(user_vecs, 100)[0].tolist(), eval_set
            ),
            "**Two-tower + recent → LightGBM**": evaluate_lists(
                ranker.rerank(test_frame, n, 100), eval_set
            ),
        }
        candidate_recall = float(
            np.mean(
                [
                    len(set(c) & r) / len(r)
                    for c, r in zip(cand_lists, test_reqs.relevant, strict=True)
                ]
            )
        )
        quality = ann_quality(retr_b, test_reqs)
        latency = latency_profile(retr_b, ranker, test_reqs, test_frame)
        importance = ranker.importance()

        for name, m in rows.items():
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            mlflow.log_metrics(_safe(m, prefix=f"{slug}_"))
        mlflow.log_metrics(
            _safe({"candidate_recall": candidate_recall, **quality, **latency})
            | _safe(fit_stats, prefix="ranker_")
        )

        serving = ARTIFACTS / "serving"
        tt_b.save(serving / "two_tower")
        retr_b.index.save(serving / "ann")
        ranker.save(serving / "ranker")
        mlflow.log_artifacts(str(serving), artifact_path="serving")

    report: dict[str, Any] = {
        "protocol": {
            "requests_per_split": args.requests,
            "horizon_minutes": HORIZON_MS // 60_000,
            "ann_k": ANN_K,
            "test_requests": n,
            "mean_relevant_per_request": float(np.mean([len(r) for r in test_reqs.relevant])),
            "git_sha": _git_sha(),
        },
        "methods": rows,
        "candidate_recall": candidate_recall,
        "candidates_per_request": test_frame.height / n,
        "ann": quality,
        "latency_ms": latency,
        "ranker": fit_stats,
        "feature_importance_gain": importance,
    }
    out = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "ranking.json").write_text(json.dumps(report, indent=2) + "\n")
    table = results_table(rows)
    (out / "ranking.md").write_text(table)
    print("\n" + table)
    print(json.dumps({k: report[k] for k in ("candidate_recall", "ann", "latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
