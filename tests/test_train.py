from streamline.training.train import results_table


def _run(v: float) -> dict[str, dict[str, float]]:
    metrics = {
        "recall@50": v,
        "ndcg@10": v,
        "mrr@50": v,
        "new_recall@50": v,
        "new_ndcg@10": v,
    }
    return {"val": metrics, "test": metrics}


def test_results_table_single_and_multi_seed() -> None:
    table = results_table(
        {
            "Popularity": {"runs": [_run(0.1)]},
            "Two-tower": {"runs": [_run(0.2), _run(0.4)]},
        }
    )
    lines = table.strip().splitlines()
    assert lines[0].startswith("| Model | Recall@50")
    assert "| Popularity | 0.1000 |" in lines[2]
    assert "0.3000 ± 0.1414" in lines[3]
