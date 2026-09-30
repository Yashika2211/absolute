"""Dataset summary: event counts, users, items, sparsity, activity distribution.

Usage: uv run python -m streamline.ingest.eda   (writes reports/eda.json and reports/eda.md)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from streamline.config import get_settings
from streamline.ingest.events import EVENT_TYPES, load_events


def _date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def summarize(df: pl.DataFrame) -> dict[str, Any]:
    n_users = df["user_id"].n_unique()
    n_items = df["item_id"].n_unique()
    n_pairs = df.select("user_id", "item_id").unique().height
    by_type = dict(df.group_by("event").len().select(pl.col("event").cast(pl.Utf8), "len").rows())
    per_user = df.group_by("user_id").len()["len"]
    per_item = df.group_by("item_id").len()["len"]
    ts_min, ts_max = int(df["ts_ms"].min()), int(df["ts_ms"].max())  # type: ignore[arg-type]
    views = by_type.get("view", 0)

    return {
        "events": df.height,
        "events_by_type": {t: int(by_type.get(t, 0)) for t in EVENT_TYPES},
        "users": n_users,
        "items": n_items,
        "user_item_pairs": n_pairs,
        "sparsity": 1.0 - n_pairs / (n_users * n_items),
        "start_date": _date(ts_min),
        "end_date": _date(ts_max),
        "days": round((ts_max - ts_min) / 86_400_000, 1),
        "events_per_user_median": float(per_user.median()),  # type: ignore[arg-type]
        "events_per_user_p99": float(per_user.quantile(0.99)),  # type: ignore[arg-type]
        "users_with_one_event_pct": round(100 * float((per_user == 1).mean()), 1),  # type: ignore[arg-type]
        "items_with_5plus_events": int((per_item >= 5).sum()),
        "view_to_cart_rate": round(by_type.get("addtocart", 0) / views, 4) if views else 0.0,
        "view_to_purchase_rate": round(by_type.get("transaction", 0) / views, 4) if views else 0.0,
    }


def to_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ("Events", f"{summary['events']:,}"),
        *((f"  {t}", f"{n:,}") for t, n in summary["events_by_type"].items()),
        ("Users (visitors)", f"{summary['users']:,}"),
        ("Items", f"{summary['items']:,}"),
        ("Distinct user-item pairs", f"{summary['user_item_pairs']:,}"),
        ("Sparsity", f"{summary['sparsity']:.6%}"),
        (
            "Date range",
            f"{summary['start_date']} to {summary['end_date']} ({summary['days']} days)",
        ),
        (
            "Events per user (median / p99)",
            f"{summary['events_per_user_median']:.0f} / {summary['events_per_user_p99']:.0f}",
        ),
        ("Users with a single event", f"{summary['users_with_one_event_pct']}%"),
        ("Items with 5+ events", f"{summary['items_with_5plus_events']:,}"),
        ("View to add-to-cart rate", f"{summary['view_to_cart_rate']:.2%}"),
        ("View to purchase rate", f"{summary['view_to_purchase_rate']:.2%}"),
    ]
    lines = ["| Metric | Value |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows)]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    summary = summarize(load_events())
    out: Path = settings.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "eda.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = to_markdown(summary)
    (out / "eda.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
