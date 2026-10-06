"""Summarise Locust runs (client view) + server stage timings into reports/loadtest.{json,md}.

Usage: uv run python -m streamline.serving.loadtest_report RUN_NAME [RUN_NAME ...]
where each RUN_NAME has reports/loadtest/<RUN_NAME>_stats.csv and _stages.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from streamline.config import get_settings


def client_stats(stats_csv: Path) -> dict[str, float]:
    with stats_csv.open() as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    requests = int(row["Request Count"])
    return {
        "requests": requests,
        "failures": int(row["Failure Count"]),
        "rps": round(float(row["Requests/s"]), 1),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
        "max_ms": float(row["Max Response Time"]),
    }


def stage_stats(stages_json: Path) -> dict[str, Any]:
    data = json.loads(stages_json.read_text())
    stages = {
        name: {
            "p50_ms": round(float(np.percentile(v, 50)), 2),
            "p99_ms": round(float(np.percentile(v, 99)), 2),
        }
        for name, v in sorted(data["stages"].items())
        if v
    }
    return {"sources": data["sources"], "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--meta", default="{}", help="JSON with per-run settings, keyed by run")
    args = parser.parse_args()
    meta = json.loads(args.meta)
    root = get_settings().reports_dir
    runs = {}
    for run in args.runs:
        runs[run] = {
            **meta.get(run, {}),
            "client": client_stats(root / "loadtest" / f"{run}_stats.csv"),
            "server": stage_stats(root / "loadtest" / f"{run}_stages.json"),
        }
    (root / "loadtest.json").write_text(json.dumps(runs, indent=2) + "\n")
    lines = [
        "| Target load | Achieved (req/s) | Requests | Failures "
        "| p50 (ms) | p95 (ms) | p99 (ms) | Server p99, warm path (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs.values():
        c, s = r["client"], r["server"]["stages"]
        target = r.get("target_rps", 0)
        label = f"{target} req/s" if target else f"saturation ({r.get('users')} users)"
        lines.append(
            f"| {label} | {c['rps']} | {c['requests']:,} | {c['failures']} "
            f"| {c['p50_ms']:.0f} | {c['p95_ms']:.0f} | {c['p99_ms']:.0f} "
            f"| {s.get('ranked:total', {}).get('p99_ms', float('nan')):.1f} |"
        )
    (root / "loadtest.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
