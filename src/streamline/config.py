"""Central settings. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(_env("STREAMLINE_DATA_DIR", str(REPO_ROOT / "data")))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(_env("STREAMLINE_REPORTS_DIR", str(REPO_ROOT / "reports")))
    )
    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:19092"))
    events_topic: str = field(default_factory=lambda: _env("EVENTS_TOPIC", "clickstream"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    mlflow_tracking_uri: str = field(
        default_factory=lambda: _env("MLFLOW_TRACKING_URI", "http://localhost:5001")
    )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_events_csv(self) -> Path:
        return self.raw_dir / "events.csv"

    @property
    def events_parquet(self) -> Path:
        return self.processed_dir / "events.parquet"


def get_settings() -> Settings:
    return Settings()
