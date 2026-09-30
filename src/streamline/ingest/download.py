"""Download the RetailRocket events.csv from Kaggle into data/raw/.

Usage: uv run python -m streamline.ingest.download
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from streamline.config import get_settings

DATASET = "retailrocket/ecommerce-dataset"
FILENAME = "events.csv"

CREDENTIALS_HELP = """\
Kaggle credentials not found. To set them up:

  1. Sign in at https://www.kaggle.com and open Settings -> API.
  2. Click "Create New Token".
  3. Either save the token string to ~/.kaggle/access_token
     (or export KAGGLE_API_TOKEN=<token>),
     or save the downloaded kaggle.json to ~/.kaggle/kaggle.json.
  4. chmod 600 the file, then re-run `make data`.
"""


def has_kaggle_credentials(home: Path | None = None, env: dict[str, str] | None = None) -> bool:
    env = dict(os.environ) if env is None else env
    kaggle_dir = (home or Path.home()) / ".kaggle"
    if env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        return True
    return (kaggle_dir / "access_token").is_file() or (kaggle_dir / "kaggle.json").is_file()


def unzip_if_needed(raw_dir: Path) -> Path:
    """Kaggle may deliver events.csv or events.csv.zip; normalise to the plain CSV."""
    target = raw_dir / FILENAME
    archive = raw_dir / f"{FILENAME}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extract(FILENAME, raw_dir)
        archive.unlink()
    return target


def download(raw_dir: Path, force: bool = False) -> Path:
    target = raw_dir / FILENAME
    if target.exists() and not force:
        print(f"{target} already exists, skipping (use --force to re-download)")
        return target
    if not has_kaggle_credentials():
        sys.exit(CREDENTIALS_HELP)
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        sys.exit("kaggle CLI not found. Run `uv sync` to install dev dependencies.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, "datasets", "download", "-d", DATASET, "-f", FILENAME, "-p", str(raw_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    path = unzip_if_needed(raw_dir)
    if not path.exists():
        sys.exit(f"Download finished but {path} is missing")
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(get_settings().raw_dir, force=args.force)


if __name__ == "__main__":
    main()
