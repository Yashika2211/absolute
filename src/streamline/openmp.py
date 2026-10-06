"""Guard against duplicate OpenMP runtimes on macOS (see scripts/fix_macos_openmp.py)."""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

FIX = "uv run python scripts/fix_macos_openmp.py  (or `make install`)"


def check_single_openmp() -> None:
    """Raise before importing faiss if its bundled libomp would clash with torch/lightgbm."""
    if platform.system() != "Darwin":
        return
    spec = importlib.util.find_spec("faiss")
    if spec is None or spec.origin is None:
        return
    lib = Path(spec.origin).parent / ".dylibs" / "libomp.dylib"
    if lib.exists() and not lib.is_symlink():
        raise RuntimeError(
            "faiss bundles its own libomp, which deadlocks with torch/lightgbm in one "
            f"process on macOS. Run: {FIX}"
        )
