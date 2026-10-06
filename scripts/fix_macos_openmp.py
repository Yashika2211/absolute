"""Make torch, faiss and lightgbm share ONE OpenMP runtime on macOS.

The macOS wheels each bundle their own libomp (lightgbm uses Homebrew's). Two
copies in one process abort with "OMP: Error #15" or deadlock, and serving needs
FAISS + LightGBM (+ torch in training) together. This replaces the bundled
copies with symlinks to Homebrew's libomp. Idempotent; a no-op off macOS.
Re-run after `uv sync` reinstalls those packages (`make install` does this).

Usage: uv run python scripts/fix_macos_openmp.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

HOMEBREW_LIBOMP = Path("/opt/homebrew/opt/libomp/lib/libomp.dylib")
BUNDLED = {"faiss": ".dylibs/libomp.dylib", "torch": "lib/libomp.dylib"}


def main() -> int:
    if platform.system() != "Darwin":
        print("not macOS: nothing to do")
        return 0
    if not HOMEBREW_LIBOMP.exists():
        print("Homebrew libomp not found. Run `brew install libomp` (LightGBM needs it too).")
        return 1
    for package, rel in BUNDLED.items():
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            continue
        lib = Path(spec.origin).parent / rel
        if lib.is_symlink() and lib.resolve() == HOMEBREW_LIBOMP.resolve():
            print(f"{package}: already shared")
            continue
        if lib.exists():
            lib.rename(lib.with_name(lib.name + ".bundled"))
        lib.symlink_to(HOMEBREW_LIBOMP)
        print(f"{package}: {lib} -> {HOMEBREW_LIBOMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
