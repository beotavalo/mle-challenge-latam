"""Open the versioned MLflow snapshot on any checkout.

MLflow's file store writes *absolute* ``file://`` URIs into its metadata, so the
``mlruns/`` snapshot committed with this repository points at the machine where it
was produced. This script rewrites those URIs to the current checkout and then
launches the MLflow UI, so a reviewer can browse the experiment history and the
model registry without retraining anything.

Usage:
    make mlflow-ui           # or: python scripts/mlflow_ui.py [--port 5000]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = REPOSITORY_ROOT / "mlruns"

#: Any absolute file URI that ends in the tracking directory, whatever machine and
#: path separator produced it.
_FOREIGN_URI = re.compile(r"file://[^\s'\"]*?/mlruns")


def rehome_metadata(tracking_dir: Path = TRACKING_DIR) -> int:
    """Point every ``file://`` URI in the store at ``tracking_dir``.

    Args:
        tracking_dir: the local ``mlruns`` directory.

    Returns:
        Number of metadata files rewritten.
    """
    local_uri = tracking_dir.as_uri()
    rewritten = 0
    for meta in tracking_dir.rglob("meta.yaml"):
        original = meta.read_text(encoding="utf-8")
        updated = _FOREIGN_URI.sub(local_uri, original)
        if updated != original:
            meta.write_text(updated, encoding="utf-8")
            rewritten += 1
    return rewritten


def main() -> int:
    """Rehome the snapshot and start the MLflow UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="5000", help="port for the MLflow UI (default: 5000)")
    arguments = parser.parse_args()

    if not TRACKING_DIR.exists():
        print(f"No MLflow store at {TRACKING_DIR}. Run `make train` first.", file=sys.stderr)
        return 1

    rewritten = rehome_metadata()
    print(f"Rehomed {rewritten} metadata file(s) to {TRACKING_DIR}")
    print(f"MLflow UI -> http://127.0.0.1:{arguments.port}")

    return subprocess.call(  # noqa: S603 - fixed argument vector, no shell
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            TRACKING_DIR.as_uri(),
            "--port",
            arguments.port,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
