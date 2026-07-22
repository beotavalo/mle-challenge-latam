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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from challenge.train import TRACKING_DIR, rehome_tracking_store


def main() -> int:
    """Rehome the snapshot and start the MLflow UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="5000", help="port for the MLflow UI (default: 5000)")
    arguments = parser.parse_args()

    if not TRACKING_DIR.exists():
        print(f"No MLflow store at {TRACKING_DIR}. Run `make train` first.", file=sys.stderr)
        return 1

    rewritten = rehome_tracking_store()
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
