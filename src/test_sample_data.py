"""End-to-end smoke test using the public synthetic sample dataset.

This is a functionality test only. Its metrics must not be compared with or
used to reproduce the manuscript results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .data_utils import load_data
    from .config import FEATURES, LABEL_COLUMN
except ImportError:
    from data_utils import load_data
    from config import FEATURES, LABEL_COLUMN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/sample/synthetic_clinical_sample.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results/sample_test",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Sample dataset not found: {data_path}")

    df = load_data(str(data_path))
    print(f"Loaded {len(df)} rows.")
    print("Columns:", FEATURES + [LABEL_COLUMN])
    print("Diagnosis counts:")
    print(df[LABEL_COLUMN].value_counts().to_string())

    command = [
        sys.executable,
        "-m",
        "src.main_experiment",
        "--data_path",
        str(data_path),
        "--output_dir",
        args.output_dir,
        "--n_splits",
        "2",
        "--quick",
        "--cpu",
    ]
    print("\nRunning smoke test:")
    print(" ".join(command))
    subprocess.run(command, check=True)

    summary = Path(args.output_dir) / "complete_input_summary.csv"
    if not summary.exists():
        raise RuntimeError(f"Expected output was not created: {summary}")

    print(f"\nSmoke test completed successfully: {summary}")


if __name__ == "__main__":
    main()
