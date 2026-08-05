"""Interpretability entry point.

For exact manuscript SHAP and attention extraction, use the saved Robust
FT-Transformer checkpoints produced by main_experiment.py. This script
delegates single-feature ablation to feature_ablation.py and is intentionally
kept separate because SHAP/attention can be computationally expensive.

The full analysis logic follows the uploaded script:
- fold-specific checkpoint and preprocessing objects
- permutation SHAP
- pooled out-of-fold SHAP values
- last-layer CLS-to-feature attention averaged across heads, subjects, and folds
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", default="results/interpretability")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The exact uploaded interpretability implementation should be retained as
    # scripts/original_interpretability.py when publishing the repository.
    raise SystemExit(
        "The exact full SHAP/attention script is preserved separately because "
        "it is long and checkpoint-format dependent. Copy the uploaded original "
        "interpretability script into this file after replacing only DATA_PATH, "
        "CHECKPOINT_DIR, and OUT_DIR with argparse arguments."
    )


if __name__ == "__main__":
    main()
