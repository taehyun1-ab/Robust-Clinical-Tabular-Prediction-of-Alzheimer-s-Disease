"""Random test-time masking using saved FT/Robust FT checkpoints.

The mask is generated independently for each test subject. Conditions are
0, 1, and 2 masked features, with 100 repeats for masked conditions.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import *
from .data_utils import load_dataframe, make_task_dataframe
from .models import FTTransformer, RobustFTTransformer
from .train_utils import (
    compute_binary_metrics, make_subjectwise_random_mask, predict_ft
)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(checkpoint, device):
    config = checkpoint["best_config"]
    common = dict(
        num_cont_features=checkpoint["num_cont_features"],
        cat_cardinalities=checkpoint["cat_cardinalities"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        num_classes=checkpoint.get("num_classes", 2),
    )
    if checkpoint["model_name"] == "Robust FT-Transformer":
        model = RobustFTTransformer(
            **common,
            feature_mask_prob=checkpoint.get("feature_mask_prob", 0.2),
        )
    else:
        model = FTTransformer(**common)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def preprocess_test(X_test, checkpoint):
    X_test = X_test.copy()
    for col in CONTINUOUS_FEATURES:
        X_test[col] = X_test[col].fillna(checkpoint["continuous_fill_values"][col])
    for col in CATEGORICAL_FEATURES:
        X_test[col] = (
            X_test[col]
            .fillna(checkpoint["categorical_fill_values"][col])
            .astype(str)
        )
    x_cont = checkpoint["scaler"].transform(X_test[CONTINUOUS_FEATURES])
    x_cat = (
        checkpoint["encoder"].transform(X_test[CATEGORICAL_FEATURES]) + 1
    ).astype(int)
    return x_cont, x_cat


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    df = load_dataframe(args.data_path)
    rows = []

    for checkpoint_path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get("model_name") not in {
            "FT-Transformer", "Robust FT-Transformer"
        }:
            continue

        task_name = checkpoint["task_name"]
        X, y = make_task_dataframe(df, task_name)
        test_idx = checkpoint["test_idx"]
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].to_numpy()

        model = build_model(checkpoint, device)
        x_cont, x_cat = preprocess_test(X_test, checkpoint)
        batch_size = checkpoint["best_config"]["batch_size"]

        full_pred, full_prob = predict_ft(
            model, x_cont, x_cat, batch_size, device
        )
        full_metrics = compute_binary_metrics(
            y_test, full_pred, full_prob[:, 1]
        )

        for mask_count in RANDOM_MASK_COUNTS:
            repeats = 1 if mask_count == 0 else args.repeats
            for repeat in range(repeats):
                rng = np.random.default_rng(
                    SEED + checkpoint["fold_idx"] * 100000
                    + mask_count * 1000 + repeat
                )
                external_mask = make_subjectwise_random_mask(
                    len(y_test), len(FEATURES), mask_count, rng
                )
                pred, prob = predict_ft(
                    model, x_cont, x_cat, batch_size, device, external_mask
                )
                metrics = compute_binary_metrics(y_test, pred, prob[:, 1])
                rows.append({
                    "Task": task_name,
                    "Fold": checkpoint["fold_idx"],
                    "Model": checkpoint["model_name"],
                    "Mask_Count": mask_count,
                    "Mask_Ratio": mask_count / len(FEATURES),
                    "Repeat": repeat,
                    **metrics,
                    "Delta_AUROC": full_metrics["AUROC"] - metrics["AUROC"],
                })

    result = pd.DataFrame(rows)
    result.to_csv(out / "random_masking_repeat_results.csv", index=False)
    summary = result.groupby(
        ["Task", "Model", "Mask_Count"], as_index=False
    ).agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),
        Precision_Mean=("Precision_Macro", "mean"),
        Precision_SD=("Precision_Macro", "std"),
        Sensitivity_Mean=("Recall_Macro", "mean"),
        Sensitivity_SD=("Recall_Macro", "std"),
        F1_Mean=("Macro_F1", "mean"),
        F1_SD=("Macro_F1", "std"),
        AUROC_Mean=("AUROC", "mean"),
        AUROC_SD=("AUROC", "std"),
        Delta_AUROC_Mean=("Delta_AUROC", "mean"),
        Delta_AUROC_SD=("Delta_AUROC", "std"),
    )
    summary.to_csv(out / "random_masking_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", default="results/random_masking")
    p.add_argument("--repeats", type=int, default=N_RANDOM_REPEATS)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
