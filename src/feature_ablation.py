"""Single-feature ablation from saved Robust FT-Transformer checkpoints."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import *
from .data_utils import load_dataframe, make_task_dataframe
from .models import RobustFTTransformer
from .random_masking import load_checkpoint, preprocess_test
from .train_utils import compute_binary_metrics, predict_ft


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataframe(args.data_path)
    rows = []

    for checkpoint_path in sorted(Path(args.checkpoint_dir).glob("*.pt")):
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint.get("model_name") != "Robust FT-Transformer":
            continue

        task_name = checkpoint["task_name"]
        X, y = make_task_dataframe(df, task_name)
        test_idx = checkpoint["test_idx"]
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].to_numpy()
        x_cont, x_cat = preprocess_test(X_test, checkpoint)

        config = checkpoint["best_config"]
        model = RobustFTTransformer(
            checkpoint["num_cont_features"],
            checkpoint["cat_cardinalities"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            feature_mask_prob=checkpoint.get("feature_mask_prob", 0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()

        full_pred, full_prob = predict_ft(
            model, x_cont, x_cat, config["batch_size"], device
        )
        full_metrics = compute_binary_metrics(
            y_test, full_pred, full_prob[:, 1]
        )

        for feature_idx, feature_name in enumerate(FEATURES):
            mask = np.zeros((len(y_test), len(FEATURES)), dtype=bool)
            mask[:, feature_idx] = True
            pred, prob = predict_ft(
                model, x_cont, x_cat, config["batch_size"], device, mask
            )
            ablated = compute_binary_metrics(y_test, pred, prob[:, 1])
            rows.append({
                "Task": task_name,
                "Fold": checkpoint["fold_idx"],
                "Feature": feature_name,
                "Complete_AUROC": full_metrics["AUROC"],
                "Ablated_AUROC": ablated["AUROC"],
                "Delta_AUROC": full_metrics["AUROC"] - ablated["AUROC"],
            })

    result = pd.DataFrame(rows)
    result.to_csv(out / "single_feature_ablation_fold_results.csv", index=False)
    summary = result.groupby(["Task", "Feature"], as_index=False).agg(
        Complete_AUROC_Mean=("Complete_AUROC", "mean"),
        Ablated_AUROC_Mean=("Ablated_AUROC", "mean"),
        Ablated_AUROC_SD=("Ablated_AUROC", "std"),
        Delta_AUROC_Mean=("Delta_AUROC", "mean"),
        Delta_AUROC_SD=("Delta_AUROC", "std"),
    )
    summary.to_csv(out / "single_feature_ablation_summary.csv", index=False)
    print(summary.sort_values(
        ["Task", "Delta_AUROC_Mean"], ascending=[True, False]
    ).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", default="results/feature_ablation")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
