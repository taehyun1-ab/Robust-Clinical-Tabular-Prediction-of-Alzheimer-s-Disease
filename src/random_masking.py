"""Subject-wise random one- and two-feature masking evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from .config import (
        CONTINUOUS_FEATURES,
        N_RANDOM_REPEATS,
        N_SPLITS,
        RANDOM_MASK_COUNTS,
        ROBUST_FT_CONFIG,
        SEED,
        TASKS,
    )
    from .data_utils import fit_preprocessor, load_data, make_task
    from .models import RobustFTTransformer
    from .train_utils import (
        binary_metrics,
        predict_ft,
        random_mask_matrix,
        set_seed,
        summarize_fold_results,
        train_torch_model,
    )
except ImportError:
    from config import (
        CONTINUOUS_FEATURES,
        N_RANDOM_REPEATS,
        N_SPLITS,
        RANDOM_MASK_COUNTS,
        ROBUST_FT_CONFIG,
        SEED,
        TASKS,
    )
    from data_utils import fit_preprocessor, load_data, make_task
    from models import RobustFTTransformer
    from train_utils import (
        binary_metrics,
        predict_ft,
        random_mask_matrix,
        set_seed,
        summarize_fold_results,
        train_torch_model,
    )


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data_path)

    config = dict(ROBUST_FT_CONFIG)
    if args.quick:
        config.update(epochs=3, patience=2)
        repeats = 2
    else:
        repeats = args.repeats

    rows = []
    for task_name in TASKS:
        x, y = make_task(df, task_name)
        cv = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed)
        for fold, (trainval_idx, test_idx) in enumerate(cv.split(x, y), 1):
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=0.10, stratify=y[trainval_idx],
                random_state=args.seed + fold,
            )
            prep = fit_preprocessor(x.iloc[train_idx])
            trc, trk, _ = prep.transform(x.iloc[train_idx])
            vac, vak, _ = prep.transform(x.iloc[val_idx])
            tec, tek, _ = prep.transform(x.iloc[test_idx])

            model = RobustFTTransformer(
                num_cont_features=len(CONTINUOUS_FEATURES),
                cat_cardinalities=prep.cat_cardinalities,
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_layers=config["n_layers"],
                dropout=config["dropout"],
                feature_mask_prob=config["feature_mask_prob"],
            )
            model = train_torch_model(
                model,
                (
                    torch.tensor(trc, dtype=torch.float32),
                    torch.tensor(trk, dtype=torch.long),
                    torch.tensor(y[train_idx], dtype=torch.long),
                ),
                (
                    torch.tensor(vac, dtype=torch.float32),
                    torch.tensor(vak, dtype=torch.long),
                    torch.tensor(y[val_idx], dtype=torch.long),
                ),
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                patience=config["patience"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                device=device,
                robust=True,
            )

            full_pred, full_prob = predict_ft(
                model, tec, tek, config["batch_size"], device
            )
            full_metrics = binary_metrics(y[test_idx], full_pred, full_prob)

            for mask_count in args.mask_counts:
                for repeat in range(1, repeats + 1):
                    rng = np.random.default_rng(
                        args.seed + fold * 10000 + mask_count * 1000 + repeat
                    )
                    mask = random_mask_matrix(
                        len(test_idx), tec.shape[1] + tek.shape[1], mask_count, rng
                    )
                    pred, prob = predict_ft(
                        model, tec, tek, config["batch_size"], device, mask
                    )
                    metrics = binary_metrics(y[test_idx], pred, prob)
                    rows.append(
                        {
                            "Task": task_name,
                            "Fold": fold,
                            "Repeat": repeat,
                            "Model": "Robust_FT_Transformer",
                            "Masked_Features": mask_count,
                            **metrics,
                            "Delta_AUROC": full_metrics["AUROC"] - metrics["AUROC"],
                        }
                    )

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "random_masking_repeat_results.csv", index=False)
    summary = (
        raw.groupby(["Task", "Model", "Masked_Features"], as_index=False)
        .agg(
            Accuracy_Mean=("Accuracy", "mean"),
            Accuracy_SD=("Accuracy", "std"),
            Precision_Mean=("Precision", "mean"),
            Precision_SD=("Precision", "std"),
            Sensitivity_Mean=("Sensitivity", "mean"),
            Sensitivity_SD=("Sensitivity", "std"),
            F1_score_Mean=("F1_score", "mean"),
            F1_score_SD=("F1_score", "std"),
            AUROC_Mean=("AUROC", "mean"),
            AUROC_SD=("AUROC", "std"),
            Delta_AUROC_Mean=("Delta_AUROC", "mean"),
            Delta_AUROC_SD=("Delta_AUROC", "std"),
        )
    )
    summary.to_csv(out / "random_masking_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="results/random_masking")
    p.add_argument("--n_splits", type=int, default=N_SPLITS)
    p.add_argument("--repeats", type=int, default=N_RANDOM_REPEATS)
    p.add_argument("--mask_counts", type=int, nargs="+", default=list(RANDOM_MASK_COUNTS))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
