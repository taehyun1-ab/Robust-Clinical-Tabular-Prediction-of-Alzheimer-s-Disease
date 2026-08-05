"""Train Robust FT-Transformer with p=0.1, 0.2, and 0.3.

This keeps the original outer 5-fold split, inner 10% validation split,
train-fold preprocessing, class-weighted cross-entropy, and
validation-AUROC early stopping.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import *
from .data_utils import load_dataframe, make_task_dataframe, fit_neural_preprocessor
from .models import RobustFTTransformer
from .train_utils import compute_binary_metrics, predict_ft, set_seed, train_neural_model


def run(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataframe(args.data_path)
    rows = []

    for mask_prob in MASK_PROBABILITIES:
        for task_name in TASKS:
            X, y_series = make_task_dataframe(df, task_name)
            y = y_series.to_numpy()
            skf = StratifiedKFold(CV_N_SPLITS, shuffle=True, random_state=SEED)

            for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y), 1):
                X_trainval, X_test = X.iloc[trainval_idx].copy(), X.iloc[test_idx].copy()
                y_trainval, y_test = y[trainval_idx], y[test_idx]
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval,
                    test_size=INNER_VALIDATION_SIZE,
                    random_state=SEED + fold_idx,
                    stratify=y_trainval,
                )
                prep = fit_neural_preprocessor(X_train, "missing")
                trc, trk, _ = prep.transform(X_train)
                vac, vak, _ = prep.transform(X_val)
                tec, tek, _ = prep.transform(X_test)

                model = RobustFTTransformer(
                    len(CONTINUOUS_FEATURES), prep.cat_cardinalities,
                    d_model=FT_CONFIG["d_model"],
                    n_heads=FT_CONFIG["n_heads"],
                    n_layers=FT_CONFIG["n_layers"],
                    dropout=FT_CONFIG["dropout"],
                    feature_mask_prob=mask_prob,
                )
                model, best_epoch, best_auc, best_loss = train_neural_model(
                    model,
                    (
                        torch.tensor(trc, dtype=torch.float32),
                        torch.tensor(trk, dtype=torch.long),
                        torch.tensor(y_train, dtype=torch.long),
                    ),
                    (
                        torch.tensor(vac, dtype=torch.float32),
                        torch.tensor(vak, dtype=torch.long),
                        torch.tensor(y_val, dtype=torch.long),
                    ),
                    FT_CONFIG, device, "robust_ft"
                )
                pred, prob = predict_ft(
                    model, tec, tek, FT_CONFIG["batch_size"], device
                )
                rows.append({
                    "Mask_Probability": mask_prob,
                    "Task": task_name,
                    "Fold": fold_idx,
                    "Best_Epoch": best_epoch,
                    "Best_Val_AUROC": best_auc,
                    "Best_Val_Loss": best_loss,
                    **compute_binary_metrics(y_test, pred, prob[:, 1]),
                })

    result = pd.DataFrame(rows)
    result.to_csv(out / "masking_probability_fold_results.csv", index=False)
    summary = result.groupby(
        ["Mask_Probability", "Task"], as_index=False
    ).agg(
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_SD=("Accuracy", "std"),
        Macro_F1_Mean=("Macro_F1", "mean"),
        Macro_F1_SD=("Macro_F1", "std"),
        AUROC_Mean=("AUROC", "mean"),
        AUROC_SD=("AUROC", "std"),
    )
    summary.to_csv(out / "masking_probability_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="results/masking_probability")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
