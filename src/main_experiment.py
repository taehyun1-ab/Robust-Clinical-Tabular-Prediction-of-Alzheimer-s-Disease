"""Exact complete-input experiment structure used for the manuscript.

RF/XGBoost:
    outer stratified 5-fold only; train on 80%, test on 20%.

MLP/FT/Robust FT:
    outer stratified 5-fold; split 10% of each outer training portion
    as validation for validation-AUROC early stopping.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from .config import *
from .data_utils import (
    fit_neural_preprocessor, fit_tree_preprocessor,
    load_dataframe, make_task_dataframe
)
from .models import MLPClassifier, FTTransformer, RobustFTTransformer
from .train_utils import (
    compute_binary_metrics, predict_ft, predict_mlp,
    set_seed, train_neural_model
)


def run(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    df = load_dataframe(args.data_path)
    all_rows = []

    for task_name in TASKS:
        X, y_series = make_task_dataframe(df, task_name)
        y = y_series.to_numpy()
        skf = StratifiedKFold(
            n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED
        )

        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_trainval = X.iloc[trainval_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_trainval = y[trainval_idx]
            y_test = y[test_idx]

            # RF and XGBoost: no inner validation split in the original scripts.
            tree_prep = fit_tree_preprocessor(X_trainval)
            X_tree_train = tree_prep.transform(X_trainval)
            X_tree_test = tree_prep.transform(X_test)

            rf = RandomForestClassifier(
                **RF_CONFIG, random_state=SEED + fold_idx, n_jobs=1
            )
            rf.fit(X_tree_train, y_trainval)
            pred = rf.predict(X_tree_test)
            prob = rf.predict_proba(X_tree_test)
            all_rows.append({
                "Task": task_name, "Fold": fold_idx, "Model": "Random Forest",
                **compute_binary_metrics(y_test, pred, prob[:, 1])
            })

            xgb = XGBClassifier(
                **XGB_CONFIG,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED + fold_idx,
                n_jobs=1,
            )
            xgb.fit(X_tree_train, y_trainval)
            pred = xgb.predict(X_tree_test)
            prob = xgb.predict_proba(X_tree_test)
            all_rows.append({
                "Task": task_name, "Fold": fold_idx, "Model": "XGBoost",
                **compute_binary_metrics(y_test, pred, prob[:, 1])
            })

            # Neural models: 10% of the outer training fold is validation.
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=INNER_VALIDATION_SIZE,
                random_state=SEED + fold_idx,
                stratify=y_trainval,
            )

            # MLP used train-fold mode for missing categorical values.
            mlp_prep = fit_neural_preprocessor(X_train, categorical_strategy="mode")
            _, _, X_mlp_train = mlp_prep.transform(X_train)
            _, _, X_mlp_val = mlp_prep.transform(X_val)
            _, _, X_mlp_test = mlp_prep.transform(X_test)

            mlp = MLPClassifier(
                input_dim=X_mlp_train.shape[1],
                hidden_dim=MLP_CONFIG["hidden_dim"],
                num_layers=MLP_CONFIG["num_layers"],
                dropout=MLP_CONFIG["dropout"],
            )
            mlp, best_epoch, best_auc, best_loss = train_neural_model(
                mlp,
                (
                    torch.tensor(X_mlp_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                ),
                (
                    torch.tensor(X_mlp_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                MLP_CONFIG, device, "mlp"
            )
            pred, prob = predict_mlp(
                mlp, X_mlp_test, MLP_CONFIG["batch_size"], device
            )
            all_rows.append({
                "Task": task_name, "Fold": fold_idx, "Model": "MLP",
                "Best_Epoch": best_epoch, "Best_Val_AUROC": best_auc,
                "Best_Val_Loss": best_loss,
                **compute_binary_metrics(y_test, pred, prob[:, 1])
            })

            # FT and Robust FT used literal "Missing" for missing categorical values.
            ft_prep = fit_neural_preprocessor(X_train, categorical_strategy="missing")
            Xc_train, Xk_train, _ = ft_prep.transform(X_train)
            Xc_val, Xk_val, _ = ft_prep.transform(X_val)
            Xc_test, Xk_test, _ = ft_prep.transform(X_test)

            for model_name, model_type, model in [
                (
                    "FT-Transformer",
                    "ft",
                    FTTransformer(
                        len(CONTINUOUS_FEATURES),
                        ft_prep.cat_cardinalities,
                        d_model=FT_CONFIG["d_model"],
                        n_heads=FT_CONFIG["n_heads"],
                        n_layers=FT_CONFIG["n_layers"],
                        dropout=FT_CONFIG["dropout"],
                    ),
                ),
                (
                    "Robust FT-Transformer",
                    "robust_ft",
                    RobustFTTransformer(
                        len(CONTINUOUS_FEATURES),
                        ft_prep.cat_cardinalities,
                        d_model=FT_CONFIG["d_model"],
                        n_heads=FT_CONFIG["n_heads"],
                        n_layers=FT_CONFIG["n_layers"],
                        dropout=FT_CONFIG["dropout"],
                        feature_mask_prob=MAIN_ROBUST_MASK_PROBABILITY,
                    ),
                ),
            ]:
                model, best_epoch, best_auc, best_loss = train_neural_model(
                    model,
                    (
                        torch.tensor(Xc_train, dtype=torch.float32),
                        torch.tensor(Xk_train, dtype=torch.long),
                        torch.tensor(y_train, dtype=torch.long),
                    ),
                    (
                        torch.tensor(Xc_val, dtype=torch.float32),
                        torch.tensor(Xk_val, dtype=torch.long),
                        torch.tensor(y_val, dtype=torch.long),
                    ),
                    FT_CONFIG, device, model_type
                )
                pred, prob = predict_ft(
                    model, Xc_test, Xk_test, FT_CONFIG["batch_size"], device
                )
                all_rows.append({
                    "Task": task_name, "Fold": fold_idx, "Model": model_name,
                    "Best_Epoch": best_epoch, "Best_Val_AUROC": best_auc,
                    "Best_Val_Loss": best_loss,
                    **compute_binary_metrics(y_test, pred, prob[:, 1])
                })

                torch.save({
                    "model_name": model_name,
                    "task_name": task_name,
                    "fold_idx": fold_idx,
                    "cv_n_splits": CV_N_SPLITS,
                    "seed": SEED,
                    "model_state_dict": model.state_dict(),
                    "best_config": FT_CONFIG,
                    "feature_mask_prob": (
                        MAIN_ROBUST_MASK_PROBABILITY
                        if model_type == "robust_ft" else 0.0
                    ),
                    "num_cont_features": len(CONTINUOUS_FEATURES),
                    "cat_cardinalities": ft_prep.cat_cardinalities,
                    "num_classes": 2,
                    "continuous_cols": CONTINUOUS_FEATURES,
                    "categorical_cols": CATEGORICAL_FEATURES,
                    "feature_names": FEATURES,
                    "label_col": LABEL_COLUMN,
                    "selected_classes": TASKS[task_name]["selected_classes"],
                    "label_map": TASKS[task_name]["label_map"],
                    "target_names": TASKS[task_name]["target_names"],
                    "scaler": ft_prep.scaler,
                    "encoder": ft_prep.encoder,
                    "continuous_fill_values": ft_prep.continuous_fill_values,
                    "categorical_fill_values": ft_prep.categorical_fill_values,
                    "trainval_idx": trainval_idx,
                    "test_idx": test_idx,
                    "train_index": X_train.index.to_numpy(),
                    "val_index": X_val.index.to_numpy(),
                    "test_index": X_test.index.to_numpy(),
                    "best_epoch": best_epoch,
                    "best_val_auc": best_auc,
                    "best_val_loss": best_loss,
                }, checkpoint_dir / f"{model_name.replace(' ', '_')}_{task_name}_fold{fold_idx}.pt")

    fold_df = pd.DataFrame(all_rows)
    fold_df.to_csv(out_dir / "complete_input_fold_results.csv", index=False)

    metrics = [
        "Accuracy", "Precision_Macro", "Recall_Macro",
        "Macro_F1", "Balanced_Accuracy", "AUROC"
    ]
    summary = fold_df.groupby(["Task", "Model"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join([str(v) for v in col if str(v)])
        for col in summary.columns.to_flat_index()
    ]
    summary.to_csv(out_dir / "complete_input_summary.csv", index=False)
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="results/main_experiment")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
