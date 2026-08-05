"""Complete-input five-fold evaluation for all five manuscript models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

try:
    from .config import (
        CONTINUOUS_FEATURES,
        FT_CONFIG,
        MLP_CONFIG,
        N_SPLITS,
        RF_CONFIG,
        ROBUST_FT_CONFIG,
        SEED,
        TASKS,
        XGB_CONFIG,
    )
    from .data_utils import fit_preprocessor, load_data, make_task
    from .models import FTTransformer, MLPClassifier, RobustFTTransformer
    from .train_utils import (
        binary_metrics,
        predict_ft,
        predict_mlp,
        save_checkpoint,
        set_seed,
        summarize_fold_results,
        train_torch_model,
    )
except ImportError:
    from config import (
        CONTINUOUS_FEATURES,
        FT_CONFIG,
        MLP_CONFIG,
        N_SPLITS,
        RF_CONFIG,
        ROBUST_FT_CONFIG,
        SEED,
        TASKS,
        XGB_CONFIG,
    )
    from data_utils import fit_preprocessor, load_data, make_task
    from models import FTTransformer, MLPClassifier, RobustFTTransformer
    from train_utils import (
        binary_metrics,
        predict_ft,
        predict_mlp,
        save_checkpoint,
        set_seed,
        summarize_fold_results,
        train_torch_model,
    )


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    fold_rows = []

    rf_config = dict(RF_CONFIG)
    xgb_config = dict(XGB_CONFIG)
    mlp_config = dict(MLP_CONFIG)
    ft_config = dict(FT_CONFIG)
    robust_config = dict(ROBUST_FT_CONFIG)

    if args.quick:
        rf_config["n_estimators"] = 20
        xgb_config["n_estimators"] = 30
        mlp_config.update(epochs=3, patience=2)
        ft_config.update(epochs=3, patience=2)
        robust_config.update(epochs=3, patience=2)

    for task_name in TASKS:
        x, y = make_task(df, task_name)
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        )

        for fold, (trainval_idx, test_idx) in enumerate(splitter.split(x, y), 1):
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=0.10,
                random_state=args.seed + fold,
                stratify=y[trainval_idx],
            )

            preprocessor = fit_preprocessor(x.iloc[train_idx])
            x_train_cont, x_train_cat, x_train_all = preprocessor.transform(x.iloc[train_idx])
            x_val_cont, x_val_cat, x_val_all = preprocessor.transform(x.iloc[val_idx])
            x_test_cont, x_test_cat, x_test_all = preprocessor.transform(x.iloc[test_idx])

            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

            classical = {
                "Random_Forest": RandomForestClassifier(
                    **rf_config,
                    random_state=args.seed + fold,
                    n_jobs=1,
                ),
                "XGBoost": XGBClassifier(
                    **xgb_config,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=args.seed + fold,
                    n_jobs=1,
                ),
            }
            for model_name, model in classical.items():
                model.fit(x_train_all, y_train)
                pred = model.predict(x_test_all)
                prob = model.predict_proba(x_test_all)[:, 1]
                fold_rows.append(
                    {"Task": task_name, "Fold": fold, "Model": model_name, **binary_metrics(y_test, pred, prob)}
                )

            mlp = MLPClassifier(
                input_dim=x_train_all.shape[1],
                hidden_dim=mlp_config["hidden_dim"],
                num_layers=mlp_config["num_layers"],
                dropout=mlp_config["dropout"],
            )
            mlp = train_torch_model(
                mlp,
                (
                    torch.tensor(x_train_all, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                ),
                (
                    torch.tensor(x_val_all, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                batch_size=mlp_config["batch_size"],
                epochs=mlp_config["epochs"],
                patience=mlp_config["patience"],
                learning_rate=mlp_config["learning_rate"],
                weight_decay=mlp_config["weight_decay"],
                device=device,
            )
            pred, prob = predict_mlp(
                mlp, x_test_all, mlp_config["batch_size"], device
            )
            fold_rows.append(
                {"Task": task_name, "Fold": fold, "Model": "MLP", **binary_metrics(y_test, pred, prob)}
            )

            for model_name, model_class, config, robust in [
                ("FT_Transformer", FTTransformer, ft_config, False),
                ("Robust_FT_Transformer", RobustFTTransformer, robust_config, True),
            ]:
                kwargs = dict(
                    num_cont_features=len(CONTINUOUS_FEATURES),
                    cat_cardinalities=preprocessor.cat_cardinalities,
                    d_model=config["d_model"],
                    n_heads=config["n_heads"],
                    n_layers=config["n_layers"],
                    dropout=config["dropout"],
                )
                if robust:
                    kwargs["feature_mask_prob"] = config["feature_mask_prob"]
                model = model_class(**kwargs)
                model = train_torch_model(
                    model,
                    (
                        torch.tensor(x_train_cont, dtype=torch.float32),
                        torch.tensor(x_train_cat, dtype=torch.long),
                        torch.tensor(y_train, dtype=torch.long),
                    ),
                    (
                        torch.tensor(x_val_cont, dtype=torch.float32),
                        torch.tensor(x_val_cat, dtype=torch.long),
                        torch.tensor(y_val, dtype=torch.long),
                    ),
                    batch_size=config["batch_size"],
                    epochs=config["epochs"],
                    patience=config["patience"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    device=device,
                    robust=robust,
                )
                pred, prob = predict_ft(
                    model, x_test_cont, x_test_cat, config["batch_size"], device
                )
                fold_rows.append(
                    {"Task": task_name, "Fold": fold, "Model": model_name, **binary_metrics(y_test, pred, prob)}
                )

                save_checkpoint(
                    checkpoint_dir / f"{model_name}_{task_name}_fold{fold}.pt",
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "task": task_name,
                        "fold": fold,
                        "config": config,
                        "preprocessor": preprocessor,
                        "cat_cardinalities": preprocessor.cat_cardinalities,
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                        "test_idx": test_idx,
                    },
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "complete_input_fold_results.csv", index=False)
    summary_df = summarize_fold_results(fold_df, ["Task", "Model"])
    summary_df.to_csv(output_dir / "complete_input_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    return summary_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="results/run")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
