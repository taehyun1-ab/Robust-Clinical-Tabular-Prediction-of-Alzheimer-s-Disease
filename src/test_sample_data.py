"""Fast smoke test for the public synthetic sample.

This validates data loading, task construction, preprocessing, and model
forward passes. It does not reproduce manuscript results.
"""

import argparse
import numpy as np
import torch

from .config import *
from .data_utils import (
    load_dataframe, make_task_dataframe,
    fit_tree_preprocessor, fit_neural_preprocessor
)
from .models import MLPClassifier, FTTransformer, RobustFTTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        default="data/sample/synthetic_clinical_sample_200.csv"
    )
    args = p.parse_args()

    df = load_dataframe(args.data_path)
    print("Rows:", len(df))
    print(df[LABEL_COLUMN].value_counts())

    for task_name in TASKS:
        X, y = make_task_dataframe(df, task_name)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]

        tree_prep = fit_tree_preprocessor(X_train)
        tree_train = tree_prep.transform(X_train)
        tree_test = tree_prep.transform(X_test)
        assert tree_train.shape[1] == len(FEATURES)
        assert tree_test.shape[1] == len(FEATURES)

        mlp_prep = fit_neural_preprocessor(X_train, "mode")
        _, _, mlp_x = mlp_prep.transform(X_test)
        mlp = MLPClassifier(mlp_x.shape[1], **{
            "hidden_dim": MLP_CONFIG["hidden_dim"],
            "num_layers": MLP_CONFIG["num_layers"],
            "dropout": MLP_CONFIG["dropout"],
        })
        mlp_logits = mlp(torch.tensor(mlp_x[:4], dtype=torch.float32))
        assert mlp_logits.shape == (4, 2)

        ft_prep = fit_neural_preprocessor(X_train, "missing")
        xc, xk, _ = ft_prep.transform(X_test)
        ft = FTTransformer(
            len(CONTINUOUS_FEATURES), ft_prep.cat_cardinalities,
            d_model=FT_CONFIG["d_model"],
            n_heads=FT_CONFIG["n_heads"],
            n_layers=FT_CONFIG["n_layers"],
            dropout=FT_CONFIG["dropout"],
        )
        robust = RobustFTTransformer(
            len(CONTINUOUS_FEATURES), ft_prep.cat_cardinalities,
            d_model=FT_CONFIG["d_model"],
            n_heads=FT_CONFIG["n_heads"],
            n_layers=FT_CONFIG["n_layers"],
            dropout=FT_CONFIG["dropout"],
            feature_mask_prob=MAIN_ROBUST_MASK_PROBABILITY,
        )
        xc_t = torch.tensor(xc[:4], dtype=torch.float32)
        xk_t = torch.tensor(xk[:4], dtype=torch.long)
        assert ft(xc_t, xk_t).shape == (4, 2)
        robust.train()
        assert robust(xc_t, xk_t, apply_feature_mask=True).shape == (4, 2)

        print(f"{task_name}: passed")

    print("Sample-data smoke test passed.")


if __name__ == "__main__":
    main()
