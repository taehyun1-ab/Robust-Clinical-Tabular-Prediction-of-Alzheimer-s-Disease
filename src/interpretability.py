"""Compact SHAP and CLS-to-feature attention analysis.

This public script intentionally writes only aggregate outputs by default.
It follows the uploaded analysis logic: pooled out-of-fold SHAP importance,
single-feature masking, and last-layer CLS-to-feature attention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from .config import CONTINUOUS_FEATURES, FEATURES, N_SPLITS, ROBUST_FT_CONFIG, SEED, TASKS
    from .data_utils import fit_preprocessor, load_data, make_task
    from .models import RobustFTTransformer
    from .train_utils import set_seed, train_torch_model
except ImportError:
    from config import CONTINUOUS_FEATURES, FEATURES, N_SPLITS, ROBUST_FT_CONFIG, SEED, TASKS
    from data_utils import fit_preprocessor, load_data, make_task
    from models import RobustFTTransformer
    from train_utils import set_seed, train_torch_model


def predict_processed(model, matrix, n_cont, batch_size, device):
    matrix = np.asarray(matrix, dtype=float)
    x_cont = torch.tensor(matrix[:, :n_cont], dtype=torch.float32, device=device)
    x_cat = torch.tensor(np.rint(matrix[:, n_cont:]).astype(int), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x_cont, x_cat, apply_feature_mask=False)
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def last_layer_cls_attention(model, x_cont, x_cat, device):
    model.eval()
    with torch.no_grad():
        cont = torch.tensor(x_cont, dtype=torch.float32, device=device)
        cat = torch.tensor(x_cat, dtype=torch.long, device=device)
        tokens = model.tokenize(cont, cat)
        cls = model.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        for layer_idx, layer in enumerate(model.transformer.layers):
            if layer_idx == len(model.transformer.layers) - 1:
                attn_out, weights = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=False
                )
                x = layer.norm1(x + layer.dropout1(attn_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff))
            else:
                x = layer(x)
        return weights[:, :, 0, 1:].mean(dim=(0, 1)).cpu().numpy()


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data_path)

    config = dict(ROBUST_FT_CONFIG)
    if args.quick:
        config.update(epochs=3, patience=2)
        explain_size = 10
        background_size = 10
    else:
        explain_size = args.explain_size
        background_size = args.background_size

    shap_rows, attention_rows = [], []
    pooled_values = {task: [] for task in TASKS}
    pooled_data = {task: [] for task in TASKS}

    for task_name in TASKS:
        x, y = make_task(df, task_name)
        cv = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed)
        for fold, (trainval_idx, test_idx) in enumerate(cv.split(x, y), 1):
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=0.10, stratify=y[trainval_idx],
                random_state=args.seed + fold,
            )
            prep = fit_preprocessor(x.iloc[train_idx])
            trc, trk, tra = prep.transform(x.iloc[train_idx])
            vac, vak, _ = prep.transform(x.iloc[val_idx])
            tec, tek, tea = prep.transform(x.iloc[test_idx])

            model = RobustFTTransformer(
                num_cont_features=len(CONTINUOUS_FEATURES),
                cat_cardinalities=prep.cat_cardinalities,
                d_model=config["d_model"], n_heads=config["n_heads"],
                n_layers=config["n_layers"], dropout=config["dropout"],
                feature_mask_prob=config["feature_mask_prob"],
            )
            model = train_torch_model(
                model,
                (torch.tensor(trc, dtype=torch.float32), torch.tensor(trk, dtype=torch.long), torch.tensor(y[train_idx], dtype=torch.long)),
                (torch.tensor(vac, dtype=torch.float32), torch.tensor(vak, dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long)),
                batch_size=config["batch_size"], epochs=config["epochs"], patience=config["patience"],
                learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
                device=device, robust=True,
            )

            attention = last_layer_cls_attention(model, tec, tek, device)
            for feature, value in zip(FEATURES, attention):
                attention_rows.append({"Task": task_name, "Fold": fold, "Feature": feature, "Attention": value})

            rng = np.random.default_rng(args.seed + fold)
            bg_idx = rng.choice(len(tra), size=min(background_size, len(tra)), replace=False)
            ex_idx = rng.choice(len(tea), size=min(explain_size, len(tea)), replace=False)
            background = tra[bg_idx]
            explain = tea[ex_idx]

            explainer = shap.Explainer(
                lambda z: predict_processed(model, z, len(CONTINUOUS_FEATURES), config["batch_size"], device),
                shap.maskers.Independent(background),
                algorithm="permutation",
                feature_names=FEATURES,
            )
            explanation = explainer(
                explain,
                max_evals=2 * len(FEATURES) + 1,
                batch_size=config["batch_size"],
            )
            values = np.asarray(explanation.values)
            if values.ndim == 3:
                values = values[:, :, -1]
            pooled_values[task_name].append(values)
            pooled_data[task_name].append(explain)

            mean_abs = np.abs(values).mean(axis=0)
            for feature, value in zip(FEATURES, mean_abs):
                shap_rows.append({"Task": task_name, "Fold": fold, "Feature": feature, "Mean_Abs_SHAP": value})

    shap_df = pd.DataFrame(shap_rows)
    attn_df = pd.DataFrame(attention_rows)
    shap_summary = shap_df.groupby(["Task", "Feature"], as_index=False).agg(
        Mean_Abs_SHAP=("Mean_Abs_SHAP", "mean"),
        SD_Across_Folds=("Mean_Abs_SHAP", "std"),
    )
    attn_summary = attn_df.groupby(["Task", "Feature"], as_index=False).agg(
        Mean_Attention=("Attention", "mean"),
        SD_Across_Folds=("Attention", "std"),
    )
    shap_summary.to_csv(out / "shap_summary.csv", index=False)
    attn_summary.to_csv(out / "attention_summary.csv", index=False)

    pivot = attn_summary.pivot(index="Feature", columns="Task", values="Mean_Attention")
    plt.figure(figsize=(7, 5))
    image = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(image, label="Mean CLS-to-feature attention")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(fig_dir / "attention_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    for task_name in TASKS:
        if not pooled_values[task_name]:
            continue
        values = np.vstack(pooled_values[task_name])
        data = np.vstack(pooled_data[task_name])
        explanation = shap.Explanation(values=values, data=data, feature_names=FEATURES)
        shap.plots.beeswarm(explanation, max_display=len(FEATURES), show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_beeswarm_{task_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(shap_summary.to_string(index=False))
    print(attn_summary.to_string(index=False))
    return shap_summary, attn_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="results/interpretability")
    p.add_argument("--n_splits", type=int, default=N_SPLITS)
    p.add_argument("--background_size", type=int, default=50)
    p.add_argument("--explain_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
