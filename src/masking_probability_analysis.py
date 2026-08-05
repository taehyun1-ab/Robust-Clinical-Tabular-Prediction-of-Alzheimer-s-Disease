"""Robust FT-Transformer masking-probability ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from .config import CONTINUOUS_FEATURES, MASK_PROBABILITIES, N_SPLITS, ROBUST_FT_CONFIG, SEED, TASKS
    from .data_utils import fit_preprocessor, load_data, make_task
    from .models import RobustFTTransformer
    from .train_utils import binary_metrics, predict_ft, set_seed, summarize_fold_results, train_torch_model
except ImportError:
    from config import CONTINUOUS_FEATURES, MASK_PROBABILITIES, N_SPLITS, ROBUST_FT_CONFIG, SEED, TASKS
    from data_utils import fit_preprocessor, load_data, make_task
    from models import RobustFTTransformer
    from train_utils import binary_metrics, predict_ft, set_seed, summarize_fold_results, train_torch_model


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data_path)
    base = dict(ROBUST_FT_CONFIG)
    if args.quick:
        base.update(epochs=3, patience=2)

    rows = []
    for probability in args.probabilities:
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
                    d_model=base["d_model"],
                    n_heads=base["n_heads"],
                    n_layers=base["n_layers"],
                    dropout=base["dropout"],
                    feature_mask_prob=probability,
                )
                model = train_torch_model(
                    model,
                    (torch.tensor(trc, dtype=torch.float32), torch.tensor(trk, dtype=torch.long), torch.tensor(y[train_idx], dtype=torch.long)),
                    (torch.tensor(vac, dtype=torch.float32), torch.tensor(vak, dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long)),
                    batch_size=base["batch_size"], epochs=base["epochs"], patience=base["patience"],
                    learning_rate=base["learning_rate"], weight_decay=base["weight_decay"],
                    device=device, robust=True,
                )
                pred, prob = predict_ft(model, tec, tek, base["batch_size"], device)
                rows.append({"Mask_Probability": probability, "Task": task_name, "Fold": fold, **binary_metrics(y[test_idx], pred, prob)})

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out / "masking_probability_fold_results.csv", index=False)
    summary = summarize_fold_results(fold_df, ["Mask_Probability", "Task"])
    summary.to_csv(out / "masking_probability_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", default="results/masking_probability")
    p.add_argument("--probabilities", type=float, nargs="+", default=list(MASK_PROBABILITIES))
    p.add_argument("--n_splits", type=int, default=N_SPLITS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
