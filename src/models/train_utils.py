"""Training, prediction, metrics, reproducibility, and masking utilities."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1_score": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "AUROC": auroc,
    }


def summarize_fold_results(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["Accuracy", "Precision", "Sensitivity", "F1_score", "AUROC"]
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            row[f"{metric}_Mean"] = group[metric].mean()
            row[f"{metric}_SD"] = group[metric].std(ddof=1)
        row["N_Folds"] = group["Fold"].nunique() if "Fold" in group else len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def train_torch_model(
    model,
    train_tensors: tuple,
    val_tensors: tuple,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    device,
    robust: bool = False,
):
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_count = 0

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 2:
                x, y = (item.to(device) for item in batch)
                logits = model(x)
            else:
                x_cont, x_cat, y = (item.to(device) for item in batch)
                logits = (
                    model(x_cont, x_cat, apply_feature_mask=True)
                    if robust
                    else model(x_cont, x_cat)
                )
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    x, y = (item.to(device) for item in batch)
                    logits = model(x)
                else:
                    x_cont, x_cat, y = (item.to(device) for item in batch)
                    logits = (
                        model(x_cont, x_cat, apply_feature_mask=False)
                        if robust
                        else model(x_cont, x_cat)
                    )
                val_losses.append(criterion(logits, y).item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_state)
    return model


def predict_mlp(model, x, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    probabilities = []
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probabilities)
    return probs.argmax(axis=1), probs[:, 1]


def predict_ft(model, x_cont, x_cat, batch_size, device, feature_mask=None):
    mask = (
        np.zeros((len(x_cont), x_cont.shape[1] + x_cat.shape[1]), dtype=bool)
        if feature_mask is None
        else feature_mask
    )
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_cont, dtype=torch.float32),
            torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch_cont, batch_cat, batch_mask in loader:
            batch_cont = batch_cont.to(device)
            batch_cat = batch_cat.to(device)
            batch_mask = batch_mask.to(device)
            if hasattr(model, "feature_mask_token"):
                logits = model(
                    batch_cont,
                    batch_cat,
                    apply_feature_mask=False,
                    external_feature_mask=batch_mask,
                )
            else:
                logits = model(batch_cont, batch_cat)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probabilities)
    return probs.argmax(axis=1), probs[:, 1]


def random_mask_matrix(n_rows: int, n_features: int, mask_count: int, rng):
    mask = np.zeros((n_rows, n_features), dtype=bool)
    for row in range(n_rows):
        selected = rng.choice(n_features, size=mask_count, replace=False)
        mask[row, selected] = True
    return mask


def replace_matrix_values(x, mask, replacement_values):
    out = x.copy()
    row_idx, col_idx = np.where(mask)
    out[row_idx, col_idx] = replacement_values[col_idx]
    return out


def save_checkpoint(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
