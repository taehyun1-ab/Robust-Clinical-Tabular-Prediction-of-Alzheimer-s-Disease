import copy
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score
)
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(y_true, y_pred, y_prob_class1):
    try:
        auroc = roc_auc_score(y_true, y_prob_class1)
    except Exception:
        auroc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall_Macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Macro_F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Weighted_F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "AUROC": auroc,
    }


def class_weight_tensor(y_train, device):
    counts = pd.Series(y_train).value_counts().sort_index().values
    weights = len(y_train) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_neural_model(
    model, train_tensors, val_tensors, config, device, model_type
):
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors), batch_size=batch_size, shuffle=False
    )

    model = model.to(device)
    y_train = train_tensors[-1].cpu().numpy()
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weight_tensor(y_train, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_auc = -np.inf
    best_val_loss = np.inf
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for batch in train_loader:
            batch = [item.to(device) for item in batch]
            optimizer.zero_grad()

            if model_type == "mlp":
                x, y = batch
                logits = model(x)
            elif model_type == "ft":
                x_cont, x_cat, y = batch
                logits = model(x_cont, x_cat)
            elif model_type == "robust_ft":
                x_cont, x_cat, y = batch
                logits = model(x_cont, x_cat, apply_feature_mask=True)
            else:
                raise ValueError(model_type)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(device) for item in batch]
                if model_type == "mlp":
                    x, y = batch
                    logits = model(x)
                elif model_type == "ft":
                    x_cont, x_cat, y = batch
                    logits = model(x_cont, x_cat)
                else:
                    x_cont, x_cat, y = batch
                    logits = model(x_cont, x_cat, apply_feature_mask=False)

                loss = criterion(logits, y)
                probs = torch.softmax(logits, dim=1)
                val_losses.append(loss.item())
                val_labels.extend(y.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_loss = float(np.mean(val_losses))
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            val_auc = -np.inf

        # Original scripts used validation AUROC as the primary early-stopping criterion.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_auc, best_val_loss


def predict_mlp(model, x, batch_size, device):
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs)
    return probs.argmax(axis=1), probs


def predict_ft(model, x_cont, x_cat, batch_size, device, external_mask=None):
    if external_mask is None:
        external_mask = np.zeros(
            (len(x_cont), x_cont.shape[1] + x_cat.shape[1]), dtype=bool
        )
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_cont, dtype=torch.float32),
            torch.tensor(x_cat, dtype=torch.long),
            torch.tensor(external_mask, dtype=torch.bool),
        ),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    probs = []
    with torch.no_grad():
        for xc, xk, xm in loader:
            xc, xk, xm = xc.to(device), xk.to(device), xm.to(device)
            if hasattr(model, "feature_mask_token"):
                logits = model(
                    xc, xk,
                    apply_feature_mask=False,
                    external_feature_mask=xm
                )
            else:
                logits = model(xc, xk)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs)
    return probs.argmax(axis=1), probs


def make_subjectwise_random_mask(n_subjects, n_features, mask_count, rng):
    mask = np.zeros((n_subjects, n_features), dtype=bool)
    if mask_count == 0:
        return mask
    for i in range(n_subjects):
        idx = rng.choice(n_features, size=mask_count, replace=False)
        mask[i, idx] = True
    return mask


def replace_masked_values(x, mask, replacement_values):
    out = x.copy()
    rows, cols = np.where(mask)
    out[rows, cols] = replacement_values[cols]
    return out
