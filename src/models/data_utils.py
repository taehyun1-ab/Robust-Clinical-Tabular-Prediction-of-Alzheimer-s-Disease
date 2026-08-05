"""Data validation, task construction, and fold-wise preprocessing."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    from .config import (
        CATEGORICAL_FEATURES,
        CONTINUOUS_FEATURES,
        FEATURES,
        LABEL_COLUMN,
        TASKS,
    )
except ImportError:
    from config import (
        CATEGORICAL_FEATURES,
        CONTINUOUS_FEATURES,
        FEATURES,
        LABEL_COLUMN,
        TASKS,
    )


@dataclass
class FoldPreprocessor:
    scaler: StandardScaler
    encoder: OrdinalEncoder
    continuous_fill_values: Dict[str, float]
    categorical_fill_values: Dict[str, str]
    cat_cardinalities: list[int]

    def transform(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = frame.copy()

        for col in CONTINUOUS_FEATURES:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].fillna(self.continuous_fill_values[col])

        for col in CATEGORICAL_FEATURES:
            x[col] = x[col].fillna(self.categorical_fill_values[col]).astype(str)

        x_cont = self.scaler.transform(x[CONTINUOUS_FEATURES]).astype(np.float32)
        x_cat = self.encoder.transform(x[CATEGORICAL_FEATURES])
        x_cat = (x_cat + 1).astype(np.int64)
        x_all = np.concatenate([x_cont, x_cat.astype(np.float32)], axis=1)
        return x_cont, x_cat, x_all

    def replacement_values(self) -> np.ndarray:
        continuous_raw = np.array(
            [[self.continuous_fill_values[c] for c in CONTINUOUS_FEATURES]],
            dtype=float,
        )
        continuous_replacement = self.scaler.transform(
            pd.DataFrame(continuous_raw, columns=CONTINUOUS_FEATURES)
        )[0]

        categorical_raw = np.array(
            [[self.categorical_fill_values[c] for c in CATEGORICAL_FEATURES]],
            dtype=object,
        )
        categorical_replacement = self.encoder.transform(categorical_raw)[0] + 1
        return np.concatenate(
            [continuous_replacement, categorical_replacement.astype(float)]
        )


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in FEATURES + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def make_task(df: pd.DataFrame, task_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task_name not in TASKS:
        raise KeyError(f"Unknown task: {task_name}")

    task = TASKS[task_name]
    out = df[FEATURES + [LABEL_COLUMN]].copy()
    out = out.dropna(subset=[LABEL_COLUMN])
    out = out[out[LABEL_COLUMN].isin(task["classes"])].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows available for task {task_name}")

    y = out[LABEL_COLUMN].map(task["label_map"]).astype(int).to_numpy()
    return out[FEATURES].copy(), y


def fit_preprocessor(train_frame: pd.DataFrame) -> FoldPreprocessor:
    x = train_frame.copy()

    continuous_fill_values: Dict[str, float] = {}
    for col in CONTINUOUS_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        value = float(x[col].median())
        if np.isnan(value):
            value = 0.0
        continuous_fill_values[col] = value
        x[col] = x[col].fillna(value)

    categorical_fill_values: Dict[str, str] = {}
    for col in CATEGORICAL_FEATURES:
        series = x[col].dropna().astype(str)
        value = str(series.mode().iloc[0]) if not series.empty else "Missing"
        categorical_fill_values[col] = value
        x[col] = x[col].fillna(value).astype(str)

    scaler = StandardScaler()
    scaler.fit(x[CONTINUOUS_FEATURES])

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(x[CATEGORICAL_FEATURES])

    cat_cardinalities = [len(categories) + 1 for categories in encoder.categories_]

    return FoldPreprocessor(
        scaler=scaler,
        encoder=encoder,
        continuous_fill_values=continuous_fill_values,
        categorical_fill_values=categorical_fill_values,
        cat_cardinalities=cat_cardinalities,
    )
