from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .config import (
    CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, FEATURES, LABEL_COLUMN, TASKS
)


def load_dataframe(path):
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in FEATURES + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def make_task_dataframe(df, task_name):
    task = TASKS[task_name]
    out = df[FEATURES + [LABEL_COLUMN]].copy()
    out = out.dropna(subset=[LABEL_COLUMN])
    out = out[out[LABEL_COLUMN].isin(task["selected_classes"])].copy()
    out["LABEL"] = out[LABEL_COLUMN].map(task["label_map"])
    out = out.reset_index(drop=True)
    return out[FEATURES].copy(), out["LABEL"].astype(int).copy()


@dataclass
class NeuralPreprocessor:
    scaler: StandardScaler
    encoder: OrdinalEncoder
    continuous_fill_values: dict
    categorical_fill_values: dict
    cat_cardinalities: list

    def transform(self, frame):
        x = frame.copy()
        for col in CONTINUOUS_FEATURES:
            x[col] = x[col].fillna(self.continuous_fill_values[col])
        for col in CATEGORICAL_FEATURES:
            x[col] = x[col].fillna(self.categorical_fill_values[col]).astype(str)

        x_cont = self.scaler.transform(x[CONTINUOUS_FEATURES])
        x_cat = (self.encoder.transform(x[CATEGORICAL_FEATURES]) + 1).astype(int)
        x_all = np.concatenate([x_cont, x_cat.astype(float)], axis=1)
        return x_cont, x_cat, x_all


def fit_neural_preprocessor(train_frame, categorical_strategy):
    x = train_frame.copy()

    continuous_fill_values = {}
    for col in CONTINUOUS_FEATURES:
        value = x[col].median()
        continuous_fill_values[col] = value
        x[col] = x[col].fillna(value)

    categorical_fill_values = {}
    for col in CATEGORICAL_FEATURES:
        if categorical_strategy == "mode":
            mode = x[col].mode()
            value = mode.iloc[0] if len(mode) > 0 else "Missing"
        elif categorical_strategy == "missing":
            value = "Missing"
        else:
            raise ValueError("categorical_strategy must be 'mode' or 'missing'")
        categorical_fill_values[col] = value
        x[col] = x[col].fillna(value).astype(str)

    scaler = StandardScaler()
    scaler.fit(x[CONTINUOUS_FEATURES])

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(x[CATEGORICAL_FEATURES])

    cat_cardinalities = [len(categories) + 1 for categories in encoder.categories_]

    return NeuralPreprocessor(
        scaler=scaler,
        encoder=encoder,
        continuous_fill_values=continuous_fill_values,
        categorical_fill_values=categorical_fill_values,
        cat_cardinalities=cat_cardinalities,
    )


@dataclass
class TreePreprocessor:
    cont_imputer: SimpleImputer
    cat_imputer: SimpleImputer
    encoder: OrdinalEncoder

    def transform(self, frame):
        x_cont = self.cont_imputer.transform(frame[CONTINUOUS_FEATURES])
        x_cat_raw = self.cat_imputer.transform(frame[CATEGORICAL_FEATURES].astype(str))
        x_cat = self.encoder.transform(x_cat_raw) + 1
        return np.concatenate([x_cont, x_cat], axis=1)

    def replacement_values(self):
        cat_mode_encoded = self.encoder.transform(
            self.cat_imputer.statistics_.reshape(1, -1)
        ) + 1
        return np.concatenate([
            self.cont_imputer.statistics_.astype(float),
            cat_mode_encoded.flatten().astype(float),
        ])


def fit_tree_preprocessor(train_frame):
    cont_imputer = SimpleImputer(strategy="median")
    cont_imputer.fit(train_frame[CONTINUOUS_FEATURES])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_raw = cat_imputer.fit_transform(train_frame[CATEGORICAL_FEATURES].astype(str))

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(cat_raw)

    return TreePreprocessor(cont_imputer, cat_imputer, encoder)
