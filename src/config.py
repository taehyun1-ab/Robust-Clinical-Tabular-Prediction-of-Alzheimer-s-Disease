"""Central configuration for the manuscript experiments."""

SEED = 17
N_SPLITS = 5
VAL_SIZE = 0.10
N_RANDOM_REPEATS = 100
RANDOM_MASK_COUNTS = (1, 2)
MASK_PROBABILITIES = (0.1, 0.2, 0.3)

CONTINUOUS_FEATURES = [
    "AGE", "CDRSB", "FAQTOTAL", "MMSCORE", "BMI", "PULSE"
]
CATEGORICAL_FEATURES = ["GENDER", "APOE4"]
FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = "DIAGNOSIS"

TASKS = {
    "AD_vs_CN": {
        "classes": ("CN", "AD"),
        "label_map": {"CN": 0, "AD": 1},
    },
    "AD_vs_MCI": {
        "classes": ("MCI", "AD"),
        "label_map": {"MCI": 0, "AD": 1},
    },
    "MCI_vs_CN": {
        "classes": ("CN", "MCI"),
        "label_map": {"CN": 0, "MCI": 1},
    },
}

RF_CONFIG = {
    "bootstrap": True,
    "class_weight": None,
    "max_depth": 5,
    "max_features": "log2",
    "min_samples_leaf": 1,
    "min_samples_split": 10,
    "n_estimators": 100,
}

XGB_CONFIG = {
    "colsample_bytree": 1.0,
    "gamma": 1.0,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 5,
    "n_estimators": 500,
    "reg_alpha": 0.01,
    "reg_lambda": 10.0,
    "subsample": 1.0,
    "scale_pos_weight": 1.0,
}

MLP_CONFIG = {
    "batch_size": 32,
    "dropout": 0.5,
    "epochs": 100,
    "hidden_dim": 16,
    "learning_rate": 0.001,
    "num_layers": 2,
    "patience": 10,
    "weight_decay": 0.0001,
}

FT_CONFIG = {
    "batch_size": 16,
    "d_model": 32,
    "dropout": 0.5,
    "epochs": 100,
    "learning_rate": 0.001,
    "n_heads": 4,
    "n_layers": 3,
    "patience": 10,
    "weight_decay": 0.001,
}

ROBUST_FT_CONFIG = {
    **FT_CONFIG,
    "feature_mask_prob": 0.2,
}
