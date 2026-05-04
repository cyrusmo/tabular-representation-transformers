from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from tabular_state_transformer.data.preprocessing import make_preprocessor


def make_baseline(name: str, task: str, frame):
    preprocessor = make_preprocessor(frame)
    if name == "linear":
        estimator = (
            LogisticRegression(max_iter=500)
            if task == "classification"
            else Ridge(alpha=1.0)
        )
    elif name == "random_forest":
        estimator = (
            RandomForestClassifier(n_estimators=50, random_state=42)
            if task == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=42)
        )
    elif name == "mlp":
        estimator = (
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=42)
            if task == "classification"
            else MLPRegressor(hidden_layer_sizes=(64,), max_iter=100, random_state=42)
        )
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install xgboost to use this baseline") from exc
        estimator = XGBClassifier(eval_metric="logloss") if task == "classification" else XGBRegressor()
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install lightgbm to use this baseline") from exc
        estimator = LGBMClassifier() if task == "classification" else LGBMRegressor()
    else:
        raise ValueError(f"Unknown baseline '{name}'")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
