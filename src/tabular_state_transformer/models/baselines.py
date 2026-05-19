from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from tabular_state_transformer.data.preprocessing import make_preprocessor


def make_baseline(name: str, task: str, frame, *, random_state: int = 42):
    preprocessor = make_preprocessor(frame)
    if name == "linear":
        estimator = (
            LogisticRegression(max_iter=500, random_state=random_state)
            if task == "classification"
            else Ridge(alpha=1.0)
        )
    elif name == "random_forest":
        estimator = (
            RandomForestClassifier(n_estimators=50, random_state=random_state)
            if task == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=random_state)
        )
    elif name == "mlp":
        estimator = (
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=random_state)
            if task == "classification"
            else MLPRegressor(hidden_layer_sizes=(64,), max_iter=100, random_state=random_state)
        )
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install xgboost to use this baseline") from exc
        estimator = (
            XGBClassifier(eval_metric="logloss", random_state=random_state)
            if task == "classification"
            else XGBRegressor(random_state=random_state)
        )
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install lightgbm to use this baseline") from exc
        estimator = (
            LGBMClassifier(random_state=random_state)
            if task == "classification"
            else LGBMRegressor(random_state=random_state)
        )
    elif name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install catboost to use this baseline") from exc
        estimator = (
            CatBoostClassifier(random_seed=random_state, verbose=False)
            if task == "classification"
            else CatBoostRegressor(random_seed=random_state, verbose=False)
        )
    else:
        raise ValueError(f"Unknown baseline '{name}'")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
