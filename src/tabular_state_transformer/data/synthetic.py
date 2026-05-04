from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X.astype("float32"), columns=[f"x{i}" for i in range(X.shape[1])])


def _uniform(
    n_samples: int,
    n_features: int,
    random_state: int,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(random_state)
    return rng, rng.uniform(low, high, size=(n_samples, n_features)).astype("float32")

def make_threshold_regression(n_samples: int = 5000, n_features: int = 20, noise: float = 0.1, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    y = ((X[:, 0] > 0.5) & (X[:, 1] < -0.25)).astype(float)
    y += np.sin(4 * X[:, 2]) * (X[:, 3] > 0)
    y += rng.normal(scale=noise, size=n_samples)
    return X.astype("float32"), y.astype("float32")

def make_regime_classification(n_samples: int = 5000, n_features: int = 20, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    regime = X[:, 0] > 0
    score = np.where(regime, X[:, 1] * X[:, 2], X[:, 3] - X[:, 4])
    y = (score + 0.2 * rng.normal(size=n_samples) > 0).astype(int)
    return X.astype("float32"), y


def make_axis_aligned_thresholds(
    n_samples: int = 5000,
    n_features: int = 20,
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state)
    y = ((X[:, 0] > 0.7) & (X[:, 1] < 0.3)).astype("int64")
    if noise:
        flip = rng.random(n_samples) < noise
        y[flip] = 1 - y[flip]
    return _frame(X), y


def make_xor_interactions(
    n_samples: int = 5000,
    n_features: int = 20,
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype("int64")
    if noise:
        flip = rng.random(n_samples) < noise
        y[flip] = 1 - y[flip]
    return _frame(X), y


def make_piecewise_non_smooth_functions(
    n_samples: int = 5000,
    n_features: int = 20,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state)
    y = np.sin(20 * X[:, 0]) + (X[:, 1] > 0.8).astype("float32")
    y += rng.normal(scale=noise, size=n_samples)
    return _frame(X), y.astype("float32")


def make_irrelevant_feature_noise(
    n_samples: int = 5000,
    n_features: int = 100,
    noise: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state)
    score = 2.0 * X[:, 0] - 1.5 * X[:, 1] + (X[:, 2] > 0.7)
    y = (score + rng.normal(scale=noise, size=n_samples) > 0.5).astype("int64")
    return _frame(X), y


def make_rotated_feature_failure_case(
    n_samples: int = 5000,
    n_features: int = 20,
    noise: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state, low=-1.0, high=1.0)
    rotated = (X[:, 0] + X[:, 1]) / np.sqrt(2.0)
    y = (rotated + rng.normal(scale=noise, size=n_samples) > 0.2).astype("int64")
    return _frame(X), y


def make_regime_switching_function(
    n_samples: int = 5000,
    n_features: int = 20,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state, low=-1.0, high=1.0)
    y = np.where(X[:, 0] < 0, X[:, 1] * X[:, 2], X[:, 3] - X[:, 4])
    y += rng.normal(scale=noise, size=n_samples)
    return _frame(X), y.astype("float32")


def make_sparse_high_order_interactions(
    n_samples: int = 5000,
    n_features: int = 50,
    noise: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng, X = _uniform(n_samples, n_features, random_state)
    active = (X[:, 0] > 0.6) & (X[:, 1] < 0.4) & (X[:, 2] > 0.5) & (X[:, 3] < 0.3)
    y = active.astype("int64")
    if noise:
        flip = rng.random(n_samples) < noise
        y[flip] = 1 - y[flip]
    return _frame(X), y


SYNTHETIC_TASKS = {
    "synthetic_axis_aligned": (make_axis_aligned_thresholds, "classification"),
    "synthetic_thresholds": (make_axis_aligned_thresholds, "classification"),
    "synthetic_xor": (make_xor_interactions, "classification"),
    "synthetic_piecewise": (make_piecewise_non_smooth_functions, "regression"),
    "synthetic_irrelevant_noise": (make_irrelevant_feature_noise, "classification"),
    "synthetic_rotated": (make_rotated_feature_failure_case, "classification"),
    "synthetic_regime": (make_regime_switching_function, "regression"),
    "synthetic_sparse_high_order": (make_sparse_high_order_interactions, "classification"),
}
