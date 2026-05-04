from __future__ import annotations

import numpy as np

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
