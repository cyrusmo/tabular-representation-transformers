from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    task_type: Literal["classification", "regression"],
    split_seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    stratify = y if task_type == "classification" and len(np.unique(y)) > 1 else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=stratify,
    )
    val_fraction = val_size / (1.0 - test_size)
    stratify_train = (
        y_train_val if task_type == "classification" and len(np.unique(y_train_val)) > 1 else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=split_seed,
        stratify=stratify_train,
    )
    return (
        X_train.reset_index(drop=True),
        np.asarray(y_train),
        X_val.reset_index(drop=True),
        np.asarray(y_val),
        X_test.reset_index(drop=True),
        np.asarray(y_test),
    )
