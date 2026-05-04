from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TabularDatasetBundle:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    numerical_features: list[str]
    categorical_features: list[str]
    task_type: Literal["classification", "regression"]
    target_name: str
    dataset_name: str
