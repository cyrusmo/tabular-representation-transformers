from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


OPENML_DATASETS: dict[str, dict[str, object]] = {
    "adult": {"id": 1590, "task": "classification"},
    "bank-marketing": {"id": 1461, "task": "classification"},
    "covertype": {"id": 1596, "task": "classification"},
    "higgs-small": {"id": 23512, "task": "classification"},
    "heloc": {"id": 45037, "task": "classification"},
    "california-housing": {"id": 43939, "task": "regression"},
    "credit-g": {"id": 31, "task": "classification"},
    "jannis": {"id": 41168, "task": "classification"},
}


def load_openml_dataset(dataset_id: int):
    try:
        import openml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install openml to use OpenML benchmark integration") from exc
    dataset = openml.datasets.get_dataset(dataset_id)
    return dataset.get_data(target=dataset.default_target_attribute)


def load_openml_named(
    name: str,
) -> tuple[pd.DataFrame, np.ndarray, Literal["classification", "regression"], str]:
    try:
        meta = OPENML_DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown OpenML dataset '{name}'.") from exc

    X, y, categorical_indicator, attribute_names = load_openml_dataset(int(meta["id"]))
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=attribute_names)
    X = X.copy()
    y_array = np.asarray(y)
    task_type = meta["task"]  # type: ignore[assignment]
    target_name = "target"
    return X, y_array, task_type, target_name

CURATED_OPENML_IDS = [int(meta["id"]) for meta in OPENML_DATASETS.values()]
TABPFN_SMALL_DATA_IDS = [31, 37, 44]
