from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .openml import OPENML_DATASETS, load_openml_named
from .schema import TabularDatasetBundle
from .splits import split_frame
from .synthetic import SYNTHETIC_TASKS


def _feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numerical]
    return numerical, categorical


def _bundle_from_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    dataset_name: str,
    task_type: Literal["classification", "regression"],
    target_name: str = "target",
    split_seed: int = 42,
) -> TabularDatasetBundle:
    numerical, categorical = _feature_lists(X)
    X_train, y_train, X_val, y_val, X_test, y_test = split_frame(
        X, y, task_type=task_type, split_seed=split_seed
    )
    return TabularDatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        numerical_features=numerical,
        categorical_features=categorical,
        task_type=task_type,
        target_name=target_name,
        dataset_name=dataset_name,
    )


def _sample_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_samples: int | None,
    split_seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if n_samples is None or len(X) <= n_samples:
        return X, y
    row_indices = np.random.default_rng(split_seed).choice(len(X), size=n_samples, replace=False)
    row_indices.sort()
    return X.iloc[row_indices].reset_index(drop=True), y[row_indices]


def load_dataset(
    name: str,
    *,
    source: str | None = None,
    split_seed: int = 42,
    task: Literal["classification", "regression"] | None = None,
    path: str | Path | None = None,
    target: str = "target",
    **kwargs,
) -> TabularDatasetBundle:
    if source is None:
        if name in SYNTHETIC_TASKS:
            source = "synthetic"
        elif name in OPENML_DATASETS:
            source = "openml"
        elif path is not None:
            source = "local"
        else:
            raise ValueError(f"Unknown dataset '{name}'.")

    if source == "synthetic":
        try:
            generator, default_task = SYNTHETIC_TASKS[name]
        except KeyError as exc:
            raise ValueError(f"Unknown synthetic dataset '{name}'.") from exc
        X, y = generator(random_state=split_seed, **kwargs)
        return _bundle_from_frame(
            X,
            y,
            dataset_name=name,
            task_type=task or default_task,
            target_name=target,
            split_seed=split_seed,
        )

    if source == "openml":
        n_samples = kwargs.pop("n_samples", None)
        X, y, task_type, target_name = load_openml_named(name)
        X, y = _sample_rows(X, y, n_samples=n_samples, split_seed=split_seed)
        return _bundle_from_frame(
            X,
            y,
            dataset_name=name,
            task_type=task or task_type,
            target_name=target_name,
            split_seed=split_seed,
        )

    if source in {"local", "local_csv", "local_parquet"}:
        n_samples = kwargs.pop("n_samples", None)
        if path is None:
            raise ValueError("A local dataset requires path=...")
        local_path = Path(path)
        if source == "local_parquet" or local_path.suffix == ".parquet":
            frame = pd.read_parquet(local_path)
        else:
            frame = pd.read_csv(local_path)
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in {local_path}")
        y = frame[target].to_numpy()
        X = frame.drop(columns=[target])
        X, y = _sample_rows(X, y, n_samples=n_samples, split_seed=split_seed)
        task_type = task or ("classification" if frame[target].nunique() <= 20 else "regression")
        return _bundle_from_frame(
            X,
            y,
            dataset_name=name,
            task_type=task_type,
            target_name=target,
            split_seed=split_seed,
        )

    if source == "hf_dataset":
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install datasets to use Hugging Face dataset loading") from exc
        n_samples = kwargs.pop("n_samples", None)
        dataset = hf_load_dataset(name, split=kwargs.pop("split", "train"), **kwargs)
        frame = dataset.to_pandas()
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' not found in Hugging Face dataset")
        X, y = _sample_rows(
            frame.drop(columns=[target]),
            frame[target].to_numpy(),
            n_samples=n_samples,
            split_seed=split_seed,
        )
        return _bundle_from_frame(
            X,
            y,
            dataset_name=name,
            task_type=task or "classification",
            target_name=target,
            split_seed=split_seed,
        )

    raise ValueError(f"Unsupported dataset source '{source}'.")


__all__ = ["OPENML_DATASETS", "SYNTHETIC_TASKS", "TabularDatasetBundle", "load_dataset"]
