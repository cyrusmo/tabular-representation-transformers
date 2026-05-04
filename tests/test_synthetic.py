from __future__ import annotations

from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.synthetic import SYNTHETIC_TASKS


def test_synthetic_registry_loads_xor():
    bundle = load_dataset("synthetic_xor", split_seed=42, n_samples=128, n_features=8)
    assert bundle.dataset_name == "synthetic_xor"
    assert bundle.task_type == "classification"
    assert bundle.X_train.shape[1] == 8
    assert set(bundle.y_train).issubset({0, 1})


def test_all_public_synthetic_tasks_generate():
    for name in SYNTHETIC_TASKS:
        bundle = load_dataset(name, split_seed=1, n_samples=64, n_features=8)
        assert len(bundle.X_train) > 0
        assert len(bundle.y_test) > 0
