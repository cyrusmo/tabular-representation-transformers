from __future__ import annotations

import pandas as pd

from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.preprocessing import make_preprocessor, transform_to_float32


def test_preprocessor_handles_numeric_and_categorical():
    frame = pd.DataFrame({"num": [1.0, 2.0, None], "cat": ["a", "b", "a"]})
    preprocessor = make_preprocessor(frame).fit(frame)
    output = transform_to_float32(preprocessor, frame)
    assert output.shape[0] == 3
    assert output.dtype == "float32"


def test_local_dataset_supports_deterministic_subsampling(tmp_path):
    path = tmp_path / "local.csv"
    pd.DataFrame(
        {
            "x": range(20),
            "cat": ["a", "b"] * 10,
            "target": [0, 1] * 10,
        }
    ).to_csv(path, index=False)

    bundle = load_dataset("local-demo", source="local_csv", path=path, n_samples=10, split_seed=7)
    total_rows = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)

    assert total_rows == 10
    assert bundle.categorical_features == ["cat"]
