from __future__ import annotations

import numpy as np

from tabular_state_transformer import TabularStateConfig, TabularStateTransformer
from tabular_state_transformer.hf import export_model


def test_save_load_round_trip(tmp_path):
    model = TabularStateTransformer(
        TabularStateConfig(
            n_features=4,
            d_token=8,
            n_heads=2,
            n_layers=1,
            pooling="cls",
            use_feature_crosses=True,
            cross_max_features=4,
        )
    )
    export_model(model, tmp_path)
    loaded = TabularStateTransformer.from_pretrained(tmp_path)
    assert loaded.config.pooling == "cls"
    assert loaded.config.use_feature_crosses is True
    assert loaded.config.cross_max_features == 4
    preds = loaded.predict_numpy(np.zeros((2, 4), dtype="float32"))
    assert preds.shape == (2,)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "example_input.json").exists()
