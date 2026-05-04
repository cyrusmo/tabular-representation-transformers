from __future__ import annotations

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.training import Trainer


def test_training_loop_smoke():
    bundle = load_dataset("synthetic_xor", split_seed=42, n_samples=128, n_features=8)
    config = TabularStateConfig(
        n_features=8,
        d_token=8,
        n_heads=2,
        n_layers=1,
        task="classification",
        max_epochs=1,
        batch_size=32,
    )
    result = Trainer(config).fit(bundle)
    assert result.train_loss >= 0
    assert result.model.config.n_features >= 8
