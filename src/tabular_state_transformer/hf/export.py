from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tabular_state_transformer.hf.model_card import make_model_card
from tabular_state_transformer.modeling import TabularStateTransformer


def export_model(
    model: TabularStateTransformer,
    output_dir: str | Path,
    *,
    example_input: np.ndarray | None = None,
    model_name: str = "Tabular State Transformer",
) -> Path:
    output_path = Path(output_dir)
    model.save_pretrained(output_path)
    if example_input is None:
        example_input = np.zeros((1, model.config.n_features), dtype="float32")
    (output_path / "example_input.json").write_text(json.dumps(example_input.tolist(), indent=2))
    (output_path / "requirements.txt").write_text(
        "torch\nnumpy\npandas\nscikit-learn\npyyaml\nsafetensors\n"
    )
    (output_path / "README.md").write_text(make_model_card(model_name))
    return output_path
