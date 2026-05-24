from __future__ import annotations

import argparse
from pathlib import Path

from tabular_state_transformer.config import TabularStateConfig
from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.training import Trainer
from tabular_state_transformer.training.artifacts import save_training_artifacts
from tabular_state_transformer.utils.io import read_merged_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/classification.yaml")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    config = read_merged_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    bundle = load_dataset(**data_config)
    model_values = {
        **model_config,
        "task": config.get("task", bundle.task_type),
        "max_epochs": config.get("max_epochs", model_config.get("max_epochs", 20)),
        "batch_size": config.get("batch_size", model_config.get("batch_size", 256)),
        "learning_rate": config.get("learning_rate", model_config.get("learning_rate", 1e-3)),
    }
    model_values.setdefault("n_features", len(bundle.numerical_features) + len(bundle.categorical_features))
    model = TabularStateConfig.from_dict(model_values)
    result = Trainer(model, device=args.device).fit(bundle)
    output_dir = Path(config.get("output_dir", "outputs/train"))
    save_training_artifacts(result, output_dir)
    print(
        {
            "dataset": bundle.dataset_name,
            "task": bundle.task_type,
            "train_loss": round(result.train_loss, 6),
            "val_metric": round(result.val_metric, 6),
            "output_dir": str(output_dir),
        }
    )

if __name__ == "__main__":
    main()
