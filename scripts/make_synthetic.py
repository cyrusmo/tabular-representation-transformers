from __future__ import annotations

import argparse
from pathlib import Path

from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/synthetic.yaml")
    parser.add_argument("--output", default="data/processed/synthetic")
    args = parser.parse_args()

    config = read_yaml(args.config)
    bundle = load_dataset(**config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.X_train.assign(target=bundle.y_train).to_csv(output_dir / "train.csv", index=False)
    bundle.X_val.assign(target=bundle.y_val).to_csv(output_dir / "val.csv", index=False)
    bundle.X_test.assign(target=bundle.y_test).to_csv(output_dir / "test.csv", index=False)
    print({"dataset": bundle.dataset_name, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
