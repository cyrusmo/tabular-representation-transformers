from __future__ import annotations

import argparse
from pathlib import Path

from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.openml import OPENML_DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted(OPENML_DATASETS), default="adult")
    parser.add_argument("--output", default="data/raw/openml")
    args = parser.parse_args()

    bundle = load_dataset(args.name, source="openml")
    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.X_train.assign(**{bundle.target_name: bundle.y_train}).to_csv(
        output_dir / "train.csv", index=False
    )
    bundle.X_val.assign(**{bundle.target_name: bundle.y_val}).to_csv(output_dir / "val.csv", index=False)
    bundle.X_test.assign(**{bundle.target_name: bundle.y_test}).to_csv(
        output_dir / "test.csv", index=False
    )
    print({"dataset": args.name, "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
