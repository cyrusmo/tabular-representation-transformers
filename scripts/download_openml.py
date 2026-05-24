from __future__ import annotations

import argparse
from pathlib import Path

from tabular_state_transformer.data import load_dataset
from tabular_state_transformer.data.openml import OPENML_DATASETS


DEFAULT_OPENML_DOWNLOADS = ["adult", "credit-g", "covertype", "california-housing"]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted(OPENML_DATASETS), default="adult")
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated OpenML dataset names. Overrides --name when provided.",
    )
    parser.add_argument("--all", action="store_true", help="Download the paper-track OpenML set.")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output", default="data/raw/openml")
    args = parser.parse_args()

    if args.all:
        names = DEFAULT_OPENML_DOWNLOADS
    elif args.names:
        names = _split_csv(args.names)
    else:
        names = [args.name]

    unknown = sorted(set(names) - set(OPENML_DATASETS))
    if unknown:
        raise ValueError(f"Unknown OpenML datasets: {', '.join(unknown)}")

    for name in names:
        bundle = load_dataset(name, source="openml", n_samples=args.n_samples)
        output_dir = Path(args.output) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle.X_train.assign(**{bundle.target_name: bundle.y_train}).to_csv(
            output_dir / "train.csv", index=False
        )
        bundle.X_val.assign(**{bundle.target_name: bundle.y_val}).to_csv(
            output_dir / "val.csv", index=False
        )
        bundle.X_test.assign(**{bundle.target_name: bundle.y_test}).to_csv(
            output_dir / "test.csv", index=False
        )
        print({"dataset": name, "output_dir": str(output_dir), "n_samples": args.n_samples})


if __name__ == "__main__":
    main()
