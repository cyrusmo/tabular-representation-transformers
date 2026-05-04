from __future__ import annotations

import argparse

from tabular_state_transformer import TabularStateConfig, TabularStateTransformer
from tabular_state_transformer.hf import export_model
from tabular_state_transformer.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model/tst_v0.yaml")
    parser.add_argument("--output", default="outputs/hf-demo")
    args = parser.parse_args()

    values = read_yaml(args.model_config)
    config = TabularStateConfig.from_dict(values)
    model = TabularStateTransformer(config)
    export_model(model, args.output)
    print({"output": args.output})


if __name__ == "__main__":
    main()
