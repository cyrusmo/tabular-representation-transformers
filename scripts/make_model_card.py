from __future__ import annotations

import argparse
from pathlib import Path

from tabular_state_transformer.hf import make_model_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="README.md")
    parser.add_argument("--model-name", default="Tabular State Transformer")
    args = parser.parse_args()
    Path(args.output).write_text(make_model_card(args.model_name))
    print({"output": args.output})


if __name__ == "__main__":
    main()
