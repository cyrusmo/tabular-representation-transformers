from __future__ import annotations

import argparse

from tabular_state_transformer.evaluation import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="synthetic", choices=["synthetic", "openml"])
    parser.add_argument("--output", default="reports/benchmark_results.md")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=2)
    args = parser.parse_args()

    results = run_benchmark(
        args.suite,
        output_path=args.output,
        n_samples=args.n_samples,
        max_epochs=args.max_epochs,
    )
    print({"results": len(results), "output": args.output})


if __name__ == "__main__":
    main()
