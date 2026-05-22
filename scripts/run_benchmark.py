from __future__ import annotations

import argparse

from tabular_state_transformer.evaluation import run_benchmark


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_int_csv(value: str | None) -> list[int] | None:
    items = _split_csv(value)
    if items is None:
        return None
    return [int(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default="synthetic",
        choices=["synthetic", "synthetic_stress", "openml"],
    )
    parser.add_argument("--output", default="reports/benchmark_results.md")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--baselines", default=None, help="Comma-separated baseline keys")
    parser.add_argument("--models", default=None, help="Comma-separated TST model labels")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    seeds = _split_int_csv(args.seeds)
    if seeds is None and args.suite == "synthetic_stress":
        seeds = [42, 43, 44]

    results = run_benchmark(
        args.suite,
        output_path=args.output,
        csv_output_path=args.output_csv,
        diagnostics_output_path=args.diagnostics_output,
        n_samples=args.n_samples,
        max_epochs=args.max_epochs,
        seeds=seeds,
        dataset_names=_split_csv(args.datasets),
        baselines=_split_csv(args.baselines),
        model_configs=_split_csv(args.models),
        continue_on_error=not args.fail_fast,
    )
    print({"results": len(results), "output": args.output})


if __name__ == "__main__":
    main()
