install:
	pip install -e ".[dev,benchmark,hub]"

test:
	pytest

lint:
	ruff check src tests scripts

benchmark-smoke:
	python scripts/run_benchmark.py --suite synthetic --n-samples 128 --max-epochs 1
