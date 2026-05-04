# Data

This repository does not commit downloaded or generated datasets.

- `raw/` stores original downloaded files, such as OpenML exports.
- `processed/` stores encoded train/validation/test artifacts.
- `cache/` stores reusable preprocessed folds.

Use `scripts/make_synthetic.py` for synthetic data and `scripts/download_openml.py` for public OpenML datasets.
