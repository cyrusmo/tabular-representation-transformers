from __future__ import annotations

import csv
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabular_state_transformer.data.synthetic import make_threshold_regression
from tabular_state_transformer.sklearn_api import TabularStateRegressor

def main() -> None:
    X, y = make_threshold_regression(n_samples=2000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = TabularStateRegressor(max_epochs=5, d_token=32).fit(X_train, y_train)
    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    out = Path("benchmarks/results.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "metric", "value"]); writer.writeheader()
        writer.writerow({"dataset": "synthetic_threshold", "model": "tabular_state_transformer", "metric": "rmse", "value": rmse})
    print({"rmse": rmse})

if __name__ == "__main__":
    main()
