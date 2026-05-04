from __future__ import annotations

from tabular_state_transformer.data.synthetic import make_threshold_regression
from tabular_state_transformer.sklearn_api import TabularStateRegressor

def main() -> None:
    X, y = make_threshold_regression(n_samples=2000, random_state=42)
    model = TabularStateRegressor(max_epochs=5, d_token=32)
    model.fit(X, y)
    print({"example_predictions": model.predict(X[:5]).round(4).tolist()})

if __name__ == "__main__":
    main()
