from __future__ import annotations

from tabular_state_transformer.data.synthetic import make_threshold_regression
from tabular_state_transformer.sklearn_api import TabularStateRegressor

def test_sklearn_regressor_smoke():
    X, y = make_threshold_regression(n_samples=64, n_features=8, random_state=1)
    model = TabularStateRegressor(max_epochs=1, d_token=8, batch_size=32).fit(X, y)
    assert model.predict(X[:3]).shape == (3,)
