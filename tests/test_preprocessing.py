from __future__ import annotations

import pandas as pd

from tabular_state_transformer.data.preprocessing import make_preprocessor, transform_to_float32


def test_preprocessor_handles_numeric_and_categorical():
    frame = pd.DataFrame({"num": [1.0, 2.0, None], "cat": ["a", "b", "a"]})
    preprocessor = make_preprocessor(frame).fit(frame)
    output = transform_to_float32(preprocessor, frame)
    assert output.shape[0] == 3
    assert output.dtype == "float32"
