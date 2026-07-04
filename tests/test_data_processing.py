import pandas as pd
from src.preprocessing import build_preprocessor


def test_preprocessor_runs_on_valid_input():
    data = pd.DataFrame({
        "age": [55], "sex": [1], "cp": [4], "trestbps": [140], "chol": [250],
        "fbs": [0], "restecg": [1], "thalach": [150], "exang": [0], "oldpeak": [1.2],
        "slope": [2], "ca": [0], "thal": [3]
    })
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(data)
    assert transformed.shape[0] == 1


def test_preprocessor_handles_missing_values():
    data = pd.DataFrame({
        "age": [55, None], "sex": [1, 0], "cp": [4, 2], "trestbps": [140, None],
        "chol": [250, 220], "fbs": [0, 1], "restecg": [1, 0], "thalach": [150, 170],
        "exang": [0, 1], "oldpeak": [1.2, None], "slope": [2, 1], "ca": [0, None],
        "thal": [3, None]
    })
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(data)
    assert transformed.shape[0] == 2
