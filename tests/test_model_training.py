from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor


def test_pipeline_contains_expected_steps():
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
