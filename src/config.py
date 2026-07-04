from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_DATA_PATH = Path("data/raw/heart_disease.csv")
PROCESSED_DATA_PATH = Path("data/processed/heart_disease_clean.csv")
MODEL_DIR = Path("models")
FINAL_MODEL_PATH = MODEL_DIR / "best_model.pkl"
REPORT_SCREENSHOT_DIR = Path("reports/screenshots")
