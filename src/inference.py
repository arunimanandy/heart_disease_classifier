import joblib
import pandas as pd
from src.config import FINAL_MODEL_PATH


def load_model(model_path=FINAL_MODEL_PATH):
    return joblib.load(model_path)


def predict_one(payload: dict, model_path=FINAL_MODEL_PATH):
    model = load_model(model_path)
    df = pd.DataFrame([payload])
    prediction = int(model.predict(df)[0])
    confidence = float(model.predict_proba(df)[0][1])
    return {
        "prediction": prediction,
        "risk_label": "Heart Disease Risk" if prediction == 1 else "Low Risk",
        "confidence": confidence,
    }
