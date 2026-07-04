import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from api.schemas import PatientInput
from src.config import FINAL_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Cloud-ready API for heart disease risk classification.",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model_path = Path(FINAL_MODEL_PATH)
    if not model_path.exists():
        logger.warning("Model file not found. Train model first: python -m src.train")
        model = None
        return
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(input_data: PatientInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train or mount model first.")

    logger.info("Received prediction request")
    data = pd.DataFrame([input_data.model_dump()])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    response = {
        "prediction": prediction,
        "risk_label": "Heart Disease Risk" if prediction == 1 else "Low Risk",
        "confidence": probability,
    }
    logger.info("Prediction response: %s", response)
    return response


Instrumentator().instrument(app).expose(app)
