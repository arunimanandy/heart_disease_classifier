import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.config import FEATURES, FINAL_MODEL_PATH, PROCESSED_DATA_PATH


def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURES]
    y = df["target_binary"]
    model = joblib.load(FINAL_MODEL_PATH)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    print(classification_report(y, preds))
    print("ROC-AUC:", roc_auc_score(y, probs))


if __name__ == "__main__":
    main()
