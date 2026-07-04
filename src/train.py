import json
import shutil
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from src.config import (
    CV_FOLDS,
    FEATURES,
    FINAL_MODEL_PATH,
    MODEL_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REPORT_SCREENSHOT_DIR,
    TEST_SIZE,
)
from src.data_ingestion import clean_dataset, download_dataset
from src.preprocessing import build_preprocessor


def ensure_data_exists():
    if not PROCESSED_DATA_PATH.exists():
        download_dataset()
        clean_dataset()


def load_data():
    ensure_data_exists()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df[FEATURES]
    y = df["target_binary"]
    return X, y


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
    }


def create_plots(model, X_test, y_test, model_name):
    REPORT_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = REPORT_SCREENSHOT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {model_name}")
    roc_path = REPORT_SCREENSHOT_DIR / f"roc_curve_{model_name}.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    return cm_path, roc_path


def train_one_model(name, classifier, params):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        scoring = ["accuracy", "precision", "recall", "roc_auc"]
        cv_results = cross_validate(pipeline, X, y, cv=CV_FOLDS, scoring=scoring)
        cv_metrics = {
            f"cv_{metric}_mean": cv_results[f"test_{metric}"].mean()
            for metric in scoring
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(cv_metrics)

        cm_path, roc_path = create_plots(pipeline, X_test, y_test, name)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        result = {"model_name": name, **params, **metrics, **cv_metrics}
        print(json.dumps(result, indent=2))
        return pipeline, result, model_path


def main():
    mlflow.set_experiment("heart-disease-classification")

    results = []
    trained = []

    models = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_split=4,
                random_state=RANDOM_STATE,
            ),
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_split": 4,
            },
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
            {
                "model_type":"XGBoost",
                "n_estimators":200,
                "max_depth":4,
                "learning_rate":0.1,
            },
        ),
    ]

    for name, clf, params in models:
        pipeline, result, path = train_one_model(name, clf, params)
        results.append(result)
        trained.append((pipeline, result, path))

    best = max(trained, key=lambda item: item[1]["roc_auc"])
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best[2], FINAL_MODEL_PATH)

    pd.DataFrame(results).to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    print(f"Best model saved to {FINAL_MODEL_PATH}: {best[1]['model_name']}")


if __name__ == "__main__":
    main()
