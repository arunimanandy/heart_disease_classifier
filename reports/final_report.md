# End-to-End MLOps Project Report: Heart Disease Risk Prediction

**Student Name:** <Your Name>  
**Course/Assignment:** MLOps Experimental Learning Assignment  
**Repository Link:** <GitHub Repository URL>  
**API URL:** <Public API URL or Local Access Instructions>  

---

## 1. Executive Summary

This project implements an end-to-end machine learning operations workflow for heart disease risk classification. The solution covers dataset acquisition, exploratory analysis, preprocessing, model development, experiment tracking, model packaging, automated testing, CI/CD, Docker-based API serving, Kubernetes deployment, and monitoring.

The goal is not only to train a classification model, but also to make the solution reproducible, testable, containerized, deployable, and observable in a production-style environment.

---

## 2. Dataset Description

The project uses the UCI Heart Disease dataset, with the processed Cleveland dataset as the primary source. It contains 303 patient records and 13 commonly used input features, such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG result, maximum heart rate, exercise-induced angina, ST depression, slope, number of vessels, and thalassemia status.

The original target column contains values from 0 to 4. For this binary classification task, the target was converted as follows:

- `0`: absence of heart disease
- `1`: presence of heart disease, obtained by mapping original target values 1, 2, 3, and 4 to 1

---

## 3. Data Acquisition and Cleaning

The dataset is downloaded using `src/data_ingestion.py`. The raw file does not include column headers, so column names are assigned manually. Missing values are represented with `?` and are converted to proper missing values during loading.

Cleaning steps include:

1. Downloading the processed Cleveland dataset.
2. Assigning the correct 14 column names.
3. Handling `?` as missing values.
4. Creating a binary target variable named `target_binary`.
5. Saving the processed dataset to `data/processed/heart_disease_clean.csv`.

---

## 4. Exploratory Data Analysis

EDA was performed using pandas, matplotlib, and seaborn. The analysis included:

- Dataset shape and schema inspection
- Missing value summary
- Descriptive statistics
- Class balance visualization
- Feature histograms
- Correlation heatmap

### EDA Screenshots

Insert screenshots here:

- `reports/screenshots/class_balance.png`
- `reports/screenshots/feature_histograms.png`
- `reports/screenshots/correlation_heatmap.png`

Key observations:

- The dataset is small but suitable for a supervised classification experiment.
- Some variables are categorical despite being represented numerically.
- `ca` and `thal` may contain missing values.
- Clinical features such as chest pain type, maximum heart rate, exercise-induced angina, and ST depression are expected to be useful predictors.

---

## 5. Feature Engineering and Preprocessing

The preprocessing pipeline was implemented using scikit-learn `Pipeline` and `ColumnTransformer`.

Numerical features:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`
- `ca`

Categorical features:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `thal`

Preprocessing operations:

- Median imputation for numerical columns
- Most frequent imputation for categorical columns
- Standard scaling for numerical features
- One-hot encoding for categorical features

This design ensures that the exact same preprocessing logic is used during both training and inference.

---

## 6. Model Development

Two classification models were trained:

1. Logistic Regression
2. Random Forest Classifier

The data was split into training and testing sets using stratification to preserve target distribution. Both models were evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- 5-fold cross-validation

The best model is selected based on ROC-AUC and saved as:

```text
models/best_model.pkl
```

The final artifact includes both preprocessing and model steps, ensuring reproducibility.

---

## 7. Experiment Tracking with MLflow

MLflow was used to track experiments. For each model run, the following were logged:

- Model parameters
- Test metrics
- Cross-validation metrics
- Confusion matrix artifact
- ROC curve artifact
- Serialized model artifact

To launch the MLflow UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Screenshots to include:

- MLflow experiment list
- Logistic Regression run details
- Random Forest run details
- Metric comparison chart

---

## 8. API Serving and Containerization

The prediction service was implemented using FastAPI.

Endpoints:

- `GET /health`: verifies API status and model loading state
- `POST /predict`: accepts patient health data and returns prediction and confidence
- `GET /metrics`: exposes Prometheus metrics

Sample prediction request:

```json
{
  "age": 55,
  "sex": 1,
  "cp": 4,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

The API is packaged using Docker. Build and run commands:

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

---

## 9. CI/CD Pipeline

A GitHub Actions workflow automates:

1. Repository checkout
2. Python setup
3. Dependency installation
4. Code linting using flake8
5. Dataset download
6. Unit and API testing using pytest
7. Model training
8. Docker image build
9. Artifact upload

The pipeline is configured to fail if linting, tests, training, or Docker build fails.

Screenshots to include:

- Successful GitHub Actions run
- Unit test logs
- Artifact upload result

---

## 10. Deployment Architecture

The API is deployed using Kubernetes manifests.

Components:

- `Deployment`: manages API replicas
- `Service`: exposes the application
- `HorizontalPodAutoscaler`: enables scaling based on CPU utilization
- Readiness and liveness probes for reliability

Deployment commands:

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl get svc
```

For Minikube:

```bash
minikube service heart-disease-service
```

---

## 11. Monitoring and Logging

Logging is implemented in the FastAPI application using Python's logging module. Prediction requests and responses are logged.

Prometheus metrics are exposed at:

```text
/metrics
```

Using Docker Compose, Prometheus and Grafana can be launched with:

```bash
docker compose up --build
```

Monitoring dashboard indicators:

- Total request count
- Request latency
- HTTP status code distribution
- Error count
- API availability

---

## 12. Architecture Diagram

```text
UCI Dataset
    ↓
Data Ingestion
    ↓
EDA and Cleaning
    ↓
Preprocessing Pipeline
    ↓
Model Training: Logistic Regression + Random Forest
    ↓
MLflow Tracking
    ↓
Best Model Artifact
    ↓
FastAPI Prediction Service
    ↓
Docker Container
    ↓
Kubernetes Deployment
    ↓
Prometheus + Grafana Monitoring
```

---

## 13. Testing Summary

Tests implemented:

- Preprocessing pipeline test
- Missing value handling test
- Model pipeline structure test
- API health endpoint test

Run tests:

```bash
pytest tests/ -v
```

---

## 14. Production Readiness Summary

The project satisfies production-readiness expectations through:

- Reproducible setup using `requirements.txt`
- Automated data ingestion
- End-to-end training script
- Serialized preprocessing and model pipeline
- Unit tests
- CI/CD automation
- Docker containerization
- Kubernetes deployment manifests
- Health checks
- Logging
- Prometheus-compatible metrics

---

## 15. Conclusion

This assignment demonstrates a complete MLOps lifecycle for a supervised classification use case. The final solution is reproducible, automated, containerized, deployable, and monitorable. It mirrors a real-world production workflow where machine learning models must be tracked, tested, packaged, deployed, and observed continuously.

---

## Appendix: Screenshot Checklist

- Dataset download output
- EDA plots
- MLflow UI
- Pytest results
- GitHub Actions workflow success
- Docker build output
- Docker container logs
- Swagger UI
- `/predict` response
- Kubernetes pods/services
- Prometheus metrics
- Grafana dashboard
