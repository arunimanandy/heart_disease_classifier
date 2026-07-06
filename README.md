# Heart Disease MLOps Project

End-to-end MLOps project for predicting heart disease risk using the UCI Cleveland Heart Disease dataset. The solution includes data ingestion, EDA, preprocessing, model training, MLflow experiment tracking, automated testing, CI/CD, FastAPI serving, Docker containerization, Kubernetes deployment, and Prometheus/Grafana monitoring.

> Educational project only. Not intended for clinical diagnosis or medical decision-making.

## Repository Structure

```text
heart-disease-mlops/
├── api/                  # FastAPI model serving app
├── data/                 # Raw and processed data folders
├── k8s/                  # Kubernetes manifests
├── monitoring/           # Prometheus/Grafana config
├── notebooks/            # EDA notebook/script
├── reports/              # Report template and screenshots folder
├── src/                  # Data, preprocessing, training, evaluation, inference code
├── tests/                # Pytest unit/API tests
├── .github/workflows/    # GitHub Actions CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── Makefile
```

## Quick Start

```bash
git clone <your-repo-url>
cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download Dataset

```bash
python -m src.data_ingestion
```

## Run EDA

```bash
python notebooks/01_eda.py
```

EDA figures are saved under `reports/screenshots/`.

## Train Models with MLflow

```bash
python -m src.train
mlflow ui --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000`.

## Run Tests

```bash
pytest tests/ -v
```

## Serve API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## Predict Example

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
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
}'
```

## Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

## Docker Compose with Monitoring

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` using admin/admin

## Kubernetes Deployment

For Minikube:

```bash
minikube start
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/
kubectl get pods
kubectl get svc
minikube service heart-disease-service
```

## CI/CD

GitHub Actions pipeline runs:

1. Dependency installation
2. Linting
3. Dataset download
4. Unit/API tests
5. Model training
6. Docker build
7. Artifact upload

## Deliverables Checklist

- [x] Data ingestion script
- [x] EDA script and plots
- [x] Two ML models
- [x] MLflow tracking
- [x] Reproducible sklearn pipeline
- [x] FastAPI `/predict` endpoint
- [x] Dockerfile and docker-compose
- [x] Pytest tests
- [x] GitHub Actions workflow
- [x] Kubernetes manifests
- [x] Prometheus/Grafana monitoring config
- [x] 10-page report template
