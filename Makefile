install:
	pip install -r requirements.txt

data:
	python -m src.data_ingestion

eda:
	python notebooks/01_eda.py

train:
	python -m src.train

test:
	pytest tests/ -v

lint:
	flake8 src api tests --max-line-length=100

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run -p 8000:8000 heart-disease-api:latest
