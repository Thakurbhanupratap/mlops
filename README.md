# Heart Disease Risk Prediction - MLOps Pipeline

End-to-end ML project: data acquisition, EDA, model training with MLflow tracking, unit tests,
GitHub Actions CI, Docker packaging, Kubernetes deployment and Prometheus/Grafana monitoring.

Dataset: UCI Heart Disease (Cleveland), 303 patients, 13 clinical features, binary target.

## Repository layout

```
src/              data processing, pipeline, training, inference
api/main.py       FastAPI service (/predict, /health, /metrics)
scripts/          download_data.py, eda.py
tests/            pytest unit and API tests
k8s/              Deployment, Service (LoadBalancer), Ingress
monitoring/       Prometheus config + Grafana provisioning/dashboard
.github/workflows ci.yml
Dockerfile, docker-compose.yml, requirements*.txt
reports/figures/  EDA plots
screenshots/      evidence for the report
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python scripts/download_data.py   # downloads raw data from UCI, writes data/processed/heart_clean.csv
python scripts/eda.py             # figures -> reports/figures
python -m src.train               # trains logreg + random forest, logs to MLflow, saves models/heart_pipeline.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
pytest -v
flake8 src api tests scripts
```

## Serve the API

Locally:
```bash
uvicorn api.main:app --port 8000
```

Docker:
```bash
docker build -t heart-api:1.0 .
docker run -p 8000:8000 heart-api:1.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Response: `{"prediction":1,"confidence":0.8,"probability_disease":0.8}` (values vary).

## Kubernetes (Docker Desktop / Minikube)

```bash
# Minikube only: minikube image load heart-api:1.0   (or: eval $(minikube docker-env) before docker build)
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
# Docker Desktop: LoadBalancer is reachable at http://localhost/health
# Minikube: minikube tunnel   (separate terminal), then use the EXTERNAL-IP
# Optional ingress (needs ingress-nginx): kubectl apply -f k8s/ingress.yaml
```

## Monitoring

```bash
docker compose up --build
```
- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090 (target `heart-api` should be UP)
- Grafana: http://localhost:3000 (admin/admin) -> dashboard "Heart API Monitoring"

Generate traffic for the dashboard with `python scripts/load_test.py`.

## Demo video

https://drive.google.com/file/d/14Asgfh7xBR8TSk25dNophkGbxzHgQu49/view?usp=sharing (also in `demo/pipeline_demo.mp4`)

## Report

See [REPORT.md](REPORT.md).
