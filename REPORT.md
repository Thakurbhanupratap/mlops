# MLOps Assignment 1 - Heart Disease Risk Prediction

**Name:** Bhanu Pratap Singh  **BITS ID:** 2025AE05693  **Course:** MLOps (AIMLCZG523)

**Code repository:** ==TODO: https://github.com/<your-username>/<repo-name>==

## 1. Problem and dataset

The goal is to predict whether a patient has heart disease from routine clinical measurements and to
serve that prediction through a monitored, containerised API. Data is the UCI Heart Disease
(Cleveland) set: 303 rows, 13 features, original target 0-4 collapsed to binary
(0 = no disease, 1 = disease). Class counts: 164 no disease, 139 disease (46 % positive), so the
classes are close to balanced and accuracy is a reasonable headline metric; recall is tracked because
missing a sick patient is the costly error.

## 2. Setup

See [README.md](README.md). Summary: `pip install -r requirements.txt`, then
`python scripts/download_data.py`, `python -m src.train`, `pytest`.

## 3. Data cleaning and EDA

- Raw file uses `?` for missing values. Six values are missing (4 in `ca`, 2 in `thal`);
  both are categorical so they were filled with the column mode.
- Target binarised; categorical columns cast to integer codes; duplicates dropped (none found).
- Figures (in `reports/figures/`):

| Figure | Observation |
|---|---|
| class_balance.png | Near-balanced classes |
| histograms.png | `thalach` skews lower and `oldpeak` skews higher for the disease class |
| correlation_heatmap.png | Strongest correlates of target: `thal`, `ca`, `oldpeak`, `exang`, `cp`, `thalach` (negative) |
| boxplots_by_target.png | Disease patients: lower max heart rate, higher ST depression |
| categorical_vs_target.png | Chest pain type 4 and exercise angina strongly associated with disease |

==TODO: insert the images here, e.g. `![class balance](reports/figures/class_balance.png)`, and
rewrite the observations in your own words after viewing the plots.==

## 4. Features and modelling

All preprocessing lives inside one scikit-learn `Pipeline`, so the exact same transformations are
applied in training and serving:

- Numeric (`age, trestbps, chol, thalach, oldpeak`): `StandardScaler`
- Categorical (`sex, cp, fbs, restecg, exang, slope, ca, thal`): `OneHotEncoder(handle_unknown="ignore")`

Data split: 80/20 stratified (random_state 42). Hyper-parameters tuned with 5-fold stratified
`GridSearchCV` on the training set, optimising ROC-AUC. The test set was only used for the final
reported numbers.

| Model | Best params | CV acc | CV precision | CV recall | CV ROC-AUC | Test acc | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | C=1 | 0.847 | 0.877 | 0.783 | 0.902 | 0.885 | 0.929 | 0.966 |
| Random Forest | 400 trees, depth 5, min leaf 3 | 0.814 | 0.831 | 0.755 | 0.904 | 0.852 | 0.893 | 0.955 |

Random Forest has marginally the highest CV ROC-AUC (0.904 vs 0.902) so it is the model selected
automatically by `src/train.py` and packaged. The difference is within noise (CV std is ~0.03-0.04);
logistic regression is simpler and equally good, which is worth noting as a trade-off.
==TODO: re-run training yourself and update this table with your numbers from models/metrics.json.==

## 5. Experiment tracking (MLflow)

Each run logs: parameters (model, CV folds, best hyper-parameters), CV and test metrics, confusion
matrix and ROC plots (artifacts), and the fitted pipeline as an MLflow model. Backend is a local
SQLite DB (`mlflow.db`) with artifacts in `mlartifacts/`. View with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

==TODO: add screenshots: MLflow experiment list, run comparison, artifacts page
(save into screenshots/ as mlflow_*.png).==

## 6. Packaging and reproducibility

- Final pipeline saved as `models/heart_pipeline.joblib` (preprocessing + classifier in one object)
  and also logged to MLflow.
- `requirements.txt` pins all training dependencies; `requirements-serve.txt` is the slim set used
  in the Docker image.
- Random seeds are fixed in `src/config.py`.

## 7. Testing and CI/CD

- `tests/` contains 15 tests: data cleaning, split, pipeline fitting, unseen-category handling,
  metrics, inference schema, and API endpoints (health, valid prediction, missing field, invalid
  `thal`, metrics).
- `.github/workflows/ci.yml` runs on every push/PR: install -> flake8 -> download data ->
  train -> pytest -> build Docker image and smoke-test `/health` and `/predict` -> upload
  models/reports/MLflow data as artifacts. Any lint, test or build failure fails the workflow.

==TODO: screenshot of a green GitHub Actions run and of the uploaded artifacts section
(screenshots/ci_run.png, ci_artifacts.png). Optionally one failing run to show it fails on errors.==

## 8. Containerisation

`Dockerfile` (python:3.12-slim, non-root user, healthcheck) serves `api.main:app` with uvicorn.

`POST /predict` accepts the 13 features as JSON and returns `prediction`, `confidence` and
`probability_disease`. Inputs are validated by pydantic (ranges, `thal` in {3, 6, 7}).

==TODO: screenshot of `docker build`, `docker run` and a successful curl (screenshots/docker_*.png).==

## 9. Deployment

Manifests in `k8s/`: Deployment (2 replicas, readiness/liveness probes, resource limits), Service
of type LoadBalancer and an optional Ingress. Deployed on ==TODO: Docker Desktop Kubernetes /
Minikube==.

==TODO: screenshots of `kubectl get pods,svc`, and the endpoint responding (screenshots/k8s_*.png).
If deployed publicly, put the URL here: ...==

## 10. Monitoring and logging

- Every request is logged (method, path, status, latency) and prediction results are logged.
- `/metrics` exposes Prometheus metrics: `api_requests_total`, `api_request_latency_seconds`,
  `model_predictions_total`.
- `docker compose up` starts the API, Prometheus and Grafana with a pre-provisioned dashboard
  (request rate, p95 latency, predictions per class, error rate).

==TODO: screenshots of Prometheus targets page and Grafana dashboard after running
`python scripts/load_test.py` (screenshots/monitoring_*.png).==

## 11. Architecture

```
 UCI repository
      |  download_data.py
      v
 raw CSV --> clean/impute --> heart_clean.csv --> EDA (figures)
                                  |
                                  v
                    train.py: Pipeline(StandardScaler + OneHot + model)
                    GridSearchCV 5-fold  ----> MLflow (params, metrics, plots, model)
                                  |
                                  v
                       models/heart_pipeline.joblib
                                  |
   GitHub push --> GitHub Actions (lint, train, test, docker build) 
                                  |
                                  v
                         Docker image (FastAPI)
                                  |
                                  v
            Kubernetes: Deployment x2 --> Service (LoadBalancer) / Ingress --> client
                                  |
                           /metrics scrape
                                  v
                       Prometheus ---> Grafana dashboard
```

==TODO (optional): redraw this as an image (draw.io) and embed it.==

## 12. Limitations and next steps

- Only 303 rows, so test metrics have high variance.
- No automatic retraining or drift detection; the next step would be data-drift metrics and a
  scheduled retraining workflow.
- Model is a decision-support demo, not a medical device.

## 13. Demo video

==TODO: link to your short screen recording of the pipeline (train -> CI -> docker -> k8s -> dashboard).==
