import pytest
from fastapi.testclient import TestClient

from src import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="model not trained yet")

SAMPLE = dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2,
              thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1


def test_predict_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_invalid_thal(client):
    assert client.post("/predict", json={**SAMPLE, "thal": 5}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
