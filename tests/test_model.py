import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluate import compute_metrics
from src.inference import predict
from src.pipeline import build_pipeline


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "age": rng.normal(54, 9, n),
        "trestbps": rng.normal(131, 17, n),
        "chol": rng.normal(246, 50, n),
        "thalach": rng.normal(150, 22, n),
        "oldpeak": rng.random(n) * 3,
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    y = ((df["oldpeak"] + df["exang"] + (df["ca"] > 0)) > 2).astype(int)
    return df[config.FEATURES], y


@pytest.mark.parametrize("name", ["logreg", "rf"])
def test_pipeline_fits_and_predicts_probabilities(toy_data, name):
    X, y = toy_data
    model = build_pipeline(name).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_pipeline("svm")


def test_pipeline_handles_unseen_category(toy_data):
    X, y = toy_data
    model = build_pipeline("logreg").fit(X, y)
    row = X.iloc[[0]].copy()
    row["cp"] = 9
    assert model.predict(row).shape == (1,)


def test_metrics_perfect_prediction():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["accuracy"] == 1 and m["roc_auc"] == 1


def test_predict_output_schema(toy_data):
    X, y = toy_data
    model = build_pipeline("logreg").fit(X, y)
    out = predict(X.iloc[0].to_dict(), model=model)
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1
