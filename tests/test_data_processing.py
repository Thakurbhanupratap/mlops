import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_processing import clean, split


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(100, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(100, 200, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.random(n) * 4,
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": rng.integers(0, 5, n),
    })
    df.loc[3, "ca"] = np.nan
    df.loc[7, "thal"] = np.nan
    return df


def test_clean_removes_missing(raw_df):
    assert clean(raw_df).isna().sum().sum() == 0


def test_clean_target_is_binary(raw_df):
    assert set(clean(raw_df)[config.TARGET].unique()) <= {0, 1}


def test_clean_categoricals_are_int(raw_df):
    out = clean(raw_df)
    for col in config.CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(out[col])


def test_split_is_stratified_and_disjoint(raw_df):
    out = clean(raw_df)
    X_tr, X_te, y_tr, y_te = split(out)
    assert len(X_tr) + len(X_te) == len(out)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert list(X_tr.columns) == config.FEATURES
