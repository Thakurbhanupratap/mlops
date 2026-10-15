"""Preprocessing + model pipelines (single object => reproducible inference)."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(model_name: str = "logreg", **params) -> Pipeline:
    if model_name == "logreg":
        defaults = dict(C=1.0, max_iter=1000, random_state=config.RANDOM_STATE)
        clf = LogisticRegression(**{**defaults, **params})
    elif model_name == "rf":
        defaults = dict(n_estimators=200, random_state=config.RANDOM_STATE)
        clf = RandomForestClassifier(**{**defaults, **params})
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])
