"""Load the saved pipeline and score patient records."""
import joblib
import pandas as pd

from src import config

_model = None


def load_model(path=config.MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def predict(record: dict, model=None) -> dict:
    model = model or load_model()
    frame = pd.DataFrame([record])[config.FEATURES]
    proba = float(model.predict_proba(frame)[0, 1])
    return {
        "prediction": int(proba >= 0.5),
        "confidence": round(max(proba, 1 - proba), 4),
        "probability_disease": round(proba, 4),
    }
