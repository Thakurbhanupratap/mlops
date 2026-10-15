"""Train, tune and log models with MLflow; save the best pipeline."""
import json
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src import config
from src.data_processing import load_clean, split
from src.evaluate import compute_metrics, save_confusion_matrix, save_roc_curve
from src.pipeline import build_pipeline

SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# Small, explicit search spaces (keys are pipeline step names).
SEARCH_SPACES = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "rf": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3],
    },
}


def run_experiment(model_name, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    base = build_pipeline(model_name)

    with mlflow.start_run(run_name=f"{model_name}-tuned") as run:
        search = GridSearchCV(base, SEARCH_SPACES[model_name], cv=cv,
                              scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
        cv_metrics = {f"cv_{m}": float(cv_res[f"test_{m}"].mean()) for m in SCORING}
        cv_metrics.update({f"cv_{m}_std": float(cv_res[f"test_{m}"].std()) for m in SCORING})

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_metrics = {f"test_{k}": float(v)
                        for k, v in compute_metrics(y_test, y_pred, y_proba).items()}

        mlflow.log_param("model", model_name)
        mlflow.log_param("cv_folds", config.CV_FOLDS)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        with tempfile.TemporaryDirectory() as tmp:
            cm = Path(tmp) / "confusion_matrix.png"
            roc = Path(tmp) / "roc_curve.png"
            save_confusion_matrix(y_test, y_pred, cm, f"{model_name} - confusion matrix")
            save_roc_curve(y_test, y_proba, roc, f"{model_name} - ROC")
            mlflow.log_artifact(str(cm), "plots")
            mlflow.log_artifact(str(roc), "plots")

        mlflow.sklearn.log_model(best, "model", serialization_format="cloudpickle")
        return {
            "model": model_name,
            "run_id": run.info.run_id,
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "estimator": best,
            **cv_metrics,
            **test_metrics,
        }


def main():
    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"))
    if not mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT):
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=(config.ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = split(load_clean())
    results = [run_experiment(name, X_train, X_test, y_train, y_test)
               for name in SEARCH_SPACES]

    winner = max(results, key=lambda r: r["cv_roc_auc"])
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(winner["estimator"], config.MODEL_PATH)

    summary = [{k: v for k, v in r.items() if k != "estimator"} for r in results]
    config.METRICS_PATH.write_text(json.dumps(
        {"selected": winner["model"], "runs": summary}, indent=2))

    for r in summary:
        print(f"{r['model']:>7}  cv_auc={r['cv_roc_auc']:.3f}  "
              f"test_auc={r['test_roc_auc']:.3f}  test_recall={r['test_recall']:.3f}")
    print(f"selected model: {winner['model']} -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
