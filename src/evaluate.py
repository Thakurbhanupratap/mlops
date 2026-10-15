"""Metric helpers and plots shared by training and tests."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_confusion_matrix(y_true, y_pred, path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
