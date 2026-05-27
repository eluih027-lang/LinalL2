from __future__ import annotations

import numpy as np


def _ratio(top: float, bottom: float) -> float:
    return 0.0 if bottom == 0 else float(top / bottom)


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    actual = np.asarray(y_true).astype(int).reshape(-1)
    predicted = np.asarray(y_pred).astype(int).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("metric arrays must have equal shape")

    return {
        "tp": int(((actual == 1) & (predicted == 1)).sum()),
        "tn": int(((actual == 0) & (predicted == 0)).sum()),
        "fp": int(((actual == 0) & (predicted == 1)).sum()),
        "fn": int(((actual == 1) & (predicted == 0)).sum()),
    }


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    box = binary_counts(y_true, y_pred)
    tp, tn, fp, fn = box["tp"], box["tn"], box["fp"], box["fn"]

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **{name: float(count) for name, count in box.items()},
    }


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return FPR, TPR and thresholds for a binary ROC curve.

    The implementation is intentionally manual and does not use sklearn.metrics.
    """
    actual = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(y_score, dtype=float).reshape(-1)
    if actual.shape != score.shape:
        raise ValueError("y_true and y_score must have equal shape")
    if not set(np.unique(actual)) <= {0, 1}:
        raise ValueError("ROC is defined for labels encoded by 0 and 1")

    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    fpr: list[float] = []
    tpr: list[float] = []
    for threshold in thresholds:
        predicted = (score >= threshold).astype(int)
        counts = binary_counts(actual, predicted)
        tpr.append(_ratio(counts["tp"], counts["tp"] + counts["fn"]))
        fpr.append(_ratio(counts["fp"], counts["fp"] + counts["tn"]))
    return np.asarray(fpr), np.asarray(tpr), thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC by trapezoidal integration over manual ROC points."""
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def extended_report(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    report = classification_report(y_true, y_pred)
    if y_score is not None:
        report["roc_auc"] = roc_auc_score(y_true, y_score)
    return report


confusion_matrix_binary = binary_counts
