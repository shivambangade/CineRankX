"""Classifier metrics — the "machine_learning" KPI group.

This is the single implementation of the strategy classifier's metrics.
`StrategyClassifier.fit()` used to compute them inline; that logic now lives
here so the evaluation report and the classifier cannot drift apart, and so
the report never has to re-derive numbers the classifier already produced.

Deliberately depends on nothing inside src/ beyond this file: the classifier
imports it, so importing the classifier back would be a cycle.
"""

from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metric_names import (
    ACCURACY,
    CONFUSION_MATRIX,
    F1,
    PRECISION,
    RECALL,
    ROC_AUC,
)


def majority_baseline_accuracy(y_train, y_test) -> tuple[int, float]:
    """Accuracy of always predicting the most common TRAINING label.

    Takes training labels explicitly because the caller may have resampled its
    training fold (e.g. SMOTE): the baseline has to describe the real class
    distribution, and reading it off a balanced fold would make "most common
    label" an arbitrary tie-break.

    Returns:
        (majority_class, accuracy_of_always_predicting_it_on_y_test)
    """
    majority_class = Counter(y_train).most_common(1)[0][0]
    return majority_class, float(np.mean(np.asarray(y_test) == majority_class))


def roc_auc(y_test, y_pred_proba, n_classes: int) -> float | None:
    """ROC-AUC, one-vs-rest for the multi-class case.

    Returns None when it is undefined (fewer than two classes present).
    Returns 0.0 if sklearn refuses the computation — which happens when a class
    is absent from y_test, so the score genuinely cannot be formed.
    """
    if n_classes == 2:
        return float(roc_auc_score(y_test, y_pred_proba[:, 1]))
    if n_classes > 2:
        try:
            return float(roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
        except ValueError:
            return 0.0
    return None


def classification_metrics(y_test, y_pred, y_pred_proba, class_names: list, y_train=None) -> dict:
    """Compute every "machine_learning" KPI for one held-out evaluation.

    Args:
        y_test / y_pred: encoded true and predicted labels for the held-out set.
        y_pred_proba: predicted probabilities, shape (n_samples, n_classes).
        class_names: decoded class names, index-aligned to the encoded labels.
        y_train: pre-resampling training labels, for the majority baseline.
            Omitted -> no baseline is reported.

    Returns:
        Dict keyed for programmatic use (accuracy, precision, recall, f1,
        per_class, confusion_matrix, confusion_matrix_labels, roc_auc,
        majority_baseline_accuracy). Use `to_report_names()` to render it under
        the exact eval_config.yaml KPI names.
    """
    class_labels = list(range(len(class_names)))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=class_labels, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(class_names, precisions, recalls, f1s, supports)
    }

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
        "confusion_matrix_labels": list(class_names),
    }

    auc = roc_auc(y_test, y_pred_proba, len(class_names))
    if auc is not None:
        metrics["roc_auc"] = auc

    if y_train is not None:
        _, baseline = majority_baseline_accuracy(y_train, y_test)
        metrics["majority_baseline_accuracy"] = baseline

    return metrics


def to_report_names(metrics: dict) -> dict:
    """Render computed metrics under the exact eval_config.yaml KPI names."""
    report = {
        ACCURACY: metrics["accuracy"],
        PRECISION: metrics["precision"],
        RECALL: metrics["recall"],
        F1: metrics["f1"],
        CONFUSION_MATRIX: metrics["confusion_matrix"],
    }
    # ROC-AUC is absent rather than zero when only one class was present, so
    # the report can say "undefined" instead of implying a real 0.0 score.
    report[ROC_AUC] = metrics.get("roc_auc")
    return report
