"""Cross-validated model comparison and one-shot held-out evaluation for the selector.

Rules enforced here: SMOTE only ever sees a training fold (resampling happens
inside each fold, after the split); candidates are ranked by cross-validated
macro-F1 on the development users only; the held-out users are scored once, for
the chosen candidate, and never used to choose anything.
"""

import numpy as np
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.ml_metrics import classification_metrics, majority_baseline_accuracy
from src.strategy_selector.classifier import StrategyClassifier

SEED = 42

# name -> (estimator factory, oversample the training fold with SMOTE)
CANDIDATES = {
    "rf_smote": (lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), True),
    "rf_balanced": (lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1,
                                                   class_weight="balanced"), False),
    "gb_smote": (lambda: GradientBoostingClassifier(random_state=SEED), True),
    "hgb_balanced": (lambda: HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"),
                     False),
}


def fit_candidate(kind: str, X, y, class_names: list):
    """Fit one candidate on a TRAINING fold (resampling it first when the candidate uses SMOTE)."""
    factory, use_smote = CANDIDATES[kind]
    if use_smote:
        X, y, _ = StrategyClassifier._resample_training_fold(X, y, class_names)
    model = factory()
    model.fit(X, y)
    return model


def _proba(model, X, n_classes: int) -> np.ndarray:
    """Probabilities aligned to all classes (a fold may have trained without a rare class)."""
    full = np.zeros((len(X), n_classes))
    full[:, list(model.classes_)] = model.predict_proba(X)
    return full


def _auc(y_true, proba) -> float | None:
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                   labels=list(range(proba.shape[1]))))
    except ValueError:
        return None


def cross_validate(kind: str, X, y, class_names: list, folds: int = 5, seed: int = SEED) -> dict:
    """Stratified k-fold CV; metrics are per-fold means plus pooled out-of-fold AUC/confusion."""
    y = np.asarray(y)
    n_classes = len(class_names)
    counts = np.bincount(y, minlength=n_classes)
    folds = int(min(folds, counts[counts > 0].min()))
    if folds < 2:
        raise ValueError("A class has fewer than 2 users; cannot cross-validate.")
    oof = np.zeros((len(y), n_classes))
    per_fold = {"macro_f1": [], "balanced_accuracy": [], "accuracy": [], "baseline_accuracy": []}
    for train_idx, test_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        model = fit_candidate(kind, X.iloc[train_idx], y[train_idx], class_names)
        proba = _proba(model, X.iloc[test_idx], n_classes)
        pred = proba.argmax(axis=1)
        oof[test_idx] = proba
        per_fold["macro_f1"].append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        per_fold["balanced_accuracy"].append(balanced_accuracy_score(y[test_idx], pred))
        per_fold["accuracy"].append(accuracy_score(y[test_idx], pred))
        per_fold["baseline_accuracy"].append(majority_baseline_accuracy(y[train_idx], y[test_idx])[1])
    pooled = classification_metrics(y, oof.argmax(axis=1), oof, class_names)
    result = {f"cv_{k}": float(np.mean(v)) for k, v in per_fold.items()}
    result["cv_macro_f1_std"] = float(np.std(per_fold["macro_f1"]))
    result.update(folds=folds, cv_roc_auc=pooled.get("roc_auc"), cv_confusion_matrix=pooled["confusion_matrix"])
    return result


def compare_candidates(X, y, class_names: list, folds: int = 5, kinds=None) -> tuple[list[dict], str]:
    """Cross-validate every candidate on the development users; best = highest mean macro-F1."""
    table = [{"model": kind, **cross_validate(kind, X, y, class_names, folds)}
             for kind in (kinds or CANDIDATES)]
    best = max(table, key=lambda row: row["cv_macro_f1"])["model"]
    return table, best


def evaluate_holdout(kind: str, X_dev, y_dev, X_hold, y_hold, class_names: list) -> dict:
    """Train on all development users, then score the untouched held-out users exactly once."""
    model = fit_candidate(kind, X_dev, np.asarray(y_dev), class_names)
    proba = _proba(model, X_hold, len(class_names))
    pred = proba.argmax(axis=1)
    metrics = classification_metrics(np.asarray(y_hold), pred, proba, class_names, y_train=np.asarray(y_dev))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_hold, pred))
    metrics["macro_f1"] = float(f1_score(y_hold, pred, average="macro", zero_division=0))
    metrics["n_test"] = int(len(y_hold))
    metrics["majority_baseline_balanced_accuracy"] = 1.0 / len(np.unique(y_dev))
    return metrics
