"""Cross-validated model comparison and one-shot held-out evaluation for the selector.

Rules enforced here: SMOTE only ever sees a training fold (resampling happens
inside each fold, after the split); candidates are ranked by cross-validated
macro-F1 on the development users only; the held-out users are scored once, for
the chosen candidate, and never used to choose anything.
"""

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluation.ml_metrics import classification_metrics, majority_baseline_accuracy
from src.strategy_selector.classifier import StrategyClassifier

SEED = 42

# name -> (estimator factory(**params), oversample the training fold with SMOTE)
def _lgbm(**p):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, class_weight="balanced", n_jobs=1, verbose=-1, **p)


def _scaled(estimator):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), estimator)


CANDIDATES = {
    "rf_smote": (lambda **p: RandomForestClassifier(**{"n_estimators": 150, "random_state": SEED,
                                                       "n_jobs": -1, **p}), True),
    "rf_balanced": (lambda **p: RandomForestClassifier(**{"n_estimators": 150, "random_state": SEED, "n_jobs": -1,
                                                          "class_weight": "balanced", **p}), False),
    "extra_trees": (lambda **p: ExtraTreesClassifier(**{"n_estimators": 150, "random_state": SEED, "n_jobs": -1,
                                                        "class_weight": "balanced", **p}), False),
    "gb_smote": (lambda **p: GradientBoostingClassifier(random_state=SEED, **p), True),
    "hgb_balanced": (lambda **p: HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", **p),
                     False),
    "lightgbm": (_lgbm, False),
    "logreg": (lambda **p: _scaled(LogisticRegression(max_iter=3000, class_weight="balanced", **p)), False),
    "svm_rbf": (lambda **p: _scaled(SVC(probability=True, class_weight="balanced", random_state=SEED, **p)),
                False),
}

# Randomised-search spaces (searched by CV macro-F1 on DEVELOPMENT users only).
TUNING = {
    # n_estimators is searched too: single-user inference time grows linearly with tree count.
    "rf_balanced": {"n_estimators": [60, 100, 150, 300], "max_depth": [4, 6, 8, 12, None],
                    "min_samples_leaf": [1, 3, 5, 10, 20], "max_features": ["sqrt", 0.3, 0.5]},
    "extra_trees": {"n_estimators": [60, 100, 150, 300], "max_depth": [4, 6, 8, 12, None],
                    "min_samples_leaf": [1, 3, 5, 10, 20], "max_features": ["sqrt", 0.3, 0.5]},
    "lightgbm": {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [4, 8, 16],
                 "min_child_samples": [10, 20, 40], "subsample": [0.7, 1.0], "colsample_bytree": [0.5, 0.8, 1.0],
                 "reg_lambda": [0.0, 1.0, 10.0]},
    "logreg": {},
    "svm_rbf": {"svc__C": [0.3, 1, 3, 10], "svc__gamma": ["scale", 0.01, 0.03]},
}


def build_estimator(kind: str, params: dict | None = None):
    """Unfitted estimator for a candidate; 'step__param' keys configure pipeline steps."""
    factory, _ = CANDIDATES[kind]
    params = params or {}
    model = factory(**{k: v for k, v in params.items() if "__" not in k})
    nested = {k: v for k, v in params.items() if "__" in k}
    return model.set_params(**nested) if nested else model


def fit_candidate(kind: str, X, y, class_names: list, params: dict | None = None):
    """Fit one candidate on a TRAINING fold (resampling it first when the candidate uses SMOTE)."""
    _, use_smote = CANDIDATES[kind]
    if use_smote:
        X, y, _ = StrategyClassifier._resample_training_fold(X, y, class_names)
    return build_estimator(kind, params).fit(X, y)


def tune_candidate(kind: str, X, y, folds: int = 3, n_iter: int = 12, seed: int = SEED) -> dict:
    """Randomised search of a candidate's hyperparameters by CV macro-F1 (development users only)."""
    space = TUNING.get(kind)
    if not space:
        return {}
    from sklearn.model_selection import RandomizedSearchCV
    factory, _ = CANDIDATES[kind]
    search = RandomizedSearchCV(factory(), space, n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
                                scoring="f1_macro", cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=1, refit=False)
    search.fit(X, y)
    return {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}


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


def cross_validate(kind: str, X, y, class_names: list, folds: int = 5, seed: int = SEED,
                   params: dict | None = None) -> dict:
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
        model = fit_candidate(kind, X.iloc[train_idx], y[train_idx], class_names, params)
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


def compare_candidates(X, y, class_names: list, folds: int = 5, kinds=None, tune: bool = False,
                       log=print) -> tuple[list[dict], str, dict]:
    """Cross-validate (optionally tuned) candidates on development users; best = top mean macro-F1."""
    table, tuned = [], {}
    for kind in (kinds or CANDIDATES):
        params = tune_candidate(kind, X, y) if tune else {}
        tuned[kind] = params
        row = {"model": kind, "params": params, **cross_validate(kind, X, y, class_names, folds, params=params)}
        table.append(row)
        log(f"  {kind:13s} macro-F1 {row['cv_macro_f1']:.3f}  bal.acc {row['cv_balanced_accuracy']:.3f}  "
            f"acc {row['cv_accuracy']:.3f}")
    best = max(table, key=lambda row: row["cv_macro_f1"])["model"]
    return table, best, tuned


def evaluate_holdout(kind: str, X_dev, y_dev, X_hold, y_hold, class_names: list,
                     params: dict | None = None) -> dict:
    """Train on all development users, then score the untouched held-out users exactly once."""
    model = fit_candidate(kind, X_dev, np.asarray(y_dev), class_names, params)
    proba = _proba(model, X_hold, len(class_names))
    pred = proba.argmax(axis=1)
    metrics = classification_metrics(np.asarray(y_hold), pred, proba, class_names, y_train=np.asarray(y_dev))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_hold, pred))
    metrics["macro_f1"] = float(f1_score(y_hold, pred, average="macro", zero_division=0))
    metrics["n_test"] = int(len(y_hold))
    metrics["majority_baseline_balanced_accuracy"] = 1.0 / len(np.unique(y_dev))
    return metrics
