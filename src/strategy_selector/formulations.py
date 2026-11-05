"""Five ways to turn user features into a strategy choice, behind one sklearn-style interface.

Classes are encoded alphabetically (0 collaborative, 1 content_based, 2 popularity) to match
StrategyClassifier's label encoder; popularity wins score ties, exactly as the labeller does.
Every formulation exposes fit(X, scores), predict_proba(X) (n x 3 strategy preference) and
classes_, so any of them can be saved and used by StrategyClassifier unchanged.

  multiclass   one classifier on the winning strategy
  ovr          one binary "is strategy s the winner?" classifier per strategy
  pairwise     three "a beats b?" classifiers (trained only where a and b differ), summed wins
  regression   predict each strategy's expected hit-rate, choose the highest
  ranking      LambdaRank over the three strategies of each user (strategy-specific features)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight

SCORE_COLS = ["cf_score", "cb_score", "pop_score"]
INNER_COLS = ["inner_cf", "inner_cb", "inner_pop"]
CLASSES = np.arange(3)


def winner(scores: np.ndarray) -> np.ndarray:
    """Index of the best strategy per row; popularity (2) wins any tie for first place."""
    top = scores.max(axis=1, keepdims=True)
    return np.where(scores[:, 2] == top[:, 0], 2, scores.argmax(axis=1))


def margin(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _fit(factory, X, y, weights=None):
    model = factory()
    try:
        return model.fit(X, y, sample_weight=weights)
    except (TypeError, ValueError):
        return model.fit(X, y)


class _Constant:
    """Stand-in for a binary classifier when the training labels contain only one outcome."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def _fit_binary(factory, X, y):
    """Fit a balanced binary classifier, or a constant when only one outcome occurs."""
    if len(np.unique(y)) < 2:
        return _Constant(float(np.mean(y)) if len(y) else 0.5)
    return _fit(factory, X, y, compute_sample_weight("balanced", y))


def _softmax(raw: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (raw - raw.max(axis=1, keepdims=True)) / temperature
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Base(BaseEstimator, ClassifierMixin):
    classes_ = CLASSES

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class MulticlassForm(_Base):
    def __init__(self, factory, margin_weight: float = 0.0):
        self.factory, self.margin_weight = factory, margin_weight

    def fit(self, X, scores):
        y = winner(scores)
        weights = compute_sample_weight("balanced", y)
        if self.margin_weight:   # trust clear-cut labels more; every user is still used
            weights = weights * (1.0 + self.margin_weight * np.minimum(margin(scores) / 0.1, 1.0))
        self.model_ = _fit(self.factory, X, y, weights)
        return self

    def predict_proba(self, X):
        full = np.zeros((len(X), 3))
        full[:, list(self.model_.classes_)] = self.model_.predict_proba(X)
        return full


class OneVsRestForm(_Base):
    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, scores):
        y = winner(scores)
        self.models_ = [_fit_binary(self.factory, X, (y == c).astype(int)) for c in CLASSES]
        return self

    def predict_proba(self, X):
        raw = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        return raw / raw.sum(axis=1, keepdims=True)


class PairwiseForm(_Base):
    PAIRS = ((0, 1), (0, 2), (1, 2))

    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, scores):
        self.models_ = {}
        for a, b in self.PAIRS:
            keep = scores[:, a] != scores[:, b]
            y = (scores[keep, a] > scores[keep, b]).astype(int)
            self.models_[(a, b)] = _fit_binary(self.factory, X[keep], y)
        return self

    def predict_proba(self, X):
        wins = np.zeros((len(X), 3))
        for (a, b), model in self.models_.items():
            p = model.predict_proba(X)[:, 1]
            wins[:, a] += p
            wins[:, b] += 1 - p
        return wins / wins.sum(axis=1, keepdims=True)


class RegressionForm(_Base):
    def __init__(self, factory, temperature: float = 0.05):
        self.factory, self.temperature = factory, temperature

    def fit(self, X, scores):
        self.models_ = [self.factory().fit(X, scores[:, c]) for c in CLASSES]
        return self

    def predict_scores(self, X):
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict_proba(self, X):
        return _softmax(self.predict_scores(X), self.temperature)


class RankingForm(_Base):
    """LambdaRank where each (user, strategy) is a row: user features + strategy id + its inner score."""

    def __init__(self, params: dict | None = None, temperature: float = 0.5):
        self.params, self.temperature = params or {}, temperature

    @staticmethod
    def _expand(X: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in CLASSES:
            part = X.copy()
            part["strategy_id"] = c
            part["own_inner"] = X[INNER_COLS[c]].to_numpy()
            part["own_gap"] = part["own_inner"] - X[INNER_COLS].max(axis=1).to_numpy()
            rows.append(part)
        stacked = pd.concat(rows, keys=CLASSES, names=["strategy"])
        return stacked.swaplevel().sort_index()      # user-major: three consecutive rows per user

    def fit(self, X, scores):
        from lightgbm import LGBMRanker
        grades = (pd.DataFrame(scores).rank(axis=1, method="dense").to_numpy() - 1).astype(int)  # 0..2
        expanded = self._expand(X.reset_index(drop=True))
        self.model_ = LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05, num_leaves=8,
                                 min_child_samples=20, verbose=-1, n_jobs=1, random_state=42, **self.params)
        self.model_.fit(expanded, grades.reshape(-1), group=[3] * len(X))
        return self

    def predict_scores(self, X):
        return self.model_.predict(self._expand(X.reset_index(drop=True))).reshape(-1, 3)

    def predict_proba(self, X):
        return _softmax(self.predict_scores(X), self.temperature)


def decision_quality(scores: np.ndarray, chosen: np.ndarray) -> dict:
    """What the choice is worth: realised hit-rate of the chosen strategy vs fixed and oracle policies."""
    n = np.arange(len(scores))
    realised = float(scores[n, chosen].mean())
    return {"realised_hit_rate": realised, "always_popularity": float(scores[:, 2].mean()),
            "always_collaborative": float(scores[:, 0].mean()),
            "oracle_hit_rate": float(scores.max(axis=1).mean()),
            "regret": float(scores.max(axis=1).mean() - realised)}
