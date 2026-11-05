"""Utility-based strategy selection with an uncertainty-aware fallback.

Instead of classifying a noisy "winning strategy" label, predict each strategy's expected
hit-rate (its utility) for the user and pick the best. A non-fallback strategy is only
chosen when its predicted utility beats the fallback's by more than `tau`: small predicted
gaps are within the model's own uncertainty, so the reliable fallback (popularity) is used.

    chosen = argmax_s ( utility_s - tau * [s != fallback] )

tau is a plain, interpretable quantity (hit-rate points) tuned by cross-validation on
development users only. predict_proba is a softmax over the same adjusted utilities, so the
saved model drops straight into StrategyClassifier (argmax == the rule above).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.strategy_selector.formulations import CLASSES

FALLBACK = 2  # popularity


class UtilitySelector(BaseEstimator, ClassifierMixin):
    classes_ = CLASSES

    def __init__(self, factory, tau: float = 0.0, temperature: float = 0.02, fallback: int = FALLBACK):
        self.factory, self.tau, self.temperature, self.fallback = factory, tau, temperature, fallback

    def fit(self, X, scores):
        self.models_ = [self.factory().fit(X, scores[:, c]) for c in CLASSES]
        return self

    def utilities(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])

    def adjusted(self, X, tau: float | None = None) -> np.ndarray:
        penalty = np.full(3, self.tau if tau is None else tau)
        penalty[self.fallback] = 0.0
        return self.utilities(X) - penalty

    def predict_proba(self, X):
        z = self.adjusted(X) / self.temperature
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.adjusted(X).argmax(axis=1)


def choose_tau(utilities: np.ndarray, scores: np.ndarray, grid) -> tuple[float, dict]:
    """Pick the fallback margin that maximises realised hit-rate on out-of-fold utilities."""
    n = np.arange(len(scores))
    table = {}
    for tau in grid:
        penalty = np.full(3, float(tau))
        penalty[FALLBACK] = 0.0
        chosen = (utilities - penalty).argmax(axis=1)
        table[float(tau)] = float(scores[n, chosen].mean())
    best = max(table, key=lambda t: (round(table[t], 6), -t))   # ties: the smaller (less cautious) margin
    return best, table
