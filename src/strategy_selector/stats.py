"""Small statistics helpers: bootstrap confidence intervals and paired significance tests.

Users are the unit of resampling (each user contributes one outcome), so intervals reflect
how much a result would move with a different sample of users, not of ratings.
"""

import numpy as np
from scipy.stats import binomtest

SEED = 42


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = SEED) -> tuple[float, float, float]:
    """(mean, low, high) of the mean of per-user values by percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Mean of (a - b) over users with a 95% CI and a one-sided bootstrap p-value for 'a is not better'."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {"mean_diff": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "p_not_better": float((means <= 0).mean())}


def mcnemar_p(correct_a, correct_b) -> float:
    """Exact McNemar p-value for 'two classifiers have equal error rates' on the same users."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)


def proportion_ci(successes: int, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """Wilson interval for an accuracy-style proportion."""
    from scipy.stats import norm
    z, p = norm.ppf(1 - (1 - level) / 2), successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(p), float(centre - half), float(centre + half)
