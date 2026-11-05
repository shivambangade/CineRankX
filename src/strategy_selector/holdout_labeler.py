"""Leak-free strategy labels: backtest on a per-user holdout the profile never sees.

Every user's ratings are split once into a PROFILE part and a HOLDOUT part.
Features come from the profile only; every strategy (and the shared popularity
ranking, similarity matrix and TF-IDF profile) is built from profile ratings
only; hit-rates are measured on the holdout. So the ratings that decide a
user's label are invisible to both the features and the strategies.
"""

import numpy as np
import pandas as pd

from src.strategy_selector.strategy_labeler import (
    _BacktestContext,
    _collaborative_hit_rate,
    _content_based_hit_rate,
    _popularity_hit_rate,
    select_best_strategy,
)

SCORE_COLUMNS = ["cb_score", "cf_score", "pop_score"]


def split_profile_holdout(ratings: pd.DataFrame, fraction: float,
                          seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user random split; each user keeps at least one rating on each side."""
    rng = np.random.default_rng(seed)
    is_holdout = np.zeros(len(ratings), dtype=bool)
    positions = ratings.groupby("userId").indices
    for rows in positions.values():
        n_hold = min(max(1, int(round(len(rows) * fraction))), len(rows) - 1)
        if n_hold > 0:
            is_holdout[rng.choice(rows, size=n_hold, replace=False)] = True
    return ratings[~is_holdout].reset_index(drop=True), ratings[is_holdout].reset_index(drop=True)


def label_holdout(profile: pd.DataFrame, holdout: pd.DataFrame, movies_df: pd.DataFrame,
                  top_n: int = 10, content_model=None) -> pd.DataFrame:
    """Hit-rate of each strategy on the holdout; raw winner uses the legacy tie-break."""
    ctx = _BacktestContext(profile, movies_df, top_n=top_n)
    hold_by_user = holdout.groupby("userId")["movieId"].apply(lambda s: s.values)
    rows = []
    for user_id, train in profile.groupby("userId"):
        test_ids = hold_by_user.get(user_id, np.array([]))
        cb = (content_model.hit_rate(train, test_ids, top_n) if content_model is not None
              else _content_based_hit_rate(ctx, train, test_ids))
        cf = _collaborative_hit_rate(ctx, user_id, test_ids)
        pop = _popularity_hit_rate(ctx, test_ids)
        rows.append({"userId": user_id, "best_strategy": select_best_strategy(cb, cf, pop),
                     "cb_score": cb, "cf_score": cf, "pop_score": pop, "n_holdout": len(test_ids)})
    return pd.DataFrame(rows)


def clean_labels(labels: pd.DataFrame, margin: float) -> tuple[pd.DataFrame, dict]:
    """Drop users with no clear winner: all-zero scores, or top-two within `margin`.

    Returns the kept rows (best_strategy is the strict winner) and a drop report.
    """
    scores = labels[SCORE_COLUMNS].to_numpy(dtype=float)
    ordered = np.sort(scores, axis=1)
    top, second = ordered[:, -1], ordered[:, -2]
    all_zero = top <= 0
    ambiguous = ~all_zero & ((top - second) < max(margin, 1e-12))
    keep = ~(all_zero | ambiguous)
    names = np.array(["content_based", "collaborative", "popularity"])
    kept = labels[keep].copy()
    kept["best_strategy"] = names[scores[keep].argmax(axis=1)]
    report = {"total": int(len(labels)), "kept": int(keep.sum()), "dropped_all_zero": int(all_zero.sum()),
              "dropped_ambiguous": int(ambiguous.sum()), "margin": float(margin)}
    return kept.reset_index(drop=True), report
