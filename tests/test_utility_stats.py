"""Tests for bootstrap statistics, the utility selector with fallback, and the content model."""

import functools

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.strategy_selector.content_model import ContentModel
from src.strategy_selector.stats import bootstrap_ci, mcnemar_p, paired_bootstrap, proportion_ci
from src.strategy_selector.utility import UtilitySelector, choose_tau


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_sample_size():
    rng = np.random.default_rng(0)
    small, large = rng.binomial(1, 0.3, 100), rng.binomial(1, 0.3, 4000)
    m1, lo1, hi1 = bootstrap_ci(small)
    m2, lo2, hi2 = bootstrap_ci(large)
    assert lo1 <= m1 <= hi1 and lo2 <= m2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)


def test_paired_bootstrap_detects_a_real_gain_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.binomial(1, 0.4, 1500).astype(float)
    better = np.clip(base + (rng.random(1500) < 0.15), 0, 1)
    assert paired_bootstrap(better, base)["ci_low"] > 0
    same = paired_bootstrap(base, base.copy())
    assert same["mean_diff"] == 0.0 and same["p_not_better"] == 1.0


def test_mcnemar_and_wilson_interval():
    a = np.array([1] * 60 + [0] * 40, dtype=bool)
    assert mcnemar_p(a, a) == 1.0
    assert mcnemar_p(np.ones(50, bool), np.zeros(50, bool)) < 0.001
    p, lo, hi = proportion_ci(60, 100)
    assert lo < p < hi and p == pytest.approx(0.6)


def _scores(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    scores = np.column_stack([0.15 + 0.08 * X["a"], np.full(n, 0.02), np.full(n, 0.12)]).clip(0, 1)
    return X, scores + rng.normal(0, 0.01, scores.shape)


def test_utility_selector_uses_fallback_when_gap_is_below_tau():
    X, scores = _scores()
    model = UtilitySelector(functools.partial(RandomForestRegressor, n_estimators=40, random_state=0), tau=0.0).fit(X, scores)
    eager = model.predict(X)
    cautious = UtilitySelector(model.factory, tau=0.5).fit(X, scores).predict(X)
    assert (cautious == 2).all()                 # nothing beats popularity by 0.5 hit-rate points
    assert (eager != 2).any()                    # without a margin it does pick collaborative sometimes
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0) and (proba.argmax(axis=1) == eager).all()


def test_choose_tau_prefers_the_margin_with_best_realised_hit_rate():
    scores = np.array([[0.30, 0.0, 0.10]] * 50 + [[0.05, 0.0, 0.20]] * 50)
    util = np.array([[0.31, 0.0, 0.10]] * 50 + [[0.22, 0.0, 0.20]] * 50)   # second group: predicted gap is wrong
    tau, table = choose_tau(util, scores, [0.0, 0.05, 0.1])
    assert tau == 0.05 and table[0.05] > table[0.0]


def test_content_model_recommends_similar_unseen_items_and_scores_hits():
    movies = pd.DataFrame({
        "movieId": range(1, 9), "genres": ["Action|Sci-Fi"] * 4 + ["Romance|Drama"] * 4,
        "text_chunk_clean": ["space ship laser battle"] * 4 + ["love wedding family tears"] * 4,
        "vote_average": [7, 8, 6, 7, 7, 6, 8, 7], "vote_count": [100, 500, 50, 300, 200, 80, 400, 150]})
    train = pd.DataFrame({"movieId": [1, 2], "rating": [5.0, 4.5]})
    for options in ({}, {"profile": "liked_centered"}, {"genre_weight": 1.0}, {"quality_alpha": 1.0}):
        model = ContentModel(movies, **options)
        top = model.top_items(train, 2)
        assert set(top) == {3, 4} and not ({1, 2} & set(top))
        assert model.hit_rate(train, [3, 7], 2) == pytest.approx(0.5)
