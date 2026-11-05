"""Tests for the strategy-choice formulations and the decision-quality metric."""

import functools
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.strategy_selector.formulations import (CLASSES, INNER_COLS, MulticlassForm, OneVsRestForm,
                                                PairwiseForm, RankingForm, RegressionForm,
                                                decision_quality, margin, winner)


def synthetic(n=240, seed=0):
    """Strategy scores really depend on two features, plus noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    scores = np.column_stack([
        0.2 + 0.1 * X["a"], 0.05 + 0.02 * X["b"], 0.2 - 0.1 * X["a"]]).clip(0.0, 1.0)
    scores += rng.normal(0, 0.02, scores.shape)
    scores = scores.clip(0, 1)
    for i, col in enumerate(INNER_COLS):         # strategy-specific inner scores track the truth noisily
        X[col] = scores[:, i] + rng.normal(0, 0.03, n)
    return X, scores


def test_winner_breaks_ties_towards_popularity_and_margin_is_gap():
    scores = np.array([[0.3, 0.1, 0.3], [0.5, 0.1, 0.2], [0.0, 0.0, 0.0], [0.1, 0.4, 0.2]])
    assert winner(scores).tolist() == [2, 0, 2, 1]
    assert margin(scores) == pytest.approx([0.0, 0.3, 0.0, 0.2])


@pytest.mark.parametrize("builder", [
    lambda: MulticlassForm(functools.partial(RandomForestClassifier, n_estimators=30, random_state=0)),
    lambda: MulticlassForm(functools.partial(RandomForestClassifier, n_estimators=30, random_state=0), 2.0),
    lambda: OneVsRestForm(functools.partial(RandomForestClassifier, n_estimators=30, random_state=0)),
    lambda: PairwiseForm(functools.partial(RandomForestClassifier, n_estimators=30, random_state=0)),
    lambda: RegressionForm(functools.partial(RandomForestRegressor, n_estimators=30, random_state=0)),
    lambda: RankingForm(),
])
def test_every_formulation_learns_the_signal_and_returns_valid_probabilities(builder):
    X, scores = synthetic()
    train, test = slice(0, 180), slice(180, None)
    model = builder().fit(X.iloc[train].reset_index(drop=True), scores[train])
    proba = model.predict_proba(X.iloc[test].reset_index(drop=True))
    assert proba.shape == (60, 3) and np.allclose(proba.sum(axis=1), 1.0)
    assert (model.classes_ == CLASSES).all()
    chosen = proba.argmax(axis=1)
    value = decision_quality(scores[test], chosen)
    assert value["realised_hit_rate"] > np.mean([scores[test][:, 1].mean()])    # beats the weak strategy
    assert value["realised_hit_rate"] <= value["oracle_hit_rate"] + 1e-12


def test_pairwise_ignores_users_where_two_strategies_tie():
    X, scores = synthetic()
    scores[:, 0] = scores[:, 1]                     # strategies 0 and 1 always tie
    model = PairwiseForm(functools.partial(RandomForestClassifier, n_estimators=10, random_state=0)).fit(X, scores)
    assert (0, 1) in model.models_                 # fitted with an empty-difference guard, not crashed
    assert model.predict_proba(X).shape == (len(X), 3)


def test_ranking_expansion_keeps_three_consecutive_rows_per_user():
    X, _ = synthetic(n=5)
    expanded = RankingForm._expand(X)
    assert len(expanded) == 15
    assert expanded["strategy_id"].tolist() == [0, 1, 2] * 5
    first = expanded.iloc[:3]
    assert first["own_inner"].tolist() == [X.loc[0, "inner_cf"], X.loc[0, "inner_cb"], X.loc[0, "inner_pop"]]


def test_decision_quality_hand_computed():
    scores = np.array([[0.4, 0.0, 0.1], [0.0, 0.2, 0.2]])
    quality = decision_quality(scores, np.array([0, 2]))
    assert quality["realised_hit_rate"] == pytest.approx(0.3)
    assert quality["oracle_hit_rate"] == pytest.approx(0.3)
    assert quality["regret"] == pytest.approx(0.0)
    assert quality["always_popularity"] == pytest.approx(0.15)


def test_fitted_formulation_can_be_pickled_for_deployment():
    X, scores = synthetic(n=90)
    model = MulticlassForm(functools.partial(RandomForestClassifier, n_estimators=10, random_state=0)).fit(X, scores)
    clone = pickle.loads(pickle.dumps(model))
    assert np.allclose(clone.predict_proba(X), model.predict_proba(X))
