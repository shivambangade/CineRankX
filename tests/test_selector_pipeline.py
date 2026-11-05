"""Tests for the richer features, leak-free labelling, margin cleaning and CV evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.strategy_selector.config import load_selector_config
from src.strategy_selector.holdout_labeler import (clean_labels, label_holdout,
                                                   split_profile_holdout)
from src.strategy_selector.model_selection import (compare_candidates, cross_validate,
                                                   evaluate_holdout)
from src.strategy_selector.profile_features import GENRES, extract_profile_features


@pytest.fixture
def movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "genres": ["Action|Comedy", "Comedy", "Drama", "Action"],
        "vote_count": [1000, 500, 10, 5],
        "text_chunk_clean": ["action comedy", "comedy", "drama", "action"]})


@pytest.fixture
def ratings():
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2],
        "movieId": [1, 2, 3, 3, 4],
        "rating": [5.0, 3.0, 4.0, 2.0, 2.0],
        "timestamp": ["2020-01-01 00:00:00", "2020-01-11 00:00:00", "2020-01-21 00:00:00",
                      "2020-02-01 00:00:00", "2020-02-01 00:00:00"]})


def test_new_features_present_and_legacy_columns_kept(movies, ratings):
    feats = extract_profile_features(ratings, movies).set_index("userId")
    for legacy in ("num_ratings", "avg_rating", "rating_std", "num_genres", "genre_entropy",
                   "top_genre_share", "genre_diversity", "rating_min", "rating_max"):
        assert legacy in feats.columns
    assert {f"affinity_{g}" for g in GENRES} <= set(feats.columns)
    assert feats.loc[1, "num_ratings"] == 3
    assert feats.loc[1, "rating_variance"] == pytest.approx(1.0)
    assert feats.loc[1, "log_num_ratings"] == pytest.approx(np.log1p(3))
    assert feats.loc[1, "affinity_Comedy"] == pytest.approx(2 / 3)
    assert feats.loc[1, "affinity_Drama"] == pytest.approx(1 / 3)
    assert feats.loc[1, "num_genres"] == 3


def test_popularity_recency_and_burst_features(movies, ratings):
    feats = extract_profile_features(ratings, movies).set_index("userId")
    # popularity percentiles are 1.0, .75, .5, .25: only movie 4 is below the 0.5 niche line
    assert feats.loc[2, "niche_fraction"] == pytest.approx(0.5)
    assert feats.loc[1, "niche_fraction"] == 0.0
    assert feats.loc[1, "log_active_span_days"] == pytest.approx(np.log1p(20))
    assert feats.loc[1, "recent_activity_share"] == pytest.approx(1 / 3)
    assert feats.loc[2, "recent_activity_share"] == 0.0     # one sitting, no sustained span


def test_features_use_only_the_ratings_given(movies, ratings):
    profile = ratings[ratings["movieId"] != 3]
    feats = extract_profile_features(profile, movies).set_index("userId")
    assert feats.loc[1, "affinity_Drama"] == 0.0
    assert feats.loc[1, "num_ratings"] == 2


def test_profile_holdout_split_is_disjoint_and_per_user():
    rows = pd.DataFrame({"userId": np.repeat([1, 2], 10), "movieId": list(range(10)) * 2,
                         "rating": 4.0})
    profile, holdout = split_profile_holdout(rows, 0.2, seed=1)
    assert len(profile) + len(holdout) == len(rows)
    assert holdout.groupby("userId").size().tolist() == [2, 2]
    merged = profile.merge(holdout, on=["userId", "movieId"])
    assert merged.empty


def test_holdout_labels_never_use_holdout_ratings(movies):
    profile = pd.DataFrame({"userId": [1, 1, 2, 2], "movieId": [1, 2, 1, 2], "rating": [5, 4, 5, 4]})
    holdout = pd.DataFrame({"userId": [1, 2], "movieId": [3, 4], "rating": [5, 5]})
    labels = label_holdout(profile, holdout, movies).set_index("userId")
    assert labels.loc[1, "n_holdout"] == 1
    assert {"cb_score", "cf_score", "pop_score"} <= set(labels.columns)
    # Movies 3 and 4 appear only in holdout, so the profile-built popularity list cannot contain them.
    assert labels.loc[1, "pop_score"] == 0.0 and labels.loc[2, "pop_score"] == 0.0


def test_clean_labels_drops_zero_and_ambiguous_users():
    labels = pd.DataFrame({
        "userId": [1, 2, 3, 4], "best_strategy": ["popularity"] * 4,
        "cb_score": [0.0, 0.50, 0.10, 0.00], "cf_score": [0.0, 0.49, 0.40, 0.30],
        "pop_score": [0.0, 0.10, 0.00, 0.00]})
    kept, report = clean_labels(labels, margin=0.05)
    assert kept["userId"].tolist() == [3, 4]
    assert kept["best_strategy"].tolist() == ["collaborative", "collaborative"]
    assert report["dropped_all_zero"] == 1 and report["dropped_ambiguous"] == 1
    loose, _ = clean_labels(labels, margin=0.0)
    assert loose["userId"].tolist() == [2, 3, 4]


def test_margin_comes_from_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("strategy_selector:\n  label_margin: 0.1\n")
    assert load_selector_config(cfg)["label_margin"] == 0.1
    assert load_selector_config(tmp_path / "missing.yaml")["label_margin"] == 0.02
    assert load_selector_config("eval_config.yaml")["cv_folds"] == 5


def _synthetic(n_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = {0: (0, 0), 1: (3, 0), 2: (0, 3)}
    rows, labels = [], []
    for cls, (cx, cy) in centers.items():
        pts = rng.normal(size=(n_per_class, 2)) + (cx, cy)
        rows.append(pts)
        labels += [cls] * n_per_class
    X = pd.DataFrame(np.vstack(rows), columns=["a", "b"])
    return X, np.array(labels), ["c0", "c1", "c2"]


def test_cross_validation_reports_baseline_and_beats_it_on_separable_data():
    X, y, names = _synthetic()
    result = cross_validate("rf_balanced", X, y, names, folds=5)
    assert result["folds"] == 5
    assert result["cv_baseline_accuracy"] == pytest.approx(1 / 3, abs=0.05)
    assert result["cv_macro_f1"] > 0.8 and result["cv_balanced_accuracy"] > 0.8
    assert np.array(result["cv_confusion_matrix"]).sum() == len(y)


def test_cross_validation_on_noise_does_not_beat_chance():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=list("abc"))
    y = rng.integers(0, 3, 150)
    result = cross_validate("rf_balanced", X, y, ["c0", "c1", "c2"], folds=5)
    assert result["cv_balanced_accuracy"] < 0.5


def test_cross_validation_shrinks_folds_for_rare_classes_and_smote_stays_in_fold():
    X, y, names = _synthetic(30)
    y[:3] = 1  # class 1 now has 33 users; make class 2 rare instead
    y = np.where(np.arange(len(y)) >= 84, 2, y)
    rare = np.where(y == 2)[0][3:]
    y[rare] = 0
    result = cross_validate("rf_smote", X, y, names, folds=5)
    assert result["folds"] == 3  # smallest class has 3 users


def test_compare_and_holdout_use_separate_users():
    X, y, names = _synthetic()
    order = np.random.default_rng(2).permutation(len(y))
    X, y = X.iloc[order].reset_index(drop=True), y[order]
    dev, hold = slice(0, 90), slice(90, None)
    table, best, _ = compare_candidates(X.iloc[dev].reset_index(drop=True), y[dev], names, 3,
                                     kinds=["rf_balanced", "rf_smote"], log=lambda *_: None)
    assert best in {"rf_balanced", "rf_smote"} and len(table) == 2
    held = evaluate_holdout(best, X.iloc[dev], y[dev], X.iloc[hold], y[hold], names)
    assert held["n_test"] == 30 and held["balanced_accuracy"] > 0.8
    assert "majority_baseline_accuracy" in held
