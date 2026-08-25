"""Tests for the Adaptive ML Strategy Selection Engine."""

import numpy as np
import pandas as pd
import pytest

from src.strategy_selector.classifier import STRATEGY_CLASSES
from src.strategy_selector import (
    StrategyClassifier,
    extract_profile_features,
    label_users_with_best_strategy,
    select_best_strategy,
)


class TestProfileFeatures:
    @pytest.fixture
    def sample_data(self):
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1, 2, 2, 2, 3],
                "movieId": [1, 2, 3, 4, 1, 2, 5, 1],
                "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 2.0, 5.0],
                "timestamp": ["2020-01-01"] * 8,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5],
                "title": ["A", "B", "C", "D", "E"],
                "genres": ["Action", "Comedy", "Action|Drama", "Drama", "Comedy|Romance"],
            }
        )
        return ratings_df, movies_df

    def test_extract_features_basic(self, sample_data):
        ratings_df, movies_df = sample_data
        features = extract_profile_features(ratings_df, movies_df)

        assert len(features) == 3  # 3 users
        assert "userId" in features.columns
        assert "num_ratings" in features.columns
        assert "avg_rating" in features.columns

    def test_extract_features_user1(self, sample_data):
        ratings_df, movies_df = sample_data
        features = extract_profile_features(ratings_df, movies_df)

        user1 = features[features["userId"] == 1].iloc[0]
        assert user1["num_ratings"] == 4
        assert user1["avg_rating"] == pytest.approx(4.25)

    def test_extract_features_includes_genre_diversity(self, sample_data):
        ratings_df, movies_df = sample_data
        features = extract_profile_features(ratings_df, movies_df)

        assert "num_genres" in features.columns
        assert "genre_diversity" in features.columns
        assert (features["num_genres"] >= 0).all()

    def test_extract_features_empty_input(self):
        empty_ratings = pd.DataFrame()
        empty_movies = pd.DataFrame()

        features = extract_profile_features(empty_ratings, empty_movies)
        assert features.empty

    def test_extract_features_includes_genre_identity_features(self, sample_data):
        ratings_df, movies_df = sample_data
        features = extract_profile_features(ratings_df, movies_df)

        assert "top_genre_share" in features.columns
        assert "genre_entropy" in features.columns
        assert (features["top_genre_share"] >= 0).all() and (features["top_genre_share"] <= 1).all()
        assert (features["genre_entropy"] >= 0).all() and (features["genre_entropy"] <= 1).all()

    def test_top_genre_share_and_entropy_for_focused_user(self):
        # A user who only ever watches one genre -> maximally concentrated:
        # top_genre_share = 1.0, genre_entropy = 0.0.
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 1],
                "movieId": [1, 2, 3],
                "rating": [5.0, 4.0, 5.0],
                "timestamp": ["2020-01-01"] * 3,
            }
        )
        movies_df = pd.DataFrame(
            {"movieId": [1, 2, 3], "title": ["A", "B", "C"], "genres": ["Action", "Action", "Action"]}
        )

        features = extract_profile_features(ratings_df, movies_df)
        user1 = features[features["userId"] == 1].iloc[0]

        assert user1["top_genre_share"] == pytest.approx(1.0)
        assert user1["genre_entropy"] == pytest.approx(0.0)

    def test_genre_entropy_for_eclectic_user(self):
        # A user spread evenly across 4 distinct genres -> maximally diffuse:
        # top_genre_share = 0.25, genre_entropy = 1.0 (normalized maximum).
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1],
                "movieId": [1, 2, 3, 4],
                "rating": [4.0, 4.0, 4.0, 4.0],
                "timestamp": ["2020-01-01"] * 4,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "title": ["A", "B", "C", "D"],
                "genres": ["Action", "Comedy", "Drama", "Horror"],
            }
        )

        features = extract_profile_features(ratings_df, movies_df)
        user1 = features[features["userId"] == 1].iloc[0]

        assert user1["top_genre_share"] == pytest.approx(0.25)
        assert user1["genre_entropy"] == pytest.approx(1.0)

    def test_focused_user_has_lower_entropy_than_eclectic_user(self):
        # The comparison that actually matters for the classifier: a focused
        # user must rank below an eclectic user on entropy and above them on
        # top-genre share, regardless of exact values.
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 2, 2],
                "movieId": [1, 2, 3, 4, 5, 6, 7],
                "rating": [5.0] * 7,
                "timestamp": ["2020-01-01"] * 7,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5, 6, 7],
                "title": list("ABCDEFG"),
                "genres": ["Action", "Action", "Action", "Action", "Comedy", "Drama", "Horror"],
            }
        )

        features = extract_profile_features(ratings_df, movies_df)
        focused = features[features["userId"] == 1].iloc[0]
        eclectic = features[features["userId"] == 2].iloc[0]

        assert focused["genre_entropy"] < eclectic["genre_entropy"]
        assert focused["top_genre_share"] > eclectic["top_genre_share"]


class TestSelectBestStrategy:
    def test_clear_winner_content_based(self):
        assert select_best_strategy(cb_score=0.5, cf_score=0.1, pop_score=0.0) == "content_based"

    def test_clear_winner_collaborative(self):
        assert select_best_strategy(cb_score=0.0, cf_score=0.3, pop_score=0.1) == "collaborative"

    def test_all_zero_tie_defaults_to_popularity(self):
        # Most common real-world case: sparse backtest data, no strategy beats
        # blind guessing. Must NOT silently fall through to content_based just
        # because it's listed first in the scores dict.
        assert select_best_strategy(cb_score=0.0, cf_score=0.0, pop_score=0.0) == "popularity"

    def test_content_and_collaborative_tie_excludes_popularity(self):
        # Tie between two non-popularity strategies: popularity isn't among the
        # tied options, so it must not be force-selected.
        result = select_best_strategy(cb_score=0.4, cf_score=0.4, pop_score=0.1)
        assert result in ("content_based", "collaborative")
        assert result != "popularity"

    def test_popularity_can_win_outright(self):
        assert select_best_strategy(cb_score=0.1, cf_score=0.1, pop_score=0.5) == "popularity"


class TestPopularityWeighting:
    def test_prefers_tmdb_popularity_column_when_available(self):
        from src.strategy_selector.strategy_labeler import _BacktestContext

        # Movie 1 has a single 5.0 rating (would win a naive mean-rating top-10)
        # but low TMDB popularity; movie 2 has broad support and high TMDB
        # popularity. 10 decoy movies with mid-range popularity fill out the
        # top-10 so the ranking is actually forced to choose. The TMDB column
        # must be trusted over the noisy mean.
        decoy_ids = list(range(3, 13))
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4],
                "movieId": [1, 2, 2, 2],
                "rating": [5.0, 4.0, 4.2, 4.1],
                "timestamp": ["2020-01-01"] * 4,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2] + decoy_ids,
                "title": ["Obscure", "Blockbuster"] + [f"Decoy{i}" for i in decoy_ids],
                "popularity": [1.0, 500.0] + [50.0] * len(decoy_ids),
            }
        )

        ctx = _BacktestContext(ratings_df, movies_df)

        assert 2 in ctx.top_popular_movies
        assert 1 not in ctx.top_popular_movies

    def test_fallback_ranks_broad_support_above_single_rating(self):
        from src.strategy_selector.strategy_labeler import _BacktestContext

        # No "popularity" column -> falls back to ratings_df. Movie 1 has one
        # perfect rating (naive mean = 5.0, would rank #1 under the old bug);
        # movie 2 has broad strong support (naive mean = 4.2 from 20 raters).
        # 15 decoy movies at a realistic lower baseline (mean=3.0, count=10
        # each, matching typical vote counts) establish a population prior
        # that isn't itself distorted by movie 1's single extreme rating —
        # against that realistic prior, the volume-weighted fallback must
        # rank movie 2 above movie 1, the exact ordering the naive-mean
        # version got backwards.
        decoy_ids = list(range(3, 18))
        decoy_rows = []
        for mid in decoy_ids:
            for u in range(10):
                decoy_rows.append(
                    {"userId": f"decoy_{mid}_{u}", "movieId": mid, "rating": 3.0, "timestamp": "2020-01-01"}
                )

        ratings_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "userId": [1] + list(range(2, 22)),
                        "movieId": [1] + [2] * 20,
                        "rating": [5.0] + [4.2] * 20,
                        "timestamp": ["2020-01-01"] * 21,
                    }
                ),
                pd.DataFrame(decoy_rows),
            ],
            ignore_index=True,
        )
        movies_df = pd.DataFrame(
            {"movieId": [1, 2] + decoy_ids, "title": ["Obscure", "Broadly Liked"] + [f"Decoy{i}" for i in decoy_ids]}
        )

        ctx = _BacktestContext(ratings_df, movies_df)
        stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
        global_mean = ratings_df["rating"].mean()
        min_votes = stats["count"].median()
        weighted = (stats["count"] / (stats["count"] + min_votes)) * stats["mean"] + (
            min_votes / (stats["count"] + min_votes)
        ) * global_mean

        # The regression this guards: under the old naive-mean ranking, movie 1
        # (a single perfect rating) would outrank movie 2 (broad, strong
        # support). The weighted score must now put movie 2 ahead.
        assert weighted[2] > weighted[1]
        assert 2 in ctx.top_popular_movies


class TestCollaborativeCandidateBudget:
    def test_caps_candidates_at_ten_not_twenty_five(self):
        # Regression guard: 5 similar users x top-5 movies each could offer up
        # to 25 candidates -- an unfair advantage over content-based and
        # popularity, which each only get 10. Build a context by hand where
        # each of 5 neighbors proposes 5 DISTINCT movies at a different
        # similarity level, so the top-10-by-similarity cutoff is unambiguous.
        from src.strategy_selector.strategy_labeler import _BacktestContext, _collaborative_hit_rate

        ctx = _BacktestContext.__new__(_BacktestContext)
        ctx.user_similarity = np.array(
            [
                [-1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                [0.9, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]
        )
        ctx.user_id_to_idx = {100: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        ctx.user_ids = np.array([100, 1, 2, 3, 4, 5])
        # Highest-similarity neighbors (1: 0.9, 2: 0.8) alone fill the top-10
        # budget; neighbors 3/4/5's movies should be crowded out.
        ctx.user_top_movies = {
            1: {101, 102, 103, 104, 105},
            2: {201, 202, 203, 204, 205},
            3: {301, 302, 303, 304, 305},
            4: {401, 402, 403, 404, 405},
            5: {501, 502, 503, 504, 505},
        }

        # A movie proposed only by the lowest-similarity neighbor must NOT
        # count as a hit once capped to the top-10 budget.
        assert _collaborative_hit_rate(ctx, 100, [501]) == 0.0
        # A movie from the highest-similarity neighbor must still count.
        assert _collaborative_hit_rate(ctx, 100, [101]) == 1.0


class TestContentBasedProfileWeighting:
    def _build_ctx(self, movie_ids, vectors):
        from scipy.sparse import csr_matrix

        from src.strategy_selector.strategy_labeler import _BacktestContext

        ctx = _BacktestContext.__new__(_BacktestContext)
        ctx.tfidf_matrix = csr_matrix(np.array(vectors, dtype=float))
        ctx.tfidf_movie_ids = np.array(movie_ids)
        ctx.movie_id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
        return ctx

    def test_profile_weights_by_rating(self):
        # Two trained movies pointing in orthogonal directions A and B. A
        # rated much higher than B -> profile must lean toward A, not sit
        # halfway between them the way an unweighted average would.
        from src.strategy_selector.strategy_labeler import _build_cb_profile_vector

        ctx = self._build_ctx([1, 2], [[1.0, 0.0], [0.0, 1.0]])
        train_ratings = pd.DataFrame({"movieId": [1, 2], "rating": [5.0, 1.0]})

        profile = _build_cb_profile_vector(ctx, train_ratings)

        assert profile[0, 0] > profile[0, 1]

    def test_profile_ignores_movies_beyond_the_cap(self):
        # Regression guard for profile dilution: once a user's trained
        # movies exceed the cap, the profile must be built from the SAME
        # top-N-by-rating movies regardless of how many extra low-rated
        # movies exist beyond that -- it must not keep diluting as a user's
        # history grows arbitrarily large.
        from src.strategy_selector.strategy_labeler import _CB_PROFILE_MAX_MOVIES, _build_cb_profile_vector

        n = _CB_PROFILE_MAX_MOVIES
        # n movies fill the cap exactly; ratings 1..n so ranking is unambiguous.
        movie_ids = list(range(1, n + 1))
        vectors = [[1.0, 0.0]] * n
        ctx = self._build_ctx(movie_ids, vectors)

        train_small = pd.DataFrame({"movieId": movie_ids, "rating": list(range(1, n + 1))})
        profile_small = _build_cb_profile_vector(ctx, train_small)

        # Add 50 extra movies rated below every movie already in the cap --
        # they must never displace the existing top-N or change the profile.
        extra_ids = list(range(n + 1, n + 51))
        extra_ratings = pd.DataFrame({"movieId": extra_ids, "rating": [0.1] * 50})
        train_large = pd.concat([train_small, extra_ratings], ignore_index=True)
        profile_large = _build_cb_profile_vector(ctx, train_large)

        assert np.allclose(profile_small, profile_large)

    def test_no_trained_movie_in_catalog_returns_none(self):
        from src.strategy_selector.strategy_labeler import _build_cb_profile_vector

        ctx = self._build_ctx([1, 2], [[1.0, 0.0], [0.0, 1.0]])
        train_ratings = pd.DataFrame({"movieId": [999], "rating": [5.0]})

        assert _build_cb_profile_vector(ctx, train_ratings) is None


class TestStrategyLabeler:
    @pytest.fixture
    def sample_data(self):
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "movieId": [1, 2, 3, 4, 5, 1, 2, 6, 7, 8],
                "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 4.0, 2.0, 3.0, 5.0],
                "timestamp": ["2020-01-01"] * 10,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5, 6, 7, 8],
                "title": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "genres": ["Action"] * 8,
                "text_chunk_clean": ["action"] * 8,
            }
        )
        return ratings_df, movies_df

    def test_label_users_returns_best_strategy(self, sample_data):
        ratings_df, movies_df = sample_data
        labels = label_users_with_best_strategy(ratings_df, movies_df)

        assert len(labels) == 2
        assert "userId" in labels.columns
        assert "best_strategy" in labels.columns
        assert all(s in ["content_based", "collaborative", "popularity"] for s in labels["best_strategy"])

    def test_label_users_includes_scores(self, sample_data):
        ratings_df, movies_df = sample_data
        labels = label_users_with_best_strategy(ratings_df, movies_df)

        assert "cb_score" in labels.columns
        assert "cf_score" in labels.columns
        assert "pop_score" in labels.columns

    def test_label_users_handles_few_ratings(self):
        ratings_df = pd.DataFrame(
            {
                "userId": [1, 2],
                "movieId": [1, 2],
                "rating": [5.0, 4.0],
                "timestamp": ["2020-01-01"] * 2,
            }
        )
        movies_df = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["A", "B"],
                "genres": ["Action"] * 2,
                "text_chunk_clean": ["action"] * 2,
            }
        )

        labels = label_users_with_best_strategy(ratings_df, movies_df)
        assert len(labels) == 2
        assert all(s in ["content_based", "collaborative", "popularity"] for s in labels["best_strategy"])


class TestStrategyClassifier:
    @pytest.fixture
    def sample_features_and_labels(self):
        features_df = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4, 5, 6],
                "num_ratings": [10, 20, 5, 15, 25, 8],
                "avg_rating": [3.5, 4.0, 2.5, 4.5, 3.8, 4.2],
                "rating_std": [1.0, 0.8, 1.5, 0.6, 0.9, 1.1],
                "rating_min": [1.0, 2.0, 1.0, 2.5, 2.0, 2.0],
                "rating_max": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                "num_genres": [5, 8, 3, 6, 10, 4],
                "genre_diversity": [0.5, 0.4, 0.6, 0.4, 0.4, 0.5],
            }
        )
        labels_df = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4, 5, 6],
                "best_strategy": [
                    "content_based",
                    "collaborative",
                    "popularity",
                    "collaborative",
                    "content_based",
                    "popularity",
                ],
            }
        )
        return features_df, labels_df

    def test_classifier_fit(self, sample_features_and_labels):
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()

        metrics = classifier.fit(features_df, labels_df)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_classifier_fit_reports_majority_baseline(self, sample_features_and_labels):
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()

        metrics = classifier.fit(features_df, labels_df)

        assert "majority_baseline_accuracy" in metrics
        assert 0 <= metrics["majority_baseline_accuracy"] <= 1

    def test_classifier_fit_reports_per_class_metrics(self, sample_features_and_labels):
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()

        metrics = classifier.fit(features_df, labels_df)

        assert "per_class" in metrics
        for cls_name in metrics["classes_present"]:
            assert cls_name in metrics["per_class"]
            pc = metrics["per_class"][cls_name]
            assert set(pc.keys()) == {"precision", "recall", "f1", "support"}

    def test_classifier_fit_reports_missing_classes(self, sample_features_and_labels):
        # Fixture has all 3 strategies represented, so nothing should be missing.
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()

        metrics = classifier.fit(features_df, labels_df)

        assert metrics["classes_present"] == sorted(metrics["classes_present"])
        assert set(metrics["classes_present"]) == {"collaborative", "content_based", "popularity"}
        assert metrics["classes_missing"] == []

    def test_classifier_fit_flags_absent_class(self):
        # Only two strategies ever appear in labels -> popularity must be
        # reported as missing/unreachable, not silently ignored.
        features_df = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4, 5, 6],
                "num_ratings": [10, 20, 5, 15, 25, 8],
                "avg_rating": [3.5, 4.0, 2.5, 4.5, 3.8, 4.2],
            }
        )
        labels_df = pd.DataFrame(
            {
                "userId": [1, 2, 3, 4, 5, 6],
                "best_strategy": [
                    "content_based",
                    "collaborative",
                    "content_based",
                    "collaborative",
                    "content_based",
                    "collaborative",
                ],
            }
        )
        classifier = StrategyClassifier()
        metrics = classifier.fit(features_df, labels_df)

        assert metrics["classes_missing"] == ["popularity"]
        assert "popularity" not in metrics["classes_present"]

    def test_classifier_holdout_predictions_match_confusion_matrix_accuracy(self, sample_features_and_labels):
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()

        metrics = classifier.fit(features_df, labels_df)

        assert classifier.holdout_predictions is not None
        assert set(classifier.holdout_predictions.columns) >= {
            "userId",
            "true_strategy",
            "predicted_strategy",
            "confidence",
            "correct",
        }
        holdout_accuracy = classifier.holdout_predictions["correct"].mean()
        assert holdout_accuracy == pytest.approx(metrics["accuracy"])

    def test_classifier_predict(self, sample_features_and_labels):
        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()
        classifier.fit(features_df, labels_df)

        predictions = classifier.predict(features_df)

        assert len(predictions) == len(features_df)
        assert "userId" in predictions.columns
        assert "predicted_strategy" in predictions.columns
        assert "confidence" in predictions.columns
        assert all(s in ["content_based", "collaborative", "popularity"] for s in predictions["predicted_strategy"])

    def test_classifier_requires_fit(self):
        classifier = StrategyClassifier()
        features_df = pd.DataFrame({"userId": [1], "num_ratings": [5]})

        with pytest.raises(ValueError, match="not fitted"):
            classifier.predict(features_df)

    def test_classifier_save_and_load(self, sample_features_and_labels, tmp_path, monkeypatch):
        import src.strategy_selector.classifier as clf_module

        monkeypatch.setattr(clf_module, "_MODEL_PATH", tmp_path / "model.pkl")
        monkeypatch.setattr(clf_module, "_LE_PATH", tmp_path / "le.pkl")

        features_df, labels_df = sample_features_and_labels
        classifier = StrategyClassifier()
        classifier.fit(features_df, labels_df)
        classifier.save()

        classifier2 = StrategyClassifier()
        loaded = classifier2.load()
        assert loaded
        assert classifier2.model is not None
        assert classifier2.label_encoder is not None

        pred1 = classifier.predict(features_df)
        pred2 = classifier2.predict(features_df)
        assert (pred1["predicted_strategy"].values == pred2["predicted_strategy"].values).all()


class TestSMOTEOversampling:
    """SMOTE must only ever touch the training fold, after the split."""

    @pytest.fixture
    def imbalanced_features_and_labels(self):
        """40 users, deliberately imbalanced: 30 collaborative / 8 popularity / 2 content_based."""
        rng = np.random.default_rng(0)
        n = 40
        features_df = pd.DataFrame(
            {
                "userId": range(1, n + 1),
                "num_ratings": rng.integers(5, 300, n),
                "avg_rating": rng.uniform(2.0, 5.0, n),
                "rating_std": rng.uniform(0.3, 1.5, n),
                "num_genres": rng.integers(2, 18, n),
                "genre_diversity": rng.uniform(0.1, 0.9, n),
                "top_genre_share": rng.uniform(0.1, 0.9, n),
                "genre_entropy": rng.uniform(0.1, 1.0, n),
            }
        )
        labels_df = pd.DataFrame(
            {
                "userId": range(1, n + 1),
                "best_strategy": ["collaborative"] * 30 + ["popularity"] * 8 + ["content_based"] * 2,
            }
        )
        return features_df, labels_df

    def test_smote_balances_the_training_fold(self, imbalanced_features_and_labels):
        features_df, labels_df = imbalanced_features_and_labels
        metrics = StrategyClassifier().fit(features_df, labels_df, use_smote=True)

        smote = metrics["smote"]
        assert smote["applied"] is True
        assert len(set(smote["after"].values())) == 1, "training fold not balanced after SMOTE"
        assert smote["rows_after"] > smote["rows_before"]

    def test_smote_does_not_change_the_test_set(self, imbalanced_features_and_labels):
        """The strongest no-leak signal: identical holdout users either way.

        If SMOTE ran before the split, synthetic rows would land in the test
        fold and the held-out user set would differ between the two runs.
        """
        features_df, labels_df = imbalanced_features_and_labels

        baseline = StrategyClassifier()
        baseline.fit(features_df, labels_df, use_smote=False)
        smoted = StrategyClassifier()
        smoted.fit(features_df, labels_df, use_smote=True)

        assert list(baseline.holdout_predictions["userId"]) == list(smoted.holdout_predictions["userId"])
        assert list(baseline.holdout_predictions["true_strategy"]) == list(
            smoted.holdout_predictions["true_strategy"]
        )

    def test_holdout_users_are_all_real_and_never_synthetic(self, imbalanced_features_and_labels):
        """Every evaluated user must be a genuine row from the input frame."""
        features_df, labels_df = imbalanced_features_and_labels
        classifier = StrategyClassifier()
        classifier.fit(features_df, labels_df, use_smote=True)

        real_user_ids = set(features_df["userId"])
        assert set(classifier.holdout_predictions["userId"]) <= real_user_ids

    def test_smote_row_count_matches_train_fold_not_full_dataset(self, imbalanced_features_and_labels):
        """rows_before must equal the TRAIN fold size, not the whole dataset.

        Catches the leak directly: if SMOTE were fed the full frame before the
        split, rows_before would be 40 rather than the 32-row training fold.
        """
        features_df, labels_df = imbalanced_features_and_labels
        metrics = StrategyClassifier().fit(features_df, labels_df, use_smote=True)

        n_total = len(features_df)
        expected_train_rows = n_total - int(round(n_total * 0.2))
        assert metrics["smote"]["rows_before"] == expected_train_rows
        assert metrics["smote"]["rows_before"] < n_total

    def test_majority_baseline_uses_real_distribution_not_resampled(self, imbalanced_features_and_labels):
        """The baseline must describe the real class mix, not the balanced one.

        Reading the majority class off a SMOTE-balanced y_train would make it an
        arbitrary tie-break between equal classes; the baseline would then be
        wrong and the reported lift meaningless.
        """
        features_df, labels_df = imbalanced_features_and_labels

        baseline = StrategyClassifier().fit(features_df, labels_df, use_smote=False)
        smoted = StrategyClassifier().fit(features_df, labels_df, use_smote=True)

        assert smoted["majority_baseline_accuracy"] == pytest.approx(
            baseline["majority_baseline_accuracy"]
        )

    def test_smote_off_by_default(self, imbalanced_features_and_labels):
        features_df, labels_df = imbalanced_features_and_labels
        metrics = StrategyClassifier().fit(features_df, labels_df)
        assert metrics["smote"]["applied"] is False

    def test_smote_skipped_when_minority_too_small_to_interpolate(self):
        """A skip must be reported as a skip, never silently pass as a SMOTE run.

        Exercises the guard directly rather than through fit(): building a frame
        whose *training fold* lands on a single minority row depends on
        stratified-split arithmetic, which is not what this test is about.
        """
        X_train = pd.DataFrame({"num_ratings": [10, 20, 30, 40], "avg_rating": [3.0, 3.5, 4.0, 4.5]})
        y_train = np.array([0, 0, 0, 1])  # one lone minority sample

        _, _, info = StrategyClassifier._resample_training_fold(
            X_train, y_train, ["collaborative", "content_based"]
        )

        assert info["applied"] is False
        assert "skipped_reason" in info

    def test_smote_helper_leaves_data_untouched_when_skipped(self):
        X_train = pd.DataFrame({"num_ratings": [10, 20, 30, 40], "avg_rating": [3.0, 3.5, 4.0, 4.5]})
        y_train = np.array([0, 0, 0, 1])

        X_out, y_out, _ = StrategyClassifier._resample_training_fold(
            X_train, y_train, ["collaborative", "content_based"]
        )

        assert len(X_out) == len(X_train)
        assert list(y_out) == list(y_train)

    def test_k_neighbors_lowered_for_small_minority(self, imbalanced_features_and_labels):
        features_df, labels_df = imbalanced_features_and_labels
        metrics = StrategyClassifier().fit(features_df, labels_df, use_smote=True)
        # content_based has ~2 users total, so k must drop below the default 5.
        assert 1 <= metrics["smote"]["k_neighbors"] <= 5

    def test_smote_model_still_predicts_real_class_names(self, imbalanced_features_and_labels):
        features_df, labels_df = imbalanced_features_and_labels
        classifier = StrategyClassifier()
        classifier.fit(features_df, labels_df, use_smote=True)

        predictions = classifier.predict(features_df)
        assert set(predictions["predicted_strategy"]) <= set(STRATEGY_CLASSES)
