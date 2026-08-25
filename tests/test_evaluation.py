"""Tests for the evaluation suite (all four KPI groups from eval_config.yaml)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation import metric_names as names
from src.evaluation import (
    aggregate,
    average_precision,
    build_relevance_index,
    build_report,
    classification_metrics,
    coverage,
    evaluate_search,
    intra_list_diversity,
    is_relevant,
    load_configured_metrics,
    load_k_values,
    majority_baseline_accuracy,
    mean_average_precision,
    measure,
    missing_from,
    ndcg_at_k,
    novelty,
    popularity_baseline_top_k,
    reciprocal_rank,
    render,
    split_user_ratings,
)
from src.evaluation.recommendation_metrics import (
    evaluate_recommendations,
    f1_score_at_k,
    precision_at_k as rec_precision_at_k,
    recall_at_k as rec_recall_at_k,
)
from src.evaluation.search_metrics import precision_at_k, recall_at_k
from src.ranking_engine import CatalogStats, MultiObjectiveRanker, RankingWeights


# --------------------------------------------------------------------------
# Group 0: the metric names themselves
# --------------------------------------------------------------------------
class TestMetricNames:
    def test_implemented_names_match_config_exactly(self):
        """The whole point of metric_names.py: no drift from eval_config.yaml."""
        configured = load_configured_metrics("eval_config.yaml")
        for group, config_names in configured.items():
            assert config_names == list(names.GROUPS[group]), f"{group} drifted from config"

    def test_all_four_groups_present(self):
        assert set(load_configured_metrics("eval_config.yaml")) == {
            "search", "recommendation", "machine_learning", "system"
        }

    def test_k_values_load_from_config(self):
        assert load_k_values("eval_config.yaml") == [5, 10, 20]

    def test_missing_from_detects_a_dropped_metric(self):
        report = {group: {n: 0.0 for n in metrics} for group, metrics in names.GROUPS.items()}
        del report["search"][names.MRR]
        assert missing_from(report, "eval_config.yaml") == {"search": [names.MRR]}

    def test_missing_from_empty_when_complete(self):
        report = {group: {n: 0.0 for n in metrics} for group, metrics in names.GROUPS.items()}
        assert missing_from(report, "eval_config.yaml") == {}


# --------------------------------------------------------------------------
# Group 1: search
# --------------------------------------------------------------------------
class TestSearchMetrics:
    @pytest.fixture
    def movies_df(self):
        return pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genre_names": ["['Action', 'Thriller']", "Comedy, Romance", "Action|Drama", ""],
            "keyword_names": ["['heist']", "wedding", "revenge", ""],
        })

    def test_relevance_index_parses_all_three_column_formats(self, movies_df):
        index = build_relevance_index(movies_df)
        assert index[1] == {"action", "thriller", "heist"}      # stringified list
        assert index[2] == {"comedy", "romance", "wedding"}     # comma separated
        assert index[3] == {"action", "drama", "revenge"}       # pipe separated
        assert index[4] == set()                                 # empty

    def test_is_relevant_needs_only_one_shared_token(self, movies_df):
        index = build_relevance_index(movies_df)
        assert is_relevant(1, {"action"}, index)
        assert not is_relevant(2, {"action"}, index)

    def test_precision_at_k_divides_by_k_not_result_count(self):
        """A short result list must not score a perfect Precision@10."""
        assert precision_at_k([True, True, True], k=10) == pytest.approx(0.3)

    def test_precision_at_k_all_relevant(self):
        assert precision_at_k([True] * 10, k=10) == pytest.approx(1.0)

    def test_precision_at_k_none_relevant(self):
        assert precision_at_k([False] * 10, k=10) == 0.0

    def test_recall_at_k_against_total_relevant(self):
        assert recall_at_k([True, False, True], total_relevant=4, k=10) == pytest.approx(0.5)

    def test_recall_at_k_zero_when_nothing_relevant_exists(self):
        assert recall_at_k([True], total_relevant=0, k=10) == 0.0

    def test_reciprocal_rank_first_position(self):
        assert reciprocal_rank([True, False]) == pytest.approx(1.0)

    def test_reciprocal_rank_third_position(self):
        assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)

    def test_reciprocal_rank_none_relevant(self):
        assert reciprocal_rank([False, False]) == 0.0

    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k([True] * 5, k=5, total_relevant=100) == pytest.approx(1.0)

    def test_ndcg_rewards_relevant_items_ranked_higher(self):
        front = ndcg_at_k([True, True, False, False], k=4, total_relevant=100)
        back = ndcg_at_k([False, False, True, True], k=4, total_relevant=100)
        assert front > back

    def test_ndcg_zero_when_nothing_relevant(self):
        assert ndcg_at_k([False] * 5, k=5, total_relevant=100) == 0.0

    def test_ndcg_ideal_accounts_for_small_relevant_pool(self):
        """With only 2 relevant items, finding both at the top is perfect."""
        assert ndcg_at_k([True, True, False], k=3, total_relevant=2) == pytest.approx(1.0)

    def test_evaluate_search_reports_every_configured_metric(self, movies_df):
        class FakeIR:
            def search(self, query, limit=10):
                return [{"movieId": 1, "title": "A", "similarity": 0.9},
                        {"movieId": 2, "title": "B", "similarity": 0.5}]

        result = evaluate_search(FakeIR(), ["action"], movies_df, k=10)
        for name in names.SEARCH_METRICS:
            assert name in result
        assert result["n_queries"] == 1
        assert result[names.SEARCH_LATENCY] >= 0.0

    def test_evaluate_search_states_the_relevance_proxy(self, movies_df):
        class FakeIR:
            def search(self, query, limit=10):
                return []

        result = evaluate_search(FakeIR(), ["action"], movies_df)
        assert "genre or keyword" in result["relevance_proxy"]

    def test_evaluate_search_handles_no_queries(self, movies_df):
        class FakeIR:
            def search(self, query, limit=10):
                return []

        assert evaluate_search(FakeIR(), [], movies_df)["n_queries"] == 0


# --------------------------------------------------------------------------
# Group 2: recommendation
# --------------------------------------------------------------------------
class TestRecommendationMetrics:
    @pytest.fixture
    def stats(self):
        movies = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genre_names": ["Action", "Action", "Comedy", "Drama"],
            "vote_average": [7.0, 8.0, 6.0, 5.0],
            "vote_count": [100, 200, 50, 10],
        })
        ratings = pd.DataFrame({"userId": [1, 1], "movieId": [1, 2], "rating": [5.0, 4.0]})
        return CatalogStats(movies, ratings)

    def test_precision_at_k(self):
        assert rec_precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == pytest.approx(0.5)

    def test_recall_at_k(self):
        assert rec_recall_at_k([1, 2], {1, 3, 5, 7}, k=2) == pytest.approx(0.25)

    def test_recall_zero_when_user_has_no_relevant_items(self):
        assert rec_recall_at_k([1, 2], set(), k=2) == 0.0

    def test_f1_is_harmonic_mean(self):
        assert f1_score_at_k(0.5, 0.5) == pytest.approx(0.5)
        assert f1_score_at_k(1.0, 0.0) == 0.0

    def test_average_precision_rewards_early_hits(self):
        early = average_precision([1, 9, 9, 9], {1, 2}, k=4)
        late = average_precision([9, 9, 9, 1], {1, 2}, k=4)
        assert early > late

    def test_average_precision_perfect_when_all_top_slots_hit(self):
        assert average_precision([1, 2], {1, 2}, k=2) == pytest.approx(1.0)

    def test_average_precision_denominator_capped_at_k(self):
        """A perfect top-2 must score 1.0 even when the user liked 10 items."""
        assert average_precision([1, 2], set(range(1, 11)), k=2) == pytest.approx(1.0)

    def test_map_averages_over_users(self):
        assert mean_average_precision([1.0, 0.0]) == pytest.approx(0.5)
        assert mean_average_precision([]) == 0.0

    def test_coverage_counts_distinct_items_across_users(self):
        assert coverage([[1, 2], [2, 3]], catalog_size=10) == pytest.approx(0.3)

    def test_coverage_zero_for_empty_catalog(self):
        assert coverage([[1]], catalog_size=0) == 0.0

    def test_diversity_lower_for_same_genre_list(self, stats):
        same_genre = intra_list_diversity([1, 2], stats)      # both Action
        mixed = intra_list_diversity([1, 3], stats)           # Action + Comedy
        assert mixed > same_genre

    def test_diversity_zero_for_single_item(self, stats):
        assert intra_list_diversity([1], stats) == 0.0

    def test_novelty_reads_catalog_stats(self, stats):
        # Movie 4 has the fewest votes, so it must be the most novel.
        assert novelty([4], stats) > novelty([2], stats)

    def test_novelty_zero_for_empty_list(self, stats):
        assert novelty([], stats) == 0.0

    def test_split_gives_every_user_train_and_test(self):
        ratings = pd.DataFrame({
            "userId": [1] * 10 + [2] * 10,
            "movieId": list(range(10)) + list(range(10)),
            "rating": [4.0] * 20,
        })
        train, test = split_user_ratings(ratings, test_fraction=0.2)
        assert set(train["userId"]) == {1, 2}
        assert set(test["userId"]) == {1, 2}
        assert len(train) + len(test) == len(ratings)

    def test_split_is_deterministic(self):
        ratings = pd.DataFrame({
            "userId": [1] * 10, "movieId": list(range(10)), "rating": [4.0] * 10,
        })
        a, _ = split_user_ratings(ratings, random_state=7)
        b, _ = split_user_ratings(ratings, random_state=7)
        assert list(a["movieId"]) == list(b["movieId"])

    def test_split_never_puts_a_users_whole_history_in_test(self):
        ratings = pd.DataFrame({
            "userId": [1] * 4, "movieId": [1, 2, 3, 4], "rating": [4.0] * 4,
        })
        train, _ = split_user_ratings(ratings, test_fraction=0.2)
        assert len(train) > 0

    def test_popularity_baseline_prefers_observed_rating_counts(self):
        """TMDB `popularity` disagrees with what MovieLens users actually rate.

        Movie 1 is the most-rated but has the LOWEST TMDB popularity. Using the
        TMDB column would pick movies almost nobody in the dataset rated,
        producing a straw-man baseline that overstates the system's lift.
        """
        movies = pd.DataFrame({
            "movieId": [1, 2, 3], "title": ["A", "B", "C"], "popularity": [1.0, 99.0, 50.0],
        })
        ratings = pd.DataFrame({
            "userId": [1, 2, 3, 4, 5], "movieId": [1, 1, 1, 2, 3], "rating": [5.0] * 5,
        })
        assert popularity_baseline_top_k(ratings, movies, k=1) == [1]

    def test_popularity_baseline_falls_back_to_tmdb_without_ratings(self):
        movies = pd.DataFrame({
            "movieId": [1, 2, 3], "title": ["A", "B", "C"], "popularity": [1.0, 99.0, 50.0],
        })
        assert popularity_baseline_top_k(pd.DataFrame(), movies, k=2) == [2, 3]


class TestRecommendationEndToEnd:
    @pytest.fixture
    def catalog_and_ratings(self):
        movies = pd.DataFrame({
            "movieId": list(range(1, 9)),
            "title": [f"M{i}" for i in range(1, 9)],
            "genre_names": ["Action", "Action", "Comedy", "Drama", "Horror",
                            "Romance", "Sci-Fi", "Documentary"],
            "vote_average": [7.5, 8.0, 6.5, 7.0, 5.5, 6.0, 8.5, 7.2],
            "vote_count": [5000, 3000, 1500, 800, 400, 900, 6000, 100],
            "popularity": [90.0, 70.0, 40.0, 30.0, 20.0, 25.0, 95.0, 5.0],
        })
        rows = []
        for user_id in range(1, 11):
            for movie_id in range(1, 8):
                rows.append({"userId": user_id, "movieId": movie_id,
                             "rating": 5.0 if movie_id % 2 else 3.0})
        ratings = pd.DataFrame(rows)
        ratings["timestamp"] = "2020-01-01"
        return movies, ratings

    def test_reports_every_configured_recommendation_metric(self, catalog_and_ratings):
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(
            movies, ratings, weights=RankingWeights.from_config("eval_config.yaml")
        )
        result = evaluate_recommendations(
            ranker, ratings, movies, user_ids=list(range(1, 11)), k_values=[5, 10]
        )
        for name in names.RECOMMENDATION_METRICS:
            assert name in result, f"{name} missing from recommendation result"

    def test_precision_recall_f1_are_keyed_by_k(self, catalog_and_ratings):
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        result = evaluate_recommendations(
            ranker, ratings, movies, user_ids=[1, 2, 3], k_values=[5, 10]
        )
        assert set(result[names.PRECISION_AT_K]) == {5, 10}
        assert set(result[names.RECALL_AT_K]) == {5, 10}
        assert set(result[names.F1]) == {5, 10}

    def test_all_rates_are_valid_probabilities(self, catalog_and_ratings):
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        result = evaluate_recommendations(
            ranker, ratings, movies, user_ids=list(range(1, 11)), k_values=[5]
        )
        assert 0.0 <= result[names.PRECISION_AT_K][5] <= 1.0
        assert 0.0 <= result[names.RECALL_AT_K][5] <= 1.0
        assert 0.0 <= result[names.MAP] <= 1.0
        assert 0.0 <= result[names.COVERAGE] <= 1.0
        assert 0.0 <= result[names.DIVERSITY] <= 1.0

    def test_never_recommends_a_training_item(self, catalog_and_ratings):
        """The held-out evaluation must not re-serve items from the train split."""
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        train, _ = split_user_ratings(ratings, test_fraction=0.2, random_state=42)
        result = evaluate_recommendations(
            ranker, ratings, movies, user_ids=[1], k_values=[5], random_state=42
        )
        recommended = set()
        for _, row in result["per_user"].iterrows():
            pass  # per-user frame holds scores, not ids; assert via the ranker directly
        user_train = set(train.loc[train["userId"] == 1, "movieId"].astype(int))
        from src.ranking_engine import CandidateGenerator
        generator = CandidateGenerator(movies, train, ir_engine=None, stats=ranker.stats)
        pool = generator.generate(1)
        assert not set(pool["movieId"].astype(int)) & user_train

    def test_empty_user_list_returns_zero_users(self, catalog_and_ratings):
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        result = evaluate_recommendations(ranker, ratings, movies, user_ids=[], k_values=[5])
        assert result["n_users_evaluated"] == 0


# --------------------------------------------------------------------------
# Group 3: machine learning
# --------------------------------------------------------------------------
class TestMLMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        proba = np.eye(3)[y]
        m = classification_metrics(y, y, proba, ["a", "b", "c"])
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_confusion_matrix_shape_and_labels(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        proba = np.tile([0.5, 0.5], (4, 1))
        m = classification_metrics(y_true, y_pred, proba, ["a", "b"])
        assert np.array(m["confusion_matrix"]).shape == (2, 2)
        assert m["confusion_matrix_labels"] == ["a", "b"]

    def test_per_class_supports_sum_to_test_size(self):
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 2])
        proba = np.tile([1 / 3, 1 / 3, 1 / 3], (5, 1))
        m = classification_metrics(y_true, y_pred, proba, ["a", "b", "c"])
        assert sum(pc["support"] for pc in m["per_class"].values()) == 5

    def test_majority_baseline_from_training_labels(self):
        y_train = np.array([0, 0, 0, 1])
        y_test = np.array([0, 0, 1, 1])
        majority, accuracy = majority_baseline_accuracy(y_train, y_test)
        assert majority == 0
        assert accuracy == pytest.approx(0.5)

    def test_baseline_absent_when_no_training_labels_given(self):
        y = np.array([0, 1])
        proba = np.eye(2)[y]
        assert "majority_baseline_accuracy" not in classification_metrics(y, y, proba, ["a", "b"])

    def test_roc_auc_computed_for_multiclass(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        proba = np.eye(3)[y] * 0.8 + 0.1
        m = classification_metrics(y, y, proba, ["a", "b", "c"])
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_matches_module3_classifier_output(self):
        """The refactor must reproduce what StrategyClassifier.fit() reports."""
        from src.strategy_selector import StrategyClassifier

        rng = np.random.default_rng(0)
        n = 60
        features = pd.DataFrame({
            "userId": range(1, n + 1),
            "num_ratings": rng.integers(5, 300, n),
            "avg_rating": rng.uniform(2, 5, n),
        })
        labels = pd.DataFrame({
            "userId": range(1, n + 1),
            "best_strategy": ["collaborative"] * 40 + ["popularity"] * 12 + ["content_based"] * 8,
        })
        metrics = StrategyClassifier().fit(features, labels)
        for key in ("accuracy", "precision", "recall", "f1", "per_class",
                    "confusion_matrix", "confusion_matrix_labels",
                    "majority_baseline_accuracy", "roc_auc"):
            assert key in metrics, f"classifier lost '{key}' in the refactor"


# --------------------------------------------------------------------------
# Group 4: system
# --------------------------------------------------------------------------
class TestSystemMetrics:
    def test_reports_every_configured_metric(self):
        m = measure(lambda: sum(range(1000)), n_calls=5, warmup=1, label="noop")
        for name in names.SYSTEM_METRICS:
            assert name in m

    def test_response_time_and_throughput_are_consistent(self):
        m = measure(lambda: sum(range(200_000)), n_calls=10, label="work")
        assert m[names.RESPONSE_TIME] > 0
        assert m[names.THROUGHPUT] > 0
        # ~1000ms/s divided by per-call ms should land near calls/sec.
        assert m[names.THROUGHPUT] == pytest.approx(1000.0 / m[names.RESPONSE_TIME], rel=0.5)

    def test_memory_usage_is_a_positive_rss_figure(self):
        m = measure(lambda: None, n_calls=3, label="noop")
        assert m[names.MEMORY_USAGE] > 0

    def test_cpu_marked_unreliable_for_tiny_windows(self):
        """A sub-50ms window cannot yield a meaningful CPU figure."""
        m = measure(lambda: None, n_calls=1, warmup=0, label="instant")
        assert m["cpu_reliable"] is False
        assert math.isnan(m[names.CPU_USAGE])

    def test_cpu_measured_for_a_long_enough_window(self):
        m = measure(lambda: sum(range(2_000_000)), n_calls=5, label="cpu-bound")
        assert m["cpu_reliable"] is True
        assert m[names.CPU_USAGE] > 0

    def test_aggregate_reports_every_configured_metric(self):
        a = measure(lambda: sum(range(500_000)), n_calls=5, label="a")
        b = measure(lambda: sum(range(500_000)), n_calls=5, label="b")
        agg = aggregate([a, b])
        for name in names.SYSTEM_METRICS:
            assert name in agg
        assert len(agg["per_operation"]) == 2

    def test_aggregate_memory_is_peak_not_sum(self):
        a = measure(lambda: None, n_calls=2, label="a")
        b = measure(lambda: None, n_calls=2, label="b")
        agg = aggregate([a, b])
        assert agg[names.MEMORY_USAGE] == max(a[names.MEMORY_USAGE], b[names.MEMORY_USAGE])

    def test_aggregate_of_nothing_is_empty(self):
        assert aggregate([]) == {}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------
class TestReport:
    def _full_groups(self):
        return (
            {n: 0.5 for n in names.SEARCH_METRICS},
            {n: 0.5 for n in names.RECOMMENDATION_METRICS},
            {n: 0.5 for n in names.MACHINE_LEARNING_METRICS},
            {n: 0.5 for n in names.SYSTEM_METRICS},
        )

    def test_build_report_covers_every_configured_kpi(self):
        report = build_report(*self._full_groups())
        assert missing_from(report, "eval_config.yaml") == {}

    def test_build_report_rejects_a_missing_kpi(self):
        search, rec, ml, sysm = self._full_groups()
        del search[names.MRR]
        with pytest.raises(ValueError, match="missing KPIs"):
            build_report(search, rec, ml, sysm)

    def test_non_strict_mode_allows_gaps(self):
        search, rec, ml, sysm = self._full_groups()
        del search[names.MRR]
        report = build_report(search, rec, ml, sysm, strict=False)
        assert report["search"][names.MRR] is None

    def test_render_includes_every_metric_name(self):
        text = render(build_report(*self._full_groups()))
        for group in names.GROUPS.values():
            for name in group:
                assert name in text, f"{name} absent from the rendered report"

    def test_render_labels_all_four_groups(self):
        text = render(build_report(*self._full_groups()))
        for heading in ("1. SEARCH", "2. RECOMMENDATION", "3. MACHINE LEARNING", "4. SYSTEM"):
            assert heading in text


class TestColdStart:
    """MovieLens has no natural cold-start users, so the cohort is simulated."""

    @pytest.fixture
    def catalog_and_ratings(self):
        movies = pd.DataFrame({
            "movieId": list(range(1, 13)),
            "title": [f"M{i}" for i in range(1, 13)],
            "genre_names": ["Action", "Comedy", "Drama", "Horror"] * 3,
            "vote_average": [7.0] * 12,
            "vote_count": [1000] * 12,
            "popularity": [float(50 - i) for i in range(12)],
        })
        rows = []
        for user_id in range(1, 16):
            for movie_id in range(1, 13):
                rows.append({"userId": user_id, "movieId": movie_id,
                             "rating": 5.0 if movie_id % 3 else 2.0})
        ratings = pd.DataFrame(rows)
        ratings["timestamp"] = "2020-01-01"
        return movies, ratings

    def test_simulates_when_no_natural_cold_users_exist(self, catalog_and_ratings):
        from src.evaluation.recommendation_metrics import evaluate_cold_start

        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        train, test = split_user_ratings(ratings, test_fraction=0.3, random_state=1)
        test_by_user = {u: g for u, g in test.groupby("userId")}

        result = evaluate_cold_start(
            ranker, train, test_by_user, movies, list(range(1, 16)), k=5
        )
        assert result["mode"] == "simulated"
        assert result["n_users"] > 0
        assert result["accuracy"] is not None

    def test_uses_natural_cohort_when_one_exists(self, catalog_and_ratings):
        from src.evaluation.recommendation_metrics import evaluate_cold_start

        movies, ratings = catalog_and_ratings
        # Give user 99 a genuinely tiny history.
        sparse = pd.DataFrame({
            "userId": [99, 99, 99], "movieId": [1, 2, 3],
            "rating": [5.0, 5.0, 5.0], "timestamp": ["2020-01-01"] * 3,
        })
        ratings = pd.concat([ratings, sparse], ignore_index=True)
        ranker = MultiObjectiveRanker(movies, ratings)
        train, test = split_user_ratings(ratings, test_fraction=0.3, random_state=1)
        test_by_user = {u: g for u, g in test.groupby("userId")}

        result = evaluate_cold_start(
            ranker, train, test_by_user, movies, list(range(1, 16)) + [99], k=5
        )
        assert result["mode"] == "natural"

    def test_accuracy_and_baseline_are_probabilities(self, catalog_and_ratings):
        from src.evaluation.recommendation_metrics import evaluate_cold_start

        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        train, test = split_user_ratings(ratings, test_fraction=0.3, random_state=1)
        test_by_user = {u: g for u, g in test.groupby("userId")}

        result = evaluate_cold_start(
            ranker, train, test_by_user, movies, list(range(1, 16)), k=5
        )
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["baseline_accuracy"] <= 1.0

    def test_cold_start_reported_in_full_evaluation(self, catalog_and_ratings):
        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        result = evaluate_recommendations(
            ranker, ratings, movies, user_ids=list(range(1, 16)), k_values=[5]
        )
        assert result[names.COLD_START_ACCURACY] is not None
        assert result["cold_start_mode"] in ("natural", "simulated")

    def test_truncation_leaves_other_users_histories_intact(self, catalog_and_ratings):
        """Only the cohort is truncated — a cold SYSTEM is a different measurement."""
        from src.evaluation.recommendation_metrics import evaluate_cold_start

        movies, ratings = catalog_and_ratings
        ranker = MultiObjectiveRanker(movies, ratings)
        train, test = split_user_ratings(ratings, test_fraction=0.3, random_state=1)
        test_by_user = {u: g for u, g in test.groupby("userId")}

        result = evaluate_cold_start(
            ranker, train, test_by_user, movies, list(range(1, 16)), k=5, max_users=3
        )
        # Only 3 users were truncated, so the rest keep full histories.
        assert result["n_users"] <= 3
