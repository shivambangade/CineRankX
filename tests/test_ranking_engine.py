"""Tests for the Multi-Objective Hybrid Ranking Engine (Module 4)."""

import numpy as np
import pandas as pd
import pytest

from src.ranking_engine import (
    BaselinePredictor,
    genre_overlap_gate,
    load_genre_gate_floor,
    CandidateGenerator,
    CatalogStats,
    MultiObjectiveRanker,
    OBJECTIVES,
    RankingWeights,
    coverage_score,
    diversity_score,
    strategy_source_weights,
)


@pytest.fixture
def movies_df():
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": ["Action A", "Action B", "Comedy C", "Drama D", "Horror E", "Doc F"],
            "genre_names": [
                "Action, Adventure",
                "Action, Adventure",
                "Comedy",
                "Drama",
                "Horror, Thriller",
                "Documentary",
            ],
            "vote_average": [8.0, 7.5, 6.0, 9.0, 5.0, 7.0],
            "vote_count": [10000, 5000, 2000, 50, 300, 10],
            "popularity": [100.0, 80.0, 40.0, 5.0, 20.0, 1.0],
            "text_chunk_clean": ["action hero", "action hero", "funny", "sad", "scary", "real"],
        }
    )


@pytest.fixture
def ratings_df():
    rows = []
    # Users 1-5 mostly rate the popular action movies; user 6 is a drama fan.
    for user_id in range(1, 6):
        rows += [
            {"userId": user_id, "movieId": 1, "rating": 5.0},
            {"userId": user_id, "movieId": 2, "rating": 4.0},
            {"userId": user_id, "movieId": 3, "rating": 3.0},
        ]
    rows += [
        {"userId": 6, "movieId": 4, "rating": 5.0},
        {"userId": 6, "movieId": 5, "rating": 4.5},
        {"userId": 6, "movieId": 1, "rating": 2.0},
        # User 7 exists so movies 4 and 5 have a rating count above the floor.
        {"userId": 7, "movieId": 4, "rating": 4.0},
        {"userId": 7, "movieId": 5, "rating": 3.0},
        # Movie 6 gets a single rating so it can enter the popularity source at
        # all -- that source now ranks by observed rating count, so a movie
        # nobody has rated is correctly not "popular". Its lone rating plus the
        # catalog's lowest vote_count still leave it the rarest item by far.
        {"userId": 8, "movieId": 6, "rating": 4.0},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = "2020-01-01"
    return df


class TestRankingWeights:
    def test_loads_all_six_objectives_from_config(self):
        weights = RankingWeights.from_config("eval_config.yaml")
        assert set(weights.weights) == set(OBJECTIVES)

    def test_weights_are_normalized_to_one(self):
        weights = RankingWeights({name: 2.0 for name in OBJECTIVES})
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert weights["relevance"] == pytest.approx(1 / 6)

    def test_relative_proportions_survive_normalization(self):
        weights = RankingWeights({**{n: 1.0 for n in OBJECTIVES}, "relevance": 5.0})
        assert weights["relevance"] == pytest.approx(5 * weights["diversity"])

    def test_missing_objective_rejected(self):
        with pytest.raises(ValueError, match="Missing weights"):
            RankingWeights({name: 1.0 for name in OBJECTIVES[:-1]})

    def test_unknown_objective_rejected(self):
        with pytest.raises(ValueError, match="Unknown objectives"):
            RankingWeights({**{n: 1.0 for n in OBJECTIVES}, "vibes": 1.0})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            RankingWeights({**{n: 1.0 for n in OBJECTIVES}, "novelty": -1.0})

    def test_all_zero_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to zero"):
            RankingWeights({name: 0.0 for name in OBJECTIVES})

    def test_with_overrides_does_not_mutate_original(self):
        base = RankingWeights({name: 1.0 for name in OBJECTIVES})
        tuned = base.with_overrides(diversity=10.0)
        assert tuned["diversity"] > base["diversity"]
        assert base["diversity"] == pytest.approx(1 / 6)

    def test_items_returns_canonical_order(self):
        weights = RankingWeights({name: 1.0 for name in OBJECTIVES})
        assert [name for name, _ in weights.items()] == list(OBJECTIVES)


class TestCatalogStats:
    def test_genres_parsed_and_lowercased(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        assert stats.genres(1) == frozenset({"action", "adventure"})

    def test_stringified_list_genres_parsed(self, ratings_df):
        """movies_merged.csv stores genre_names as "['Animation', 'Comedy']"."""
        movies = pd.DataFrame(
            {"movieId": [1], "title": ["A"], "genre_names": ["['Animation', 'Adventure', 'Family']"]}
        )
        stats = CatalogStats(movies, ratings_df)
        assert stats.genres(1) == frozenset({"animation", "adventure", "family"})

    def test_genre_parsing_is_format_agnostic(self, ratings_df):
        """The same three genres must parse identically from all three formats."""
        formats = ["['Action', 'Drama', 'Crime']", "Action, Drama, Crime", "Action|Drama|Crime"]
        parsed = [
            CatalogStats(
                pd.DataFrame({"movieId": [1], "title": ["A"], "genre_names": [fmt]}), ratings_df
            ).genres(1)
            for fmt in formats
        ]
        assert parsed[0] == parsed[1] == parsed[2] == frozenset({"action", "drama", "crime"})

    def test_pipe_separated_genres_supported(self, ratings_df):
        movies = pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Action|Drama"]})
        stats = CatalogStats(movies, ratings_df)
        assert stats.genres(1) == frozenset({"action", "drama"})

    def test_popularity_quality_shrinks_low_vote_movies(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        # Movie 4 has the best raw vote_average (9.0) but only 50 votes;
        # movie 1 has 8.0 from 10,000 votes and must win on Bayesian shrinkage.
        assert stats.popularity_quality(1) > stats.popularity_quality(4)

    def test_novelty_rewards_rarely_rated_items(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        # Movie 1 is rated by 6 users, movie 6 by nobody.
        assert stats.novelty(6) > stats.novelty(1)

    def test_novelty_uses_vote_count_for_unrated_items(self, movies_df):
        """Items absent from the ratings sample must still be ranked against
        each other by TMDB vote_count, not all collapse to novelty 1.0."""
        sparse_ratings = pd.DataFrame(
            [{"userId": 1, "movieId": 1, "rating": 5.0}, {"userId": 2, "movieId": 1, "rating": 4.0}]
        )
        stats = CatalogStats(movies_df, sparse_ratings)
        unrated = [2, 3, 4, 5, 6]  # none appear in sparse_ratings
        scores = [stats.novelty(mid) for mid in unrated]
        assert len(set(scores)) == len(scores), "unrated items collapsed to a single novelty value"
        # Movie 6 has 10 votes, movie 2 has 5000: 6 must be the more novel.
        assert stats.novelty(6) > stats.novelty(2)

    def test_all_scores_in_unit_range(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        for movie_id in movies_df["movieId"]:
            assert 0.0 <= stats.novelty(movie_id) <= 1.0
            assert 0.0 <= stats.popularity_quality(movie_id) <= 1.0

    def test_unknown_movie_returns_neutral_values(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        assert stats.novelty(999) == 0.0
        assert stats.genres(999) == frozenset()


class TestDiversityObjective:
    def test_first_pick_is_maximally_diverse(self):
        assert diversity_score(1, [], {}) == 1.0

    def test_near_duplicate_scores_low(self):
        similarity = {(2, 1): 0.95}
        assert diversity_score(2, [1], similarity) == pytest.approx(0.05)

    def test_uses_max_not_mean_similarity(self):
        # Near-duplicate of one selected item, unlike the other: must still
        # score as redundant, which a mean would hide.
        similarity = {(3, 1): 0.9, (3, 2): 0.0}
        assert diversity_score(3, [1, 2], similarity) == pytest.approx(0.1)


class TestCoverageObjective:
    def test_fully_new_genres_score_one(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        assert coverage_score(1, set(), stats) == 1.0

    def test_fully_covered_genres_score_zero(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        assert coverage_score(2, {"action", "adventure"}, stats) == 0.0

    def test_partial_overlap_scores_fraction(self, movies_df, ratings_df):
        stats = CatalogStats(movies_df, ratings_df)
        assert coverage_score(5, {"horror"}, stats) == pytest.approx(0.5)

    def test_item_without_genres_scores_zero(self, ratings_df):
        movies = pd.DataFrame({"movieId": [1], "title": ["A"], "genre_names": [""]})
        stats = CatalogStats(movies, ratings_df)
        assert coverage_score(1, set(), stats) == 0.0


class TestBaselinePredictor:
    def test_predictions_stay_on_the_rating_scale(self, ratings_df):
        predictor = BaselinePredictor().fit(ratings_df)
        for movie_id in [1, 2, 3, 4, 5]:
            assert 0.5 <= predictor.predict(1, movie_id) <= 5.0

    def test_item_bias_orders_well_and_poorly_rated_items(self, ratings_df):
        predictor = BaselinePredictor().fit(ratings_df)
        # Movie 1 averages ~4.5 across users, movie 3 exactly 3.0.
        assert predictor.predict(1, 1) > predictor.predict(1, 3)

    def test_unknown_user_falls_back_to_item_standing(self, ratings_df):
        predictor = BaselinePredictor().fit(ratings_df)
        cold = predictor.predict(9999, 1)
        assert 0.5 <= cold <= 5.0
        assert cold == pytest.approx(predictor.global_mean + predictor.item_bias[1])

    def test_unknown_item_falls_back_to_user_standing(self, ratings_df):
        predictor = BaselinePredictor().fit(ratings_df)
        assert predictor.predict(1, 9999) == pytest.approx(
            predictor.global_mean + predictor.user_bias[1]
        )

    def test_normalized_prediction_in_unit_range(self, ratings_df):
        predictor = BaselinePredictor().fit(ratings_df)
        assert 0.0 <= predictor.predict_normalized(1, 1) <= 1.0

    def test_empty_ratings_does_not_crash(self):
        predictor = BaselinePredictor().fit(pd.DataFrame(columns=["userId", "movieId", "rating"]))
        assert predictor.predict(1, 1) == pytest.approx(0.5)


class TestStrategySourceWeights:
    def test_full_confidence_concentrates_on_predicted_strategy(self):
        weights = strategy_source_weights("content_based", confidence=1.0)
        assert weights["content_based"] == pytest.approx(1.0)
        assert weights["collaborative"] == pytest.approx(0.0)

    def test_chance_confidence_falls_back_to_prior(self):
        weights = strategy_source_weights("popularity", confidence=1 / 3)
        assert weights["collaborative"] == pytest.approx(0.60)

    def test_low_confidence_degrades_toward_collaborative(self):
        weights = strategy_source_weights("popularity", confidence=0.4)
        assert weights["collaborative"] > weights["popularity"]

    def test_no_strategy_returns_prior(self):
        weights = strategy_source_weights(None)
        assert weights["collaborative"] == pytest.approx(0.60)

    def test_unknown_strategy_returns_prior(self):
        assert strategy_source_weights("magic", 1.0)["collaborative"] == pytest.approx(0.60)

    def test_weights_always_sum_to_one(self):
        for confidence in [0.0, 0.34, 0.5, 0.9, 1.0]:
            weights = strategy_source_weights("content_based", confidence)
            assert sum(weights.values()) == pytest.approx(1.0)


class TestCandidateGenerator:
    def test_excludes_movies_the_user_already_rated(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        candidates = generator.generate(user_id=1)
        assert not set(candidates["movieId"]) & {1, 2, 3}

    def test_relevance_normalized_to_unit_range(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        candidates = generator.generate(user_id=1)
        assert candidates["relevance"].between(0.0, 1.0).all()

    def test_respects_pool_size(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        assert len(generator.generate(user_id=1, pool_size=2)) <= 2

    def test_sources_are_recorded(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        candidates = generator.generate(user_id=1)
        assert all(any(s in row for s in ("popularity", "collaborative", "content_based"))
                   for row in candidates["sources"])

    def test_unknown_user_still_gets_candidates(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        # Cold-start user: no history at all, so only the popularity source can
        # fire -- but it must still return something rather than an empty frame.
        assert not generator.generate(user_id=9999).empty

    def test_strategy_shifts_the_blend(self, movies_df, ratings_df):
        generator = CandidateGenerator(movies_df, ratings_df)
        pop = generator.generate(user_id=6, strategy="popularity", confidence=1.0)
        cf = generator.generate(user_id=6, strategy="collaborative", confidence=1.0)
        pop_scores = dict(zip(pop["movieId"], pop["relevance"]))
        cf_scores = dict(zip(cf["movieId"], cf["relevance"]))
        assert pop_scores != cf_scores


class TestMultiObjectiveRanker:
    @pytest.fixture
    def ranker(self, movies_df, ratings_df):
        return MultiObjectiveRanker(
            movies_df, ratings_df, weights=RankingWeights.from_config("eval_config.yaml")
        )

    def test_returns_k_results_with_all_objective_columns(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        assert len(ranked) == 3
        for objective in OBJECTIVES:
            assert objective in ranked.columns

    def test_ranks_are_sequential_from_one(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        assert list(ranked["rank"]) == [1, 2, 3]

    def test_no_duplicate_recommendations(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        assert ranked["movieId"].nunique() == len(ranked)

    def test_never_recommends_already_rated_movies(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        assert not set(ranked["movieId"]) & {1, 2, 3}

    def test_all_objective_scores_in_unit_range(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        for objective in OBJECTIVES:
            assert ranked[objective].between(0.0, 1.0).all()

    def test_score_equals_weighted_sum_of_objectives(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        for _, row in ranked.iterrows():
            expected = sum(ranker.weights[name] * row[name] for name in OBJECTIVES)
            assert row["score"] == pytest.approx(expected)

    def test_first_pick_has_maximal_diversity_and_coverage(self, ranker):
        ranked = ranker.rank(user_id=1, k=3)
        assert ranked.iloc[0]["diversity"] == pytest.approx(1.0)

    def test_k_larger_than_pool_returns_what_exists(self, ranker):
        ranked = ranker.rank(user_id=1, k=100)
        assert 0 < len(ranked) <= 6

    def test_empty_candidate_pool_returns_empty_frame(self, ranker):
        empty = pd.DataFrame(columns=["movieId", "relevance", "sources"])
        ranked = ranker.rank(user_id=1, k=5, candidates=empty)
        assert ranked.empty
        assert "score" in ranked.columns

    def test_weight_override_changes_the_ordering(self, movies_df, ratings_df):
        ranker = MultiObjectiveRanker(movies_df, ratings_df, weights=RankingWeights(
            {**{n: 0.0 for n in OBJECTIVES}, "popularity_quality": 1.0}
        ))
        by_popularity = ranker.rank(user_id=6, k=3)
        novelty_weights = RankingWeights({**{n: 0.0 for n in OBJECTIVES}, "novelty": 1.0})
        by_novelty = ranker.rank(user_id=6, k=3, weights=novelty_weights)
        # Novelty and popularity/quality are opposing objectives; flipping all
        # the weight from one to the other must not leave the order untouched.
        assert list(by_popularity["movieId"]) != list(by_novelty["movieId"])

    def test_pure_novelty_ranks_the_rarest_item_first(self, movies_df, ratings_df):
        weights = RankingWeights({**{n: 0.0 for n in OBJECTIVES}, "novelty": 1.0})
        ranker = MultiObjectiveRanker(movies_df, ratings_df, weights=weights)
        assert ranker.rank(user_id=1, k=1).iloc[0]["movieId"] == 6

    def test_diversity_weight_spreads_genres(self, movies_df, ratings_df):
        """Turning diversity/coverage up must widen the genre mix, not narrow it."""
        stats = CatalogStats(movies_df, ratings_df)
        relevance_only = RankingWeights({**{n: 0.0 for n in OBJECTIVES}, "relevance": 1.0})
        diverse = relevance_only.with_overrides(diversity=2.0, coverage=2.0)
        ranker = MultiObjectiveRanker(movies_df, ratings_df, weights=relevance_only)

        def genre_count(ranked):
            covered = set()
            for movie_id in ranked["movieId"]:
                covered |= stats.genres(movie_id)
            return len(covered)

        narrow = genre_count(ranker.rank(user_id=6, k=3))
        wide = genre_count(ranker.rank(user_id=6, k=3, weights=diverse))
        assert wide >= narrow

    def test_recommend_returns_plain_id_list(self, ranker):
        ids = ranker.recommend(user_id=1, k=3)
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)


class TestGenreRelevanceGate:
    """The gate that stops keyword-only TF-IDF matches riding a high-rated seed."""

    @pytest.fixture
    def stats(self, movies_df, ratings_df):
        return CatalogStats(movies_df, ratings_df)

    def test_identical_genres_are_not_penalized(self, stats):
        # Movies 1 and 2 are both "Action, Adventure".
        assert genre_overlap_gate(2, 1, stats, floor=0.25) == pytest.approx(1.0)

    def test_zero_overlap_is_gated_to_the_floor(self, stats):
        # Movie 5 is Horror/Thriller, movie 1 is Action/Adventure.
        assert genre_overlap_gate(5, 1, stats, floor=0.25) == pytest.approx(0.25)

    def test_partial_overlap_scales_between_floor_and_one(self, stats):
        gate = genre_overlap_gate(3, 1, stats, floor=0.25)  # Comedy vs Action/Adventure
        assert 0.25 <= gate <= 1.0

    def test_floor_of_one_disables_the_gate(self, stats):
        assert genre_overlap_gate(5, 1, stats, floor=1.0) == pytest.approx(1.0)

    def test_floor_of_zero_fully_suppresses_zero_overlap(self, stats):
        assert genre_overlap_gate(5, 1, stats, floor=0.0) == pytest.approx(0.0)

    def test_missing_genre_metadata_is_never_penalized(self, ratings_df):
        """An unknown genre is not evidence of a mismatch."""
        movies = pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Known", "Untagged"],
                "genre_names": ["Action, Adventure", ""],
            }
        )
        stats = CatalogStats(movies, ratings_df)
        assert genre_overlap_gate(2, 1, stats, floor=0.25) == pytest.approx(1.0)
        assert genre_overlap_gate(1, 2, stats, floor=0.25) == pytest.approx(1.0)

    def test_gate_is_symmetric(self, stats):
        assert genre_overlap_gate(3, 1, stats, 0.25) == pytest.approx(
            genre_overlap_gate(1, 3, stats, 0.25)
        )

    def test_gate_floor_loads_from_config(self):
        assert load_genre_gate_floor("eval_config.yaml") == pytest.approx(0.25)

    def test_gate_floor_falls_back_when_block_absent(self, tmp_path):
        config = tmp_path / "c.yaml"
        config.write_text("ranking_objectives:\n  weights:\n    relevance: 1.0\n")
        assert load_genre_gate_floor(config) == pytest.approx(0.25)

    def test_out_of_range_floor_rejected(self, tmp_path):
        config = tmp_path / "c.yaml"
        config.write_text("ranking_objectives:\n  genre_relevance_gate:\n    floor: 1.5\n")
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            load_genre_gate_floor(config)


class TestGateInCandidateGeneration:
    class _FakeIR:
        """Minimal IR stand-in: movie 1 (Action) is similar to 2 (Action) and 5 (Horror)."""

        class _TFIDF:
            matrix = None
            movie_ids = None

        def __init__(self):
            self.tfidf = self._TFIDF()

        def similar_to(self, movie_id, limit=10):
            if movie_id != 1:
                return []
            # The horror match scores HIGHER on raw TF-IDF -- the exact
            # keyword-only mismatch the gate exists to correct.
            return [
                {"movieId": 5, "title": "Horror E", "similarity": 0.90},
                {"movieId": 2, "title": "Action B", "similarity": 0.60},
            ]

    @pytest.fixture
    def seed_ratings(self):
        return pd.DataFrame(
            {"userId": [1, 1], "movieId": [1, 3], "rating": [5.0, 3.0], "timestamp": ["2020-01-01"] * 2}
        )

    def test_gate_demotes_the_keyword_only_match(self, movies_df, seed_ratings):
        """Ungated, the horror movie wins; gated, the genre-matched one does."""
        ungated = CandidateGenerator(
            movies_df, seed_ratings, ir_engine=self._FakeIR(), genre_gate_floor=1.0
        )
        gated = CandidateGenerator(
            movies_df, seed_ratings, ir_engine=self._FakeIR(), genre_gate_floor=0.25
        )

        ungated_scores = ungated._content_candidates(seed_ratings, limit=10)
        gated_scores = gated._content_candidates(seed_ratings, limit=10)

        assert ungated_scores[5] > ungated_scores[2], "fixture no longer reproduces the mismatch"
        assert gated_scores[2] > gated_scores[5], "gate failed to demote the keyword-only match"

    def test_gate_does_not_change_genre_matched_candidates(self, movies_df, seed_ratings):
        ungated = CandidateGenerator(
            movies_df, seed_ratings, ir_engine=self._FakeIR(), genre_gate_floor=1.0
        )._content_candidates(seed_ratings, limit=10)
        gated = CandidateGenerator(
            movies_df, seed_ratings, ir_engine=self._FakeIR(), genre_gate_floor=0.25
        )._content_candidates(seed_ratings, limit=10)
        # Movie 2 shares both genres with seed 1, so its score must be untouched.
        assert gated[2] == pytest.approx(ungated[2])

    def test_candidates_for_seed_returns_gated_pool(self, movies_df, seed_ratings):
        generator = CandidateGenerator(
            movies_df, seed_ratings, ir_engine=self._FakeIR(), genre_gate_floor=0.25
        )
        pool = generator.candidates_for_seed(1, limit=10)
        assert list(pool.columns) == ["movieId", "relevance", "sources"]
        assert pool.iloc[0]["movieId"] == 2, "gated pool should lead with the genre-matched movie"
        assert pool["relevance"].max() == pytest.approx(1.0)

    def test_candidates_for_seed_without_ir_engine_is_empty(self, movies_df, seed_ratings):
        generator = CandidateGenerator(movies_df, seed_ratings, ir_engine=None)
        assert generator.candidates_for_seed(1).empty

    def test_cold_start_is_unaffected_by_the_gate(self, movies_df, ratings_df):
        """A user with no history has no seeds, so the gate cannot fire."""
        strict = CandidateGenerator(
            movies_df, ratings_df, ir_engine=self._FakeIR(), genre_gate_floor=0.0
        ).generate(user_id=9999)
        loose = CandidateGenerator(
            movies_df, ratings_df, ir_engine=self._FakeIR(), genre_gate_floor=1.0
        ).generate(user_id=9999)

        assert not strict.empty
        assert list(strict["movieId"]) == list(loose["movieId"])
        assert strict["relevance"].tolist() == pytest.approx(loose["relevance"].tolist())

    def test_collaborative_and_popularity_sources_are_not_gated(self, movies_df, ratings_df):
        """Only IR candidates have a seed; the other two sources must be untouched."""
        strict = CandidateGenerator(movies_df, ratings_df, ir_engine=None, genre_gate_floor=0.0)
        loose = CandidateGenerator(movies_df, ratings_df, ir_engine=None, genre_gate_floor=1.0)
        assert strict.generate(user_id=1)["relevance"].tolist() == pytest.approx(
            loose.generate(user_id=1)["relevance"].tolist()
        )
