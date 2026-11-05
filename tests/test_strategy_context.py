"""Tests for the strategy-aware features (inner backtest) and their train/serve consistency."""

import numpy as np
import pandas as pd
import pytest

from src.strategy_selector.strategy_context import (LiteContext, build_features, inner_scores,
                                                    load_context, save_context)
from src.strategy_selector.strategy_labeler import _BacktestContext, _content_based_hit_rate


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    genres = ["Action", "Comedy", "Drama", "Horror"]
    movies = pd.DataFrame({
        "movieId": range(1, 41),
        "genres": [genres[i % 4] for i in range(40)],
        "text_chunk_clean": [f"{genres[i % 4].lower()} word{i % 7} tag{i % 5}" for i in range(40)],
        "vote_count": rng.integers(1, 1000, 40)})
    rows = []
    for u in range(1, 31):
        liked = rng.choice(np.arange(1, 41), size=18, replace=False)
        rows += [(u, int(m), float(rng.choice([3.0, 4.0, 5.0]))) for m in liked]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    return movies, ratings


def test_fast_content_score_matches_the_reference_implementation(world):
    movies, ratings = world
    lite = LiteContext(ratings, movies, top_n=5)
    ref = _BacktestContext(ratings, movies, top_n=5)
    for user in (1, 2, 3):
        mine = ratings[ratings["userId"] == user]
        train, test = mine.iloc[:12], mine["movieId"].iloc[12:].to_numpy()
        assert lite.cb_hit_rate(train, test) == pytest.approx(_content_based_hit_rate(ref, train, test))


def test_inner_scores_are_deterministic_and_bounded(world):
    movies, ratings = world
    ctx = LiteContext(ratings, movies, top_n=10)
    mine = ratings[ratings["userId"] == 4]
    a, b = inner_scores(ctx, 4, mine), inner_scores(ctx, 4, mine)
    assert a == b
    assert set(a) == {"inner_cb", "inner_cf", "inner_pop"}
    assert all(0.0 <= v <= 1.0 for v in a.values())


def test_a_users_own_profile_is_never_their_own_neighbour(world):
    movies, ratings = world
    ctx = LiteContext(ratings, movies, top_n=10)
    mine = ratings[ratings["userId"] == 5]
    test_ids = mine["movieId"].to_numpy()[:5]
    # If the user's own stored row were allowed to match, their own best movies would
    # be recommended back and the score would be inflated; excluding it must not raise it.
    assert ctx.cf_hit_rate(5, mine.iloc[5:], test_ids) <= ctx.cf_hit_rate(999, mine.iloc[5:], test_ids)


def test_features_use_only_the_ratings_passed_in(world):
    movies, ratings = world
    ctx = LiteContext(ratings, movies, top_n=10)
    mine = ratings[ratings["userId"] == 7]
    few = build_features(mine.iloc[:9], movies, ctx).iloc[0]
    assert few["num_ratings"] == 9
    assert few["top100_share"] == pytest.approx(1.0)      # tiny catalog: everything is in the top 100


def test_training_and_serving_build_identical_columns(world, tmp_path):
    movies, ratings = world
    ctx = LiteContext(ratings, movies, top_n=10)
    batch = build_features(ratings[ratings["userId"] <= 5], movies, ctx)
    path = tmp_path / "ctx.pkl"
    save_context(ctx, path)
    served = build_features(ratings[ratings["userId"] == 3], movies, load_context(path))
    assert list(served.columns) == list(batch.columns)
    pd.testing.assert_frame_equal(served.reset_index(drop=True),
                                  batch[batch["userId"] == 3].reset_index(drop=True))
    for col in ("inner_cb", "inner_cf", "inner_pop", "top100_share", "top1000_share"):
        assert col in served.columns


def test_missing_context_file_returns_none(tmp_path):
    assert load_context(tmp_path / "nope.pkl") is None
