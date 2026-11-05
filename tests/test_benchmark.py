"""Tests for the frozen benchmark: deterministic split, metrics, frozen baseline, EASE, evaluation."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark import baseline as baseline_module
from src.benchmark.baseline import FrozenBaseline
from src.benchmark.config import load_benchmark_config
from src.benchmark.evaluate import evaluate_strategy, exclusions
from src.benchmark.metrics import average_precision, hit_rate, ndcg, precision, rating_errors, recall
from src.benchmark.split import (TEST, TRAIN, VALIDATION, assign_splits, build_benchmark,
                                 uniform_hash)
from src.ranking_engine.ease import EaseModel

CONFIG = {**load_benchmark_config(), "overall_users": 6, "cold_users": 4, "tuning_users": 3,
          "min_train_ratings": 5, "cold_history": 2}


def synthetic_frame(n_users=40, n_items=60, per_user=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        taste = u % 2  # two taste groups prefer disjoint halves of the catalog
        pool = np.arange(1, n_items + 1)
        favoured = pool[(pool % 2) == taste]
        items = rng.choice(favoured, size=per_user, replace=False)
        rows += [(u, int(m), float(rng.choice([4.0, 4.5, 5.0]))) for m in items]
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    frame["split"] = assign_splits(frame["userId"].to_numpy(), frame["movieId"].to_numpy(), CONFIG)
    frame["order"] = uniform_hash(frame["userId"].to_numpy(), frame["movieId"].to_numpy(), 99).astype("float32")
    return frame


def test_split_is_deterministic_and_independent_of_row_order():
    frame = synthetic_frame()
    again = assign_splits(frame["userId"].to_numpy(), frame["movieId"].to_numpy(), CONFIG)
    shuffled = frame.sample(frac=1.0, random_state=3)
    reordered = pd.Series(assign_splits(shuffled["userId"].to_numpy(), shuffled["movieId"].to_numpy(), CONFIG),
                          index=shuffled.index).sort_index().to_numpy()
    assert (again == frame["split"].to_numpy()).all() and (reordered == frame["split"].to_numpy()).all()
    shares = pd.Series(frame["split"]).value_counts(normalize=True)
    assert shares[TRAIN] == pytest.approx(0.70, abs=0.08) and shares[TEST] == pytest.approx(0.20, abs=0.08)
    assert set(frame["split"]) == {TRAIN, VALIDATION, TEST}


def test_split_changes_with_seed_only():
    frame = synthetic_frame()
    other = assign_splits(frame["userId"].to_numpy(), frame["movieId"].to_numpy(), {**CONFIG, "seed": 1})
    assert (other != frame["split"].to_numpy()).any()


def test_benchmark_cohorts_disjoint_cold_truncated_and_test_never_in_train():
    bench = build_benchmark(synthetic_frame(), CONFIG)
    groups = [set(map(int, v)) for v in bench.cohorts.values()]
    assert sum(len(g) for g in groups) == len(set().union(*groups))   # no user in two cohorts
    sizes = bench.train.groupby("userId").size()
    for cold_user in bench.cohorts["test_cold"]:
        assert sizes[int(cold_user)] == CONFIG["cold_history"]
    train_pairs = set(zip(bench.train["userId"], bench.train["movieId"]))
    for user, info in bench.truth["test"].items():
        assert not (train_pairs & {(user, m) for m in info["rated"]})
    assert bench.fingerprint == build_benchmark(synthetic_frame(), CONFIG).fingerprint


def test_exclusions_never_use_test_items():
    bench = build_benchmark(synthetic_frame(), CONFIG)
    user = int(bench.cohorts["test_overall"][0])
    test_items = set(bench.truth["test"][user]["rated"])
    assert not (exclusions(bench, "test", [user])[user] & test_items)
    assert not (exclusions(bench, "validation", [user])[user] & test_items)


def test_ranking_metrics_hand_computed():
    rec, rel = [1, 2, 3, 4], {2, 4, 9}
    assert hit_rate(rec, rel, 1) == 0.0 and hit_rate(rec, rel, 2) == 1.0
    assert precision(rec, rel, 4) == 0.5 and recall(rec, rel, 4) == pytest.approx(2 / 3)
    assert average_precision(rec, rel, 4) == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert ndcg(rec, rel, 4) == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) /
                                              (1 + 1 / np.log2(3) + 1 / np.log2(4)))
    assert ndcg([2, 4, 9], rel, 3) == pytest.approx(1.0)
    assert rating_errors([4.0, 3.0], [5.0, 3.0])["rmse"] == pytest.approx(np.sqrt(0.5))


def test_baseline_is_frozen_independent_and_deterministic():
    tree = ast.parse(Path(baseline_module.__file__).read_text(encoding="utf-8"))
    imported = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert imported <= {"numpy", "pandas"}                      # nothing from CineRankX can reach it
    train = pd.DataFrame({"userId": [1, 1, 2, 2, 3], "movieId": [10, 20, 10, 30, 10], "rating": [4, 5, 3, 2, 4]})
    a, b = FrozenBaseline(train), FrozenBaseline(train.sample(frac=1, random_state=1))
    assert a.ranking == b.ranking == [10, 20, 30]
    assert a.recommend({10}, 2) == [20, 30]
    assert 0.5 <= a.predict(10) <= 5.0 and a.predict(999) == pytest.approx(a.global_mean)


def test_baseline_does_not_change_when_cinerankx_is_retrained():
    bench = build_benchmark(synthetic_frame(), CONFIG)
    users = [int(u) for u in bench.cohorts["test_overall"]]

    def run():
        base = FrozenBaseline(bench.train)
        return evaluate_strategy(bench, "test", "overall",
                                 lambda us, ex, k: {u: base.recommend(ex[u], k) for u in us}, catalog_size=60)

    before = run()
    model_a = EaseModel.fit(bench.train, 60, 10.0)                 # "retrain" CineRankX twice, differently
    EaseModel.fit(bench.train.sample(frac=0.5, random_state=0), 30, 500.0)
    assert model_a is not None and users
    assert run() == before


def test_ease_beats_popularity_on_structured_data_with_same_ground_truth():
    bench = build_benchmark(synthetic_frame(60, 80, 40), {**CONFIG, "overall_users": 12})
    base = FrozenBaseline(bench.train)
    ease = EaseModel.fit(bench.train, 80, 20.0)
    profiles = {int(u): g["movieId"].tolist() for u, g in bench.train.groupby("userId")}

    def ease_many(users, ex, k):
        scores = ease.blended_scores(ease.user_matrix(profiles, users), 1.0)
        out = {}
        for i, u in enumerate(users):
            for m in ex[u]:
                if m in ease.index:
                    scores[i, ease.index[m]] = -np.inf
            out[u] = [int(ease.item_ids[c]) for c in np.argsort(-scores[i])[:k]]
        return out

    pop = evaluate_strategy(bench, "test", "overall",
                            lambda us, ex, k: {u: base.recommend(ex[u], k) for u in us}, catalog_size=80)
    new = evaluate_strategy(bench, "test", "overall", ease_many, catalog_size=80)
    assert pop["n_users"] == new["n_users"]
    assert new["ndcg@10"] > pop["ndcg@10"] and new["hit_rate@10"] >= pop["hit_rate@10"]
