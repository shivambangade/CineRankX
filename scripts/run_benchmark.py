"""Reproducible strategy comparison: frozen baseline vs. CineRankX on one fixed benchmark.

Usage:
    PYTHONPATH=. python scripts/run_benchmark.py [--refresh-split] [--retune] [--skip-before]

1. Builds (or loads) the frozen split and cohorts from eval_config.yaml `benchmark:`.
2. Scores the FROZEN baseline; compares it with the stored record and aborts on any drift.
3. Tunes CineRankX on the validation split only (or reuses data/benchmark/cinerankx_params.json).
4. Scores the previous CineRankX pipeline ("before") and the improved one ("after") on the
   untouched test split, overall and cold-start, and writes reports/strategy_comparison.json.
5. Retrains the EASE model on all ratings for the live app and saves the tuned blend.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import baseline as baseline_module
from src.benchmark import metrics as metrics_module
from src.benchmark import split as split_module
from src.benchmark.baseline import BASELINE_VERSION, FrozenBaseline
from src.benchmark.cinerankx import CineRankXStrategy, weights_from
from src.benchmark.config import BENCH_DIR, load_benchmark_config
from src.benchmark.evaluate import evaluate_strategy
from src.benchmark.split import SPLIT_VERSION, get_benchmark
from src.benchmark.tuning import tune_relevance, tune_weights
from src.evaluation.recommendation_metrics import intra_list_diversity, novelty
from src.ir_engine import IREngine
from src.ranking_engine import MultiObjectiveRanker
from src.ranking_engine.config import RankingWeights
from src.ranking_engine.ease import EaseModel
from src.ranking_engine.objectives import CatalogStats
from src.ranking_engine.predictor import BaselinePredictor

FROZEN_PATH = BENCH_DIR / "baseline_frozen.json"
PARAMS_PATH = BENCH_DIR / "cinerankx_params.json"
REPORT_PATH = Path("reports/strategy_comparison.json")
RATING_CSV = "newdata/movielens/rating.csv"


def code_hash() -> str:
    h = hashlib.sha256()
    for module in (baseline_module, metrics_module, split_module):
        h.update(Path(module.__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def run_baseline(bench, catalog_size, stats):
    base = FrozenBaseline(bench.train)
    div, nov = (lambda r: intra_list_diversity(r, stats)), (lambda r: novelty(r, stats))
    out = {}
    for cohort in ("overall", "cold"):
        out[cohort] = evaluate_strategy(
            bench, "test", cohort, lambda us, ex, k: {u: base.recommend(ex[u], k) for u in us},
            predict=lambda u, m: base.predict(m), catalog_size=catalog_size,
            diversity_fn=div, novelty_fn=nov)
    return base, out


def freeze_baseline(bench, results) -> None:
    record = {"baseline_version": BASELINE_VERSION, "split_version": SPLIT_VERSION,
              "code_sha256": code_hash(), "benchmark_fingerprint": bench.fingerprint, "results": results}
    if not FROZEN_PATH.exists():
        FROZEN_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Baseline recorded in {FROZEN_PATH} (first run)")
        return
    old = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if old["code_sha256"] != record["code_sha256"] or old["benchmark_fingerprint"] != record["benchmark_fingerprint"]:
        raise SystemExit("Frozen baseline/benchmark CODE or SPLIT changed. Bump the version and delete "
                         f"{FROZEN_PATH} only if you intend to start a new benchmark.")
    if json.dumps(old["results"], sort_keys=True) != json.dumps(results, sort_keys=True):
        raise SystemExit("Frozen baseline results drifted from the recorded values.")
    print("Frozen baseline verified: code, split and results identical to the record.")


def before_pipeline(bench, movies, ir, stats, predictor):
    """The previous CineRankX pipeline (IR + user-kNN + popularity candidates), unchanged."""
    pool_users = bench.train["userId"].unique()[:5000]
    members = np.concatenate(list(bench.cohorts.values()))
    keep = bench.train["userId"].isin(np.union1d(pool_users, members))
    small = bench.train[keep].reset_index(drop=True)
    ranker = MultiObjectiveRanker(movies, small, ir_engine=ir, predictor=predictor, stats=stats)
    return lambda users, ex, k: {
        u: [int(m) for m in ranker.rank(u, k=k, candidates=ranker.generator.generate(
            u, pool_size=200).pipe(lambda c: c[~c["movieId"].isin(ex[u])]))["movieId"]] for u in users}


def main() -> None:
    config = load_benchmark_config()
    movies = pd.read_csv("data/processed/movies_merged.csv")
    ids = set(movies["movieId"].astype(int))
    bench = get_benchmark(config, RATING_CSV, ids, refresh="--refresh-split" in sys.argv)
    stats = CatalogStats(movies, bench.train)
    size = len(movies)
    print(f"Benchmark fingerprint {bench.fingerprint[:16]}; cohorts "
          f"{ {k: len(v) for k, v in bench.cohorts.items()} }")

    _, base_results = run_baseline(bench, size, stats)
    freeze_baseline(bench, base_results)

    ir = IREngine()
    ir = ir if ir.load() else None
    strategy = CineRankXStrategy(bench, movies, ir)
    if PARAMS_PATH.exists() and "--retune" not in sys.argv:
        params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
        strategy.fit(params["l2"])
        strategy.beta.update(params["beta"])
        for cohort, block in params["weights"].items():
            strategy.weights[cohort] = weights_from(block["weights"])
        print(f"Reusing tuned parameters from {PARAMS_PATH}")
    else:
        print("Tuning relevance model on the validation split...")
        params = tune_relevance(strategy, size)
        print("Tuning objective weights on the validation split...")
        params["weights"] = tune_weights(strategy, FrozenBaseline(bench.train), size, config["seed"])
        PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")

    div, nov = (lambda r: intra_list_diversity(r, stats)), (lambda r: novelty(r, stats))
    after, before, relevance_only = {}, {}, {}
    old = None if "--skip-before" in sys.argv else before_pipeline(bench, movies, ir, stats, strategy.predictor)
    for cohort in ("overall", "cold"):
        after[cohort] = evaluate_strategy(
            bench, "test", cohort, lambda us, ex, k, c=cohort: strategy.recommend_many(us, ex, k, c),
            predict=strategy.predict, catalog_size=size, diversity_fn=div, novelty_fn=nov)
        relevance_only[cohort] = evaluate_strategy(
            bench, "test", cohort,
            lambda us, ex, k, c=cohort: strategy.recommend_many(us, ex, k, c, weights=None),
            catalog_size=size, diversity_fn=div, novelty_fn=nov)
        if old:
            before[cohort] = evaluate_strategy(bench, "test", cohort, old, catalog_size=size,
                                               diversity_fn=div, novelty_fn=nov)
        print(f"{cohort}: baseline hit@10={base_results[cohort]['hit_rate@10']:.4f} "
              f"after={after[cohort]['hit_rate@10']:.4f} "
              + (f"before={before[cohort]['hit_rate@10']:.4f}" if old else ""))

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "headline_metric": f"hit_rate@{config['headline_k']}", "config": config,
        "benchmark_fingerprint": bench.fingerprint, "split_version": SPLIT_VERSION,
        "baseline": {"version": BASELINE_VERSION, "code_sha256": code_hash(), "results": base_results},
        "cinerankx": {"after": after, "before": before or None, "relevance_only": relevance_only,
                      "params": params},
    }, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

    # Deployment model for the live app: same tuned hyperparameters, refit on ALL ratings.
    every = pd.read_csv(RATING_CSV, usecols=["userId", "movieId", "rating"])
    EaseModel.fit(every[every["movieId"].isin(ids)], strategy.n_items, params["l2"]).save()
    Path("data/processed/cinerankx_model.json").write_text(json.dumps({
        "beta": params["beta"], "weights": {c: b["weights"] for c, b in params["weights"].items()},
        "l2": params["l2"], "n_items": strategy.n_items}, indent=2), encoding="utf-8")
    print("Saved deployment model (data/processed/ease_model.npz, cinerankx_model.json)")


if __name__ == "__main__":
    main()
