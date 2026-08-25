"""End-to-end KPI report — the project's final results table.

Computes every metric defined in eval_config.yaml from real data, across all
four engines, and refuses to finish if any configured KPI is missing.

Usage:
    PYTHONPATH=. python scripts/generate_kpi_report.py [n_ratings] [n_users]

Requires the Module 1/2/3 artifacts in data/processed/ (run the ingestion
pipeline and the Module 2/3 verification scripts first).
"""

import sys
import time

import pandas as pd

from src.evaluation import (
    aggregate,
    build_report,
    evaluate_recommendations,
    evaluate_search,
    load_k_values,
    measure,
    missing_from,
    render,
)
from src.evaluation.ml_metrics import to_report_names
from src.evaluation.search_metrics import build_relevance_index
from src.ir_engine import IREngine
from src.ranking_engine import MultiObjectiveRanker
from src.strategy_selector import (
    StrategyClassifier,
    extract_profile_features,
    label_users_with_best_strategy,
)

N_RATINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
N_USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

# Queries spanning genre terms, keyword terms and multi-word phrases, so the
# search KPIs are not measured on one narrow query shape.
QUERIES = [
    "action", "comedy", "horror", "romance", "documentary", "thriller",
    "space adventure", "world war", "murder mystery", "high school",
    "superhero", "time travel", "heist", "zombie apocalypse", "coming of age",
    "revenge", "friendship", "dystopian future", "serial killer", "musical",
]


def log(message: str) -> None:
    print(f"  {message}", flush=True)


print("=" * 78)
print("  Building CineRankX KPI report".center(78))
print("=" * 78)

log("Loading catalog and ratings...")
movies_df = pd.read_csv("data/processed/movies_merged.csv")
ratings_df = pd.read_csv("newdata/movielens/rating.csv", nrows=N_RATINGS)
log(f"{len(movies_df):,} movies | {len(ratings_df):,} ratings "
    f"from {ratings_df['userId'].nunique():,} users")

log("Loading Module 2 IR engine...")
ir = IREngine()
if not ir.load():
    raise SystemExit("IR engine artifacts missing — run scripts/verify_ir_engine.py first")

log("Loading Module 3 classifier...")
clf = StrategyClassifier()
classifier_loaded = clf.load()

log("Building Module 4 ranker...")
ranker = MultiObjectiveRanker(movies_df, ratings_df, ir_engine=ir)

# ---------------------------------------------------------------- group 1
print("\n[1/4] Search metrics...", flush=True)
relevance_index = build_relevance_index(movies_df)
search_results = evaluate_search(ir, QUERIES, movies_df, k=10, relevance_index=relevance_index)
log(f"{search_results['n_queries']} queries, mean latency "
    f"{search_results['Search Latency']:.2f} ms")

# ---------------------------------------------------------------- group 3
# Run before recommendations so the strategy predictions can feed the ranker.
print("\n[2/4] Machine learning metrics (strategy classifier)...", flush=True)
eval_user_ids = [int(u) for u in ratings_df["userId"].unique()[:N_USERS]]
user_ratings = ratings_df[ratings_df["userId"].isin(eval_user_ids)]

features = extract_profile_features(user_ratings, movies_df)
_t = time.time()
labels = label_users_with_best_strategy(user_ratings, movies_df, test_fraction=0.2)
log(f"labeled {len(labels)} users in {time.time() - _t:.0f}s")

fresh_clf = StrategyClassifier()
ml_raw = fresh_clf.fit(features, labels)
ml_results = to_report_names(ml_raw)
ml_results.update({k: v for k, v in ml_raw.items() if k not in ml_results})
log(f"accuracy {ml_raw['accuracy']:.4f} vs majority baseline "
    f"{ml_raw['majority_baseline_accuracy']:.4f}")

predictions = fresh_clf.predict(features)
strategies = {
    int(r["userId"]): (r["predicted_strategy"], float(r["confidence"]))
    for _, r in predictions.iterrows()
}

# ---------------------------------------------------------------- group 2
print("\n[3/4] Recommendation metrics...", flush=True)
k_values = load_k_values("eval_config.yaml")
_t = time.time()
rec_results = evaluate_recommendations(
    ranker, user_ratings, movies_df, user_ids=eval_user_ids,
    k_values=k_values, strategies=strategies,
)
log(f"evaluated {rec_results['n_users_evaluated']} users at K={k_values} "
    f"in {time.time() - _t:.0f}s")

# ---------------------------------------------------------------- group 4
print("\n[4/4] System metrics...", flush=True)
probe_user = eval_user_ids[0]
probe_features = features[features["userId"] == probe_user]
probe_candidates = ranker.generator.generate(probe_user)

measurements = [
    measure(lambda: ir.search("space adventure", limit=10), n_calls=60,
            label="IR search (Module 2)"),
    measure(lambda: ir.similar_to(1, limit=10), n_calls=40,
            label="IR similar_to (Module 2)"),
    measure(lambda: fresh_clf.predict(probe_features), n_calls=40,
            label="Strategy predict (Module 3)"),
    measure(lambda: ranker.rank(probe_user, k=10, candidates=probe_candidates), n_calls=20,
            label="Rank top-10 (Module 4)"),
]
system_results = aggregate(measurements)
log(f"aggregate throughput {system_results['Throughput']:.1f} ops/sec")

# ---------------------------------------------------------------- assemble
report = build_report(search_results, rec_results, ml_results, system_results, strict=True)
gaps = missing_from(report, "eval_config.yaml")
print(f"\nKPI coverage check: {'ALL METRICS PRESENT' if not gaps else f'MISSING {gaps}'}\n")

print(render(report))

print(f"\nRun parameters: {len(ratings_df):,} ratings loaded, {len(eval_user_ids)} users "
      f"evaluated, {len(QUERIES)} search queries, K values {k_values}.")
if not classifier_loaded:
    print("Note: no saved classifier was found on disk; one was trained fresh for this run.")
