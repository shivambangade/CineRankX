"""Verification script for the Multi-Objective Hybrid Ranking Engine (Module 4)."""

import sys
import time

import pandas as pd

from src.ir_engine import IREngine
from src.ranking_engine import (
    CandidateGenerator,
    MultiObjectiveRanker,
    OBJECTIVES,
    RankingWeights,
    genre_overlap_gate,
)
from src.strategy_selector import StrategyClassifier, extract_profile_features

N_RATINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
BAR = "=" * 100


def show(ranked, weights, note=""):
    print(f"  {'#':>2}  {'Title':42s} {'score':>6s} " +
          " ".join(f"{name[:9]:>9s}" for name in OBJECTIVES) + "  sources")
    for _, row in ranked.iterrows():
        print(f"  {int(row['rank']):>2}  {str(row['title'])[:42]:42s} {row['score']:6.3f} " +
              " ".join(f"{row[name]:9.3f}" for name in OBJECTIVES) +
              f"  {row['sources']}")
    if note:
        print(f"  {note}")


print("Loading catalog and ratings...")
movies_df = pd.read_csv("data/processed/movies_merged.csv")
ratings_df = pd.read_csv("newdata/movielens/rating.csv", nrows=N_RATINGS)
print(f"  {len(movies_df):,} movies | {len(ratings_df):,} ratings "
      f"from {ratings_df['userId'].nunique():,} users\n")

print("Loading Module 2 IR engine from disk...")
ir = IREngine()
print(f"  TF-IDF loaded: {ir.load()} ({ir.tfidf.matrix.shape[0]:,} movies x "
      f"{ir.tfidf.matrix.shape[1]:,} terms)\n")

print("Loading Module 3 strategy classifier from disk...")
clf = StrategyClassifier()
print(f"  Classifier loaded: {clf.load()}\n")

print("Building Module 4 ranker (catalog stats + baseline predictor + candidate sources)...")
_t = time.time()
ranker = MultiObjectiveRanker(movies_df, ratings_df, ir_engine=ir)
print(f"  Built in {time.time() - _t:.1f}s")
print(f"  Weights loaded from eval_config.yaml (normalized):")
for name, weight in ranker.weights.items():
    print(f"    {name:20s} {weight:.4f}")
print()

# Pick a user with a substantial history so every candidate source can fire.
counts = ratings_df["userId"].value_counts()
sample_user = int(counts[(counts > 80) & (counts < 400)].index[0])
user_ratings = ratings_df[ratings_df["userId"] == sample_user]

features = extract_profile_features(user_ratings, movies_df)
prediction = clf.predict(features).iloc[0]
strategy, confidence = prediction["predicted_strategy"], float(prediction["confidence"])

print(BAR)
print(f"SAMPLE USER {sample_user}: {len(user_ratings)} ratings, "
      f"mean {user_ratings['rating'].mean():.2f}")
print(BAR)
top_rated = user_ratings.nlargest(5, "rating").merge(movies_df[["movieId", "title", "genre_names"]], on="movieId")
print("Their top-rated movies (the IR seeds):")
for _, row in top_rated.iterrows():
    print(f"  {row['rating']:.1f}  {str(row['title'])[:45]:45s}  {str(row['genre_names'])[:40]}")
print(f"\nModule 3 predicted strategy: {strategy}  (confidence {confidence:.3f})")
from src.ranking_engine import strategy_source_weights
sw = strategy_source_weights(strategy, confidence)
print("Resulting candidate-source blend (strategy as weighted prior, not hard routing):")
for source, weight in sw.items():
    print(f"    {source:16s} {weight:.4f}")

candidates = ranker.generator.generate(sample_user, strategy=strategy, confidence=confidence)
print(f"\nCandidate pool: {len(candidates)} movies")

print("\n" + BAR)
print("VERIFICATION 1: top-10 with the DEFAULT weights from eval_config.yaml")
print(BAR)
_t = time.time()
default_ranked = ranker.rank(sample_user, k=10, candidates=candidates)
print(f"  (ranked in {time.time() - _t:.2f}s)")
show(default_ranked, ranker.weights)

print("\n" + BAR)
print("VERIFICATION 2: weight sensitivity — diversity 0.15 -> 0.60, coverage 0.10 -> 0.40")
print(BAR)
diverse_weights = ranker.weights.with_overrides(diversity=0.60, coverage=0.40)
print("  Weights now (renormalized):", ", ".join(f"{n}={w:.3f}" for n, w in diverse_weights.items()))
diverse_ranked = ranker.rank(sample_user, k=10, candidates=candidates, weights=diverse_weights)
show(diverse_ranked, diverse_weights)

genres_of = ranker.stats.genres
def spread(ranked):
    covered = set()
    for mid in ranked["movieId"]:
        covered |= genres_of(int(mid))
    return covered

d_default, d_diverse = spread(default_ranked), spread(diverse_ranked)
overlap = len(set(default_ranked["movieId"]) & set(diverse_ranked["movieId"]))
print(f"\n  Distinct genres covered: default={len(d_default)}  diversity-heavy={len(d_diverse)}")
print(f"  Titles shared between the two lists: {overlap}/10")
print(f"  Mean diversity score:    default={default_ranked['diversity'].mean():.3f}  "
      f"diversity-heavy={diverse_ranked['diversity'].mean():.3f}")
print(f"  Mean relevance score:    default={default_ranked['relevance'].mean():.3f}  "
      f"diversity-heavy={diverse_ranked['relevance'].mean():.3f}   <- the trade-off being paid")

print("\n" + BAR)
print("VERIFICATION 3: pure single-objective rankings (does each weight actually drive the list?)")
print(BAR)
for objective in OBJECTIVES:
    solo = RankingWeights({**{n: 0.0 for n in OBJECTIVES}, objective: 1.0})
    top = ranker.rank(sample_user, k=3, candidates=candidates, weights=solo)
    titles = " | ".join(str(t)[:28] for t in top["title"])
    print(f"  {objective:20s} -> {titles}")

print("\n" + BAR)
print("VERIFICATION 4: genre-overlap relevance gate on the Toy Story tonal mismatch")
print(BAR)
toy = ir.similar_to(1, limit=25)
toy_genres = ranker.stats.genres(1)
print(f"  Seed: Toy Story (1995) — genres: {', '.join(sorted(toy_genres))}")
print(f"  Gate floor from eval_config.yaml: {ranker.generator.genre_gate_floor}\n")
print("  Raw Module 2 IR similar_to(Toy Story) — top 10, with the gate each would receive:")
for i, hit in enumerate(toy[:10], 1):
    mid = int(hit["movieId"])
    gate = genre_overlap_gate(mid, 1, ranker.stats, ranker.generator.genre_gate_floor)
    flag = "  <-- GATED" if gate <= ranker.generator.genre_gate_floor + 1e-9 else ""
    print(f"    {i:>2}. {str(hit['title'])[:40]:40s} sim={hit['similarity']:.3f} "
          f"gate={gate:.3f} -> {hit['similarity'] * gate:.3f}  "
          f"{', '.join(sorted(ranker.stats.genres(mid)))[:32]}{flag}")

ungated_pool = pd.DataFrame({
    "movieId": [h["movieId"] for h in toy],
    "relevance": [h["similarity"] for h in toy],
    "sources": ["content_based"] * len(toy),
})
ungated_pool["relevance"] = ungated_pool["relevance"] / ungated_pool["relevance"].max()
gated_pool = ranker.generator.candidates_for_seed(1, limit=25)

print("\n  A) Module 4 ranking, gate DISABLED (floor=1.0) — top 10:")
ungated_ranked = ranker.rank(sample_user, k=10, candidates=ungated_pool)
show(ungated_ranked, ranker.weights)

print("\n  B) Module 4 ranking, gate ENABLED (floor=0.25) — top 10:")
gated_ranked = ranker.rank(sample_user, k=10, candidates=gated_pool)
show(gated_ranked, ranker.weights)

def horror_like(ranked):
    """Titles sharing NO genre with the Toy Story seed."""
    return [(int(r["rank"]), str(r["title"])[:38]) for _, r in ranked.iterrows()
            if not (ranker.stats.genres(int(r["movieId"])) & toy_genres)]

print("\n  Candidates sharing NO genre with the seed (the tonal mismatches):")
print(f"    gate DISABLED: {horror_like(ungated_ranked)}")
print(f"    gate ENABLED : {horror_like(gated_ranked)}")
print(f"\n    count in top 10 — disabled: {len(horror_like(ungated_ranked))}, "
      f"enabled: {len(horror_like(gated_ranked))}")

print("\n" + BAR)
print("VERIFICATION 5: cold-start user (no rating history at all)")
print(BAR)
cold = ranker.rank(user_id=999_999_999, k=5)
show(cold, ranker.weights)
print("  (only the popularity source can fire; the list is still populated and scored)")

# A cold-start user has no rated seeds, so no IR candidate exists to gate.
# Re-running with the strictest possible gate must therefore change nothing.
strict_generator = CandidateGenerator(
    movies_df, ratings_df, ir_engine=ir, stats=ranker.stats, genre_gate_floor=0.0
)
strict_cold = ranker.rank(
    user_id=999_999_999, k=5, candidates=strict_generator.generate(999_999_999)
)
identical = list(cold["movieId"]) == list(strict_cold["movieId"])
print(f"\n  Same 5 titles with the gate at its strictest (floor=0.0): {identical}")
print("  (no rated seeds -> no IR candidates -> the gate has nothing to act on)")
assert identical, "gate leaked into the cold-start path"

print("\n" + BAR)
print("Module 4 verification complete.")
print(BAR)
