"""Recommendation KPIs, driven off Module 4's final ranked output.

Ground truth is a held-out split of each user's own ratings: the ranker sees
the user's training ratings, and a recommendation counts as a hit if it appears
among the items that user rated POSITIVELY in the held-out portion. Items the
user rated poorly are not hits — recommending a movie someone disliked is not
a success just because they happened to watch it.

Cold-Start Accuracy uses the documented definition: users with <= 5 ratings,
scored against the popularity baseline's top-K on their held-out ratings.
"""

import numpy as np
import pandas as pd

from src.evaluation.metric_names import (
    COLD_START_ACCURACY,
    COVERAGE,
    DIVERSITY,
    F1,
    MAP,
    NOVELTY,
    PRECISION_AT_K,
    RECALL_AT_K,
)

POSITIVE_RATING_THRESHOLD = 4.0  # >= this counts as a genuine "liked" item
COLD_START_MAX_RATINGS = 5       # documented definition of a cold-start user


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-K recommendations the user actually liked."""
    if k <= 0:
        return 0.0
    return len([m for m in recommended[:k] if m in relevant]) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the user's liked held-out items that made the top-K."""
    if not relevant:
        return 0.0
    return len([m for m in recommended[:k] if m in relevant]) / len(relevant)


def f1_score_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(recommended: list[int], relevant: set[int], k: int) -> float:
    """Average Precision for one user — the per-user term inside MAP.

    Precision is sampled at each rank where a relevant item appears, then
    divided by min(k, |relevant|). Dividing by |relevant| alone would make a
    perfect top-K unreachable whenever the user has more than K liked items.
    """
    if not relevant:
        return 0.0
    hits, precision_sum = 0, 0.0
    for position, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            hits += 1
            precision_sum += hits / position
    denominator = min(k, len(relevant))
    return precision_sum / denominator if denominator > 0 else 0.0


def mean_average_precision(per_user_ap: list[float]) -> float:
    """MAP: Average Precision averaged over users."""
    return float(np.mean(per_user_ap)) if per_user_ap else 0.0


def coverage(all_recommended: list[list[int]], catalog_size: int) -> float:
    """Coverage: share of the catalog that appears in ANY user's list.

    Catalog coverage, not per-user coverage: it answers "how much of the
    library does this system ever surface", the standard filter-bubble check.
    """
    if catalog_size <= 0:
        return 0.0
    distinct = set()
    for recommended in all_recommended:
        distinct |= set(recommended)
    return len(distinct) / catalog_size


def intra_list_diversity(recommended: list[int], stats) -> float:
    """Diversity: mean pairwise genre dissimilarity within one list.

    1 - mean Jaccard genre similarity over all pairs. A single-item (or empty)
    list has no pairs, so it scores 0.0 rather than a misleading 1.0.
    """
    ids = [int(m) for m in recommended]
    if len(ids) < 2:
        return 0.0
    similarities = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = stats.genres(ids[i]), stats.genres(ids[j])
            union = a | b
            similarities.append(len(a & b) / len(union) if union else 0.0)
    return float(1.0 - np.mean(similarities))


def novelty(recommended: list[int], stats) -> float:
    """Novelty: mean self-information of the recommended items.

    Reads CatalogStats' novelty scores, so the reported KPI is the same
    quantity the ranker optimises rather than a parallel definition.
    """
    if not recommended:
        return 0.0
    return float(np.mean([stats.novelty(int(m)) for m in recommended]))


def split_user_ratings(ratings_df: pd.DataFrame, test_fraction: float = 0.2,
                       random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user random hold-out split.

    Split per user rather than globally so every evaluated user has both a
    profile to recommend from and held-out items to be scored against; a global
    split would leave some users with no training history at all.
    """
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    for _, user_ratings in ratings_df.groupby("userId"):
        n = len(user_ratings)
        shuffled = user_ratings.iloc[rng.permutation(n)]
        n_test = max(1, int(round(n * test_fraction))) if n > 1 else 0
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])
    return (
        pd.concat(train_parts) if train_parts else ratings_df.iloc[0:0],
        pd.concat(test_parts) if test_parts else ratings_df.iloc[0:0],
    )


def popularity_baseline_top_k(train_ratings: pd.DataFrame, movies_df: pd.DataFrame,
                              k: int) -> list[int]:
    """Top-K most-rated movies in the training data — the cold-start reference.

    Uses OBSERVED rating counts, not TMDB's `popularity` column. TMDB
    popularity is a live, recency-weighted metric; MovieLens 20M's ratings stop
    in 2015, so the two disagree almost completely. On a 900k-rating sample the
    TMDB top-20 shares just 1 title with the 20 most-rated films, and includes
    movies rated by 0 and 5 users respectively. Scored as a baseline it
    captures 9,426 ratings against the observed baseline's 46,864 — a straw man
    that would overstate any lift measured against it by a wide margin.

    Falls back to TMDB popularity only when no ratings are available at all.
    """
    if train_ratings is not None and not train_ratings.empty:
        counts = train_ratings["movieId"].value_counts().head(k)
        return [int(m) for m in counts.index]
    if "popularity" in movies_df.columns:
        return [int(m) for m in movies_df.nlargest(k, "popularity")["movieId"]]
    return []


def evaluate_cold_start(ranker, train_df: pd.DataFrame, test_by_user: dict,
                        movies_df: pd.DataFrame, user_ids: list[int], k: int,
                        max_history: int = COLD_START_MAX_RATINGS,
                        max_users: int = 100) -> dict:
    """Cold-Start Accuracy: hit-rate for users with <= `max_history` ratings.

    MovieLens 20M only admits users who rated at least 20 movies, so NO natural
    cold-start user exists in it -- the minimum profile size in any sample is
    exactly 20. Measured naturally the KPI is permanently undefined, so when no
    genuine cold-start user is present the condition is *simulated*: a cohort of
    real users has its profile truncated to `max_history` ratings, and the model
    is scored on their untouched held-out items. The returned `mode` says which
    happened, so a simulated figure is never mistaken for a natural one.

    Every other user keeps their full history, so the collaborative neighbourhood
    the cold user is matched against stays realistic -- truncating everybody
    would measure a cold *system*, not a cold user.
    """
    from src.ranking_engine import CandidateGenerator

    history_sizes = train_df.groupby("userId").size()
    natural = [u for u in user_ids if history_sizes.get(u, 0) <= max_history]

    if natural:
        mode, cohort, cold_train = "natural", natural[:max_users], train_df
    else:
        mode = "simulated"
        cohort = [u for u in user_ids if history_sizes.get(u, 0) > max_history][:max_users]
        if not cohort:
            return {"mode": "unavailable", "n_users": 0, "accuracy": None, "baseline_accuracy": None}
        cohort_set = set(cohort)
        truncated = [
            g.head(max_history) if uid in cohort_set else g
            for uid, g in train_df.groupby("userId")
        ]
        cold_train = pd.concat(truncated)

    generator = CandidateGenerator(
        movies_df, cold_train, ir_engine=ranker.ir_engine, stats=ranker.stats
    )
    baseline = popularity_baseline_top_k(cold_train, movies_df, k)

    rows = []
    for user_id in cohort:
        test_ratings = test_by_user.get(user_id)
        if test_ratings is None or test_ratings.empty:
            continue
        relevant = set(
            test_ratings.loc[test_ratings["rating"] >= POSITIVE_RATING_THRESHOLD, "movieId"].astype(int)
        )
        if not relevant:
            continue
        candidates = generator.generate(user_id)
        ranked = ranker.rank(user_id, k=k, candidates=candidates)
        recommended = [int(m) for m in ranked["movieId"]] if not ranked.empty else []
        rows.append({
            "userId": user_id,
            "system_hit": float(any(m in relevant for m in recommended)),
            "baseline_hit": float(any(m in relevant for m in baseline)),
            "system_precision": precision_at_k(recommended, relevant, k),
            "baseline_precision": precision_at_k(baseline, relevant, k),
        })

    if not rows:
        return {"mode": mode, "n_users": 0, "accuracy": None, "baseline_accuracy": None}

    detail = pd.DataFrame(rows)
    return {
        "mode": mode,
        "n_users": len(detail),
        "max_history": max_history,
        # Accuracy = share of cold users for whom at least one of the top-K
        # recommendations was an item they actually liked.
        "accuracy": float(detail["system_hit"].mean()),
        "baseline_accuracy": float(detail["baseline_hit"].mean()),
        "system_precision": float(detail["system_precision"].mean()),
        "baseline_precision": float(detail["baseline_precision"].mean()),
    }


def evaluate_recommendations(ranker, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                             user_ids: list[int], k_values: list[int],
                             strategies: dict[int, tuple[str, float]] | None = None,
                             test_fraction: float = 0.2,
                             random_state: int = 42) -> dict:
    """Run Module 4 for each user and compute the recommendation KPI group.

    Args:
        ranker: a fitted MultiObjectiveRanker.
        user_ids: users to evaluate.
        k_values: the K's from eval_config.yaml's recommendation.k_values.
        strategies: optional {userId: (strategy, confidence)} from Module 3.

    Returns:
        Dict keyed by the exact eval_config.yaml metric names. Precision@K,
        Recall@K and F1 are themselves dicts keyed by K.
    """
    strategies = strategies or {}
    max_k = max(k_values)

    train_df, test_df = split_user_ratings(ratings_df, test_fraction, random_state)
    train_by_user = {uid: g for uid, g in train_df.groupby("userId")}
    test_by_user = {uid: g for uid, g in test_df.groupby("userId")}

    # The ranker must only ever see training ratings. Rebuilding its candidate
    # generator on the training split is what keeps held-out items genuinely
    # unseen -- reusing the ranker's original full-history generator would let
    # test items leak in as candidates the user "already rated".
    from src.ranking_engine import CandidateGenerator

    generator = CandidateGenerator(
        movies_df, train_df, ir_engine=ranker.ir_engine, stats=ranker.stats
    )

    per_user, all_lists = [], []
    for user_id in user_ids:
        test_ratings = test_by_user.get(user_id)
        if test_ratings is None or test_ratings.empty:
            continue
        relevant = set(
            test_ratings.loc[
                test_ratings["rating"] >= POSITIVE_RATING_THRESHOLD, "movieId"
            ].astype(int)
        )

        strategy, confidence = strategies.get(user_id, (None, 1.0))
        candidates = generator.generate(user_id, strategy=strategy, confidence=confidence)
        ranked = ranker.rank(user_id, k=max_k, candidates=candidates)
        recommended = [int(m) for m in ranked["movieId"]] if not ranked.empty else []
        all_lists.append(recommended)

        row = {"userId": user_id, "n_relevant": len(relevant)}
        for k in k_values:
            p = precision_at_k(recommended, relevant, k)
            r = recall_at_k(recommended, relevant, k)
            row[f"P@{k}"] = p
            row[f"R@{k}"] = r
            row[f"F1@{k}"] = f1_score_at_k(p, r)
        row["AP"] = average_precision(recommended, relevant, max_k)
        row["diversity"] = intra_list_diversity(recommended, ranker.stats)
        row["novelty"] = novelty(recommended, ranker.stats)
        per_user.append(row)

    if not per_user:
        return {"n_users_evaluated": 0}

    detail = pd.DataFrame(per_user)
    cold = evaluate_cold_start(
        ranker, train_df, test_by_user, movies_df, user_ids, k=max_k
    )

    result = {
        PRECISION_AT_K: {k: float(detail[f"P@{k}"].mean()) for k in k_values},
        RECALL_AT_K: {k: float(detail[f"R@{k}"].mean()) for k in k_values},
        F1: {k: float(detail[f"F1@{k}"].mean()) for k in k_values},
        MAP: mean_average_precision(detail["AP"].tolist()),
        COVERAGE: coverage(all_lists, len(movies_df)),
        DIVERSITY: float(detail["diversity"].mean()),
        NOVELTY: float(detail["novelty"].mean()),
        COLD_START_ACCURACY: cold["accuracy"],
        "cold_start_baseline_accuracy": cold["baseline_accuracy"],
        "cold_start_n_users": cold["n_users"],
        "cold_start_mode": cold["mode"],
        "cold_start_detail": cold,
        "n_users_evaluated": len(detail),
        "max_k": max_k,
        "per_user": detail,
    }
    return result
