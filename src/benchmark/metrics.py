"""Ranking and rating-prediction metrics for the frozen benchmark.

"Accuracy" in the dashboard is Hit Rate@K: the share of evaluated users for whom at
least one of the top-K recommendations is a held-out movie they rated >= the positive
threshold (4.0). It is stricter than it sounds (a user needs a hit among ~27,000
titles) and saturates for heavy users, which is why Precision, Recall, NDCG and MAP
are always reported next to it.
"""

import numpy as np


def hit_rate(rec: list[int], relevant: set[int], k: int) -> float:
    return float(any(m in relevant for m in rec[:k]))


def precision(rec: list[int], relevant: set[int], k: int) -> float:
    return sum(m in relevant for m in rec[:k]) / k


def recall(rec: list[int], relevant: set[int], k: int) -> float:
    return sum(m in relevant for m in rec[:k]) / len(relevant) if relevant else 0.0


def ndcg(rec: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@K with the ideal ranking limited to min(K, |relevant|) hits."""
    gains = sum(1.0 / np.log2(i + 2) for i, m in enumerate(rec[:k]) if m in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return gains / ideal if ideal > 0 else 0.0


def average_precision(rec: list[int], relevant: set[int], k: int) -> float:
    hits, total = 0, 0.0
    for pos, m in enumerate(rec[:k], start=1):
        if m in relevant:
            hits += 1
            total += hits / pos
    denom = min(k, len(relevant))
    return total / denom if denom else 0.0


def rating_errors(predicted: list[float], actual: list[float]) -> dict[str, float]:
    err = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "n": int(len(err))}


def summarize(lists: dict[int, list[int]], positives: dict[int, set[int]], ks: list[int],
              catalog_size: int, diversity_fn=None, novelty_fn=None) -> dict:
    """Mean metrics over users; every list is scored against the same positives."""
    users = [u for u in lists if positives.get(u)]
    out: dict = {"n_users": len(users)}
    for k in ks:
        out[f"hit_rate@{k}"] = float(np.mean([hit_rate(lists[u], positives[u], k) for u in users]))
        out[f"precision@{k}"] = float(np.mean([precision(lists[u], positives[u], k) for u in users]))
        out[f"recall@{k}"] = float(np.mean([recall(lists[u], positives[u], k) for u in users]))
        out[f"ndcg@{k}"] = float(np.mean([ndcg(lists[u], positives[u], k) for u in users]))
        out[f"map@{k}"] = float(np.mean([average_precision(lists[u], positives[u], k) for u in users]))
    max_k = max(ks)
    shown = set().union(*[set(lists[u][:max_k]) for u in users]) if users else set()
    out["coverage"] = len(shown) / catalog_size
    if diversity_fn:
        out["diversity"] = float(np.mean([diversity_fn(lists[u][:max(ks)]) for u in users]))
    if novelty_fn:
        out["novelty"] = float(np.mean([novelty_fn(lists[u][:max(ks)]) for u in users]))
    return out
