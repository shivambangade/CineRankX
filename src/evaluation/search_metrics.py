"""Search KPIs — Precision@10, Recall@10, MRR, NDCG, Search Latency.

RELEVANCE IS A PROXY, NOT GROUND TRUTH
--------------------------------------
This project has no hand-labeled relevance judgements. Every number in this
module rests on the documented substitute:

    a search result is RELEVANT if it shares at least one genre or keyword
    with the query.

That is a lenient, recall-oriented definition. Two consequences are stated
wherever these metrics are reported rather than buried here:

1. The relevant pool is enormous. A one-word genre query like "action" marks
   thousands of the ~26.7k catalog movies relevant, so **Recall@10 is bounded
   above by 10/|relevant| and is structurally near zero** — it measures the
   catalog's breadth, not the engine's quality. `relevant_pool_size` is
   reported alongside it so the figure stays interpretable.
2. Precision@10 is correspondingly generous: matching one genre is a low bar,
   so a high score here does not imply the results are *good*, only on-topic.
"""

import time

import numpy as np
import pandas as pd

from src.evaluation.metric_names import (
    MRR,
    NDCG,
    PRECISION_AT_10,
    RECALL_AT_10,
    SEARCH_LATENCY,
)

RELEVANCE_PROXY = (
    "A result is relevant if it shares >=1 genre or keyword with the query "
    "(documented proxy - no hand-labeled relevance set exists)."
)

_DEFAULT_K = 10


def _tokenize(raw) -> set[str]:
    """Split a genre/keyword cell into a lowercase token set.

    Handles all three shapes these columns take: a stringified Python list
    ("['Animation', 'Comedy']", how movies_merged.csv round-trips), plain
    comma-separated text, and MovieLens's pipe-separated genres.
    """
    if not isinstance(raw, str) or not raw.strip():
        return set()
    parts = raw.replace("|", ",").split(",")
    return {p.strip().strip("[]'\" ").strip().lower() for p in parts if p.strip(" []'\"")}


def build_relevance_index(movies_df: pd.DataFrame) -> dict[int, set[str]]:
    """Map movieId -> its genre+keyword token set, for proxy relevance."""
    genre_col = "genre_names" if "genre_names" in movies_df.columns else "genres"
    index = {}
    for row in movies_df.itertuples(index=False):
        tokens = _tokenize(getattr(row, genre_col, ""))
        tokens |= _tokenize(getattr(row, "keyword_names", ""))
        index[int(row.movieId)] = tokens
    return index


def is_relevant(movie_id: int, query_tokens: set[str], relevance_index: dict) -> bool:
    """Apply the documented proxy to one result."""
    return bool(relevance_index.get(int(movie_id), set()) & query_tokens)


def precision_at_k(relevance_flags: list[bool], k: int = _DEFAULT_K) -> float:
    """Precision@K: fraction of the top-K results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = relevance_flags[:k]
    # Divide by k, not len(top_k): a query returning 3 results of which 3 are
    # relevant has not achieved Precision@10 of 1.0 -- it failed to fill the
    # slate, and dividing by the short length would hide that.
    return sum(top_k) / k


def recall_at_k(relevance_flags: list[bool], total_relevant: int, k: int = _DEFAULT_K) -> float:
    """Recall@K: fraction of ALL relevant items that appear in the top-K.

    Structurally tiny under the genre/keyword proxy — see the module docstring.
    """
    if total_relevant <= 0:
        return 0.0
    return sum(relevance_flags[:k]) / total_relevant


def reciprocal_rank(relevance_flags: list[bool]) -> float:
    """1 / rank of the first relevant result; 0.0 if none are relevant."""
    for position, relevant in enumerate(relevance_flags, start=1):
        if relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevance_flags: list[bool], k: int = _DEFAULT_K, total_relevant: int | None = None) -> float:
    """NDCG@K with binary gains.

    The ideal ranking puts min(k, total_relevant) relevant items at the top. If
    total_relevant is not supplied it defaults to k, which is correct here
    because the proxy's relevant pool is far larger than any K used.
    """
    gains = np.asarray(relevance_flags[:k], dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    n_ideal = k if total_relevant is None else min(k, total_relevant)
    if n_ideal <= 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, n_ideal + 2))
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_search(ir_engine, queries: list[str], movies_df: pd.DataFrame,
                    k: int = _DEFAULT_K, relevance_index: dict | None = None) -> dict:
    """Run every query through the IR engine and compute the search KPI group.

    Returns a dict keyed by the exact eval_config.yaml metric names, plus
    per-query detail and the relevance-proxy caveat for the report to print.
    """
    if relevance_index is None:
        relevance_index = build_relevance_index(movies_df)

    all_tokens = set()
    for tokens in relevance_index.values():
        all_tokens |= tokens

    per_query, latencies = [], []
    for query in queries:
        query_tokens = {t.strip().lower() for t in query.split() if t.strip()} & all_tokens

        started = time.perf_counter()
        results = ir_engine.search(query, limit=k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        flags = [is_relevant(r["movieId"], query_tokens, relevance_index) for r in results]
        total_relevant = sum(
            1 for tokens in relevance_index.values() if tokens & query_tokens
        ) if query_tokens else 0

        per_query.append({
            "query": query,
            "n_results": len(results),
            PRECISION_AT_10: precision_at_k(flags, k),
            RECALL_AT_10: recall_at_k(flags, total_relevant, k),
            MRR: reciprocal_rank(flags),
            NDCG: ndcg_at_k(flags, k, total_relevant),
            "relevant_pool_size": total_relevant,
            "latency_ms": latency_ms,
        })

    if not per_query:
        return {"n_queries": 0, "relevance_proxy": RELEVANCE_PROXY}

    detail = pd.DataFrame(per_query)
    latencies = np.asarray(latencies)
    return {
        PRECISION_AT_10: float(detail[PRECISION_AT_10].mean()),
        RECALL_AT_10: float(detail[RECALL_AT_10].mean()),
        MRR: float(detail[MRR].mean()),
        NDCG: float(detail[NDCG].mean()),
        SEARCH_LATENCY: float(latencies.mean()),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "mean_relevant_pool_size": float(detail["relevant_pool_size"].mean()),
        "n_queries": len(per_query),
        "relevance_proxy": RELEVANCE_PROXY,
        "per_query": detail,
    }
