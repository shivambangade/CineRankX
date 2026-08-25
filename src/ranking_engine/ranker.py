"""Multi-Objective Hybrid Ranking Engine (Module 4).

Re-ranks a blended candidate pool on six weighted objectives. Selection is
*greedy and sequential*, not score-then-sort: diversity and coverage are
defined relative to what has already been chosen, so an item's score is not
fixed until the items above it are. At each step every remaining candidate is
rescored against the current selection and the best-scoring one is taken.
"""

import numpy as np
import pandas as pd

from src.ranking_engine.candidates import CandidateGenerator
from src.ranking_engine.config import OBJECTIVES, RankingWeights
from src.ranking_engine.objectives import CatalogStats, coverage_score, diversity_score
from src.ranking_engine.predictor import BaselinePredictor


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MultiObjectiveRanker:
    """Ranks candidates on relevance, diversity, novelty, coverage,
    popularity/quality and predicted rating, combined by tunable weights."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        ir_engine=None,
        weights: RankingWeights | None = None,
        predictor: BaselinePredictor | None = None,
    ):
        self.movies_df = movies_df
        self.ir_engine = ir_engine
        self.weights = weights if weights is not None else RankingWeights.from_config()
        self.stats = CatalogStats(movies_df, ratings_df)
        self.predictor = predictor if predictor is not None else BaselinePredictor().fit(ratings_df)
        self.generator = CandidateGenerator(
            movies_df, ratings_df, ir_engine=ir_engine, stats=self.stats
        )
        self.titles = dict(zip(movies_df["movieId"].astype(int), movies_df["title"]))

    def _similarity_lookup(self, movie_ids: list[int]) -> dict:
        """Pairwise similarity among candidates, for the diversity objective.

        Prefers the Module 2 TF-IDF vectors (content similarity over overview +
        genres + keywords). Pairs where either item is missing from the TF-IDF
        matrix fall back to genre Jaccard, so an item without a TF-IDF row is
        not silently treated as maximally diverse from everything.
        """
        rows = {}
        if self.ir_engine is not None and getattr(self.ir_engine.tfidf, "matrix", None) is not None:
            index = {int(mid): i for i, mid in enumerate(self.ir_engine.tfidf.movie_ids)}
            rows = {mid: index[mid] for mid in movie_ids if mid in index}

        similarity: dict = {}
        if rows:
            from sklearn.metrics.pairwise import cosine_similarity

            ordered = list(rows.keys())
            matrix = cosine_similarity(self.ir_engine.tfidf.matrix[[rows[m] for m in ordered]])
            for i, a in enumerate(ordered):
                for j, b in enumerate(ordered):
                    if i != j:
                        similarity[(a, b)] = float(matrix[i, j])

        missing = [mid for mid in movie_ids if mid not in rows]
        for a in missing:
            for b in movie_ids:
                if a == b:
                    continue
                value = _jaccard(self.stats.genres(a), self.stats.genres(b))
                similarity[(a, b)] = value
                similarity[(b, a)] = value
        return similarity

    def rank(
        self,
        user_id: int,
        k: int = 10,
        strategy: str | None = None,
        confidence: float = 1.0,
        candidates: pd.DataFrame | None = None,
        weights: RankingWeights | None = None,
        pool_size: int = 200,
    ) -> pd.DataFrame:
        """Return the top-k re-ranked recommendations for one user.

        Args:
            candidates: optional precomputed pool [movieId, relevance]; when
                omitted, generated from the blended IR + strategy sources.
            weights: optional per-call weight override, for sensitivity runs.

        Returns:
            DataFrame [rank, movieId, title, score, sources, <one column per
            objective>]. The per-objective columns are the values that actually
            went into `score` at the moment the item was selected — for
            diversity and coverage that is their value against the items ranked
            above them, which is what makes the breakdown auditable.
        """
        active_weights = weights if weights is not None else self.weights

        if candidates is None:
            candidates = self.generator.generate(
                user_id, strategy=strategy, confidence=confidence, pool_size=pool_size
            )
        if candidates.empty:
            return pd.DataFrame(columns=["rank", "movieId", "title", "score", "sources", *OBJECTIVES])

        movie_ids = [int(mid) for mid in candidates["movieId"]]
        relevance = dict(zip(movie_ids, candidates["relevance"].astype(float)))
        sources = dict(zip(movie_ids, candidates.get("sources", pd.Series([""] * len(candidates)))))
        similarity = self._similarity_lookup(movie_ids)

        # Item-independent objectives: fixed regardless of selection order.
        fixed = {
            mid: {
                "relevance": relevance[mid],
                "novelty": self.stats.novelty(mid),
                "popularity_quality": self.stats.popularity_quality(mid),
                "predicted_rating": self.predictor.predict_normalized(user_id, mid),
            }
            for mid in movie_ids
        }

        remaining = list(movie_ids)
        selected: list[int] = []
        covered_genres: set = set()
        rows = []

        while remaining and len(selected) < k:
            best_id, best_score, best_scores = None, -np.inf, None
            for mid in remaining:
                scores = dict(fixed[mid])
                scores["diversity"] = diversity_score(mid, selected, similarity)
                scores["coverage"] = coverage_score(mid, covered_genres, self.stats)
                total = sum(active_weights[name] * scores[name] for name in OBJECTIVES)
                if total > best_score:
                    best_id, best_score, best_scores = mid, total, scores

            remaining.remove(best_id)
            selected.append(best_id)
            covered_genres |= self.stats.genres(best_id)
            rows.append(
                {
                    "rank": len(selected),
                    "movieId": best_id,
                    "title": self.titles.get(best_id, ""),
                    "score": float(best_score),
                    "sources": sources.get(best_id, ""),
                    **{name: float(best_scores[name]) for name in OBJECTIVES},
                }
            )

        return pd.DataFrame(rows)

    def recommend(self, user_id: int, k: int = 10, strategy: str | None = None,
                  confidence: float = 1.0) -> list[int]:
        """Convenience wrapper returning just the ranked movieIds."""
        ranked = self.rank(user_id, k=k, strategy=strategy, confidence=confidence)
        return [int(mid) for mid in ranked["movieId"]] if not ranked.empty else []
