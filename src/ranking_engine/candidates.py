"""Candidate generation: blends Module 2 (IR) and Module 3 (strategy) output.

Module 3's documented guidance (DATASET_MIGRATION.md) is that the predicted
strategy should be treated as a *weighted prior*, not a hard routing decision,
and should degrade toward `collaborative` when classifier confidence is low.
That is implemented here: every candidate source always contributes, and the
predicted strategy only shifts how much each one contributes.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.ranking_engine.config import load_genre_gate_floor
from src.ranking_engine.objectives import CatalogStats

SOURCES = ("content_based", "collaborative", "popularity")

# Where source weights sit when the classifier tells us nothing useful. Not
# uniform: `collaborative` is the majority strategy by a wide margin (~84% of
# users in Module 3's labels), so leaning on it is the correct low-information
# default rather than an arbitrary preference.
_FALLBACK_PRIOR = {"collaborative": 0.60, "content_based": 0.20, "popularity": 0.20}
_CHANCE_CONFIDENCE = 1.0 / len(SOURCES)  # confidence at/below this carries no information

_CF_MAX_USERS = 5000       # bounds the neighbor pool scanned per user
_CF_MAX_MOVIES = 2000      # bounds user-item matrix width (most-rated movies)
_CF_NEIGHBORS = 25
_CF_PER_NEIGHBOR = 20
_IR_SEEDS = 10             # user's top-rated movies used as IR query seeds
_POPULARITY_POOL = 200


def strategy_source_weights(strategy: str | None, confidence: float = 1.0) -> dict[str, float]:
    """Blend the fallback prior toward the predicted strategy by confidence.

    At chance-level confidence the result is the fallback prior unchanged; at
    confidence 1.0 it is fully the predicted strategy. In between it
    interpolates, so a barely-confident prediction nudges the mix instead of
    swinging it. An unrecognized or missing strategy returns the prior as-is.
    """
    if strategy not in SOURCES:
        return dict(_FALLBACK_PRIOR)

    strength = (float(confidence) - _CHANCE_CONFIDENCE) / (1.0 - _CHANCE_CONFIDENCE)
    strength = float(np.clip(strength, 0.0, 1.0))

    return {
        source: (1.0 - strength) * _FALLBACK_PRIOR[source] + strength * (1.0 if source == strategy else 0.0)
        for source in SOURCES
    }


def genre_overlap_gate(candidate_id: int, seed_id: int, stats: CatalogStats, floor: float) -> float:
    """Multiplier in [floor, 1.0] scaling a candidate's IR relevance by genre fit.

    TF-IDF matches on shared vocabulary, which is not the same as shared tone:
    querying neighbours of Toy Story surfaces "Silent Night, Deadly Night 5: The
    Toy Maker" and "Dollman vs. Demonic Toys" because they talk about toys, not
    because anyone who liked one would like the other. Genre overlap is the
    cheap, explainable signal that separates those cases, so the candidate's
    relevance is scaled by its Jaccard overlap with the seed it came from.

    Returns 1.0 (no penalty) when either side has no genre metadata: an unknown
    genre is not evidence of a mismatch, and penalising it would push every
    sparsely-tagged movie out of the catalog.
    """
    seed_genres = stats.genres(seed_id)
    candidate_genres = stats.genres(candidate_id)
    if not seed_genres or not candidate_genres:
        return 1.0

    overlap = len(seed_genres & candidate_genres) / len(seed_genres | candidate_genres)
    return floor + (1.0 - floor) * overlap


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max a source's raw scores into [0, 1] so sources are comparable.

    Different sources produce scores on incompatible scales (cosine similarity,
    neighbor-similarity mass, TMDB popularity in the hundreds). Without this,
    the source weights would be meaningless — popularity's raw magnitude alone
    would swamp everything.
    """
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo <= 0:
        return {mid: 1.0 for mid in scores}
    return {mid: float((score - lo) / (hi - lo)) for mid, score in scores.items()}


class CandidateGenerator:
    """Builds a per-user candidate pool from all three strategy sources."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        ir_engine=None,
        stats: CatalogStats | None = None,
        genre_gate_floor: float | None = None,
    ):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.ir_engine = ir_engine
        # Reuse the ranker's CatalogStats when one is supplied -- rebuilding it
        # here would parse every genre string in the catalog a second time.
        self.stats = stats if stats is not None else CatalogStats(movies_df, ratings_df)
        self.genre_gate_floor = (
            genre_gate_floor if genre_gate_floor is not None else load_genre_gate_floor()
        )
        self._build_popularity()
        self._build_collaborative()

    def _build_popularity(self) -> None:
        if "popularity" in self.movies_df.columns:
            top = self.movies_df.nlargest(_POPULARITY_POOL, "popularity")
            self._popular = dict(zip(top["movieId"].astype(int), top["popularity"].astype(float)))
            return
        # Fallback for callers without TMDB metadata: rating volume, which is
        # the same "how many people engaged with this" signal popularity encodes.
        counts = self.ratings_df["movieId"].value_counts().head(_POPULARITY_POOL)
        self._popular = {int(k): float(v) for k, v in counts.items()}

    def _build_collaborative(self) -> None:
        self._user_item = None
        self._cf_user_ids = np.array([])
        self._movie_to_col: dict[int, int] = {}

        user_pool = self.ratings_df["userId"].unique()[:_CF_MAX_USERS]
        sub = self.ratings_df[self.ratings_df["userId"].isin(user_pool)]
        if sub.empty or sub["userId"].nunique() < 2:
            return

        top_movies = sub["movieId"].value_counts().nlargest(_CF_MAX_MOVIES).index
        sub = sub[sub["movieId"].isin(top_movies)]
        if sub.empty:
            return

        self._cf_user_ids = sub["userId"].unique()
        user_to_row = {uid: i for i, uid in enumerate(self._cf_user_ids)}
        self._movie_to_col = {int(mid): i for i, mid in enumerate(top_movies)}

        self._user_item = csr_matrix(
            (
                sub["rating"].values,
                (sub["userId"].map(user_to_row).values, sub["movieId"].map(self._movie_to_col).values),
            ),
            shape=(len(self._cf_user_ids), len(top_movies)),
        )
        # Neighbors' highly-rated movies come from their FULL history, not the
        # bounded column set used for similarity — restricting candidates to the
        # most-rated movies too would make this source a second popularity
        # source rather than a genuinely collaborative one.
        self._neighbor_top = (
            self.ratings_df[self.ratings_df["userId"].isin(self._cf_user_ids)]
            .sort_values("rating", ascending=False)
            .groupby("userId")["movieId"]
            .apply(lambda s: list(s.head(_CF_PER_NEIGHBOR)))
            .to_dict()
        )

    def _content_candidates(self, user_ratings: pd.DataFrame, limit: int) -> dict[int, float]:
        """IR engine (Module 2) similarity to the user's best-loved movies."""
        if self.ir_engine is None or user_ratings.empty:
            return {}

        seeds = user_ratings.nlargest(_IR_SEEDS, "rating")
        scores: dict[int, float] = {}
        for movie_id, rating in zip(seeds["movieId"], seeds["rating"]):
            seed_id = int(movie_id)
            for hit in self.ir_engine.similar_to(seed_id, limit=limit):
                mid = int(hit["movieId"])
                # Weight each neighbour by how much the user liked its seed, so
                # a 5-star seed's neighbours outrank a 3-star seed's, then gate
                # on genre fit so a keyword-only match cannot ride a 5-star seed
                # to the top of the pool.
                gate = genre_overlap_gate(mid, seed_id, self.stats, self.genre_gate_floor)
                weighted = hit["similarity"] * float(rating) * gate
                scores[mid] = max(scores.get(mid, 0.0), weighted)
        return scores

    def _collaborative_candidates(self, user_id: int, user_ratings: pd.DataFrame) -> dict[int, float]:
        """User-user KNN over the bounded user-item matrix."""
        if self._user_item is None or user_ratings.empty:
            return {}

        # Build the target user's vector from their own ratings rather than
        # looking up a row, so users outside the sampled pool (and brand-new
        # users) still get collaborative candidates instead of nothing.
        columns, values = [], []
        for movie_id, rating in zip(user_ratings["movieId"], user_ratings["rating"]):
            col = self._movie_to_col.get(int(movie_id))
            if col is not None:
                columns.append(col)
                values.append(float(rating))
        if not columns:
            return {}

        user_vector = csr_matrix(
            (values, ([0] * len(columns), columns)), shape=(1, self._user_item.shape[1])
        )
        sims = cosine_similarity(user_vector, self._user_item)[0]
        # A perfect self-match means this user IS one of the pooled rows;
        # recommending from themselves would just echo their own history back.
        sims[np.isclose(sims, 1.0)] = -1.0

        neighbors = np.argsort(sims)[::-1][:_CF_NEIGHBORS]
        scores: dict[int, float] = {}
        for idx in neighbors:
            similarity = float(sims[idx])
            if similarity <= 0:
                continue
            for movie_id in self._neighbor_top.get(self._cf_user_ids[idx], []):
                mid = int(movie_id)
                scores[mid] = scores.get(mid, 0.0) + similarity
        return scores

    def candidates_for_seed(self, seed_movie_id: int, limit: int = 25) -> pd.DataFrame:
        """Gated IR neighbours of one movie, as a ready-to-rank candidate pool.

        Exists so a caller inspecting a single seed (e.g. "what is similar to
        Toy Story?") goes through the same gating path as real recommendations
        instead of reimplementing it.
        """
        if self.ir_engine is None:
            return pd.DataFrame(columns=["movieId", "relevance", "sources"])

        hits = self.ir_engine.similar_to(int(seed_movie_id), limit=limit)
        rows = []
        for hit in hits:
            mid = int(hit["movieId"])
            gate = genre_overlap_gate(mid, int(seed_movie_id), self.stats, self.genre_gate_floor)
            rows.append({"movieId": mid, "relevance": hit["similarity"] * gate, "sources": "content_based"})
        if not rows:
            return pd.DataFrame(columns=["movieId", "relevance", "sources"])

        pool = pd.DataFrame(rows)
        pool["relevance"] = pool["relevance"] / pool["relevance"].max()
        return pool.sort_values("relevance", ascending=False).reset_index(drop=True)

    def generate(
        self,
        user_id: int,
        strategy: str | None = None,
        confidence: float = 1.0,
        pool_size: int = 200,
    ) -> pd.DataFrame:
        """Produce a blended candidate pool for one user.

        Returns:
            DataFrame [movieId, relevance, sources] sorted by relevance
            descending, at most `pool_size` rows. `relevance` is objective 1,
            already scaled to [0, 1]. `sources` records which generators
            proposed the item, so a ranked result stays explainable.
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        already_rated = set(user_ratings["movieId"].astype(int))
        source_weights = strategy_source_weights(strategy, confidence)

        per_source = {
            "content_based": self._content_candidates(user_ratings, limit=30),
            "collaborative": self._collaborative_candidates(user_id, user_ratings),
            "popularity": dict(self._popular),
        }

        blended: dict[int, float] = {}
        sources: dict[int, list[str]] = {}
        for source, raw_scores in per_source.items():
            weight = source_weights[source]
            if weight <= 0:
                continue
            for movie_id, score in _normalize(raw_scores).items():
                if movie_id in already_rated:
                    continue
                # Summing across sources (rather than taking a max) means an
                # item proposed by two independent sources beats one proposed
                # by a single source at the same strength -- cross-source
                # agreement is real evidence.
                blended[movie_id] = blended.get(movie_id, 0.0) + weight * score
                sources.setdefault(movie_id, []).append(source)

        if not blended:
            return pd.DataFrame(columns=["movieId", "relevance", "sources"])

        normalized = _normalize(blended)
        candidates = pd.DataFrame(
            {
                "movieId": list(normalized.keys()),
                "relevance": list(normalized.values()),
                "sources": ["+".join(sources[mid]) for mid in normalized],
            }
        )
        return candidates.sort_values("relevance", ascending=False).head(pool_size).reset_index(drop=True)
