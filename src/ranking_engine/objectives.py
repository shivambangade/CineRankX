"""The six ranking objectives, plus the catalog statistics they read from.

Two of the six — diversity and coverage — are *set-dependent*: their value for
a candidate depends on which items have already been selected, so they cannot
be precomputed per item. That is why the ranker selects greedily (see
ranker.py) instead of scoring everything once and sorting. The other four are
per-item and are precomputed here.
"""

import numpy as np
import pandas as pd

_MIN_NOVELTY_COUNT = 1  # unrated items are treated as seen once, not zero times

# `genre_names` in movies_merged.csv round-trips through CSV as a *stringified
# Python list* -- "['Animation', 'Adventure']" -- so a bare comma split leaves
# bracket and quote characters glued to the first and last genre. Left
# unstripped, "['animation'" and "'animation'," count as different genres, which
# silently corrupts both the coverage objective and the genre-Jaccard fallback.
_GENRE_JUNK = "[]'\" "


def _clean_genre(token: str) -> str:
    """Strip list/quote punctuation off one genre token and lowercase it."""
    return token.strip().strip(_GENRE_JUNK).strip().lower()


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A constant array maps to all-zeros (no discrimination)."""
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


class CatalogStats:
    """Per-item statistics for the item-independent objectives, built once.

    Args:
        movies_df: catalog with [movieId, genre_names or genres, vote_average,
            vote_count, popularity].
        ratings_df: optional ratings used for novelty's observed-popularity
            counts. When absent, TMDB `vote_count` is used instead.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame | None = None):
        self.movie_ids = movies_df["movieId"].values
        self._genres = self._build_genres(movies_df)
        self._popularity_quality = self._build_popularity_quality(movies_df)
        self._novelty = self._build_novelty(movies_df, ratings_df)

    @staticmethod
    def _build_genres(movies_df: pd.DataFrame) -> dict[int, frozenset]:
        # `genre_names` (TMDB, comma-separated) is preferred over MovieLens's
        # pipe-separated `genres`; fall back so unit fixtures with only one of
        # the two still work.
        column = "genre_names" if "genre_names" in movies_df.columns else "genres"
        if column not in movies_df.columns:
            return {}

        genres = {}
        for movie_id, raw in zip(movies_df["movieId"], movies_df[column]):
            if not isinstance(raw, str) or not raw.strip():
                genres[int(movie_id)] = frozenset()
                continue
            parts = raw.replace("|", ",").split(",")
            genres[int(movie_id)] = frozenset(
                cleaned for cleaned in (_clean_genre(p) for p in parts) if cleaned
            )
        return genres

    @staticmethod
    def _build_popularity_quality(movies_df: pd.DataFrame) -> dict[int, float]:
        """Bayesian-weighted rating, min-max scaled to [0, 1].

        A raw `vote_average` lets a movie with three 10/10 votes outrank a
        classic rated 8.4 by 20,000 people, so votes are shrunk toward the
        catalog mean by vote count. Min-max scaling is applied afterwards
        because the Bayesian scores themselves bunch into a narrow band (~5-7
        on TMDB's 0-10 scale); dividing by 10 would leave this objective with
        almost no spread to contribute against the other five.
        """
        if "vote_average" not in movies_df.columns or "vote_count" not in movies_df.columns:
            return {}

        votes = movies_df["vote_count"].fillna(0).to_numpy(dtype=float)
        averages = movies_df["vote_average"].fillna(0).to_numpy(dtype=float)

        rated = votes > 0
        # The prior is the UNWEIGHTED mean across movies that have votes -- the
        # rating of a typical *movie*. A vote-count-weighted mean would instead
        # be the rating of a typical *vote*, which blockbusters dominate and
        # which pulls the prior well above the catalog's centre; thinly-voted
        # movies would then shrink toward that inflated prior and could outrank
        # heavily-voted ones, inverting the point of the shrinkage. This is the
        # standard IMDB weighted-rating formulation.
        global_mean = float(np.mean(averages[rated])) if rated.any() else 0.0
        min_votes = float(np.median(votes[rated])) if rated.any() else 0.0

        denominator = votes + min_votes
        with np.errstate(invalid="ignore", divide="ignore"):
            bayesian = np.where(
                denominator > 0,
                (votes / denominator) * averages + (min_votes / denominator) * global_mean,
                global_mean,
            )
        return dict(zip(movies_df["movieId"].astype(int), _minmax(bayesian)))

    @staticmethod
    def _build_novelty(movies_df: pd.DataFrame, ratings_df: pd.DataFrame | None) -> dict[int, float]:
        """Self-information -log2(p(item)), min-max scaled to [0, 1].

        Novelty is the standard information-theoretic "how unexpected is this
        item" measure: the rarer an item is in the observed interaction stream,
        the higher its novelty. Deliberately in tension with
        popularity_quality — resolving that tension is exactly what the weights
        are for.
        """
        movie_ids = movies_df["movieId"].astype(int)

        # Observed engagement = MovieLens rating count + TMDB vote count. Both
        # are "how many people engaged with this item", and combining them keeps
        # the measure defined across the WHOLE catalog. Rating counts alone are
        # not enough: any run over a subset of the 20M ratings leaves most of
        # the catalog at zero, and after the floor + min-max below those all tie
        # at novelty 1.0 (15,481 of 26,743 movies on a 1M-rating sample),
        # flattening the objective into a near-constant and letting genuine
        # obscurities and merely-unsampled films score identically.
        observed = np.zeros(len(movie_ids), dtype=float)
        if ratings_df is not None and not ratings_df.empty:
            counts = ratings_df["movieId"].value_counts()
            observed += movie_ids.map(counts).fillna(0).to_numpy(dtype=float)
        if "vote_count" in movies_df.columns:
            observed += movies_df["vote_count"].fillna(0).to_numpy(dtype=float)
        if not observed.any():
            return {}

        observed = np.maximum(observed, _MIN_NOVELTY_COUNT)
        total = observed.sum()
        self_information = -np.log2(observed / total)
        return dict(zip(movie_ids, _minmax(self_information)))

    def genres(self, movie_id: int) -> frozenset:
        return self._genres.get(int(movie_id), frozenset())

    def popularity_quality(self, movie_id: int) -> float:
        """Objective 5: quality weighted by how many people actually voted."""
        return self._popularity_quality.get(int(movie_id), 0.0)

    def novelty(self, movie_id: int) -> float:
        """Objective 3: how far off the beaten path this item is."""
        return self._novelty.get(int(movie_id), 0.0)


def diversity_score(movie_id: int, selected_ids: list[int], similarity: dict) -> float:
    """Objective 2: how unlike the already-selected items this candidate is.

    Defined as 1 - max similarity to anything already in the list (the MMR
    formulation). Using the *max* rather than the mean is deliberate: a
    candidate that is a near-duplicate of one selected item is redundant even
    if it is unlike the other nine, and a mean would dilute that away.

    An empty selection scores 1.0 — the first pick cannot be redundant.
    """
    if not selected_ids:
        return 1.0
    sims = [similarity.get((int(movie_id), int(other)), 0.0) for other in selected_ids]
    return float(1.0 - max(sims))


def coverage_score(movie_id: int, covered_genres: set, stats: CatalogStats) -> float:
    """Objective 4: how much of the catalog's genre space this item opens up.

    Per-item proxy for the catalog-Coverage metric: the fraction of this
    item's genres not already represented in the selected list. An item whose
    genres are all already covered adds nothing new and scores 0.0; an item
    with no genre metadata also scores 0.0, since it cannot be shown to
    broaden coverage.
    """
    item_genres = stats.genres(movie_id)
    if not item_genres:
        return 0.0
    new_genres = item_genres - covered_genres
    return len(new_genres) / len(item_genres)
