"""Extract user profile features from rating history (vectorized over all users)."""

import numpy as np
import pandas as pd

# MovieLens genre vocabulary: a fixed list keeps the affinity columns identical
# between training and serving, whatever genres a given user happens to have.
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
          "Sci-Fi", "Thriller", "War", "Western"]
NICHE_PERCENTILE = 0.5  # a title is "niche" if it is in the less-popular half of the catalog
RECENT_SPAN_SHARE = 0.2  # "recent" = the last 20% of a user's own active time span


def _parse_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


_CATALOG_CACHE: dict = {}


def _catalog_tables(movies_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(movieId -> genres, movieId -> popularity percentile), built once per catalog object.

    Rebuilding them on every call dominated single-user inference time (a 27k-row merge and
    rank per prediction); the cache key includes the catalog's identity and shape.
    """
    key = (id(movies_df), movies_df.shape)
    if key not in _CATALOG_CACHE:
        _CATALOG_CACHE.clear()
        unique = movies_df.drop_duplicates("movieId")
        genres = (unique.set_index("movieId")["genres"] if "genres" in unique.columns
                  else pd.Series(dtype=object))
        _CATALOG_CACHE[key] = (genres, _popularity_percentile(movies_df))
    return _CATALOG_CACHE[key]


def _popularity_percentile(movies_df: pd.DataFrame) -> pd.Series:
    """Catalog popularity percentile per movieId (higher = more popular), if a signal exists."""
    for col in ("vote_count", "popularity"):
        if col in movies_df.columns:
            values = pd.to_numeric(movies_df.set_index("movieId")[col], errors="coerce")
            values = values[~values.index.duplicated()]
            return values.rank(pct=True)
    return pd.Series(dtype=float)


def _genre_features(rated: pd.DataFrame, n_ratings: pd.Series) -> pd.DataFrame:
    """Legacy genre columns plus per-genre affinity (share of a user's ratings in each genre)."""
    exploded = rated[["userId", "movieId", "genres"]].dropna(subset=["genres"]).copy()
    exploded["genre"] = exploded["genres"].astype(str).str.split("|")
    exploded = exploded.explode("genre")
    exploded["genre"] = exploded["genre"].str.strip()
    exploded = exploded[exploded["genre"] != ""]
    counts = exploded.groupby(["userId", "genre"]).size().rename("n").reset_index()
    out = pd.DataFrame(index=n_ratings.index)
    totals = counts.groupby("userId")["n"].sum()
    num_genres = counts.groupby("userId")["genre"].nunique()
    top_share = counts.groupby("userId")["n"].max() / totals
    props = counts["n"] / counts["userId"].map(totals)
    entropy = (-(props * np.log(props))).groupby(counts["userId"]).sum()
    max_entropy = np.log(num_genres.where(num_genres > 1, 1.0))
    out["num_genres"] = num_genres.reindex(out.index).fillna(0).astype(int)
    out["genre_diversity"] = out["num_genres"] / n_ratings.clip(lower=1)
    out["top_genre_share"] = top_share.reindex(out.index).fillna(0.0)
    out["genre_entropy"] = (entropy / max_entropy.where(max_entropy > 0)).reindex(out.index).fillna(0.0)
    # Affinity: fraction of the user's ratings whose movie carries the genre.
    pivot = counts[counts["genre"].isin(GENRES)].pivot_table(
        index="userId", columns="genre", values="n", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=GENRES, fill_value=0).reindex(out.index, fill_value=0)
    affinity = pivot.div(n_ratings.clip(lower=1), axis=0).clip(upper=1.0)
    affinity.columns = [f"affinity_{g}" for g in GENRES]
    return out.join(affinity)


def _time_features(user_ratings: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=index, columns=["log_active_span_days", "recent_activity_share"])
    if "timestamp" not in user_ratings.columns:
        return out
    frame = user_ratings[["userId"]].assign(t=_parse_time(user_ratings["timestamp"])).dropna()
    if frame.empty:
        return out
    grouped = frame.groupby("userId")["t"]
    first, last = grouped.transform("min"), grouped.transform("max")
    span = (last - first).dt.total_seconds()
    cutoff = last - pd.to_timedelta(span * RECENT_SPAN_SHARE, unit="s")
    recent = (frame["t"] >= cutoff).groupby(frame["userId"]).mean()
    span_days = span.groupby(frame["userId"]).max() / 86400.0
    out["log_active_span_days"] = np.log1p(span_days).reindex(index).fillna(0.0)
    # A single-burst user (zero span) has all activity "recent" by definition; report 0
    # instead so the feature separates sustained from one-sitting raters.
    out["recent_activity_share"] = recent.where(span_days > 0, 0.0).reindex(index).fillna(0.0)
    return out


def extract_profile_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for each user from their rating history and movie metadata.

    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
            (timestamp optional)
        movies_df: DataFrame with columns [movieId, genres, ...]; vote_count or
            popularity, when present, drives the niche/popularity features.

    Returns:
        One row per user. Original columns: userId, num_ratings, avg_rating,
        rating_std, rating_min, rating_max, num_genres, genre_diversity,
        top_genre_share, genre_entropy (taste concentration, 0 = one genre,
        1 = even spread). Added columns: rating_variance, log_num_ratings,
        affinity_<Genre> (share of ratings in each genre), niche_fraction,
        mean_popularity_pct, log_active_span_days, recent_activity_share.
        Everything is computed from the ratings passed in, so callers control
        exactly which ratings a profile may see.
    """
    if ratings_df.empty or movies_df.empty:
        return pd.DataFrame()

    ratings = ratings_df.copy()
    genre_map, pct = _catalog_tables(movies_df)
    rated = ratings.copy()
    rated["genres"] = rated["movieId"].map(genre_map) if len(genre_map) else np.nan

    by_user = ratings.groupby("userId")["rating"]
    n_ratings = by_user.size()
    feats = pd.DataFrame({
        "num_ratings": n_ratings,
        "avg_rating": by_user.mean(),
        "rating_std": by_user.std().fillna(0.0),
        "rating_min": by_user.min(),
        "rating_max": by_user.max(),
    })
    feats["rating_variance"] = feats["rating_std"] ** 2
    feats["log_num_ratings"] = np.log1p(n_ratings)
    feats = feats.join(_genre_features(rated, n_ratings))

    feats["niche_fraction"] = 0.0
    feats["mean_popularity_pct"] = 0.0
    if not pct.empty:
        rated_pct = rated["movieId"].map(pct)
        keyed = rated_pct.groupby(rated["userId"])
        feats["mean_popularity_pct"] = keyed.mean().reindex(feats.index).fillna(0.0)
        niche = (rated_pct < NICHE_PERCENTILE).where(rated_pct.notna()).astype(float)
        feats["niche_fraction"] = niche.groupby(rated["userId"]).mean().reindex(feats.index).fillna(0.0).astype(float)

    feats = feats.join(_time_features(ratings, feats.index))
    ordered = ["num_ratings", "avg_rating", "rating_std", "rating_min", "rating_max", "num_genres",
               "genre_diversity", "top_genre_share", "genre_entropy"]
    rest = [c for c in feats.columns if c not in ordered]
    return feats[ordered + rest].reset_index().rename(columns={"index": "userId"})
