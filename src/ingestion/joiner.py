"""Joins MovieLens movies to TMDB metadata via links.csv, and builds the
per-movie raw text chunk (overview + genres + keywords) used by the IR
engine (Module 2).

TMDB 930K has no cast/crew data (no credits file, no cast/crew columns), so
text_chunk is overview + genres + keywords only — a deliberate scope decision,
not a placeholder. Its genres/keywords are plain comma-separated strings
(unlike TMDB 5000's JSON list-of-dicts), so extraction is a simple split.
"""

import pandas as pd


def _split_names(value: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def join_movielens_tmdb(
    ml: dict[str, pd.DataFrame], tmdb: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, int]]:
    ml_movies = ml["movies"]
    ml_links = ml["links"].copy()
    ml_links["tmdbId"] = pd.to_numeric(ml_links["tmdbId"], errors="coerce")

    merged = ml_movies.merge(ml_links, on="movieId", how="left")
    n_with_tmdb_id = merged["tmdbId"].notna().sum()

    merged = merged.merge(
        tmdb["movies"], left_on="tmdbId", right_on="id", how="inner", suffixes=("", "_tmdb")
    )

    merged["genre_names"] = merged["genres_tmdb" if "genres_tmdb" in merged.columns else "genres"].apply(
        _split_names
    )
    merged["keyword_names"] = merged["keywords"].apply(_split_names)

    merged["overview"] = merged["overview"].fillna("")
    merged["text_chunk"] = (
        merged["overview"]
        + " " + merged["genre_names"].apply(" ".join)
        + " " + merged["keyword_names"].apply(" ".join)
    )

    keep_cols = [
        "movieId",
        "tmdbId",
        "title",
        "genres",
        "genre_names",
        "keyword_names",
        "overview",
        "vote_average",
        "vote_count",
        "popularity",
        "text_chunk",
    ]
    result = merged[keep_cols].reset_index(drop=True)

    report = {
        "movielens_movies": len(ml_movies),
        "movielens_with_tmdb_id": int(n_with_tmdb_id),
        "tmdb_movies": len(tmdb["movies"]),
        "joined_rows": len(result),
    }
    return result, report
