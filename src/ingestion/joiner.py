"""Joins MovieLens movies to TMDB metadata via links.csv, and builds the
per-movie raw text chunk (overview + genres + keywords + cast + crew) used
by the IR engine (Module 2).
"""

import json

import pandas as pd

TOP_N_CAST = 5


def _parse_json_list(value: str) -> list[dict]:
    if not isinstance(value, str) or not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_names(json_str: str, limit: int | None = None) -> list[str]:
    items = _parse_json_list(json_str)
    names = [item.get("name", "") for item in items if item.get("name")]
    return names[:limit] if limit else names


def _extract_directors(crew_json_str: str) -> list[str]:
    items = _parse_json_list(crew_json_str)
    return [item.get("name", "") for item in items if item.get("job") == "Director"]


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
    merged = merged.merge(
        tmdb["credits"], left_on="tmdbId", right_on="movie_id", how="left", suffixes=("", "_credits")
    )

    merged["genre_names"] = merged["genres_tmdb" if "genres_tmdb" in merged.columns else "genres"].apply(
        _extract_names
    )
    merged["keyword_names"] = merged["keywords"].apply(_extract_names)
    merged["cast_names"] = merged["cast"].apply(lambda v: _extract_names(v, limit=TOP_N_CAST))
    merged["crew_names"] = merged["crew"].apply(_extract_directors)

    merged["overview"] = merged["overview"].fillna("")
    merged["text_chunk"] = (
        merged["overview"]
        + " " + merged["genre_names"].apply(" ".join)
        + " " + merged["keyword_names"].apply(" ".join)
        + " " + merged["cast_names"].apply(" ".join)
        + " " + merged["crew_names"].apply(" ".join)
    )

    keep_cols = [
        "movieId",
        "tmdbId",
        "title",
        "genres",
        "genre_names",
        "keyword_names",
        "cast_names",
        "crew_names",
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
