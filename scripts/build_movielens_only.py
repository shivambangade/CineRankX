"""Build data/processed/ from MovieLens alone (no TMDB), for local demos.

The full pipeline (src.ingestion.build_dataset) needs the TMDB 2023 CSV, which
requires a Kaggle login. This builds the same artifacts from MovieLens 20M only:
overview is empty, keywords come from user tags, and popularity/quality columns
are derived from the ratings themselves.

Usage:
    PYTHONPATH=. python scripts/build_movielens_only.py [n_ratings]
"""

import sys
from pathlib import Path

import pandas as pd

from src.ingestion.loaders import DEFAULT_MOVIELENS_DIR
from src.ingestion.text_cleaner import clean_text

N_RATINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
OUT = Path("data/processed")
TOP_TAGS = 8


def main() -> None:
    base = DEFAULT_MOVIELENS_DIR

    def read(name: str, **kw) -> pd.DataFrame:  # 20M uses singular names, "small" plural
        path = base / f"{name}.csv"
        return pd.read_csv(path if path.exists() else base / f"{name}s.csv", **kw)

    ratings = read("rating", nrows=N_RATINGS)
    movies = read("movie")
    links = read("link")
    tags = read("tag", usecols=["movieId", "tag"])

    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    movies = movies.merge(stats, left_on="movieId", right_index=True, how="inner")

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    top_tags = (tags.groupby("movieId")["tag"]
                .agg(lambda s: list(s.value_counts().index[:TOP_TAGS])))
    movies = movies.merge(top_tags.rename("keyword_names"), left_on="movieId",
                          right_index=True, how="left")
    movies["keyword_names"] = movies["keyword_names"].apply(lambda v: v if isinstance(v, list) else [])
    movies["genre_names"] = movies["genres"].apply(
        lambda g: [x for x in str(g).split("|") if x and x != "(no genres listed)"])

    movies["overview"] = ""
    movies["vote_average"] = movies["mean"] * 2  # MovieLens 0.5-5 -> TMDB-style 0-10
    movies["vote_count"] = movies["count"]
    movies["popularity"] = movies["count"].astype(float)
    movies["text_chunk"] = (movies["genre_names"].apply(" ".join) + " "
                            + movies["keyword_names"].apply(" ".join)).str.strip()
    movies["text_chunk_clean"] = movies["text_chunk"].apply(clean_text)

    cols = ["movieId", "tmdbId", "title", "genres", "genre_names", "keyword_names", "overview",
            "vote_average", "vote_count", "popularity", "text_chunk", "text_chunk_clean"]
    OUT.mkdir(parents=True, exist_ok=True)
    movies[cols].reset_index(drop=True).to_csv(OUT / "movies_merged.csv", index=False)
    ratings.to_csv(OUT / "ratings_clean.csv", index=False)
    print(f"Wrote {len(movies):,} movies and {len(ratings):,} ratings to {OUT}/")


if __name__ == "__main__":
    main()
