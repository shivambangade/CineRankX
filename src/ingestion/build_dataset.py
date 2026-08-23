"""CLI entrypoint: loads raw MovieLens + TMDB data, joins them, cleans text,
and caches the results to data/processed/ so downstream modules never
recompute the join.

Usage: python -m src.ingestion.build_dataset
"""

from pathlib import Path

from src.ingestion.joiner import join_movielens_tmdb
from src.ingestion.loaders import load_movielens, load_tmdb
from src.ingestion.text_cleaner import clean_text

PROCESSED_DIR = Path("data/processed")


def build() -> None:
    ml = load_movielens()
    tmdb = load_tmdb()

    merged, report = join_movielens_tmdb(ml, tmdb)
    merged["text_chunk_clean"] = merged["text_chunk"].apply(clean_text)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED_DIR / "movies_merged.csv", index=False)
    ml["ratings"].to_csv(PROCESSED_DIR / "ratings_clean.csv", index=False)

    print("Join report:")
    print(f"  MovieLens movies:           {report['movielens_movies']}")
    print(f"  MovieLens movies w/ tmdbId: {report['movielens_with_tmdb_id']}")
    print(f"  TMDB movies available:      {report['tmdb_movies']}")
    print(f"  Successfully joined rows:   {report['joined_rows']}")
    print(f"Wrote {PROCESSED_DIR / 'movies_merged.csv'} and {PROCESSED_DIR / 'ratings_clean.csv'}")


if __name__ == "__main__":
    build()
