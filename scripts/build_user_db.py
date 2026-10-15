"""Build data/processed/cinerankx.db (ratings, movies, user stats, run history).

Streams newdata/movielens/rating.csv in chunks, so the 20M-row file never sits in memory.

Usage:
    PYTHONPATH=. python scripts/build_user_db.py
"""

from pathlib import Path

import pandas as pd

from src.ingestion.loaders import DEFAULT_MOVIELENS_DIR
from src.storage import DB_PATH, build_database

MERGED = Path("data/processed/movies_merged.csv")


def main() -> None:
    movies = pd.read_csv(DEFAULT_MOVIELENS_DIR / "movie.csv")
    merged = pd.read_csv(MERGED) if MERGED.exists() else None
    chunks = pd.read_csv(DEFAULT_MOVIELENS_DIR / "rating.csv", chunksize=1_000_000)
    stats = build_database(DB_PATH, chunks, movies, merged)
    print(f"Wrote {DB_PATH}: {stats['ratings']:,} ratings, {stats['users']:,} users")


if __name__ == "__main__":
    main()
