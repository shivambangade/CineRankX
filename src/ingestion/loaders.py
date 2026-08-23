"""Loaders for the raw MovieLens (ml-latest-small) and TMDB 5000 CSVs.

Per the build plan's Decision #0, these read directly from data/MovieLens/
and data/tmdb/ (the actual on-disk folder names), not data/raw/.
"""

from pathlib import Path

import pandas as pd

DEFAULT_MOVIELENS_DIR = Path("data/MovieLens")
DEFAULT_TMDB_DIR = Path("data/tmdb")


def load_movielens(base_dir: Path = DEFAULT_MOVIELENS_DIR) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    return {
        "ratings": pd.read_csv(base_dir / "ratings.csv"),
        "movies": pd.read_csv(base_dir / "movies.csv"),
        "links": pd.read_csv(base_dir / "links.csv"),
        "tags": pd.read_csv(base_dir / "tags.csv"),
    }


def load_tmdb(base_dir: Path = DEFAULT_TMDB_DIR) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    return {
        "movies": pd.read_csv(base_dir / "tmdb_5000_movies.csv"),
        "credits": pd.read_csv(base_dir / "tmdb_5000_credits.csv"),
    }
