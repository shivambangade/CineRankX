"""Loaders for the raw MovieLens 20M and TMDB 930K CSVs.

Reads from newdata/movielens/ and newdata/tmdb/ (the actual on-disk folder
names for this dataset pair). MovieLens 20M's genome_scores.csv and
genome_tags.csv are intentionally not loaded here — unused by this pipeline.
"""

from pathlib import Path

import pandas as pd

DEFAULT_MOVIELENS_DIR = Path("newdata/movielens")
DEFAULT_TMDB_DIR = Path("newdata/tmdb")


def load_movielens(base_dir: Path = DEFAULT_MOVIELENS_DIR) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    return {
        "ratings": pd.read_csv(base_dir / "rating.csv"),
        "movies": pd.read_csv(base_dir / "movie.csv"),
        "links": pd.read_csv(base_dir / "link.csv"),
        "tags": pd.read_csv(base_dir / "tag.csv"),
    }


def load_tmdb(base_dir: Path = DEFAULT_TMDB_DIR) -> dict[str, pd.DataFrame]:
    base_dir = Path(base_dir)
    return {
        "movies": pd.read_csv(base_dir / "TMDB_movie_dataset_v11.csv"),
    }
