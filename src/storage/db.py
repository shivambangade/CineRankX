"""SQLite schema and bulk ingestion for ratings, movies, per-user stats and run history."""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/processed/cinerankx.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    movieId INTEGER PRIMARY KEY, title TEXT NOT NULL, year INTEGER, genres TEXT,
    original_language TEXT, release_date TEXT, poster_path TEXT);
CREATE TABLE IF NOT EXISTS ratings (
    userId INTEGER NOT NULL, movieId INTEGER NOT NULL, rating REAL NOT NULL, ts TEXT);
CREATE TABLE IF NOT EXISTS user_stats (
    userId INTEGER PRIMARY KEY, n_ratings INTEGER, avg_rating REAL, first_ts TEXT, last_ts TEXT);
CREATE TABLE IF NOT EXISTS recommendation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, userId INTEGER NOT NULL, created_at TEXT NOT NULL,
    source TEXT, strategy TEXT, confidence REAL, k INTEGER, weights TEXT, status TEXT NOT NULL,
    error TEXT, elapsed_ms REAL, results TEXT);
CREATE INDEX IF NOT EXISTS idx_runs_user ON recommendation_runs (userId, created_at);
"""

_YEAR = r"\((\d{4})\)\s*$"
_EXTRA = ["original_language", "release_date", "poster_path"]


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_ready(path: Path | str = DB_PATH) -> bool:
    try:
        if not Path(path).exists():
            return False
        conn = connect(path)
        try:
            return bool(conn.execute("SELECT 1 FROM user_stats LIMIT 1").fetchone())
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _movie_frame(movies: pd.DataFrame, merged: pd.DataFrame | None) -> pd.DataFrame:
    frame = movies[["movieId", "title", "genres"]].copy()
    frame["year"] = pd.to_numeric(frame["title"].str.extract(_YEAR)[0], errors="coerce")
    for col in _EXTRA:
        frame[col] = None
    if merged is not None:
        have = [c for c in _EXTRA if c in merged.columns]
        if have:
            frame = frame.drop(columns=have).merge(
                merged[["movieId", *have]].drop_duplicates("movieId"), on="movieId", how="left")
    return frame.drop_duplicates("movieId")


def build_database(path: Path | str, ratings_chunks, movies: pd.DataFrame,
                   merged: pd.DataFrame | None = None) -> dict[str, int]:
    """(Re)build the store. ratings_chunks yields frames [userId, movieId, rating, timestamp]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = connect(path)
    conn.executescript(SCHEMA)
    _movie_frame(movies, merged).to_sql("movies", conn, if_exists="append", index=False)
    total = 0
    for chunk in ratings_chunks:
        out = chunk.rename(columns={"timestamp": "ts"})[["userId", "movieId", "rating", "ts"]].copy()
        out["ts"] = out["ts"].astype(str)
        out.to_sql("ratings", conn, if_exists="append", index=False, chunksize=100_000)
        total += len(out)
    conn.execute("CREATE INDEX idx_ratings_user ON ratings (userId, ts)")
    conn.execute("INSERT INTO user_stats SELECT userId, COUNT(*), AVG(rating), MIN(ts), MAX(ts) "
                 "FROM ratings GROUP BY userId")
    conn.commit()
    users = conn.execute("SELECT COUNT(*) FROM user_stats").fetchone()[0]
    conn.close()
    return {"ratings": total, "users": users}
