"""Server-side user queries: profile summary, paginated history, preferences, timelines."""

import sqlite3

import pandas as pd

SORTABLE = {"timestamp": "r.ts", "rating": "r.rating", "title": "m.title", "year": "m.year"}


def user_summary(conn: sqlite3.Connection, user_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM user_stats WHERE userId = ?", (int(user_id),)).fetchone()
    if row is None:
        return None
    genres = conn.execute(
        "SELECT m.genres FROM ratings r JOIN movies m USING (movieId) WHERE r.userId = ?",
        (int(user_id),)).fetchall()
    unique = {g for (value,) in genres if value for g in value.split("|")
              if g and g != "(no genres listed)"}
    return {"userId": int(row["userId"]), "n_ratings": int(row["n_ratings"]),
            "avg_rating": float(row["avg_rating"]), "first_ts": row["first_ts"],
            "last_ts": row["last_ts"], "n_genres": len(unique)}


def _filters(user_id: int, search: str, genre: str, min_rating: float, max_rating: float):
    where = ["r.userId = ?", "r.rating BETWEEN ? AND ?"]
    args = [int(user_id), min_rating, max_rating]
    if search.strip():
        where.append("m.title LIKE ?")
        args.append(f"%{search.strip()}%")
    if genre and genre != "All":
        where.append("('|' || m.genres || '|') LIKE ?")
        args.append(f"%|{genre}|%")
    return " AND ".join(where), args


def history_page(conn: sqlite3.Connection, user_id: int, *, search: str = "", genre: str = "All",
                 min_rating: float = 0.5, max_rating: float = 5.0, sort: str = "timestamp",
                 descending: bool = True, page: int = 0,
                 page_size: int = 25) -> tuple[pd.DataFrame, int]:
    """One page of a user's ratings plus the total matching rows (filtering done in SQL)."""
    column = SORTABLE.get(sort, SORTABLE["timestamp"])
    where, args = _filters(user_id, search, genre, min_rating, max_rating)
    base = "FROM ratings r JOIN movies m USING (movieId) WHERE " + where
    total = conn.execute("SELECT COUNT(*) " + base, args).fetchone()[0]
    order = "DESC" if descending else "ASC"
    page_size = max(1, min(int(page_size), 200))
    rows = conn.execute(
        f"SELECT r.movieId, m.title, m.year, m.genres, r.rating, r.ts AS timestamp, "
        f"m.original_language, m.poster_path {base} ORDER BY {column} {order}, r.movieId "
        "LIMIT ? OFFSET ?", [*args, page_size, max(0, int(page)) * page_size]).fetchall()
    return pd.DataFrame([dict(r) for r in rows]), int(total)


def _joined(conn: sqlite3.Connection, user_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT r.rating, r.ts, m.title, m.year, m.genres, m.original_language "
        "FROM ratings r JOIN movies m USING (movieId) WHERE r.userId = ?", conn,
        params=(int(user_id),))


def rating_distribution(conn: sqlite3.Connection, user_id: int) -> pd.Series:
    rows = conn.execute("SELECT rating, COUNT(*) FROM ratings WHERE userId = ? GROUP BY rating "
                        "ORDER BY rating", (int(user_id),)).fetchall()
    return pd.Series({float(r[0]): int(r[1]) for r in rows}, dtype="int64")


def genre_preferences(conn: sqlite3.Connection, user_id: int) -> pd.DataFrame:
    """Per-genre rating count and mean rating, most-watched first."""
    frame = _joined(conn, user_id)
    if frame.empty:
        return pd.DataFrame(columns=["genre", "count", "avg_rating"])
    frame["genre"] = frame["genres"].fillna("").str.split("|")
    exploded = frame.explode("genre")
    exploded = exploded[exploded["genre"].notna() & ~exploded["genre"].isin(["", "(no genres listed)"])]
    out = exploded.groupby("genre")["rating"].agg(count="count", avg_rating="mean").reset_index()
    return out.sort_values(["count", "avg_rating"], ascending=False).reset_index(drop=True)


def user_timeline(conn: sqlite3.Connection, user_id: int) -> dict[str, pd.DataFrame]:
    """Monthly activity, release-decade mix and language mix, all from real rows."""
    frame = _joined(conn, user_id)
    if frame.empty:
        empty = pd.DataFrame()
        return {"monthly": empty, "decades": empty, "languages": empty}
    monthly = (frame.assign(month=frame["ts"].str.slice(0, 7)).groupby("month")["rating"]
               .agg(ratings="count", avg_rating="mean").reset_index())
    known = frame.dropna(subset=["year"]).copy()
    known["decade"] = (known["year"] // 10 * 10).astype(int).astype(str) + "s"
    decades = known.groupby("decade")["rating"].agg(ratings="count", avg_rating="mean").reset_index()
    langs = frame.dropna(subset=["original_language"])
    languages = (langs.groupby("original_language")["rating"].agg(ratings="count", avg_rating="mean")
                 .reset_index().sort_values("ratings", ascending=False))
    return {"monthly": monthly, "decades": decades, "languages": languages}


def active_users(conn: sqlite3.Connection, prefix: str = "", limit: int = 20) -> pd.DataFrame:
    """Users matching an ID prefix, most active first (served from user_stats)."""
    where, args = "", []
    if prefix.strip().isdigit():
        where, args = "WHERE CAST(userId AS TEXT) LIKE ?", [f"{prefix.strip()}%"]
    rows = conn.execute(f"SELECT userId, n_ratings, avg_rating, last_ts FROM user_stats {where} "
                        "ORDER BY n_ratings DESC LIMIT ?", [*args, int(limit)]).fetchall()
    return pd.DataFrame([dict(r) for r in rows])
