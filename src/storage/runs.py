"""Persisted recommendation runs: what was generated, when, with which engine and weights."""

import json
import sqlite3
from datetime import datetime, timezone
from io import StringIO

import pandas as pd


def record_run(conn: sqlite3.Connection, user_id: int, *, source: str, strategy: str | None,
               confidence: float | None, k: int, weights: dict | None, status: str,
               error: str | None = None, elapsed_ms: float | None = None,
               results: pd.DataFrame | None = None) -> int:
    payload = results.to_json(orient="records") if results is not None else None
    cursor = conn.execute(
        "INSERT INTO recommendation_runs (userId, created_at, source, strategy, confidence, k, "
        "weights, status, error, elapsed_ms, results) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (int(user_id), datetime.now(timezone.utc).isoformat(timespec="seconds"), source, strategy,
         confidence, int(k), json.dumps(weights) if weights else None, status, error, elapsed_ms,
         payload))
    conn.commit()
    return int(cursor.lastrowid)


def list_runs(conn: sqlite3.Connection, user_id: int, limit: int = 20, offset: int = 0) -> list[dict]:
    rows = conn.execute("SELECT * FROM recommendation_runs WHERE userId = ? ORDER BY id DESC "
                        "LIMIT ? OFFSET ?", (int(user_id), int(limit), int(offset))).fetchall()
    runs = []
    for row in rows:
        run = dict(row)
        run["weights"] = json.loads(run["weights"]) if run["weights"] else None
        run["results"] = (pd.read_json(StringIO(run["results"]), orient="records")
                          if run["results"] else None)
        runs.append(run)
    return runs
