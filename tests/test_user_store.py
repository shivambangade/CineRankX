"""Tests for the SQLite user store: lookup, history paging/filtering, run history."""

import pandas as pd
import pytest

from src.storage import (active_users, build_database, connect, db_ready, genre_preferences,
                         history_page, list_runs, rating_distribution, record_run, user_summary,
                         user_timeline)


@pytest.fixture
def conn(tmp_path):
    movies = pd.DataFrame({
        "movieId": [1, 2, 3],
        "title": ["Toy Story (1995)", "Heat (1995)", "Up (2009)"],
        "genres": ["Animation|Comedy", "Action|Crime", "Animation|Adventure"]})
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2], "movieId": [1, 2, 3, 1], "rating": [5.0, 3.0, 4.5, 2.0],
        "timestamp": ["2020-01-01 10:00:00", "2020-02-01 10:00:00", "2020-03-01 10:00:00",
                      "2021-01-01 00:00:00"]})
    path = tmp_path / "t.db"
    build_database(path, iter([ratings]), movies)
    assert db_ready(path)
    connection = connect(path)
    yield connection
    connection.close()


def test_summary_and_unknown_user(conn):
    summary = user_summary(conn, 1)
    assert summary["n_ratings"] == 3 and summary["n_genres"] == 5
    assert summary["first_ts"].startswith("2020-01") and summary["last_ts"].startswith("2020-03")
    assert user_summary(conn, 999) is None


def test_history_paginates_filters_and_sorts(conn):
    page, total = history_page(conn, 1, page=0, page_size=2)
    assert total == 3 and len(page) == 2 and page.iloc[0]["title"] == "Up (2009)"
    page2, _ = history_page(conn, 1, page=1, page_size=2)
    assert len(page2) == 1
    filtered, total = history_page(conn, 1, genre="Animation", sort="rating", descending=False)
    assert total == 2 and filtered["rating"].tolist() == [4.5, 5.0]
    found, total = history_page(conn, 1, search="heat")
    assert total == 1 and found.iloc[0]["movieId"] == 2
    _, total = history_page(conn, 1, min_rating=4.0)
    assert total == 2


def test_history_is_scoped_to_user_and_safe_sort(conn):
    page, total = history_page(conn, 2, sort="title; DROP TABLE ratings")
    assert total == 1 and page.iloc[0]["rating"] == 2.0
    assert history_page(conn, 1)[1] == 3


def test_preferences_and_timeline(conn):
    prefs = genre_preferences(conn, 1)
    assert prefs.iloc[0]["genre"] == "Animation" and prefs.iloc[0]["count"] == 2
    assert rating_distribution(conn, 1).to_dict() == {3.0: 1, 4.5: 1, 5.0: 1}
    timeline = user_timeline(conn, 1)
    assert timeline["monthly"]["ratings"].sum() == 3
    assert set(timeline["decades"]["decade"]) == {"1990s", "2000s"}
    assert user_timeline(conn, 999)["monthly"].empty


def test_active_users_prefix(conn):
    assert active_users(conn).iloc[0]["userId"] == 1
    assert active_users(conn, "2")["userId"].tolist() == [2]


def test_run_history_round_trip(conn):
    results = pd.DataFrame({"rank": [1], "movieId": [1], "title": ["Toy Story (1995)"], "score": [0.9]})
    record_run(conn, 1, source="live", strategy="popularity", confidence=0.7, k=5,
               weights={"relevance": 0.5}, status="success", elapsed_ms=12.0, results=results)
    record_run(conn, 1, source="live", strategy=None, confidence=None, k=5, weights=None,
               status="error", error="boom")
    runs = list_runs(conn, 1)
    assert [r["status"] for r in runs] == ["error", "success"]
    assert runs[1]["weights"] == {"relevance": 0.5} and runs[1]["results"].iloc[0]["title"].startswith("Toy")
    assert list_runs(conn, 2) == []
