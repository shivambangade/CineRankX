"""Unit tests for output displayed by the recommendation UI."""

import pandas as pd
import pytest

from src.ranking_engine.config import OBJECTIVES
from src.ui.results import output_kpis, prepare_results


@pytest.fixture
def ranked():
    return pd.DataFrame([
        {"rank": 2, "movieId": 20, "title": "Second", "score": 0.4,
         "sources": "popularity", **{name: 0.2 for name in OBJECTIVES}},
        {"rank": 1, "movieId": 10, "title": "First", "score": 0.8,
         "sources": None, **{name: 0.6 for name in OBJECTIVES}},
    ])


def test_prepares_and_sorts_ranked_output(ranked):
    result = prepare_results(ranked)
    assert result["movieId"].tolist() == [10, 20]
    assert result["sources"].tolist() == ["", "popularity"]
    assert ranked["rank"].tolist() == [2, 1]


def test_kpis_describe_displayed_list(ranked):
    kpis = output_kpis(prepare_results(ranked))
    assert kpis == pytest.approx({
        "titles": 2, "mean_score": 0.6, "mean_relevance": 0.4, "mean_novelty": 0.4,
    })


@pytest.mark.parametrize(
    "column,value",
    [("rank", 0), ("rank", 1.5), ("movieId", 1.5), ("score", float("inf")),
     ("novelty", "not a number"), ("title", "")],
)
def test_rejects_invalid_rows(ranked, column, value):
    ranked[column] = ranked[column].astype(object)
    ranked.loc[0, column] = value
    with pytest.raises(ValueError):
        prepare_results(ranked)


def test_rejects_missing_objectives(ranked):
    with pytest.raises(ValueError, match="Missing columns: novelty"):
        prepare_results(ranked.drop(columns=["novelty"]))


def test_rejects_duplicate_movies(ranked):
    ranked.loc[0, "movieId"] = 10
    with pytest.raises(ValueError, match="unique"):
        prepare_results(ranked)


def test_empty_output_has_no_kpis(ranked):
    result = prepare_results(ranked.iloc[0:0])
    assert output_kpis(result) == {
        "titles": 0, "mean_score": 0.0, "mean_relevance": 0.0, "mean_novelty": 0.0,
    }


def test_split_title_extracts_year():
    from src.ui.results import split_title
    assert split_title("Toy Story (1995)") == ("Toy Story", "1995")
    assert split_title("No Year") == ("No Year", "")


def test_reason_uses_top_objectives(ranked):
    from src.ui.results import reason_for
    row = next(prepare_results(ranked).itertuples(index=False))
    assert reason_for(row).startswith("Ranked on strong ")


def test_genres_lookup_handles_missing_catalog():
    from src.ui.results import genres_lookup
    assert genres_lookup(None) == {}
    movies = pd.DataFrame({"movieId": [1], "genres": ["Comedy|Drama"]})
    assert genres_lookup(movies) == {1: ["Comedy", "Drama"]}
