import pandas as pd

from src.ingestion.joiner import join_movielens_tmdb
from src.ingestion.loaders import load_movielens, load_tmdb
from src.ingestion.text_cleaner import clean_text


def test_load_movielens(tmp_path):
    (tmp_path / "rating.csv").write_text("userId,movieId,rating,timestamp\n1,1,4.0,2005-04-02 23:53:47\n")
    (tmp_path / "movie.csv").write_text("movieId,title,genres\n1,Toy Story (1995),Animation\n")
    (tmp_path / "link.csv").write_text("movieId,imdbId,tmdbId\n1,114709,862\n")
    (tmp_path / "tag.csv").write_text("userId,movieId,tag,timestamp\n1,1,fun,2005-04-02 23:53:47\n")

    ml = load_movielens(tmp_path)

    assert set(ml.keys()) == {"ratings", "movies", "links", "tags"}
    assert list(ml["movies"].columns) == ["movieId", "title", "genres"]
    assert ml["ratings"].iloc[0]["rating"] == 4.0


def test_load_tmdb(tmp_path):
    (tmp_path / "TMDB_movie_dataset_v11.csv").write_text(
        "id,title,overview,genres,keywords,vote_average,vote_count,popularity\n"
        '862,Toy Story,"A cowboy doll","Animation, Comedy","toys, jealousy",8.0,100,50.0\n'
    )

    tmdb = load_tmdb(tmp_path)

    assert set(tmdb.keys()) == {"movies"}
    assert tmdb["movies"].iloc[0]["title"] == "Toy Story"


def test_join_movielens_tmdb_builds_text_chunk():
    ml = {
        "movies": pd.DataFrame(
            {"movieId": [1], "title": ["Toy Story (1995)"], "genres": ["Animation|Comedy"]}
        ),
        "links": pd.DataFrame({"movieId": [1], "imdbId": ["114709"], "tmdbId": [862]}),
    }
    tmdb = {
        "movies": pd.DataFrame(
            {
                "id": [862],
                "overview": ["A cowboy doll is jealous"],
                "genres": ["Animation, Comedy"],
                "keywords": ["toys, jealousy"],
                "vote_average": [8.0],
                "vote_count": [100],
                "popularity": [50.0],
            }
        ),
    }

    result, report = join_movielens_tmdb(ml, tmdb)

    assert report["joined_rows"] == 1
    row = result.iloc[0]
    assert row["genre_names"] == ["Animation", "Comedy"]
    assert row["keyword_names"] == ["toys", "jealousy"]
    assert "cast_names" not in result.columns
    assert "crew_names" not in result.columns
    assert "cowboy" in row["text_chunk"]
    assert "Tom Hanks" not in row["text_chunk"]


def test_join_drops_unmatched_movies():
    ml = {
        "movies": pd.DataFrame({"movieId": [1, 2], "title": ["A", "B"], "genres": ["X", "Y"]}),
        "links": pd.DataFrame({"movieId": [1, 2], "imdbId": ["1", "2"], "tmdbId": [999, None]}),
    }
    tmdb = {
        "movies": pd.DataFrame(
            {
                "id": [1],
                "overview": ["irrelevant"],
                "genres": [""],
                "keywords": [""],
                "vote_average": [1.0],
                "vote_count": [1],
                "popularity": [1.0],
            }
        ),
    }

    result, report = join_movielens_tmdb(ml, tmdb)

    assert report["movielens_movies"] == 2
    assert report["joined_rows"] == 0


def test_clean_text_lowercases_removes_stopwords_and_stems():
    cleaned = clean_text("The Runners are Running quickly in the Park")

    assert cleaned == cleaned.lower()
    assert "the" not in cleaned.split()
    assert "in" not in cleaned.split()
    assert "run" in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
