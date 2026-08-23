"""Tests for the IR engine: TF-IDF, Trie, and similarity search."""

import pandas as pd
import pytest

from src.ir_engine import IREngine, Trie, TFIDFEngine


class TestTrie:
    def test_insert_and_search_basic(self):
        trie = Trie()
        trie.insert("Toy Story", 1)
        trie.insert("Toy Story 2", 2)

        results = trie.search("toy", limit=10)
        assert len(results) == 2
        assert (1, "Toy Story") in results
        assert (2, "Toy Story 2") in results

    def test_search_case_insensitive(self):
        trie = Trie()
        trie.insert("Batman Returns", 100)

        results = trie.search("BAT", limit=10)
        assert len(results) == 1
        assert results[0] == (100, "Batman Returns")

    def test_search_empty_prefix_returns_empty(self):
        trie = Trie()
        trie.insert("Some Movie", 1)
        assert trie.search("", limit=10) == []

    def test_search_nonexistent_prefix(self):
        trie = Trie()
        trie.insert("Toy Story", 1)
        assert trie.search("xyz", limit=10) == []

    def test_search_respects_limit(self):
        trie = Trie()
        for i in range(20):
            trie.insert(f"Batman {i}", i)

        results = trie.search("bat", limit=5)
        assert len(results) <= 5

    def test_insert_ignores_empty_titles(self):
        trie = Trie()
        trie.insert("", 1)
        trie.insert(None, 2)
        assert trie.search("", limit=10) == []


class TestTFIDFEngine:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Toy Story", "Toy Story 2", "Finding Nemo"],
                "text_chunk_clean": [
                    "toy story cowboy spaceman friendship",
                    "toy story cowboy spaceman adventure sequel",
                    "fish ocean adventure ocean clownfish",
                ],
            }
        )

    def test_fit_creates_matrix(self, sample_df):
        engine = TFIDFEngine()
        engine.fit(sample_df)

        assert engine.matrix is not None
        assert engine.vectorizer is not None
        assert len(engine.movie_ids) == 3
        assert len(engine.movie_titles) == 3

    def test_search_returns_results(self, sample_df):
        engine = TFIDFEngine()
        engine.fit(sample_df)

        results = engine.search("toy cowboy", limit=10)
        assert len(results) > 0
        assert results[0]["similarity"] > 0
        assert "movieId" in results[0]
        assert "title" in results[0]

    def test_search_respects_limit(self, sample_df):
        engine = TFIDFEngine()
        engine.fit(sample_df)

        results = engine.search("adventure", limit=2)
        assert len(results) <= 2

    def test_similar_to_returns_similar_movies(self, sample_df):
        engine = TFIDFEngine()
        engine.fit(sample_df)

        results = engine.similar_to(1, limit=5)
        assert len(results) > 0
        assert all(r["movieId"] != 1 for r in results)

    def test_similar_to_nonexistent_movie_returns_empty(self, sample_df):
        engine = TFIDFEngine()
        engine.fit(sample_df)

        results = engine.similar_to(999, limit=5)
        assert results == []

    def test_search_requires_fit(self):
        engine = TFIDFEngine()
        with pytest.raises(ValueError, match="not fitted"):
            engine.search("test")

    def test_save_and_load(self, sample_df, tmp_path, monkeypatch):
        import src.ir_engine.tfidf_vectorizer as tfidf_module

        monkeypatch.setattr(tfidf_module, "_VECTORIZER_PATH", tmp_path / "vectorizer.pkl")
        monkeypatch.setattr(tfidf_module, "_MATRIX_PATH", tmp_path / "matrix.npz")

        engine = TFIDFEngine()
        engine.fit(sample_df)
        engine.save()

        engine2 = TFIDFEngine()
        loaded = engine2.load()
        assert loaded
        assert engine2.vectorizer is not None
        assert engine2.matrix is not None

        results1 = engine.search("toy", limit=5)
        results2 = engine2.search("toy", limit=5)
        assert len(results1) == len(results2)


class TestIREngine:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "title": ["Toy Story", "Toy Story 2", "Finding Nemo", "Batman Begins"],
                "text_chunk_clean": [
                    "toy story cowboy spaceman friendship",
                    "toy story cowboy spaceman adventure sequel",
                    "fish ocean adventure clownfish",
                    "batman dark knight superhero crime",
                ],
            }
        )

    def test_fit_initializes_both_tfidf_and_trie(self, sample_df):
        engine = IREngine()
        engine.fit(sample_df)

        assert engine.tfidf.vectorizer is not None
        assert engine.trie.root is not None

    def test_search_works(self, sample_df):
        engine = IREngine()
        engine.fit(sample_df)

        results = engine.search("toy story", limit=5)
        assert len(results) > 0
        assert "movieId" in results[0]
        assert "title" in results[0]

    def test_autocomplete_works(self, sample_df):
        engine = IREngine()
        engine.fit(sample_df)

        results = engine.autocomplete("toy", limit=10)
        assert len(results) == 2
        titles = [r["title"] for r in results]
        assert "Toy Story" in titles
        assert "Toy Story 2" in titles

    def test_similar_to_works(self, sample_df):
        engine = IREngine()
        engine.fit(sample_df)

        results = engine.similar_to(1, limit=5)
        assert len(results) > 0
        assert all(r["movieId"] != 1 for r in results)

    def test_autocomplete_case_insensitive(self, sample_df):
        engine = IREngine()
        engine.fit(sample_df)

        results_lower = engine.autocomplete("bat", limit=10)
        results_upper = engine.autocomplete("BAT", limit=10)
        assert len(results_lower) == len(results_upper)
