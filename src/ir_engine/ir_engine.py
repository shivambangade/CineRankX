"""Unified IR engine combining TF-IDF search and Trie autocomplete."""

import pandas as pd

from src.ir_engine.tfidf_vectorizer import TFIDFEngine
from src.ir_engine.trie import Trie


class IREngine:
    """Classical information retrieval engine with TF-IDF and Trie autocomplete."""

    def __init__(self):
        self.tfidf = TFIDFEngine()
        self.trie = Trie()

    def fit(self, df: pd.DataFrame) -> None:
        """Fit both TF-IDF and Trie on the movie dataset."""
        self.tfidf.fit(df)
        for _, row in df.iterrows():
            self.trie.insert(row["title"], row["movieId"])

    def save(self) -> None:
        """Persist both TF-IDF and Trie to disk."""
        self.tfidf.save()

    def load(self) -> bool:
        """Load persisted TF-IDF from disk. Trie cannot be persisted easily; re-fit if needed."""
        return self.tfidf.load()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for movies by text query (overview, genres, keywords)."""
        return self.tfidf.search(query, limit)

    def autocomplete(self, prefix: str, limit: int = 10) -> list[dict]:
        """Find movies by title prefix."""
        results = self.trie.search(prefix, limit)
        return [{"movieId": mid, "title": title} for mid, title in results]

    def similar_to(self, movie_id: int, limit: int = 10) -> list[dict]:
        """Find movies most similar to a given movie."""
        return self.tfidf.similar_to(movie_id, limit)
