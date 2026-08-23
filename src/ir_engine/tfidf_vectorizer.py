"""TF-IDF vectorization and cosine similarity search over movie texts."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_VECTORIZER_PATH = Path("data/processed/tfidf_vectorizer.pkl")
_MATRIX_PATH = Path("data/processed/tfidf_matrix.npz")


class TFIDFEngine:
    """TF-IDF vectorizer and similarity search engine."""

    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.movie_ids = None
        self.movie_titles = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit TF-IDF vectorizer on text_chunk_clean column."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=False,  # Already lowercased in text_chunk_clean
        )
        texts = df["text_chunk_clean"].fillna("")
        self.matrix = self.vectorizer.fit_transform(texts)
        self.movie_ids = df["movieId"].values
        self.movie_titles = df["title"].values

    def save(self) -> None:
        """Persist vectorizer and matrix to disk."""
        _VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        from scipy.sparse import save_npz

        save_npz(_MATRIX_PATH, self.matrix)
        # Save metadata
        metadata = {
            "movie_ids": self.movie_ids,
            "movie_titles": self.movie_titles,
        }
        with open(_MATRIX_PATH.parent / "tfidf_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load(self) -> bool:
        """Load vectorizer and matrix from disk. Return True if successful."""
        if not _VECTORIZER_PATH.exists() or not _MATRIX_PATH.exists():
            return False
        try:
            with open(_VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
            from scipy.sparse import load_npz

            self.matrix = load_npz(_MATRIX_PATH)
            with open(_MATRIX_PATH.parent / "tfidf_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                self.movie_ids = metadata["movie_ids"]
                self.movie_titles = metadata["movie_titles"]
            return True
        except Exception:
            return False

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for movies similar to query text. Return top-k results."""
        if self.vectorizer is None or self.matrix is None:
            raise ValueError("Engine not fitted. Call fit() first.")
        if not query or not isinstance(query, str):
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append(
                    {
                        "movieId": int(self.movie_ids[idx]),
                        "title": self.movie_titles[idx],
                        "similarity": float(similarities[idx]),
                    }
                )
        return results

    def similar_to(self, movie_id: int, limit: int = 10) -> list[dict]:
        """Find movies similar to a given movie (excluding itself)."""
        if self.vectorizer is None or self.matrix is None:
            raise ValueError("Engine not fitted. Call fit() first.")

        idx = np.where(self.movie_ids == movie_id)[0]
        if len(idx) == 0:
            return []

        idx = idx[0]
        query_vec = self.matrix[idx]
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:limit + 1]

        results = []
        for top_idx in top_indices:
            if int(self.movie_ids[top_idx]) != movie_id and similarities[top_idx] > 0:
                results.append(
                    {
                        "movieId": int(self.movie_ids[top_idx]),
                        "title": self.movie_titles[top_idx],
                        "similarity": float(similarities[top_idx]),
                    }
                )
                if len(results) >= limit:
                    break
        return results
