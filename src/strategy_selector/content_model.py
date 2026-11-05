"""Content-based recommender with configurable representation, user profile and quality prior.

Every option is something the held-out experiment can switch independently:
  vectorizer  "base"  TF-IDF over text_chunk_clean, 2,000 terms (the original strategy)
              "rich"  TF-IDF with bigrams, sublinear tf and 20,000 terms
  genre_weight        weight of a genre multi-hot block appended to the (normalised) text vector
  profile     "top20"          rating-weighted sum of the user's 20 best-rated movies (original)
              "liked_centered" weights = rating - user mean over ALL rated movies (dislikes push away)
  quality_alpha       multiplies similarity by (quality percentile) ** alpha: a mild prior towards
                      well-rated, well-voted movies (0 disables it)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BASE_OPTIONS = {"vectorizer": "base", "genre_weight": 0.0, "profile": "top20", "quality_alpha": 0.0}


def _quality_percentile(movies: pd.DataFrame) -> np.ndarray:
    """Bayesian-weighted rating percentile (votes shrunk towards the catalog mean)."""
    if "vote_average" not in movies.columns or "vote_count" not in movies.columns:
        return np.ones(len(movies))
    votes = movies["vote_count"].fillna(0).to_numpy(float)
    avg = movies["vote_average"].fillna(0).to_numpy(float)
    prior_votes = max(float(np.median(votes[votes > 0])) if (votes > 0).any() else 1.0, 1.0)
    mean = float(avg[votes > 0].mean()) if (votes > 0).any() else 0.0
    bayes = (votes * avg + prior_votes * mean) / (votes + prior_votes)
    return pd.Series(bayes).rank(pct=True).to_numpy()


class ContentModel:
    def __init__(self, movies: pd.DataFrame, **options):
        self.options = {**BASE_OPTIONS, **options}
        o = self.options
        self.movie_ids = movies["movieId"].to_numpy()
        self.row = {int(m): i for i, m in enumerate(self.movie_ids)}
        text = movies["text_chunk_clean"].fillna("")
        vec = (TfidfVectorizer(max_features=2000) if o["vectorizer"] == "base" else
               TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True))
        matrix = normalize(vec.fit_transform(text))
        if o["genre_weight"] > 0 and "genres" in movies.columns:
            genres = movies["genres"].fillna("").str.replace("|", " ", regex=False).str.replace("-", "", regex=False)
            block = normalize(TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False).fit_transform(genres))
            matrix = normalize(hstack([matrix, o["genre_weight"] * block]).tocsr())
        self.matrix = matrix.tocsr()
        self.quality = _quality_percentile(movies) ** o["quality_alpha"] if o["quality_alpha"] else None

    def profile_vector(self, train: pd.DataFrame) -> np.ndarray | None:
        ratings = train["rating"].to_numpy(float)
        if self.options["profile"] == "top20":
            order = np.argsort(-ratings)[:20]
            items, w = train["movieId"].to_numpy()[order], ratings[order]
        else:
            items, w = train["movieId"].to_numpy(), ratings - ratings.mean()
        def combine(weights_):
            idx = [(self.row[int(m)], wt) for m, wt in zip(items, weights_) if int(m) in self.row and wt != 0]
            if not idx:
                return None
            r, wt = zip(*idx)
            vec = np.asarray(self.matrix[list(r)].multiply(np.array(wt).reshape(-1, 1)).sum(axis=0)).ravel()
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None

        vec = combine(w)
        # A user who rates everything the same (or whose centred weights cancel) has no "direction":
        # fall back to the plain rating-weighted profile instead of returning nothing.
        return vec if vec is not None else combine(np.ones(len(items)))

    def top_items(self, train: pd.DataFrame, top_n: int) -> list[int]:
        vec = self.profile_vector(train)
        if vec is None:
            return []
        sims = np.asarray(self.matrix @ vec).ravel()
        if self.quality is not None:
            sims = sims * self.quality
        for m in set(train["movieId"]):
            r = self.row.get(int(m))
            if r is not None:
                sims[r] = -np.inf
        top = np.argpartition(-sims, top_n)[:top_n]
        return [int(self.movie_ids[i]) for i in top[np.argsort(-sims[top])]]

    def hit_rate(self, train: pd.DataFrame, test_ids, top_n: int) -> float:
        if len(test_ids) == 0:
            return 0.0
        return len(set(test_ids) & set(self.top_items(train, top_n))) / len(test_ids)
