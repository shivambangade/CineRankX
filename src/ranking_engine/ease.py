"""EASE item-item linear model: CineRankX's personalised relevance source.

EASE (Steck, 2019) is closed-form ridge regression of every item on all the others,
fit on the binary user-item matrix X of the TRAINING interactions:

    P = (X'X + lambda I)^-1,   B = -P / diag(P),   diag(B) = 0,   score(u) = x_u B

It is classical and explainable (a weighted item-to-item similarity), trains in seconds
on the 6,000 most-rated movies, and scores any user from their own ratings by fold-in --
including a cold-start user with five ratings. Scores are turned into per-user percentile
ranks and blended with a popularity percentile; the blend weight (beta) is tuned on the
validation split, and the multi-objective ranker still produces the final list.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

MODEL_PATH = Path("data/processed/ease_model.npz")


class EaseModel:
    def __init__(self, item_ids: np.ndarray, weights: np.ndarray, log_popularity: np.ndarray):
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        self.weights = weights.astype(np.float32)            # B, (n_items, n_items)
        self.log_popularity = log_popularity.astype(np.float32)
        self.index = {int(m): i for i, m in enumerate(self.item_ids)}
        order = self.log_popularity.argsort().argsort()
        self.pop_percentile = (order / len(order)).astype(np.float32)

    @classmethod
    def fit(cls, train: pd.DataFrame, n_items: int = 6000, l2: float = 500.0) -> "EaseModel":
        """Fit on a ratings frame [userId, movieId, rating]; every rating is an interaction."""
        counts = train["movieId"].value_counts()
        top = np.sort(counts.index[:n_items].to_numpy())
        col = {int(m): i for i, m in enumerate(top)}
        kept = train[train["movieId"].isin(col)]
        users = kept["userId"].astype("int64")
        rows = pd.factorize(users)[0]
        x = csr_matrix((np.ones(len(kept), dtype=np.float32), (rows, kept["movieId"].map(col).to_numpy())),
                       shape=(rows.max() + 1, len(top)))
        gram = (x.T @ x).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += l2
        inverse = np.linalg.inv(gram)
        b = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(b, 0.0)
        popularity = np.log1p(np.asarray(x.sum(axis=0)).ravel())
        return cls(top, b, popularity)

    def user_matrix(self, profiles: dict[int, list[int]], users: list[int]) -> csr_matrix:
        rows, cols = [], []
        for r, u in enumerate(users):
            for m in profiles.get(u, ()):
                c = self.index.get(int(m))
                if c is not None:
                    rows.append(r)
                    cols.append(c)
        return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(users), len(self.item_ids)))

    def blended_scores(self, user_rows: csr_matrix, beta: float) -> np.ndarray:
        """(users x items) = (1-beta) * popularity percentile + beta * per-user EASE percentile."""
        raw = np.asarray(user_rows @ self.weights)
        personal = raw.argsort(axis=1).argsort(axis=1) / raw.shape[1]
        return (1.0 - beta) * self.pop_percentile[None, :] + beta * personal

    def save(self, path: Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, item_ids=self.item_ids, weights=self.weights, log_popularity=self.log_popularity)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "EaseModel | None":
        if not Path(path).exists():
            return None
        data = np.load(path)
        return cls(data["item_ids"], data["weights"], data["log_popularity"])
