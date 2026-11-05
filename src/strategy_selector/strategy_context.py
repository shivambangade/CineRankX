"""Strategy-aware features: a cheap backtest of each strategy on the user's OWN profile.

The label says which strategy won on a hidden holdout. The best available predictor of
that is how the same three strategies score when the user's visible ratings are split
again (profile', pseudo-holdout). These scores, plus popularity-head shares, are computed
by one function used for training and for live prediction, so the two can never drift.
Only ratings passed in are read; the label holdout is never visible.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.strategy_selector.profile_features import extract_profile_features
from src.strategy_selector.strategy_labeler import (
    _CF_MAX_MOVIES,
    _CF_MAX_USERS,
    _BacktestContext,
    _build_cb_profile_vector,
    _popularity_hit_rate,
)

CONTEXT_PATH = Path("data/processed/strategy_context.pkl")
HEAD_SIZES = (100, 1000)  # "share of the profile inside the N most-rated titles"


class LiteContext(_BacktestContext):
    """Reference population for the inner backtest; no dense user-user matrix, so it is small."""

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, top_n: int = 10, content_model=None):
        self.top_n = top_n
        self.content_model = content_model   # optional improved content strategy (see content_model.py)
        self.cf_k = max(5, top_n // 2)
        self._build_popularity(ratings_df, movies_df)
        self._build_content_based(movies_df)
        counts = ratings_df["movieId"].value_counts()
        self.item_rank = {int(m): r for r, m in enumerate(counts.index)}
        self.item_mean = ratings_df.groupby("movieId")["rating"].mean().to_dict()
        self.item_surprisal = dict(zip(counts.index, -np.log2(counts.to_numpy() / counts.sum())))
        self._build_cf_lite(ratings_df)
        # Row-normalised once, so scoring a profile is one sparse mat-vec instead of re-normalising 27k rows.
        self.tfidf_norm = normalize(self.tfidf_matrix).tocsr() if self.tfidf_matrix is not None else None

    def _build_cf_lite(self, ratings_df: pd.DataFrame) -> None:
        pool = ratings_df["userId"].unique()[:_CF_MAX_USERS]
        sub = ratings_df[ratings_df["userId"].isin(pool)]
        cols = sub["movieId"].value_counts().nlargest(_CF_MAX_MOVIES).index
        sim = sub[sub["movieId"].isin(cols)]
        self.cf_users = sim["userId"].unique()
        self.cf_row = {int(u): i for i, u in enumerate(self.cf_users)}
        self.cf_col = {int(m): i for i, m in enumerate(cols)}
        matrix = csr_matrix((sim["rating"].to_numpy(dtype=np.float32),
                             (sim["userId"].map(self.cf_row).to_numpy(), sim["movieId"].map(self.cf_col).to_numpy())),
                            shape=(len(self.cf_users), len(cols)))
        self.cf_matrix = normalize(matrix)
        self.cf_top = (sub.sort_values("rating", ascending=False).groupby("userId")["movieId"]
                       .apply(lambda s: [int(m) for m in s.head(self.cf_k)]).to_dict())

    def cb_hit_rate(self, train: pd.DataFrame, test_ids) -> float:
        """Same quantity as strategy_labeler._content_based_hit_rate, via a precomputed normalised matrix."""
        # getattr: contexts pickled before the content_model option existed lack the attribute.
        content_model = getattr(self, "content_model", None)
        if content_model is not None:
            return content_model.hit_rate(train, test_ids, self.top_n)
        if self.tfidf_norm is None or len(test_ids) == 0 or train.empty:
            return 0.0
        profile = _build_cb_profile_vector(self, train)
        if profile is None:
            return 0.0
        vec = np.asarray(profile).ravel()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return 0.0
        sims = np.asarray(self.tfidf_norm @ (vec / norm)).ravel()
        for m in set(train["movieId"]):
            row = self.movie_id_to_row.get(m)
            if row is not None:
                sims[row] = -1.0
        top = np.argpartition(-sims, self.top_n)[:self.top_n]
        return len(set(test_ids) & set(self.tfidf_movie_ids[top])) / len(test_ids)

    def engine_scores(self, user_id: int, ratings: pd.DataFrame) -> dict[str, float]:
        """What the CB and CF engines would actually output for this user: similarity strengths.

        These are engine outputs (not hit-rates): how strongly the user's neighbourhood and
        content profile pull towards particular items, and how coherent their taste is.
        """
        out = {"cb_top_cos_mean": 0.0, "cb_coherence": 0.0, "cf_top_sim_mean": 0.0, "cf_top_sim_max": 0.0,
               "cf_cand_score_mean": 0.0}
        seen = set(ratings["movieId"])
        profile = _build_cb_profile_vector(self, ratings) if self.tfidf_norm is not None else None
        if profile is not None:
            vec = np.asarray(profile).ravel()
            norm = np.linalg.norm(vec)
            if norm > 0:
                sims = np.asarray(self.tfidf_norm @ (vec / norm)).ravel()
                for m in seen:
                    row = self.movie_id_to_row.get(m)
                    if row is not None:
                        sims[row] = -1.0
                out["cb_top_cos_mean"] = float(np.sort(sims)[-self.top_n:].mean())
            liked = ratings.nlargest(10, "rating")["movieId"].map(self.movie_id_to_row).dropna().astype(int)
            if len(liked) > 1:
                block = (self.tfidf_norm[liked.to_numpy()] @ self.tfidf_norm[liked.to_numpy()].T).toarray()
                out["cb_coherence"] = float((block.sum() - np.trace(block)) / (len(liked) * (len(liked) - 1)))
        vec = np.zeros(self.cf_matrix.shape[1], dtype=np.float32)
        for m, r in zip(ratings["movieId"], ratings["rating"]):
            c = self.cf_col.get(int(m))
            if c is not None:
                vec[c] = r
        norm = np.linalg.norm(vec)
        if norm > 0:
            sims = np.asarray(self.cf_matrix @ (vec / norm)).ravel()
            own = self.cf_row.get(int(user_id))
            if own is not None:
                sims[own] = -1.0
            top = np.sort(sims)[::-1][:self.cf_k]
            out["cf_top_sim_mean"], out["cf_top_sim_max"] = float(top.mean()), float(top.max())
            scores: dict[int, float] = {}
            for idx in np.argsort(sims)[::-1][:self.cf_k]:
                for m in self.cf_top.get(int(self.cf_users[idx]), ()):
                    if m not in seen:
                        scores[m] = max(scores.get(m, 0.0), float(sims[idx]))
            if scores:
                out["cf_cand_score_mean"] = float(np.mean(sorted(scores.values(), reverse=True)[:self.top_n]))
        return out

    def cf_hit_rate(self, user_id: int, train: pd.DataFrame, test_ids) -> float:
        if len(test_ids) == 0:
            return 0.0
        cols = [self.cf_col[int(m)] for m in train["movieId"] if int(m) in self.cf_col]
        if not cols:
            return 0.0
        vec = np.zeros(self.cf_matrix.shape[1], dtype=np.float32)
        for m, r in zip(train["movieId"], train["rating"]):
            c = self.cf_col.get(int(m))
            if c is not None:
                vec[c] = r
        norm = np.linalg.norm(vec)
        sims = np.asarray(self.cf_matrix @ (vec / norm)).ravel() if norm > 0 else np.zeros(len(self.cf_users))
        own = self.cf_row.get(int(user_id))
        if own is not None:      # never let a user's own full profile act as their neighbour
            sims[own] = -1.0
        scores: dict[int, float] = {}
        for idx in np.argsort(sims)[::-1][:self.cf_k]:
            if sims[idx] <= 0:
                continue
            for m in self.cf_top.get(int(self.cf_users[idx]), ()):
                scores[m] = max(scores.get(m, -1.0), float(sims[idx]))
        top = set(sorted(scores, key=scores.get, reverse=True)[:self.top_n])
        return len(set(test_ids) & top) / len(test_ids)


def inner_scores(ctx: LiteContext, user_id: int, ratings: pd.DataFrame, n_splits: int = 3,
                 fraction: float = 0.25) -> dict[str, float]:
    """Mean content/collaborative/popularity hit-rates over seeded splits of ONE user's ratings."""
    rng = np.random.default_rng(int(user_id) * 7919 + 17)
    n = len(ratings)
    n_test = max(1, int(round(n * fraction)))
    cb, cf, pop = [], [], []
    for _ in range(n_splits):
        order = rng.permutation(n)
        test, train = ratings.iloc[order[:n_test]], ratings.iloc[order[n_test:]]
        ids = test["movieId"].to_numpy()
        cb.append(ctx.cb_hit_rate(train, ids))
        cf.append(ctx.cf_hit_rate(user_id, train, ids))
        pop.append(_popularity_hit_rate(ctx, ids))
    return {"inner_cb": float(np.mean(cb)), "inner_cf": float(np.mean(cf)), "inner_pop": float(np.mean(pop))}


def build_features(ratings: pd.DataFrame, movies: pd.DataFrame, ctx: LiteContext, n_splits: int = 3,
                   fraction: float = 0.25) -> pd.DataFrame:
    """Profile features + popularity-head shares + inner-backtest scores, one row per user."""
    base = extract_profile_features(ratings, movies)
    if base.empty:
        return base
    rows = []
    for user_id, user_ratings in ratings.groupby("userId"):
        ranks = user_ratings["movieId"].map(ctx.item_rank)
        row = {"userId": user_id, **{f"top{h}_share": float((ranks < h).mean()) for h in HEAD_SIZES}}
        row.update(inner_scores(ctx, user_id, user_ratings, n_splits, fraction))
        row.update(ctx.engine_scores(user_id, user_ratings))
        deviation = user_ratings["rating"] - user_ratings["movieId"].map(ctx.item_mean)
        row["user_bias"] = float(deviation.dropna().mean()) if deviation.notna().any() else 0.0
        row["liked_share"] = float((user_ratings["rating"] >= 4.0).mean())
        row["profile_surprisal"] = float(user_ratings["movieId"].map(ctx.item_surprisal).dropna().mean())             if user_ratings["movieId"].isin(ctx.item_surprisal).any() else 0.0
        row["inner_gap_cb_pop"] = row["inner_cb"] - row["inner_pop"]
        row["inner_gap_cf_pop"] = row["inner_cf"] - row["inner_pop"]
        rows.append(row)
    return base.merge(pd.DataFrame(rows), on="userId")


def save_context(ctx: LiteContext, path: Path = CONTEXT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(ctx, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_context(path: Path = CONTEXT_PATH) -> "LiteContext | None":
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.UnpicklingError, AttributeError, EOFError):
        return None
