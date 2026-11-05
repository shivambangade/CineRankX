"""CineRankX as a benchmark strategy: EASE relevance -> six-objective re-ranking.

Everything here is trained and tuned on the benchmark's TRAIN and VALIDATION splits
only. The frozen baseline never imports this module.
"""

import numpy as np
import pandas as pd

from src.ranking_engine import MultiObjectiveRanker
from src.ranking_engine.config import OBJECTIVES, RankingWeights
from src.ranking_engine.ease import EaseModel
from src.ranking_engine.objectives import CatalogStats
from src.ranking_engine.predictor import BaselinePredictor

POOL_SIZE = 200


class CineRankXStrategy:
    def __init__(self, bench, movies: pd.DataFrame, ir_engine, n_items: int = 6000):
        self.bench, self.movies, self.ir_engine, self.n_items = bench, movies, ir_engine, n_items
        members = np.concatenate(list(bench.cohorts.values()))
        sub = bench.train[bench.train["userId"].isin(members)]
        self.profiles = {int(u): g["movieId"].tolist() for u, g in sub.groupby("userId")}
        self.stats = CatalogStats(movies, bench.train)
        self.predictor = BaselinePredictor().fit(bench.train)
        self.ranker = MultiObjectiveRanker(movies, bench.train, ir_engine=ir_engine, predictor=self.predictor,
                                           stats=self.stats, build_generator=False)
        self.ease: EaseModel | None = None
        self.beta = {"overall": 1.0, "cold": 0.7}
        self.weights: dict[str, RankingWeights | None] = {"overall": None, "cold": None}

    def fit(self, l2: float) -> None:
        self.ease = EaseModel.fit(self.bench.train, self.n_items, l2)

    def pools(self, users: list[int], exclude: dict[int, set[int]], beta: float) -> dict[int, pd.DataFrame]:
        scores = self.ease.blended_scores(self.ease.user_matrix(self.profiles, users), beta)
        for i, u in enumerate(users):
            for m in exclude.get(u, ()):
                c = self.ease.index.get(m)
                if c is not None:
                    scores[i, c] = -np.inf
        top = np.argpartition(-scores, POOL_SIZE, axis=1)[:, :POOL_SIZE]
        out = {}
        for i, u in enumerate(users):
            idx = top[i][np.argsort(-scores[i, top[i]])]
            rel = scores[i, idx]
            rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-12)
            out[u] = pd.DataFrame({"movieId": self.ease.item_ids[idx], "relevance": rel, "sources": "ease"})
        return out

    def recommend_many(self, users: list[int], exclude: dict[int, set[int]], k: int,
                       cohort: str = "overall", beta: float | None = None,
                       weights: RankingWeights | None | bool = True) -> dict[int, list[int]]:
        """weights=True -> tuned weights for the cohort; None -> plain relevance order (no re-rank)."""
        pools = self.pools(users, exclude, self.beta[cohort] if beta is None else beta)
        active = self.weights[cohort] if weights is True else weights
        if active is None:
            return {u: [int(m) for m in p["movieId"].head(k)] for u, p in pools.items()}
        return {u: [int(m) for m in self.ranker.rank(u, k=k, candidates=p, weights=active)["movieId"]]
                for u, p in pools.items()}

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.predictor.predict(user_id, movie_id)


def weights_from(vector: dict[str, float]) -> RankingWeights:
    return RankingWeights({name: float(vector.get(name, 0.0)) for name in OBJECTIVES})
