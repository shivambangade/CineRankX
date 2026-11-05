"""FROZEN baseline strategy. Do not edit: any change creates a different benchmark.

This module is deliberately self-contained: it imports nothing from the ranking engine,
strategy selector or UI, so training, tuning or changing CineRankX cannot alter it.
It has no learnable parameters beyond counting the benchmark's training ratings.

  recommend : non-personalised "most popular" -- movies ranked by number of TRAIN
              ratings (ties broken by movieId), minus what the user has already seen.
  predict   : item mean rating, shrunk toward the global mean with a fixed damping.

run_benchmark.py records a SHA-256 of this file next to the baseline's results and
refuses to continue if either the code or the measured numbers differ from the record.
"""

import numpy as np
import pandas as pd

BASELINE_VERSION = "popularity-itemmean-v1"
RATING_DAMPING = 10.0
RATING_MIN, RATING_MAX = 0.5, 5.0


class FrozenBaseline:
    def __init__(self, train: pd.DataFrame):
        counts = train.groupby("movieId").size()
        order = np.lexsort((counts.index.to_numpy(), -counts.to_numpy()))
        self.ranking = [int(m) for m in counts.index.to_numpy()[order]]
        self.global_mean = float(train["rating"].mean())
        grouped = train.groupby("movieId")["rating"]
        sums, n = grouped.sum(), grouped.count()
        self.item_mean = ((sums + RATING_DAMPING * self.global_mean) / (n + RATING_DAMPING)).to_dict()

    def recommend(self, seen: set[int], k: int) -> list[int]:
        out = []
        for movie in self.ranking:
            if movie not in seen:
                out.append(movie)
                if len(out) == k:
                    break
        return out

    def predict(self, movie_id: int) -> float:
        return float(np.clip(self.item_mean.get(int(movie_id), self.global_mean), RATING_MIN, RATING_MAX))
