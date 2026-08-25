"""Baseline rating predictor — the source for the predicted_rating objective.

Uses the classical bias baseline r_hat(u,i) = mu + b_u + b_i with damped
biases. It is deliberately the simplest defensible predictor: the project is
scoped to explainable classical methods, and this model has the property that
matters most for ranking — it degrades gracefully. An unknown user falls back
to mu + b_i (the item's own standing), and an unknown item to mu + b_u, so
cold-start candidates still receive a sensible score instead of a hole.
"""

import numpy as np
import pandas as pd

_DAMPING = 10.0  # shrinks biases of thinly-rated users/items toward the global mean
_RATING_MIN = 0.5
_RATING_MAX = 5.0


class BaselinePredictor:
    """Predicts a user's rating for an item from global/user/item biases."""

    def __init__(self, damping: float = _DAMPING):
        self.damping = damping
        self.global_mean = 0.0
        self.item_bias: dict[int, float] = {}
        self.user_bias: dict[int, float] = {}

    def fit(self, ratings_df: pd.DataFrame) -> "BaselinePredictor":
        """Fit global mean and damped user/item biases from a ratings frame."""
        if ratings_df.empty:
            return self

        self.global_mean = float(ratings_df["rating"].mean())
        centered = ratings_df["rating"] - self.global_mean

        # Item bias first, then user bias on the item-adjusted residual: a user
        # who only ever rates acclaimed films is not a "generous rater", and
        # fitting user bias on the raw residual would wrongly record them as one.
        item_grouped = centered.groupby(ratings_df["movieId"])
        item_bias = item_grouped.sum() / (item_grouped.count() + self.damping)
        self.item_bias = {int(k): float(v) for k, v in item_bias.items()}

        residual = centered - ratings_df["movieId"].map(item_bias).fillna(0.0)
        user_grouped = residual.groupby(ratings_df["userId"])
        user_bias = user_grouped.sum() / (user_grouped.count() + self.damping)
        self.user_bias = {int(k): float(v) for k, v in user_bias.items()}

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicted rating on the original rating scale, clipped to [0.5, 5.0]."""
        prediction = (
            self.global_mean
            + self.user_bias.get(int(user_id), 0.0)
            + self.item_bias.get(int(movie_id), 0.0)
        )
        return float(np.clip(prediction, _RATING_MIN, _RATING_MAX))

    def predict_normalized(self, user_id: int, movie_id: int) -> float:
        """Objective 6: the predicted rating rescaled to [0, 1].

        The other five objectives are already [0, 1]; rescaling here keeps all
        six commensurate so the configured weights mean what they say, rather
        than predicted_rating silently dominating on a 0.5-5.0 scale.
        """
        prediction = self.predict(user_id, movie_id)
        return (prediction - _RATING_MIN) / (_RATING_MAX - _RATING_MIN)
