"""Backtest strategies to determine the best-fit strategy per user.

Performance note: hit-rates are computed against structures built ONCE for
the whole batch (global TF-IDF matrix, a sparse user-item matrix, one
user-user similarity matrix, one popularity ranking) rather than rebuilt from
scratch inside the per-user loop. The original per-user version re-fit a
TfidfVectorizer and rebuilt full rating-vector dicts for up to 500 users on
every iteration, which made it O(users × movies) and timed out past ~1000
users; this version is O(users) after the shared setup.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_CF_MAX_USERS = 5000  # bounds the O(n^2) user-user similarity matrix
_CF_MAX_MOVIES = 2000  # bounds user-item matrix width (most-rated movies)
_CB_MAX_FEATURES = 2000  # TF-IDF vocabulary size for the content-based backtest
_CB_PROFILE_MAX_MOVIES = 20  # cap on how many trained movies build a CB profile


def select_best_strategy(cb_score: float, cf_score: float, pop_score: float) -> str:
    """Pick the winning strategy from backtested hit-rates.

    On a tie, prefer popularity: the same fallback used when a user has too
    little history to backtest at all. A tie most commonly means every score
    is 0.0 (no personalized strategy beat blind guessing), so defaulting to
    the safe global baseline is correct — not an arbitrary pick.
    """
    scores = {"content_based": cb_score, "collaborative": cf_score, "popularity": pop_score}
    max_score = max(scores.values())
    tied = [strategy for strategy, score in scores.items() if score == max_score]
    return "popularity" if "popularity" in tied else tied[0]


class _BacktestContext:
    """Structures shared across all users' backtests, built once per batch."""

    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self._build_popularity(ratings_df, movies_df)
        self._build_content_based(movies_df)
        self._build_collaborative(ratings_df)

    def _build_popularity(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        # OBSERVED rating counts first. The original code preferred TMDB's
        # `popularity` column to avoid a naive mean-rating top-10 (where one
        # 5.0 rating outranks a film rated 4.3 by 10,000 people) -- a real
        # concern, but rating COUNTS solve it just as well while staying native
        # to the dataset. TMDB popularity is live and recency-weighted whereas
        # MovieLens 20M stops in 2015: its top-10 is touched by only 39% of
        # users and covers 0.39% of ratings, against 87% and 2.87% for the
        # most-rated top-10. Backtesting the popularity strategy against a list
        # most users never touch drove pop_score to 0.0 for the majority, so
        # the `popularity` label was largely assigned by the all-zero tie-break
        # in select_best_strategy() rather than by winning on merit.
        if ratings_df is not None and not ratings_df.empty:
            counts = ratings_df["movieId"].value_counts().head(10)
            self.top_popular_movies = {int(m) for m in counts.index}
            return

        if "popularity" in movies_df.columns and "movieId" in movies_df.columns:
            top = movies_df.nlargest(10, "popularity")["movieId"]
            self.top_popular_movies = set(top)
            return

        # Neither ratings nor TMDB metadata: nothing can be called popular.
        # (The previous Bayesian volume-weighted-average branch became
        # unreachable once rating counts took priority -- it needed ratings,
        # but is only reached when there are none.)
        self.top_popular_movies = set()

    def _build_content_based(self, movies_df: pd.DataFrame) -> None:
        self.tfidf_matrix = None
        self.movie_id_to_row = {}
        if "text_chunk_clean" not in movies_df.columns or movies_df.empty:
            return
        texts = movies_df["text_chunk_clean"].fillna("")
        vectorizer = TfidfVectorizer(max_features=_CB_MAX_FEATURES)
        self.tfidf_matrix = vectorizer.fit_transform(texts)
        self.tfidf_movie_ids = movies_df["movieId"].values
        self.movie_id_to_row = {mid: i for i, mid in enumerate(self.tfidf_movie_ids)}

    def _build_collaborative(self, ratings_df: pd.DataFrame) -> None:
        self.user_similarity = None
        self.user_id_to_idx = {}
        self.user_ids = np.array([])
        self.user_top_movies = {}

        sampled_user_ids = ratings_df["userId"].unique()[:_CF_MAX_USERS]
        sub = ratings_df[ratings_df["userId"].isin(sampled_user_ids)]
        if sub["userId"].nunique() < 2:
            return

        # Similarity is computed over a bounded set of the most-rated movies —
        # a standard dimensionality reduction for the O(users^2) similarity
        # matrix. Recommendation candidates below are drawn from each user's
        # FULL history instead, not this bounded set: restricting both to the
        # same popular-movie pool would make CF's hit-rate partly re-measure
        # "is this a popular movie," confounding it with the popularity baseline.
        top_movie_ids = sub["movieId"].value_counts().nlargest(_CF_MAX_MOVIES).index
        sim_sub = sub[sub["movieId"].isin(top_movie_ids)]
        if sim_sub.empty:
            return

        self.user_ids = sim_sub["userId"].unique()
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        movie_id_to_col = {mid: i for i, mid in enumerate(top_movie_ids)}

        row_idx = sim_sub["userId"].map(self.user_id_to_idx).values
        col_idx = sim_sub["movieId"].map(movie_id_to_col).values
        user_item = csr_matrix(
            (sim_sub["rating"].values, (row_idx, col_idx)),
            shape=(len(self.user_ids), len(top_movie_ids)),
        )
        self.user_similarity = cosine_similarity(user_item)

        self.user_top_movies = (
            sub[sub["userId"].isin(self.user_ids)]
            .sort_values("rating", ascending=False)
            .groupby("userId")["movieId"]
            .apply(lambda s: set(s.head(5)))
            .to_dict()
        )


def _build_cb_profile_vector(ctx: _BacktestContext, train_ratings: pd.DataFrame):
    """Build a user's content-based profile vector.

    Uses only the top rated-N trained movies, weighted by rating, instead of
    summing the whole history. For heavy raters (hundreds of movies) an
    unweighted full-history sum blurs the profile into a generic "watches
    everything" blob and drowns out the genre-specific signal that would
    actually identify a content-based fan. Returns None if no trained movie
    is present in the TF-IDF matrix.
    """
    top_rated = train_ratings.nlargest(_CB_PROFILE_MAX_MOVIES, "rating")
    rows, weights = [], []
    for mid, rating in zip(top_rated["movieId"], top_rated["rating"]):
        row = ctx.movie_id_to_row.get(mid)
        if row is not None:
            rows.append(row)
            weights.append(rating)
    if not rows:
        return None

    weights = np.array(weights).reshape(-1, 1)
    return np.asarray(ctx.tfidf_matrix[rows].multiply(weights).sum(axis=0))


def _content_based_hit_rate(ctx: _BacktestContext, train_ratings: pd.DataFrame, test_movie_ids) -> float:
    if ctx.tfidf_matrix is None or len(test_movie_ids) == 0 or train_ratings.empty:
        return 0.0

    trained_set = set(train_ratings["movieId"])

    profile_vec = _build_cb_profile_vector(ctx, train_ratings)
    if profile_vec is None:
        return 0.0

    similarities = cosine_similarity(profile_vec, ctx.tfidf_matrix)[0]

    # Exclude ALL trained movies (not just the top-N used for the profile)
    # from candidates -- never recommend something the user already rated.
    for mid in trained_set:
        row = ctx.movie_id_to_row.get(mid)
        if row is not None:
            similarities[row] = -1.0

    top_indices = np.argsort(similarities)[::-1][:10]
    top_movie_ids = set(ctx.tfidf_movie_ids[top_indices])

    hits = len(set(test_movie_ids) & top_movie_ids)
    return hits / len(test_movie_ids)


def _collaborative_hit_rate(ctx: _BacktestContext, user_id, test_movie_ids) -> float:
    if ctx.user_similarity is None or user_id not in ctx.user_id_to_idx or len(test_movie_ids) == 0:
        return 0.0

    uidx = ctx.user_id_to_idx[user_id]
    sims = ctx.user_similarity[uidx].copy()
    sims[uidx] = -1.0  # exclude self
    top_similar = np.argsort(sims)[::-1][:5]

    # Union of top-5 movies from 5 similar users could offer up to 25
    # candidates — an unfair advantage over content-based and popularity,
    # which each only get 10. Score every candidate by the similarity of the
    # best neighbor proposing it, then keep only the top 10, matching their
    # candidate budget so all three strategies compete on equal footing.
    candidate_scores: dict = {}
    for idx in top_similar:
        sim_score = sims[idx]
        if sim_score <= 0:
            continue
        other_uid = ctx.user_ids[idx]
        for mid in ctx.user_top_movies.get(other_uid, set()):
            candidate_scores[mid] = max(candidate_scores.get(mid, -1.0), sim_score)

    if not candidate_scores:
        return 0.0

    top_movie_ids = set(sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:10])

    hits = len(set(test_movie_ids) & top_movie_ids)
    return hits / len(test_movie_ids)


def _popularity_hit_rate(ctx: _BacktestContext, test_movie_ids) -> float:
    if len(test_movie_ids) == 0:
        return 0.0
    hits = len(set(test_movie_ids) & ctx.top_popular_movies)
    return hits / len(test_movie_ids)


_DEFAULT_N_SPLITS = 5  # random train/test splits averaged per user


def label_users_with_best_strategy(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    test_fraction: float = 0.2,
    n_splits: int = _DEFAULT_N_SPLITS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Label each user with their best-performing strategy via backtesting.

    Each user's hit-rate is averaged over `n_splits` independent random
    train/test splits rather than a single positional split. A single split
    with a small test set (often 1-5 ratings) is a high-variance one-shot
    estimate: which strategy happens to "win" can flip between splits purely
    from which ratings landed in the test set, especially for a rare label
    like content_based. Averaging over several splits smooths that noise out
    before the argmax in select_best_strategy() is taken.

    Strategies: "content_based", "collaborative", "popularity"

    Returns:
        DataFrame with columns [userId, best_strategy, cb_score, cf_score, pop_score]
        where the score columns are the mean hit-rate across splits.
    """
    ctx = _BacktestContext(ratings_df, movies_df)
    rng = np.random.default_rng(random_state)

    user_labels = []
    for user_id, user_ratings in ratings_df.groupby("userId"):
        n = len(user_ratings)
        if n < 3:
            user_labels.append(
                {
                    "userId": user_id,
                    "best_strategy": "popularity",
                    "cb_score": 0.0,
                    "cf_score": 0.0,
                    "pop_score": 1.0,
                }
            )
            continue

        train_size = max(1, int(n * (1 - test_fraction)))
        cb_scores, cf_scores, pop_scores = [], [], []

        for _ in range(n_splits):
            shuffled = user_ratings.iloc[rng.permutation(n)]
            train_ratings = shuffled.iloc[:train_size]
            test_ratings = shuffled.iloc[train_size:]
            if len(test_ratings) == 0:
                test_ratings = shuffled.iloc[-1:]

            test_movie_ids = test_ratings["movieId"].values
            cb_scores.append(_content_based_hit_rate(ctx, train_ratings, test_movie_ids))
            cf_scores.append(_collaborative_hit_rate(ctx, user_id, test_movie_ids))
            pop_scores.append(_popularity_hit_rate(ctx, test_movie_ids))

        cb_score = float(np.mean(cb_scores))
        cf_score = float(np.mean(cf_scores))
        pop_score = float(np.mean(pop_scores))

        best_strategy = select_best_strategy(cb_score, cf_score, pop_score)

        user_labels.append(
            {
                "userId": user_id,
                "best_strategy": best_strategy,
                "cb_score": cb_score,
                "cf_score": cf_score,
                "pop_score": pop_score,
            }
        )

    return pd.DataFrame(user_labels)
