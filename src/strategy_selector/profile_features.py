"""Extract user profile features from rating history."""

import numpy as np
import pandas as pd


def extract_profile_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for each user from their rating history and movie metadata.

    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        movies_df: DataFrame with columns [movieId, genres, ...]

    Returns:
        DataFrame with columns [userId, num_ratings, avg_rating, rating_std,
        num_genres, genre_diversity, top_genre_share, genre_entropy, ...]

        top_genre_share and genre_entropy capture WHICH genres a user watches
        (taste concentration), not just how many distinct genres or how that
        count scales with activity -- num_genres/genre_diversity alone can't
        tell a focused single-genre fan apart from an eclectic browser who
        happens to have rated a similar number of movies.
    """
    if ratings_df.empty or movies_df.empty:
        return pd.DataFrame()

    features_list = []

    for user_id in ratings_df["userId"].unique():
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        rated_movie_ids = user_ratings["movieId"].values

        num_ratings = len(user_ratings)
        avg_rating = user_ratings["rating"].mean()
        rating_std = user_ratings["rating"].std()
        rating_min = user_ratings["rating"].min()
        rating_max = user_ratings["rating"].max()

        rated_movies = movies_df[movies_df["movieId"].isin(rated_movie_ids)]
        num_genres = 0
        genre_diversity = 0
        top_genre_share = 0.0
        genre_entropy = 0.0

        if not rated_movies.empty and "genres" in rated_movies.columns:
            genre_counts: dict = {}
            for genres_str in rated_movies["genres"].dropna():
                if isinstance(genres_str, str):
                    for genre in genres_str.split("|"):
                        genre = genre.strip()
                        if genre:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            num_genres = len(genre_counts)
            genre_diversity = num_genres / max(num_ratings, 1)

            if genre_counts:
                counts = np.array(list(genre_counts.values()), dtype=float)
                proportions = counts / counts.sum()
                top_genre_share = float(proportions.max())
                raw_entropy = float(-np.sum(proportions * np.log(proportions)))
                # Normalize to [0, 1]: 0 = concentrated on one genre, 1 = spread
                # evenly across every genre the user has touched.
                max_entropy = np.log(num_genres) if num_genres > 1 else 1.0
                genre_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        rating_variance = rating_std if pd.notna(rating_std) else 0.0

        features_list.append(
            {
                "userId": user_id,
                "num_ratings": num_ratings,
                "avg_rating": avg_rating,
                "rating_std": rating_variance,
                "rating_min": rating_min,
                "rating_max": rating_max,
                "num_genres": num_genres,
                "genre_diversity": genre_diversity,
                "top_genre_share": top_genre_share,
                "genre_entropy": genre_entropy,
            }
        )

    return pd.DataFrame(features_list)
