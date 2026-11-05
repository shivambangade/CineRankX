"""Build (and cache) the labelled training frame: sampled users -> features + backtest scores."""

import time
from pathlib import Path

import pandas as pd

from src.strategy_selector.config import load_selector_config
from src.strategy_selector.content_model import ContentModel
from src.strategy_selector.holdout_labeler import label_holdout, split_profile_holdout
from src.strategy_selector.strategy_context import LiteContext, build_features, save_context

CACHE_DIR = Path("data/processed")
MAX_USERS = 5000  # the collaborative backtest bounds its user-user matrix at this size


def sample_user_ids(conn, n_users: int, seed: int, min_history: int, max_history: int) -> list[int]:
    """Random users with enough (but not extreme) history, from the indexed user_stats table."""
    frame = pd.read_sql_query(
        "SELECT userId FROM user_stats WHERE n_ratings BETWEEN ? AND ?", conn,
        params=(int(min_history), int(max_history)))
    take = min(int(n_users), len(frame), MAX_USERS)
    return sorted(int(u) for u in frame["userId"].sample(n=take, random_state=seed))


def load_ratings(conn, user_ids: list[int], catalog_ids: set[int]) -> pd.DataFrame:
    """Ratings of the sampled users, restricted to catalog movies (the strategies' universe)."""
    marks = ",".join("?" * len(user_ids))
    frame = pd.read_sql_query(
        f"SELECT userId, movieId, rating, ts AS timestamp FROM ratings WHERE userId IN ({marks})",
        conn, params=[int(u) for u in user_ids])
    return frame[frame["movieId"].isin(catalog_ids)].reset_index(drop=True)


def cache_path(n_users: int, seed: int, config: dict, cache_dir: Path = CACHE_DIR) -> Path:
    tag = (f"v3_n{n_users}_s{seed}_h{config['holdout_fraction']}_t{config['label_top_n']}"
           f"_min{config['min_history']}_max{config['max_history']}_i{config['inner_splits']}"
           f"_c{'base' if not config.get('content_options') else 'rq'}")
    return cache_dir / f"strategy_labels_{tag}.csv"


def build_labelled_frame(conn, movies_df: pd.DataFrame, n_users: int, seed: int = 42,
                         config: dict | None = None, cache_dir: Path = CACHE_DIR,
                         refresh: bool = False, log=print) -> pd.DataFrame:
    """Features (profile ratings only) + raw backtest scores per user, cached on disk.

    Margin-based cleaning is applied by the caller, so changing label_margin in
    eval_config.yaml never forces a re-label.
    """
    config = config or load_selector_config()
    path = cache_path(n_users, seed, config, cache_dir)
    if path.exists() and not refresh:
        log(f"Using cached labelled frame {path}")
        return pd.read_csv(path)

    ids = sample_user_ids(conn, n_users, seed, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(movies_df["movieId"].astype(int)))
    counts = ratings.groupby("userId").size()
    ratings = ratings[ratings["userId"].isin(counts[counts >= config["min_history"]].index)]
    log(f"Sampled {len(ids)} users; {ratings['userId'].nunique()} keep >= {config['min_history']} "
        f"catalog ratings ({len(ratings):,} ratings)")

    profile, holdout = split_profile_holdout(ratings, config["holdout_fraction"], seed)
    started = time.time()
    content = ContentModel(movies_df, **config["content_options"]) if config.get("content_options") else None
    labels = label_holdout(profile, holdout, movies_df, config["label_top_n"], content)
    log(f"Labelled {len(labels)} users in {time.time() - started:.0f}s")
    started = time.time()
    ctx = LiteContext(profile, movies_df, top_n=config["label_top_n"], content_model=content)
    features = build_features(profile, movies_df, ctx, config["inner_splits"], config["inner_fraction"])
    log(f"Built strategy-aware features in {time.time() - started:.0f}s")
    save_context(ctx, path.with_suffix(".context.pkl"))
    frame = features.merge(labels, on="userId")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
