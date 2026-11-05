"""Does a better content representation actually raise held-out hit rate?

Hypothesis: the content-based strategy is weak (hit rate ~0.016) because its representation
(2,000-term TF-IDF, top-20-rated profile, no quality signal) is poor, not because content is
useless. Protocol: the label backtest (70% profile / 30% holdout per user, 30 recommendations);
variants are chosen on a DEVELOPMENT half of users and confirmed on a disjoint half.

Usage: PYTHONPATH=. python scripts/content_experiments.py [n_users]
Writes data/processed/content_experiments.json.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage import DB_PATH, connect
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.content_model import ContentModel
from src.strategy_selector.dataset import load_ratings, sample_user_ids
from src.strategy_selector.holdout_labeler import split_profile_holdout
from src.strategy_selector.stats import bootstrap_ci, paired_bootstrap
from src.strategy_selector.strategy_context import LiteContext

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("data/processed/content_experiments.json")
VARIANTS = {
    "base (original)": {},
    "liked-centred profile": {"profile": "liked_centered"},
    "rich TF-IDF (bigrams, 20k)": {"vectorizer": "rich"},
    "rich + liked-centred": {"vectorizer": "rich", "profile": "liked_centered"},
    "genre weight 0.5": {"genre_weight": 0.5},
    "genre weight 1.0": {"genre_weight": 1.0},
    "quality prior a=0.5": {"quality_alpha": 0.5},
    "quality prior a=1.0": {"quality_alpha": 1.0},
    "quality prior a=2.0": {"quality_alpha": 2.0},
    "quality prior a=3.0": {"quality_alpha": 3.0},
    "rich + quality a=1.0": {"vectorizer": "rich", "quality_alpha": 1.0},
    "rich + quality a=2.0": {"vectorizer": "rich", "quality_alpha": 2.0},
    "rich + genre 0.5 + quality a=2.0": {"vectorizer": "rich", "genre_weight": 0.5, "quality_alpha": 2.0},
    "liked + quality a=1": {"profile": "liked_centered", "quality_alpha": 1.0},
    "rich + liked + genre 0.5": {"vectorizer": "rich", "profile": "liked_centered", "genre_weight": 0.5},
    "rich + liked + genre 0.5 + quality 1": {"vectorizer": "rich", "profile": "liked_centered",
                                              "genre_weight": 0.5, "quality_alpha": 1.0},
}


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    config = load_selector_config()
    movies = pd.read_csv("data/processed/movies_merged.csv")
    conn = connect(DB_PATH)
    ids = sample_user_ids(conn, n_users, SEED + 11, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(movies["movieId"].astype(int)))
    profile, holdout = split_profile_holdout(ratings, config["holdout_fraction"], SEED)
    top_n = config["label_top_n"]
    users = sorted(profile["userId"].unique())
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(users))
    dev = {users[i] for i in order[: len(users) // 2]}
    by_user = {u: g for u, g in profile.groupby("userId")}
    truth = {u: g["movieId"].to_numpy() for u, g in holdout.groupby("userId")}
    print(f"{len(users)} users ({len(dev)} dev / {len(users) - len(dev)} confirm), top-{top_n} budget")

    ctx = LiteContext(profile, movies, top_n=top_n)
    refs = {"popularity (reference)": [len(set(truth[u]) & ctx.top_popular_movies) / len(truth[u]) for u in users],
            "collaborative (reference)": [ctx.cf_hit_rate(u, by_user[u], truth[u]) for u in users]}
    results = {}
    for name, options in VARIANTS.items():
        started = time.time()
        model = ContentModel(movies, **options)
        results[name] = [model.hit_rate(by_user[u], truth[u], top_n) for u in users]
        dv = np.mean([r for u, r in zip(users, results[name]) if u in dev])
        print(f"  {name:40s} dev hit-rate {dv:.4f}  ({time.time() - started:.0f}s)", flush=True)

    is_dev = np.array([u in dev for u in users])
    best = max(results, key=lambda n: np.mean(np.array(results[n])[is_dev]))
    base = np.array(results["base (original)"])[~is_dev]
    report = {"n_users": len(users), "top_n": top_n, "selected_on_dev": best, "variants": {}}
    print(f"\nSelected on development users: {best}\nConfirmation on {int((~is_dev).sum())} unseen users:")
    for name, vals in {**results, **refs}.items():
        conf = np.array(vals)[~is_dev]
        mean, lo, hi = bootstrap_ci(conf)
        entry = {"confirm_mean": mean, "ci95": [lo, hi], "dev_mean": float(np.array(vals)[is_dev].mean())}
        if name in results and name != "base (original)":
            entry["vs_base"] = paired_bootstrap(conf, base)
        report["variants"][name] = entry
        extra = f"  vs base {entry['vs_base']['mean_diff']:+.4f} [{entry['vs_base']['ci_low']:+.4f}, {entry['vs_base']['ci_high']:+.4f}]" if "vs_base" in entry else ""
        print(f"  {name:40s} {mean:.4f} [{lo:.4f}, {hi:.4f}]{extra}")
    report["selected_options"] = VARIANTS[best]
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
