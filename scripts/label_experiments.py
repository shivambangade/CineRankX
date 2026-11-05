"""Compare label-generation methods: reliability, class mix and what a perfect model could score.

For each configuration the SAME users are labelled twice with independent random holdouts
(seeds 42 and 43). Agreement between the two labelings measures how repeatable a label is;
an oracle ceiling is estimated by simulation (see analyze_selector.noise_ceiling). The
uncertainty-aware variant additionally marks a user "indistinct" when the best strategy does not
beat the runner-up by a one-sided 95% margin (binomial standard error on the holdout size), so
uncertainty is represented instead of being turned into a confident label.

Usage: PYTHONPATH=. python scripts/label_experiments.py [n_users]
Writes data/processed/label_experiments.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage import DB_PATH, connect
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.dataset import load_ratings, sample_user_ids
from src.strategy_selector.formulations import winner
from src.strategy_selector.holdout_labeler import label_holdout, split_profile_holdout

warnings.filterwarnings("ignore")
SEED = 42
SCORES = ["cf_score", "cb_score", "pop_score"]       # alphabetical class order: collaborative, content, popularity
NAMES = ["collaborative", "content_based", "popularity", "indistinct"]
CONFIGS = {"old (10 recs, 20% holdout)": (10, 0.2), "current (30 recs, 30% holdout)": (30, 0.3),
           "larger (50 recs, 40% holdout)": (50, 0.4)}
Z_ONE_SIDED_95 = 1.645


def labels_for(ratings, movies, top_n, frac, seed) -> pd.DataFrame:
    profile, holdout = split_profile_holdout(ratings, frac, seed)
    return label_holdout(profile, holdout, movies, top_n).set_index("userId")


def classify(frame: pd.DataFrame, uncertainty_aware: bool) -> np.ndarray:
    s = frame[["cf_score", "cb_score", "pop_score"]].to_numpy()
    lab = winner(s)
    if not uncertainty_aware:
        return lab
    top2 = np.sort(s, axis=1)[:, -2:]
    se = np.sqrt((top2 * (1 - top2)).sum(axis=1) / frame["n_holdout"].to_numpy())
    lab = lab.copy()
    lab[(top2[:, 1] - top2[:, 0]) < Z_ONE_SIDED_95 * se] = 3
    return lab


def oracle_ceiling(frame: pd.DataFrame, draws: int = 100) -> float:
    rng = np.random.default_rng(SEED)
    p = frame[SCORES].to_numpy().clip(0.001, 0.999)
    m = frame["n_holdout"].to_numpy()
    true = winner(p)
    return float(np.mean([(winner(rng.binomial(m[:, None], p) / m[:, None]) == true).mean() for _ in range(draws)]))


def main() -> None:
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    config = load_selector_config()
    movies = pd.read_csv("data/processed/movies_merged.csv")
    conn = connect(DB_PATH)
    ids = sample_user_ids(conn, n_users, SEED, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(movies["movieId"].astype(int)))
    report = {}
    for name, (top_n, frac) in CONFIGS.items():
        a, b = labels_for(ratings, movies, top_n, frac, SEED), labels_for(ratings, movies, top_n, frac, SEED + 1)
        a = a.rename(columns={c: c for c in a.columns})
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        row = {"top_n": top_n, "holdout_fraction": frac, "users": int(len(common)),
               "oracle_ceiling_full_population": oracle_ceiling(a)}
        for kind, aware in (("winner_labels", False), ("uncertainty_aware", True)):
            la, lb = classify(a, aware), classify(b, aware)
            clear = (la != 3) & (lb != 3)
            row[kind] = {
                "class_mix": {NAMES[i]: float((la == i).mean()) for i in range(4) if (la == i).any()},
                "agreement_all_users": float((la == lb).mean()),
                "agreement_where_both_clear": float((la[clear] == lb[clear]).mean()) if clear.any() else None,
                "share_clear_in_both": float(clear.mean())}
        report[name] = row
        w, u = row["winner_labels"], row["uncertainty_aware"]
        print(f"{name}: winner-label agreement {w['agreement_all_users']:.3f}, oracle ceiling "
              f"{row['oracle_ceiling_full_population']:.3f}, mix {{{', '.join(f'{k} {v:.2f}' for k, v in w['class_mix'].items())}}}; "
              f"uncertainty-aware: indistinct {u['class_mix'].get('indistinct', 0):.2f}, agreement where both clear "
              f"{u['agreement_where_both_clear']:.3f} ({u['share_clear_in_both']:.0%} of users)", flush=True)
    Path("data/processed/label_experiments.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
