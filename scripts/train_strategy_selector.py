"""Train and honestly evaluate the Adaptive Strategy Selector.

Pipeline: sample users (fixed seed) -> per-user profile/holdout split -> features from the
profile only (incl. strategy-aware inner backtest scores) + strategy backtest on the holdout
-> drop ambiguous labels -> hold out 20% of users untouched -> tuned 5-fold CV comparison of
eight model families on the rest (selected by macro-F1) -> one-shot held-out evaluation
-> refit on every labelled user and save model + context.

Usage:
    PYTHONPATH=. python scripts/train_strategy_selector.py [n_users] [--refresh] [--no-tune] [--reliability]

--reliability re-labels the same users with a second independent holdout and reports how often
the two labels agree: the ceiling any classifier can reach on these labels.
Writes data/processed/strategy_eval.json (read by scripts/generate_kpi_report.py).
"""

import json
import warnings
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.evaluation.ml_metrics import to_report_names
from src.storage import DB_PATH, connect
from src.strategy_selector import StrategyClassifier
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.dataset import build_labelled_frame, cache_path, load_ratings, sample_user_ids
from src.strategy_selector.holdout_labeler import SCORE_COLUMNS, clean_labels, label_holdout, split_profile_holdout
from src.strategy_selector.model_selection import (CANDIDATES, SEED, build_estimator, compare_candidates,
                                                   cross_validate, evaluate_holdout)
from src.strategy_selector.strategy_context import CONTEXT_PATH, build_features, load_context

warnings.filterwarnings("ignore", category=FutureWarning)
EVAL_PATH = Path("data/processed/strategy_eval.json")
NON_FEATURES = ["userId", "best_strategy", "n_holdout", *SCORE_COLUMNS]
STRATEGY_AWARE = ["inner_cb", "inner_cf", "inner_pop", "inner_gap_cb_pop", "inner_gap_cf_pop",
                  "top100_share", "top1000_share"]


def label_reliability(conn, movies, n_users, config) -> dict:
    """Agreement between labels from two independent holdouts of the same users."""
    ids = sample_user_ids(conn, n_users, SEED, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(movies["movieId"].astype(int)))
    kept = {}
    for seed in (SEED, SEED + 1):
        profile, holdout = split_profile_holdout(ratings, config["holdout_fraction"], seed)
        labels, _ = clean_labels(label_holdout(profile, holdout, movies, config["label_top_n"]),
                                 config["label_margin"])
        kept[seed] = labels.set_index("userId")["best_strategy"]
    common = kept[SEED].index.intersection(kept[SEED + 1].index)
    agree = float((kept[SEED].loc[common] == kept[SEED + 1].loc[common]).mean())
    return {"users_kept_in_both": int(len(common)), "label_agreement": agree}


def inference_latency(clf, ctx, movies, conn, config, n=100) -> dict:
    ids = sample_user_ids(conn, n, SEED + 5, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(movies["movieId"].astype(int)))
    timings = []
    for _, group in ratings.groupby("userId"):
        started = time.perf_counter()
        feats = build_features(group, movies, ctx, config["inner_splits"], config["inner_fraction"])
        clf.predict(feats)
        timings.append((time.perf_counter() - started) * 1000)
    return {"mean_ms": float(np.mean(timings)), "p95_ms": float(np.percentile(timings, 95)), "users": len(timings)}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(args[0]) if args else 5000
    config = load_selector_config()
    movies = pd.read_csv("data/processed/movies_merged.csv")
    conn = connect(DB_PATH)
    frame = build_labelled_frame(conn, movies, n_users, SEED, config, refresh="--refresh" in sys.argv)

    kept, drop = clean_labels(frame, config["label_margin"])
    print(f"\nLabelled {drop['total']} users; kept {drop['kept']}, dropped {drop['dropped_all_zero']} "
          f"(all strategies scored 0) and {drop['dropped_ambiguous']} (top-two within margin {drop['margin']})")
    print("Cleaned label mix:", kept["best_strategy"].value_counts().to_dict())

    feature_cols = [c for c in kept.columns if c not in NON_FEATURES]
    base_cols = [c for c in feature_cols if c not in STRATEGY_AWARE]
    encoder = LabelEncoder().fit(kept["best_strategy"])
    y, names = encoder.transform(kept["best_strategy"]), list(encoder.classes_)
    X = kept[feature_cols].fillna(0)
    X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_dev = X_dev.reset_index(drop=True)
    print(f"Development users: {len(y_dev)}   untouched held-out users: {len(y_hold)}")

    ablation = {}
    for label, cols in (("profile features only", base_cols), ("profile + strategy-aware", feature_cols)):
        res = cross_validate("rf_balanced", X_dev[cols], y_dev, names, config["cv_folds"])
        ablation[label] = {"cv_macro_f1": res["cv_macro_f1"], "cv_balanced_accuracy": res["cv_balanced_accuracy"],
                           "cv_accuracy": res["cv_accuracy"], "n_features": len(cols)}
        print(f"Ablation (rf_balanced) {label:28s} macro-F1 {res['cv_macro_f1']:.3f}  acc {res['cv_accuracy']:.3f}")

    tune = "--no-tune" not in sys.argv
    print(f"\n{config['cv_folds']}-fold CV, {'tuned ' if tune else ''}candidates (selected by macro-F1):")
    started = time.time()
    table, best, tuned = compare_candidates(X_dev, y_dev, names, config["cv_folds"], tune=tune)
    print(f"Selected {best} (comparison took {time.time() - started:.0f}s)")

    started = time.time()
    held = evaluate_holdout(best, X_dev, y_dev, X_hold, y_hold, names, tuned[best])
    train_seconds = time.time() - started
    print(f"\nHeld-out ({held['n_test']} users, scored once, model={best}):")
    print(f"  accuracy {held['accuracy']:.3f} (majority baseline {held['majority_baseline_accuracy']:.3f})  "
          f"balanced {held['balanced_accuracy']:.3f}  macro-F1 {held['macro_f1']:.3f}  AUC {held.get('roc_auc', 0):.3f}")
    for name, m in held["per_class"].items():
        print(f"  {name:14s} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  n={m['support']}")
    print(f"  confusion {held['confusion_matrix_labels']}: {held['confusion_matrix']}")

    clf = StrategyClassifier()
    clf.fit_full(kept[["userId", *feature_cols]], kept[["userId", "best_strategy"]],
                 build_estimator(best, tuned[best]), CANDIDATES[best][1])
    clf.save()
    ctx_src = cache_path(n_users, SEED, config).with_suffix(".context.pkl")
    shutil.copyfile(ctx_src, CONTEXT_PATH)
    ctx = load_context()
    latency = inference_latency(clf, ctx, movies, conn, config)
    print(f"Inference (features + predict, per user): mean {latency['mean_ms']:.1f} ms, p95 {latency['p95_ms']:.1f} ms")

    reliability = label_reliability(conn, movies, min(n_users, 3000), config) if "--reliability" in sys.argv else None
    if reliability:
        print(f"Label reliability: two independent holdouts agree on {reliability['label_agreement']:.3f} of "
              f"{reliability['users_kept_in_both']} users (ceiling for any classifier)")

    report = to_report_names(held)
    report.update({k: v for k, v in held.items() if k not in report})
    EVAL_PATH.write_text(json.dumps({
        "machine_learning": report, "selected_model": best, "tuned_params": tuned, "cv": table,
        "ablation": ablation, "label_cleaning": drop, "label_reliability": reliability,
        "latency_ms": latency, "holdout_fit_seconds": train_seconds,
        "n_users_requested": n_users, "n_dev": int(len(y_dev)), "n_holdout": int(len(y_hold)),
        "features": feature_cols, "config": config}, indent=2,
        default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(f"\nSaved model, context and {EVAL_PATH}")


if __name__ == "__main__":
    main()
