"""Train and honestly evaluate the Adaptive Strategy Selector.

Pipeline: sample users (fixed seed) -> per-user profile/holdout split -> features from
the profile only + strategy backtest on the holdout -> drop ambiguous labels -> hold out
20% of users untouched -> 5-fold CV model comparison on the rest (selected by macro-F1)
-> one-shot held-out evaluation -> refit on every labelled user and save.

Usage:
    PYTHONPATH=. python scripts/train_strategy_selector.py [n_users] [--refresh]

Writes data/processed/strategy_eval.json (read by scripts/generate_kpi_report.py).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.evaluation.ml_metrics import to_report_names
from src.storage import DB_PATH, connect
from src.strategy_selector import StrategyClassifier
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.dataset import build_labelled_frame
from src.strategy_selector.holdout_labeler import SCORE_COLUMNS, clean_labels
from src.strategy_selector.model_selection import (CANDIDATES, SEED, compare_candidates,
                                                   evaluate_holdout)

EVAL_PATH = Path("data/processed/strategy_eval.json")
NON_FEATURES = ["userId", "best_strategy", "n_holdout", *SCORE_COLUMNS]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(args[0]) if args else 4000
    config = load_selector_config()
    movies = pd.read_csv("data/processed/movies_merged.csv")
    frame = build_labelled_frame(connect(DB_PATH), movies, n_users, SEED, config,
                                 refresh="--refresh" in sys.argv)

    kept, drop = clean_labels(frame, config["label_margin"])
    print(f"\nLabelled {drop['total']} users; kept {drop['kept']}, dropped "
          f"{drop['dropped_all_zero']} (all strategies scored 0) and {drop['dropped_ambiguous']} "
          f"(top-two within margin {drop['margin']})")
    print("Raw label mix:     ", frame["best_strategy"].value_counts().to_dict())
    print("Cleaned label mix: ", kept["best_strategy"].value_counts().to_dict())

    feature_cols = [c for c in kept.columns if c not in NON_FEATURES]
    X = kept[feature_cols].fillna(0)
    encoder = LabelEncoder().fit(kept["best_strategy"])
    y, names = encoder.transform(kept["best_strategy"]), list(encoder.classes_)

    X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\nDevelopment users: {len(y_dev)}   untouched held-out users: {len(y_hold)}")

    table, best = compare_candidates(X_dev.reset_index(drop=True), y_dev, names, config["cv_folds"])
    print(f"\n{config['cv_folds']}-fold CV on development users (selected by macro-F1):")
    print(f"  {'model':14s}{'macro-F1':>10s}{'bal.acc':>9s}{'acc':>8s}{'baseline':>10s}{'AUC':>7s}")
    for row in table:
        auc = f"{row['cv_roc_auc']:.3f}" if row["cv_roc_auc"] is not None else "n/a"
        print(f"  {row['model']:14s}{row['cv_macro_f1']:10.3f}{row['cv_balanced_accuracy']:9.3f}"
              f"{row['cv_accuracy']:8.3f}{row['cv_baseline_accuracy']:10.3f}{auc:>7s}"
              + ("   <- selected" if row["model"] == best else ""))

    held = evaluate_holdout(best, X_dev, y_dev, X_hold, y_hold, names)
    print(f"\nHeld-out ({held['n_test']} users, scored once, model={best}):")
    print(f"  accuracy {held['accuracy']:.3f}  (majority baseline {held['majority_baseline_accuracy']:.3f})")
    print(f"  balanced accuracy {held['balanced_accuracy']:.3f}  (chance {held['majority_baseline_balanced_accuracy']:.3f})")
    print(f"  macro-F1 {held['macro_f1']:.3f}   ROC-AUC (OvR) {held.get('roc_auc', float('nan')):.3f}")
    print(f"  confusion matrix {held['confusion_matrix_labels']}: {held['confusion_matrix']}")

    factory, use_smote = CANDIDATES[best]
    clf = StrategyClassifier()
    clf.fit_full(kept[["userId", *feature_cols]], kept[["userId", "best_strategy"]], factory(), use_smote)
    clf.save()

    report = to_report_names(held)
    report.update({k: v for k, v in held.items() if k not in report})
    EVAL_PATH.write_text(json.dumps({
        "machine_learning": report, "selected_model": best, "cv": table, "label_cleaning": drop,
        "n_users_requested": n_users, "n_dev": int(len(y_dev)), "n_holdout": int(len(y_hold)),
        "features": feature_cols, "config": config}, indent=2, default=lambda o: o.item()
        if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(f"\nSaved model and {EVAL_PATH}")


if __name__ == "__main__":
    main()
