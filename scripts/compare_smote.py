"""Before/after comparison: Module 3's classifier with and without SMOTE.

Both arms share one labeling pass, one feature frame and one train/test split
(train_test_split is seeded at random_state=42 on identical inputs), so the
ONLY difference between them is SMOTE oversampling of the training fold.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.strategy_selector import (
    StrategyClassifier,
    extract_profile_features,
    label_users_with_best_strategy,
)

N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BAR = "=" * 92

print("Loading MovieLens ratings...")
ratings_df = pd.read_csv("newdata/movielens/rating.csv")
sample_users = ratings_df["userId"].unique()[:N_USERS]
ratings_df = ratings_df[ratings_df["userId"].isin(sample_users)]
movies_df = pd.read_csv("data/processed/movies_merged.csv")
print(f"  {len(ratings_df):,} ratings from {len(sample_users):,} users | {len(movies_df):,} movies\n")

print("Extracting user profile features...")
_t = time.time()
features = extract_profile_features(ratings_df, movies_df)
print(f"  {len(features):,} users in {time.time() - _t:.0f}s\n")

print("Labeling users with best strategy (backtest, n_splits=5)...")
_t = time.time()
labels = label_users_with_best_strategy(ratings_df, movies_df, test_fraction=0.2)
print(f"  Labeled {len(labels):,} users in {time.time() - _t:.0f}s")
counts = labels["best_strategy"].value_counts()
for strategy, count in counts.items():
    print(f"    {strategy:16s} {count:6,d} users ({100 * count / len(labels):5.2f}%)")
print()

print(BAR)
print("ARM A — BASELINE (current Module 3 behaviour, no SMOTE)")
print(BAR)
arm_a = StrategyClassifier()
metrics_a = arm_a.fit(features, labels, use_smote=False)
print(f"  SMOTE applied: {metrics_a['smote']['applied']}\n")

print(BAR)
print("ARM B — SMOTE (identical inputs; oversampling applied to the train fold only)")
print(BAR)
arm_b = StrategyClassifier()
metrics_b = arm_b.fit(features, labels, use_smote=True)
smote = metrics_b["smote"]
print(f"  SMOTE applied: {smote['applied']}  (k_neighbors={smote['k_neighbors']})")
print(f"  Training fold BEFORE: {smote['before']}  -> {smote['rows_before']:,} rows")
print(f"  Training fold AFTER : {smote['after']}  -> {smote['rows_after']:,} rows\n")

print(BAR)
print("SPLIT INTEGRITY CHECK — both arms must have evaluated the exact same held-out users")
print(BAR)
holdout_a = arm_a.holdout_predictions
holdout_b = arm_b.holdout_predictions
same_users = list(holdout_a["userId"]) == list(holdout_b["userId"])
same_truth = list(holdout_a["true_strategy"]) == list(holdout_b["true_strategy"])
print(f"  Held-out user IDs identical:       {same_users}")
print(f"  Held-out true labels identical:    {same_truth}")
print(f"  Held-out set size (both arms):     {len(holdout_a):,} users")
print(f"  Held-out users are all real rows:  {set(holdout_a['userId']) <= set(features['userId'])}")
print(f"  Train fold size ({smote['rows_before']:,}) + test ({len(holdout_a):,}) = "
      f"{smote['rows_before'] + len(holdout_a):,} vs. {len(features):,} total users")
assert same_users and same_truth, "SPLIT DIFFERS BETWEEN ARMS — comparison is invalid"
print("  => Same split confirmed. Differences below are attributable to SMOTE alone.\n")

print(BAR)
print("PER-CLASS COMPARISON (held-out test set)")
print(BAR)
print(f"  {'Strategy':16s} {'Support':>8s} | {'Prec A':>7s} {'Prec B':>7s} {'delta':>7s} |"
      f" {'Rec A':>7s} {'Rec B':>7s} {'delta':>7s} | {'F1 A':>7s} {'F1 B':>7s} {'delta':>7s}")
print("  " + "-" * 88)
for cls in metrics_a["confusion_matrix_labels"]:
    a, b = metrics_a["per_class"][cls], metrics_b["per_class"][cls]
    print(f"  {cls:16s} {a['support']:8d} |"
          f" {a['precision']:7.4f} {b['precision']:7.4f} {b['precision'] - a['precision']:+7.4f} |"
          f" {a['recall']:7.4f} {b['recall']:7.4f} {b['recall'] - a['recall']:+7.4f} |"
          f" {a['f1']:7.4f} {b['f1']:7.4f} {b['f1'] - a['f1']:+7.4f}")

print("\n" + BAR)
print("OVERALL COMPARISON")
print(BAR)
rows = [
    ("Accuracy", metrics_a["accuracy"], metrics_b["accuracy"]),
    ("Majority baseline", metrics_a["majority_baseline_accuracy"], metrics_b["majority_baseline_accuracy"]),
    ("Lift over baseline",
     metrics_a["accuracy"] - metrics_a["majority_baseline_accuracy"],
     metrics_b["accuracy"] - metrics_b["majority_baseline_accuracy"]),
    ("Precision (weighted)", metrics_a["precision"], metrics_b["precision"]),
    ("Recall (weighted)", metrics_a["recall"], metrics_b["recall"]),
    ("F1 (weighted)", metrics_a["f1"], metrics_b["f1"]),
    ("ROC-AUC", metrics_a.get("roc_auc", float("nan")), metrics_b.get("roc_auc", float("nan"))),
]
print(f"  {'Metric':24s} {'Baseline':>10s} {'SMOTE':>10s} {'delta':>10s}")
print("  " + "-" * 56)
for name, a, b in rows:
    print(f"  {name:24s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

print("\n" + BAR)
print("CONFUSION MATRICES (rows = true, cols = predicted)")
print(BAR)
labels_order = metrics_a["confusion_matrix_labels"]
for name, metrics in (("BASELINE", metrics_a), ("SMOTE", metrics_b)):
    print(f"  {name}")
    print("  " + " " * 18 + "  ".join(f"{lbl[:12]:>12s}" for lbl in labels_order))
    for true_label, row in zip(labels_order, metrics["confusion_matrix"]):
        print(f"    true {true_label:12s} " + "  ".join(f"{val:12d}" for val in row))
    print()

print(BAR)
print("DIAGNOSTIC — interaction between SMOTE and class_weight='balanced'")
print(BAR)
encoded = np.array(sorted({*range(len(labels_order))}))
original_counts = np.array([metrics_b["smote"]["before"][c] for c in labels_order])
smoted_counts = np.array([metrics_b["smote"]["after"][c] for c in labels_order])
for name, counts_arr in (("train fold as-is", original_counts), ("after SMOTE", smoted_counts)):
    y_fake = np.repeat(encoded, counts_arr)
    weights = compute_class_weight("balanced", classes=encoded, y=y_fake)
    pretty = ", ".join(f"{lbl}={w:.3f}" for lbl, w in zip(labels_order, weights))
    print(f"  class_weight='balanced' resolves to -> {name:18s}: {pretty}")
print("\n  Both arms keep class_weight='balanced' as requested, but note that once SMOTE")
print("  equalises the classes those weights collapse to 1.0 — so ARM B is effectively")
print("  'SMOTE instead of class weighting', not 'SMOTE on top of it'.")

print("\n" + BAR)
print("Comparison complete. Nothing committed.")
print(BAR)
