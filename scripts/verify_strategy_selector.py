"""Verification script for Strategy Selector on real MovieLens data."""

import pandas as pd

from src.strategy_selector import (
    StrategyClassifier,
    extract_profile_features,
    label_users_with_best_strategy,
)
from src.strategy_selector.classifier import STRATEGY_CLASSES

print("Loading MovieLens ratings...")
ratings_df = pd.read_csv("newdata/movielens/rating.csv")
print(f"Loaded {len(ratings_df)} ratings from {ratings_df['userId'].nunique()} users")

import sys

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
print(f"\nSampling {SAMPLE_SIZE} users for the demo (vectorized backtest is O(users) after shared setup)...")
sample_users = ratings_df["userId"].unique()[:SAMPLE_SIZE]
ratings_df = ratings_df[ratings_df["userId"].isin(sample_users)]
print(f"Using {len(ratings_df)} ratings from {len(sample_users)} sampled users\n")

print("Loading movies metadata...")
movies_df = pd.read_csv("data/processed/movies_merged.csv")
print(f"Loaded {len(movies_df)} movies\n")

print("Extracting user profile features...")
features = extract_profile_features(ratings_df, movies_df)
print(f"Extracted features for {len(features)} users")
print(f"\nFeature summary:")
print(features[["num_ratings", "avg_rating", "rating_std", "num_genres"]].describe())

print("\n" + "=" * 70)
print("Labeling users with best strategy via backtesting (vectorized)...")
print("=" * 70)
import time as _time

_start = _time.time()
labels = label_users_with_best_strategy(ratings_df, movies_df, test_fraction=0.2)
print(f"Labeled {len(labels)} users in {_time.time() - _start:.1f}s\n")

strategy_counts = labels["best_strategy"].value_counts()
print("Strategy distribution:")
for strategy, count in strategy_counts.items():
    pct = 100 * count / len(labels)
    print(f"  {strategy:20s}: {count:6d} users ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("Training strategy classifier...")
print("=" * 70)
classifier = StrategyClassifier()
metrics = classifier.fit(features, labels)
classifier.save()

print(f"\nClasses present in training labels: {metrics['classes_present']}")
if metrics["classes_missing"]:
    print(
        f"Classes ABSENT from training labels (model cannot predict these — "
        f"never saw an example): {metrics['classes_missing']}"
    )
else:
    print("All strategy classes are represented in the training labels.")

print(f"\nClassifier metrics on held-out test set:")
print(f"  Accuracy:                 {metrics['accuracy']:.4f}")
print(f"  Majority-class baseline:  {metrics['majority_baseline_accuracy']:.4f}  <- always predicting the most common label")
lift = metrics["accuracy"] - metrics["majority_baseline_accuracy"]
print(f"  Lift over baseline:       {lift:+.4f}")
print(f"  Precision (weighted):     {metrics['precision']:.4f}")
print(f"  Recall (weighted):        {metrics['recall']:.4f}")
print(f"  F1 (weighted):            {metrics['f1']:.4f}")
if "roc_auc" in metrics:
    print(f"  ROC-AUC:                  {metrics['roc_auc']:.4f}")

print(f"\nPer-class metrics (this is what shows whether the model is actually")
print(f"distinguishing classes, vs. just riding the majority class):")
print(f"  {'Strategy':16s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
for cls_name in metrics["confusion_matrix_labels"]:
    pc = metrics["per_class"][cls_name]
    print(f"  {cls_name:16s} {pc['precision']:10.4f} {pc['recall']:10.4f} {pc['f1']:10.4f} {pc['support']:8d}")

print(f"\nConfusion matrix (rows: true label, cols: predicted; held-out test set only):")
cm = metrics["confusion_matrix"]
cm_labels = metrics["confusion_matrix_labels"]
header = " " * 16 + "  ".join(f"{lbl[:10]:>10s}" for lbl in cm_labels)
print("  Predicted ->" + " " * 4 + header[16:])
for true_label, row in zip(cm_labels, cm):
    row_str = "  ".join(f"{val:10d}" for val in row)
    print(f"  True {true_label:16s} {row_str}")

print("\n" + "=" * 70)
print("Sample predictions — drawn from the ACTUAL held-out test set")
print("(same population the confusion matrix above was computed on)")
print("=" * 70)
holdout = classifier.holdout_predictions
print(f"Held-out test set size: {len(holdout)} users\n")

for _, row in holdout.head(10).iterrows():
    scores = labels[labels["userId"] == row["userId"]].iloc[0]
    status = "correct" if row["correct"] else "WRONG"
    print(f"User {row['userId']:6d}: predicted={row['predicted_strategy']:14s} "
          f"true={row['true_strategy']:14s} confidence={row['confidence']:.2f}  [{status}]")
    print(f"             backtest scores - CB: {scores['cb_score']:.3f}, "
          f"CF: {scores['cf_score']:.3f}, Pop: {scores['pop_score']:.3f}")

holdout_accuracy = holdout["correct"].mean()
print(f"\nHeld-out accuracy from this table: {holdout_accuracy:.4f} "
      f"(should match 'Accuracy' above: {metrics['accuracy']:.4f})")
assert abs(holdout_accuracy - metrics["accuracy"]) < 1e-9, "Holdout table disagrees with reported accuracy!"

print("\n" + "=" * 70)
print("Strategy selector verification complete!")
print("Classifier saved to data/processed/strategy_classifier.pkl")
print("=" * 70)
