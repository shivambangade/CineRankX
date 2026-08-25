"""Multi-seed robustness check for the SMOTE result.

The headline comparison rests on ONE train/test split, where content_based's
true positives moved 5 -> 13 out of 70. Those are small counts, so this script
repeats the identical two-arm comparison across many seeds to separate a real
effect from split luck.

Deliberately standalone: it replicates the classifier's fit path rather than
modifying it, so production code keeps its fixed random_state=42.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.strategy_selector import extract_profile_features, label_users_with_best_strategy

N_USERS = 5000
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
# Cached under data/processed/ (gitignored) so the ~3-minute labeling pass is
# paid once and shared with the other SMOTE scripts.
CACHE = Path("data/processed")
CACHE.mkdir(parents=True, exist_ok=True)
FEATURES_CACHE = CACHE / "smote_features_5000.csv"
LABELS_CACHE = CACHE / "smote_labels_5000.csv"

if FEATURES_CACHE.exists() and LABELS_CACHE.exists():
    print("Reusing cached features/labels...")
    features, labels = pd.read_csv(FEATURES_CACHE), pd.read_csv(LABELS_CACHE)
else:
    print("Loading + labeling (one-off, ~3 min)...")
    ratings_df = pd.read_csv("newdata/movielens/rating.csv")
    ratings_df = ratings_df[ratings_df["userId"].isin(ratings_df["userId"].unique()[:N_USERS])]
    movies_df = pd.read_csv("data/processed/movies_merged.csv")
    features = extract_profile_features(ratings_df, movies_df)
    _t = time.time()
    labels = label_users_with_best_strategy(ratings_df, movies_df, test_fraction=0.2)
    print(f"  labeled in {time.time() - _t:.0f}s")
    features.to_csv(FEATURES_CACHE, index=False)
    labels.to_csv(LABELS_CACHE, index=False)

merged = features.merge(labels[["userId", "best_strategy"]], on="userId")
feature_names = [c for c in merged.columns if c not in ("userId", "best_strategy")]
X = merged[feature_names].fillna(0)
encoder = LabelEncoder()
y = encoder.fit_transform(merged["best_strategy"])
classes = list(encoder.classes_)
print(f"  {len(merged):,} users | classes {classes}\n")


def run_arm(seed: int, use_smote: bool) -> dict:
    """One arm at one seed. Split first; SMOTE only ever sees the train fold."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    y_tr_original = y_tr
    if use_smote:
        k = min(5, min(Counter(y_tr).values()) - 1)
        X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    p, r, f, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    majority = Counter(y_tr_original).most_common(1)[0][0]
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "majority_baseline": float(np.mean(y_te == majority)),
        "roc_auc": roc_auc_score(y_te, proba, multi_class="ovr"),
        **{f"{cls}_f1": f[i] for i, cls in enumerate(classes)},
        **{f"{cls}_recall": r[i] for i, cls in enumerate(classes)},
        **{f"{cls}_precision": p[i] for i, cls in enumerate(classes)},
    }


seeds = list(range(42, 42 + N_SEEDS))
base = pd.DataFrame([run_arm(s, False) for s in seeds], index=seeds)
smote = pd.DataFrame([run_arm(s, True) for s in seeds], index=seeds)
print(f"Ran {N_SEEDS} seeds per arm.\n")

BAR = "=" * 86
print(BAR)
print(f"ACROSS {N_SEEDS} SEEDS — mean +/- std, and how often SMOTE beats baseline")
print(BAR)
print(f"  {'Metric':28s} {'Baseline':>16s} {'SMOTE':>16s} {'delta':>9s} {'SMOTE wins':>12s}")
print("  " + "-" * 82)
for col in ["content_based_f1", "content_based_recall", "content_based_precision",
            "popularity_f1", "collaborative_f1", "accuracy", "roc_auc"]:
    a, b = base[col], smote[col]
    wins = int((b > a).sum())
    print(f"  {col:28s} {a.mean():7.4f}+/-{a.std():6.4f} {b.mean():7.4f}+/-{b.std():6.4f} "
          f"{b.mean() - a.mean():+9.4f} {wins:>8d}/{N_SEEDS}")

print("\n" + BAR)
print("PER-SEED content_based F1 (the class SMOTE is meant to rescue)")
print(BAR)
print(f"  {'seed':>6s} {'baseline':>10s} {'smote':>10s} {'delta':>10s}")
for seed in seeds:
    a, b = base.loc[seed, "content_based_f1"], smote.loc[seed, "content_based_f1"]
    print(f"  {seed:>6d} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

d = smote["content_based_f1"] - base["content_based_f1"]
print(f"\n  content_based F1 delta: mean {d.mean():+.4f}, min {d.min():+.4f}, max {d.max():+.4f}")
print(f"  Improved in {int((d > 0).sum())}/{N_SEEDS} seeds; regressed in {int((d < 0).sum())}/{N_SEEDS}.")
