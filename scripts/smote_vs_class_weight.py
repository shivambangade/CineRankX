"""Does SMOTE do anything that cheaper class_weight tuning does not?

The 10-seed result shows SMOTE lifting content_based RECALL while leaving
ROC-AUC flat. Flat AUC means the probability ranking did not improve — the
model is not discriminating better, it is just pushing the decision boundary
toward the minority class. If that reading is right, simply scaling up the
minority class weights should reproduce the same effect without synthetic
data or an extra dependency. This tests exactly that.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Reuses the cache written by smote_seed_robustness.py -- run that first.
CACHE = Path("data/processed")
FEATURES_CACHE = CACHE / "smote_features_5000.csv"
LABELS_CACHE = CACHE / "smote_labels_5000.csv"
if not (FEATURES_CACHE.exists() and LABELS_CACHE.exists()):
    raise SystemExit("Cache missing. Run: PYTHONPATH=. python scripts/smote_seed_robustness.py")

features = pd.read_csv(FEATURES_CACHE)
labels = pd.read_csv(LABELS_CACHE)

merged = features.merge(labels[["userId", "best_strategy"]], on="userId")
feature_names = [c for c in merged.columns if c not in ("userId", "best_strategy")]
X = merged[feature_names].fillna(0)
encoder = LabelEncoder()
y = encoder.fit_transform(merged["best_strategy"])
classes = list(encoder.classes_)
CB = classes.index("content_based")


def run(seed: int, mode: str, boost: float = 1.0) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    y_tr_original = y_tr
    class_weight = "balanced"

    if mode == "smote":
        k = min(5, min(Counter(y_tr).values()) - 1)
        X_tr, y_tr = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
    elif mode == "boosted_weights":
        # 'balanced' weights, with the two minority classes scaled up further.
        encoded = np.arange(len(classes))
        base = compute_class_weight("balanced", classes=encoded, y=y_tr_original)
        majority = Counter(y_tr_original).most_common(1)[0][0]
        class_weight = {
            c: (w if c == majority else w * boost) for c, w in zip(encoded, base)
        }

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight
    ).fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    return {
        "cb_f1": f[CB], "cb_recall": r[CB], "cb_precision": p[CB],
        "accuracy": accuracy_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, model.predict_proba(X_te), multi_class="ovr"),
    }


seeds = range(42, 52)
arms = {
    "A baseline (class_weight=balanced)": ("baseline", 1.0),
    "B SMOTE (weights collapse to 1.0)": ("smote", 1.0),
    "C balanced x2 on minorities": ("boosted_weights", 2.0),
    "D balanced x4 on minorities": ("boosted_weights", 4.0),
    "E balanced x8 on minorities": ("boosted_weights", 8.0),
}

results = {
    name: pd.DataFrame([run(s, mode, boost) for s in seeds])
    for name, (mode, boost) in arms.items()
}

BAR = "=" * 94
print(BAR)
print("SMOTE vs. simply turning class_weight up — 10 seeds each, identical splits")
print(BAR)
print(f"  {'Arm':36s} {'cb F1':>14s} {'cb recall':>14s} {'cb prec':>14s} {'accuracy':>10s} {'ROC-AUC':>9s}")
print("  " + "-" * 90)
for name, df in results.items():
    print(f"  {name:36s} {df['cb_f1'].mean():7.4f}+/-{df['cb_f1'].std():5.4f} "
          f"{df['cb_recall'].mean():7.4f}+/-{df['cb_recall'].std():5.4f} "
          f"{df['cb_precision'].mean():7.4f}+/-{df['cb_precision'].std():5.4f} "
          f"{df['accuracy'].mean():10.4f} {df['roc_auc'].mean():9.4f}")

print("\n" + BAR)
print("ACCURACY COST PER UNIT OF content_based RECALL GAINED (vs arm A)")
print(BAR)
base_df = results["A baseline (class_weight=balanced)"]
for name, df in results.items():
    if name.startswith("A"):
        continue
    d_recall = df["cb_recall"].mean() - base_df["cb_recall"].mean()
    d_acc = df["accuracy"].mean() - base_df["accuracy"].mean()
    ratio = (d_acc / d_recall) if abs(d_recall) > 1e-9 else float("nan")
    print(f"  {name:36s} recall {d_recall:+.4f}  accuracy {d_acc:+.4f}  "
          f"accuracy lost per recall point: {ratio:+.3f}")
