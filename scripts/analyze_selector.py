"""Diagnose why the strategy classifier plateaus: labels, features, errors and attainable accuracy.

Usage:  PYTHONPATH=. python scripts/analyze_selector.py [labelled_frame.csv]
Everything is computed from the cached labelled frame (profile-only features + holdout backtest
scores) and the saved model settings; nothing is tuned here and the held-out users are only
used for reporting. Writes data/processed/selector_analysis.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FRAME = "data/processed/strategy_labels_v3_n5000_s42_h0.3_t30_min30_max1000_i3.csv"
SCORES = ["cb_score", "cf_score", "pop_score"]
NAMES = ["content_based", "collaborative", "popularity"]
NON_FEATURES = ["userId", "best_strategy", "n_holdout", *SCORES]
SEED = 42


def margins(frame: pd.DataFrame) -> pd.Series:
    ordered = np.sort(frame[SCORES].to_numpy(), axis=1)
    return pd.Series(ordered[:, -1] - ordered[:, -2], index=frame.index)


def noise_ceiling(frame: pd.DataFrame, draws: int = 200) -> float:
    """Accuracy of an oracle that knows each user's TRUE hit-rates, scored against noisy labels.

    The observed scores are the best available estimate of the true rates; the holdout size
    gives the binomial noise of the label. Ties follow the labeller's rule (popularity).
    """
    rng = np.random.default_rng(SEED)
    p = frame[SCORES].to_numpy().clip(0.001, 0.999)
    m = frame["n_holdout"].to_numpy()

    def winner(s):
        best = s.max(axis=1, keepdims=True)
        tied = s == best
        out = np.where(tied[:, 2], 2, s.argmax(axis=1))  # popularity wins ties
        return out

    true_label = winner(p)
    hits = [(winner(rng.binomial(m[:, None], p) / m[:, None]) == true_label).mean() for _ in range(draws)]
    return float(np.mean(hits))


def main() -> None:
    frame = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else FRAME)
    feats = [c for c in frame.columns if c not in NON_FEATURES]
    out: dict = {"n_users": len(frame)}
    y = frame["best_strategy"]
    mg = margins(frame)
    out["class_counts"] = y.value_counts().to_dict()
    out["label_clarity"] = {"all_zero": int((frame[SCORES].max(axis=1) == 0).sum()),
                            "margin_lt_0.02": int(((mg < 0.02) & (frame[SCORES].max(axis=1) > 0)).sum()),
                            "clear": int((mg >= 0.02).sum())}
    # Binomial noise: a winner is only meaningful if its lead exceeds the sampling error of the two scores.
    p = frame[SCORES].to_numpy()
    top2 = np.sort(p, axis=1)[:, -2:]
    se = np.sqrt((top2 * (1 - top2)).sum(axis=1) / frame["n_holdout"].to_numpy())
    out["winner_within_1_standard_error"] = float((mg.to_numpy() < se).mean())
    out["noise_ceiling_full_population"] = noise_ceiling(frame)
    print(f"{len(frame)} users; classes {out['class_counts']}; clarity {out['label_clarity']}")
    print(f"Winner indistinguishable from runner-up (within 1 SE): {out['winner_within_1_standard_error']:.1%}")
    print(f"Accuracy ceiling for an oracle that knows the true hit-rates: {out['noise_ceiling_full_population']:.3f}")

    X = frame[feats].fillna(0)
    enc = {n: i for i, n in enumerate(sorted(y.unique()))}
    yi = y.map(enc).to_numpy()
    f_stat, _ = f_classif(X, yi)
    mi = mutual_info_classif(X, yi, random_state=SEED)
    sep = pd.DataFrame({"feature": feats, "anova_f": f_stat, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    out["top_features"] = sep.head(12).round(4).to_dict("records")
    corr = X.corr().abs().to_numpy()
    out["redundant_pairs_corr_gt_0.95"] = int(((corr > 0.95).sum() - len(feats)) // 2)
    out["constant_features"] = [f for f in feats if X[f].nunique() <= 1]
    out["missing_values"] = int(frame[feats].isna().sum().sum())
    out["exact_duplicate_rows"] = int(X.duplicated().sum())

    scaled = StandardScaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=2).fit(scaled).kneighbors(scaled)
    near = dist[:, 1] < np.percentile(dist[:, 1], 10)
    disagree = (yi[idx[:, 1]] != yi)
    chance = 1 - (np.bincount(yi) / len(yi)) @ (np.bincount(yi) / len(yi))
    out["nearest_neighbour_label_disagreement"] = {"closest_10pct_pairs": float(disagree[near].mean()),
                                                   "chance_level": float(chance)}
    print(f"Nearest-neighbour label disagreement (closest 10% pairs): {disagree[near].mean():.3f} (chance {chance:.3f})")

    X_dev, X_hold, y_dev, y_hold, m_dev, m_hold = train_test_split(
        X, yi, mg, test_size=0.2, random_state=SEED, stratify=yi)
    params = json.loads(Path("data/processed/strategy_eval.json").read_text())["tuned_params"]["extra_trees"]
    model = ExtraTreesClassifier(**{"n_estimators": 150, "random_state": SEED, "n_jobs": -1,
                                    "class_weight": "balanced", **params})
    cv_pred = cross_val_predict(model, X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    model.fit(X_dev, y_dev)
    pred = model.predict(X_hold)
    out["current_model_full_population"] = {
        "train_accuracy": float(accuracy_score(y_dev, model.predict(X_dev))),
        "cv_accuracy": float(accuracy_score(y_dev, cv_pred)),
        "cv_macro_f1": float(f1_score(y_dev, cv_pred, average="macro")),
        "holdout_accuracy": float(accuracy_score(y_hold, pred)),
        "holdout_macro_f1": float(f1_score(y_hold, pred, average="macro")),
        "majority_baseline": float(np.mean(y_hold == np.bincount(y_dev).argmax()))}
    print("Current model, FULL population (nothing dropped):", {k: round(v, 3) for k, v in out["current_model_full_population"].items()})
    bins = pd.qcut(m_hold, 4, duplicates="drop")
    by_margin = pd.DataFrame({"correct": pred == y_hold, "bin": bins}).groupby("bin", observed=True)["correct"].agg(["mean", "size"])
    out["accuracy_by_label_margin"] = {str(k): {"accuracy": float(v["mean"]), "users": int(v["size"])} for k, v in by_margin.iterrows()}
    print("Accuracy by label margin (low margin = ambiguous label):")
    print(by_margin.round(3).to_string())
    errors = pd.crosstab(pd.Series(y_hold, name="true"), pd.Series(pred, name="pred"))
    out["error_confusion"] = errors.to_dict()
    Path("data/processed/selector_analysis.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("Saved data/processed/selector_analysis.json")


if __name__ == "__main__":
    main()
