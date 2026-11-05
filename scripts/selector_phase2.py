"""Strategy selection, evaluated on TWO separate objectives, with uncertainty and significance.

  Objective A (classification): predict each user's labelled winning strategy.
  Objective B (recommendation value): realised hit-rate of the strategy the selector chooses,
                                       against always-popularity and an oracle.

Protocol: all users kept; stratified 80/20 development/held-out split (seed 42); every model gets the
same 5-fold CV on development users; selection uses CV only; held-out users are scored once per model
for reporting. CIs are user-level bootstraps; significance uses paired bootstraps and McNemar.

Usage: PYTHONPATH=. python scripts/selector_phase2.py [n_users] [--deploy] [--tag NAME]
Writes data/processed/selector_phase2_<tag>.json (tag defaults to the content variant).
"""

import functools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, "scripts")
import selector_experiments as se  # noqa: E402  (shared helpers: make, evaluate, NAMES, NON_FEATURES)

from src.storage import DB_PATH, connect  # noqa: E402
from src.strategy_selector.config import load_selector_config  # noqa: E402
from src.strategy_selector.dataset import build_labelled_frame  # noqa: E402
from src.strategy_selector.formulations import (CLASSES, SCORE_COLS, MulticlassForm, PairwiseForm,  # noqa: E402
                                                RankingForm, decision_quality, winner)
from src.strategy_selector.stats import bootstrap_ci, mcnemar_p, paired_bootstrap, proportion_ci  # noqa: E402
from src.strategy_selector.utility import UtilitySelector, choose_tau  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
TAU_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]


def per_user_value(scores, chosen):
    return scores[np.arange(len(scores)), chosen]


def oof_utilities(factory, X, scores, folds=5):
    y = winner(scores)
    out = np.zeros((len(X), 3))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        model = UtilitySelector(factory).fit(X.iloc[tr].reset_index(drop=True), scores[tr])
        out[te] = model.utilities(X.iloc[te].reset_index(drop=True))
    return out


def tune_thresholds(proba, y, grid=(0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.6)):
    """Per-class probability multipliers that maximise macro-F1 on OUT-OF-FOLD predictions."""
    from sklearn.metrics import f1_score
    best, best_f1 = np.ones(3), -1.0
    for a in grid:
        for b in grid:
            w = np.array([a, b, 1.0])
            f1 = f1_score(y, (proba * w).argmax(axis=1), average="macro", labels=CLASSES)
            if f1 > best_f1:
                best, best_f1 = w, f1
    return best


class Reweighted:
    """Wraps a fitted probabilistic model with per-class multipliers (threshold optimisation)."""

    classes_ = CLASSES

    def __init__(self, model, weights):
        self.model, self.weights = model, np.asarray(weights)

    def predict_proba(self, X):
        p = self.model.predict_proba(X) * self.weights
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(args[0]) if args else 5000
    config = load_selector_config()
    if "--base-content" in sys.argv:        # reproduce the previous (original content strategy) labels
        config["content_options"] = None
    for flag, key, cast in (("--top-n", "label_top_n", int), ("--holdout", "holdout_fraction", float)):
        if flag in sys.argv:                  # label-generation experiments: budget / holdout size
            config[key] = cast(sys.argv[sys.argv.index(flag) + 1])
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else (
        ("rq" if config.get("content_options") else "base") + f"_t{config['label_top_n']}_h{config['holdout_fraction']}")
    movies = pd.read_csv("data/processed/movies_merged.csv")
    frame = build_labelled_frame(connect(DB_PATH), movies, n_users, SEED, config)
    scores = frame[SCORE_COLS].to_numpy()
    y = winner(scores)
    feats = [c for c in frame.columns if c not in se.NON_FEATURES]
    X = frame[feats].fillna(0)
    idx_dev, idx_hold = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=y)
    X_dev, X_hold = X.iloc[idx_dev].reset_index(drop=True), X.iloc[idx_hold].reset_index(drop=True)
    s_dev, s_hold, y_dev, y_hold = scores[idx_dev], scores[idx_hold], y[idx_dev], y[idx_hold]
    corr = X_dev.corr().abs()
    drop = {c for i, c in enumerate(corr.columns) if (corr.iloc[:i][c] > 0.98).any()} | \
           {c for c in feats if X_dev[c].nunique() <= 1}
    keep = [c for c in feats if c not in drop]
    X, X_dev, X_hold = X[keep], X_dev[keep], X_hold[keep]
    print(f"[{tag}] {len(frame)} users (none dropped), {len(keep)} features; class mix {np.bincount(y).tolist()}; "
          f"mean strategy hit-rates cf/cb/pop {scores.mean(axis=0).round(4).tolist()}; oracle {scores.max(axis=1).mean():.4f}")

    saved = json.loads(Path("data/processed/selector_experiments_inner3.json").read_text())["tuned"]["lightgbm"]
    lgb_params = saved["params"]
    lgb = functools.partial(se.make, "lightgbm", **lgb_params)
    lgb_reg = functools.partial(se.make, "lightgbm_reg", **lgb_params)
    results: dict = {}

    def record(name, model, proba_oof=None):
        started = time.time()
        proba_oof = se.cv_oof(lambda: model, X_dev, s_dev) if proba_oof is None else proba_oof
        cv = se.evaluate(proba_oof, s_dev)
        fitted = model.fit(X_dev, s_dev)
        proba_hold = fitted.predict_proba(X_hold)
        hold = se.evaluate(proba_hold, s_hold)
        results[name] = {"cv": cv, "holdout": hold, "_pred_hold": proba_hold.argmax(axis=1), "_model": fitted}
        print(f"  {name:36s} CV macro-F1 {cv['macro_f1']:.3f} acc {cv['accuracy']:.3f} | CV hit-rate {cv['realised_hit_rate']:.4f}"
              f" | held-out acc {hold['accuracy']:.3f} F1 {hold['macro_f1']:.3f} hit-rate {hold['realised_hit_rate']:.4f}  ({time.time() - started:.0f}s)", flush=True)

    print("\nObjective A + B candidates:")
    record("A. multiclass LightGBM (existing)", MulticlassForm(lgb, saved["margin_weight"]))
    record("B. multiclass + sigmoid calibration", MulticlassForm(functools.partial(
        CalibratedClassifierCV, lgb(), method="sigmoid", cv=3), saved["margin_weight"]))
    oof = se.cv_oof(lambda: MulticlassForm(lgb, saved["margin_weight"]), X_dev, s_dev)
    w = tune_thresholds(oof, y_dev)
    base_model = MulticlassForm(lgb, saved["margin_weight"]).fit(X_dev, s_dev)
    oof_w = oof * w
    results["B2. multiclass + tuned class thresholds"] = {
        "cv": se.evaluate(oof_w / oof_w.sum(1, keepdims=True), s_dev),
        "holdout": se.evaluate(Reweighted(base_model, w).predict_proba(X_hold), s_hold),
        "_pred_hold": Reweighted(base_model, w).predict(X_hold), "_model": Reweighted(base_model, w)}
    r = results["B2. multiclass + tuned class thresholds"]
    print(f"  {'B2. multiclass + tuned thresholds':36s} CV macro-F1 {r['cv']['macro_f1']:.3f} acc {r['cv']['accuracy']:.3f} | "
          f"held-out acc {r['holdout']['accuracy']:.3f} F1 {r['holdout']['macro_f1']:.3f} hit-rate {r['holdout']['realised_hit_rate']:.4f}")
    record("D. pairwise LightGBM", PairwiseForm(lgb))
    record("E. LambdaRank over strategies", RankingForm())

    util_oof = oof_utilities(lgb_reg, X_dev, s_dev)
    tau, tau_table = choose_tau(util_oof, s_dev, TAU_GRID)
    print(f"  utility margin tau tuned on out-of-fold utilities: {tau} (realised hit-rate by tau {({k: round(v, 4) for k, v in tau_table.items()})})")
    for label, t in (("C. utility regression (tau 0)", 0.0), (f"F. utility + fallback (tau {tau})", tau)):
        sel = UtilitySelector(lgb_reg, tau=t)
        pen = np.array([t, t, 0.0])
        p_oof = np.exp((util_oof - pen) / sel.temperature)
        p_oof /= p_oof.sum(axis=1, keepdims=True)
        record(label, sel, p_oof)

    # ---- references, significance and breakdowns -------------------------------------------------
    n = np.arange(len(s_hold))
    pop_value = s_hold[:, 2]
    best_B = max((k for k in results), key=lambda k: results[k]["cv"]["realised_hit_rate"])
    best_A = max((k for k in results), key=lambda k: results[k]["cv"]["macro_f1"])
    print(f"\nSelected for Objective A (CV macro-F1): {best_A}\nSelected for Objective B (CV hit-rate): {best_B}")
    summary = {"tag": tag, "n_users": len(frame), "features": keep, "tau": tau, "tau_table": tau_table,
               "selected_A": best_A, "selected_B": best_B, "candidates": {}}
    for name, r in results.items():
        chosen = r["_pred_hold"]
        value = per_user_value(s_hold, chosen)
        correct = chosen == y_hold
        entry = {"cv": r["cv"], "holdout": r["holdout"],
                 "holdout_hit_rate_ci95": bootstrap_ci(value)[1:],
                 "holdout_accuracy_ci95": proportion_ci(int(correct.sum()), len(correct))[1:],
                 "vs_popularity": paired_bootstrap(value, pop_value),
                 "accuracy_vs_majority_p": mcnemar_p(correct, y_hold == np.bincount(y_dev).argmax())}
        summary["candidates"][name] = entry
    final = results[best_B]
    chosen = final["_pred_hold"]
    value = per_user_value(s_hold, chosen)
    sizes = X_hold["num_ratings"].to_numpy()
    cuts = np.quantile(sizes, [1 / 3, 2 / 3])
    buckets = np.digitize(sizes, cuts)
    summary["final"] = {
        "name": best_B, "strategy_distribution": {se.NAMES[i]: float((chosen == i).mean()) for i in range(3)},
        "true_label_distribution": {se.NAMES[i]: float((y_hold == i).mean()) for i in range(3)},
        "decision": decision_quality(s_hold, chosen),
        "by_profile_size": [{"bucket": ["smallest third", "middle third", "largest third"][b],
                             "ratings_range": [int(sizes[buckets == b].min()), int(sizes[buckets == b].max())],
                             "users": int((buckets == b).sum()), "selector_hit_rate": float(value[buckets == b].mean()),
                             "popularity_hit_rate": float(pop_value[buckets == b].mean()),
                             "oracle_hit_rate": float(s_hold[buckets == b].max(axis=1).mean())} for b in range(3)]}
    d = summary["final"]["decision"]
    ci = summary["candidates"][best_B]["holdout_hit_rate_ci95"]
    vp = summary["candidates"][best_B]["vs_popularity"]
    print(f"\nFINAL (held-out {len(s_hold)} users): selector hit-rate {d['realised_hit_rate']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] vs always-popularity "
          f"{d['always_popularity']:.4f}, oracle {d['oracle_hit_rate']:.4f}; paired gain {vp['mean_diff']:+.4f} [{vp['ci_low']:+.4f}, {vp['ci_high']:+.4f}]")
    for b in summary["final"]["by_profile_size"]:
        print(f"  {b['bucket']:15s} ratings {b['ratings_range']} n={b['users']}: selector {b['selector_hit_rate']:.4f} popularity {b['popularity_hit_rate']:.4f} oracle {b['oracle_hit_rate']:.4f}")

    # TreeSHAP contributions (LightGBM's native pred_contrib) for the multiclass model
    try:
        gm = results["A. multiclass LightGBM (existing)"]["_model"].model_
        contrib = np.asarray(gm.predict(X_hold, pred_contrib=True))
        per_class = contrib.reshape(len(X_hold), 3, len(keep) + 1)[:, :, :-1]
        importance = np.abs(per_class).mean(axis=(0, 1))
        top = np.argsort(-importance)[:12]
        summary["shap_top_features"] = [{"feature": keep[i], "mean_abs_shap": float(importance[i])} for i in top]
        print("Top features by mean |TreeSHAP|:", ", ".join(f"{keep[i]} ({importance[i]:.3f})" for i in top[:8]))
    except Exception as error:   # noqa: BLE001 - explainability is best effort
        print("SHAP skipped:", error)

    out = Path(f"data/processed/selector_phase2_{tag}.json")
    out.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(f"Saved {out}")
    if "--deploy" in sys.argv:
        import pickle
        pickle.dump(results[best_B]["_model"], open("data/processed/_phase2_model.pkl", "wb"))


if __name__ == "__main__":
    main()
