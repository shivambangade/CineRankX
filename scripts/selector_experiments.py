"""Systematic strategy-selector experiments on the FULL labelled population (no user dropped).

Usage:  PYTHONPATH=. python scripts/selector_experiments.py [n_users] [--trials N] [--deploy]

Protocol (fixed before any result was seen):
  * every user is kept, labels are the labeller's argmax with the documented popularity tie-break;
  * 80/20 stratified development/held-out split (seed 42); the held-out users are never used to
    tune, select or early-stop anything;
  * Optuna tunes each model family by 3-fold macro-F1 on development users only;
  * all candidates (families x formulations x ensembles) get the same 5-fold CV on development users;
  * the winner is the highest CV macro-F1 (ties: higher CV realised hit-rate); held-out numbers are
    then reported for it, and shown for the other candidates only for transparency.
--deploy refits the winner on all users and saves it where StrategyClassifier loads from.
Writes data/processed/selector_experiments.json.
"""

import functools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.storage import DB_PATH, connect
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.dataset import build_labelled_frame
from src.strategy_selector.estimators import make
from src.strategy_selector.formulations import (CLASSES, SCORE_COLS, MulticlassForm, OneVsRestForm, PairwiseForm,
                                                RankingForm, RegressionForm, decision_quality, winner)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
OUT = Path("data/processed/selector_experiments.json")
if "--inner" in sys.argv:
    OUT = Path(f"data/processed/selector_experiments_inner{sys.argv[sys.argv.index('--inner') + 1]}.json")
NON_FEATURES = ["userId", "best_strategy", "n_holdout", "cb_score", "cf_score", "pop_score"]
NAMES = ["collaborative", "content_based", "popularity"]


def space(family: str, t: optuna.Trial) -> dict:
    if family == "lightgbm":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500), learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=t.suggest_int("num_leaves", 4, 32), min_child_samples=t.suggest_int("min_child_samples", 5, 60),
                    subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0), reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True))
    if family == "xgboost":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500), max_depth=t.suggest_int("max_depth", 2, 6),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0), min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 30, log=True))
    if family == "catboost":
        return dict(iterations=t.suggest_int("iterations", 100, 400), depth=t.suggest_int("depth", 3, 7),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 20, log=True))
    if family in ("extra_trees", "random_forest"):
        return dict(n_estimators=t.suggest_int("n_estimators", 60, 300), max_depth=t.suggest_int("max_depth", 3, 14),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 30), max_features=t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5, 0.8]))
    return dict(max_iter=t.suggest_int("max_iter", 50, 400), learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_leaf_nodes=t.suggest_int("max_leaf_nodes", 4, 31), l2_regularization=t.suggest_float("l2_regularization", 1e-3, 30, log=True))


def evaluate(proba: np.ndarray, scores: np.ndarray) -> dict:
    y = winner(scores)
    pred = proba.argmax(axis=1)
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", labels=CLASSES),
           "balanced_accuracy": balanced_accuracy_score(y, pred)}
    try:
        out["roc_auc"] = roc_auc_score(y, proba / proba.sum(axis=1, keepdims=True), multi_class="ovr", labels=list(CLASSES))
    except ValueError:
        out["roc_auc"] = None
    out.update(decision_quality(scores, pred))
    return {k: (float(v) if v is not None else None) for k, v in out.items()}


def cv_oof(builder, X: pd.DataFrame, scores: np.ndarray, folds: int = 5) -> np.ndarray:
    y = winner(scores)
    oof = np.zeros((len(X), 3))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y):
        oof[te] = builder().fit(X.iloc[tr].reset_index(drop=True), scores[tr]).predict_proba(X.iloc[te].reset_index(drop=True))
    return oof


def tune(family: str, X, scores, trials: int) -> tuple[dict, float]:
    def objective(trial):
        weight = trial.suggest_float("margin_weight", 0.0, 4.0)
        params = space(family, trial)
        builder = lambda: MulticlassForm(functools.partial(make, family, **params), weight)  # noqa: E731
        y = winner(scores)
        folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
        pred = np.zeros(len(y), dtype=int)
        for tr, te in folds.split(X, y):
            pred[te] = builder().fit(X.iloc[tr].reset_index(drop=True), scores[tr]).predict(X.iloc[te].reset_index(drop=True))
        return f1_score(y, pred, average="macro", labels=CLASSES)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    weight = best.pop("margin_weight")
    if family == "lightgbm":
        best["subsample_freq"] = 1
    return {"params": best, "margin_weight": weight}, float(study.best_value)


def deploy_winner(builder, name, X, scores, X_dev, s_dev, X_hold, s_hold, keep, config, table, extras) -> None:
    """Refit the selected candidate on every user, save it for the app, and write the KPI-facing results."""
    import shutil

    from sklearn.preprocessing import LabelEncoder
    from src.evaluation.ml_metrics import classification_metrics, to_report_names
    from src.strategy_selector import StrategyClassifier
    from src.strategy_selector.dataset import cache_path
    from src.strategy_selector.strategy_context import CONTEXT_PATH, build_features, load_context
    from src.strategy_selector.dataset import load_ratings, sample_user_ids

    held_model = builder().fit(X_dev, s_dev)
    proba = held_model.predict_proba(X_hold)
    truth, pred = winner(s_hold), proba.argmax(axis=1)
    held = classification_metrics(truth, pred, proba, NAMES, y_train=winner(s_dev))
    held.update(balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                macro_f1=float(f1_score(truth, pred, average="macro")), n_test=int(len(truth)),
                majority_baseline_balanced_accuracy=1 / 3, **{f"decision_{k}": v for k, v in decision_quality(s_hold, pred).items()})
    report = to_report_names(held)
    report.update({k: v for k, v in held.items() if k not in report})

    final = builder().fit(X[keep], scores)
    clf = StrategyClassifier()
    clf.model, clf.feature_names = final, list(keep)
    clf.label_encoder = LabelEncoder().fit(NAMES)
    clf.save()
    shutil.copyfile(cache_path(extras["n_users"], SEED, config).with_suffix(".context.pkl"), CONTEXT_PATH)
    ctx, conn = load_context(), connect(DB_PATH)
    ids = sample_user_ids(conn, 100, SEED + 5, config["min_history"], config["max_history"])
    ratings = load_ratings(conn, ids, set(pd.read_csv("data/processed/movies_merged.csv")["movieId"].astype(int)))
    clf.load()
    timings = []
    movies = pd.read_csv("data/processed/movies_merged.csv")
    for _, group in ratings.groupby("userId"):
        started = time.perf_counter()
        clf.predict(build_features(group, movies, ctx, config["inner_splits"], config["inner_fraction"]))
        timings.append((time.perf_counter() - started) * 1000)
    previous = json.loads(Path("data/processed/strategy_eval.json").read_text(encoding="utf-8"))
    analysis = json.loads(Path("data/processed/selector_analysis.json").read_text(encoding="utf-8"))         if Path("data/processed/selector_analysis.json").exists() else {}
    cv = {n: row["cv"] for n, row in table.items()}
    Path("data/processed/strategy_eval.json").write_text(json.dumps({
        "machine_learning": {**report, "noise_ceiling": analysis.get("noise_ceiling_full_population"),
                             "winner_within_1_se": analysis.get("winner_within_1_standard_error")},
        "selected_model": name, "cv": cv, "label_cleaning": {"total": len(scores), "kept": len(scores),
                                                             "dropped_all_zero": 0, "dropped_ambiguous": 0, "margin": None},
        "label_reliability": previous.get("label_reliability"),
        "latency_ms": {"mean_ms": float(np.mean(timings)), "p95_ms": float(np.percentile(timings, 95)), "users": len(timings)},
        "n_dev": int(len(s_dev)), "n_holdout": int(len(s_hold)), "n_users_requested": extras["n_users"],
        "features": list(keep), "config": config, "ablation": previous.get("ablation"),
        "evaluation_population": "all labelled users; none dropped"}, indent=2,
        default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
    print(f"Deployed {name}: refit on {len(scores)} users; inference {np.mean(timings):.0f} ms/user")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n_users = int(args[0]) if args else 5000
    trials = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 20
    config = load_selector_config()
    if "--inner" in sys.argv:   # information experiment: more inner backtest splits = less noisy features
        config["inner_splits"] = int(sys.argv[sys.argv.index("--inner") + 1])
    quick = "--quick" in sys.argv
    movies = pd.read_csv("data/processed/movies_merged.csv")
    frame = build_labelled_frame(connect(DB_PATH), movies, n_users, SEED, config)
    scores = frame[SCORE_COLS].to_numpy()
    y = winner(scores)
    feats = [c for c in frame.columns if c not in NON_FEATURES]
    X = frame[feats].fillna(0)
    print(f"{len(frame)} users (none dropped); classes {dict(zip(NAMES, np.bincount(y)))}; {len(feats)} features")

    idx_dev, idx_hold = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=y)
    X_dev, X_hold = X.iloc[idx_dev].reset_index(drop=True), X.iloc[idx_hold].reset_index(drop=True)
    s_dev, s_hold = scores[idx_dev], scores[idx_hold]
    corr = X_dev.corr().abs()
    drop = [c for i, c in enumerate(corr.columns) if (corr.iloc[:i][c] > 0.98).any()] + \
           [c for c in feats if X_dev[c].nunique() <= 1]
    keep = [c for c in feats if c not in set(drop)]
    X_dev, X_hold = X_dev[keep], X_hold[keep]
    print(f"Removed {len(drop)} redundant/constant features: {sorted(set(drop))}")

    saved = json.loads(Path("data/processed/selector_experiments_inner3.json").read_text())         if "--deploy-saved" in sys.argv else None
    tuned, cv_scores = (dict(saved["tuned"]) if saved else {}), {}
    for family in (() if saved else ("lightgbm", "extra_trees") if quick else
                   ("lightgbm", "xgboost", "catboost", "extra_trees", "random_forest", "hist_gb")):
        started = time.time()
        tuned[family], cv_scores[family] = tune(family, X_dev, s_dev, 10 if family == "catboost" else trials)
        print(f"  tuned {family:13s} inner-CV macro-F1 {cv_scores[family]:.3f}  ({time.time() - started:.0f}s)", flush=True)

    def fac(family): return functools.partial(make, family, **tuned[family]["params"])
    builders = {f"multiclass/{f}": (lambda f=f: MulticlassForm(fac(f), tuned[f]["margin_weight"])) for f in tuned}
    lgb = fac("lightgbm")
    builders.update({
        "multiclass/lightgbm (no weights)": lambda: MulticlassForm(lgb, 0.0),
        "one-vs-rest/lightgbm": lambda: OneVsRestForm(lgb),
        "pairwise/lightgbm": lambda: PairwiseForm(lgb),
        "regression/lightgbm": lambda: RegressionForm(functools.partial(make, "lightgbm_reg", **{**tuned["lightgbm"]["params"]})),
        "regression/catboost": lambda: RegressionForm(functools.partial(make, "catboost_reg", **tuned["catboost"]["params"])),
        "ranking/lambdarank": lambda: RankingForm(),
        "ensemble/voting(lgb+et+cat)": lambda: MulticlassForm(functools.partial(
            VotingClassifier, [("lgb", lgb()), ("et", fac("extra_trees")()), ("cat", fac("catboost")())], voting="soft"), 0.0),
        "ensemble/stacking(lgb+et+cat)": lambda: MulticlassForm(functools.partial(
            StackingClassifier, [("lgb", lgb()), ("et", fac("extra_trees")()), ("cat", fac("catboost")())],
            final_estimator=LogisticRegression(max_iter=2000), cv=3), 0.0),
    })
    if saved:   # redeploy a winner of the earlier full comparison without repeating it
        pos = sys.argv.index("--deploy-saved") + 1
        name = sys.argv[pos] if pos < len(sys.argv) and not sys.argv[pos].startswith("--") else saved["selected"]
        print(f"Deploying {name}")
        deploy_winner(builders[name], name, X, scores, X_dev, s_dev, X_hold, s_hold, keep, config,
                      saved["table"], {"n_users": n_users})
        return
    table = {}
    print("\n5-fold CV on development users (all candidates identical protocol):")
    for name, builder in builders.items():
        started = time.time()
        table[name] = {"cv": evaluate(cv_oof(builder, X_dev, s_dev), s_dev)}
        c = table[name]["cv"]
        print(f"  {name:34s} macro-F1 {c['macro_f1']:.3f}  acc {c['accuracy']:.3f}  bal {c['balanced_accuracy']:.3f}  "
              f"realised hit-rate {c['realised_hit_rate']:.4f}  regret {c['regret']:.4f}  ({time.time() - started:.0f}s)", flush=True)

    best = max(table, key=lambda n: (round(table[n]["cv"]["macro_f1"], 4), table[n]["cv"]["realised_hit_rate"]))
    print(f"\nSelected by CV macro-F1: {best}")
    for name, builder in builders.items():
        model = builder().fit(X_dev, s_dev)
        table[name]["holdout"] = evaluate(model.predict_proba(X_hold), s_hold)
        if name == best:
            proba = model.predict_proba(X_hold)
            pred, truth = proba.argmax(axis=1), winner(s_hold)
            p, r, f, sup = precision_recall_fscore_support(truth, pred, labels=CLASSES, zero_division=0)
            report = {"per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
                                    for i, n in enumerate(NAMES)},
                      "confusion_matrix": confusion_matrix(truth, pred, labels=CLASSES).tolist(), "labels": NAMES,
                      "majority_baseline_accuracy": float(np.mean(truth == np.bincount(winner(s_dev)).argmax()))}
            train_acc = float(accuracy_score(winner(s_dev), model.predict(X_dev)))
    h = table[best]["holdout"]
    print(f"HELD-OUT ({len(idx_hold)} users, every user kept): accuracy {h['accuracy']:.3f} (majority {report['majority_baseline_accuracy']:.3f}) "
          f"macro-F1 {h['macro_f1']:.3f} balanced {h['balanced_accuracy']:.3f} AUC {h['roc_auc']:.3f}; train accuracy {train_acc:.3f}")
    print(f"Decision value: realised hit-rate {h['realised_hit_rate']:.4f} vs always-popularity {h['always_popularity']:.4f}, "
          f"always-collaborative {h['always_collaborative']:.4f}, oracle {h['oracle_hit_rate']:.4f}")
    for name, m in report["per_class"].items():
        print(f"  {name:14s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} n={m['support']}")
    print("  confusion", report["confusion_matrix"])
    OUT.write_text(json.dumps({"selected": best, "table": table, "tuned": tuned, "features_kept": keep, "features_dropped": sorted(set(drop)),
                               "report": report, "train_accuracy_of_selected": train_acc, "n_dev": len(idx_dev), "n_holdout": len(idx_hold),
                               "trials": trials}, indent=2, default=str), encoding="utf-8")
    print(f"Saved {OUT}")
    if "--deploy" in sys.argv:
        deploy_winner(builders[best], best, X, scores, X_dev, s_dev, X_hold, s_hold, keep, config, table,
                      {"n_users": n_users})


if __name__ == "__main__":
    main()
