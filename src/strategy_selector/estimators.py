"""Importable estimator factory: saved models reference it by module path, so it must not live in a script."""

import numpy as np  # noqa: F401  (kept for parity with callers that pickle numpy-backed estimators)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

SEED = 42


def make(family: str, **p):
    """Unfitted estimator (picklable: classes + keyword arguments only)."""
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from xgboost import XGBClassifier, XGBRegressor
    table = {
        "lightgbm": (LGBMClassifier, dict(verbose=-1, n_jobs=1, random_state=SEED)),
        "lightgbm_reg": (LGBMRegressor, dict(verbose=-1, n_jobs=1, random_state=SEED)),
        "xgboost": (XGBClassifier, dict(n_jobs=1, random_state=SEED, eval_metric="mlogloss", verbosity=0)),
        "xgboost_reg": (XGBRegressor, dict(n_jobs=1, random_state=SEED, verbosity=0)),
        "catboost": (CatBoostClassifier, dict(verbose=0, thread_count=4, random_seed=SEED)),
        "catboost_reg": (CatBoostRegressor, dict(verbose=0, thread_count=4, random_seed=SEED)),
        "extra_trees": (ExtraTreesClassifier, dict(n_jobs=-1, random_state=SEED)),
        "random_forest": (RandomForestClassifier, dict(n_jobs=-1, random_state=SEED)),
        "hist_gb": (HistGradientBoostingClassifier, dict(random_state=SEED)),
    }
    cls, base = table[family]
    return cls(**{**base, **p})
