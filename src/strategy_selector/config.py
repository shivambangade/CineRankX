"""Strategy-selector settings, read from eval_config.yaml (never hardcoded in the logic)."""

from pathlib import Path

import yaml

_DEFAULTS = {
    "holdout_fraction": 0.3,
    "label_top_n": 30,
    "inner_splits": 3,
    "inner_fraction": 0.25,
    "label_margin": 0.02,
    "min_history": 30,
    "max_history": 1000,
    "cv_folds": 5,
    "content_options": None,   # None = the original TF-IDF content strategy
}


def load_selector_config(path: Path | str = "eval_config.yaml") -> dict:
    """Return the strategy_selector block merged over defaults (older configs still load)."""
    try:
        with open(path) as f:
            block = (yaml.safe_load(f) or {}).get("strategy_selector") or {}
    except OSError:
        block = {}
    config = {**_DEFAULTS, **{k: v for k, v in block.items() if k in _DEFAULTS}}
    if not 0 < float(config["holdout_fraction"]) < 1:
        raise ValueError("strategy_selector.holdout_fraction must be in (0, 1)")
    if float(config["label_margin"]) < 0:
        raise ValueError("strategy_selector.label_margin must be >= 0")
    return config
