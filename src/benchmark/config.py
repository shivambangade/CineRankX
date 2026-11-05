"""Benchmark settings from eval_config.yaml (the `benchmark:` block)."""

from pathlib import Path

import yaml

BENCH_DIR = Path("data/benchmark")

_DEFAULTS = {
    "seed": 20260101, "train_fraction": 0.70, "validation_fraction": 0.10,
    "positive_threshold": 4.0, "min_train_ratings": 20, "overall_users": 2000,
    "cold_users": 1000, "tuning_users": 1000, "cold_history": 5, "ks": [5, 10, 20],
    "headline_k": 10,
}


def load_benchmark_config(path: Path | str = "eval_config.yaml") -> dict:
    try:
        with open(path) as f:
            block = (yaml.safe_load(f) or {}).get("benchmark") or {}
    except OSError:
        block = {}
    config = {**_DEFAULTS, **{k: v for k, v in block.items() if k in _DEFAULTS}}
    if config["train_fraction"] + config["validation_fraction"] >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave room for a test split")
    if config["headline_k"] not in config["ks"]:
        raise ValueError("headline_k must be one of ks")
    return config
