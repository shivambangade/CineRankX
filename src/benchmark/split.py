"""Frozen benchmark: deterministic train/validation/test split and fixed evaluation cohorts.

Every rating is assigned to a split by a hash of (userId, movieId, seed), so the split
is reproducible without stored state and independent of file order or sample size. The
test interactions are the ground truth for BOTH the frozen baseline and CineRankX;
neither training nor tuning ever reads them.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark.config import BENCH_DIR

TRAIN, VALIDATION, TEST = 0, 1, 2
SPLIT_VERSION = "hash-split-v1"
_M1, _M2, _GOLDEN = (np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB),
                     np.uint64(0x9E3779B97F4A7C15))


def uniform_hash(users: np.ndarray, items: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 of (user, item, seed) mapped to [0, 1); stable across machines and runs."""
    x = (users.astype(np.uint64) << np.uint64(32)) | items.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = x + _GOLDEN * np.uint64(int(seed) + 1)
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def assign_splits(users, items, config: dict) -> np.ndarray:
    u = uniform_hash(users, items, config["seed"])
    split = np.full(len(u), TEST, dtype=np.uint8)
    split[u < config["train_fraction"] + config["validation_fraction"]] = VALIDATION
    split[u < config["train_fraction"]] = TRAIN
    return split


def load_interactions(rating_csv: Path | str, catalog_ids: set[int], config: dict) -> pd.DataFrame:
    """Ratings restricted to catalog movies, with split and a per-rating ordering key."""
    frame = pd.read_csv(rating_csv, usecols=["userId", "movieId", "rating"],
                        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
    frame = frame[frame["movieId"].isin(catalog_ids)].reset_index(drop=True)
    users, items = frame["userId"].to_numpy(), frame["movieId"].to_numpy()
    frame["split"] = assign_splits(users, items, config)
    # A second, independent hash orders each user's ratings, so "the first 5 ratings"
    # of a simulated cold-start user is deterministic and not tied to file order.
    frame["order"] = uniform_hash(users, items, config["seed"] + 7919).astype(np.float32)
    return frame


@dataclass
class Benchmark:
    config: dict
    train: pd.DataFrame            # training ratings (cold cohorts truncated to cold_history)
    cohorts: dict[str, np.ndarray]
    truth: dict[str, dict[int, dict]] = field(default_factory=dict)  # split -> user -> info
    fingerprint: str = ""


def _eligible(frame: pd.DataFrame, config: dict) -> np.ndarray:
    thr = config["positive_threshold"]
    n_train = frame[frame["split"] == TRAIN].groupby("userId").size()
    pos = frame[(frame["rating"] >= thr) & (frame["split"] != TRAIN)]
    has_val = set(pos.loc[pos["split"] == VALIDATION, "userId"])
    has_test = set(pos.loc[pos["split"] == TEST, "userId"])
    ok = n_train[n_train >= config["min_train_ratings"]].index
    return np.array(sorted(set(ok) & has_val & has_test), dtype=np.int64)


def build_benchmark(frame: pd.DataFrame, config: dict) -> Benchmark:
    eligible = _eligible(frame, config)
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(eligible)
    need = config["overall_users"] + config["cold_users"] + 2 * config["tuning_users"]
    if len(order) < need:
        raise ValueError(f"Only {len(order)} eligible users; benchmark needs {need}")
    n_o, n_c, n_t = config["overall_users"], config["cold_users"], config["tuning_users"]
    cohorts = {
        "test_overall": np.sort(order[:n_o]),
        "test_cold": np.sort(order[n_o:n_o + n_c]),
        "tune_overall": np.sort(order[n_o + n_c:n_o + n_c + n_t]),
        "tune_cold": np.sort(order[n_o + n_c + n_t:n_o + n_c + 2 * n_t]),
    }
    cold_ids = np.concatenate([cohorts["test_cold"], cohorts["tune_cold"]])
    train = frame[frame["split"] == TRAIN]
    is_cold = train["userId"].isin(cold_ids)
    cold_rows = train[is_cold].sort_values(["userId", "order"]).groupby("userId").head(config["cold_history"])
    train = pd.concat([train[~is_cold], cold_rows], ignore_index=True)
    bench = Benchmark(config, train[["userId", "movieId", "rating"]].reset_index(drop=True), cohorts)
    members = np.concatenate(list(cohorts.values()))
    sub = frame[frame["userId"].isin(members)]
    for name, code in (("validation", VALIDATION), ("test", TEST)):
        part = sub[sub["split"] == code]
        bench.truth[name] = {
            int(u): {"rated": dict(zip(g["movieId"].astype(int), g["rating"].astype(float))),
                     "positives": set(g.loc[g["rating"] >= config["positive_threshold"], "movieId"].astype(int))}
            for u, g in part.groupby("userId")}
    bench.truth["train_full"] = {int(u): set(g["movieId"].astype(int))
                                 for u, g in sub[sub["split"] == TRAIN].groupby("userId")}
    bench.fingerprint = fingerprint(bench)
    return bench


def fingerprint(bench: Benchmark) -> str:
    """Hash of the cohorts and test ground truth: detects any change to the benchmark."""
    h = hashlib.sha256(SPLIT_VERSION.encode())
    for name in sorted(bench.cohorts):
        h.update(name.encode() + bench.cohorts[name].tobytes())
    for user in sorted(bench.truth["test"]):
        h.update(f"{user}:{sorted(bench.truth['test'][user]['rated'].items())}".encode())
    return h.hexdigest()


def get_benchmark(config: dict, rating_csv: Path | str, catalog_ids: set[int],
                  cache_dir: Path = BENCH_DIR, refresh: bool = False, log=print) -> Benchmark:
    """Build the benchmark once and cache it; the cache is keyed by the benchmark settings."""
    import pickle

    key = hashlib.sha256(repr(sorted(config.items())).encode() + SPLIT_VERSION.encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"benchmark_{key}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            bench = pickle.load(fh)
        if fingerprint(bench) != bench.fingerprint:
            raise RuntimeError("Cached benchmark does not match its fingerprint; rebuild with --refresh")
        log(f"Loaded cached benchmark {path.name}")
        return bench
    log("Loading ratings and assigning the frozen split (about a minute)...")
    frame = load_interactions(rating_csv, catalog_ids, config)
    bench = build_benchmark(frame, config)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bench, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Built benchmark {path.name}: fingerprint {bench.fingerprint[:16]}")
    return bench
