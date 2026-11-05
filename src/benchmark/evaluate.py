"""Strategy-agnostic evaluation: every strategy is scored on identical users and ground truth."""

from src.benchmark.metrics import rating_errors, summarize
from src.benchmark.split import Benchmark

PHASES = {"validation": "validation", "test": "test"}


def exclusions(bench: Benchmark, phase: str, users) -> dict[int, set[int]]:
    """Items a strategy must not recommend: everything the user rated BEFORE the scored split.

    Tuning (validation) excludes the user's training items; the final test additionally
    excludes validation items. Test interactions are never used to build either mask.
    """
    out = {}
    for u in users:
        seen = set(bench.truth["train_full"].get(int(u), ()))
        if phase == "test":
            seen |= set(bench.truth["validation"].get(int(u), {}).get("rated", {}))
        out[int(u)] = seen
    return out


def evaluate_strategy(bench: Benchmark, phase: str, cohort: str, recommend_many, predict=None,
                      catalog_size: int = 1, diversity_fn=None, novelty_fn=None,
                      ks: list[int] | None = None, limit: int | None = None) -> dict:
    """Score one strategy on one cohort.

    recommend_many(users, exclude, k) -> {user: [movieId, ...]}; predict(user, movie) -> rating.
    """
    ks = ks or bench.config["ks"]
    users = [int(u) for u in bench.cohorts[f"{'test' if phase == 'test' else 'tune'}_{cohort}"]]
    truth = bench.truth["test" if phase == "test" else "validation"]
    users = [u for u in users if truth.get(u, {}).get("positives")]
    if limit:  # evenly spaced subset: deterministic and not biased towards low user ids
        users = users[::max(1, len(users) // limit)][:limit]
    lists = recommend_many(users, exclusions(bench, phase, users), max(ks))
    positives = {u: truth[u]["positives"] for u in users}
    result = summarize(lists, positives, ks, catalog_size, diversity_fn, novelty_fn)
    if predict is not None:
        pairs = [(predict(u, m), r) for u in users for m, r in truth[u]["rated"].items()]
        result["rating"] = rating_errors([p for p, _ in pairs], [r for _, r in pairs])
    return result
