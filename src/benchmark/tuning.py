"""Hyperparameter tuning on the VALIDATION split only (test interactions are never read)."""

import numpy as np

from src.benchmark.baseline import FrozenBaseline
from src.benchmark.cinerankx import CineRankXStrategy, weights_from
from src.benchmark.evaluate import evaluate_strategy
from src.evaluation.recommendation_metrics import intra_list_diversity
from src.ranking_engine.config import OBJECTIVES, RankingWeights

L2_GRID = (200.0, 500.0, 1500.0)
BETA_GRID = {"overall": (0.8, 0.9, 1.0), "cold": (0.5, 0.6, 0.7, 0.8, 0.9)}
WEIGHT_TRIALS = 14
WEIGHT_USERS = 300


def _score(res: dict) -> tuple[float, float]:
    return res["ndcg@10"], res["hit_rate@10"]


def tune_relevance(s: CineRankXStrategy, catalog_size: int, log=print) -> dict:
    """Pick the EASE regularisation and per-cohort popularity blend by validation NDCG@10."""
    best = None
    for l2 in L2_GRID:
        s.fit(l2)
        chosen, total = {}, 0.0
        for cohort, betas in BETA_GRID.items():
            runs = {b: evaluate_strategy(
                s.bench, "validation", cohort,
                lambda us, ex, k, c=cohort, b=b: s.recommend_many(us, ex, k, c, b, weights=None),
                catalog_size=catalog_size) for b in betas}
            top = max(runs, key=lambda b: _score(runs[b]))
            chosen[cohort] = top
            total += runs[top]["ndcg@10"]
            log(f"  l2={l2:g} {cohort}: best beta={top} ndcg@10={runs[top]['ndcg@10']:.4f} "
                f"hit@10={runs[top]['hit_rate@10']:.4f}")
        if best is None or total > best[0]:
            best = (total, l2, chosen)
    _, l2, betas = best
    s.fit(l2)
    s.beta.update(betas)
    return {"l2": l2, "beta": betas}


def _weight_candidates(seed: int) -> list[RankingWeights]:
    rng = np.random.default_rng(seed)
    cands = [RankingWeights.from_config(), weights_from({"relevance": 1.0})]
    for _ in range(WEIGHT_TRIALS):
        rel = rng.uniform(0.5, 0.95)
        rest = rng.dirichlet(np.ones(len(OBJECTIVES) - 1)) * (1 - rel)
        vec = {"relevance": rel, **dict(zip([o for o in OBJECTIVES if o != "relevance"], rest))}
        cands.append(weights_from(vec))
    return cands


def tune_weights(s: CineRankXStrategy, baseline: FrozenBaseline, catalog_size: int, seed: int,
                 log=print) -> dict:
    """Objective weights: maximise validation NDCG@10 while staying at least as diverse as the baseline."""
    div = lambda rec: intra_list_diversity(rec, s.stats)  # noqa: E731
    report = {}
    for cohort in ("overall", "cold"):
        base = evaluate_strategy(
            s.bench, "validation", cohort,
            lambda us, ex, k: {u: baseline.recommend(ex[u], k) for u in us},
            catalog_size=catalog_size, diversity_fn=div, limit=WEIGHT_USERS)
        floor = base["diversity"]
        rows = []
        for w in _weight_candidates(seed):
            res = evaluate_strategy(
                s.bench, "validation", cohort,
                lambda us, ex, k, w=w: s.recommend_many(us, ex, k, cohort, weights=w),
                catalog_size=catalog_size, diversity_fn=div, limit=WEIGHT_USERS)
            rows.append((w, res))
        ok = [r for r in rows if r[1]["diversity"] >= floor] or [max(rows, key=lambda r: r[1]["diversity"])]
        w, res = max(ok, key=lambda r: _score(r[1]))
        s.weights[cohort] = w
        report[cohort] = {"weights": dict(w.items()), "ndcg@10": res["ndcg@10"],
                          "hit_rate@10": res["hit_rate@10"], "diversity": res["diversity"],
                          "baseline_diversity_floor": floor}
        log(f"  {cohort}: ndcg@10={res['ndcg@10']:.4f} hit@10={res['hit_rate@10']:.4f} "
            f"diversity={res['diversity']:.3f} (floor {floor:.3f})")
    return report
