"""Evaluation suite: every KPI defined in eval_config.yaml.

Metric names are the exact strings from that file (see metric_names.py), so a
number in the final report and the code that produced it cannot drift apart.
"""

from src.evaluation import metric_names
from src.evaluation.metric_names import (
    GROUPS,
    load_configured_metrics,
    load_k_values,
    missing_from,
)
from src.evaluation.ml_metrics import (
    classification_metrics,
    majority_baseline_accuracy,
    roc_auc,
)
from src.evaluation.recommendation_metrics import (
    average_precision,
    coverage,
    evaluate_recommendations,
    intra_list_diversity,
    mean_average_precision,
    novelty,
    popularity_baseline_top_k,
    split_user_ratings,
)
from src.evaluation.report import build_report, render
from src.evaluation.search_metrics import (
    RELEVANCE_PROXY,
    build_relevance_index,
    evaluate_search,
    is_relevant,
    ndcg_at_k,
    reciprocal_rank,
)
from src.evaluation.system_metrics import aggregate, measure

__all__ = [
    "metric_names", "GROUPS", "load_configured_metrics", "load_k_values", "missing_from",
    "classification_metrics", "majority_baseline_accuracy", "roc_auc",
    "average_precision", "coverage", "evaluate_recommendations", "intra_list_diversity",
    "mean_average_precision", "novelty", "popularity_baseline_top_k", "split_user_ratings",
    "build_report", "render",
    "RELEVANCE_PROXY", "build_relevance_index", "evaluate_search", "is_relevant",
    "ndcg_at_k", "reciprocal_rank",
    "aggregate", "measure",
]
