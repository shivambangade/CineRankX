"""The exact KPI name strings from eval_config.yaml.

Every metric this package reports is keyed by one of these constants rather
than an ad-hoc string, so a number in the report and the code that produced it
can never drift apart. `load_configured_metrics()` reads the config back and
`missing_from()` checks a report actually covers it — the guard against a KPI
being quietly dropped from the table.
"""

from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path("eval_config.yaml")

# --- Group 1: search ---
PRECISION_AT_10 = "Precision@10"
RECALL_AT_10 = "Recall@10"
MRR = "MRR"
NDCG = "NDCG"
SEARCH_LATENCY = "Search Latency"

# --- Group 2: recommendation ---
PRECISION_AT_K = "Precision@K"
RECALL_AT_K = "Recall@K"
F1 = "F1"
MAP = "MAP"
COVERAGE = "Coverage"
DIVERSITY = "Diversity"
NOVELTY = "Novelty"
COLD_START_ACCURACY = "Cold-Start Accuracy"

# --- Group 3: machine learning (strategy classifier) ---
ACCURACY = "Accuracy"
PRECISION = "Precision"
RECALL = "Recall"
ROC_AUC = "ROC-AUC"
CONFUSION_MATRIX = "Confusion Matrix"

# --- Group 4: system ---
RESPONSE_TIME = "Response Time"
MEMORY_USAGE = "Memory Usage"
CPU_USAGE = "CPU Usage"
THROUGHPUT = "Throughput"

SEARCH_METRICS = (PRECISION_AT_10, RECALL_AT_10, MRR, NDCG, SEARCH_LATENCY)
RECOMMENDATION_METRICS = (
    PRECISION_AT_K, RECALL_AT_K, F1, MAP, COVERAGE, DIVERSITY, NOVELTY, COLD_START_ACCURACY,
)
MACHINE_LEARNING_METRICS = (ACCURACY, PRECISION, RECALL, F1, ROC_AUC, CONFUSION_MATRIX)
SYSTEM_METRICS = (RESPONSE_TIME, MEMORY_USAGE, CPU_USAGE, THROUGHPUT)

GROUPS = {
    "search": SEARCH_METRICS,
    "recommendation": RECOMMENDATION_METRICS,
    "machine_learning": MACHINE_LEARNING_METRICS,
    "system": SYSTEM_METRICS,
}


def load_configured_metrics(path: Path | str = _DEFAULT_CONFIG_PATH) -> dict[str, list[str]]:
    """Read the metric-name lists straight out of eval_config.yaml."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return {group: list(config[group]["metrics"]) for group in GROUPS if group in config}


def load_k_values(path: Path | str = _DEFAULT_CONFIG_PATH) -> list[int]:
    """Read recommendation.k_values — the K's for Precision@K / Recall@K."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return [int(k) for k in config["recommendation"]["k_values"]]


def missing_from(report: dict, path: Path | str = _DEFAULT_CONFIG_PATH) -> dict[str, list[str]]:
    """Metric names in eval_config.yaml that a report does not contain.

    An empty dict means full coverage. Used by both the test suite and the
    report script so an unreported KPI fails loudly instead of silently
    vanishing from the results table.
    """
    configured = load_configured_metrics(path)
    gaps = {}
    for group, names in configured.items():
        absent = [name for name in names if name not in report.get(group, {})]
        if absent:
            gaps[group] = absent
    return gaps
