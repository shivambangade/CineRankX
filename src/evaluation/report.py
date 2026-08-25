"""Assembles and renders the full KPI report.

Every value is keyed by the exact metric-name string from eval_config.yaml, and
`build_report()` refuses to return a report that is missing a configured KPI —
so a metric cannot silently drop out of the final results table.
"""

import math

from src.evaluation import metric_names as names

_WIDTH = 78
_UNSET = "not computed"


def build_report(search: dict, recommendation: dict, machine_learning: dict,
                 system: dict, config_path="eval_config.yaml", strict: bool = True) -> dict:
    """Collect the four KPI groups into one report keyed by exact metric names.

    Raises:
        ValueError: if strict and any metric in eval_config.yaml is absent.
    """
    # Only carry over metrics the group actually supplied. Using .get() here
    # would insert the key with a None value, so missing_from() -- which tests
    # key presence -- would report full coverage for a KPI that was never
    # computed, defeating the strict check below. A metric that was computed
    # but is genuinely undefined (ROC-AUC with one class, Cold-Start Accuracy
    # with no cold-start users) still sets its key explicitly, to None.
    report = {
        "search": {n: search[n] for n in names.SEARCH_METRICS if n in search},
        "recommendation": {n: recommendation[n] for n in names.RECOMMENDATION_METRICS if n in recommendation},
        "machine_learning": {n: machine_learning[n] for n in names.MACHINE_LEARNING_METRICS if n in machine_learning},
        "system": {n: system[n] for n in names.SYSTEM_METRICS if n in system},
    }
    gaps = names.missing_from(report, config_path)
    if gaps:
        if strict:
            raise ValueError(f"Report is missing KPIs defined in {config_path}: {gaps}")
        # Non-strict: surface the gap as an explicit None so the rendered table
        # shows "not computed" rather than dropping the row.
        for group, absent in gaps.items():
            for name in absent:
                report[group][name] = None
    report["_detail"] = {
        "search": search, "recommendation": recommendation,
        "machine_learning": machine_learning, "system": system,
    }
    return report


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return _UNSET
    if isinstance(value, float) and math.isnan(value):
        return "n/a (window too short)"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rule(char: str = "=") -> str:
    return char * _WIDTH


def _header(title: str) -> list[str]:
    return [_rule(), f"  {title}", _rule()]


def render(report: dict) -> str:
    """Render the report as the project's final results table."""
    detail = report.get("_detail", {})
    lines = [
        _rule(),
        "  CineRankX — FINAL KPI REPORT".center(_WIDTH),
        "  Metric names are verbatim from eval_config.yaml".center(_WIDTH),
        _rule(),
        "",
    ]

    # --- 1. Search ---
    search = detail.get("search", {})
    lines += _header("1. SEARCH  (Module 2 — Classical IR Engine)")
    lines.append(f"  Queries evaluated: {search.get('n_queries', 0)}")
    if "relevance_proxy" in search:
        lines.append(f"  Relevance basis:   {search['relevance_proxy']}")
    lines.append("")
    lines.append(f"  {'Metric':<24}{'Value':>18}   Notes")
    lines.append(f"  {'-' * (_WIDTH - 4)}")
    pool = search.get("mean_relevant_pool_size")
    notes = {
        names.PRECISION_AT_10: "share of top-10 sharing a genre/keyword",
        names.RECALL_AT_10: f"bounded by 10/{pool:,.0f} — see caveat" if pool else "",
        names.MRR: "mean 1/rank of first relevant result",
        names.NDCG: "binary-gain NDCG@10",
        names.SEARCH_LATENCY: f"ms/query (p95 {_fmt(search.get('latency_p95_ms'), 2)} ms)",
    }
    for name in names.SEARCH_METRICS:
        digits = 2 if name == names.SEARCH_LATENCY else 4
        lines.append(f"  {name:<24}{_fmt(report['search'][name], digits):>18}   {notes.get(name, '')}")
    if pool:
        lines += [
            "",
            f"  CAVEAT: the genre/keyword proxy marks on average {pool:,.0f} of the catalog",
            "  relevant per query, so Recall@10 is structurally near zero. It reflects",
            "  catalog breadth, not retrieval quality. Precision@10 is correspondingly",
            "  generous — a shared genre is a low bar for 'relevant'.",
        ]
    lines.append("")

    # --- 2. Recommendation ---
    rec = detail.get("recommendation", {})
    lines += _header("2. RECOMMENDATION  (Module 4 — Multi-Objective Hybrid Ranking)")
    lines.append(f"  Users evaluated: {rec.get('n_users_evaluated', 0)}   "
                 f"(held-out split; a hit = an item rated >= 4.0 in the held-out portion)")
    lines.append("")
    per_k = report["recommendation"].get(names.PRECISION_AT_K)
    k_values = sorted(per_k) if isinstance(per_k, dict) and per_k else []
    if k_values:
        lines.append(f"  {'Metric':<24}" + "".join(f"{'K=' + str(k):>14}" for k in k_values))
        lines.append(f"  {'-' * (_WIDTH - 4)}")
        for name in (names.PRECISION_AT_K, names.RECALL_AT_K, names.F1):
            values = report["recommendation"].get(name) or {}
            lines.append(f"  {name:<24}" + "".join(f"{_fmt(values.get(k)):>14}" for k in k_values))
    else:
        # These three are normally keyed by K. If they arrive in any other
        # shape they must STILL appear by name -- a KPI silently missing from
        # the results table is the exact failure this report guards against.
        lines.append(f"  {'Metric':<24}{'Value':>18}")
        lines.append(f"  {'-' * (_WIDTH - 4)}")
        for name in (names.PRECISION_AT_K, names.RECALL_AT_K, names.F1):
            lines.append(f"  {name:<24}{_fmt(report['recommendation'].get(name)):>18}")
    lines.append("")
    lines.append(f"  {'Metric':<24}{'Value':>18}   Notes")
    lines.append(f"  {'-' * (_WIDTH - 4)}")
    single = {
        names.MAP: f"mean average precision @{rec.get('max_k', '')}",
        names.COVERAGE: "share of the catalog surfaced across all users",
        names.DIVERSITY: "1 - mean pairwise genre similarity in-list",
        names.NOVELTY: "mean self-information of recommended items",
        names.COLD_START_ACCURACY: (
            f"{rec.get('cold_start_mode', '?')} cold-start cohort, "
            f"n={rec.get('cold_start_n_users', 0)}, profile <= "
            f"{(rec.get('cold_start_detail') or {}).get('max_history', 5)} ratings"
        ),
    }
    for name in (names.MAP, names.COVERAGE, names.DIVERSITY, names.NOVELTY, names.COLD_START_ACCURACY):
        lines.append(f"  {name:<24}{_fmt(report['recommendation'][name]):>18}   {single.get(name, '')}")
    baseline = rec.get("cold_start_baseline_accuracy")
    if baseline is not None:
        system_cs = report["recommendation"][names.COLD_START_ACCURACY] or 0.0
        lines += [
            "",
            f"  Cold-start comparison: CineRankX {_fmt(system_cs)} vs. popularity baseline "
            f"{_fmt(baseline)}",
            f"  Lift over the popularity baseline: {_fmt(system_cs - baseline):>8}",
        ]
        if rec.get("cold_start_mode") == "simulated":
            lines += [
                "",
                "  NOTE: MovieLens 20M only admits users with >= 20 ratings, so it contains",
                "  NO natural cold-start users. This cohort is SIMULATED — real users whose",
                "  profiles were truncated to 5 ratings, scored on their untouched held-out",
                "  items. Every other user keeps a full history, so the collaborative",
                "  neighbourhood stays realistic.",
            ]
    lines.append("")

    # --- 3. Machine learning ---
    ml = detail.get("machine_learning", {})
    lines += _header("3. MACHINE LEARNING  (Module 3 — Strategy Selection Classifier)")
    lines.append(f"  {'Metric':<24}{'Value':>18}   Notes")
    lines.append(f"  {'-' * (_WIDTH - 4)}")
    ml_notes = {
        names.ACCURACY: f"majority baseline {_fmt(ml.get('majority_baseline_accuracy'))}",
        names.PRECISION: "weighted across the three strategies",
        names.RECALL: "weighted across the three strategies",
        names.F1: "weighted across the three strategies",
        names.ROC_AUC: "one-vs-rest; 0.50 = chance",
    }
    for name in (names.ACCURACY, names.PRECISION, names.RECALL, names.F1, names.ROC_AUC):
        lines.append(f"  {name:<24}{_fmt(report['machine_learning'][name]):>18}   {ml_notes.get(name, '')}")

    matrix = report["machine_learning"].get(names.CONFUSION_MATRIX)
    labels = ml.get("confusion_matrix_labels", [])
    lines += ["", f"  {names.CONFUSION_MATRIX} (rows = true, columns = predicted):", ""]
    if isinstance(matrix, list) and matrix and labels:
        lines.append("   " + " " * 18 + "".join(f"{lbl[:13]:>15}" for lbl in labels))
        for label, row in zip(labels, matrix):
            lines.append(f"    true {label:<14}" + "".join(f"{v:>15,}" for v in row))
    else:
        # Name still printed even when the matrix is absent or malformed, so a
        # reader can see the KPI was requested and why no grid is shown.
        lines.append(f"    {_fmt(matrix)}")
    per_class = ml.get("per_class")
    if per_class:
        lines += ["", "  Per-class detail:", "",
                  f"    {'Strategy':<18}{'Precision':>11}{'Recall':>11}{'F1':>11}{'Support':>10}"]
        for label in labels:
            pc = per_class.get(label, {})
            lines.append(f"    {label:<18}{_fmt(pc.get('precision')):>11}{_fmt(pc.get('recall')):>11}"
                         f"{_fmt(pc.get('f1')):>11}{pc.get('support', 0):>10,}")
    lines.append("")

    # --- 4. System ---
    sysm = detail.get("system", {})
    lines += _header("4. SYSTEM  (end-to-end instrumentation across all engines)")
    lines.append(f"  {'Metric':<24}{'Value':>18}   Notes")
    lines.append(f"  {'-' * (_WIDTH - 4)}")
    n_cores = sysm.get("n_cores", "?")
    sys_notes = {
        names.RESPONSE_TIME: "ms/call, averaged across instrumented operations",
        names.MEMORY_USAGE: "MB resident set size (peak observed)",
        names.CPU_USAGE: f"% of ONE core ({_fmt(sysm.get('cpu_percent_of_machine'), 1)}% of {n_cores} cores)",
        names.THROUGHPUT: "operations/second across all instrumented calls",
    }
    for name in names.SYSTEM_METRICS:
        digits = 1 if name in (names.CPU_USAGE, names.THROUGHPUT) else 2
        lines.append(f"  {name:<24}{_fmt(report['system'][name], digits):>18}   {sys_notes.get(name, '')}")

    per_op = sysm.get("per_operation", [])
    if per_op:
        lines += ["", "  Per-operation breakdown:", "",
                  f"    {'Operation':<26}{'ms/call':>10}{'p95 ms':>10}{'ops/sec':>11}{'CPU %':>9}"]
        for m in per_op:
            lines.append(f"    {m.get('label', '?'):<26}{m[names.RESPONSE_TIME]:>10.3f}"
                         f"{m['response_time_p95_ms']:>10.3f}{m[names.THROUGHPUT]:>11.1f}"
                         f"{_fmt(m[names.CPU_USAGE], 1):>9}")
    lines += ["", _rule(), "  End of report".center(_WIDTH), _rule()]
    return "\n".join(lines)
