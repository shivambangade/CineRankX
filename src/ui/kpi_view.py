"""Evaluation workspace rendered from reports/kpi_report.json (generate_kpi_report.py)."""

import pandas as pd
import streamlit as st

from src.ui.theme import card, kpi_row, meter


def _num(value, fmt: str = "{:.3f}") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "n/a"


def _comparison(label: str, system, baseline) -> str:
    if not isinstance(system, (int, float)) or not isinstance(baseline, (int, float)):
        return ""
    delta = system - baseline
    badge = "ok" if delta >= 0 else "warn"
    word = "above" if delta >= 0 else "below"
    return (meter(f"CineRankX {label}", system, "o") + meter(f"Baseline {label}", baseline, "t")
            + f'<span class="badge {badge}">{abs(delta):.2f} {word} baseline</span>')


def render_report(report: dict, last_run_ms: float | None) -> None:
    search, rec = report.get("search", {}), report.get("recommendation", {})
    ml, system = report.get("machine_learning", {}), report.get("system", {})
    run, detail = report.get("run", {}), report.get("detail", {})
    rec_d, ml_d = detail.get("recommendation", {}), detail.get("machine_learning", {})
    st.caption(f"Offline evaluation on {run.get('users', '?')} users, {run.get('ratings', 0):,} ratings and "
               f"{run.get('queries', '?')} search queries. Regenerate with "
               "`PYTHONPATH=. python scripts/generate_kpi_report.py`.")

    kpi_row([("Search MRR", _num(search.get("MRR"))),
             ("MAP", _num(rec.get("MAP"))),
             ("Classifier F1", _num(ml.get("F1"))),
             ("Response time", _num(system.get("Response Time"), "{:.1f}"), "ms")])
    st.write("")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with card("topk", "Top-K ranking quality", "bar-chart-2",
                  "Precision, recall and F1 at each list length."):
            ks = sorted(rec.get("Precision@K", {}), key=int)
            if ks:
                topk = pd.DataFrame({m: [rec.get(m, {}).get(k) for k in ks]
                                     for m in ("Precision@K", "Recall@K", "F1")},
                                    index=[f"K={k}" for k in ks])
                st.bar_chart(topk, color=["#f97316", "#2563eb", "#0d9488"], height=260)
                with st.expander("View as table"):
                    st.dataframe(topk.style.format("{:.3f}"), width="stretch")
            else:
                st.caption("Top-K metrics are not in the report.")
    with right:
        with card("listq", "List-level quality", "grid", "Measured over all evaluated users."):
            st.markdown("".join(
                meter(n, rec[n], "o" if n == "MAP" else "") for n in
                ("MAP", "Coverage", "Diversity", "Novelty") if isinstance(rec.get(n), (int, float))),
                unsafe_allow_html=True)

    a, b = st.columns(2, gap="medium")
    with a:
        with card("search", "Retrieval and latency", "search", "IR engine on a fixed query set."):
            st.markdown("".join(meter(n, search[n], "") for n in ("Precision@10", "Recall@10", "MRR", "NDCG")
                                if isinstance(search.get(n), (int, float)))
                        + f'<div class="row"><span>Mean search latency</span>'
                          f'<b>{_num(search.get("Search Latency"), "{:.2f}")} ms</b></div>',
                        unsafe_allow_html=True)
    with b:
        with card("strategy", "Strategy comparison", "git-merge", "CineRankX against simple baselines."):
            body = _comparison("strategy accuracy", ml.get("Accuracy"), ml_d.get("majority_baseline_accuracy"))
            body += "<div style='height:10px'></div>" + _comparison(
                "cold-start accuracy", rec.get("Cold-Start Accuracy"), rec_d.get("cold_start_baseline_accuracy"))
            st.markdown(body, unsafe_allow_html=True)
            if rec_d.get("cold_start_mode") == "simulated":
                st.caption("Cold-start users are simulated by truncating real profiles to 5 ratings.")

    with card("classifier", "Strategy classifier", "cpu", "Held-out classification details."):
        kpi_row([(n, _num(ml.get(n))) for n in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")])
        matrix, labels = ml_d.get("confusion_matrix"), ml_d.get("confusion_matrix_labels")
        per_class = ml_d.get("per_class")
        c1, c2 = st.columns(2, gap="medium")
        if matrix and labels:
            frame = pd.DataFrame(matrix, index=[f"true {c}" for c in labels],
                                 columns=[f"pred {c}" for c in labels])
            c1.markdown("**Confusion matrix**")
            c1.dataframe(frame.style.background_gradient(cmap="Oranges"), width="stretch")
        if per_class:
            c2.markdown("**Per-class metrics**")
            c2.dataframe(pd.DataFrame(per_class).T.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}", "support": "{:.0f}"}),
                width="stretch")

    with card("system", "System performance", "activity", "Measured end to end across all engines."):
        kpi_row([("Response time", _num(system.get("Response Time"), "{:.1f}"), "ms"),
                 ("Throughput", _num(system.get("Throughput"), "{:.1f}"), "ops/s"),
                 ("Memory", _num(system.get("Memory Usage"), "{:.0f}"), "MB"),
                 ("CPU", _num(system.get("CPU Usage"), "{:.0f}"), "% of one core")])
        if last_run_ms is not None:
            st.caption(f"Last live recommendation in this session: {last_run_ms:,.0f} ms.")
