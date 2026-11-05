"""Evaluation workspace rendered from reports/kpi_report.json (generate_kpi_report.py)."""

import pandas as pd
import streamlit as st

from src.ui.comparison_view import render_comparison
from src.ui.theme import card, kpi_row, meter


def _num(value, fmt: str = "{:.3f}") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "n/a"


def _comparison(label: str, system, reference, ref_name: str = "Reference") -> str:
    """Two percentage bars plus a points-above/below badge; empty when either value is missing."""
    if not isinstance(system, (int, float)) or not isinstance(reference, (int, float)):
        return ""
    diff = (system - reference) * 100
    badge, word = ("ok", "above") if diff >= 0.05 else ("warn", "below") if diff <= -0.05 else ("", "")
    text = f"{abs(diff):.1f} pts {word} {ref_name.lower()}" if word else f"Same as {ref_name.lower()}"
    return (meter(f"Selector {label}", system, "o", "{:.1%}") + meter(f"{ref_name} {label}", reference, "t", "{:.1%}")
            + f'<span class="badge {badge}">{text}</span><div style="height:10px"></div>')


def render_report(report: dict, last_run_ms: float | None, comparison: dict | None = None) -> None:
    search, rec = report.get("search", {}), report.get("recommendation", {})
    ml, system = report.get("machine_learning", {}), report.get("system", {})
    run, detail = report.get("run", {}), report.get("detail", {})
    rec_d, ml_d = detail.get("recommendation", {}), detail.get("machine_learning", {})
    st.caption(f"Offline evaluation on {run.get('users', '?')} users, {run.get('ratings', 0):,} ratings and "
               f"{run.get('queries', '?')} search queries. Regenerate with "
               "`PYTHONPATH=. python scripts/generate_kpi_report.py`.")

    cards = [("Search MRR", _num(search.get("MRR"))),
             ("MAP", _num(rec.get("MAP"))),
             ("Classifier F1", _num(ml.get("F1"))),
             ("Response time", _num(system.get("Response Time"), "{:.1f}"), "ms")]
    if comparison:
        headline = comparison["headline_metric"]
        now = comparison["cinerankx"]["after"]["overall"]
        base = comparison["baseline"]["results"]["overall"]
        cards = [("Overall accuracy", _num(now.get(headline)), "",
                  f"{headline.replace('_', ' ').title()} | frozen baseline {_num(base.get(headline))}"),
                 ("Cold-start accuracy", _num(comparison["cinerankx"]["after"]["cold"].get(headline)), "",
                  f"Baseline {_num(comparison['baseline']['results']['cold'].get(headline))}"),
                 *cards[:3]]
    kpi_row(cards)
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
        with card("strategy", "Strategy selector quality", "git-merge",
                  "Predicting which strategy suits a user (a labelling task, not recommendation accuracy)."):
            roc = ml.get("ROC-AUC")
            body = _comparison("balanced accuracy", ml_d.get("balanced_accuracy"),
                               ml_d.get("majority_baseline_balanced_accuracy"), "Chance")
            body += _comparison("ROC-AUC", roc, 0.5, "Chance")
            st.markdown(body, unsafe_allow_html=True)
            acc, maj = ml.get("Accuracy"), ml_d.get("majority_baseline_accuracy")
            if isinstance(acc, (int, float)) and isinstance(maj, (int, float)):
                relation = ("above" if acc - maj > 0.005 else "below" if acc - maj < -0.005 else "level with")
                st.caption(f"Plain accuracy {acc:.3f} is {relation} the majority-class rate {maj:.3f}: most users are "
                           "best served by popularity, so balanced accuracy, macro-F1 and AUC are the fair measures.")
            if isinstance(ml_d.get("noise_ceiling"), (int, float)):
                st.caption(f"Label noise limits accuracy: even an oracle that knows each user's true hit-rates "
                           f"would reach about {ml_d['noise_ceiling']:.2f}.")

    if comparison:
        render_comparison(comparison)

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
            macro = {m: sum(v[m] for v in per_class.values()) / len(per_class) for m in ("precision", "recall", "f1")}
            notes = [f"Macro precision {macro['precision']:.3f}, recall {macro['recall']:.3f}, F1 {macro['f1']:.3f}"]
            if ml_d.get("selected_model"):
                notes.append(f"model: {ml_d['selected_model']}")
            if isinstance(ml_d.get("latency_ms"), dict):
                notes.append(f"inference {ml_d['latency_ms']['mean_ms']:.0f} ms/user")
            rel = ml_d.get("label_reliability")
            if isinstance(rel, dict) and rel.get("label_agreement") is not None:
                notes.append(f"label agreement across two independent holdouts {rel['label_agreement']:.2f} "
                             "(practical accuracy ceiling)")
            st.caption("; ".join(notes) + ".")

    with card("system", "System performance", "activity", "Measured end to end across all engines."):
        kpi_row([("Response time", _num(system.get("Response Time"), "{:.1f}"), "ms"),
                 ("Throughput", _num(system.get("Throughput"), "{:.1f}"), "ops/s"),
                 ("Memory", _num(system.get("Memory Usage"), "{:.0f}"), "MB"),
                 ("CPU", _num(system.get("CPU Usage"), "{:.0f}"), "% of one core")])
        if last_run_ms is not None:
            st.caption(f"Last live recommendation in this session: {last_run_ms:,.0f} ms.")
