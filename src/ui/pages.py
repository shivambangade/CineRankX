"""Overview and engines screens. Everything shown comes from loaded data, config or reports."""

import html
import inspect

import pandas as pd
import streamlit as st

from src.ranking_engine import MultiObjectiveRanker
from src.ranking_engine.config import RankingWeights
from src.ui.results import objective_label, split_title
from src.ui.theme import card, card_html, kpi_row, meter

FUNNEL_COLORS = ["#2563eb", "#3b82f6", "#f97316", "#0d9488"]


def _pool_size() -> int | None:
    try:
        return int(inspect.signature(MultiObjectiveRanker.rank).parameters["pool_size"].default)
    except (KeyError, ValueError, TypeError):
        return None


def _funnel(movies: pd.DataFrame, ratings: pd.DataFrame, shown: int | None) -> str:
    steps = [("Catalog titles", len(movies)),
             ("Titles with ratings in sample", int(ratings["movieId"].nunique()))]
    pool = _pool_size()
    if pool:
        steps.append(("Candidate pool per user", min(pool, steps[-1][1])))
    if shown:
        steps.append(("Latest ranked slate", shown))
    top = max(steps[0][1], 1)
    out = []
    for i, (label, n) in enumerate(steps):
        pct = n / top
        out.append(f'<div class="funnel-step"><div class="row"><span>{html.escape(label)}</span>'
                   f'<b>{n:,} <span class="brand-sub">{pct:.1%}</span></b></div>'
                   f'<div class="funnel-bar" style="width:{max(pct * 100, 4):.1f}%;'
                   f'background:{FUNNEL_COLORS[i % 4]}"></div></div>')
    return "".join(out)


def _routing(report: dict | None) -> str:
    detail = (report or {}).get("detail", {}).get("machine_learning", {})
    matrix, labels = detail.get("confusion_matrix"), detail.get("confusion_matrix_labels")
    if not matrix or not labels:
        return '<div class="brand-sub">Generate the offline KPI report to see strategy routing.</div>'
    predicted = [sum(row[j] for row in matrix) for j in range(len(labels))]
    total = max(sum(predicted), 1)
    body = "".join(meter(f"{lab.replace('_', ' ').title()} ({n})", n / total, "o" if i == 0 else "",
                         "{:.0%}") for i, (lab, n) in enumerate(zip(labels, predicted)))
    return body + f'<div class="brand-sub">Predicted strategy share over {total} held-out users.</div>'


def render_overview(movies, ratings, classifier_ready: bool, ranked, report: dict | None,
                    last_run_ms: float | None) -> None:
    st.subheader("Overview")
    if movies is None or ratings is None:
        st.info("Local datasets are not ready. Run the ingestion pipeline to see live platform metrics.")
        return
    search = (report or {}).get("search", {})
    ml = (report or {}).get("machine_learning", {})
    kpi_row([
        ("Catalog titles", f"{len(movies):,}", "", "Merged catalog"),
        ("Ratings in sample", f"{len(ratings):,}", "", f"{ratings['userId'].nunique():,} users"),
        ("Strategy model", "Ready" if classifier_ready else "Missing", "",
         f"Accuracy {ml['Accuracy']:.2f}" if isinstance(ml.get("Accuracy"), (int, float)) else "No report yet"),
        ("Search MRR", f"{search['MRR']:.3f}" if isinstance(search.get("MRR"), (int, float)) else "n/a",
         "", "Offline benchmark"),
    ])
    st.write("")
    left, right = st.columns([7, 5], gap="medium")
    with left:
        with card("funnel", "Candidate funnel", "layers", "From the full catalog down to a ranked slate."):
            st.markdown(_funnel(movies, ratings, None if ranked is None else len(ranked)),
                        unsafe_allow_html=True)
    with right:
        with card("harmony", "Objective harmony", "bar-chart-2", "Configured weights, normalized to 1.0."):
            try:
                items = RankingWeights.from_config().items()
                st.markdown("".join(meter(objective_label(n), w, "o" if i == 0 else "")
                                    for i, (n, w) in enumerate(items)), unsafe_allow_html=True)
            except (OSError, ValueError):
                st.caption("Objective weights are unavailable in eval_config.yaml.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with card("routing", "Strategy routing", "git-merge", "How users are dispatched."):
            st.markdown(_routing(report), unsafe_allow_html=True)
    with c2:
        with card("telemetry", "Operational telemetry", "activity", "Measured latency and resources."):
            system = (report or {}).get("system", {})
            rows = [("Response time", system.get("Response Time"), "ms", "{:.1f}"),
                    ("Throughput", system.get("Throughput"), "ops/s", "{:.1f}"),
                    ("Memory", system.get("Memory Usage"), "MB", "{:.0f}"),
                    ("Last live run", last_run_ms, "ms", "{:,.0f}")]
            shown = [f'<div class="row"><span>{l}</span><b>{f.format(v)} {u}</b></div><div class="bar"></div>'
                     for l, v, u, f in rows if isinstance(v, (int, float))]
            st.markdown("".join(shown) or '<div class="brand-sub">Run the KPI report or generate '
                        'recommendations to see telemetry.</div>', unsafe_allow_html=True)
    with c3:
        with card("candidates", "Surfaced candidates", "film", "Top results of the latest run."):
            if ranked is not None and not ranked.empty:
                st.markdown("".join(
                    f'<div class="row"><span>{r.rank}. {html.escape(split_title(r.title)[0])}</span>'
                    f'<b>{r.score:.3f}</b></div><div class="bar o"><span style="width:{r.score * 100:.0f}%">'
                    f'</span></div>' for r in ranked.head(5).itertuples(index=False)), unsafe_allow_html=True)
            else:
                st.caption("Generate recommendations in Run Engine to surface candidates here.")


def render_engines(report: dict | None, ir_ready: bool, classifier_ready: bool, data_ready: bool) -> None:
    st.subheader("Recommendation engines")
    search = (report or {}).get("search", {})
    ml = (report or {}).get("machine_learning", {})
    rec = (report or {}).get("recommendation", {})
    ml_detail = (report or {}).get("detail", {}).get("machine_learning", {})

    def num(d, key, fmt="{:.3f}"):
        return fmt.format(d[key]) if isinstance(d.get(key), (int, float)) else "n/a"

    modules = [
        ("search", "Classical IR engine", ir_ready,
         "TF-IDF vector space model over genres, tags and overview text with cosine similarity, "
         "plus a Trie for prefix autocomplete on titles.",
         [("Precision@10", num(search, "Precision@10")), ("MRR", num(search, "MRR")),
          ("NDCG", num(search, "NDCG")), ("Latency", num(search, "Search Latency", "{:.2f} ms"))]),
        ("cpu", "Adaptive strategy selector", classifier_ready,
         "A classifier predicts per user whether collaborative, content-based or popularity "
         "recommendations fit best, from rating-history profile features.",
         [("Accuracy", num(ml, "Accuracy")), ("F1", num(ml, "F1")), ("ROC-AUC", num(ml, "ROC-AUC")),
          ("Majority baseline", num(ml_detail, "majority_baseline_accuracy"))]),
        ("git-merge", "Multi-objective hybrid ranker", data_ready,
         "Re-ranks candidates from the IR and strategy sources on six weighted objectives: "
         "relevance, diversity, novelty, coverage, popularity/quality and predicted rating.",
         [("MAP", num(rec, "MAP")), ("Diversity", num(rec, "Diversity")),
          ("Novelty", num(rec, "Novelty")), ("Coverage", num(rec, "Coverage"))]),
    ]
    for col, (icon_name, title, ready, text, stats) in zip(st.columns(3, gap="medium"), modules):
        badge = '<span class="badge ok">Loaded</span>' if ready else '<span class="badge warn">Unavailable</span>'
        stat_html = "".join(f'<div class="row"><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>'
                            '<div class="bar"></div>' for a, b in stats)
        col.markdown(card_html(title, icon_name,
                               f'<p class="brand-sub">{html.escape(text)}</p>{badge}'
                               f'<div style="height:12px"></div>{stat_html}'), unsafe_allow_html=True)
    st.caption("Metrics come from the latest offline KPI report. Weights and K are set in Run Engine.")
