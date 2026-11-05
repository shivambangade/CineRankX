"""Strategy Comparison: frozen baseline vs CineRankX on the fixed benchmark (reports/strategy_comparison.json)."""

import html

import pandas as pd
import streamlit as st

from src.ui.theme import card, meter

ROWS = [("NDCG@10", "ndcg@10"), ("Precision@10", "precision@10"), ("Recall@10", "recall@10"),
        ("MAP@10", "map@10"), ("Hit Rate@20", "hit_rate@20"), ("Coverage", "coverage"),
        ("Diversity", "diversity")]


def _delta_badge(after: float, base: float) -> str:
    diff = (after - base) * 100
    if abs(diff) < 0.05:
        return '<span class="badge">No difference from baseline</span>'
    kind, word = ("ok", "above") if diff > 0 else ("warn", "below")
    return f'<span class="badge {kind}">{abs(diff):.1f} pts {word} baseline</span>'


def _panel(title: str, icon_name: str, key: str, cohort: str, comp: dict, headline: str) -> None:
    base = comp["baseline"]["results"][cohort]
    new = comp["cinerankx"]["after"][cohort]
    before = (comp["cinerankx"].get("before") or {}).get(cohort)
    with card(key, title, icon_name, f"{base['n_users']:,} users, same held-out test interactions for both"):
        body = meter("Frozen baseline (most popular)", base[headline], "t", "{:.1%}")
        if before:
            body += meter("CineRankX before (old pipeline)", before[headline], "", "{:.1%}")
        body += meter("CineRankX now", new[headline], "o", "{:.1%}")
        st.markdown(body + _delta_badge(new[headline], base[headline]), unsafe_allow_html=True)
        rows = {"Metric": [], "Baseline": [], "CineRankX": [], "Change": []}
        for label, key_ in ROWS:
            if key_ in base and key_ in new:
                rows["Metric"].append(label)
                rows["Baseline"].append(base[key_])
                rows["CineRankX"].append(new[key_])
                rows["Change"].append(new[key_] - base[key_])
        if "rating" in base and "rating" in new:
            for name in ("rmse", "mae"):
                rows["Metric"].append(name.upper() + " (lower is better)")
                rows["Baseline"].append(base["rating"][name])
                rows["CineRankX"].append(new["rating"][name])
                rows["Change"].append(new["rating"][name] - base["rating"][name])
        st.dataframe(pd.DataFrame(rows).style.format({"Baseline": "{:.3f}", "CineRankX": "{:.3f}",
                                                      "Change": "{:+.3f}"}), hide_index=True, width="stretch")


def render_comparison(comp: dict) -> None:
    headline = comp["headline_metric"]
    k = headline.split("@")[1]
    st.markdown("#### Strategy comparison")
    st.caption(
        f"Accuracy here is Hit Rate@{k}: the share of users with at least one liked (rated 4.0 or more) "
        "held-out movie among their top-%s recommendations. The baseline is frozen (most-rated movies, "
        "benchmark %s) and both strategies are scored on identical test interactions; "
        "tuning used only the validation split." % (k, html.escape(comp["benchmark_fingerprint"][:8])))
    left, right = st.columns(2, gap="medium")
    with left:
        _panel("Overall accuracy", "bar-chart-2", "cmp_overall", "overall", comp, headline)
    with right:
        _panel("Cold-start accuracy", "user", "cmp_cold", "cold", comp, headline)
    st.caption("Cold-start users are simulated: real users whose profile is cut to %d ratings and who "
               "are excluded from training beyond those ratings." % comp["config"]["cold_history"])
