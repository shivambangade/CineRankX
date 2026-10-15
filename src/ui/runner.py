"""Recommendation runner screen: source, user, K, objective weights, ranked results."""

import html
import time

import pandas as pd
import streamlit as st

from src.ranking_engine.config import OBJECTIVES, RankingWeights
from src.ui.results import (OBJECTIVE_HELP, genres_lookup, objective_label, output_kpis,
                            prepare_results, reason_for, split_title)
from src.ui.nav import open_profile
from src.ui.theme import card, kpi_row, meter

MAX_UPLOAD_MB = 25
SOURCES = ["Live engine", "Upload ranked CSV"]


def _default_weights() -> dict[str, float] | None:
    try:
        return RankingWeights.from_config().weights
    except (OSError, ValueError, KeyError):
        return None


def _reset_weights(defaults: dict[str, float]) -> None:
    for name in OBJECTIVES:
        st.session_state[f"w_{name}"] = float(round(defaults[name], 2))


def _weights_panel(defaults: dict[str, float]) -> dict[str, float]:
    """Render the six sliders plus a total/normalization indicator; return raw values."""
    with card("weights", "Objective weights", "sliders",
              "Relative importance of each ranking objective, normalized to 100% on run."):
        for name in OBJECTIVES:
            st.session_state.setdefault(f"w_{name}", float(round(defaults[name], 2)))
        left, right = st.columns(2, gap="large")
        raw = {}
        for i, name in enumerate(OBJECTIVES):
            raw[name] = (left if i % 2 == 0 else right).slider(
                objective_label(name), 0.0, 1.0, step=0.05, key=f"w_{name}",
                help=f"{OBJECTIVE_HELP[name]} Range 0.00 to 1.00.")
        total = sum(raw.values())
        if total > 0:
            segments = "".join(
                f'<span style="flex:{v / total:.4f}" class="badge {"hot" if i == 0 else ""}">'
                f'{objective_label(n).split()[0]} {v / total:.0%}</span>'
                for i, (n, v) in enumerate(raw.items()) if v > 0)
            st.markdown(
                f'<div class="sum-pill"><span>Raw total <b>{total:.2f}</b> &rarr; normalized to '
                f'<b>1.00</b></span><span class="badge ok">Valid</span></div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:10px">{segments}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="sum-pill"><span>Raw total <b>0.00</b>: set at least one '
                        'weight above zero.</span><span class="badge warn">Invalid</span></div>',
                        unsafe_allow_html=True)
        st.button("Reset to configured defaults", on_click=_reset_weights, args=(defaults,),
                  key="reset_weights", help="Restores the weights from eval_config.yaml.")
    return raw


def _live_controls(users: set[int], max_user: int) -> tuple[int, int, bool]:
    with card("target", "Target user", "user", "Who to recommend for, and how many titles."):
        c1, c2 = st.columns([1, 1])
        user_id = c1.number_input("MovieLens user ID", min_value=1, max_value=max(max_user, 1),
                                  value=1, step=1, key="user_id",
                                  help=f"Valid IDs are in the loaded ratings sample (1 to {max_user}).")
        k = c2.number_input("Recommendations (K)", min_value=1, max_value=20, value=10, step=1, key="k",
                            help="How many ranked titles to return (1 to 20).")
        valid = int(user_id) in users
        if valid and st.session_state.get("profile_db_ready"):
            st.button("Open full profile", key="open_profile", on_click=open_profile,
                      args=(int(user_id),), help="History, preferences and past runs for this user.")
        if valid:
            st.markdown(f'<span class="badge ok">User {int(user_id)} found</span> '
                        f'<span class="brand-sub">{len(users):,} users in the loaded sample</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge warn">User {int(user_id)} not in sample</span> '
                        f'<span class="brand-sub">Pick an ID between 1 and {max_user}.</span>',
                        unsafe_allow_html=True)
    return int(user_id), int(k), valid


def show_results(results: pd.DataFrame, context: str, genres: dict[int, list[str]]) -> None:
    if results.empty:
        st.warning("No recommendations were returned for this selection.")
        return
    kp = output_kpis(results)
    st.markdown("#### Ranked recommendations")
    st.caption(context)
    kpi_row([("Movies", str(kp["titles"])), ("Avg rank score", f'{kp["mean_score"]:.3f}'),
             ("Avg relevance", f'{kp["mean_relevance"]:.3f}'), ("Avg novelty", f'{kp["mean_novelty"]:.3f}')])
    st.caption("Summaries of this list's ranking objectives, not held-out precision, recall or coverage.")
    st.download_button("Export ranked CSV", results.to_csv(index=False).encode("utf-8"),
                       file_name="recommendations.csv", mime="text/csv", key="export")
    cards = []
    for row in results.itertuples(index=False):
        name, year = split_title(row.title)
        chips = "".join(f'<span class="chip2">{html.escape(g)}</span>' for g in genres.get(int(row.movieId), [])[:4])
        bars = "".join(f"<div>{meter(objective_label(o), getattr(row, o), 'o' if o == 'relevance' else '')}</div>"
                       for o in OBJECTIVES)
        meta = f"{year} · " if year else ""
        cards.append(
            f'<article class="movie{" top" if row.rank == 1 else ""}"><div class="movie-head">'
            f'<div class="movie-rank">{row.rank}</div><div><div class="movie-title">{html.escape(name)}</div>'
            f'<div class="movie-meta">{meta}Movie ID {row.movieId}</div></div>'
            f'<div class="score"><span class="eyebrow">Rank score</span><b>{row.score:.3f}</b></div></div>'
            f'<div class="chips">{chips}</div><div class="reason">{html.escape(reason_for(row))}</div>'
            f'<div class="mini">{bars}</div></article>')
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _csv_source() -> pd.DataFrame | None:
    with card("upload", "Ranked CSV", "database",
              f"Open an existing Module 4 output (up to {MAX_UPLOAD_MB} MB)."):
        uploaded = st.file_uploader("Ranked output file", type="csv", key="csv_upload",
                                    help="Needs rank, movieId, title, score, sources and the six objectives.")
        if uploaded is None:
            st.caption("Expected columns: rank, movieId, title, score, sources, plus the six objectives.")
            return None
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {uploaded.size / 1e6:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.")
            return None
        try:
            frame = prepare_results(pd.read_csv(uploaded))
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
            st.error(f"Could not read ranked output: {error}")
            return None
        st.markdown(f'<span class="badge ok">Validated</span> '
                    f'<span class="brand-sub">{len(frame)} rows from {html.escape(uploaded.name)}</span>',
                    unsafe_allow_html=True)
        st.session_state["csv_name"] = uploaded.name
        return frame


def render_runner(data_ready: bool, load_engine, recommend) -> None:
    st.subheader("Recommendation runner")
    source = st.segmented_control("Source", SOURCES, default=SOURCES[0], key="source")
    if source == SOURCES[1]:
        frame = _csv_source()
        if frame is not None:
            genres = {}
            if data_ready:
                genres = genres_lookup(load_engine()[0])
            show_results(frame, st.session_state.get("csv_name", "Uploaded file"), genres)
        return
    if not data_ready:
        st.info("Local datasets are not ready. Run `python -m src.ingestion.build_dataset` "
                "(or `scripts/build_movielens_only.py`), or switch to Upload ranked CSV.")
        return
    movies, ratings, _, _ = load_engine()
    users = set(ratings["userId"].astype(int).unique())
    defaults = _default_weights()
    left, right = st.columns([5, 7], gap="medium")
    with left:
        user_id, k, user_ok = _live_controls(users, max(users) if users else 1)
    with right:
        raw = _weights_panel(defaults) if defaults else None
        if raw is None:
            st.warning("Objective weights could not be read from eval_config.yaml; engine defaults apply.")
    weights_ok = raw is None or sum(raw.values()) > 0
    ready = user_ok and weights_ok
    run = st.button("Generate recommendations", type="primary", disabled=not ready,
                    width="stretch", key="run")
    if not ready:
        st.caption("Enter a valid user ID and at least one non-zero weight to enable.")
    if run:
        try:
            weights = RankingWeights(raw) if raw else None
            started = time.perf_counter()
            with st.status("Predicting strategy and ranking candidates...", expanded=False) as status:
                ranked, strategy, confidence = recommend(user_id, k, weights)
                elapsed = (time.perf_counter() - started) * 1000
                status.update(label=f"Done in {elapsed:,.0f} ms", state="complete")
            st.session_state.update(
                ranked=ranked, run_ms=elapsed,
                run_context=(f"User {user_id} | {strategy.replace('_', ' ').title()} "
                             f"({confidence:.0%} confidence) | {elapsed:,.0f} ms" if strategy else
                             f"User {user_id} | Strategy model unavailable; default source blend | "
                             f"{elapsed:,.0f} ms"))
            st.session_state["run_strategy"] = strategy
        except (ValueError, FileNotFoundError) as error:
            st.session_state.pop("ranked", None)
            st.error(str(error))
    results = st.session_state.get("ranked")
    if results is None:
        st.info("Set a user and weights, then generate recommendations. Results appear here.")
    else:
        show_results(results, st.session_state.get("run_context", ""), genres_lookup(movies))
