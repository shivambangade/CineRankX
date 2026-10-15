"""User 360 screen: summary, history, preferences, past runs and current recommendations."""

import pandas as pd
import streamlit as st

from src.ranking_engine.config import OBJECTIVES, RankingWeights
from src.storage import (active_users, genre_preferences, rating_distribution, user_summary,
                         user_timeline)
from src.ui.nav import selected_user, set_user
from src.ui.profile_history import render_history, render_runs
from src.ui.runner import show_results
from src.ui.results import genres_lookup
from src.ui.theme import card, kpi_row, meter

DB_HINT = "`PYTHONPATH=. python scripts/build_user_db.py`"


def _picker(conn) -> int | None:
    current = selected_user()
    c1, c2 = st.columns([1, 2])
    value = c1.number_input("User ID", min_value=1, value=current or 1, step=1, key="picker_id",
                            help="Enter any MovieLens user ID.")
    prefix = c2.text_input("Find users by ID prefix", key="picker_prefix",
                           placeholder="e.g. 12 lists the most active users starting with 12")
    if prefix.strip():
        found = active_users(conn, prefix, 8)
        if found.empty:
            c2.caption("No users match that prefix.")
        else:
            choice = c2.selectbox("Matches (most active first)", found["userId"].tolist(),
                                  format_func=lambda u: f"User {u} · "
                                  f"{int(found.set_index('userId').loc[u, 'n_ratings']):,} ratings",
                                  key="picker_choice")
            value = choice
    chosen = int(value)
    if chosen != current:
        set_user(chosen)
    return chosen


def _weights_from_session() -> RankingWeights | None:
    raw = {n: st.session_state.get(f"w_{n}") for n in OBJECTIVES}
    if all(isinstance(v, (int, float)) for v in raw.values()) and sum(raw.values()) > 0:
        return RankingWeights(raw)
    return None


def _overview(conn, user_id: int, summary: dict) -> None:
    kpi_row([("Ratings", f"{summary['n_ratings']:,}", "", "Movies rated"),
             ("Average rating", f"{summary['avg_rating']:.2f}", "/ 5"),
             ("Unique genres", str(summary["n_genres"])),
             ("First activity", summary["first_ts"][:10]),
             ("Latest activity", summary["last_ts"][:10])])
    st.write("")
    left, right = st.columns(2, gap="medium")
    dist = rating_distribution(conn, user_id)
    with left:
        with card("p_dist", "Rating distribution", "bar-chart-2", "How this user scores titles."):
            st.bar_chart(dist.rename("ratings"), color="#f97316", height=240)
    monthly = user_timeline(conn, user_id)["monthly"]
    with right:
        with card("p_trend", "Activity over time", "activity", "Ratings submitted per month."):
            st.line_chart(monthly.set_index("month")["ratings"], color="#2563eb", height=240) \
                if len(monthly) > 1 else st.caption("Not enough dated activity to chart a trend.")
    st.info("Profile attributes: MovieLens 20M contains no demographic or account fields (name, age, "
            "location, plan). Only the identifier and rating activity are available; supplying them "
            "would require a user-attributes source in the backend.")


def _preferences(conn, user_id: int) -> None:
    prefs = genre_preferences(conn, user_id)
    if prefs.empty:
        st.info("No genre data is available for this user.")
        return
    timeline = user_timeline(conn, user_id)
    left, right = st.columns(2, gap="medium")
    with left:
        with card("p_genre", "Favorite genres", "film", "By number of ratings; bar shows mean rating."):
            top = prefs.head(8)
            st.markdown("".join(meter(f"{r.genre} ({r.count})", r.avg_rating / 5, "o" if i == 0 else "")
                                for i, r in enumerate(top.itertuples())), unsafe_allow_html=True)
            loved = prefs[prefs["count"] >= 3].sort_values("avg_rating", ascending=False).head(3)
            if not loved.empty:
                st.caption("Highest-rated genres (3+ ratings): " + ", ".join(
                    f"{r.genre} {r.avg_rating:.2f}" for r in loved.itertuples()))
    with right:
        with card("p_era", "Preferred eras", "clock", "Ratings by release decade."):
            decades = timeline["decades"]
            if decades.empty:
                st.caption("Release years are unavailable for this user's titles.")
            else:
                st.bar_chart(decades.set_index("decade")["ratings"], color="#0d9488", height=220)
    languages = timeline["languages"]
    with card("p_lang", "Languages", "grid", "Original language, where TMDB metadata is available."):
        if languages.empty:
            st.caption("Language data is unavailable: it comes from the TMDB join, which has not "
                       "been loaded into the user store. Re-run the ingestion and build_user_db.")
        else:
            st.dataframe(languages.head(8).rename(columns={"original_language": "Language"}),
                         hide_index=True, width="stretch")


def _current(conn, user_id: int, data_ready: bool, load_engine, recommend) -> None:
    if not data_ready:
        st.info("Engine data is not loaded; build the processed catalog to generate recommendations.")
        return
    movies, ratings, _, _ = load_engine()
    if user_id not in set(ratings["userId"].astype(int).unique()):
        st.warning(f"User {user_id} is outside the engine's loaded ratings sample "
                   f"({ratings['userId'].nunique():,} users). Their profile and history are complete, but "
                   "live recommendations are limited to the sample.")
        return
    k = st.number_input("Recommendations (K)", 1, 20, 10, key="profile_k")
    weights = _weights_from_session()
    st.caption("Using the objective weights from Run Engine." if weights else
               "Using the configured default weights.")
    if st.button("Generate recommendations", type="primary", key="profile_run"):
        try:
            with st.status("Predicting strategy and ranking candidates...") as status:
                ranked, strategy, confidence = recommend(user_id, int(k), weights)
                status.update(label="Done", state="complete")
            st.session_state["profile_result"] = (user_id, ranked, strategy, confidence)
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))
    saved = st.session_state.get("profile_result")
    if saved and saved[0] == user_id:
        _, ranked, strategy, confidence = saved
        label = (f"User {user_id} | {strategy.replace('_', ' ').title()} ({confidence:.0%} confidence)"
                 if strategy else f"User {user_id} | default source blend")
        show_results(ranked, label, genres_lookup(movies))
    else:
        st.info("No recommendations generated for this user in this session.")


def render_profile(conn, data_ready: bool, load_engine, recommend) -> None:
    st.subheader("User profile")
    user_id = _picker(conn)
    summary = user_summary(conn, user_id)
    if summary is None:
        st.error(f"User {user_id} was not found in the ratings store. Pick another ID or search "
                 "by prefix above.")
        return
    st.markdown(f'<span class="badge hot">User {user_id}</span> <span class="brand-sub">'
                f'{summary["n_ratings"]:,} ratings between {summary["first_ts"][:10]} and '
                f'{summary["last_ts"][:10]}</span>', unsafe_allow_html=True)
    genres = genre_preferences(conn, user_id)["genre"].tolist()
    tabs = st.tabs(["Overview", "History", "Preferences", "Past runs", "Current recommendations"])
    with tabs[0]:
        _overview(conn, user_id, summary)
    with tabs[1]:
        render_history(conn, user_id, sorted(genres))
    with tabs[2]:
        _preferences(conn, user_id)
    with tabs[3]:
        render_runs(conn, user_id)
    with tabs[4]:
        _current(conn, user_id, data_ready, load_engine, recommend)
