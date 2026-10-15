"""Profile tabs backed by the store: paginated rating history and past recommendation runs."""

import html

import pandas as pd
import streamlit as st

from src.storage import history_page, list_runs
from src.ui.results import objective_label

POSTER_BASE = "https://image.tmdb.org/t/p/w92"
PAGE_SIZES = [10, 25, 50, 100]
SORTS = {"Most recent": ("timestamp", True), "Oldest": ("timestamp", False),
         "Highest rated": ("rating", True), "Lowest rated": ("rating", False),
         "Title A-Z": ("title", False), "Newest release": ("year", True)}


def render_history(conn, user_id: int, genres: list[str]) -> None:
    c1, c2, c3 = st.columns([3, 2, 2])
    search = c1.text_input("Search titles", key=f"h_search_{user_id}", placeholder="e.g. toy story")
    genre = c2.selectbox("Genre", ["All", *genres], key=f"h_genre_{user_id}")
    order = c3.selectbox("Sort by", list(SORTS), key=f"h_sort_{user_id}")
    c4, c5 = st.columns([3, 2])
    low, high = c4.slider("Rating range", 0.5, 5.0, (0.5, 5.0), 0.5, key=f"h_range_{user_id}")
    size = c5.selectbox("Rows per page", PAGE_SIZES, index=1, key=f"h_size_{user_id}")

    signature = (search, genre, order, low, high, size)
    pkey = f"h_page_{user_id}"
    if st.session_state.get(f"{pkey}_sig") != signature:
        st.session_state[pkey], st.session_state[f"{pkey}_sig"] = 0, signature
    column, descending = SORTS[order]
    frame, total = history_page(conn, user_id, search=search, genre=genre, min_rating=low,
                                max_rating=high, sort=column, descending=descending,
                                page=st.session_state[pkey], page_size=size)
    pages = max(1, -(-total // size))
    st.session_state[pkey] = min(st.session_state[pkey], pages - 1)
    if total == 0:
        st.info("No ratings match these filters." if (search or genre != "All" or low > 0.5 or high < 5)
                else "This user has no rating history.")
        return
    frame = frame.assign(
        Poster=frame["poster_path"].apply(lambda p: f"{POSTER_BASE}{p}" if isinstance(p, str) and p else None),
        Event="Rating", Genres=frame["genres"].fillna("").str.replace("|", ", ", regex=False),
        Year=frame["year"].astype("Int64"))
    shown = frame[["Poster", "title", "Year", "Genres", "rating", "timestamp", "Event"]].rename(
        columns={"title": "Title", "rating": "Rating", "timestamp": "Timestamp"})
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "Poster": st.column_config.ImageColumn("Poster", width="small"),
        "Rating": st.column_config.NumberColumn(format="%.1f"),
        "Year": st.column_config.NumberColumn(format="%d")})
    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("Previous", disabled=st.session_state[pkey] <= 0, key=f"{pkey}_prev"):
        st.session_state[pkey] -= 1
        st.rerun()
    p2.markdown(f'<div style="text-align:center" class="brand-sub">Page {st.session_state[pkey] + 1} of '
                f'{pages} · {total:,} ratings</div>', unsafe_allow_html=True)
    if p3.button("Next", disabled=st.session_state[pkey] >= pages - 1, key=f"{pkey}_next"):
        st.session_state[pkey] += 1
        st.rerun()


def render_runs(conn, user_id: int) -> None:
    runs = list_runs(conn, user_id, limit=25)
    if not runs:
        st.info("No recommendation runs recorded for this user yet. Generate recommendations "
                "in Current recommendations or Run Engine and they will be listed here.")
        return
    for run in runs:
        ok = run["status"] == "success"
        strategy = (run["strategy"] or "default blend").replace("_", " ").title()
        label = f"{run['created_at'].replace('T', ' ')} UTC · {strategy} · K={run['k']} · {run['status']}"
        with st.expander(label, expanded=run is runs[0]):
            conf = f"{run['confidence']:.0%}" if run["confidence"] is not None else "n/a"
            elapsed = f"{run['elapsed_ms']:,.0f} ms" if run["elapsed_ms"] is not None else "n/a"
            badge = '<span class="badge ok">Success</span>' if ok else '<span class="badge warn">Error</span>'
            st.markdown(f"{badge} <span class='brand-sub'>Source: {html.escape(str(run['source']))} · "
                        f"Confidence {conf} · {elapsed}</span>", unsafe_allow_html=True)
            if run["error"]:
                st.error(run["error"])
            if run["weights"]:
                total = sum(run["weights"].values()) or 1
                st.caption("Objective weights: " + ", ".join(
                    f"{objective_label(n)} {v / total:.0%}" for n, v in run["weights"].items()))
            results: pd.DataFrame | None = run["results"]
            if results is not None and not results.empty:
                cols = [c for c in ("rank", "title", "score", "sources") if c in results.columns]
                st.dataframe(results[cols], hide_index=True, width="stretch")
                st.download_button("Export this run", results.to_csv(index=False).encode("utf-8"),
                                   file_name=f"run_{run['id']}_user_{user_id}.csv", mime="text/csv",
                                   key=f"export_run_{run['id']}")
