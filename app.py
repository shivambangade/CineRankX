"""Streamlit UI for live or exported Module 4 recommendation output."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ranking_engine import MultiObjectiveRanker, OBJECTIVES
from src.strategy_selector import StrategyClassifier, extract_profile_features
from src.ui.results import output_kpis, prepare_results

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data/processed/movies_merged.csv"
RATINGS = ROOT / "data/processed/ratings_clean.csv"
RAW_RATINGS = ROOT / "newdata/movielens/rating.csv"
RATINGS_LIMIT = 400_000


@st.cache_resource(show_spinner="Loading catalog and ranking engine...")
def load_engine():
    from src.ir_engine import IREngine

    movies = pd.read_csv(CATALOG)
    ratings_path = RATINGS if RATINGS.exists() else RAW_RATINGS
    ratings = pd.read_csv(ratings_path, nrows=RATINGS_LIMIT)
    ir = IREngine()
    ir_engine = ir if ir.load() else None
    ranker = MultiObjectiveRanker(movies, ratings, ir_engine=ir_engine)
    classifier = StrategyClassifier()
    return movies, ratings, ranker, classifier if classifier.load() else None


def recommend(user_id: int, k: int):
    movies, ratings, ranker, classifier = load_engine()
    if user_id not in ratings["userId"].values:
        raise ValueError(f"User {user_id} is not in the loaded ratings sample.")
    strategy, confidence = None, 1.0
    if classifier is not None:
        profile = extract_profile_features(ratings[ratings["userId"] == user_id], movies)
        prediction = classifier.predict(profile).iloc[0]
        strategy = str(prediction["predicted_strategy"])
        confidence = float(prediction["confidence"])
    ranked = prepare_results(ranker.rank(user_id, k=k, strategy=strategy, confidence=confidence))
    return ranked, strategy, confidence


st.set_page_config(page_title="CineRankX | Recommendations", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f7f8f7; color: #172522; }
    .block-container { max-width: 1160px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    .brand { font-size: .8rem; font-weight: 800; color: #16735c; letter-spacing: 0; }
    .stMetric { background: #fff; border: 1px solid #dce4df; border-radius: 6px; padding: 14px 18px; }
    .movie-line { border-top: 1px solid #dce4df; padding: 16px 0 4px; }
    .movie-rank { color: #16735c; font-weight: 800; font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<span class="brand">CINERANKX / RESULTS</span>', unsafe_allow_html=True)
st.title("Movie recommendations")

mode = st.segmented_control("Source", ["Run engine", "Upload ranked CSV"], default="Run engine")
results = None
context = ""

if mode == "Run engine":
    if not CATALOG.exists() or not (RATINGS.exists() or RAW_RATINGS.exists()):
        st.info(
            "Local datasets are not ready. Run the ingestion pipeline, or select "
            "Upload ranked CSV to view an existing Module 4 result."
        )
    else:
        st.caption(f"Using the first {RATINGS_LIMIT:,} ratings from the local dataset.")
        with st.form("recommend"):
            left, right = st.columns([2, 1])
            user_id = left.number_input("User ID", min_value=1, value=1, step=1)
            k = right.slider("Recommendations", min_value=1, max_value=20, value=10)
            submitted = st.form_submit_button("Generate recommendations", type="primary")
        if submitted:
            try:
                with st.spinner("Ranking movies..."):
                    ranked, strategy, confidence = recommend(int(user_id), k)
                st.session_state["ranked"] = ranked
                st.session_state["run_context"] = (
                    f"User {user_id} | {strategy.replace('_', ' ').title()} "
                    f"({confidence:.0%} confidence)" if strategy else
                    f"User {user_id} | Strategy model unavailable; using default source blend"
                )
            except (ValueError, FileNotFoundError) as error:
                st.session_state.pop("ranked", None)
                st.error(str(error))
        results = st.session_state.get("ranked")
        context = st.session_state.get("run_context", "")
else:
    uploaded = st.file_uploader("Module 4 ranked output", type="csv")
    if uploaded is not None:
        try:
            results = prepare_results(pd.read_csv(uploaded))
            context = uploaded.name
        except (ValueError, pd.errors.ParserError, UnicodeError) as error:
            st.error(f"Could not read ranked output: {error}")
    else:
        st.caption("Upload a CSV with the columns returned by MultiObjectiveRanker.rank().")

if results is not None:
    if results.empty:
        st.warning("No recommendations were returned for this selection.")
    else:
        st.subheader("Ranked for you")
        st.caption(context)
        kpis = output_kpis(results)
        a, b, c, d = st.columns(4)
        a.metric("Movies", kpis["titles"])
        b.metric("Average rank score", f'{kpis["mean_score"]:.3f}')
        c.metric("Average relevance", f'{kpis["mean_relevance"]:.3f}')
        d.metric("Average novelty", f'{kpis["mean_novelty"]:.3f}')
        st.caption(
            "These are summaries of this list's ranking objectives, not held-out "
            "precision, recall, or catalog-wide coverage."
        )
        st.download_button(
            "Download ranked CSV", results.to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv", mime="text/csv",
        )
        for row in results.itertuples(index=False):
            st.markdown('<div class="movie-line"></div>', unsafe_allow_html=True)
            rank_col, title_col, score_col = st.columns([1, 8, 2])
            rank_col.markdown(f'<span class="movie-rank">{row.rank:02d}</span>', unsafe_allow_html=True)
            title_col.write(row.title)
            title_col.caption(f"Movie ID {row.movieId} | Sources: {row.sources or 'Unspecified'}")
            score_col.metric("Rank score", f"{row.score:.3f}")
            with st.expander("Score breakdown"):
                st.dataframe(
                    pd.DataFrame(
                        {"Objective": [name.replace("_", " ").title() for name in OBJECTIVES],
                         "Score": [getattr(row, name) for name in OBJECTIVES]}
                    ),
                    hide_index=True, use_container_width=True,
                )
