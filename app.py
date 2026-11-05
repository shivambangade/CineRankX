"""Streamlit UI for live or exported Module 4 recommendation output."""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ranking_engine import MultiObjectiveRanker
from src.ranking_engine.config import RankingWeights
from src.strategy_selector import StrategyClassifier
from src.strategy_selector.config import load_selector_config
from src.strategy_selector.strategy_context import build_features, load_context
from src.storage import DB_PATH, connect, db_ready, record_run
from src.ui.kpi_view import render_report
from src.ui.nav import LABEL_TO_VIEW, VIEWS, selected_user, sync_view
from src.ui.profile import DB_HINT, render_profile
from src.ui.pages import render_engines, render_overview
from src.ui.results import prepare_results
from src.ui.runner import render_runner
from src.ui.theme import apply_theme, header

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data/processed/movies_merged.csv"
RATINGS = ROOT / "data/processed/ratings_clean.csv"
RAW_RATINGS = ROOT / "newdata/movielens/rating.csv"
REPORT = ROOT / "reports/kpi_report.json"
COMPARISON = ROOT / "reports/strategy_comparison.json"
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
    return movies, ratings, ranker, classifier if classifier.load() else None, ir_engine is not None


@st.cache_resource(show_spinner=False)
def strategy_context():
    """Reference population the strategy-aware features are computed against (None if untrained)."""
    return load_context()


def engine_parts():
    return load_engine()[:4]


def _log_run(user_id, k, weights, status, **fields) -> None:
    """Persist a run to the user store when it exists; never let logging break a run."""
    if not db_ready(DB_PATH):
        return
    try:
        used = (weights or RankingWeights.from_config()).weights
        conn = db_connection()
        record_run(conn, user_id, source="live", k=k, weights=used, status=status, **fields)
    except Exception:  # noqa: BLE001 - logging is best effort
        pass


def recommend(user_id: int, k: int, weights: RankingWeights | None = None):
    started = time.perf_counter()
    try:
        movies, ratings, ranker, classifier = engine_parts()
        if user_id not in ratings["userId"].values:
            raise ValueError(f"User {user_id} is not in the loaded ratings sample.")
        if not 1 <= k <= 20:
            raise ValueError("Recommendations must be between 1 and 20.")
        strategy, confidence = None, 1.0
        context = strategy_context()
        if classifier is not None and context is not None:
            cfg = load_selector_config()
            profile = build_features(ratings[ratings["userId"] == user_id], movies, context,
                                     cfg["inner_splits"], cfg["inner_fraction"])
            prediction = classifier.predict(profile).iloc[0]
            strategy = str(prediction["predicted_strategy"])
            confidence = float(prediction["confidence"])
        ranked = prepare_results(ranker.rank(user_id, k=k, strategy=strategy, confidence=confidence,
                                             weights=weights))
    except (ValueError, FileNotFoundError) as error:
        _log_run(user_id, k, weights, "error", strategy=None, confidence=None, error=str(error),
                 elapsed_ms=(time.perf_counter() - started) * 1000)
        raise
    _log_run(user_id, k, weights, "success", strategy=strategy, confidence=confidence,
             elapsed_ms=(time.perf_counter() - started) * 1000, results=ranked)
    return ranked, strategy, confidence


@st.cache_resource
def db_connection():
    return connect(DB_PATH)


@st.cache_data(show_spinner=False)
def load_report(mtime: float, path: str = str(REPORT)) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


st.set_page_config(page_title="CineRankX", layout="wide")
apply_theme()
header()

if "nav" not in st.session_state:
    st.session_state["nav"] = VIEWS.get(st.query_params.get("view") or "", VIEWS["run"])
st.segmented_control("Section", list(VIEWS.values()), key="nav", on_change=sync_view,
                     label_visibility="collapsed")
view = LABEL_TO_VIEW.get(st.session_state.get("nav") or VIEWS["run"], "run")

has_db = db_ready(DB_PATH)
st.session_state["profile_db_ready"] = has_db
if selected_user() and "profile_user" not in st.session_state:
    st.session_state["profile_user"] = selected_user()
data_ready = CATALOG.exists() and (RATINGS.exists() or RAW_RATINGS.exists())
report = load_report(REPORT.stat().st_mtime) if REPORT.exists() else None
last_run_ms = st.session_state.get("run_ms")

try:
    if view == "overview":
        parts = load_engine() if data_ready else None
        render_overview(parts[0] if parts else None, parts[1] if parts else None,
                        bool(parts and parts[3] is not None), st.session_state.get("ranked"),
                        report, last_run_ms)
    elif view == "engines":
        parts = load_engine() if data_ready else None
        render_engines(report, bool(parts and parts[4]), bool(parts and parts[3] is not None), data_ready)
    elif view == "profile":
        if has_db:
            render_profile(db_connection(), data_ready, engine_parts, recommend)
        else:
            st.subheader("User profile")
            st.info(f"The user store has not been built yet. Run {DB_HINT} to index the ratings.")
    elif view == "kpi":
        st.subheader("KPI benchmark suite")
        if report:
            comparison = (load_report(COMPARISON.stat().st_mtime, str(COMPARISON))
                          if COMPARISON.exists() else None)
            render_report(report, last_run_ms, comparison)
        else:
            st.info("No offline report found. Run `PYTHONPATH=. python scripts/generate_kpi_report.py` "
                    "to compute the KPIs defined in eval_config.yaml.")
    else:
        render_runner(data_ready, engine_parts, recommend)
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as error:
    st.error(f"CineRankX could not load its data: {error}. Rebuild data/processed/ and try again.")
