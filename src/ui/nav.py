"""Navigation state shared by screens: section labels, deep links and the selected user."""

import streamlit as st

VIEWS = {"overview": "Overview", "engines": "Recommendation Engines", "run": "Run Engine",
         "profile": "User Profile", "kpi": "KPI Benchmarks"}
LABEL_TO_VIEW = {label: key for key, label in VIEWS.items()}


def selected_user() -> int | None:
    """Selected user from session state, falling back to the ?user= deep link."""
    value = st.session_state.get("profile_user", st.query_params.get("user"))
    try:
        return int(value) if value is not None and int(value) > 0 else None
    except (TypeError, ValueError):
        return None


def set_user(user_id: int | None) -> None:
    if user_id:
        st.session_state["profile_user"] = int(user_id)
        st.query_params["user"] = str(int(user_id))
    else:
        st.session_state.pop("profile_user", None)
        st.query_params.pop("user", None)


def open_profile(user_id: int) -> None:
    """Button callback: jump to a user's full profile."""
    set_user(user_id)
    st.session_state["nav"] = VIEWS["profile"]
    st.query_params["view"] = "profile"


def sync_view() -> None:
    label = st.session_state.get("nav")
    if label in LABEL_TO_VIEW:
        st.query_params["view"] = LABEL_TO_VIEW[label]
