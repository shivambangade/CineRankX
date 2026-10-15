"""CineRankX visual system: porcelain canvas, orange/cobalt accents, hairline cards."""

import html
from contextlib import contextmanager

import streamlit as st

from src.ui.icons import icon

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root { --canvas:#f6f8fb; --line:#e3e8ef; --ink:#0f172a; --mute:#64748b; --soft:#eef2f7;
        --orange:#f97316; --orange-d:#ea580c; --cobalt:#2563eb; --teal:#0d9488; --rose:#e11d48; }
html, body, .stApp { font-family: Inter, system-ui, sans-serif; }
.stApp { background: var(--canvas); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1320px; padding: 1.25rem 16px 3rem; }
@media (min-width: 900px) { .block-container { padding: 1.5rem 40px 4rem; } }
h1, h2, h3, h4, .brand-name, .kpi-value { font-family: 'Plus Jakarta Sans', Inter, sans-serif; letter-spacing: -.01em; }
h2, h3 { color: var(--ink); } h4 { margin: 1.4rem 0 .6rem; font-size: 1.05rem; }
.fi { vertical-align: -4px; }

/* top bar + navigation */
.topbar { display:flex; align-items:center; gap:12px; }
.logo { background:linear-gradient(135deg,var(--orange),var(--orange-d)); color:#fff; border-radius:12px;
        padding:8px; display:inline-flex; box-shadow:0 4px 12px rgba(249,115,22,.35); }
.brand-name { font-weight:800; font-size:1.3rem; line-height:1.1; }
.brand-sub { color:var(--mute); font-size:.82rem; }
.st-key-nav { margin: 6px 0 18px; }
.st-key-nav div[data-testid="stSegmentedControl"] { background:#fff; border:1px solid var(--line); border-radius:14px;
        padding:4px; overflow-x:auto; max-width:100%; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.st-key-nav button { border:0 !important; border-radius:10px !important; font-weight:600; background:transparent !important; color:var(--mute); padding:8px 16px; }
.st-key-nav button[aria-checked="true"], .st-key-nav button[aria-pressed="true"], .st-key-nav button[kind="segmented_controlActive"] {
        background:var(--ink) !important; color:#fff !important; }

/* cards */
[class*="st-key-card"] { background:#fff; border:1px solid var(--line) !important; border-radius:20px !important;
        padding:20px 22px !important; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px rgba(15,23,42,.04); }
.card-title { display:flex; align-items:center; gap:10px; font-family:'Plus Jakarta Sans',Inter,sans-serif;
              font-weight:700; font-size:1.05rem; margin-bottom:2px; }
.card-title .chip { background:var(--soft); color:var(--cobalt); border-radius:10px; padding:6px; display:inline-flex; }
.card-sub { color:var(--mute); font-size:.84rem; margin-bottom:14px; }
.eyebrow { color:var(--mute); font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:18px; padding:16px 18px;
       box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px rgba(15,23,42,.04); height:100%; }
.kpi-value { font-weight:800; font-size:1.9rem; font-variant-numeric:tabular-nums; margin-top:4px; }
.kpi-unit { color:var(--mute); font-size:.85rem; font-weight:500; margin-left:4px; }
.kpi-note { color:var(--mute); font-size:.8rem; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:600;
         background:#e8efff; color:var(--cobalt); white-space:nowrap; }
.badge.ok { background:#ccfbf1; color:#0f766e; } .badge.warn { background:#ffe4e6; color:#be123c; }
.badge.hot { background:#ffedd5; color:var(--orange-d); }
.bar { height:6px; background:var(--soft); border-radius:99px; overflow:hidden; margin:5px 0 12px; }
.bar > span { display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,#06b6d4,var(--cobalt)); }
.bar.o > span { background:linear-gradient(90deg,#fb923c,var(--orange-d)); }
.bar.t > span { background:linear-gradient(90deg,#2dd4bf,var(--teal)); }
.row { display:flex; justify-content:space-between; gap:10px; font-size:.88rem; font-variant-numeric:tabular-nums; }
.funnel-step { margin-bottom:14px; }
.funnel-bar { height:34px; border-radius:10px; display:flex; align-items:center; padding:0 12px; color:#fff;
              font-weight:600; font-size:.85rem; min-width:64px; white-space:nowrap; }

/* weights */
.sum-pill { display:flex; justify-content:space-between; align-items:center; gap:10px; background:var(--soft);
            border-radius:12px; padding:10px 14px; margin-top:6px; font-size:.88rem; }

/* results */
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr)); gap:14px; }
.movie { background:#fff; border:1px solid var(--line); border-radius:18px; padding:16px;
         box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 22px rgba(15,23,42,.04); transition:box-shadow .15s, transform .15s; }
.movie:hover { box-shadow:0 12px 30px rgba(15,23,42,.1); transform:translateY(-1px); }
.movie.top { border-color:#fdba74; background:linear-gradient(180deg,#fff7ed,#fff 60%); }
.movie-head { display:flex; gap:12px; align-items:flex-start; }
.movie-rank { background:var(--soft); font-weight:800; border-radius:12px; min-width:44px; height:44px;
              display:flex; align-items:center; justify-content:center; font-variant-numeric:tabular-nums; }
.movie.top .movie-rank { background:var(--orange); color:#fff; }
.movie-title { font-weight:700; font-size:1rem; line-height:1.3; }
.movie-meta { color:var(--mute); font-size:.8rem; margin-top:2px; }
.score { margin-left:auto; text-align:right; }
.score b { display:block; color:var(--orange-d); font-size:1.35rem; font-weight:800; font-variant-numeric:tabular-nums; }
.reason { background:var(--soft); border-radius:10px; padding:8px 10px; font-size:.82rem; margin:12px 0; color:#334155; }
.mini { display:grid; grid-template-columns:repeat(2,1fr); gap:2px 18px; }
.mini .row { font-size:.76rem; color:var(--mute); }
.mini > div { min-width:0; }
.st-key-reset_weights { margin-top:12px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip2 { background:#e8efff; color:var(--cobalt); border-radius:999px; padding:1px 9px; font-size:.7rem; font-weight:600; }

/* native widgets */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius:12px; font-weight:600; min-height:44px; }
.stButton > button[kind="primary"] { background:var(--orange); border-color:var(--orange); color:#fff; box-shadow:0 6px 16px rgba(249,115,22,.3); }
.stButton > button[kind="primary"]:hover { background:var(--orange-d); border-color:var(--orange-d); }
.stButton > button[kind="primary"]:disabled { background:#fed7aa; border-color:#fed7aa; box-shadow:none; color:#fff; }
div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div { border-radius:12px; }
div[data-testid="stSlider"] label p { font-size:.85rem; font-weight:600; }
div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:14px 18px; }
button:focus-visible, input:focus-visible, [role="slider"]:focus-visible, a:focus-visible {
  outline:2px solid var(--cobalt) !important; outline-offset:2px; }
@media (max-width: 640px) { .kpi-value { font-size:1.45rem; } [class*="st-key-card"] { padding:16px !important; border-radius:16px !important; }
  .mini { grid-template-columns:1fr; } }
@media (prefers-reduced-motion: reduce) { .movie { transition:none; } }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(
        f'<div class="topbar"><span class="logo">{icon("film", 22)}</span><div>'
        '<div class="brand-name">CineRankX</div>'
        '<div class="brand-sub">Adaptive, multi-objective movie recommendation</div></div></div>',
        unsafe_allow_html=True,
    )


def card_title(title: str, icon_name: str, sub: str = "") -> None:
    sub_html = f'<div class="card-sub">{html.escape(sub)}</div>' if sub else '<div class="card-sub"></div>'
    st.markdown(f'<div class="card-title"><span class="chip">{icon(icon_name, 16)}</span>'
                f'{html.escape(title)}</div>{sub_html}', unsafe_allow_html=True)


@contextmanager
def card(key: str, title: str, icon_name: str, sub: str = ""):
    """Bordered card that can hold native widgets as well as HTML."""
    with st.container(border=True, key=f"card_{key}"):
        card_title(title, icon_name, sub)
        yield


def card_html(title: str, icon_name: str, body: str, sub: str = "") -> str:
    """Static card as one HTML string (for grids that contain no widgets)."""
    sub_html = f'<div class="card-sub">{html.escape(sub)}</div>' if sub else '<div class="card-sub"></div>'
    return (f'<div class="kpi"><div class="card-title"><span class="chip">{icon(icon_name, 16)}</span>'
            f'{html.escape(title)}</div>{sub_html}{body}</div>')


def kpi(label: str, value: str, unit: str = "", note: str = "") -> str:
    unit_html = f'<span class="kpi-unit">{html.escape(unit)}</span>' if unit else ""
    note_html = f'<div class="kpi-note">{html.escape(note)}</div>' if note else ""
    return (f'<div class="kpi"><div class="eyebrow">{html.escape(label)}</div>'
            f'<div class="kpi-value" style="{"font-size:1.25rem;white-space:nowrap" if len(value) > 9 else ""}">'
            f'{html.escape(value)}{unit_html}</div>{note_html}</div>')


def kpi_row(items: list[tuple]) -> None:
    for col, item in zip(st.columns(len(items)), items):
        col.markdown(kpi(*item), unsafe_allow_html=True)


def meter(label: str, value: float, accent: str = "", fmt: str = "{:.2f}") -> str:
    """HTML progress row for a value in [0, 1]."""
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="row"><span>{html.escape(label)}</span><b>{fmt.format(value)}</b></div>'
        f'<div class="bar {accent}"><span style="width:{pct:.0f}%"></span></div>'
    )
