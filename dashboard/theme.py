"""Visual design system: CSS and small HTML building blocks.

Pure presentation: no figure is computed here.  Every helper receives values
already formatted (or ``None``) by the caller, so a missing figure is rendered
as missing - never replaced by a placeholder number.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, List, Optional, Sequence, Tuple

ACCENT = "#7d2ae8"
CYAN = "#00c4cc"
GREEN = "#0f9d58"
RED = "#d93654"
AMBER = "#b26a00"
MUTED = "#5b6475"

CSS = """
<style>
:root {
  --erai-bg: #f6f7fb;
  --erai-card: #ffffff;
  --erai-border: #eceef5;
  --erai-border-strong: #e3e6ef;
  --erai-text: #0e1318;
  --erai-muted: #5b6475;
  --erai-accent: #7d2ae8;
  --erai-cyan: #00c4cc;
  --erai-accent-soft: #f1e9fd;
  --erai-hover: #f4f5f9;
  --erai-green: #0f9d58;
  --erai-red: #d93654;
  --erai-amber: #b26a00;
  --erai-radius: 18px;
  --erai-shadow: 0 1px 2px rgba(16, 24, 40, .04), 0 4px 16px rgba(16, 24, 40, .06);
  --erai-shadow-hover: 0 2px 4px rgba(16, 24, 40, .05), 0 12px 28px rgba(16, 24, 40, .10);
  --erai-gradient: linear-gradient(135deg, #7d2ae8, #00c4cc);
  --erai-mono: "Plus Jakarta Sans", Inter, "Segoe UI", sans-serif;
}

/* ---------- canvas ---------- */
.stApp, .stApp [data-testid="stAppViewContainer"] { background: var(--erai-bg) !important; }
header[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: none !important;
}
.block-container { padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1480px; }
.block-container > div { animation: erai-fade-up .4s ease both; }
@keyframes erai-fade-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea {
  font-family: "Plus Jakarta Sans", Inter, "Segoe UI", sans-serif;
}
h1, h2, h3, h4 { color: var(--erai-text); letter-spacing: -0.02em; font-weight: 800; }
h3 { margin-top: .6rem; }
hr { border-color: var(--erai-border) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #d8dce8; border-radius: 10px; border: 2px solid var(--erai-bg); }
::-webkit-scrollbar-thumb:hover { background: #c2c8d8; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid #e8eaf2; }
section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }
.erai-brand { display: flex; align-items: center; gap: 10px; margin: 0 0 6px; }
.erai-brand .logo {
  width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center;
  background: linear-gradient(135deg, #7d2ae8, #00c4cc);
  color: #fff; font-weight: 800; font-size: 14px; letter-spacing: .02em;
  box-shadow: 0 4px 12px rgba(125, 42, 232, .25);
}
.erai-brand .name { font-weight: 800; font-size: 15.5px; line-height: 1.1; color: var(--erai-text); }
.erai-brand .tag { font-size: 11.5px; color: var(--erai-muted); }
.erai-company-card {
  border: 1px solid var(--erai-border); border-radius: 16px; padding: 12px 14px; margin: 6px 0 4px;
  background: linear-gradient(160deg, rgba(125,42,232,.08), rgba(0,196,204,.06)), #ffffff;
  box-shadow: var(--erai-shadow);
}
.erai-company-card .t { font-weight: 700; font-size: 13.5px; color: var(--erai-text); }
.erai-company-card .s { font-size: 11.5px; color: var(--erai-muted); margin: 2px 0 6px; }
.erai-side-label {
  font-size: 10.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: #8a92a3; margin: 14px 0 6px;
}

/* Navigation: the page radio becomes a vertical menu (Streamlit >= 1.60 markup) */
.st-key-nav_page, .st-key-nav_page [data-testid="stRadio"],
.st-key-nav_page [role="radiogroup"], .st-key-nav_page [role="radiogroup"] > div { width: 100% !important; }
.st-key-nav_page [role="radiogroup"] { gap: 2px; }
.st-key-nav_page label[data-testid="stRadioOption"] {
  width: 100%; padding: 7px 10px; border-radius: 12px; margin: 0; min-height: 0;
  border: 1px solid transparent; cursor: pointer;
  transition: background .15s ease, color .15s ease;
}
.st-key-nav_page label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-nav_page label[data-testid="stRadioOption"] p { font-size: 13.5px; font-weight: 500; color: #3b4352; }
.st-key-nav_page label[data-testid="stRadioOption"] p span[role="img"] { margin-right: 6px; color: #8a92a3; }
.st-key-nav_page label[data-testid="stRadioOption"]:hover { background: #f4f5f9; }
.st-key-nav_page label[data-testid="stRadioOption"][data-selected="true"] {
  background: #f1e9fd; border-color: transparent;
}
.st-key-nav_page label[data-testid="stRadioOption"][data-selected="true"] p { font-weight: 700; color: #7d2ae8; }
.st-key-nav_page label[data-testid="stRadioOption"][data-selected="true"] p span[role="img"] { color: #7d2ae8; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, .stLinkButton > a {
  font-weight: 600; border-radius: 10px; border: 1px solid var(--erai-border-strong);
  background: #ffffff; color: var(--erai-text);
  transition: transform .12s ease, box-shadow .2s ease, border-color .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover, .stLinkButton > a:hover {
  transform: translateY(-1px); border-color: #c9cee0; box-shadow: var(--erai-shadow-hover);
  color: var(--erai-text);
}
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #7d2ae8, #00c4cc) !important;
  border: none !important; color: #fff !important; border-radius: 999px !important;
  box-shadow: 0 4px 14px rgba(125, 42, 232, .28);
}
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  box-shadow: 0 8px 20px rgba(125, 42, 232, .34); color: #fff !important;
}
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff !important; }

/* inputs stay visible on white cards */
div[data-baseweb="input"] > div, div[data-baseweb="base-input"], div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea, div[data-testid="stTextArea"] textarea {
  background: #ffffff !important; border-color: #d9dde9 !important; border-radius: 10px !important;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] div, div[data-testid="stNumberInput"] input {
  color: var(--erai-text) !important;
}

div[role="tablist"] {
  gap: 4px; background: #eef0f6; padding: 4px; border-radius: 999px; border-bottom: none !important;
  width: fit-content; max-width: 100%; overflow-x: auto; margin-bottom: 12px;
}
div[role="tablist"] [data-testid="stTab"] {
  border-radius: 999px; padding: 6px 16px; height: auto; margin: 0; transition: background .15s ease, color .15s ease;
}
div[role="tablist"] [data-testid="stTab"] p { font-size: 13px; font-weight: 600; color: var(--erai-muted); }
div[role="tablist"] [data-testid="stTab"]:hover { background: rgba(255,255,255,.6); }
div[role="tablist"] [data-testid="stTab"][data-selected="true"] {
  background: #ffffff; box-shadow: 0 1px 2px rgba(16,24,40,.06), 0 2px 8px rgba(16,24,40,.08);
}
div[role="tablist"] [data-testid="stTab"][data-selected="true"] p { color: var(--erai-text); font-weight: 700; }
div[role="tablist"] .react-aria-SelectionIndicator { display: none; }
div[data-testid="stExpander"] details {
  border-radius: var(--erai-radius); border: 1px solid var(--erai-border); background: var(--erai-card);
  box-shadow: var(--erai-shadow);
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border-radius: 14px; overflow: hidden; border: 1px solid var(--erai-border); background: var(--erai-card);
}
div[data-testid="stPlotlyChart"] {
  background: var(--erai-card); border: 1px solid var(--erai-border);
  border-radius: var(--erai-radius); padding: 10px 10px 4px; box-shadow: var(--erai-shadow);
  transition: transform .18s ease, box-shadow .18s ease;
}
div[data-testid="stPlotlyChart"]:hover { transform: translateY(-2px); box-shadow: var(--erai-shadow-hover); }
div[data-testid="stAlert"] { border-radius: 14px; }
div[data-testid="stMetric"] {
  background: var(--erai-card); border: 1px solid var(--erai-border); border-radius: 16px; padding: 12px 14px;
  box-shadow: var(--erai-shadow);
}
div[data-testid="stChatMessage"] {
  border-radius: 16px; border: 1px solid var(--erai-border); background: var(--erai-card); box-shadow: var(--erai-shadow);
}

/* ---------- hero header ---------- */
.erai-hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 24px 28px 22px; margin: 0 0 18px;
  border: 1px solid var(--erai-border); background: var(--erai-card); box-shadow: var(--erai-shadow);
}
.erai-hero::before {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--erai-gradient);
}
.erai-eyebrow {
  font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: var(--erai-accent);
}
.erai-hero-title { font-size: 30px; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 8px; line-height: 1.15; color: var(--erai-text); }
.erai-hero-title .tk {
  font-size: 13px; font-weight: 700; vertical-align: middle; margin-left: 10px;
  padding: 3px 10px; border-radius: 999px; background: var(--erai-accent-soft); color: var(--erai-accent);
  font-variant-numeric: tabular-nums;
}
.erai-hero-sub { color: var(--erai-muted); font-size: 14px; margin-bottom: 10px; max-width: 900px; }

/* ---------- KPI cards ---------- */
.erai-kpi {
  position: relative; height: 100%; min-height: 138px; border-radius: var(--erai-radius); padding: 16px 18px 14px;
  margin-bottom: 14px;
  display: flex; flex-direction: column; justify-content: space-between;
  border: 1px solid var(--erai-border); background: var(--erai-card); box-shadow: var(--erai-shadow);
  transition: transform .18s ease, box-shadow .18s ease;
}
.erai-kpi:hover { transform: translateY(-2px); box-shadow: var(--erai-shadow-hover); }
.erai-kpi .label { font-size: 11.5px; font-weight: 600; color: var(--erai-muted); }
.erai-kpi .value {
  font-size: 24px; font-weight: 800; margin-top: 6px; letter-spacing: -0.02em; color: var(--erai-text);
  font-variant-numeric: tabular-nums;
}
.erai-kpi .sub { font-size: 11.5px; color: var(--erai-muted); margin-top: 4px; }
.erai-kpi .row { display: flex; align-items: flex-end; justify-content: space-between; gap: 8px; }
.erai-kpi svg { display: block; }
.erai-delta {
  display: inline-flex; align-items: center; gap: 3px; font-size: 11.5px; font-weight: 700;
  padding: 2px 8px; border-radius: 999px; margin-top: 6px; font-variant-numeric: tabular-nums;
}
.erai-delta.up { color: var(--erai-green); background: #e7f8ef; }
.erai-delta.down { color: var(--erai-red); background: #fdecee; }
.erai-delta.flat { color: var(--erai-muted); background: #f0f2f7; }

/* ---------- pills, notes ---------- */
.erai-pill {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px; font-size: 11.5px;
  font-weight: 600; color: var(--erai-muted); background: #f0f2f7; margin: 0 6px 6px 0;
}
.erai-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.erai-pill.ok { color: var(--erai-green); background: #e7f8ef; }
.erai-pill.warn { color: var(--erai-amber); background: #fff4e0; }
.erai-pill.bad { color: var(--erai-red); background: #fdecee; }
.erai-pill.info { color: var(--erai-accent); background: #f1e9fd; }
.erai-prov { font-size: 11.5px; color: var(--erai-muted); margin: 6px 4px 10px; }
.erai-section {
  border-left: 3px solid var(--erai-accent); padding-left: 10px; margin: 6px 0 14px; color: var(--erai-muted);
}
.erai-quote, .erai-warn, .erai-danger {
  padding: 12px 16px; border-radius: 14px; font-size: 13px; margin: 6px 0 12px; color: var(--erai-text);
}
.erai-quote { background: #f1e9fd; border-left: 3px solid var(--erai-accent); }
.erai-warn { background: #fff4e0; border-left: 3px solid #ffb020; }
.erai-danger { background: #fdecee; border-left: 3px solid var(--erai-red); }
.erai-tool { font-size: 12px; color: var(--erai-muted); font-family: var(--erai-mono); font-variant-numeric: tabular-nums; }

/* ---------- section titles & steps ---------- */
.erai-h { display: flex; align-items: center; gap: 10px; margin: 20px 0 10px; }
.erai-h .bar { width: 4px; height: 18px; border-radius: 4px; background: var(--erai-gradient); }
.erai-h .t { font-size: 16.5px; font-weight: 800; color: var(--erai-text); }
.erai-h .s { font-size: 12px; color: var(--erai-muted); margin-left: 4px; }
.erai-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 8px 0 14px; }
.erai-step {
  border: 1px solid var(--erai-border); border-radius: var(--erai-radius); padding: 16px 18px;
  background: var(--erai-card); box-shadow: var(--erai-shadow);
  transition: transform .18s ease, box-shadow .18s ease;
}
.erai-step:hover { transform: translateY(-2px); box-shadow: var(--erai-shadow-hover); }
.erai-step .n {
  width: 28px; height: 28px; border-radius: 10px; display: grid; place-items: center; font-weight: 800; font-size: 12.5px;
  background: var(--erai-gradient); color: #fff; margin-bottom: 10px;
}
.erai-step .t { font-weight: 700; font-size: 13.5px; margin-bottom: 4px; color: var(--erai-text); }
.erai-step .d { font-size: 12.5px; color: var(--erai-muted); }
.erai-footer {
  margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--erai-border);
  font-size: 11.5px; color: var(--erai-muted);
}

@media (max-width: 900px) {
  .erai-hero-title { font-size: 23px; }
  .erai-kpi .value { font-size: 19px; }
}
</style>
"""


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------


def pill(text: str, kind: str = "") -> str:
    return f'<span class="erai-pill {escape(kind)}">{escape(str(text))}</span>'


def brand(tagline: str) -> str:
    return (
        '<div class="erai-brand"><div class="logo">ER</div><div>'
        '<div class="name">Equity Research AI</div>'
        f'<div class="tag">{escape(tagline)}</div></div></div>'
    )


def company_card(name: str, subtitle: str, pills_html: str = "") -> str:
    return (
        f'<div class="erai-company-card"><div class="t">{escape(name)}</div>'
        f'<div class="s">{escape(subtitle)}</div>{pills_html}</div>'
    )


def side_label(text: str) -> str:
    return f'<div class="erai-side-label">{escape(text)}</div>'


def hero(eyebrow: str, title: str, ticker: Optional[str] = None,
         subtitle: Optional[str] = None, pills_html: str = "") -> str:
    ticker_html = f'<span class="tk">{escape(ticker)}</span>' if ticker else ""
    subtitle_html = f'<div class="erai-hero-sub">{escape(subtitle)}</div>' if subtitle else ""
    return (
        f'<div class="erai-hero"><div class="erai-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="erai-hero-title">{escape(title)}{ticker_html}</div>'
        f"{subtitle_html}<div>{pills_html}</div></div>"
    )


def section_title(title: str, subtitle: Optional[str] = None) -> str:
    sub = f'<span class="s">{escape(subtitle)}</span>' if subtitle else ""
    return f'<div class="erai-h"><span class="bar"></span><span class="t">{escape(title)}</span>{sub}</div>'


def steps(items: Sequence[Tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="erai-step"><div class="n">{i}</div><div class="t">{escape(t)}</div>'
        f'<div class="d">{escape(d)}</div></div>'
        for i, (t, d) in enumerate(items, start=1)
    )
    return f'<div class="erai-steps">{cards}</div>'


def footer(text: str) -> str:
    return f'<div class="erai-footer">{escape(text)}</div>'


def delta_badge(delta: Optional[float], invert: bool = False) -> str:
    """YoY change badge from a fraction (0.12 = +12 %).  ``None`` renders nothing."""
    if delta is None:
        return ""
    if abs(delta) < 0.0005:
        kind, arrow = "flat", "▬"
    else:
        positive = delta > 0
        good = positive != invert
        kind = "up" if good else "down"
        arrow = "▲" if positive else "▼"
    return f'<span class="erai-delta {kind}">{arrow} {delta * 100:+.1f} %</span>'


def sparkline(values: Iterable[Optional[float]], width: int = 96, height: int = 30,
              color: str = ACCENT) -> str:
    """Tiny inline SVG trend.  Missing points are skipped, not interpolated as zero."""
    points: List[float] = [float(v) for v in values if v is not None]
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    step = width / (len(points) - 1)
    coords = [
        (round(i * step, 1), round(height - 3 - (v - lo) / span * (height - 6), 1))
        for i, v in enumerate(points)
    ]
    line = " ".join(f"{x},{y}" for x, y in coords)
    area = f"0,{height} " + line + f" {width},{height}"
    gid = f"g{abs(hash((line, color))) % 10**8}"
    last_x, last_y = coords[-1]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.22"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.6" fill="{color}"/></svg>'
    )


def kpi_card(label: str, value: str, sub: str = "", delta: Optional[float] = None,
             spark: Optional[Iterable[Optional[float]]] = None, invert_delta: bool = False,
             color: str = ACCENT) -> str:
    spark_svg = sparkline(spark, color=color) if spark is not None else ""
    sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="erai-kpi"><div class="label">{escape(label)}</div>'
        f'<div class="row"><div><div class="value">{escape(value)}</div>'
        f"{delta_badge(delta, invert_delta)}</div>{spark_svg}</div>{sub_html}</div>"
    )
