"""Accendio brand theme — global CSS and shared UI components."""

import copy
import datetime
import os

import streamlit as st

# ── Color constants ───────────────────────────────────────────────────────────
VOID    = "#060912"
ABYSS   = "#09102A"
DEPTH   = "#0C1228"
SIGNAL  = "#7B9CFF"
COAL    = "#1A2A5E"
ICE     = "#EEF2FF"
ICE_MID = "rgba(238,242,255,0.55)"
ICE_LOW = "rgba(238,242,255,0.28)"
ASCEND  = "#3DB87A"
DESCEND = "#D94F4F"
AMBER   = "#F59E0B"
BORDER  = "rgba(123,156,255,0.14)"

# Plotly layout defaults — apply with `fig.update_layout(**PLOTLY_LAYOUT)`
PLOTLY_LAYOUT = dict(
    paper_bgcolor=DEPTH,
    plot_bgcolor=ABYSS,
    font=dict(color=ICE, size=11, family="Arial, Helvetica Neue, sans-serif"),
    title=dict(text="", font=dict(color=ICE, size=13)),
    xaxis=dict(
        gridcolor="rgba(123,156,255,0.08)",
        zerolinecolor="rgba(123,156,255,0.15)",
        linecolor="rgba(123,156,255,0.1)",
        tickfont=dict(color=ICE_MID, size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(123,156,255,0.08)",
        zerolinecolor="rgba(123,156,255,0.15)",
        linecolor="rgba(123,156,255,0.1)",
        tickfont=dict(color=ICE_MID, size=10),
    ),
    legend=dict(
        bgcolor="rgba(9,16,42,0.8)",
        bordercolor="rgba(123,156,255,0.2)",
        borderwidth=0.5,
        font=dict(color=ICE_MID, size=10),
    ),
    hoverlabel=dict(
        bgcolor=ABYSS,
        bordercolor="rgba(123,156,255,0.3)",
        font=dict(color=ICE, size=11),
    ),
)

# ── Depth Zones (spec §A) — approved mockup values; minor tuning in visual QA OK
ZONES = {
    "data":    dict(label="MARKETS & DATA",     accent="#5A8CFF", bg_top="#060B1A",
                    bg_bot="#0A1430", panel="#0A1430", border="rgba(90,140,255,0.22)",
                    glow="rgba(90,140,255,0.5)"),
    "signals": dict(label="SIGNALS & RESEARCH", accent="#A78BFF", bg_top="#0A0920",
                    bg_bot="#141238", panel="#12102E", border="rgba(167,139,255,0.25)",
                    glow="rgba(167,139,255,0.5)"),
    "risk":    dict(label="PORTFOLIO & RISK",   accent="#F5A65B", bg_top="#140D08",
                    bg_bot="#2A1A0E", panel="#1E130A", border="rgba(245,166,91,0.25)",
                    glow="rgba(245,166,91,0.5)"),
    "macro":   dict(label="MACRO CONTEXT",      accent="#4EC9A8", bg_top="#061410",
                    bg_bot="#0D2A20", panel="#0B241B", border="rgba(78,201,168,0.25)",
                    glow="rgba(78,201,168,0.5)"),
}


def _ecosystem_on() -> bool:
    return os.getenv("ECOSYSTEM_UI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def zone_plotly_layout(zone: str | None = None) -> dict:
    """Per-zone PLOTLY_LAYOUT variant. Falls back to the brand default."""
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    if zone in ZONES and _ecosystem_on():
        z = ZONES[zone]
        layout["paper_bgcolor"] = z["panel"]
        layout["plot_bgcolor"] = z["bg_top"]
        layout["legend"]["bordercolor"] = z["border"]
        layout["hoverlabel"]["bordercolor"] = z["border"]
    return layout


_CSS = """
<style>
/* ── Accendio Brand Theme ────────────────────────────────────────────────── */
:root {
  --ac-void:    #060912;
  --ac-abyss:   #09102A;
  --ac-depth:   #0C1228;
  --ac-signal:  #7B9CFF;
  --ac-coal:    #1A2A5E;
  --ac-ice:     #EEF2FF;
  --ac-ascend:  #3DB87A;
  --ac-descend: #D94F4F;
  --ac-amber:   #F59E0B;
  --ac-border:  rgba(123,156,255,0.14);
}

/* Page & app background */
[data-testid="stAppViewContainer"], .stApp {
  background-color: var(--ac-void) !important;
}
.main .block-container {
  background-color: var(--ac-void);
  padding-top: 56px !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  max-width: 100% !important;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Sidebar background */
[data-testid="stSidebar"] {
  background-color: var(--ac-abyss) !important;
  border-right: 0.5px solid rgba(123,156,255,0.15) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
  color: rgba(238,242,255,0.65) !important;
}

/* Sidebar auto-nav links */
[data-testid="stSidebarNavLink"] {
  color: rgba(238,242,255,0.5) !important;
  border-radius: 6px !important;
  padding: 5px 10px !important;
  margin: 1px 4px !important;
  font-size: 13px !important;
}
[data-testid="stSidebarNavLink"]:hover {
  background: rgba(123,156,255,0.08) !important;
  color: var(--ac-ice) !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavLink"][aria-selected="true"] {
  background: rgba(123,156,255,0.12) !important;
  color: var(--ac-signal) !important;
  border-left: 2px solid var(--ac-signal) !important;
  padding-left: 8px !important;
}

/* Typography */
h1 {
  color: var(--ac-ice) !important;
  font-weight: 300 !important;
  font-size: 1.4rem !important;
  letter-spacing: 0.02em;
  margin-bottom: 0.2rem !important;
}
h2 {
  color: rgba(238,242,255,0.9) !important;
  font-weight: 300 !important;
  font-size: 1.1rem !important;
}
h3 {
  color: rgba(238,242,255,0.75) !important;
  font-weight: 400 !important;
  font-size: 0.9rem !important;
}
h4, h5, h6 {
  color: rgba(238,242,255,0.70) !important;
  font-weight: 400 !important;
  font-size: 0.85rem !important;
}
p, span, label, .stMarkdown p {
  color: rgba(238,242,255,0.75) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--ac-depth) !important;
  border: 0.5px solid var(--ac-border) !important;
  border-radius: 8px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] > div {
  color: rgba(238,242,255,0.4) !important;
  font-size: 10px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
  color: var(--ac-ice) !important;
  font-size: 1.3rem !important;
  font-weight: 300 !important;
}

/* Buttons */
.stButton > button {
  background: rgba(123,156,255,0.07) !important;
  color: var(--ac-signal) !important;
  border: 0.5px solid rgba(123,156,255,0.28) !important;
  border-radius: 6px !important;
  font-size: 12px !important;
  letter-spacing: 0.05em !important;
  font-weight: 400 !important;
}
.stButton > button:hover {
  background: rgba(123,156,255,0.14) !important;
  border-color: rgba(123,156,255,0.5) !important;
  color: var(--ac-ice) !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 0.5px solid rgba(123,156,255,0.18) !important;
  gap: 2px !important;
  overflow-x: auto !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(238,242,255,0.62) !important;
  border-bottom: 2px solid transparent !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 8px 16px !important;
  font-weight: 400 !important;
  white-space: nowrap !important;
}
[data-baseweb="tab"]:hover {
  color: rgba(238,242,255,0.88) !important;
  background: rgba(123,156,255,0.06) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--ac-signal) !important;
  border-bottom: 2px solid var(--ac-signal) !important;
}
[data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 1.25rem !important;
}

/* Selectboxes */
[data-testid="stSelectbox"] > div > div {
  background: var(--ac-depth) !important;
  border: 0.5px solid rgba(123,156,255,0.25) !important;
  color: var(--ac-ice) !important;
  border-radius: 6px !important;
}

/* Expander */
details {
  background: var(--ac-depth) !important;
  border: 0.5px solid var(--ac-border) !important;
  border-radius: 8px !important;
}
summary { color: rgba(238,242,255,0.6) !important; font-size: 13px !important; }

/* Horizontal rules */
hr { border-color: rgba(123,156,255,0.1) !important; margin: 0.6rem 0 !important; }

/* Caption / small text */
[data-testid="stCaptionContainer"],
.stCaption, small {
  color: rgba(238,242,255,0.3) !important;
  font-size: 11px !important;
  letter-spacing: 0.03em;
}

/* DataFrames */
[data-testid="stDataFrame"] {
  border: 0.5px solid rgba(123,156,255,0.1) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

/* Alert banners */
[data-testid="stAlert"] {
  border-radius: 6px !important;
  font-size: 13px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li {
  color: rgba(238,242,255,0.85) !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
  border-top-color: var(--ac-signal) !important;
}

/* Busy indicator — cycling emojis, top-right, auto-shown when any spinner is active */
@keyframes _ac_busy_cycle {
  0%,   20% { opacity: 1;  transform: scale(1.12) translateY(-1px); }
  28%, 100% { opacity: 0;  transform: scale(0.82) translateY(0px);  }
}
#_ac_busy_bar {
  position: fixed;
  top: 0;
  right: 16px;
  height: 44px;
  width: 24px;
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 10000000;
  pointer-events: none;
}
body:has([data-testid="stSpinner"]) #_ac_busy_bar {
  display: flex !important;
}
._ac_busy_e {
  font-size: 15px;
  line-height: 1;
  opacity: 0;
  position: absolute;
  animation: _ac_busy_cycle 3.2s ease-in-out infinite;
}
._ac_busy_e:nth-child(1) { animation-delay: 0.0s; }
._ac_busy_e:nth-child(2) { animation-delay: 0.8s; }
._ac_busy_e:nth-child(3) { animation-delay: 1.6s; }
._ac_busy_e:nth-child(4) { animation-delay: 2.4s; }

/* Multiselect / radio / checkbox */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
  color: rgba(238,242,255,0.65) !important;
}
</style>
"""


def theme_css(zone: str | None = None) -> str:
    """Full CSS for a page. zone=None (or flag off) → legacy CSS, unchanged."""
    if zone not in ZONES or not _ecosystem_on():
        return _CSS
    z = ZONES[zone]
    return _CSS + f"""<style>
/* ── Depth Zone override: {zone} ── (topbar + sidebar deliberately untouched) */
[data-testid="stAppViewContainer"], .stApp {{
  background: linear-gradient(180deg, {z['bg_top']} 0%, {z['bg_bot']} 140%) !important;
}}
[data-testid="stMetric"], details {{
  background: {z['panel']} !important;
  border-color: {z['border']} !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  color: {z['accent']} !important;
  border-bottom-color: {z['accent']} !important;
}}
:root {{ --zone-accent: {z['accent']}; --zone-border: {z['border']}; }}
</style>"""


def apply_theme(zone: str | None = None):
    """Inject the Accendio CSS theme (+ optional Depth Zone override).

    Call once per page after set_page_config. zone=None or flag off → legacy.
    """
    st.markdown(theme_css(zone), unsafe_allow_html=True)


def _macro_trigger_pill_html() -> str:
    """
    Build the MACRO topbar pill: "MACRO N/M" where N is the number of active
    triggers (5-day lookback) and M is the count of trigger-aware modules in
    this codebase. Hidden when MACRO_TRIGGERS_ENABLED is false.

    Lives next to SIG in the topbar so the user always has a visible signal
    that the trigger ecosystem is wired up.
    """
    import os
    enabled = os.getenv("MACRO_TRIGGERS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return (
            '<div style="margin-right:18px;flex-shrink:0;display:flex;align-items:center;gap:6px">'
            '<span style="font-size:10px;color:rgba(238,242,255,0.28);letter-spacing:.1em">MACRO</span>'
            f'<span style="font-size:11px;color:{DESCEND};font-weight:500">OFF</span>'
            '</div>'
        )

    # Count of dashboard modules that read from features.macro_features.
    # Update this if you wire a new module into the trigger surface.
    n_modules = 6   # cascade_orchestrator, macro_router, sector_model, meta_predictor, portfolio_optimizer, ripple

    try:
        from utils.theme import _cached_active_trigger_count
        n_active = _cached_active_trigger_count()
    except Exception:
        n_active = 0

    color = ASCEND if n_active > 0 else "rgba(238,242,255,0.4)"
    return (
        '<div style="margin-right:18px;flex-shrink:0;display:flex;align-items:center;gap:6px">'
        '<span style="font-size:10px;color:rgba(238,242,255,0.28);letter-spacing:.1em">MACRO</span>'
        f'<span style="font-size:12px;color:{color};font-weight:500">{n_active}/{n_modules}</span>'
        '</div>'
    )


@st.cache_data(ttl=120)
def _cached_active_trigger_count() -> int:
    """Fetch active-trigger count once per 2 minutes so the topbar stays cheap."""
    try:
        from features.macro_features import get_active_triggers
        import pandas as _pd
        return len(get_active_triggers(_pd.Timestamp.utcnow(), lookback_days=5))
    except Exception:
        return 0


def _zone_dots_html(zone: str | None) -> str:
    """Topbar zone indicator (spec §C). Empty string when flag off / no zone."""
    if zone not in ZONES or not _ecosystem_on():
        return ""
    dots = ""
    for key in ("data", "signals", "risk", "macro"):
        z = ZONES[key]
        if key == zone:
            dots += (f'<span style="width:7px;height:7px;border-radius:50%;background:{z["accent"]};'
                     f'box-shadow:0 0 6px {z["glow"]};display:inline-block"></span>')
        else:
            dots += (f'<span style="width:5px;height:5px;border-radius:50%;background:{z["accent"]};'
                     f'opacity:.35;display:inline-block"></span>')
    word = ZONES[zone]["label"].split(" ")[0]
    return (f'<div style="display:flex;align-items:center;gap:4px;margin-right:18px;flex-shrink:0">{dots}'
            f'<span style="font-size:8px;color:{ZONES[zone]["accent"]};letter-spacing:.1em;'
            f'margin-left:4px">{word}</span></div>')


def render_topbar(df=None, zone=None):
    """
    Render the fixed top context bar (44px, always visible):
      Accendio mark | zone dots | sector momentum | signal count | market sessions | timestamp
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    ts  = now.strftime("%H:%M UTC")
    h   = now.hour + now.minute / 60

    asia_open   = h >= 23 or h < 8
    europe_open = 7.0 <= h < 16.5
    us_open     = 13.5 <= h < 20.0

    def _sess(label, is_open):
        dot = ASCEND if is_open else "rgba(238,242,255,0.2)"
        op  = "1"    if is_open else "0.38"
        return (
            f'<span style="opacity:{op};margin-right:10px;font-size:10px;'
            f'letter-spacing:.08em;color:rgba(238,242,255,0.7)">'
            f'<span style="display:inline-block;width:5px;height:5px;border-radius:50%;'
            f'background:{dot};margin-right:4px;vertical-align:middle"></span>'
            f'{label}</span>'
        )

    sessions = _sess("ASIA", asia_open) + _sess("EU", europe_open) + _sess("US", us_open)

    if df is not None and not df.empty and "Pct_Change" in df.columns:
        n_sig     = int((df["Pct_Change"].abs() > 1.0).sum())
        sig_color = SIGNAL if n_sig > 0 else "rgba(238,242,255,0.25)"
    else:
        n_sig, sig_color = "—", "rgba(238,242,255,0.25)"

    if df is not None and not df.empty and "Sector" in df.columns:
        groups = [("NRG", "Energy"), ("MET", "Metals"), ("AGR", "Agriculture"), ("LVS", "Livestock")]
        gpills = ""
        for label, sector in groups:
            sub = df[df["Sector"] == sector]
            if sub.empty:
                gpills += (
                    f'<span style="margin-right:14px;font-size:10px;color:rgba(238,242,255,0.2);'
                    f'letter-spacing:.08em">{label}&nbsp;—</span>'
                )
            else:
                avg = sub["Pct_Change"].mean()
                c   = ASCEND if avg > 0.2 else (DESCEND if avg < -0.2 else "rgba(238,242,255,0.4)")
                s   = "+" if avg >= 0 else ""
                gpills += (
                    f'<span style="margin-right:14px;font-size:10px;color:{c};letter-spacing:.08em">'
                    f'{label}&nbsp;<span style="opacity:.9">{s}{avg:.2f}%</span></span>'
                )
    else:
        gpills = '<span style="font-size:10px;color:rgba(238,242,255,0.15)">—</span>'

    st.markdown(f"""
<div style="
  position:fixed;top:0;left:0;right:0;height:44px;
  background:#09102A;
  border-bottom:0.5px solid rgba(123,156,255,0.18);
  z-index:999999;
  display:flex;align-items:center;
  padding:0 20px 0 16px;
  font-family:'Arial','Helvetica Neue',Helvetica,sans-serif;
">
  <div style="display:flex;align-items:center;gap:9px;margin-right:18px;flex-shrink:0">
    <svg width="17" height="17" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
      <line x1="5" y1="26" x2="13" y2="7" stroke="#EEF2FF" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="13" y1="7" x2="26" y2="18" stroke="#7B9CFF" stroke-width="2.5" stroke-linecap="round"/>
    </svg>
    <span style="color:#EEF2FF;font-size:11px;font-weight:400;letter-spacing:.16em">ACCENDIO</span>
  </div>
  <div style="height:18px;width:0.5px;background:rgba(123,156,255,0.2);margin-right:18px;flex-shrink:0"></div>
  {_zone_dots_html(zone)}
  <div style="display:flex;align-items:center;flex:1;min-width:0;overflow:hidden">{gpills}</div>
  <div style="height:18px;width:0.5px;background:rgba(123,156,255,0.2);margin:0 16px;flex-shrink:0"></div>
  <div style="margin-right:18px;flex-shrink:0;display:flex;align-items:center;gap:6px">
    <span style="font-size:10px;color:rgba(238,242,255,0.28);letter-spacing:.1em">SIG</span>
    <span style="font-size:12px;color:{sig_color};font-weight:500">{n_sig}</span>
  </div>
  {_macro_trigger_pill_html()}
  <div style="margin-right:14px;flex-shrink:0">{sessions}</div>
  <div style="flex-shrink:0">
    <span style="font-size:10px;color:rgba(238,242,255,0.22);letter-spacing:.06em;font-family:'Courier New',monospace">{ts}</span>
  </div>
</div>
<div id="_ac_busy_bar">
  <span class="_ac_busy_e">📈</span>
  <span class="_ac_busy_e">🧮</span>
  <span class="_ac_busy_e">🔬</span>
  <span class="_ac_busy_e">📊</span>
</div>
""", unsafe_allow_html=True)


def _nav_flag_on(name: str) -> bool:
    """True when an env-var feature flag is set to an affirmative value."""
    import os
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _nav_section(label: str):
    """Render a four-layer section header in the sidebar directory."""
    st.markdown(
        f"<div style='font-size:0.68rem; letter-spacing:0.12em; font-weight:600; "
        f"text-transform:uppercase; color:{ICE_MID}; margin:0.6rem 0 0.15rem 0;'>"
        f"{label}</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar_nav_v2():
    """
    Four-layer page taxonomy (spec 4.1) — the sidebar mirrors how the system
    thinks, so a user navigates the four architectural layers:

      Markets & Data     → data-layer surface  (pricing, charts, data-health)
      Signals & Research → signal-layer surface (models, causal/cascade engine)
      Portfolio & Risk   → risk-layer surface   (target book, scenarios, alerts)
      Macro Context      → cross-cutting narrative (news, events, macro exposure)

    Flagged surfaces are listed inside the layer they belong to, only when their
    own feature flag is on.
    """
    _nav_section("Markets & Data")
    st.page_link("app.py",                           label="Home")
    if _ecosystem_on():
        st.page_link("pages/0_Ecosystem.py",         label="Ecosystem")
    st.page_link("pages/1_Pricing.py",               label="Pricing")
    st.page_link("pages/2_Charts.py",                label="Charts")
    st.page_link("pages/5_Database.py",              label="Data Health")

    _nav_section("Signals & Research")
    st.page_link("pages/4_Models.py",                label="Models")
    st.page_link("pages/6_Causal_QS_Engine.py",      label="Causal QS Engine")
    st.page_link("pages/7_Macro_Market_Cascade.py",  label="Macro-Market Cascade")
    if _nav_flag_on("SIGNAL_RESEARCH_ENABLED"):
        st.page_link("pages/13_Signal_Lab.py",       label="Signal Lab ⚗️")
    if _nav_flag_on("RESEARCH_LIBRARY_ENABLED"):
        st.page_link("pages/15_Research_Library.py", label="Research Library 📚")

    _nav_section("Portfolio & Risk")
    st.page_link("pages/8_Portfolio.py",             label="Portfolio")
    st.page_link("pages/9_Scenarios.py",             label="Scenarios")
    st.page_link("pages/12_Alerts.py",               label="Alerts")
    if _nav_flag_on("PRODUCTION_PORTFOLIO_ENABLED"):
        st.page_link("pages/14_Live_Portfolio.py",   label="Live Portfolio ⚙️")

    _nav_section("Macro Context")
    st.page_link("pages/3_News.py",                  label="News")
    st.page_link("pages/10_Event_Ribbon.py",         label="Event Ribbon")
    st.page_link("pages/11_Macro_Exposure.py",       label="Macro Exposure")


def _render_sidebar_nav_legacy():
    """Original three-group directory. Default until the v2 taxonomy is proven."""
    # Group 1 — core data
    st.page_link("app.py",                           label="Home")
    st.page_link("pages/1_Pricing.py",               label="Pricing")
    st.page_link("pages/2_Charts.py",                label="Charts")
    st.page_link("pages/3_News.py",                  label="News")
    st.page_link("pages/4_Models.py",                label="Models")
    st.page_link("pages/5_Database.py",              label="Database")
    st.divider()

    # Group 2 — analytics
    st.page_link("pages/6_Causal_QS_Engine.py",      label="Causal QS Engine")
    st.page_link("pages/7_Macro_Market_Cascade.py",  label="Macro-Market Cascade")
    st.page_link("pages/8_Portfolio.py",             label="Portfolio")
    st.page_link("pages/9_Scenarios.py",             label="Scenarios")
    st.divider()

    # Group 3 — live signals
    st.page_link("pages/10_Event_Ribbon.py",         label="Event Ribbon")
    st.page_link("pages/11_Macro_Exposure.py",       label="Macro Exposure")
    st.page_link("pages/12_Alerts.py",               label="Alerts")

    # Research-grade surface — only listed when its feature flag is on.
    if _nav_flag_on("SIGNAL_RESEARCH_ENABLED"):
        st.divider()
        st.page_link("pages/13_Signal_Lab.py",       label="Signal Lab ⚗️")
    if _nav_flag_on("PRODUCTION_PORTFOLIO_ENABLED"):
        if not _nav_flag_on("SIGNAL_RESEARCH_ENABLED"):
            st.divider()
        st.page_link("pages/14_Live_Portfolio.py",   label="Live Portfolio ⚙️")
    if _nav_flag_on("RESEARCH_LIBRARY_ENABLED"):
        if not (_nav_flag_on("SIGNAL_RESEARCH_ENABLED") or _nav_flag_on("PRODUCTION_PORTFOLIO_ENABLED")):
            st.divider()
        st.page_link("pages/15_Research_Library.py", label="Research Library 📚")


def render_sidebar_nav():
    """
    Render the canonical Accendio sidebar directory.

    Single source of truth for the page list. Call this at the top of every
    page (inside or outside a `with st.sidebar:` block — it manages its own).
    Page-specific filters/controls must live in the main page body, NOT here,
    so the sidebar stays a clean directory across all pages.

    The four-layer taxonomy (spec 4.1) is gated behind NAV_TAXONOMY_V2_ENABLED.
    The legacy three-group directory stays the default until v2 is proven, so
    rollback in production is a flag flip — no redeploy.
    """
    with st.sidebar:
        st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
        st.divider()
        if _nav_flag_on("NAV_TAXONOMY_V2_ENABLED"):
            _render_sidebar_nav_v2()
        else:
            _render_sidebar_nav_legacy()

        from components.docent import docent_toggle, _enabled as _docent_enabled
        if _docent_enabled():
            st.divider()
            docent_toggle()


def panel_header(title: str, badge: str = "", badge_color: str = SIGNAL):
    """Styled panel section label with optional badge."""
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="font-size:10px;color:{badge_color};'
            f'background:rgba(123,156,255,0.08);padding:2px 8px;border-radius:20px;'
            f'border:0.5px solid rgba(123,156,255,0.2)">{badge}</span>'
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;margin-top:2px">'
        f'<span style="font-size:10px;color:rgba(238,242,255,0.32);letter-spacing:.15em;'
        f'text-transform:uppercase;font-weight:500">{title}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )
