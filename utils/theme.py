"""Accendio brand theme — global CSS and shared UI components."""

import datetime
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
    title_font=dict(color=ICE, size=13),
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

/* Multiselect / radio / checkbox */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
  color: rgba(238,242,255,0.65) !important;
}
</style>
"""


def apply_theme():
    """Inject the Accendio CSS theme. Call once per page after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_topbar(df=None):
    """
    Render the fixed top context bar (44px, always visible):
      Accendio mark | sector momentum | signal count | market sessions | timestamp
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
  <div style="display:flex;align-items:center;flex:1;min-width:0;overflow:hidden">{gpills}</div>
  <div style="height:18px;width:0.5px;background:rgba(123,156,255,0.2);margin:0 16px;flex-shrink:0"></div>
  <div style="margin-right:18px;flex-shrink:0;display:flex;align-items:center;gap:6px">
    <span style="font-size:10px;color:rgba(238,242,255,0.28);letter-spacing:.1em">SIG</span>
    <span style="font-size:12px;color:{sig_color};font-weight:500">{n_sig}</span>
  </div>
  <div style="margin-right:14px;flex-shrink:0">{sessions}</div>
  <div style="flex-shrink:0">
    <span style="font-size:10px;color:rgba(238,242,255,0.22);letter-spacing:.06em;font-family:'Courier New',monospace">{ts}</span>
  </div>
</div>
""", unsafe_allow_html=True)


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
