"""
Causal Chain Visualization — Real-Time Macro Cascade View.

Five-panel page:
  1. Macro State     — DXY z-score, VIX level, TLT momentum (colour-coded)
  2. Cascade Flow    — Sankey: macro channels → sectors → commodities
  3. Forecast Table  — base vs. macro-adjusted vs. final for each sector
  4. Narrative       — auto-generated plain-English summary
  5. Historical Accuracy — past macro shock events vs. actual sector outcomes
"""

import warnings
warnings.filterwarnings("ignore")

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from typing import Dict, List, Optional

from utils.theme import apply_theme, render_topbar, PLOTLY_LAYOUT
from utils.macro_narrative import (
    SECTORS, MACRO_EFFECTS, MACRO_EFFECTS_RISK_ON, MACRO_EFFECTS_RISK_OFF,
    load_macro, get_macro_state, compute_forecasts, build_narrative,
)
from models.cascade_validator import validate as _cascade_validate

st.set_page_config(
    page_title="Accendio | Causal Chains",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()
render_topbar()

# ── Brand colors ───────────────────────────────────────────────────────────────
VOID   = "#060912"
ABYSS  = "#09102A"
DEPTH  = "#0C1228"
GRID   = "#1A2A5E"
GREEN  = "#3DB87A"
RED    = "#D94F4F"
BLUE   = "#7B9CFF"
AMBER  = "#F59E0B"
PURPLE = "#AB47BC"
TEAL   = "#26C6DA"
ICE    = "#EEF2FF"


def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _signal_strength(final_pct: float, uncertainty_band: float):
    """
    Returns (label, bar_color) based on how clearly final_pct clears the
    uncertainty band.  A general audience sees 'Inconclusive' instead of
    a misleading Bullish/Bearish when the bar is buried in noise.

    Thresholds (relative to the 90% CI half-width):
      |final| < 0.5× band  →  Inconclusive  (signal is noise)
      0.5–1.0×             →  Weak
      1.0–2.0×             →  Moderate
      > 2.0×               →  Strong
    """
    if uncertainty_band <= 0:
        uncertainty_band = 1.0
    ratio = abs(final_pct) / uncertainty_band
    direction = "▲" if final_pct >= 0 else "▼"
    if ratio < 0.5:
        return "— Inconclusive", _hex_rgba(ICE, 0.18)
    if ratio < 1.0:
        color = _hex_rgba(GREEN, 0.55) if final_pct >= 0 else _hex_rgba(RED, 0.55)
        return f"{direction} Weak", color
    if ratio < 2.0:
        color = GREEN if final_pct >= 0 else RED
        return f"{direction} Moderate", color
    color = GREEN if final_pct >= 0 else RED
    return f"{direction} Strong", color


def _color_metric_html(label: str, value: str, sub: str, color: str) -> str:
    return f"""
<div style="
  background:{DEPTH};border:0.5px solid {_hex_rgba(color, 0.4)};
  border-left:3px solid {color};border-radius:8px;
  padding:14px 18px;text-align:left;
">
  <div style="font-size:10px;color:{_hex_rgba(ICE, 0.35)};letter-spacing:.12em;
              text-transform:uppercase;font-weight:500;margin-bottom:4px">{label}</div>
  <div style="font-size:1.5rem;font-weight:300;color:{color}">{value}</div>
  <div style="font-size:11px;color:{_hex_rgba(ICE, 0.45)};margin-top:4px">{sub}</div>
</div>"""


# ── Sector / commodity map ─────────────────────────────────────────────────────
# SECTORS and MACRO_EFFECTS are imported from utils.macro_narrative above.

SECTOR_COLORS = {
    "Energy":    AMBER,
    "Metals":    BLUE,
    "Grains":    GREEN,
    "Livestock": PURPLE,
}

ALL_COMMODITIES = [c for comms in SECTORS.values() for c in comms]

TICKERS = {
    "WTI Crude Oil":           "CL=F",
    "Brent Crude Oil":         "BZ=F",
    "Natural Gas (Henry Hub)": "NG=F",
    "Gold (COMEX)":            "GC=F",
    "Silver (COMEX)":          "SI=F",
    "Copper (COMEX)":          "HG=F",
    "Corn (CBOT)":             "ZC=F",
    "Wheat (CBOT SRW)":        "ZW=F",
    "Soybeans (CBOT)":         "ZS=F",
    "Feeder Cattle":           "GF=F",
    "Lean Hogs":               "HE=F",
}


# Cascade sector connections from the three macro channels
# {macro_channel_idx: {sector_idx: weight}}
MACRO_SECTOR_WIRING = {
    0: {1: 1.0},                               # Real Yields → Metals
    1: {0: 0.7, 1: 0.5},                       # USD → Energy + Metals
    2: {0: 0.5, 1: 0.6, 2: 0.5, 3: 0.6},      # Risk-Off → all sectors
}


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(period_days: int = 756) -> pd.DataFrame:
    from services.data_contract import fetch_price_matrix
    df = fetch_price_matrix(names=ALL_COMMODITIES, days=period_days)
    if not df.empty:
        cols = [c for c in ALL_COMMODITIES if c in df.columns]
        return df[cols].ffill().dropna(how="all")
    try:
        import yfinance as yf
        raw = yf.download(
            list(TICKERS.values()),
            period=f"{period_days // 365 + 1}y",
            interval="1d", progress=False, auto_adjust=True,
        )
        prices = raw["Close"].rename(columns={v: k for k, v in TICKERS.items()})
        return prices.ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


# load_macro, get_macro_state, compute_forecasts, build_narrative
# are imported from utils.macro_narrative above.


# ── Cascade Topology Diagram ───────────────────────────────────────────────────

def build_topology_diagram() -> go.Figure:
    """
    Static node-link diagram of the full 9-edge cross-sector causal topology.

    Node layout (data coords, x in [0,10], y in [0,6]):
      Energy(1,3)  Metals(3.5,4.8)  Agriculture(6.5,4.8)  Livestock(9,3)  Digital(5.5,1.2)

    Arrow colour = source sector colour so edge bundles are readable at a glance.
    Edge labels are rendered in the reference table below the chart (not inline).
    """
    nodes = {
        "Energy":      (1.0, 3.0),
        "Metals":      (3.5, 4.8),
        "Agriculture": (6.5, 4.8),
        "Livestock":   (9.0, 3.0),
        "Digital":     (5.5, 1.2),
    }

    # (source, target) — mirrors UPSTREAM_MAP in cascade_orchestrator.py
    edges = [
        ("Energy",      "Metals"),
        ("Energy",      "Agriculture"),
        ("Energy",      "Livestock"),
        ("Energy",      "Digital"),
        ("Metals",      "Agriculture"),
        ("Metals",      "Livestock"),
        ("Metals",      "Digital"),
        ("Agriculture", "Livestock"),
        ("Agriculture", "Digital"),
    ]

    # Source-sector colour map (Digital not a source in UPSTREAM_MAP)
    _src_color = {
        "Energy":      AMBER,
        "Metals":      BLUE,
        "Agriculture": GREEN,
    }

    fig = go.Figure()

    # Arrows first so node circles sit on top
    for src, tgt in edges:
        x0, y0 = nodes[src]
        x1, y1 = nodes[tgt]
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3, arrowsize=1.1, arrowwidth=1.8,
            arrowcolor=_hex_rgba(_src_color.get(src, ICE), 0.65),
            text="",
        )

    # Node circles + sector labels
    _node_sector_colors = {**SECTOR_COLORS, "Digital": TEAL}
    for sector, (x, y) in nodes.items():
        color = _node_sector_colors.get(sector, ICE)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=60,
                color=_hex_rgba(color, 0.12),
                line=dict(color=color, width=2),
            ),
            text=[sector],
            textposition="middle center",
            textfont=dict(color=color, size=11),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=270, margin=dict(t=10, l=10, r=10, b=10))
    fig.update_xaxes(visible=False, range=[0, 10.5])
    fig.update_yaxes(visible=False, range=[0, 6])
    return fig


# Edge reference table rendered below the topology diagram
_TOPOLOGY_EDGE_TABLE = [
    ("Energy",      "Metals",      AMBER, "Mining & smelting electricity; ore processing fuel"),
    ("Energy",      "Agriculture", AMBER, "Fertilizer production & farm machinery fuel costs"),
    ("Energy",      "Livestock",   AMBER, "Facility heating/cooling; cold-chain transport fuel"),
    ("Energy",      "Digital",     AMBER, "Electricity is Bitcoin's dominant variable cost"),
    ("Metals",      "Agriculture", BLUE,  "Farm equipment steel; irrigation copper; storage capex"),
    ("Metals",      "Livestock",   BLUE,  "Equipment & barn construction material costs"),
    ("Metals",      "Digital",     BLUE,  "ASIC hardware metals; data-centre cooling infrastructure"),
    ("Agriculture", "Livestock",   GREEN, "Feed-grain prices are the primary rearing cost"),
    ("Agriculture", "Digital",     GREEN, "Food-CPI inflation signal → rate expectations → BTC demand"),
]


def render_topology_edge_table() -> None:
    """Render a compact styled HTML reference table for all 9 topology edges."""
    rows_html = ""
    for src, tgt, color, mechanism in _TOPOLOGY_EDGE_TABLE:
        rows_html += (
            f"<tr>"
            f"<td style='color:{color};font-weight:500;white-space:nowrap;padding:4px 10px'>{src}</td>"
            f"<td style='color:{_hex_rgba(ICE,0.4)};padding:4px 6px'>→</td>"
            f"<td style='color:{color};font-weight:500;white-space:nowrap;padding:4px 10px'>{tgt}</td>"
            f"<td style='color:{_hex_rgba(ICE,0.55)};font-size:11px;padding:4px 12px'>{mechanism}</td>"
            f"</tr>"
        )
    st.markdown(
        f"""<div style='background:{DEPTH};border:0.5px solid {_hex_rgba(GRID,0.6)};
        border-radius:8px;padding:10px 4px;margin-top:6px;overflow-x:auto'>
        <table style='border-collapse:collapse;width:100%;font-size:12px'>
          <thead><tr>
            <th style='color:{_hex_rgba(ICE,0.3)};font-size:9px;letter-spacing:.1em;
                       text-transform:uppercase;padding:4px 10px;text-align:left'>From</th>
            <th></th>
            <th style='color:{_hex_rgba(ICE,0.3)};font-size:9px;letter-spacing:.1em;
                       text-transform:uppercase;padding:4px 10px;text-align:left'>To</th>
            <th style='color:{_hex_rgba(ICE,0.3)};font-size:9px;letter-spacing:.1em;
                       text-transform:uppercase;padding:4px 12px;text-align:left'>Economic mechanism</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table></div>""",
        unsafe_allow_html=True,
    )


# ── Cascade Sankey ─────────────────────────────────────────────────────────────

def build_cascade_sankey(macro_state: Dict, forecasts: pd.DataFrame) -> go.Figure:
    """
    Three-layer Sankey:
      Layer 0 (x≈0.02): Macro channels — Real Yields, USD Strength, Risk Sentiment
      Layer 1 (x≈0.45): Sectors
      Layer 2 (x≈0.88): Key commodities (2 per sector)
    Link widths encode signal magnitude; link colours encode forecast direction.
    """
    ry  = abs(macro_state["real_yields_signal"])
    usd = abs(macro_state["usd_signal"])
    ro  = abs(macro_state["risk_off_signal"])

    channel_strengths = [ry, usd, ro]
    channel_labels = [
        f"Real Yields {'↑' if macro_state['real_yields_signal'] > 0 else '↓'}",
        f"USD {'Strong' if macro_state['usd_signal'] > 0 else 'Weak'}",
        f"Risk {'Off' if macro_state['risk_off'] else 'On'}",
    ]
    channel_colors = [
        RED if macro_state["real_yields_signal"] > 0 else GREEN,
        RED if macro_state["usd_signal"] > 0 else GREEN,
        RED if macro_state["risk_off"] else GREEN,
    ]

    sector_list = list(SECTORS.keys())

    # Representative commodities per sector (first two that exist in forecasts)
    repr_comms = {}
    for sector, comms in SECTORS.items():
        available = [c for c in comms if c in forecasts["commodity"].values]
        repr_comms[sector] = available[:2] if len(available) >= 2 else available

    # ── Build node arrays ──────────────────────────────────────────────────────
    labels, colors, xs, ys = [], [], [], []

    # Layer 0: macro channels
    for i, (lbl, col) in enumerate(zip(channel_labels, channel_colors)):
        labels.append(lbl)
        colors.append(col)
        xs.append(0.02)
        ys.append([0.2, 0.5, 0.8][i])

    # Layer 1: sectors
    sector_start = len(labels)
    sector_ys    = [0.15, 0.40, 0.65, 0.87]
    for sector, y in zip(sector_list, sector_ys):
        labels.append(sector)
        colors.append(SECTOR_COLORS[sector])
        xs.append(0.45)
        ys.append(y)

    # Layer 2: commodities
    comm_start = len(labels)
    all_repr   = [c for sector in sector_list for c in repr_comms.get(sector, [])]
    n_comms    = len(all_repr)
    comm_ys    = np.linspace(0.05, 0.95, max(n_comms, 1)).tolist()

    for comm, y in zip(all_repr, comm_ys):
        frow  = forecasts[forecasts["commodity"] == comm]
        fp    = frow.iloc[0]["final_pct"] if not frow.empty else 0.0
        color = GREEN if fp > 0.1 else (RED if fp < -0.1 else BLUE)
        short = (comm.replace(" (COMEX)", "").replace(" (CBOT)", "")
                 .replace(" (Henry Hub)", "").replace(" (CBOT SRW)", ""))
        labels.append(f"{short}\n{fp:+.2f}%")
        colors.append(color)
        xs.append(0.88)
        ys.append(y)

    # ── Build link arrays ──────────────────────────────────────────────────────
    src, tgt, val, lcol = [], [], [], []

    # Macro → Sector
    for macro_i, sector_dict in MACRO_SECTOR_WIRING.items():
        for sector_j, base_w in sector_dict.items():
            strength = channel_strengths[macro_i] * base_w
            if strength < 0.01:
                continue
            src.append(macro_i)
            tgt.append(sector_start + sector_j)
            val.append(max(strength, 0.04))
            lcol.append(_hex_rgba(channel_colors[macro_i], 0.35))

    # Sector → Commodity
    comm_idx = comm_start
    for s_i, sector in enumerate(sector_list):
        for comm in repr_comms.get(sector, []):
            frow = forecasts[forecasts["commodity"] == comm]
            fp   = abs(frow.iloc[0]["final_pct"]) if not frow.empty else 0.0
            node_col = colors[comm_idx]
            src.append(sector_start + s_i)
            tgt.append(comm_idx)
            val.append(max(fp / 2.0, 0.04))
            lcol.append(_hex_rgba(node_col, 0.3))
            comm_idx += 1

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color=GRID, width=0.8),
            label=labels,
            color=colors,
            x=xs,
            y=ys,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=460,
        margin=dict(t=20, l=10, r=10, b=10),
    )
    fig.update_layout(font_size=12)
    return fig


# ── Forecast bar charts ────────────────────────────────────────────────────────

def build_sector_bar(forecasts: pd.DataFrame, sector: str) -> go.Figure:
    df = forecasts[forecasts["sector"] == sector].copy()
    if df.empty:
        return go.Figure()

    short_names = (
        df["commodity"]
        .str.replace(r" \(.*?\)", "", regex=True)
        .str.replace(" (Henry Hub)", "", regex=False)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Base",
        x=short_names,
        y=df["base_pct"],
        marker_color=_hex_rgba(BLUE, 0.55),
        error_y=dict(
            type="data", array=df["uncertainty_band"] * 0.5,
            visible=True, color=_hex_rgba(ICE, 0.25), thickness=1.2, width=4,
        ),
        hovertemplate="%{x}<br>Base: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Macro Adj",
        x=short_names,
        y=df["macro_adj_pct"],
        marker_color=[GREEN if v > 0 else RED for v in df["macro_adj_pct"]],
        opacity=0.75,
        hovertemplate="%{x}<br>Macro adj: %{y:+.2f}%<extra></extra>",
    ))
    _final_colors = [
        _signal_strength(fp, ub)[1]
        for fp, ub in zip(df["final_pct"], df["uncertainty_band"])
    ]
    fig.add_trace(go.Bar(
        name="Final",
        x=short_names,
        y=df["final_pct"],
        marker_color=_final_colors,
        error_y=dict(
            type="data", array=df["uncertainty_band"],
            visible=True, color=_hex_rgba(AMBER, 0.5), thickness=1.5, width=5,
        ),
        hovertemplate="%{x}<br>Final: %{y:+.2f}%<br>±%{error_y.array:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=_hex_rgba(ICE, 0.18), line_width=0.8)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=230,
        barmode="group",
        margin=dict(t=4, l=10, r=10, b=30),
        showlegend=True,
        yaxis_title="1-wk Forecast (%)",
        yaxis_tickformat="+.2f",
    )
    fig.update_layout(legend=dict(orientation="h", y=1.18, x=0, font=dict(size=9)))
    return fig


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    df = forecasts.copy()
    df["Short Name"] = (
        df["commodity"]
        .str.replace(r" \(.*?\)", "", regex=True)
        .str.replace(" (Henry Hub)", "", regex=False)
    )

    def _fmt(v):
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"

    tbl = pd.DataFrame({
        "Sector":               df["sector"],
        "Commodity":            df["Short Name"],
        "Base Forecast":        df["base_pct"].map(_fmt),
        "Macro Adjustment":     df["macro_adj_pct"].map(_fmt),
        "Final Forecast":       df["final_pct"].map(_fmt),
        "90% Band (±)":        df["uncertainty_band"].map(lambda v: f"{v:.2f}%"),
        "Signal Strength":      df.apply(
            lambda r: _signal_strength(r["final_pct"], r["uncertainty_band"])[0], axis=1
        ),
    })
    return tbl.reset_index(drop=True)


# ── Historical accuracy ────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def compute_historical_accuracy(
    _prices: pd.DataFrame, _macro_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify macro shock events (VIX crossings, DXY spikes) in history and
    check whether the cascade's directional predictions matched actual sector
    returns over the following 10 trading days.
    """
    if _macro_df is None or _macro_df.empty or _prices.empty:
        return pd.DataFrame()

    required = {"vix", "dxy_zscore63"}
    if not required.issubset(_macro_df.columns):
        return pd.DataFrame()

    # Detect shock events
    shocks = []
    vix_s = _macro_df["vix"].dropna()
    dxy_s = _macro_df.get("dxy_zscore63", pd.Series(dtype=float)).dropna()
    tlt_s = _macro_df.get("tlt_mom21",    pd.Series(dtype=float)).dropna()

    # Rising VIX crosses above 20
    for i in range(1, len(vix_s)):
        if vix_s.iloc[i] >= 20 and vix_s.iloc[i - 1] < 20:
            shocks.append({
                "date":  vix_s.index[i],
                "event": f"VIX spike → {vix_s.iloc[i]:.1f}",
                "type":  "risk_off",
            })

    # DXY z-score spikes above 1.5
    for i in range(1, len(dxy_s)):
        if dxy_s.iloc[i] >= 1.5 and dxy_s.iloc[i - 1] < 1.5:
            shocks.append({
                "date":  dxy_s.index[i],
                "event": f"DXY z-score → {dxy_s.iloc[i]:.2f}",
                "type":  "usd_strength",
            })

    # TLT momentum turns sharply negative (real yields rising)
    for i in range(1, len(tlt_s)):
        if tlt_s.iloc[i] <= -0.05 and tlt_s.iloc[i - 1] > -0.05:
            shocks.append({
                "date":  tlt_s.index[i],
                "event": f"TLT momentum → {tlt_s.iloc[i]*100:.1f}%",
                "type":  "real_yields",
            })

    if not shocks:
        return pd.DataFrame()

    # Cascade directional predictions per shock type
    TYPE_EXPECTED = {
        "risk_off":    {"Energy": -1, "Metals": +1, "Grains": -1, "Livestock": -1},
        "usd_strength":{"Energy": -1, "Metals": -1, "Grains": -1, "Livestock": -1},
        "real_yields": {"Energy": -1, "Metals": -1, "Grains":  0, "Livestock":  0},
    }

    rows = []
    for shock in shocks[-30:]:  # last 30 events max
        shock_date = shock["date"]
        expected   = TYPE_EXPECTED.get(shock["type"], {})

        for sector, comms in SECTORS.items():
            exp_dir = expected.get(sector, 0)
            if exp_dir == 0:
                continue
            avail = [c for c in comms if c in _prices.columns]
            if not avail:
                continue
            try:
                pos = _prices.index.searchsorted(shock_date)
                if pos >= len(_prices) - 11:
                    continue
                p0 = _prices[avail].iloc[pos]
                p1 = _prices[avail].iloc[pos + 10]
                fwd_ret  = float(((p1 / p0) - 1).mean())
                actual   = 1 if fwd_ret > 0 else -1
                correct  = actual == exp_dir
                rows.append({
                    "Date":           shock_date.strftime("%Y-%m-%d"),
                    "Event":          shock["event"],
                    "Sector":         sector,
                    "Expected":       "▲ Bullish" if exp_dir > 0 else "▼ Bearish",
                    "10d Return":     f"{fwd_ret*100:+.2f}%",
                    "Outcome":        "✅ Correct" if correct else "❌ Incorrect",
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )


def accuracy_scorecard(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty or "Outcome" not in hist_df.columns:
        return pd.DataFrame()
    hist_df = hist_df.copy()
    hist_df["_correct"] = hist_df["Outcome"].str.startswith("✅")
    grp = hist_df.groupby("Sector")["_correct"]
    sc = pd.DataFrame({
        "Events":   grp.count(),
        "Correct":  grp.sum().astype(int),
        "Accuracy": (grp.mean() * 100).round(1),
    }).reset_index()
    sc["Accuracy"] = sc["Accuracy"].map(lambda x: f"{x:.0f}%")
    return sc.sort_values("Events", ascending=False)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.divider()
    st.page_link("app.py",                           label="Home")
    st.page_link("pages/1_Pricing.py",               label="Pricing")
    st.page_link("pages/2_Charts.py",                label="Charts")
    st.page_link("pages/3_News.py",                  label="News")
    st.page_link("pages/4_Models.py",                label="Models")
    st.page_link("pages/5_Database.py",              label="Database")
    st.divider()
    st.page_link("pages/6_Causal_QS_Engine.py",      label="Causal QS Engine")
    st.page_link("pages/7_Macro_Market_Cascade.py",  label="Macro-Market Cascade")
    st.page_link("pages/8_Portfolio.py",             label="Portfolio")
    st.divider()
    st.caption("Macro signals require network · price data from latest DB ingest")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("Macro-Market Cascade")
st.markdown(
    """
<div style="background:#0C1228;border:0.5px solid rgba(249,158,11,0.25);
border-left:3px solid #F59E0B;border-radius:8px;padding:14px 18px;margin-bottom:12px">
  <div style="font-size:10px;color:rgba(238,242,255,0.35);letter-spacing:.12em;
              text-transform:uppercase;margin-bottom:6px">EXTERNAL INPUT LAYER</div>
  <div style="font-size:13px;color:rgba(238,242,255,0.80);line-height:1.65">
    The Macro-Market Cascade is the <b style="color:#EEF2FF">external input layer</b> — macro signals
    (DXY, VIX, TLT) shape the environment that individual commodity forecasts operate within.
    This page visualises how those signals flow through sectors into specific commodity outlooks,
    and how confident the model is in each channel.
  </div>
  <div style="font-size:11px;color:rgba(238,242,255,0.35);margin-top:8px">
    Pair with <b>Causal QS Engine</b> to trace how a specific trigger event responds to these conditions. →
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
_load_col, _status_col = st.columns([5, 1])
with _load_col:
    with st.spinner("Loading price history and macro overlay…"):
        _prices      = load_prices(period_days=756)
        _macro       = load_macro(period="2y")

_macro_state = get_macro_state(_macro)
_forecasts   = compute_forecasts(_prices, _macro_state)

data_ok  = not _prices.empty
macro_ok = not _macro.empty

with _status_col:
    st.markdown(
        f"<div style='text-align:right;font-size:10px;color:{_hex_rgba(ICE, 0.35)};margin-top:8px'>"
        f"{'✓ DB prices' if data_ok else '⚠ No price data'}<br>"
        f"{'✓ Macro data' if macro_ok else '⚠ Macro unavailable'}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Cascade Health Badge ──────────────────────────────────────────────────────
try:
    _vr = _cascade_validate(_forecasts)
except Exception as _ve:
    from models.cascade_validator import ValidationResult as _VR
    _vr = _VR(
        status="no_data", sector_accuracy={}, sector_lift={},
        lag_confirmed_pct=float("nan"), flags=[f"Validator error: {_ve}"],
        last_run=None, n_shock_events=0,
    )

# Badge colour and label
if _vr.status == "healthy" and not _vr.flags:
    _badge_col   = GREEN
    _badge_label = "Cascade Healthy"
elif _vr.status == "no_data":
    _badge_col   = AMBER
    _badge_label = "No Validation Data"
elif _vr.status in ("degraded", "stale_coefficients") or _vr.flags:
    _badge_col   = AMBER
    _badge_label = "Cascade Degraded"
else:
    _badge_col   = RED
    _badge_label = "Cascade Failed"

_last_run_str = (
    f"Last run: {_vr.last_run[:19].replace('T', ' ')} UTC"
    if _vr.last_run else "Never run"
)
_events_str = f" · {_vr.n_shock_events} shock events" if _vr.n_shock_events else ""

st.markdown(
    f"""
<div style="
  background:{_hex_rgba(_badge_col, 0.08)};
  border:0.5px solid {_hex_rgba(_badge_col, 0.35)};
  border-left:3px solid {_badge_col};
  border-radius:8px;padding:10px 18px;margin-bottom:8px;
  display:flex;align-items:center;gap:18px;flex-wrap:wrap;
">
  <div style="
    background:{_hex_rgba(_badge_col, 0.15)};
    border:1px solid {_hex_rgba(_badge_col, 0.45)};
    border-radius:20px;padding:3px 13px;
    font-size:11px;font-weight:600;color:{_badge_col};
    letter-spacing:.09em;text-transform:uppercase;white-space:nowrap;
  ">● {_badge_label}</div>
  <div style="font-size:11px;color:{_hex_rgba(ICE, 0.45)}">
    {_last_run_str}{_events_str}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("🔍 Cascade validator details", expanded=(_vr.status not in ("healthy", "no_data"))):
    if _vr.status == "no_data":
        st.info(
            "No backtest runs found. The validator reads from `cascade_validation_summary` — "
            "run `BacktestHarness.validate_causal_chains()` (or `daily_retrain.py`) "
            "to populate it.",
            icon="📋",
        )
    else:
        # Flags
        if _vr.flags:
            for _f in _vr.flags:
                st.markdown(
                    f"<div style='background:{_hex_rgba(AMBER,0.08)};border-left:3px solid {AMBER};"
                    f"border-radius:4px;padding:6px 12px;font-size:12px;"
                    f"color:{_hex_rgba(ICE,0.80)};margin-bottom:4px'>⚠ {_f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='color:{GREEN};font-size:12px;margin-bottom:6px'>"
                f"✓ No flags — all chains within healthy thresholds.</div>",
                unsafe_allow_html=True,
            )

        # Sector accuracy table
        if _vr.sector_accuracy:
            _acc_rows = []
            for _sec in ("Energy", "Metals", "Grains", "Livestock"):
                _acc = _vr.sector_accuracy.get(_sec, float("nan"))
                _lift = _vr.sector_lift.get(_sec, float("nan"))
                _ok = (not math.isnan(_acc)) and _acc >= 0.50
                _acc_rows.append({
                    "Sector":              _sec,
                    "Dir. Accuracy":       f"{_acc*100:.1f}%" if not math.isnan(_acc) else "—",
                    "vs. Baseline":        "✅ Beats baseline" if _ok else "❌ Below floor (50%)",
                    "Avg MAE Lift (pp)":   f"{_lift:+.3f}" if not math.isnan(_lift) else "—",
                })
            st.dataframe(
                pd.DataFrame(_acc_rows),
                use_container_width=True, hide_index=True,
                height=min(len(_acc_rows) * 38 + 44, 220),
            )

        # Lag structure
        _lag_pct = _vr.lag_confirmed_pct
        if not math.isnan(_lag_pct):
            _lag_color = GREEN if _lag_pct >= 0.5 else AMBER
            st.markdown(
                f"<div style='font-size:12px;color:{_hex_rgba(ICE, 0.60)};margin-top:6px'>"
                f"Energy → Metals lag confirmed in "
                f"<span style='color:{_lag_color};font-weight:600'>"
                f"{_lag_pct*100:.0f}%</span> of shock events.</div>",
                unsafe_allow_html=True,
            )

    st.caption(
        f"Status: {_vr.status.upper()} · {_last_run_str}{_events_str} · "
        "Powered by cascade_validation_summary table."
    )


# ── SECTION 0: Cascade Topology ──────────────────────────────────────────────
st.subheader("⓪ Cascade Topology")
st.caption(
    "Static causal dependency map — how price shocks propagate across commodity sectors "
    "before any macro channel overlay is applied."
)
st.plotly_chart(build_topology_diagram(), use_container_width=True,
                config={"displayModeBar": False})
render_topology_edge_table()
st.divider()


# ── SECTION 1: Macro State ────────────────────────────────────────────────────
st.subheader("① Current Macro State")
st.caption(
    f"Signals as of {_macro_state['date']} — "
    "green = bullish for commodities, red = headwind."
)

dxy_z  = _macro_state["dxy_zscore"]
vix    = _macro_state["vix"]
tlt21  = _macro_state["tlt_mom21"]

# DXY z-score: negative z = weak dollar = bullish for commodities
if dxy_z < -0.5:
    dxy_color, dxy_label = GREEN, "Weak Dollar — Commodity Tailwind"
elif dxy_z > 1.5:
    dxy_color, dxy_label = RED,   "Strong Dollar — Commodity Headwind"
elif dxy_z > 0.5:
    dxy_color, dxy_label = AMBER, "Mild Dollar Strength"
else:
    dxy_color, dxy_label = BLUE,  "Dollar Neutral"

# VIX: lower is better for risk assets
if vix >= 30:
    vix_color, vix_label = RED,   "Crisis — Extreme Risk-Off"
elif vix >= 20:
    vix_color, vix_label = RED,   "Risk-Off — Elevated Volatility"
elif vix >= 15:
    vix_color, vix_label = AMBER, "Caution — Vol Elevated"
else:
    vix_color, vix_label = GREEN, "Risk-On — Low Volatility"

# TLT momentum: positive = rates falling = commodity tailwind
if tlt21 >= 0.02:
    tlt_color, tlt_label = GREEN, "Rates Falling — Commodity Tailwind"
elif tlt21 <= -0.02:
    tlt_color, tlt_label = RED,   "Rates Rising — Commodity Headwind"
else:
    tlt_color, tlt_label = BLUE,  "Rates Stable"

m1, m2, m3, m4 = st.columns(4)
_sub_dim = f"color:{_hex_rgba(ICE, 0.32)};font-size:10px"
with m1:
    st.markdown(
        _color_metric_html(
            "DXY Z-Score (63d)", f"{dxy_z:+.2f}",
            f"{dxy_label}<br><span style='{_sub_dim}'>"
            "Dollar strength suppresses USD-priced commodity returns</span>",
            dxy_color,
        ),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        _color_metric_html(
            "VIX Level", f"{vix:.1f}",
            f"{vix_label}<br><span style='{_sub_dim}'>"
            "Elevated volatility compresses risk appetite and weakens industrial demand</span>",
            vix_color,
        ),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _color_metric_html(
            "TLT 21d Momentum", f"{tlt21*100:+.2f}%",
            f"{tlt_label}<br><span style='{_sub_dim}'>"
            "Rising rates raise inventory costs and reinforce USD headwinds</span>",
            tlt_color,
        ),
        unsafe_allow_html=True,
    )
with m4:
    # Overall signal composite
    ry_s  = _macro_state["real_yields_signal"]
    usd_s = _macro_state["usd_signal"]
    ro_s  = _macro_state["risk_off_signal"]
    # Composite: positive = headwind (bearish for most commodities)
    composite = ry_s * 0.3 + usd_s * 0.4 + ro_s * 0.3
    comp_color = RED if composite > 0.25 else (GREEN if composite < -0.25 else AMBER)
    comp_label = ("Bearish Macro Environment" if composite > 0.25
                  else ("Bullish Macro Environment" if composite < -0.25 else "Mixed Signals"))
    st.markdown(
        _color_metric_html(
            "Macro Composite", f"{composite:+.2f}",
            f"{comp_label}<br><span style='{_sub_dim}'>"
            "Weighted: real yields 30% · USD 40% · risk sentiment 30%</span>",
            comp_color,
        ),
        unsafe_allow_html=True,
    )

# ── Regime coefficient badge + comparison ─────────────────────────────────────
_is_risk_off  = _macro_state.get("risk_off", False)
_active_table = MACRO_EFFECTS_RISK_OFF if _is_risk_off else MACRO_EFFECTS_RISK_ON
_regime_color = RED if _is_risk_off else GREEN
_regime_label = "RISK-OFF  ·  VIX ≥ 20" if _is_risk_off else "RISK-ON  ·  VIX < 20"
_regime_desc  = (
    "Regime-conditional coefficients active: safe-haven amplification for Gold/Silver, "
    "deepened demand-destruction discount for Copper and Energy."
    if _is_risk_off else
    "Regime-conditional coefficients active: standard inverse macro sensitivities, "
    "safe-haven demand for Gold muted."
)

st.markdown(
    f"""
<div style="
  background:{DEPTH};border:0.5px solid {_hex_rgba(_regime_color, 0.4)};
  border-left:3px solid {_regime_color};border-radius:8px;
  padding:10px 18px;margin-bottom:6px;
  display:flex;align-items:center;gap:16px;
">
  <div>
    <div style="font-size:9px;color:{_hex_rgba(ICE, 0.35)};letter-spacing:.12em;
                text-transform:uppercase;margin-bottom:3px">Active macro-effect table</div>
    <div style="font-size:1rem;font-weight:600;color:{_regime_color}">{_regime_label}</div>
  </div>
  <div style="font-size:11px;color:{_hex_rgba(ICE, 0.55)};line-height:1.5">
    {_regime_desc}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("🔢 Causal weight matrix", expanded=False):
    # Column header definitions — shown as subtext inside the header cells
    _COL_DEFS = {
        "real_yields": (
            "real_yields",
            "Real-rate signal (−TLT 21d momentum, normalised to [−1,1]). "
            "Positive signal = rates rising = headwind for most commodities via carry cost.",
        ),
        "usd": (
            "usd",
            "Dollar-strength signal (DXY 63d z-score, normalised to [−1,1]). "
            "Positive = stronger USD = lower USD-priced commodity returns.",
        ),
        "risk_off": (
            "risk_off",
            "Risk-sentiment signal (VIX − 15, normalised to [0,1]). "
            "Positive = risk-off; lifts safe havens (Gold), crushes industrial demand.",
        ),
    }

    # Build rows: commodities in SECTORS order, with sector group labels
    def _short(name: str) -> str:
        return (
            name.replace(" (COMEX)", "").replace(" (CBOT)", "")
                .replace(" (Henry Hub)", "").replace(" (CBOT SRW)", "")
        )

    def _cell_color(v: float) -> str:
        if v > 0.01:
            return GREEN
        if v < -0.01:
            return RED
        return _hex_rgba(ICE, 0.35)

    # Header row HTML
    _th_style = (
        f"padding:8px 14px;text-align:right;font-size:11px;font-weight:600;"
        f"color:{_hex_rgba(ICE, 0.75)};border-bottom:1px solid {_hex_rgba(GRID, 0.8)};"
        f"white-space:nowrap;vertical-align:bottom"
    )
    _th_left = (
        f"padding:8px 14px;text-align:left;font-size:11px;font-weight:600;"
        f"color:{_hex_rgba(ICE, 0.75)};border-bottom:1px solid {_hex_rgba(GRID, 0.8)};"
        f"vertical-align:bottom"
    )
    _def_style = (
        f"display:block;font-size:9px;font-weight:400;color:{_hex_rgba(ICE, 0.38)};"
        f"line-height:1.4;max-width:160px;white-space:normal;margin-top:3px"
    )

    _header_html = (
        f"<tr>"
        f"<th style='{_th_left}'>Sector</th>"
        f"<th style='{_th_left}'>Commodity</th>"
    )
    for _key, (_label, _def) in _COL_DEFS.items():
        _header_html += (
            f"<th style='{_th_style}'>{_label}"
            f"<span style='{_def_style}'>{_def}</span></th>"
        )
    _header_html += "</tr>"

    # Data rows
    _td_base = f"padding:7px 14px;font-size:12px;border-bottom:1px solid {_hex_rgba(GRID, 0.25)}"
    _rows_html = ""
    _prev_sector = None
    for _sector, _comms in SECTORS.items():
        for _comm in _comms:
            _efx = _active_table.get(_comm, {})
            _is_new_sector = _sector != _prev_sector
            _prev_sector   = _sector
            _sec_color     = SECTOR_COLORS.get(_sector, ICE)

            _rows_html += "<tr>"
            # Sector cell — only show label on the first commodity in each group
            if _is_new_sector:
                _rows_html += (
                    f"<td style='{_td_base};color:{_sec_color};"
                    f"font-weight:600;font-size:10px;letter-spacing:.07em;"
                    f"text-transform:uppercase'>{_sector}</td>"
                )
            else:
                _rows_html += f"<td style='{_td_base}'></td>"

            # Commodity name
            _rows_html += (
                f"<td style='{_td_base};color:{_hex_rgba(ICE, 0.75)}'>"
                f"{_short(_comm)}</td>"
            )

            # Coefficient cells
            for _key in ("real_yields", "usd", "risk_off"):
                _v = _efx.get(_key, 0.0)
                _c = _cell_color(_v)
                _rows_html += (
                    f"<td style='{_td_base};text-align:right;"
                    f"color:{_c};font-weight:500;font-variant-numeric:tabular-nums'>"
                    f"{_v:+.2f}</td>"
                )
            _rows_html += "</tr>"

    st.markdown(
        f"""
<div style="overflow-x:auto;margin-bottom:4px">
<table style="border-collapse:collapse;width:100%;font-family:monospace">
  <thead>{_header_html}</thead>
  <tbody>{_rows_html}</tbody>
</table>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Active table: {_regime_label}. "
        "Coefficients multiply normalised signals to produce the macro adjustment in percentage points "
        "(e.g. coefficient −0.50 × usd_signal 0.80 = −0.40 pp adjustment). "
        "Sources: Baur & Lucey (2010), Reboredo (2013), Hamilton (2009), IMF WEO Apr-2012."
    )

with st.expander("📊 Compare risk-on vs risk-off coefficient tables", expanded=False):
    st.markdown(
        f"""
<div style="font-size:11px;color:{_hex_rgba(ICE, 0.45)};margin-bottom:8px;line-height:1.6">
  Coefficients are multiplied by normalised signals (real_yields ∈ [−1,1],
  usd ∈ [−1,1], risk_off ∈ [0,1]) to produce the macro adjustment in percentage points.
  The <b style="color:{_regime_color}">{_regime_label}</b> table is currently active.
  Sources: Baur &amp; Lucey (2010), Hood &amp; Malik (2013), Erb &amp; Harvey (2013),
  Hamilton (2009), IMF WEO Apr-2012.
</div>
""",
        unsafe_allow_html=True,
    )

    _cmp_rows = []
    for sector, comms in SECTORS.items():
        for comm in comms:
            ron = MACRO_EFFECTS_RISK_ON.get(comm, {})
            rof = MACRO_EFFECTS_RISK_OFF.get(comm, {})
            short = comm.replace(" (COMEX)", "").replace(" (CBOT)", "") \
                        .replace(" (Henry Hub)", "").replace(" (CBOT SRW)", "")
            _cmp_rows.append({
                "Sector":                 sector,
                "Commodity":              short,
                "RiskOn · real_yields":  f"{ron.get('real_yields', 0):+.2f}",
                "RiskOff · real_yields": f"{rof.get('real_yields', 0):+.2f}",
                "RiskOn · usd":          f"{ron.get('usd', 0):+.2f}",
                "RiskOff · usd":         f"{rof.get('usd', 0):+.2f}",
                "RiskOn · risk_off":     f"{ron.get('risk_off', 0):+.2f}",
                "RiskOff · risk_off":    f"{rof.get('risk_off', 0):+.2f}",
            })

    _cmp_df = pd.DataFrame(_cmp_rows)
    st.dataframe(
        _cmp_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(_cmp_df) * 38 + 44, 450),
    )
    st.caption(
        "Highlighted cells: Gold risk_off +0.30 (risk-on) → +0.80 (risk-off); "
        "Copper risk_off −0.40 → −0.70; WTI/Brent risk_off −0.30 → −0.60. "
        "Gold's usd coefficient also flips from −0.20 to +0.15 in risk-off, "
        "reflecting simultaneous dollar/gold safe-haven rallies (Reboredo 2013)."
    )

st.divider()


# ── SECTION 2: Cascade Flow Diagram ──────────────────────────────────────────
st.subheader("② Real-Time Cascade Flow")
st.caption(
    "How macro channels route through sectors to individual commodity forecasts. "
    "Link width = signal magnitude · Link colour = forecast direction (green = bullish, red = bearish)."
)

if _forecasts.empty:
    st.warning("No forecast data — price history unavailable. Run the ingestion pipeline first.", icon="⚠️")
else:
    fig_sankey = build_cascade_sankey(_macro_state, _forecasts)
    st.plotly_chart(fig_sankey, use_container_width=True, config={"displayModeBar": False})

st.divider()


# ── SECTION 3: Sector Forecast Breakdown ─────────────────────────────────────
st.subheader("③ Sector Forecast Breakdown")
st.caption(
    "Base forecast = 21-day momentum signal (5 trading-day horizon). "
    "Macro adjustment = causal overlay from DXY, VIX, TLT channels. "
    "Error bars = 90% confidence interval."
)

if not _forecasts.empty:
    tabs = st.tabs(list(SECTORS.keys()) + ["📋 Full Table"])

    for tab, sector in zip(tabs[:-1], SECTORS.keys()):
        with tab:
            sc = SECTOR_COLORS[sector]
            st.markdown(
                f"<div style='border-left:3px solid {sc};padding-left:10px;"
                f"margin-bottom:10px'>"
                f"<span style='color:{sc};font-size:12px;font-weight:500'>"
                f"{sector}</span></div>",
                unsafe_allow_html=True,
            )
            bar_col, tbl_col = st.columns([3, 2])
            with bar_col:
                fig_bar = build_sector_bar(_forecasts, sector)
                st.plotly_chart(fig_bar, use_container_width=True,
                                config={"displayModeBar": False})
            with tbl_col:
                df_s = _forecasts[_forecasts["sector"] == sector].copy()
                def _row_color(v):
                    if v > 0.1:
                        return f"color:{GREEN}"
                    if v < -0.1:
                        return f"color:{RED}"
                    return f"color:{BLUE}"

                display = pd.DataFrame({
                    "Commodity":     df_s["commodity"].str.replace(r" \(.*?\)", "", regex=True),
                    "Base":          df_s["base_pct"].map(lambda v: f"{v:+.2f}%"),
                    "Macro Adj":     df_s["macro_adj_pct"].map(lambda v: f"{v:+.2f}%"),
                    "Final":         df_s["final_pct"].map(lambda v: f"{v:+.2f}%"),
                    "90% CI (±)":   df_s["uncertainty_band"].map(lambda v: f"{v:.2f}%"),
                })
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    height=min(len(display) * 38 + 44, 300),
                )

    with tabs[-1]:
        full_tbl = build_comparison_table(_forecasts)
        st.dataframe(full_tbl, use_container_width=True, hide_index=True, height=420)
        st.caption(
            "All values are 1-week (5 trading day) point forecasts in percentage points. "
            "Macro adjustment is rule-based from the causal chain matrix; "
            "base forecast uses 21-day rolling mean log-return."
        )

st.divider()


# ── SECTION 4: Narrative ──────────────────────────────────────────────────────
st.subheader("④ Cascade Narrative")
st.caption("Auto-generated plain-English explanation of current macro signals and their commodity implications.")

if not _forecasts.empty:
    narrative = build_narrative(_macro_state, _forecasts)
    for para in narrative.split("\n\n"):
        if para.strip():
            st.markdown(
                f"""<div style="
                  background:{DEPTH};border-left:3px solid {_hex_rgba(BLUE, 0.5)};
                  border-radius:6px;padding:14px 18px;margin-bottom:10px;
                  color:{_hex_rgba(ICE, 0.85)};font-size:0.93rem;line-height:1.75;
                ">{para}</div>""",
                unsafe_allow_html=True,
            )
else:
    st.info("Narrative requires price and macro data. Check your ingestion pipeline.", icon="📖")

st.divider()


# ── SECTION 5: Historical Cascade Accuracy ───────────────────────────────────
st.subheader("⑤ Historical Cascade Accuracy")
st.caption(
    "For each identified macro shock event (VIX crossing, DXY spike, TLT momentum reversal), "
    "did the cascade correctly predict the direction of sector returns over the following 10 trading days?"
)

with st.spinner("Computing historical accuracy…"):
    hist_df = compute_historical_accuracy(_prices, _macro)

if hist_df.empty:
    st.info(
        "No historical shock events found in available data. "
        "This section populates once sufficient macro history is loaded "
        "(requires 2+ years of DXY, VIX, TLT data).",
        icon="📊",
    )
else:
    # Scorecard summary at the top
    scorecard = accuracy_scorecard(hist_df)
    if not scorecard.empty:
        sc_cols = st.columns(len(scorecard))
        for col, (_, row) in zip(sc_cols, scorecard.iterrows()):
            acc_val = float(row["Accuracy"].replace("%", ""))
            acc_color = GREEN if acc_val >= 60 else (AMBER if acc_val >= 50 else RED)
            with col:
                st.markdown(
                    _color_metric_html(
                        row["Sector"],
                        row["Accuracy"],
                        f"{row['Correct']}/{row['Events']} events",
                        acc_color,
                    ),
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # Overall accuracy badge
    total_events   = len(hist_df)
    correct_events = hist_df["Outcome"].str.startswith("✅").sum()
    overall_acc    = correct_events / total_events * 100 if total_events else 0
    overall_color  = GREEN if overall_acc >= 60 else (AMBER if overall_acc >= 50 else RED)

    st.markdown(
        f"<div style='background:{DEPTH};border:0.5px solid {_hex_rgba(overall_color, 0.4)};"
        f"border-radius:8px;padding:12px 18px;margin-bottom:14px;display:inline-block'>"
        f"<span style='font-size:10px;color:{_hex_rgba(ICE, 0.4)};letter-spacing:.1em;"
        f"text-transform:uppercase'>Overall Cascade Accuracy</span>"
        f"<span style='font-size:1.3rem;font-weight:300;color:{overall_color};margin-left:14px'>"
        f"{overall_acc:.0f}%</span>"
        f"<span style='font-size:11px;color:{_hex_rgba(ICE, 0.4)};margin-left:8px'>"
        f"({correct_events}/{total_events} events)</span></div>",
        unsafe_allow_html=True,
    )

    # Detail table — collapsible
    with st.expander(f"📋 Event-level detail ({total_events} shock events)", expanded=False):
        st.dataframe(hist_df, use_container_width=True, hide_index=True, height=420)
        st.caption(
            "Events detected: VIX crossings above 20, DXY z-score spikes above 1.5, "
            "TLT momentum reversals below −5%. Expected direction derived from causal "
            "chain rules; actual direction from realized sector returns over the 10 "
            "trading days following the event."
        )

st.divider()
st.caption(
    "Macro-Market Cascade · Accendio Commodities Intelligence · "
    "Forecasts are model-generated and for informational purposes only. "
    "Macro overlay based on DXY, VIX, and TLT signals."
)

st.divider()
st.markdown(
    """
<div style="background:#09102A;border:0.5px solid rgba(123,156,255,0.15);
border-radius:8px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center">
  <div>
    <div style="font-size:10px;color:rgba(238,242,255,0.3);letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px">
      NEXT STEP
    </div>
    <div style="font-size:13px;color:rgba(238,242,255,0.75)">
      Trace how a specific trigger event responds to these macro conditions →
      <b style="color:#EEF2FF">Causal QS Engine</b>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.page_link("pages/6_Causal_QS_Engine.py", label="→ Open Causal QS Engine")
