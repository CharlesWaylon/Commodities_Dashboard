"""
Causal Chain Explorer — Phase 4 of the Model Integration Roadmap.

Visualises how a macro trigger event ripples through the model ecosystem:

    Trigger
        │  (how did realized vol shift?)
    Statistical layer  — rolling vol before vs. after
        │  (what was the macro regime?)
    Regime layer       — VIX / DXY state on trigger date
        │  (what did the ensemble decide?)
    Ensemble layer     — MetaPredictor weight decision
        │  (what does the portfolio do?)
    Portfolio layer    — rule-based position recommendation

UI sections
───────────
1. Controls — pick commodity, trigger family, date, strength
2. Sankey diagram — 5-layer flow, confidence-weighted, direction-coloured
3. Narrative panel — plain-English chain explanation
4. Layer cards — expandable detail per node
5. Recent events — last 60 days from SQLite trigger log (if populated)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import sys, os

from utils.theme import apply_theme, render_topbar, PLOTLY_LAYOUT
from models.causal_chain import ChainNode, CausalChainResult

st.set_page_config(
    page_title="Accendio | Causal Chain",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()
render_topbar()

# ── Theme constants ────────────────────────────────────────────────────────────
BG      = "#060912"
PLOT_BG = "#0C1228"
GRID    = "#1A2A5E"
AMBER   = "#F59E0B"
BLUE    = "#4FC3F7"
GREEN   = "#66BB6A"
RED     = "#EF5350"
PURPLE  = "#AB47BC"
TEAL    = "#26C6DA"

DIRECTION_COLORS = {
    "bullish":       GREEN,
    "bearish":       RED,
    "neutral":       BLUE,
    "heightened_vol": PURPLE,
    "reduced_vol":   AMBER,
}

LAYER_LABELS = {
    "trigger":     "① Trigger<br><sup>What set the signal in motion</sup>",
    "statistical": "② Vol Shift<br><sup>How volatility responded</sup>",
    "regime":      "③ Regime<br><sup>What market environment we're in</sup>",
    "ensemble":    "④ Ensemble<br><sup>How the models voted</sup>",
    "portfolio":   "⑤ Portfolio<br><sup>What it means for your positions</sup>",
}

LAYER_ORDER = ["trigger", "statistical", "regime", "ensemble", "portfolio"]

LAYER_ICONS = {
    "trigger":     "⚡",
    "statistical": "📊",
    "regime":      "🌡️",
    "ensemble":    "🤖",
    "portfolio":   "💼",
}

# ── Tickers / commodities (mirrors 4_Models.py) ────────────────────────────────
CORE_TICKERS = {
    "WTI Crude Oil":           "CL=F",
    "Brent Crude Oil":         "BZ=F",
    "Natural Gas (Henry Hub)": "NG=F",
    "Gasoline (RBOB)":         "RB=F",
    "Heating Oil":             "HO=F",
    "Gold (COMEX)":            "GC=F",
    "Silver (COMEX)":          "SI=F",
    "Copper (COMEX)":          "HG=F",
    "Corn (CBOT)":             "ZC=F",
    "Wheat (CBOT SRW)":        "ZW=F",
    "Soybeans (CBOT)":         "ZS=F",
}

# Map from CORE_TICKERS display name → (VAR group key, DB canonical column name).
# The DB canonical names match COMMODITY_GROUPS entries in models/statistical/var_vecm.py.
COMMODITY_TO_GROUP: dict[str, tuple[str, str]] = {
    "WTI Crude Oil":           ("energy",           "WTI Crude Oil"),
    "Brent Crude Oil":         ("energy",           "Brent Crude Oil"),
    "Natural Gas (Henry Hub)": ("energy",           "Natural Gas"),
    "Gasoline (RBOB)":         ("energy",           "Gasoline (RBOB)"),
    "Heating Oil":             ("energy",           "Heating Oil"),
    "Gold (COMEX)":            ("precious_metals",  "Gold"),
    "Silver (COMEX)":          ("precious_metals",  "Silver"),
    "Copper (COMEX)":          ("industrial_metals","Copper"),
    "Corn (CBOT)":             ("grains",           "Corn"),
    "Wheat (CBOT SRW)":        ("grains",           "Wheat"),
    "Soybeans (CBOT)":         ("grains",           "Soybeans"),
}

# Reverse: DB canonical name → display name (for labelling ripple nodes)
_DB_TO_DISPLAY: dict[str, str] = {v[1]: k for k, v in COMMODITY_TO_GROUP.items()}


# ── Data loaders (cached, DB-first) ────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(period: str = "3y") -> pd.DataFrame:
    from services.data_contract import fetch_price_matrix
    days = int(period[:-1]) * 365 if period.endswith("y") else 365
    df = fetch_price_matrix(names=list(CORE_TICKERS.keys()), days=days)
    if not df.empty:
        return df
    import yfinance as yf
    raw = yf.download(
        list(CORE_TICKERS.values()), period=period,
        interval="1d", progress=False, auto_adjust=True,
    )
    prices = raw["Close"].rename(columns={v: k for k, v in CORE_TICKERS.items()})
    return prices.ffill().dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(period: str = "2y", _price_index=None) -> pd.DataFrame:
    from features.macro_overlays import build_macro_overlay_features
    try:
        return build_macro_overlay_features(period=period, index=_price_index)
    except Exception:
        return pd.DataFrame()


# ── Sparkline builder ──────────────────────────────────────────────────────────

_SPARK_DAYS_BEFORE = 30   # trading days of history shown left of the trigger
_SPARK_DAYS_AFTER  = 10   # trading days shown right of the trigger (context)
_SPARK_HEIGHT      = 115  # px — compact but readable


def _build_layer_sparkline(
    layer: str,
    node: "ChainNode",
    prices: pd.DataFrame,
    macro_df: pd.DataFrame,
    commodity: str,
    trigger_date: str,
    line_color: str,
    ic_trend_df: pd.DataFrame,
) -> "go.Figure | None":
    """
    Return a compact Plotly sparkline for the given chain layer, or None
    if the layer has no time-series signal or the data are insufficient.

    Statistical → rolling 21-day annualised vol from prices.
    Regime      → VIX from macro_df (horizontal reference at VIX=20).
    Ensemble    → mean IC across all tiers from ic_log (reference at IC=0.05).
    Trigger / Portfolio → None (point-in-time; no meaningful trailing series).

    The trigger date is marked with a dashed amber vertical line.
    """
    try:
        t_ts = pd.Timestamp(trigger_date)
    except Exception:
        return None

    t_str = t_ts.strftime("%Y-%m-%d")   # Plotly vline needs string for date axes

    def _make_fig(
        x: pd.Index,
        y: np.ndarray,
        title: str,
        yline: float | None = None,
        yline_label: str = "",
    ) -> "go.Figure | None":
        if len(x) < 5:
            return None
        x_list = [str(d)[:10] for d in x]   # ISO date strings for Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_list,
            y=y,
            mode="lines",
            line=dict(color=line_color, width=1.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(line_color, 0.10),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ))
        # Trigger-date vertical line — only when it falls within the x range.
        # Use add_shape + add_annotation instead of add_vline: Plotly's add_vline
        # internally calls sum() on the string x-range to place the annotation,
        # which raises TypeError when the axis uses ISO date strings.
        if x_list[0] <= t_str <= x_list[-1]:
            fig.add_shape(
                type="line",
                x0=t_str, x1=t_str, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=AMBER, width=1.2, dash="dash"),
                opacity=0.80,
            )
            fig.add_annotation(
                x=t_str, y=0.98,
                xref="x", yref="paper",
                text="trigger", showarrow=False,
                font=dict(size=7, color=AMBER),
                xanchor="left", yanchor="top",
            )
        # Optional horizontal reference line (threshold)
        if yline is not None:
            fig.add_hline(
                y=yline,
                line_width=0.8, line_dash="dot",
                line_color="rgba(255,255,255,0.22)",
                annotation_text=yline_label,
                annotation_position="top right",
                annotation_font=dict(size=7, color="rgba(255,255,255,0.35)"),
            )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=_SPARK_HEIGHT,
            margin=dict(t=20, b=2, l=2, r=2),
            xaxis=dict(
                showgrid=False, showticklabels=False,
                showline=False, zeroline=False,
                range=[x_list[0], x_list[-1]],
            ),
            yaxis=dict(
                showgrid=False, showticklabels=False,
                showline=False, zeroline=False,
            ),
            showlegend=False,
            title_text=title,
            title_font=dict(size=8, color="rgba(238,242,255,0.30)"),
            title_x=0.0,
            title_pad=dict(t=2, l=4),
        )
        return fig

    # ── Statistical: rolling 21-day annualised realised vol ────────────────────
    if layer == "statistical":
        if commodity not in prices.columns:
            return None
        log_ret = np.log(prices[commodity] / prices[commodity].shift(1)).dropna()
        vol     = log_ret.rolling(21).std() * np.sqrt(252) * 100
        vol     = vol.dropna()
        # Extra buffer before trigger so the rolling window is fully warmed up
        start   = t_ts - pd.offsets.BDay(_SPARK_DAYS_BEFORE + 5)
        end     = t_ts + pd.offsets.BDay(_SPARK_DAYS_AFTER)
        sliced  = vol.loc[start:end].iloc[-(_SPARK_DAYS_BEFORE + _SPARK_DAYS_AFTER + 5):]
        return _make_fig(sliced.index, sliced.values, "Rolling 21d realised vol  (%)")

    # ── Regime: VIX ────────────────────────────────────────────────────────────
    if layer == "regime":
        if macro_df is None or macro_df.empty or "vix" not in macro_df.columns:
            return None
        vix    = macro_df["vix"].dropna()
        start  = t_ts - pd.offsets.BDay(_SPARK_DAYS_BEFORE)
        end    = t_ts + pd.offsets.BDay(_SPARK_DAYS_AFTER)
        sliced = vix.loc[start:end]
        return _make_fig(sliced.index, sliced.values, "VIX", yline=20.0, yline_label="risk-off ≥ 20")

    # ── Ensemble: mean IC across tiers from ic_log ─────────────────────────────
    if layer == "ensemble":
        if ic_trend_df.empty:
            return None
        # ic_trend() returns (computed_at, tier, mean_ic, n_commodities).
        # Collapse to one point per retrain run — average mean_ic across all tiers.
        agg = (
            ic_trend_df.groupby("computed_at")["mean_ic"]
            .mean()
            .reset_index()
            .rename(columns={"mean_ic": "ic"})
            .sort_values("computed_at")
        )
        agg["computed_at"] = pd.to_datetime(agg["computed_at"])
        agg = agg.tail(30)   # cap at 30 retrain runs
        if len(agg) < 2:
            return None
        return _make_fig(
            agg["computed_at"],
            agg["ic"].values,
            "Mean IC across tiers  (from ic_log)",
            yline=0.05,
            yline_label="actionable ≥ 0.05",
        )

    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _direction_label(d: str) -> str:
    return {
        "bullish":       "🟢 Bullish",
        "bearish":       "🔴 Bearish",
        "neutral":       "🔵 Neutral",
        "heightened_vol": "🟣 Heightened Vol",
        "reduced_vol":   "🟡 Reduced Vol",
    }.get(d, d.title())


def _confidence_gauge_html(confidence: float) -> str:
    """
    Three-zone horizontal progress bar for portfolio confidence.
    Red < 40 %, amber 40–65 %, green > 65 %.
    Includes a tooltip explaining what the score represents.
    """
    pct = confidence * 100
    if pct < 40:
        bar_color, zone_label = "#EF5350", "Low agreement"
    elif pct < 65:
        bar_color, zone_label = "#F59E0B", "Moderate agreement"
    else:
        bar_color, zone_label = "#66BB6A", "Strong agreement"

    tooltip = (
        "Confidence = fraction of upstream signals in agreement. "
        "The trigger, vol-shift, regime, and ensemble layers each cast a directional vote; "
        "confidence reflects how many of those votes align with the portfolio recommendation. "
        "Below 40 % the signal is noisy; 40–65 % is actionable with caution; above 65 % is high conviction."
    )

    return f"""
<div title="{tooltip}" style="margin-top:12px; cursor:help;">
  <div style="
      display:flex;
      justify-content:space-between;
      font-size:0.78rem;
      color:#8899BB;
      margin-bottom:4px;
  ">
    <span>Confidence</span>
    <span style="color:{bar_color}; font-weight:600;">{pct:.0f}% &nbsp;·&nbsp; {zone_label}</span>
  </div>
  <div style="
      background:#1A2A5E;
      border-radius:6px;
      height:8px;
      overflow:hidden;
  ">
    <div style="
        width:{pct:.1f}%;
        height:100%;
        background:{bar_color};
        border-radius:6px;
        transition:width 0.4s ease;
    "></div>
  </div>
  <div style="
      display:flex;
      justify-content:space-between;
      font-size:0.68rem;
      color:#445577;
      margin-top:3px;
  ">
    <span>0%</span><span>40%</span><span>65%</span><span>100%</span>
  </div>
</div>
"""


# ── Sankey builder ─────────────────────────────────────────────────────────────

def build_sankey(
    nodes: list,
    height: int = 380,
    ripple_spillovers: "list[tuple[str, float]] | None" = None,
) -> go.Figure:
    """
    Build a Plotly Sankey diagram for the 5-layer causal chain.

    Each link width is proportional to the *destination* node's confidence.
    Link colour matches the destination node's direction.
    Missing layers are shown dimmed in grey.

    If ripple_spillovers is provided (list of (peer_display_name, irf_h5) tuples),
    the 5 core nodes are compressed leftward and peer-commodity nodes are appended
    at the right edge with links from the Portfolio node; link width = |irf_h5|.
    """
    layer_map = {n.layer: n for n in nodes}
    has_ripple = bool(ripple_spillovers)

    # Core node x positions — compress leftward when ripple nodes are present
    if has_ripple:
        core_x = [0.01, 0.19, 0.37, 0.56, 0.74]
    else:
        core_x = [0.02, 0.27, 0.50, 0.73, 0.97]

    # Build node specs (5 core layers)
    n_labels, n_colors, n_customdata, n_x, n_y = [], [], [], [], []
    for i, layer in enumerate(LAYER_ORDER):
        node = layer_map.get(layer)
        lbl  = LAYER_LABELS.get(layer, layer.title())
        if node:
            conf_pct = f"{node.confidence:.0%}"
            color    = DIRECTION_COLORS.get(node.direction, BLUE)
            n_labels.append(f"{lbl}<br><sup>{_direction_label(node.direction)} · {conf_pct}</sup>")
            n_colors.append(color)
            n_customdata.append(node.summary[:120])
        else:
            n_labels.append(f"{lbl}<br><sup>— no data —</sup>")
            n_colors.append("rgba(80,80,100,0.4)")
            n_customdata.append("Layer not computed")
        n_x.append(core_x[i])
        n_y.append(0.50)

    # Append ripple (peer) nodes at the right
    if has_ripple:
        n_peers = len(ripple_spillovers)
        _y_positions = {
            1: [0.50],
            2: [0.28, 0.72],
            3: [0.18, 0.50, 0.82],
            4: [0.13, 0.38, 0.63, 0.88],
            5: [0.10, 0.30, 0.50, 0.70, 0.90],
        }.get(n_peers, [round(0.10 + 0.80 * i / max(n_peers - 1, 1), 2) for i in range(n_peers)])

        for j, (peer_name, irf5) in enumerate(ripple_spillovers):
            short = peer_name.split("(")[0].strip()
            sign  = "+" if irf5 >= 0 else "−"
            n_labels.append(f"<b>{short}</b><br><sup>IRF h=5: {sign}{abs(irf5):.3f}</sup>")
            n_colors.append(_hex_to_rgba(TEAL, 0.85))
            n_customdata.append(
                f"Spillover from trigger → {short} | IRF coupling at horizon 5: {irf5:+.4f}"
            )
            n_x.append(0.97)
            n_y.append(_y_positions[j] if j < len(_y_positions) else 0.50)

    # Build link specs (sequential core links)
    link_src, link_tgt, link_val, link_color, link_lbl = [], [], [], [], []
    for i in range(len(LAYER_ORDER) - 1):
        tgt_layer = LAYER_ORDER[i + 1]
        tgt_node  = layer_map.get(tgt_layer)
        val       = max(tgt_node.confidence if tgt_node else 0.08, 0.05)
        d_color   = DIRECTION_COLORS.get(tgt_node.direction if tgt_node else "neutral", BLUE)
        link_src.append(i)
        link_tgt.append(i + 1)
        link_val.append(val)
        link_color.append(_hex_to_rgba(d_color, 0.45))
        link_lbl.append(tgt_node.summary[:80] if tgt_node else "")

    # Ripple links: Portfolio node (index 4) → each peer node
    if has_ripple:
        port_node = layer_map.get("portfolio")
        for j, (peer_name, irf5) in enumerate(ripple_spillovers):
            peer_node_idx = 5 + j
            link_src.append(4)
            link_tgt.append(peer_node_idx)
            link_val.append(max(abs(irf5), 0.02))
            link_color.append(_hex_to_rgba(TEAL, 0.35))
            link_lbl.append(f"IRF spillover h=5: {irf5:+.4f}")

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=40,
            thickness=28,
            line=dict(color=GRID, width=1.5),
            label=n_labels,
            color=n_colors,
            customdata=n_customdata,
            hovertemplate="%{customdata}<extra>%{label}</extra>",
            x=n_x,
            y=n_y,
        ),
        link=dict(
            source=link_src,
            target=link_tgt,
            value=link_val,
            color=link_color,
            label=link_lbl,
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=height,
        margin=dict(t=30, l=10, r=10, b=10),
        title_text="",
    )
    fig.update_layout(font_size=13)
    return fig


# ── CausalRipple helpers ───────────────────────────────────────────────────────

def _build_trigger_band(
    source_col: str,
    direction: str,
    strength: float,
    tdate: str,
    horizon: int = 10,
) -> "ScenarioBand":
    """
    Synthesise a ScenarioBand from a trigger's direction and strength.

    The per-step mean log-return is ±strength × 0.015 (≈ daily commodity vol).
    Bear / bull envelopes are ±0.01 on top of that.
    """
    from models.scenarios.band import ScenarioBand

    direction_sign = {"bullish": 1.0, "bearish": -1.0, "heightened_vol": 0.5,
                      "reduced_vol": -0.2}.get(direction, 0.0)
    daily_mu = direction_sign * strength * 0.015
    spread   = 0.010
    mean_arr = np.full(horizon, daily_mu)
    return ScenarioBand(
        commodity    = source_col,
        source_model = "QS Engine Trigger",
        asof         = pd.Timestamp(tdate),
        horizon      = horizon,
        bear_q       = 0.10,
        bull_q       = 0.90,
        mean         = mean_arr,
        bear         = mean_arr - spread,
        bull         = mean_arr + spread,
    )



def _run_ripple(
    commodity_display: str,
    direction: str,
    strength: float,
    tdate: str,
    prices: pd.DataFrame,
) -> list[tuple[str, float]]:
    """
    Rename prices columns to DB canonical names, fit CausalRipple, and return
    (peer_display_name, irf_h5) pairs sorted by |irf_h5| descending.

    The IRF coupling at horizon h=5 is taken as index [4] (0-based) of the
    coupling array returned by CausalRipple.coupling().
    """
    from models.scenarios.ripple import CausalRipple

    info = COMMODITY_TO_GROUP.get(commodity_display)
    if info is None:
        return []
    group, source_col = info

    # Rename price columns from display names to DB canonical names
    rename = {disp: grp_col for disp, (_, grp_col) in COMMODITY_TO_GROUP.items()}
    prices_r = prices.rename(columns=rename)

    # Keep only columns present in the VAR group
    from models.statistical.var_vecm import COMMODITY_GROUPS
    group_members = COMMODITY_GROUPS.get(group, [])
    available = [c for c in group_members if c in prices_r.columns]
    if source_col not in available or len(available) < 2:
        return []

    prices_group = prices_r[available].dropna()
    if len(prices_group) < 60:
        return []

    band = _build_trigger_band(source_col, direction, strength, tdate)

    ripple = CausalRipple(prices_group, group=group)

    results = []
    for target_col in ripple.members():
        if target_col == source_col:
            continue
        try:
            propagated = ripple.propagate(band, target_col)
        except Exception:
            continue
        if propagated is None:
            continue
        # Extract IRF coupling at horizon 5 from the diagnostics list (0-indexed → index 4)
        coup = propagated.diagnostics.get("coupling", [])
        irf5 = float(coup[4]) if len(coup) > 4 else 0.0
        if abs(irf5) < 0.005:
            continue
        display = _DB_TO_DISPLAY.get(target_col, target_col)
        results.append((display, irf5))

    results.sort(key=lambda t: abs(t[1]), reverse=True)
    return results


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


# ── Page header ────────────────────────────────────────────────────────────────
st.title("Causal QS Engine")
st.markdown(
    """
<div style="background:#0C1228;border:0.5px solid rgba(123,156,255,0.2);
border-left:3px solid #7B9CFF;border-radius:8px;padding:14px 18px;margin-bottom:12px">
  <div style="font-size:10px;color:rgba(238,242,255,0.35);letter-spacing:.12em;
              text-transform:uppercase;margin-bottom:6px">INTERNAL RESPONSE LAYER</div>
  <div style="font-size:13px;color:rgba(238,242,255,0.80);line-height:1.65">
    The QS Engine is the <b style="color:#EEF2FF">internal response layer</b> — a live trigger fires,
    propagates through statistical, regime, and ensemble models, and produces a position recommendation.
    Use this page to trace the mechanics: pick a trigger family, set a commodity, and follow the signal
    from raw event through to a directional output.
  </div>
  <div style="font-size:11px;color:rgba(238,242,255,0.35);margin-top:8px">
    Pair with <b>Macro-Market Cascade</b> to see what drove the macro conditions that seeded the trigger. →
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.divider()


# ── Control panel ──────────────────────────────────────────────────────────────
st.subheader("Configure a trace")

from models.triggers import all_trigger_families

_families     = all_trigger_families()
_family_names = {f.description: f.name for f in _families}
_commodity_list = list(CORE_TICKERS.keys())

col_a, col_b, col_c, col_d = st.columns([2, 2, 1.5, 1])

with col_a:
    _selected_commodity = st.selectbox(
        "Commodity to trace",
        _commodity_list,
        index=0,
        key="causal_commodity",
    )

with col_b:
    _selected_family_desc = st.selectbox(
        "Trigger family",
        list(_family_names.keys()),
        index=0,
        key="causal_family",
    )
    _selected_family = _family_names[_selected_family_desc]

with col_c:
    _trigger_date = st.date_input(
        "Trigger date",
        value=date.today() - timedelta(days=30),
        min_value=date(2020, 1, 1),
        max_value=date.today(),
        key="causal_date",
    )

with col_d:
    st.write("")  # spacer to align button with inputs
    st.write("")
    _trace_btn = st.button("⚡ Trace Chain", type="primary", key="trace_chain_btn")

# Strength slider on its own row
_trigger_strength = st.slider(
    "Trigger strength",
    min_value=0.0, max_value=1.0, value=0.70, step=0.05,
    key="causal_strength",
    help="Override the trigger's measured strength for scenario analysis.",
)

if _trace_btn:
    st.session_state["causal_trace_requested"] = True
    st.session_state["causal_trace_config"] = {
        "commodity":  _selected_commodity,
        "family":     _selected_family,
        "date":       _trigger_date.strftime("%Y-%m-%d"),
        "strength":   _trigger_strength,
    }

st.divider()


# ── Example trace (shown on landing before first run) ─────────────────────────
_EXAMPLE_RESULT = CausalChainResult(
    trigger_family   = "opec_action",
    trigger_date     = "2023-11-30",
    trigger_strength = 0.78,
    affected_commodities = ["WTI Crude Oil (CL=F)", "Brent Crude (BZ=F)", "Natural Gas (NG=F)"],
    commodity        = "WTI Crude Oil (CL=F)",
    sector           = "Energy",
    narrative        = (
        "On 30 Nov 2023, an OPEC+ production-cut announcement (strength 0.78) "
        "sent a bullish jolt through the energy complex. Realized volatility in "
        "WTI crude spiked 38 % in the 10-day window after the meeting versus the "
        "prior 10 days, confirming the market absorbed the news as a genuine supply "
        "shock. The macro regime was already risk-on (VIX below its 1-year median, "
        "DXY z-score –0.4), providing fertile ground for an upside move. The ensemble "
        "arbitrator assigned a 71 % weight to bullish models. The portfolio layer "
        "recommends a modest long position with a tighter-than-usual stop, given that "
        "the vol expansion raises gap risk on the downside."
    ),
    portfolio_recommendation = "Modest long WTI, ~0.6× normal size; stop –4 % from entry.",
    portfolio_confidence     = 0.71,
    traced_at = "2023-12-01T09:00:00+00:00",
    nodes = [
        ChainNode(
            layer       = "trigger",
            model_name  = "OPEC Action Monitor",
            summary     = "OPEC+ announced a voluntary cut of 1 mb/d effective Jan 2024; "
                          "trigger strength 0.78 (strong).",
            confidence  = 0.78,
            before_value= None,
            after_value = None,
            direction   = "bullish",
            metadata    = {"event": "OPEC+ Nov 2023 ministerial", "cut_mbd": 1.0},
        ),
        ChainNode(
            layer       = "statistical",
            model_name  = "Realized Vol Comparator",
            summary     = "10-day realized vol rose from 18.2 % to 25.1 % (+38 %) after "
                          "the trigger — confirming the market treated this as a genuine shock.",
            confidence  = 0.74,
            before_value= 18.2,
            after_value = 25.1,
            change_pct  = 0.38,
            direction   = "heightened_vol",
        ),
        ChainNode(
            layer       = "regime",
            model_name  = "Macro Regime Overlay",
            summary     = "Risk-on regime on trigger date: VIX 14.2 (below 1yr median 17.8), "
                          "DXY z-score –0.4. Supportive backdrop for a bullish move.",
            confidence  = 0.68,
            before_value= 17.8,
            after_value = 14.2,
            change_pct  = -0.20,
            direction   = "bullish",
            metadata    = {"vix": 14.2, "dxy_zscore": -0.4},
        ),
        ChainNode(
            layer       = "ensemble",
            model_name  = "MetaPredictor Arbitrator",
            summary     = "Ensemble voted 71 % bullish weight; LSTM and XGBoost aligned, "
                          "Prophet slightly more cautious due to seasonal headwinds.",
            confidence  = 0.71,
            direction   = "bullish",
            metadata    = {"bullish_weight": 0.71, "model_count": 3},
        ),
        ChainNode(
            layer       = "portfolio",
            model_name  = "Rule-Based Sizer",
            summary     = "Modest long WTI, ~0.6× normal size; stop –4 % from entry. "
                          "Vol expansion raises gap risk, so position is sized conservatively.",
            confidence  = 0.71,
            direction   = "bullish",
        ),
    ],
)


def _render_example_trace(result: CausalChainResult) -> None:
    """Render the landing example trace with an [EXAMPLE] badge."""
    st.markdown(
        """
        <div style="
            display:inline-block;
            background:#1A2A5E;
            color:#F59E0B;
            font-size:0.72rem;
            font-weight:700;
            letter-spacing:0.08em;
            border-radius:4px;
            padding:2px 8px;
            margin-bottom:10px;
        ">EXAMPLE &nbsp;·&nbsp; WTI + OPEC action &nbsp;·&nbsp; 30 Nov 2023</div>
        <span style="color:#708090; font-size:0.82rem; margin-left:8px;">
            Run your own trace above — this is what the output looks like.
        </span>
        """,
        unsafe_allow_html=True,
    )

    banner_col, info_col = st.columns([3, 1])
    with banner_col:
        st.markdown(
            f"### Chain traced · 5/5 layers · "
            f"{result.trigger_date} · WTI Crude Oil"
        )
    with info_col:
        st.metric("Portfolio confidence", f"{result.portfolio_confidence:.0%}")

    st.plotly_chart(
        build_sankey(result.nodes, height=400),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    narr_col, rec_col = st.columns([3, 2])
    with narr_col:
        st.markdown("#### 📖 Chain narrative")
        st.markdown(
            f"""
            <div style="
                background:{PLOT_BG};
                border-left:4px solid {AMBER};
                border-radius:6px;
                padding:16px 20px;
                line-height:1.7;
                color:#D0D8F0;
                font-size:0.95rem;
            ">
            {result.narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rec_col:
        st.markdown("#### 💼 Portfolio recommendation")
        port_node = next((n for n in result.nodes if n.layer == "portfolio"), None)
        if port_node:
            dir_color = DIRECTION_COLORS.get(port_node.direction, BLUE)
            dir_label = _direction_label(port_node.direction)
            st.markdown(
                f"""
                <div style="
                    background:{PLOT_BG};
                    border-left:4px solid {dir_color};
                    border-radius:6px;
                    padding:16px 20px;
                    color:#D0D8F0;
                    font-size:0.95rem;
                    line-height:1.7;
                ">
                <div style="color:{dir_color}; font-weight:600; margin-bottom:6px;">
                    {dir_label}
                </div>
                {port_node.summary}
                {_confidence_gauge_html(port_node.confidence)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### 🔍 Layer-by-layer breakdown")
    layer_map = {n.layer: n for n in result.nodes}
    for layer in LAYER_ORDER:
        node     = layer_map.get(layer)
        icon     = LAYER_ICONS.get(layer, "•")
        label    = LAYER_LABELS.get(layer, layer.title())
        label_md = label.split("<br>")[0]
        if node:
            d_color    = DIRECTION_COLORS.get(node.direction, BLUE)
            header_md  = f"{icon} **{label_md}** — {_direction_label(node.direction)} · confidence {node.confidence:.0%}"
        else:
            d_color   = "rgba(80,80,100,0.5)"
            header_md = f"{icon} **{label_md}** — *not computed (insufficient data)*"

        with st.expander(header_md, expanded=(layer in ("trigger", "portfolio"))):
            if not node:
                st.info("Layer skipped in example.", icon="ℹ️")
                continue
            st.markdown(
                f"""
                <div style="
                    border-left:3px solid {d_color};
                    padding-left:12px;
                    margin-bottom:10px;
                    color:#C0CDE0;
                ">
                {node.summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if node.before_value is not None and node.after_value is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Before trigger", f"{node.before_value:.1f}")
                m2.metric("After trigger",  f"{node.after_value:.1f}",
                          delta=f"{node.change_pct:+.0%}" if node.change_pct is not None else None)
                m3.metric("Confidence", f"{node.confidence:.0%}")


# ── Trace output ───────────────────────────────────────────────────────────────
if not st.session_state.get("causal_trace_requested"):
    _render_example_trace(_EXAMPLE_RESULT)
else:
    cfg = st.session_state.get("causal_trace_config", {})
    commodity = cfg.get("commodity", _selected_commodity)
    family    = cfg.get("family",    _selected_family)
    tdate     = cfg.get("date",      _trigger_date.strftime("%Y-%m-%d"))
    strength  = cfg.get("strength",  _trigger_strength)

    with st.spinner(f"Loading price & macro data…"):
        _prices = load_prices(period="3y")
        _macro  = load_macro(period="2y", _price_index=_prices.index if not _prices.empty else None)

    with st.spinner(f"Running causal trace for {commodity}…"):
        from models.causal_chain import CausalChain
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family=family,
            trigger_date=tdate,
            prices=_prices,
            macro_df=_macro,
            commodity=commodity,
            trigger_strength=strength,
        )

    # ── Chain status banner ────────────────────────────────────────────────────
    n_layers = len(result.nodes)
    layer_names = [n.layer for n in result.nodes]
    missing = [l for l in LAYER_ORDER if l not in layer_names]

    banner_col, info_col = st.columns([3, 1])
    with banner_col:
        st.markdown(
            f"### Chain traced · {n_layers}/5 layers · "
            f"{result.trigger_date} · {commodity.split('(')[0].strip()}"
        )
    with info_col:
        port_conf = result.portfolio_confidence
        st.metric(
            "Portfolio confidence",
            f"{port_conf:.0%}",
            delta=None,
        )

    if missing:
        st.warning(
            f"⚠️ Layers skipped (insufficient data): {', '.join(missing)}",
            icon="⚠️",
        )

    # ── CausalRipple — compute peer spillovers ────────────────────────────────
    _ripple_spillovers: list[tuple[str, float]] = []
    _ripple_error: str | None = None
    _ripple_group: str | None = COMMODITY_TO_GROUP.get(commodity, (None, None))[0]

    if _ripple_group:
        with st.spinner("Computing cross-commodity ripple via VAR/IRF…"):
            try:
                _ripple_spillovers = _run_ripple(
                    commodity_display=commodity,
                    direction=(
                        next((n.direction for n in result.nodes if n.layer == "trigger"), "neutral")
                    ),
                    strength=strength,
                    tdate=tdate,
                    prices=_prices,
                )
            except Exception as _e:
                _ripple_error = str(_e)

    # ── Sankey diagram (extended with ripple nodes when available) ─────────────
    st.plotly_chart(
        build_sankey(
            result.nodes,
            height=420 if _ripple_spillovers else 400,
            ripple_spillovers=_ripple_spillovers or None,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Narrative + recommendation ─────────────────────────────────────────────
    narr_col, rec_col = st.columns([3, 2])

    with narr_col:
        st.markdown("#### 📖 Chain narrative")
        st.markdown(
            f"""
            <div style="
                background: {PLOT_BG};
                border-left: 4px solid {AMBER};
                border-radius: 6px;
                padding: 16px 20px;
                line-height: 1.7;
                color: #D0D8F0;
                font-size: 0.95rem;
            ">
            {result.narrative}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rec_col:
        st.markdown("#### 💼 Portfolio recommendation")
        port_node = next((n for n in result.nodes if n.layer == "portfolio"), None)
        if port_node:
            dir_color = DIRECTION_COLORS.get(port_node.direction, BLUE)
            dir_label = _direction_label(port_node.direction)
            st.markdown(
                f"""
                <div style="
                    background: {PLOT_BG};
                    border-left: 4px solid {dir_color};
                    border-radius: 6px;
                    padding: 16px 20px;
                    color: #D0D8F0;
                    font-size: 0.95rem;
                    line-height: 1.7;
                ">
                <div style="color:{dir_color}; font-weight:600; margin-bottom:6px;">
                    {dir_label}
                </div>
                {port_node.summary}
                {_confidence_gauge_html(port_node.confidence)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Portfolio layer not computed.")

    # ── Cross-commodity ripple panel ───────────────────────────────────────────
    st.divider()
    st.markdown("#### 🌊 Cross-commodity ripple (CausalRipple · VAR/IRF)")

    if _ripple_error:
        st.warning(
            f"CausalRipple could not run for this selection: {_ripple_error}",
            icon="⚠️",
        )
    elif not _ripple_group:
        st.info(
            f"No VAR group defined for **{commodity}** — ripple analysis unavailable.",
            icon="ℹ️",
        )
    elif not _ripple_spillovers:
        st.info(
            f"No peer commodities in the **{_ripple_group}** group produced a meaningful "
            "IRF response (|coupling h=5| < 0.005). This can happen when the group "
            "has too few overlapping observations or when the trigger direction is neutral.",
            icon="ℹ️",
        )
    else:
        _trigger_dir = next(
            (n.direction for n in result.nodes if n.layer == "trigger"), "neutral"
        )
        _dir_color = DIRECTION_COLORS.get(_trigger_dir, BLUE)
        _dir_label = _direction_label(_trigger_dir)

        st.markdown(
            f"""
<div style="
    background:{PLOT_BG};
    border-left:4px solid {TEAL};
    border-radius:6px;
    padding:12px 18px;
    font-size:0.88rem;
    color:#B0C4DE;
    line-height:1.6;
    margin-bottom:12px;
">
  A <span style="color:{_dir_color};font-weight:600">{_dir_label}</span> trigger
  on <b>{commodity.split("(")[0].strip()}</b> propagates through the
  <b>{_ripple_group}</b> VAR complex via orthogonalised IRFs.
  Link widths in the Sankey above and the table below show the coupling coefficient
  at forecast horizon 5 — the fraction of the source shock still present in each
  peer after 5 trading days.
</div>
""",
            unsafe_allow_html=True,
        )

        _rip_df = pd.DataFrame(
            _ripple_spillovers,
            columns=["Peer commodity", "IRF coupling (h=5)"],
        )
        _rip_df["Direction"] = _rip_df["IRF coupling (h=5)"].apply(
            lambda v: "↑ Co-moves" if v > 0 else "↓ Counter-moves"
        )
        _rip_df["Strength"] = _rip_df["IRF coupling (h=5)"].apply(
            lambda v: "Strong" if abs(v) > 0.15 else ("Moderate" if abs(v) > 0.07 else "Weak")
        )
        _rip_df["IRF coupling (h=5)"] = _rip_df["IRF coupling (h=5)"].map("{:+.4f}".format)
        st.dataframe(
            _rip_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"{len(_ripple_spillovers)} peer(s) with |IRF h=5| ≥ 0.005 in the "
            f"{_ripple_group} complex. Orthogonalised impulse response functions "
            "via statsmodels VAR."
        )

    st.divider()

    # ── Per-layer detail cards ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Layer-by-layer breakdown")

    # Pre-load IC trend once — used only by the ensemble sparkline.
    # Wrapped in try/except so a missing ic_log table doesn't break the page.
    _ic_trend_df: pd.DataFrame = pd.DataFrame()
    try:
        from models.ic_tracker import ic_trend as _load_ic_trend
        _ic_trend_df = _load_ic_trend(days=60)
    except Exception:
        pass

    layer_map = {n.layer: n for n in result.nodes}

    for layer in LAYER_ORDER:
        node  = layer_map.get(layer)
        icon  = LAYER_ICONS.get(layer, "•")
        label     = LAYER_LABELS.get(layer, layer.title())
        label_md  = label.split("<br>")[0]   # strip HTML gloss for markdown context

        if node:
            d_color = DIRECTION_COLORS.get(node.direction, BLUE)
            header_md = (
                f"{icon} **{label_md}** — {_direction_label(node.direction)} "
                f"· confidence {node.confidence:.0%}"
            )
        else:
            d_color  = "rgba(80,80,100,0.5)"
            header_md = f"{icon} **{label_md}** — *not computed (insufficient data)*"

        # Untrained-ensemble warning card — rendered above the expander so it's
        # immediately visible without expanding the section.
        if layer == "ensemble" and node and node.metadata.get("warning"):
            st.markdown(
                f"""
<div style="
  background:#1C1400;
  border:0.5px solid rgba(245,158,11,0.5);
  border-left:3px solid {AMBER};
  border-radius:8px;
  padding:10px 16px;
  margin-bottom:6px;
  display:flex;
  align-items:flex-start;
  gap:10px;
">
  <span style="font-size:1.1rem;margin-top:1px">⚠️</span>
  <div>
    <div style="font-size:0.82rem;font-weight:600;color:{AMBER};
                letter-spacing:0.04em;margin-bottom:3px">
      Ensemble layer inactive
    </div>
    <div style="font-size:0.80rem;color:rgba(238,242,255,0.70);line-height:1.55">
      {node.metadata['warning']}
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with st.expander(header_md, expanded=(layer in ("trigger", "portfolio"))):
            if not node:
                st.info("This layer was skipped. Common reasons: not enough price history around the trigger date, or macro columns not available.", icon="ℹ️")
                continue

            # Summary
            st.markdown(
                f"""
                <div style="
                    border-left: 3px solid {d_color};
                    padding-left: 12px;
                    margin-bottom: 10px;
                    color: #C0CDE0;
                ">
                {node.summary}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 30-day sparkline — shows the layer's key signal in temporal context
            _spark = _build_layer_sparkline(
                layer, node, _prices, _macro, commodity, tdate,
                d_color, _ic_trend_df,
            )
            if _spark is not None:
                st.plotly_chart(
                    _spark,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            elif layer == "ensemble" and _ic_trend_df.empty:
                st.caption(
                    "Ensemble sparkline unavailable — ic_log is empty. "
                    "Run daily_retrain.py to populate IC history."
                )

            # Before / after values (for statistical layer)
            if node.before_value is not None and node.after_value is not None:
                v1, v2 = st.columns(2)
                with v1:
                    st.metric("Before trigger", f"{node.before_value:.2f}")
                with v2:
                    delta_str = None
                    if node.change_pct is not None:
                        delta_str = f"{node.change_pct:+.1%}"
                    st.metric("After trigger", f"{node.after_value:.2f}", delta=delta_str)

            # Metadata table (filter out big dicts)
            if node.metadata:
                meta_display = {}
                for k, v in node.metadata.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            meta_display[f"{k}.{kk}"] = (
                                f"{vv:.3f}" if isinstance(vv, float) else str(vv)
                            )
                    elif isinstance(v, float):
                        meta_display[k] = f"{v:.4f}" if not (v != v) else "NaN"
                    else:
                        meta_display[k] = str(v)

                if meta_display:
                    st.dataframe(
                        pd.DataFrame.from_dict(
                            meta_display, orient="index", columns=["value"]
                        ).rename_axis("field"),
                        use_container_width=True,
                        height=min(35 * len(meta_display) + 40, 300),
                    )

    st.divider()

    # ── Raw JSON (collapsible) ─────────────────────────────────────────────────
    with st.expander("🗂️ Raw chain JSON", expanded=False):
        st.code(result.to_json(), language="json")


# ── Recent events from trigger log ────────────────────────────────────────────
st.subheader("📋 Recent trigger events (from log)")
st.caption("Events detected by the trigger detectors and saved to the local SQLite log.")

if st.button("🔄 Load recent events", key="load_recent_events"):
    st.session_state["load_recent_requested"] = True

if st.session_state.get("load_recent_requested"):
    try:
        from models.trigger_log import recent_trigger_events
        _log_df = recent_trigger_events(days=60)
    except Exception as e:
        _log_df = pd.DataFrame()
        st.warning(f"Could not read trigger log: {e}")

    if _log_df.empty:
        st.info(
            "No trigger events found in the last 60 days. "
            "Events are logged automatically when the Models page runs `detect_all()`. "
            "Open the **Models** page to populate the log.",
            icon="📭",
        )
    else:
        # Pretty-format for display
        _display = _log_df.copy()
        if "strength" in _display.columns:
            _display["strength"] = _display["strength"].apply(lambda x: f"{x:.0%}")
        st.dataframe(
            _display,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"{len(_log_df)} event(s) found. "
            "Use the controls above to trace any of these — "
            "enter the family name and date manually."
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
      See what macro conditions set the stage for these triggers →
      <b style="color:#EEF2FF">Macro-Market Cascade</b>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.page_link("pages/7_Macro_Market_Cascade.py", label="→ Open Macro-Market Cascade")
