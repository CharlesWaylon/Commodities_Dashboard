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
    "trigger":     "① Trigger",
    "statistical": "② Vol Shift",
    "regime":      "③ Regime",
    "ensemble":    "④ Ensemble",
    "portfolio":   "⑤ Portfolio",
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


# ── Sankey builder ─────────────────────────────────────────────────────────────

def build_sankey(nodes: list, height: int = 380) -> go.Figure:
    """
    Build a Plotly Sankey diagram for the 5-layer causal chain.

    Each link width is proportional to the *destination* node's confidence.
    Link colour matches the destination node's direction.
    Missing layers are shown dimmed in grey.
    """
    layer_map = {n.layer: n for n in nodes}

    # Build node specs
    n_labels, n_colors, n_customdata = [], [], []
    for layer in LAYER_ORDER:
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

    # Build link specs (sequential)
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
            # Fixed horizontal positions (left → right)
            x=[0.02, 0.27, 0.50, 0.73, 0.97],
            y=[0.50,  0.50, 0.50, 0.50, 0.50],
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
        font=dict(color="#E0E0E0", size=13),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.divider()
    st.page_link("app.py",              label="Home")
    st.page_link("pages/1_Pricing.py",  label="Pricing")
    st.page_link("pages/2_Charts.py",   label="Charts")
    st.page_link("pages/3_News.py",     label="News")
    st.page_link("pages/4_Models.py",   label="Models")
    st.page_link("pages/5_Database.py", label="Database")
    st.page_link("pages/6_Causal.py",   label="Causal Chain")
    st.divider()


# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔗 Causal Chain Explorer")
st.caption(
    "Trace how a macro trigger ripples through the model ecosystem: "
    "vol shift → regime → ensemble → portfolio recommendation."
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


# ── Trace output ───────────────────────────────────────────────────────────────
if not st.session_state.get("causal_trace_requested"):
    st.info(
        "👆 Configure a trigger above and click **⚡ Trace Chain** to see how "
        "the signal propagates through the model ecosystem.",
        icon="🔗",
    )
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

    # ── Sankey diagram ─────────────────────────────────────────────────────────
    st.plotly_chart(
        build_sankey(result.nodes, height=400),
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
                <div style="margin-top:10px; opacity:0.6; font-size:0.82rem;">
                    Confidence: {port_node.confidence:.0%}
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Portfolio layer not computed.")

    st.divider()

    # ── Per-layer detail cards ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Layer-by-layer breakdown")

    layer_map = {n.layer: n for n in result.nodes}

    for layer in LAYER_ORDER:
        node  = layer_map.get(layer)
        icon  = LAYER_ICONS.get(layer, "•")
        label = LAYER_LABELS.get(layer, layer.title())

        if node:
            d_color = DIRECTION_COLORS.get(node.direction, BLUE)
            header_md = (
                f"{icon} **{label}** — {_direction_label(node.direction)} "
                f"· confidence {node.confidence:.0%}"
            )
        else:
            d_color  = "rgba(80,80,100,0.5)"
            header_md = f"{icon} **{label}** — *not computed (insufficient data)*"

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
