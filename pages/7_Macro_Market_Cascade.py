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
SECTORS: Dict[str, List[str]] = {
    "Energy":    ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas (Henry Hub)"],
    "Metals":    ["Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)"],
    "Grains":    ["Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)"],
    "Livestock": ["Feeder Cattle", "Lean Hogs"],
}

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

# ── Causal relationship matrix ─────────────────────────────────────────────────
# Effect of each macro channel on 1-week commodity return, per unit of signal.
# Signals are normalised to [−1, +1]; effects are in percentage points.
#   real_yields: TLT momentum inverted (rising rates = real yields up)
#   usd:         DXY z-score (positive = strong dollar)
#   risk_off:    scaled VIX above 15 (positive = risk-off, flight to safety)
MACRO_EFFECTS: Dict[str, Dict[str, float]] = {
    # Energy — hurt by strong dollar; mild risk-off sensitivity
    "WTI Crude Oil":           {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.40},
    "Brent Crude Oil":         {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.40},
    "Natural Gas (Henry Hub)": {"real_yields":  0.00, "usd": -0.20, "risk_off": -0.20},
    # Metals — gold safe-haven offsets USD headwind; copper cyclical
    "Gold (COMEX)":            {"real_yields": -0.60, "usd":  0.30, "risk_off":  0.50},
    "Silver (COMEX)":          {"real_yields": -0.40, "usd": -0.10, "risk_off":  0.25},
    "Copper (COMEX)":          {"real_yields":  0.10, "usd": -0.30, "risk_off": -0.50},
    # Grains — USD-sensitive, weak risk-off link
    "Corn (CBOT)":             {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.25},
    "Wheat (CBOT SRW)":        {"real_yields":  0.00, "usd": -0.25, "risk_off": -0.20},
    "Soybeans (CBOT)":         {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.25},
    # Livestock — demand-led; sensitive to risk-off via feed cost / consumption
    "Feeder Cattle":           {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.30},
    "Lean Hogs":               {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.30},
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(period: str = "2y") -> pd.DataFrame:
    from features.macro_overlays import build_macro_overlay_features
    try:
        return build_macro_overlay_features(period=period)
    except Exception:
        return pd.DataFrame()


# ── Macro state ────────────────────────────────────────────────────────────────

def get_macro_state(macro_df: pd.DataFrame) -> Dict:
    """Extract the latest macro state, returning a normalised signal dict."""
    if macro_df is None or macro_df.empty:
        return {
            "dxy_zscore": 0.0, "vix": 18.0, "tlt_mom21": 0.0,
            "real_yields_signal": 0.0, "usd_signal": 0.0, "risk_off_signal": 0.0,
            "risk_off": False, "crisis": False,
            "date": str(date.today()),
        }
    row   = macro_df.iloc[-1]
    dxy_z = float(row.get("dxy_zscore63", 0.0) or 0.0)
    vix   = float(row.get("vix", 18.0) or 18.0)
    tlt21 = float(row.get("tlt_mom21", 0.0) or 0.0)

    # Normalised signals — positive = headwind direction named by label
    real_yields_signal = max(-1.0, min(1.0, -tlt21 / 0.05))
    usd_signal         = max(-1.0, min(1.0,  dxy_z / 2.0))
    risk_off_signal    = max(0.0,  min(1.0,  (vix - 15.0) / 20.0))

    return {
        "dxy_zscore":          round(dxy_z, 3),
        "vix":                 round(vix, 1),
        "tlt_mom21":           round(tlt21, 4),
        "real_yields_signal":  round(real_yields_signal, 3),
        "usd_signal":          round(usd_signal, 3),
        "risk_off_signal":     round(risk_off_signal, 3),
        "risk_off":            vix >= 20.0,
        "crisis":              vix >= 30.0,
        "date": str(macro_df.index[-1].date()),
    }


# ── Forecast computation ───────────────────────────────────────────────────────

def compute_forecasts(prices: pd.DataFrame, macro_state: Dict) -> pd.DataFrame:
    """
    Return a DataFrame with one row per commodity:
      commodity, sector, base_pct, macro_adj_pct, final_pct, uncertainty_band
    All values are 1-week (5 trading day) forecasts in percentage points.
    """
    ry  = macro_state["real_yields_signal"]
    usd = macro_state["usd_signal"]
    ro  = macro_state["risk_off_signal"]

    rows = []
    for sector, comms in SECTORS.items():
        for comm in comms:
            if comm in prices.columns and len(prices[comm].dropna()) >= 22:
                log_ret  = np.log(prices[comm] / prices[comm].shift(1)).dropna()
                tail21   = log_ret.tail(21)
                base_pct = float(tail21.mean() * 5 * 100)
                std21    = float(tail21.std() * np.sqrt(5) * 100)
            else:
                base_pct = 0.0
                std21    = 1.0

            efx = MACRO_EFFECTS.get(comm, {})
            macro_adj = (
                efx.get("real_yields", 0.0) * ry  +
                efx.get("usd",         0.0) * usd +
                efx.get("risk_off",    0.0) * ro
            )
            final = base_pct + macro_adj
            rows.append({
                "commodity":        comm,
                "sector":           sector,
                "base_pct":         round(base_pct, 3),
                "macro_adj_pct":    round(macro_adj, 3),
                "final_pct":        round(final, 3),
                "uncertainty_band": round(1.65 * std21, 3),
            })
    return pd.DataFrame(rows)


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


# ── Narrative ──────────────────────────────────────────────────────────────────

def build_narrative(macro_state: Dict, forecasts: pd.DataFrame) -> str:
    ry   = macro_state["real_yields_signal"]
    usd  = macro_state["usd_signal"]
    ro   = macro_state["risk_off_signal"]
    vix  = macro_state["vix"]
    dxyz = macro_state["dxy_zscore"]
    tlt  = macro_state["tlt_mom21"]
    crisis   = macro_state["crisis"]
    risk_off = macro_state["risk_off"]

    paras = []

    # Macro regime headline
    if crisis:
        regime_lbl = "crisis conditions (VIX ≥ 30)"
    elif risk_off:
        regime_lbl = "risk-off conditions (VIX ≥ 20)"
    else:
        regime_lbl = "risk-on conditions"
    paras.append(
        f"**Macro environment ({macro_state['date']}):** Markets are in {regime_lbl} "
        f"with VIX at {vix:.1f}. "
        f"The DXY z-score stands at {dxyz:+.2f} ({'above' if dxyz > 0 else 'below'} "
        f"its 63-day average) and TLT 21-day momentum is "
        f"{'positive' if tlt > 0 else 'negative'} at {tlt*100:+.1f}%."
    )

    # Real yields channel
    if abs(ry) > 0.15:
        dir_str = "rising" if ry > 0 else "falling"
        gold_row = forecasts[forecasts["commodity"] == "Gold (COMEX)"]
        gold_adj = f"{gold_row.iloc[0]['macro_adj_pct']:+.2f}%" if not gold_row.empty else "N/A"
        cu_row   = forecasts[forecasts["commodity"] == "Copper (COMEX)"]
        cu_adj   = f"{cu_row.iloc[0]['macro_adj_pct']:+.2f}%" if not cu_row.empty else "N/A"
        paras.append(
            f"**Real yields channel:** Real yields are {dir_str} (TLT 21d momentum "
            f"{tlt*100:+.1f}%), "
            + (f"creating a headwind for Gold (macro adjustment: {gold_adj}) "
               f"as the opportunity cost of holding bullion rises. "
               f"Industrial Copper is less sensitive to rate changes "
               f"(macro adjustment: {cu_adj})."
               if ry > 0.15 else
               f"acting as a tailwind for Gold (macro adjustment: {gold_adj}) "
               f"and other precious metals.")
        )

    # USD channel
    if abs(usd) > 0.15:
        dir_str  = "strong" if usd > 0 else "weak"
        wti_row  = forecasts[forecasts["commodity"] == "WTI Crude Oil"]
        wti_adj  = f"{wti_row.iloc[0]['macro_adj_pct']:+.2f}%" if not wti_row.empty else "N/A"
        paras.append(
            f"**USD channel:** The dollar is {dir_str} (DXY z-score {dxyz:+.2f}). "
            + (f"As most commodity futures are USD-denominated, "
               f"non-US buyers face higher costs — WTI receives a macro adjustment of {wti_adj} "
               f"and most Agricultural prices face similar pressure."
               if usd > 0.15 else
               f"A weak dollar provides broad support for USD-denominated commodity prices; "
               f"WTI macro adjustment: {wti_adj}.")
        )

    # Risk sentiment channel
    if ro > 0.15:
        cu_row  = forecasts[forecasts["commodity"] == "Copper (COMEX)"]
        cu_adj  = f"{cu_row.iloc[0]['macro_adj_pct']:+.2f}%" if not cu_row.empty else "N/A"
        gld_row = forecasts[forecasts["commodity"] == "Gold (COMEX)"]
        gld_adj = f"{gld_row.iloc[0]['macro_adj_pct']:+.2f}%" if not gld_row.empty else "N/A"
        paras.append(
            f"**Risk sentiment:** VIX at {vix:.1f} signals risk-off positioning. "
            f"Industrial commodities face demand destruction headwinds — "
            f"Copper macro adjustment: {cu_adj}. "
            f"Gold benefits from safe-haven flows (macro adjustment: {gld_adj})."
        )

    # Sector-level summaries
    sector_summaries = []
    for sector in SECTORS:
        df_s     = forecasts[forecasts["sector"] == sector]
        if df_s.empty:
            continue
        avg_final = df_s["final_pct"].mean()
        avg_base  = df_s["base_pct"].mean()
        adj       = avg_final - avg_base
        dir_word  = "adds" if adj > 0 else "subtracts"
        sector_summaries.append(
            f"{sector} ({dir_word} {abs(adj):.2f}pp → {avg_final:+.2f}%)"
        )
    if sector_summaries:
        paras.append(
            f"**Sector-level macro overlay:** " + "; ".join(sector_summaries) + "."
        )

    # Overall conclusion
    sector_avgs = forecasts.groupby("sector")["final_pct"].mean()
    if not sector_avgs.empty:
        best  = sector_avgs.idxmax()
        worst = sector_avgs.idxmin()
        paras.append(
            f"**Overall:** Under current macro conditions, **{best}** is the "
            f"most favoured sector (avg 1-wk forecast: {sector_avgs[best]:+.2f}%). "
            f"**{worst}** faces the greatest headwinds "
            f"(avg forecast: {sector_avgs[worst]:+.2f}%). "
            f"These are model-generated outlooks; macro signal quality depends on "
            f"the freshness of DXY, VIX, and TLT data."
        )

    return "\n\n".join(paras)


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
