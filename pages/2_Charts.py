"""
Charts page — interactive historical price charts with technical overlays.

Uses Plotly for rich interactivity: zoom, pan, hover tooltips, range selectors.
Includes candlestick + volume charts and simple moving averages (SMA).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.price_data import COMMODITY_TICKERS, COMMODITY_UNITS, fetch_historical

st.set_page_config(page_title="Charts | Commodities", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("📈 Commodities Hub")
    st.caption("Future of Commodities Club")
    st.divider()
    st.markdown("""
**Navigation**
- 🏠 [Overview](/)
- 💰 [Pricing](/Pricing)
- 📊 **Charts** ← you are here
- 📰 [News](/News)
- 🤖 [Models](/Models)
    """)
    st.divider()

st.title("📊 Price Charts")
st.caption("Interactive historical price charts for all tracked commodities")

# ── Controls ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

with col1:
    commodity = st.selectbox("Select Commodity", list(COMMODITY_TICKERS.keys()))

with col2:
    period = st.selectbox("Time Period", {
        "1 Week":  "5d",
        "1 Month": "1mo",
        "3 Months":"3mo",
        "6 Months":"6mo",
        "1 Year":  "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }.keys())
    period_map = {
        "1 Week":  "5d",
        "1 Month": "1mo",
        "3 Months":"3mo",
        "6 Months":"6mo",
        "1 Year":  "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }
    period_code = period_map[period]

with col3:
    chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Area"])

with col4:
    show_sma = st.multiselect("Moving Averages", [20, 50, 200], default=[20, 50])

ticker = COMMODITY_TICKERS[commodity]
unit   = COMMODITY_UNITS.get(commodity, "USD")

# ── Fetch Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {commodity} history..."):
    # Use weekly interval for long periods to reduce data points
    interval = "1wk" if period_code in ("2y", "5y") else "1d"
    df = fetch_historical(ticker, period=period_code, interval=interval)

if df.empty:
    st.warning("No historical data available for this commodity.")
    st.stop()

# Ensure we have OHLCV columns
required = ["Open", "High", "Low", "Close"]
missing  = [c for c in required if c not in df.columns]
if missing:
    st.warning(f"Missing columns: {missing}")
    st.stop()

# ── Compute SMAs ───────────────────────────────────────────────────────────────
for window in show_sma:
    if len(df) >= window:
        df[f"SMA{window}"] = df["Close"].rolling(window).mean()

# ── Build Chart ────────────────────────────────────────────────────────────────
has_volume = "Volume" in df.columns and df["Volume"].sum() > 0

if has_volume:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
else:
    fig = make_subplots(rows=1, cols=1)

price_row = 1

# Main price trace
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=commodity,
        increasing_line_color="#00C851",
        decreasing_line_color="#ff4444",
    ), row=price_row, col=1)

elif chart_type == "Area":
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        fill="tozeroy",
        fillcolor="rgba(245, 166, 35, 0.15)",
        line=dict(color="#F5A623", width=2),
        name=commodity,
    ), row=price_row, col=1)

else:  # Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        line=dict(color="#F5A623", width=2),
        name=commodity,
    ), row=price_row, col=1)

# SMA overlays
sma_colors = {20: "#4FC3F7", 50: "#FF8A65", 200: "#CE93D8"}
for window in show_sma:
    col_name = f"SMA{window}"
    if col_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            line=dict(color=sma_colors.get(window, "#AAA"), width=1.5, dash="dot"),
            name=f"SMA {window}",
        ), row=price_row, col=1)

# Volume bar chart
if has_volume:
    colors = ["#00C851" if c >= o else "#ff4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors,
        name="Volume",
        opacity=0.6,
    ), row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

# Layout
fig.update_layout(
    title=f"{commodity} ({ticker}) — {period}",
    yaxis_title=f"Price ({unit})",
    xaxis_rangeslider_visible=(chart_type == "Candlestick"),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#1C2333",
    font_color="#FAFAFA",
    legend=dict(bgcolor="#1C2333", bordercolor="#444"),
    hovermode="x unified",
    height=550,
    margin=dict(t=60, l=60, r=20, b=40),
)
fig.update_xaxes(gridcolor="#2C3347", zeroline=False)
fig.update_yaxes(gridcolor="#2C3347", zeroline=False)

st.plotly_chart(fig, width='stretch')

# ── Stats Row ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"{commodity} — Period Statistics")

close = df["Close"].dropna()
if not close.empty:
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Current",  f"{close.iloc[-1]:,.4f}")
    s2.metric("Period High", f"{close.max():,.4f}")
    s3.metric("Period Low",  f"{close.min():,.4f}")
    pct_return = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
    s4.metric("Period Return", f"{pct_return:+.2f}%")
    s5.metric("Volatility (σ)", f"{close.pct_change().std() * 100:.2f}%")

# ── Multi-Commodity Comparison ─────────────────────────────────────────────────
st.divider()
st.subheader("📈 Multi-Commodity Comparison")
st.caption("Normalized to 100 at start of period — compare relative performance")

compare_names = st.multiselect(
    "Select commodities to compare",
    list(COMMODITY_TICKERS.keys()),
    default=["Gold (COMEX)", "WTI Crude Oil", "Wheat (CBOT SRW)"],
)

if compare_names:
    comp_fig = go.Figure()
    for name in compare_names:
        t = COMMODITY_TICKERS[name]
        with st.spinner(f"Loading {name}..."):
            hist = fetch_historical(t, period=period_code,
                                    interval="1wk" if period_code in ("2y", "5y") else "1d")
        if hist.empty or "Close" not in hist.columns:
            continue
        series = hist["Close"].dropna()
        # Normalize to 100
        normalized = (series / series.iloc[0]) * 100
        comp_fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized,
            name=name, mode="lines",
        ))

    comp_fig.update_layout(
        title="Normalized Performance (base = 100)",
        yaxis_title="Indexed Price (100 = start)",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1C2333",
        font_color="#FAFAFA",
        legend=dict(bgcolor="#1C2333"),
        hovermode="x unified",
        height=400,
        margin=dict(t=60, l=60, r=20, b=40),
    )
    comp_fig.update_xaxes(gridcolor="#2C3347")
    comp_fig.update_yaxes(gridcolor="#2C3347")
    st.plotly_chart(comp_fig, width='stretch')
