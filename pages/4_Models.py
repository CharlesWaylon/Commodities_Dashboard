"""
Models page — analytics, pipeline roadmap, and a working demo model.

Phase 1 (now):     Simple statistical analysis on live data
Phase 2 (soon):    Data ingestion pipeline → PostgreSQL storage
Phase 3 (future):  Predictive models (Prophet, ARIMA, LSTM, XGBoost)

This page documents the roadmap and provides a working demo of
a basic statistical model you can run right now on live data.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services.price_data import COMMODITY_TICKERS, COMMODITY_IS_PROXY, fetch_historical
from models.config import MODELING_COMMODITIES

st.set_page_config(page_title="Models | Commodities", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("📈 Commodities Hub")
    st.caption("Future of Commodities Club")
    st.divider()
    st.markdown("""
**Navigation**
- 🏠 [Overview](/)
- 💰 [Pricing](/Pricing)
- 📊 [Charts](/Charts)
- 📰 [News](/News)
- 🤖 **Models** ← you are here
    """)
    st.divider()

    # ── Signal / data-source coverage ─────────────────────────────────────────
    st.markdown("**📡 Active Data Sources**")

    # Price data — always active if we got this far
    st.caption("🟢 Yahoo Finance (yfinance) — prices")

    # Database pipeline
    _db_active = False
    try:
        from database.db import db_info as _db_info
        _dbstats = _db_info()
        _db_active = _dbstats.get("price_rows", 0) > 0
    except Exception:
        pass
    st.caption(
        ("🟢 Database pipeline — " + f"{_dbstats['price_rows']:,} rows stored")
        if _db_active
        else "🔴 Database pipeline — not populated (run pipeline/ingest.py)"
    )

    # Optional enrichment keys
    _news_key     = bool(os.getenv("NEWS_API_KEY"))
    _alphavantage = bool(os.getenv("ALPHA_VANTAGE_KEY"))
    st.caption(
        "🟢 NewsAPI key — configured" if _news_key
        else "🔴 NewsAPI key — not set (RSS-only news)"
    )
    st.caption(
        "🟢 Alpha Vantage key — configured" if _alphavantage
        else "🔴 Alpha Vantage — not set (yfinance-only prices)"
    )

    st.divider()
    st.markdown("**⚗️ Model status**")
    st.caption("🟢 Statistical analysis — live (this page)")
    st.caption("🟡 Walk-forward CV — implemented (CLI only)")
    st.caption("🔴 ARIMA / Prophet / XGBoost — Phase 3 roadmap")
    st.caption("⚠️ Quantum hybrid — classical simulator only (PennyLane `default.qubit`)")
    st.divider()

st.title("🤖 Analytics & Models")
st.caption("Statistical analysis now — predictive pipeline coming soon")

# ── Pipeline Roadmap ───────────────────────────────────────────────────────────
st.subheader("🗺️ Pipeline Roadmap")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
**Phase 1 — Display Layer** ✅
> *You are here*

- Live prices via Yahoo Finance
- Interactive charts with Plotly
- RSS news aggregation
- Statistical analysis on live data

**Tools:** Python, Streamlit, yfinance
    """)
with col2:
    st.markdown("""
**Phase 2 — Data Pipeline** 🔜

- Scheduled data ingestion scripts
- Raw OHLCV stored in PostgreSQL
- Data cleaning & normalization
- Feature engineering (rolling stats, spreads)
- REST API to serve stored data

**Tools:** Python (pandas), PostgreSQL,
SQLAlchemy, Apache Airflow / cron
    """)
with col3:
    st.markdown("""
**Phase 3 — Predictive Models** 🔮

- Time-series forecasting (Prophet, ARIMA)
- Regime detection (HMM, clustering)
- Correlation & spread analysis
- Backtesting framework
- Model results fed back into this dashboard

**Tools:** scikit-learn, statsmodels,
Facebook Prophet, PyTorch (LSTM)
    """)

st.divider()

# ── Live Demo: Statistical Analysis ───────────────────────────────────────────
st.subheader("📐 Statistical Analysis Demo")
st.caption("Run descriptive and rolling statistics on live price data right now")

dcol1, dcol2 = st.columns([3, 2])
with dcol1:
    commodity = st.selectbox("Commodity", list(COMMODITY_TICKERS.keys()), index=5)
with dcol2:
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1,
                          format_func=lambda x: {"6mo":"6 Months","1y":"1 Year",
                                                  "2y":"2 Years","5y":"5 Years"}[x])

ticker = COMMODITY_TICKERS[commodity]

if COMMODITY_IS_PROXY.get(commodity, False):
    st.warning(
        f"⚠️ **{commodity}** is an ETF/equity proxy, not a direct futures price. "
        "Statistics here reflect the proxy instrument's equity-market behaviour and may "
        "include stock-market beta unrelated to the underlying commodity."
    )

with st.spinner(f"Loading {commodity} data..."):
    df = fetch_historical(ticker, period=period)

if df.empty or "Close" not in df.columns:
    st.warning("No data available.")
    st.stop()

close   = df["Close"].dropna()
returns = close.pct_change().dropna()

# Surface gaps: if rows with NaN close exist, forward-fill silently drops them
n_raw  = len(df)
n_used = len(close)
if n_used < n_raw * 0.95:
    st.warning(
        f"⚠️ {n_raw - n_used} of {n_raw} rows were dropped (missing close prices). "
        "Statistics below reflect the remaining {n_used} trading days only."
    )

# ── Key Stats ──────────────────────────────────────────────────────────────────
st.markdown("#### Descriptive Statistics")
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Mean Price",   f"${close.mean():,.2f}")
s2.metric("Std Dev",      f"${close.std():,.2f}")
s3.metric("Min",          f"${close.min():,.2f}")
s4.metric("Max",          f"${close.max():,.2f}")
s5.metric("Avg Daily Return", f"{returns.mean()*100:.3f}%")
s6.metric("Daily Volatility", f"{returns.std()*100:.2f}%")

# Annualised stats
ann_vol    = returns.std() * np.sqrt(252) * 100
ann_return = ((close.iloc[-1] / close.iloc[0]) ** (252 / len(close)) - 1) * 100
sharpe     = ann_return / ann_vol if ann_vol != 0 else 0

s7, s8, s9 = st.columns(3)
s7.metric("Annualised Volatility", f"{ann_vol:.1f}%")
s8.metric("Annualised Return",     f"{ann_return:+.1f}%")
s9.metric("Sharpe Ratio (rf=0)",   f"{sharpe:.2f}",
          help="Return / Volatility. Above 1 = good, above 2 = great.")

st.divider()

# ── Rolling Volatility Chart ───────────────────────────────────────────────────
st.markdown("#### Rolling Volatility & Returns")

window = st.slider("Rolling window (days)", 10, 60, 21)

roll_vol    = returns.rolling(window).std() * np.sqrt(252) * 100
roll_return = returns.rolling(window).mean() * 252 * 100

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=["Price", "Rolling Annualised Volatility (%)"],
                    row_heights=[0.6, 0.4], vertical_spacing=0.08)

fig.add_trace(go.Scatter(x=close.index, y=close,
                          line=dict(color="#F5A623", width=1.5),
                          name=commodity), row=1, col=1)

fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol,
                          fill="tozeroy",
                          fillcolor="rgba(79,195,247,0.15)",
                          line=dict(color="#4FC3F7", width=1.5),
                          name=f"{window}d Rolling Vol"), row=2, col=1)

fig.update_layout(
    paper_bgcolor="#0E1117", plot_bgcolor="#1C2333",
    font_color="#FAFAFA", showlegend=True,
    legend=dict(bgcolor="#1C2333"),
    height=500, margin=dict(t=60, l=60, r=20, b=40),
)
fig.update_xaxes(gridcolor="#2C3347")
fig.update_yaxes(gridcolor="#2C3347")
st.plotly_chart(fig, use_container_width=True)

# ── Return Distribution ────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Return Distribution")
st.caption("How frequently does this commodity move by X% in a day?")

import plotly.express as px

hist_fig = px.histogram(
    returns * 100,
    nbins=60,
    labels={"value": "Daily Return (%)"},
    title=f"{commodity} — Daily Return Distribution",
    color_discrete_sequence=["#F5A623"],
)
hist_fig.add_vline(x=0, line_color="#fff", line_dash="dash", opacity=0.5)
hist_fig.update_layout(
    paper_bgcolor="#0E1117", plot_bgcolor="#1C2333",
    font_color="#FAFAFA", showlegend=False,
    height=350, margin=dict(t=60, l=60, r=20, b=40),
)
st.plotly_chart(hist_fig, use_container_width=True)

# ── Correlation Matrix ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🔗 Correlation Matrix")
st.caption("How correlated are commodity returns? High correlation = they move together.")

corr_commodities = st.multiselect(
    "Select commodities",
    list(COMMODITY_TICKERS.keys()),
    default=list(MODELING_COMMODITIES.keys()),
    help="Defaults to direct futures only. ETF/equity proxies (marked *) carry equity-market beta.",
)

if len(corr_commodities) >= 2:
    price_series = {}
    for name in corr_commodities:
        t    = COMMODITY_TICKERS[name]
        hist = fetch_historical(t, period=period)
        if not hist.empty and "Close" in hist.columns:
            price_series[name] = hist["Close"].dropna().pct_change()

    if len(price_series) >= 2:
        corr_df  = pd.DataFrame(price_series).dropna()
        corr_mat = corr_df.corr()

        corr_fig = px.imshow(
            corr_mat,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            text_auto=".2f",
            title="Return Correlation Matrix",
        )
        corr_fig.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            font_color="#FAFAFA",
            height=450, margin=dict(t=60, l=60, r=20, b=40),
        )
        st.plotly_chart(corr_fig, use_container_width=True)
else:
    st.info("Select at least 2 commodities to compute correlations.")

# ── Model inventory ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Model Inventory")
st.caption(
    "What exists in the codebase today vs what is planned. "
    "Run `python -m models.run_experiment` to execute the implemented models from the CLI."
)

st.markdown("""
| Model | Status | Limitation / Note | Run via |
|---|---|---|---|
| Persistence (lag-1 return) | ✅ Implemented | Naive baseline only | `models/run_experiment.py` |
| Rolling Mean (5-day) | ✅ Implemented | Naive baseline only | `models/run_experiment.py` |
| **Quantum Hybrid Regressor** | ⚠️ Prototype | **Classical simulator** — uses PennyLane `default.qubit`, not real quantum hardware. O(n²) runtime; limited to ~50 training samples. Not a production model. | `models/run_experiment.py` |
| Walk-forward CV | ✅ Implemented | 5-fold anchored CV added to `models/data_loader.py`; not yet wired into the UI | `models/data_loader.py` |
| Rolling Z-Score | 🔜 Phase 3 | Planned mean-reversion signal | — |
| ARIMA | 🔜 Phase 3 | Grid search parameters TBD; requires ≥50 observations per series | — |
| Prophet | 🔜 Phase 3 | Trend + seasonality decomposition | — |
| Hidden Markov Model | 🔜 Phase 3 | Market regime detection (bull/bear) | — |
| LSTM Neural Net | 🔜 Phase 3 | Requires Phase 2 database for sufficient training data | — |
| XGBoost | 🔜 Phase 3 | Feature-based price direction classification | — |
""")

with st.expander("⚠️ Known data pipeline caveats"):
    st.markdown("""
**Forward-fill limit (3 bars):** `models/data_loader.py` forward-fills missing price bars up
to a limit of 3 days before dropping the row entirely. Any commodity with a data gap wider than
3 consecutive trading days will have those rows silently removed from the feature matrix.
This can cause the training set to shrink unexpectedly for thinly-traded instruments
(e.g., Rough Rice, Oats). No warning is currently raised — check row counts in experiment output.

**Quantum model scope:** The quantum hybrid regressor in `models/quantum/` runs on PennyLane's
`default.qubit` classical simulator. It has **not** been connected to IBM Quantum, AWS Braket,
or any other real quantum backend. Runtime is O(n²) in the number of training samples — the
default cap of 50 training samples exists to keep it tractable on a laptop. Treat as a
research prototype, not a production forecasting tool.

**Single train-test split (legacy):** The original 80/20 chronological split in
`models/run_experiment.py` tests only one market regime. Walk-forward cross-validation
(`models/data_loader.walk_forward_splits`) was added to address this but is not yet
integrated into the experiment runner — Spearman IC scores from the current CLI output
are optimistic.
    """)
