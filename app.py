"""
Commodities Dashboard — main entry point.

Streamlit multipage apps use the `pages/` directory convention:
  - app.py        → the home/landing page
  - pages/1_*.py  → numbered pages shown in the sidebar in order

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from services.price_data import fetch_current_prices, COMMODITY_SECTORS
from utils.formatting import color_delta, sector_emoji, format_price

st.set_page_config(
    page_title="Commodities Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Commodities Hub")
    st.caption("Future of Commodities Club")
    st.divider()
    st.markdown("""
**Navigation**
- 🏠 **Overview** ← you are here
- 💰 **Pricing** — full price table
- 📊 **Charts** — interactive price history
- 📰 **News** — live market news
- 🤖 **Models** — analytics pipeline
    """)
    st.divider()
    st.caption("Data: Yahoo Finance · Refreshes every 5 min")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏠 Market Overview")
st.caption("Live commodity prices across Energy, Metals, Agriculture & Livestock")

# ── Load Data ──────────────────────────────────────────────────────────────────
with st.spinner("Fetching live prices..."):
    df = fetch_current_prices()

if df.empty:
    st.error("Could not load price data. Check your internet connection.")
    st.stop()

# ── Market Snapshot KPIs ───────────────────────────────────────────────────────
st.subheader("Market Snapshot")

gainers = df[df["Pct_Change"] > 0]
losers  = df[df["Pct_Change"] < 0]
best    = df.loc[df["Pct_Change"].idxmax()]
worst   = df.loc[df["Pct_Change"].idxmin()]

k1, k2, k3, k4 = st.columns(4)
k1.metric("📦 Commodities Tracked", len(df))
k2.metric("🟢 Advancing", len(gainers), f"{len(gainers)/len(df)*100:.0f}% of market")
k3.metric("🏆 Top Gainer", best["Name"],
          f"{best['Pct_Change']:+.2f}%", delta_color="normal")
k4.metric("📉 Top Loser", worst["Name"],
          f"{worst['Pct_Change']:+.2f}%", delta_color="inverse")

st.divider()

# ── Price Cards by Sector ──────────────────────────────────────────────────────
sectors = df["Sector"].unique()

for sector in ["Energy", "Metals", "Agriculture", "Livestock"]:
    sector_df = df[df["Sector"] == sector]
    if sector_df.empty:
        continue

    st.subheader(f"{sector_emoji(sector)} {sector}")
    cols = st.columns(min(len(sector_df), 5))

    for i, (_, row) in enumerate(sector_df.iterrows()):
        col = cols[i % len(cols)]
        delta_str = f"{row['Pct_Change']:+.2f}%"
        col.metric(
            label=row["Name"],
            value=f"{row['Price']:,.2f} {row['Unit']}",
            delta=delta_str,
        )

    st.divider()

# ── Quick Table ────────────────────────────────────────────────────────────────
with st.expander("📋 Full Price Table", expanded=False):
    display = df[["Name", "Sector", "Price", "Unit", "Change", "Pct_Change"]].copy()
    display["Pct_Change"] = display["Pct_Change"].apply(lambda x: f"{x:+.2f}%")
    display["Change"]     = display["Change"].apply(lambda x: f"{x:+.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
