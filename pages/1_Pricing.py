"""
Pricing page — sortable, filterable table of all commodity prices.

This page is the "data analyst" view: you can sort by any column,
filter by sector, and see both absolute and percentage changes at a glance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from services.price_data import fetch_current_prices
from utils.formatting import sector_emoji
from utils.theme import apply_theme, render_topbar

st.set_page_config(page_title="Accendio | Pricing", page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()

with st.spinner(""):
    df = fetch_current_prices()

render_topbar(df)

with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.divider()
    st.page_link("app.py",              label="Home")
    st.page_link("pages/1_Pricing.py",  label="Pricing")
    st.page_link("pages/2_Charts.py",   label="Charts")
    st.page_link("pages/3_News.py",     label="News")
    st.page_link("pages/4_Models.py",   label="Models")
    st.page_link("pages/5_Database.py", label="Database")
    st.divider()
    if st.button("↻  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.title("Pricing")
st.caption("Sorted, filtered pricing data for all tracked commodities")

if df.empty:
    st.error("Could not load price data.")
    st.stop()

# ── Filters ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    sectors = ["All"] + sorted(df["Sector"].unique().tolist())
    selected_sector = st.selectbox("Filter by Sector", sectors)

with col2:
    movement = st.selectbox("Filter by Movement", ["All", "Gainers", "Losers", "Unchanged"])

with col3:
    sort_by = st.selectbox("Sort by", ["Name", "Pct_Change", "Price", "Change", "Sector"])

# Apply filters
filtered = df.copy()
if selected_sector != "All":
    filtered = filtered[filtered["Sector"] == selected_sector]
if movement == "Gainers":
    filtered = filtered[filtered["Pct_Change"] > 0]
elif movement == "Losers":
    filtered = filtered[filtered["Pct_Change"] < 0]
elif movement == "Unchanged":
    filtered = filtered[filtered["Pct_Change"] == 0]

ascending = sort_by == "Name"
filtered = filtered.sort_values(sort_by, ascending=ascending)

st.divider()

# ── Summary Strip ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Showing",        f"{len(filtered)} commodities")
c2.metric("Avg Daily Move", f"{filtered['Pct_Change'].mean():+.2f}%")
c3.metric("Largest Gain",   f"{filtered['Pct_Change'].max():+.2f}%")
c4.metric("Largest Loss",   f"{filtered['Pct_Change'].min():+.2f}%")

st.divider()

# ── Styled DataFrame ───────────────────────────────────────────────────────────
def highlight_change(val):
    """Color-code the Pct_Change column."""
    try:
        num = float(str(val).replace("%", "").replace("+", ""))
        if num > 0:
            return "color: #00C851; font-weight: bold"
        elif num < 0:
            return "color: #ff4444; font-weight: bold"
    except Exception:
        pass
    return ""

display = filtered[["Name", "Sector", "Price", "Unit", "Change", "Pct_Change"]].copy()

# Format numbers for display
display["Price"]      = display.apply(lambda r: f"{r['Price']:,.4f}", axis=1)
display["Change"]     = display["Change"].apply(lambda x: f"{x:+.4f}")
display["Pct_Change"] = display["Pct_Change"].apply(lambda x: f"{x:+.2f}%")
display["Sector"]     = display["Sector"].apply(lambda s: f"{sector_emoji(s)} {s}")

display.columns = ["Commodity", "Sector", "Price", "Unit", "Change", "% Change"]

styled = display.style.map(highlight_change, subset=["% Change"])
st.dataframe(styled, width='stretch', hide_index=True, height=500)

st.divider()

# ── Heatmap by Sector ──────────────────────────────────────────────────────────
st.subheader("Performance Heatmap")
st.caption("Daily % change across all commodities — green = up, red = down")

heatmap_df = filtered[["Name", "Sector", "Pct_Change"]].copy()

fig = px.treemap(
    heatmap_df,
    path=["Sector", "Name"],
    values=heatmap_df["Pct_Change"].abs() + 0.1,  # size by magnitude, +0.1 so zeros are visible
    color="Pct_Change",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=0,
    hover_data={"Pct_Change": ":.2f"},
    title="Commodity Performance Treemap",
)
fig.update_layout(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font_color="#FAFAFA",
    margin=dict(t=50, l=10, r=10, b=10),
)
fig.update_traces(
    textfont_size=14,
    hovertemplate="<b>%{label}</b><br>Daily Change: %{color:.2f}%<extra></extra>",
)
st.plotly_chart(fig, width='stretch')
