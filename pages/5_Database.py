"""
Database Inspector page — see exactly what's stored in the database.

This page is your window into the pipeline. It shows:
  - How many rows are stored, for which commodities, across what date range
  - The raw stored data for any commodity (so you can verify it looks right)
  - A chart of stored close prices (using DB data, NOT live yfinance)
  - Tools to trigger a manual ingestion run or backfill from the UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Database | Commodities", page_icon="🗄️", layout="wide")

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
- 🤖 [Models](/Models)
- 🗄️ **Database** ← you are here
    """)
    st.divider()

st.title("🗄️ Database Inspector")
st.caption("Inspect the local data pipeline — what's stored, when it was ingested, and trigger manual runs")

# ── Try to connect ─────────────────────────────────────────────────────────────
try:
    from database.db import init_db, get_db, db_info
    from database.models import Commodity, PriceHistory
    from pipeline.ingest import run_ingestion
    init_db()  # Ensure tables exist
    info = db_info()
    db_available = True
except Exception:
    st.info(
        "🗄️ **Database Inspector is only available when running locally.**\n\n"
        "This page connects to a local SQLite database that lives on your machine. "
        "When viewing the dashboard online, live prices and charts are still fully "
        "functional — this inspector is just for pipeline management.\n\n"
        "To use this page, run `streamlit run app.py` on your own computer."
    )
    st.stop()

# ── DB Status ──────────────────────────────────────────────────────────────────

st.subheader("📊 Database Status")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Engine",          "SQLite" if "sqlite" in info["url"] else "PostgreSQL")
c2.metric("Commodities",     info["commodities"])
c3.metric("Price Rows",      f"{info['price_rows']:,}")
c4.metric("Oldest Date",     str(info["oldest_date"]) if info["oldest_date"] else "—")
c5.metric("Newest Date",     str(info["newest_date"]) if info["newest_date"] else "—")

if info["price_rows"] == 0:
    st.warning("⚠️ Database is empty. Run an ingestion below to populate it.")

st.caption(f"DB location: `{info['url']}`")
st.divider()

# ── Manual Ingestion Trigger ───────────────────────────────────────────────────
st.subheader("⚙️ Manual Ingestion")
st.markdown("""
You can trigger a data pull right from this page without using the terminal.
- **Incremental** — pulls the last 5 days only (fast, good for daily top-ups)
- **Full Backfill** — pulls 5 years of history (slow first time, ~2–3 minutes)
""")

col_inc, col_back = st.columns(2)

with col_inc:
    if st.button("▶️ Run Incremental Ingestion", use_container_width=True):
        with st.spinner("Running incremental ingestion (last 5 days)..."):
            try:
                inserted, skipped = run_ingestion(backfill=False)
                st.success(f"Done! Inserted {inserted:,} new rows, skipped {skipped:,} duplicates.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

with col_back:
    if st.button("⏮️ Run Full Backfill (5 years)", use_container_width=True,
                 type="secondary"):
        confirm = st.checkbox("I understand this will take 2–3 minutes")
        if confirm:
            with st.spinner("Running full backfill — pulling 5 years of history..."):
                try:
                    inserted, skipped = run_ingestion(backfill=True)
                    st.success(f"Backfill complete! Inserted {inserted:,} rows, skipped {skipped:,} duplicates.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Backfill failed: {e}")

st.divider()

# ── Per-Commodity Coverage ─────────────────────────────────────────────────────
st.subheader("📋 Coverage by Commodity")
st.caption("How many stored rows exist per commodity, and what date range they cover")

if info["price_rows"] > 0:
    with get_db() as db:
        rows = (
            db.query(
                Commodity.name,
                Commodity.sector,
                Commodity.ticker,
                PriceHistory.date,
                PriceHistory.close,
            )
            .join(PriceHistory, Commodity.id == PriceHistory.commodity_id)
            .all()
        )

    df = pd.DataFrame(rows, columns=["Name", "Sector", "Ticker", "Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])

    coverage = (
        df.groupby(["Name", "Sector", "Ticker"])
        .agg(
            Rows       = ("Close", "count"),
            From       = ("Date", "min"),
            To         = ("Date", "max"),
            Latest_Close = ("Close", "last"),
        )
        .reset_index()
        .sort_values("Sector")
    )
    coverage["From"] = coverage["From"].dt.strftime("%Y-%m-%d")
    coverage["To"]   = coverage["To"].dt.strftime("%Y-%m-%d")
    coverage["Latest_Close"] = coverage["Latest_Close"].apply(lambda x: f"{x:,.4f}")

    st.dataframe(coverage, use_container_width=True, hide_index=True)
    st.divider()

    # ── Raw Data Viewer ────────────────────────────────────────────────────────
    st.subheader("🔍 Raw Data Viewer")
    commodities_in_db = sorted(df["Name"].unique().tolist())
    selected = st.selectbox("Select commodity to inspect", commodities_in_db)

    sub = df[df["Name"] == selected].sort_values("Date", ascending=False)

    col_tbl, col_chart = st.columns([2, 3])
    with col_tbl:
        st.caption(f"Last 30 rows stored for **{selected}**")
        display = sub.head(30)[["Date", "Close"]].copy()
        display["Date"]  = display["Date"].dt.strftime("%Y-%m-%d")
        display["Close"] = display["Close"].apply(lambda x: f"{x:,.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    with col_chart:
        st.caption(f"All stored close prices for **{selected}**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sub["Date"].sort_values(),
            y=sub.sort_values("Date")["Close"],
            mode="lines",
            line=dict(color="#F5A623", width=1.5),
            name="Close (from DB)",
        ))
        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1C2333",
            font_color="#FAFAFA",
            height=300,
            margin=dict(t=20, l=50, r=20, b=40),
            xaxis=dict(gridcolor="#2C3347"),
            yaxis=dict(gridcolor="#2C3347"),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No data in the database yet. Use the ingestion buttons above to populate it.")
