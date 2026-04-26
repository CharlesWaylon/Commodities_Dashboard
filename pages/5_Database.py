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
    from database.db import init_db, get_db, db_info, staleness_info
    from database.models import Commodity, PriceHistory, IngestionLog
    from pipeline.ingest import run_ingestion
    db_available = True
except Exception as e:
    db_available = False
    st.error(f"Could not load database module: {e}")
    st.stop()

# ── DB Status ──────────────────────────────────────────────────────────────────
init_db()  # Ensure tables exist
info = db_info()

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

# ── Staleness check ────────────────────────────────────────────────────────────
if info["price_rows"] > 0:
    stale = staleness_info()
    days  = stale["days_stale"]

    if days is None:
        pass
    elif days <= 1:
        st.success(f"✅ Data is current — newest record: **{stale['newest_date']}**")
    elif days <= 3:
        st.warning(
            f"⚠️ Data is **{days} day(s) old** (newest: {stale['newest_date']}). "
            "This may be a weekend/holiday gap. If today is a weekday, check that the "
            "scheduler (`python -m pipeline.scheduler`) is running."
        )
    else:
        st.error(
            f"🚨 Data is **{days} days stale** (newest: {stale['newest_date']}). "
            "The ingestion pipeline has likely missed multiple runs — check the "
            "Ingestion Log below for errors, then trigger a manual run."
        )

    if stale["per_type"]:
        per_type_lines = "  |  ".join(
            f"{t}: {d}d" for t, d in sorted(stale["per_type"].items()) if t != "unknown"
        )
        st.caption(
            f"Staleness by instrument type — {per_type_lines}.  "
            "Note: equity/ETF proxies may lag futures by 1–2 days on NYSE-only holidays "
            "(e.g. MLK Day, Juneteenth) where NYMEX futures trade but NYSE closes."
        )

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

# ── Ingestion Log ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Ingestion Log")
st.caption(
    "One row per ticker per run. Failures are flagged with status=error or status=empty. "
    "run_id (first 8 chars) groups all tickers from the same run_ingestion() call."
)

try:
    with get_db() as db:
        log_entries = (
            db.query(IngestionLog)
            .order_by(IngestionLog.started_at.desc())
            .limit(300)
            .all()
        )

    if not log_entries:
        st.info("No ingestion log entries yet. Run an ingestion above to populate the log.")
    else:
        log_df = pd.DataFrame([
            {
                "Run ID":      l.run_id[:8],
                "Started (UTC)": str(l.started_at)[:19] if l.started_at else "—",
                "Ticker":      l.ticker,
                "Name":        l.name,
                "Status":      l.status,
                "Inserted":    l.rows_inserted,
                "Skipped":     l.rows_skipped,
                "Duration ms": l.duration_ms,
                "Error":       l.error_msg or "",
            }
            for l in log_entries
        ])

        # Summary counts for the most recent run
        latest_run = log_df["Run ID"].iloc[0]
        run_slice  = log_df[log_df["Run ID"] == latest_run]
        n_ok    = (run_slice["Status"] == "ok").sum()
        n_empty = (run_slice["Status"] == "empty").sum()
        n_err   = (run_slice["Status"] == "error").sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Last run — OK",    n_ok,    delta=None)
        s2.metric("Last run — Empty", n_empty, delta=None)
        s3.metric("Last run — Error", n_err,   delta=None if n_err == 0 else f"{n_err} tickers failed",
                  delta_color="inverse" if n_err > 0 else "normal")

        if n_err > 0:
            st.warning(
                f"⚠️ {n_err} ticker(s) errored in the last run. "
                "Expand the table below to see the Error column for details."
            )

        status_filter = st.selectbox(
            "Filter by status", ["all", "error", "empty", "ok"], index=0
        )
        display_df = log_df if status_filter == "all" else log_df[log_df["Status"] == status_filter]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Could not load ingestion log: {e}")
