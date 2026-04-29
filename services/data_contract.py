"""
Data Contract — canonical name registry and DB-first data access layer.

WHY THIS EXISTS
--------------
Two naming systems exist in the codebase:

  Display names  (services/price_data.py COMMODITY_TICKERS):
    "Gold (COMEX)", "Corn (CBOT)", "Natural Gas (Henry Hub)", ...
    Used in the UI: labels, metrics, selectboxes.

  DB canonical names  (commodities table):
    "Gold", "Corn", "Natural Gas", ...
    Used in all SQL queries.

Any code that queries the DB with display names silently returns empty results —
the bug that caused the "populate DB" false error on the Command Centre.

HOW IT WORKS
------------
The ticker symbol (GC=F, ZC=F, NG=F ...) is the same in both systems.
This module uses ticker as a bridge to build the mapping at runtime,
so it self-heals if names change in either system.

USAGE
-----
  from services.data_contract import fetch_price_history, fetch_current_prices_db

  # Historical OHLCV from DB — pass display names, get display names back
  df = fetch_price_history(names=["Gold (COMEX)", "WTI Crude Oil"], days=90)

  # Drop-in replacement for fetch_current_prices() using DB data
  df = fetch_current_prices_db()

  # Name conversion utilities
  db_name  = display_to_db("Gold (COMEX)")   # → "Gold"
  disp     = db_to_display("Gold")           # → "Gold (COMEX)"
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional


# ── Name registry ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_name_map() -> dict[str, str]:
    """
    Returns {display_name → db_canonical_name}.

    Built live by joining COMMODITY_TICKERS (display→ticker) with the DB
    commodities table (ticker→canonical_name). Self-healing: survives renames
    in either system as long as the ticker stays the same.
    """
    try:
        from database.db import get_engine
        from services.price_data import COMMODITY_TICKERS
        engine = get_engine()
        conn = engine.raw_connection()
        try:
            db = pd.read_sql("SELECT name, ticker FROM commodities", conn)
        finally:
            conn.close()
        ticker_to_db = dict(zip(db["ticker"], db["name"]))
        return {
            display: ticker_to_db.get(ticker, display)
            for display, ticker in COMMODITY_TICKERS.items()
        }
    except Exception:
        return {}


def display_to_db(display_name: str) -> str:
    """Convert a UI display name to the DB canonical name."""
    return get_name_map().get(display_name, display_name)


def db_to_display(db_name: str) -> str:
    """Convert a DB canonical name to the UI display name."""
    rev = {v: k for k, v in get_name_map().items()}
    return rev.get(db_name, db_name)


def _rev_map() -> dict[str, str]:
    """Cached reverse map: {db_canonical_name → display_name}."""
    return {v: k for k, v in get_name_map().items()}


# ── DB-first price history ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_price_history(
    names: Optional[list[str]] = None,
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV history from PostgreSQL/SQLite.

    Args:
        names:  Display names to filter (e.g. ["Gold (COMEX)", "WTI Crude Oil"]).
                Pass None to fetch all 41 commodities.
        days:   Calendar days of history to return.

    Returns:
        DataFrame with columns:
            name (display), sector, ticker, date,
            open, high, low, close, adjusted_close, volume.
    """
    try:
        from database.db import get_engine
        engine = get_engine()
        cutoff = (datetime.now() - timedelta(days=days)).date()

        name_filter = ""
        if names:
            db_names = [display_to_db(n) for n in names]
            quoted   = ", ".join(f"'{n}'" for n in db_names)
            name_filter = f"AND c.name IN ({quoted})"

        conn = engine.raw_connection()
        try:
            df = pd.read_sql(
                f"""
                SELECT c.name  AS db_name,
                       c.sector, c.ticker,
                       ph.date,
                       ph.open, ph.high, ph.low, ph.close,
                       COALESCE(ph.adjusted_close, ph.close) AS adjusted_close,
                       ph.volume
                FROM   price_history ph
                JOIN   commodities   c  ON c.id = ph.commodity_id
                WHERE  ph.date     >= '{cutoff}'
                  AND  ph.interval  = '1d'
                  {name_filter}
                ORDER BY c.name, ph.date
                """,
                conn,
                parse_dates=["date"],
            )
        finally:
            conn.close()

        if df.empty:
            return df

        rev       = _rev_map()
        df["name"] = df["db_name"].map(rev).fillna(df["db_name"])
        return df.drop(columns=["db_name"])

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_current_prices_db() -> pd.DataFrame:
    """
    Latest price + 1-day change for every commodity, sourced entirely from the DB.
    Returns the same schema as services.price_data.fetch_current_prices() and is
    a drop-in replacement when live streaming isn't required.

    The DB is the authoritative source; yfinance is only needed for the live
    top-bar price updates that need <5-minute freshness.
    """
    try:
        from database.db import get_engine
        from services.price_data import COMMODITY_UNITS, COMMODITY_IS_PROXY
        engine = get_engine()

        # Fetch the two most recent rows per commodity for 1-day delta
        conn = engine.raw_connection()
        try:
            df = pd.read_sql(
                """
                SELECT db_name, sector, ticker, unit, date, price, rn
                FROM (
                    SELECT c.name   AS db_name,
                           c.sector, c.ticker, c.unit,
                           ph.date,
                           COALESCE(ph.adjusted_close, ph.close) AS price,
                           ROW_NUMBER() OVER (
                               PARTITION BY c.id ORDER BY ph.date DESC
                           ) AS rn
                    FROM   price_history ph
                    JOIN   commodities   c ON c.id = ph.commodity_id
                    WHERE  ph.interval = '1d'
                ) ranked
                WHERE rn <= 2
                """,
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            return pd.DataFrame()

        rev = _rev_map()
        rows = []
        for db_name, grp in df.groupby("db_name"):
            grp    = grp.sort_values("rn")
            latest = grp[grp["rn"] == 1].iloc[0]
            prev   = grp[grp["rn"] == 2]

            price  = float(latest["price"])
            if not prev.empty:
                prev_p = float(prev.iloc[0]["price"])
                change = price - prev_p
                pct    = (change / prev_p * 100) if prev_p else 0.0
            else:
                change, pct = 0.0, 0.0

            display = rev.get(db_name, db_name)
            rows.append({
                "Name":       display,
                "Sector":     latest["sector"],
                "Price":      round(price, 4),
                "Change":     round(change, 4),
                "Pct_Change": round(pct, 2),
                "Unit":       COMMODITY_UNITS.get(display, latest.get("unit", "USD")),
                "Ticker":     latest["ticker"],
                "Is_Proxy":   COMMODITY_IS_PROXY.get(display, False),
                "Date":       latest["date"],
            })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


# ── Freshness check ────────────────────────────────────────────────────────────

def data_freshness() -> dict:
    """
    Returns a freshness report for the price_history table:
      newest_date   — most recent date in DB
      days_stale    — calendar days since newest_date
      is_fresh      — True if data is ≤1 trading day old
      recommend_live — True if yfinance fallback is recommended
    """
    try:
        from database.db import get_engine
        engine  = get_engine()
        conn = engine.raw_connection()
        try:
            result = pd.read_sql(
                "SELECT MAX(date) AS newest FROM price_history WHERE interval = '1d'",
                conn,
            )
        finally:
            conn.close()
        newest  = pd.to_datetime(result["newest"].iloc[0]).date()
        today   = datetime.now(timezone.utc).date()
        stale   = (today - newest).days
        # weekends don't count as stale
        is_fresh = stale == 0 or (stale == 1 and today.weekday() == 0) or today.weekday() >= 5
        return {
            "newest_date":    newest,
            "days_stale":     stale,
            "is_fresh":       is_fresh,
            "recommend_live": stale > 3,
        }
    except Exception:
        return {
            "newest_date": None,
            "days_stale":  None,
            "is_fresh":    False,
            "recommend_live": True,
        }


# ── Convenience: wide price matrix for models ──────────────────────────────────

@st.cache_data(ttl=600)
def fetch_price_matrix(
    names: Optional[list[str]] = None,
    days: int = 1095,
    price_col: str = "adjusted_close",
) -> pd.DataFrame:
    """
    Returns a wide DataFrame: date index × commodity columns.
    Suitable for feeding directly into models (HMM, XGBoost, correlations, etc.).

    Args:
        names:     Display names to include. None = all 41 commodities.
        days:      Calendar days of history.
        price_col: "adjusted_close" (default) or "close".
    """
    df = fetch_price_history(names=names, days=days)
    if df.empty:
        return pd.DataFrame()

    col = price_col if price_col in df.columns else "close"
    wide = df.pivot(index="date", columns="name", values=col)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index().ffill(limit=3)
