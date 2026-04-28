"""
Data ingestion pipeline — Yahoo Finance → SQLite/PostgreSQL.

HOW THIS WORKS (end to end):

  1. SEED  — On first run, insert one row per commodity into the `commodities`
             reference table (name, ticker, sector, unit). Skips if already seeded.

  2. FETCH — For each commodity, call yfinance to download OHLCV history.
             We ask for the last 5 years on first run (FULL_BACKFILL_PERIOD),
             then daily incremental updates on every subsequent run.

  3. UPSERT — For each OHLCV row from yfinance, try to INSERT it into
              `price_history`. If a row for that (commodity, date, interval)
              already exists (UniqueConstraint), skip it silently.
              This makes the script safe to run repeatedly — no duplicates.

  4. LOG   — Print a summary: how many rows were inserted vs skipped.

RUN MANUALLY:
  python -m pipeline.ingest              # ingest all commodities
  python -m pipeline.ingest --backfill   # force full 5-year history re-pull
"""

import argparse
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from sqlalchemy.exc import IntegrityError

from database.db import init_db, get_db
from database.models import Commodity, PriceHistory, IngestionLog
from services.price_data import COMMODITY_TICKERS, COMMODITY_SECTORS, COMMODITY_UNITS

# ── Logging setup ──────────────────────────────────────────────────────────────
# logging is Python's built-in way to print timestamped messages to the console.
# Using logging instead of print() means we can later redirect output to a file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FULL_BACKFILL_PERIOD = "5y"    # Used on first run to load historical data
INCREMENTAL_PERIOD   = "5d"    # Used on subsequent runs (a few extra days for safety)


# ── Step 1: Seed reference table ───────────────────────────────────────────────
def seed_commodities(db) -> dict[str, int]:
    """
    Insert all known commodities into the `commodities` table.
    Returns a dict of {ticker: commodity_id} for use in the fetch step.
    """
    ticker_to_id = {}

    for name, ticker in COMMODITY_TICKERS.items():
        existing = db.query(Commodity).filter_by(ticker=ticker).first()
        if existing:
            ticker_to_id[ticker] = existing.id
            continue

        commodity = Commodity(
            name   = name,
            ticker = ticker,
            sector = COMMODITY_SECTORS.get(name, "Other"),
            unit   = COMMODITY_UNITS.get(name, "USD"),
        )
        db.add(commodity)
        db.flush()  # flush() sends the INSERT to the DB but doesn't commit yet,
                    # which is how we get the auto-generated `id` back immediately.
        ticker_to_id[ticker] = commodity.id
        log.info(f"  Seeded commodity: {name} ({ticker})")

    return ticker_to_id


# ── Step 2 & 3: Fetch + Upsert ─────────────────────────────────────────────────
def ingest_commodity(
    db, commodity_id: int, ticker: str, name: str, backfill: bool
) -> tuple[int, int, str, str | None]:
    """
    Download OHLCV data for one commodity and write new rows to the DB.
    Returns (inserted_count, skipped_count, status, error_msg).
      status: 'ok' | 'empty' | 'error'
      error_msg: exception text if status == 'error', else None
    """
    period   = FULL_BACKFILL_PERIOD if backfill else INCREMENTAL_PERIOD
    interval = "1d"

    log.info(f"  Fetching {name} ({ticker})  period={period} ...")
    try:
        raw = yf.download(ticker, period=period, interval=interval,
                          progress=False, auto_adjust=True)
    except Exception as e:
        log.warning(f"  yfinance error for {ticker}: {e}")
        return 0, 0, "error", str(e)[:500]

    if raw.empty:
        log.warning(f"  No data returned for {ticker}")
        return 0, 0, "empty", None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    inserted = 0
    skipped  = 0

    for row_date, row in raw.iterrows():
        price_date = row_date.date() if hasattr(row_date, "date") else row_date

        close_val = row.get("Close")
        if pd.isna(close_val):
            continue

        price_row = PriceHistory(
            commodity_id = commodity_id,
            date         = price_date,
            open         = float(row["Open"])   if not pd.isna(row.get("Open"))   else None,
            high         = float(row["High"])   if not pd.isna(row.get("High"))   else None,
            low          = float(row["Low"])    if not pd.isna(row.get("Low"))    else None,
            close        = float(close_val),
            volume       = int(row["Volume"])   if not pd.isna(row.get("Volume")) else 0,
            interval     = interval,
        )

        try:
            # begin_nested() creates a SAVEPOINT so a duplicate row only rolls
            # back this single insert, not the entire session.
            with db.begin_nested():
                db.add(price_row)
            inserted += 1
        except IntegrityError:
            skipped += 1

    return inserted, skipped, "ok", None


# ── Main entry point ───────────────────────────────────────────────────────────
def run_ingestion(backfill: bool = False):
    """
    Full ingestion run: seed → fetch → upsert all commodities.
    Writes one IngestionLog row per commodity so every failure is traceable.
    Called by the scheduler and by the CLI entry point below.
    """
    log.info("=" * 60)
    log.info(f"Ingestion started  (backfill={backfill})")
    log.info("=" * 60)

    init_db()
    log.info("Database tables verified.")

    run_id         = str(uuid.uuid4())
    total_inserted = 0
    total_skipped  = 0
    log_rows       = []

    with get_db() as db:
        log.info("Seeding commodities reference table...")
        ticker_to_id = seed_commodities(db)

        log.info(f"\nIngesting price history for {len(ticker_to_id)} commodities...")
        for name, ticker in COMMODITY_TICKERS.items():
            commodity_id = ticker_to_id.get(ticker)
            if commodity_id is None:
                continue

            t0 = time.monotonic()
            ins, skip, status, error_msg = ingest_commodity(
                db, commodity_id, ticker, name, backfill
            )
            duration_ms = int((time.monotonic() - t0) * 1000)

            total_inserted += ins
            total_skipped  += skip
            log.info(
                f"  {name:25s}  status={status:<5}  "
                f"inserted={ins:4d}  skipped={skip:4d}"
                + (f"  ERROR: {error_msg}" if error_msg else "")
            )

            log_rows.append(IngestionLog(
                run_id        = run_id,
                started_at    = datetime.now(timezone.utc),
                ticker        = ticker,
                name          = name,
                status        = status,
                rows_inserted = ins,
                rows_skipped  = skip,
                error_msg     = error_msg,
                duration_ms   = duration_ms,
            ))

        # Write all log rows in the same session so they commit atomically
        # with the price data.
        db.add_all(log_rows)

    log.info("=" * 60)
    log.info(f"Ingestion complete.  Inserted={total_inserted}  Skipped={total_skipped}")
    log.info("=" * 60)

    # ── Roll adjustment ────────────────────────────────────────────────────────
    # Always run after ingestion so adjusted_close stays current.
    # Even on incremental runs (few new rows), roll detection is fast because
    # it reads the full series and recomputes only what changed.
    if total_inserted > 0:
        log.info("New rows detected — running roll adjustment...")
        from pipeline.roll_adjust import run_roll_adjust
        run_roll_adjust()
        log.info("Roll adjustment complete.")
    else:
        log.info("No new rows inserted — skipping roll adjustment.")

    return total_inserted, total_skipped


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commodity price ingestion pipeline")
    parser.add_argument("--backfill", action="store_true",
                        help="Pull full 5-year history instead of incremental update")
    args = parser.parse_args()
    run_ingestion(backfill=args.backfill)
