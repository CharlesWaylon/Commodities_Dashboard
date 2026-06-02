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

  4. M2    — After the front-month pass, fetch second-nearby contract prices
             (see FUTURES_M2_TICKERS in services/price_data.py) and store them
             with interval="1d_m2" under the same commodity_id.  ETF/equity
             proxies are absent from FUTURES_M2_TICKERS and are silently skipped.

  5. LOG   — Print a summary: how many rows were inserted vs skipped.

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
from database.models import Commodity, PriceHistory, IngestionLog, PriceValidationLog
from pipeline.price_validator import validate_price_series, AnomalyRecord
from services.price_data import (
    COMMODITY_TICKERS,
    COMMODITY_SECTORS,
    COMMODITY_UNITS,
    FUTURES_M2_TICKERS,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
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


# ── Step 2 & 3: Fetch + Validate + Upsert ──────────────────────────────────────
def ingest_commodity(
    db, commodity_id: int, ticker: str, name: str, backfill: bool, run_id: str,
    interval: str = "1d",
) -> tuple[int, int, str, str | None, list[AnomalyRecord]]:
    """
    Download OHLCV data for one commodity, validate it, then write to the DB.

    Parameters
    ----------
    interval : str
        Storage interval tag.  Use "1d" for the front-month series (default)
        and "1d_m2" for second-nearby contract data.  The UniqueConstraint on
        (commodity_id, date, interval) keeps them separate and idempotent.

    Returns (inserted_count, skipped_count, status, error_msg, anomalies).
      status: 'ok' | 'empty' | 'error' | 'circuit_breaker'
      error_msg: description if status is not 'ok', else None
      anomalies: list of AnomalyRecord from price_validator (may be empty)
    """
    period = FULL_BACKFILL_PERIOD if backfill else INCREMENTAL_PERIOD

    log.info(f"  Fetching {name} ({ticker})  period={period} interval={interval} ...")
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          progress=False, auto_adjust=True)
    except Exception as e:
        log.warning(f"  yfinance error for {ticker}: {e}")
        return 0, 0, "error", str(e)[:500], []

    if raw.empty:
        log.warning(f"  No data returned for {ticker}")
        return 0, 0, "empty", None, []

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ── Validate before touching the DB ────────────────────────────────────────
    validation = validate_price_series(ticker, name, raw)

    if validation.circuit_breaker_triggered:
        log.error(
            "  CIRCUIT BREAKER: %s — skipping all %d fetched rows. Reason: %s",
            ticker, len(raw), validation.circuit_breaker_reason,
        )
        return (
            0, 0,
            "circuit_breaker",
            validation.circuit_breaker_reason[:500],
            validation.anomalies,
        )

    # Use the cleaned DataFrame (corrections applied, excluded rows dropped)
    clean = validation.clean_df
    if clean.empty:
        return 0, 0, "empty", None, validation.anomalies

    inserted = 0
    skipped  = 0

    for row_date, row in clean.iterrows():
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

    return inserted, skipped, "ok", None, validation.anomalies


# ── Step 4: M2 (second-nearby) ingest ─────────────────────────────────────────
def _run_m2_ingestion(run_id: str, backfill: bool = False) -> tuple[int, int]:
    """
    Fetch and upsert second-nearby (M2) price series for each futures commodity
    listed in FUTURES_M2_TICKERS (services/price_data.py).

    Storage: existing ``price_history`` table with ``interval = "1d_m2"``.
    The UniqueConstraint on (commodity_id, date, interval) ensures idempotency.
    M2 rows are stored under the same commodity_id as the front-month series,
    so no new commodities table rows are needed.

    Design notes
    ─────────────
    • FUTURES_M2_TICKERS lists specific dated contract tickers (e.g. CLQ26.NYM)
      rather than continuous tickers.  yfinance typically returns 6-12 months of
      history per contract.  Historical depth grows as successive contracts are
      added to the mapping over time (semi-annual update).
    • ETF/equity proxies (names ending with "*") have no futures term structure
      and are absent from FUTURES_M2_TICKERS — they produce no rows, no errors.
    • Failures per ticker are non-fatal: logged at WARNING, then skipped.
    • ⚠️  MAINTENANCE: Update FUTURES_M2_TICKERS every ~6 months as contracts roll.
    """
    log.info("=" * 60)
    log.info("M2 (second-nearby) ingestion started")
    log.info("=" * 60)

    total_ins = 0
    total_sk  = 0

    with get_db() as db:
        # Build name→commodity_id map for M2 candidates using the seeded DB
        name_to_id: dict[str, int] = {}
        for name in FUTURES_M2_TICKERS:
            row = db.query(Commodity).filter_by(name=name).first()
            if row is None:
                log.warning("M2 skip — commodity not seeded yet: %r", name)
                continue
            name_to_id[name] = row.id

        for name, m2_ticker in FUTURES_M2_TICKERS.items():
            commodity_id = name_to_id.get(name)
            if commodity_id is None:
                continue

            ins, sk, status, err, _ = ingest_commodity(
                db, commodity_id, m2_ticker, name,
                backfill=backfill, run_id=run_id, interval="1d_m2",
            )
            total_ins += ins
            total_sk  += sk
            log.info(
                "  M2 %-28s (%s)  status=%-12s  inserted=%4d  skipped=%4d%s",
                name, m2_ticker, status, ins, sk,
                f"  MSG: {err}" if err else "",
            )

    log.info("M2 ingestion complete.  Inserted=%d  Skipped=%d", total_ins, total_sk)
    return total_ins, total_sk


# ── Main entry point ───────────────────────────────────────────────────────────
def run_ingestion(backfill: bool = False):
    """
    Full ingestion run: seed → fetch M1 → fetch M2 → upsert all commodities.
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
        validation_log_rows: list = []

        for name, ticker in COMMODITY_TICKERS.items():
            commodity_id = ticker_to_id.get(ticker)
            if commodity_id is None:
                continue

            t0 = time.monotonic()
            ins, skip, status, error_msg, anomalies = ingest_commodity(
                db, commodity_id, ticker, name, backfill, run_id
            )
            duration_ms = int((time.monotonic() - t0) * 1000)

            total_inserted += ins
            total_skipped  += skip

            anomaly_summary = (
                f"  anomalies={len(anomalies)}" if anomalies else ""
            )
            log.info(
                f"  {name:25s}  status={status:<16}  "
                f"inserted={ins:4d}  skipped={skip:4d}{anomaly_summary}"
                + (f"  MSG: {error_msg}" if error_msg else "")
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

            # Persist each anomaly to the audit table
            for a in anomalies:
                validation_log_rows.append(PriceValidationLog(
                    run_id          = run_id,
                    ticker          = a.ticker,
                    name            = a.name,
                    date            = a.date,
                    raw_close       = a.raw_close,
                    corrected_close = a.corrected_close,
                    reason_code     = a.reason_code,
                    action          = a.action,
                    details         = a.details[:500] if a.details else None,
                ))

        # Write all log rows in the same session so they commit atomically
        # with the price data.
        db.add_all(log_rows)
        if validation_log_rows:
            db.add_all(validation_log_rows)
            log.info(
                "Persisted %d validation anomaly records to price_validation_log.",
                len(validation_log_rows),
            )

    log.info("=" * 60)
    log.info(f"Ingestion complete.  Inserted={total_inserted}  Skipped={total_skipped}")
    log.info("=" * 60)

    # ── M2 (second-nearby) ingest ──────────────────────────────────────────────
    # Stored with interval="1d_m2" under the same commodity_id as the front
    # series.  ETF/equity proxies are not in FUTURES_M2_TICKERS and are skipped.
    # Failures are non-fatal: if a specific contract ticker is expired or
    # unavailable on yfinance, that commodity is silently skipped for this run.
    _run_m2_ingestion(run_id=run_id, backfill=backfill)

    # ── Alert report ───────────────────────────────────────────────────────────
    try:
        from pipeline.alert_reporter import generate_alert_report
        report_path = generate_alert_report(run_id)
        log.info("Alert report: %s", report_path)
    except Exception as exc:
        log.warning("Alert report failed (non-fatal): %s", exc)

    # ── Roll adjustment ────────────────────────────────────────────────────────
    # Always run after ingestion so adjusted_close stays current.
    # Even on incremental runs (few new rows), roll detection is fast because
    # it reads the full series and recomputes only what changed.
    if total_inserted > 0:
        log.info("New rows detected — running roll adjustment...")
        from pipeline.roll_adjust import run_roll_adjust
        run_roll_adjust()
        log.info("Roll adjustment complete.")

        # Rebuild aligned_prices (calendar-aligned forward-filled table).
        # Must run after roll_adjust so adjusted_close is current before alignment.
        log.info("Aligning calendar (updating aligned_prices)...")
        try:
            from pipeline.align_calendar import run_alignment
            run_alignment()
            log.info("Calendar alignment complete.")
        except Exception as exc:
            log.warning("Calendar alignment failed (non-fatal): %s", exc)

        # Recompute 21-day rolling correlations and persist to correlation_snapshots.
        # Must run after align_calendar so aligned_prices is current.
        log.info("Storing correlation snapshot...")
        try:
            from models.cross_asset import store_correlation_snapshot
            n_pairs = store_correlation_snapshot()
            log.info("Correlation snapshot stored (%d pairs).", n_pairs)
        except Exception as exc:
            log.warning("Correlation snapshot failed (non-fatal): %s", exc)
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
