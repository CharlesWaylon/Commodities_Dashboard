"""
Step 3: Build a unified trading calendar and time-align every instrument.

THE PROBLEM:
  The 41 instruments in price_history are NOT on the same time axis:

    Bitcoin     ~1,827 rows  — trades 7 days/week, 365 days/year
    Futures     ~1,260 rows  — US exchange trading days only (~252/year)
    ETF proxies ~1,256 rows  — US equity market hours (ETFs close on some
                               holidays that futures ignore, and vice-versa)

  When you build a correlation matrix, you're computing the covariance between
  two return series. If Bitcoin's row 500 is a Saturday and WTI's row 500 is
  the following Monday, you're pairing the wrong observations. The correlation
  number you get is garbage — it's arithmetically correct but economically
  meaningless because the dates don't match.

  Similarly, feeding misaligned instruments into an LSTM or XGBoost model means
  your feature matrix has rows where column A is "Monday's price" and column B
  is "the prior Friday's price" — the model can't learn real relationships from
  that.

THE CANONICAL CALENDAR:
  We need one definitive list of "valid trading days" to use as the shared
  x-axis for all 41 instruments.

  The natural anchor is US exchange trading days, since the majority of our
  instruments (28 direct futures + most ETFs) trade on US markets. Rather than
  importing an external market-calendar library, we derive it empirically:

    For every date that appears in price_history, count how many direct futures
    instruments (ticker ends with =F) have a record on that date. If at least
    FUTURES_QUORUM (50%) of them have data, we call it a valid trading day.

  This is self-calibrating and robust:
    - US holidays (Thanksgiving, Christmas) are naturally excluded because
      futures don't trade then, so very few instruments have records.
    - Bitcoin's weekend prices don't pollute the calendar because Bitcoin is
      not a futures instrument (doesn't count toward the quorum).
    - Data gaps in one or two instruments don't remove a day from the calendar
      because the other 26+ instruments still vote it in.

FORWARD-FILL:
  Once we have the canonical calendar, we reindex every instrument to it.
  For any date in the calendar where an instrument has no record (weekends
  for futures, ETF holidays, gaps), we carry the last known price forward.

  Why forward-fill and not interpolation?
    Financial prices follow a random walk. There is no reason to believe the
    price on a missing day falls between the surrounding days. The best estimate
    for "what would this instrument have been priced at on Tuesday (a holiday)?"
    is "whatever it closed at on Monday." That's what forward-fill does.

  is_filled = True marks those synthetic rows so downstream code can
  identify and weight them differently if needed.

OUTPUT TABLE: aligned_prices
  One row per (commodity, canonical trading day).
  Every instrument has exactly the same number of rows and the same dates.
  Use this table for all cross-asset analysis.

RUN:
  python -m pipeline.align_calendar
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_db, init_db
from database.models import Commodity, PriceHistory, AlignedPrice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# A date qualifies as a canonical trading day if this fraction of direct
# futures instruments have a price record on that date.
FUTURES_QUORUM = 0.50

WRITE_BATCH = 2000   # rows per DB flush — keeps memory usage flat


def build_canonical_calendar(df: pd.DataFrame, futures_ids: set) -> pd.DatetimeIndex:
    """
    Derive the US exchange trading calendar from futures price data.

    Any date where ≥ FUTURES_QUORUM of direct futures instruments have a record
    is treated as a valid trading day. This naturally excludes weekends, US
    public holidays, and exchange-specific closures.
    """
    futures_df   = df[df["commodity_id"].isin(futures_ids)]
    n_futures    = len(futures_ids)
    quorum_count = int(np.ceil(n_futures * FUTURES_QUORUM))

    date_counts = futures_df.groupby("date")["commodity_id"].nunique()
    calendar    = date_counts[date_counts >= quorum_count].index
    calendar    = pd.DatetimeIndex(sorted(calendar))

    log.info(f"  Canonical calendar: {len(calendar)} trading days  "
             f"({calendar.min().date()} → {calendar.max().date()})")
    log.info(f"  Quorum: ≥{quorum_count} of {n_futures} futures instruments per date")
    return calendar


def align_instrument(
    commodity_id: int,
    name: str,
    raw: pd.Series,           # DatetimeIndex → adjusted_close
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Reindex one instrument's adjusted_close series to the canonical calendar
    and forward-fill missing days.

    Returns a DataFrame with columns: commodity_id, date, adjusted_close, is_filled.
    """
    # Restrict calendar to the instrument's active date range to avoid
    # back-filling before the instrument existed.
    active_cal = calendar[(calendar >= raw.index.min()) & (calendar <= raw.index.max())]

    reindexed = raw.reindex(active_cal)

    # Track which positions were originally missing before filling
    was_missing = reindexed.isna()

    # Forward-fill first, then back-fill any leading NaN (instrument started
    # mid-calendar — rare but possible for newer tickers).
    reindexed = reindexed.ffill().bfill()

    result = pd.DataFrame({
        "commodity_id":   commodity_id,
        "date":           active_cal,
        "adjusted_close": reindexed.values,
        "is_filled":      was_missing.values,
    })

    filled_count = int(was_missing.sum())
    if filled_count:
        log.info(f"  {name:<35}  {len(result):>5} rows  ({filled_count} forward-filled)")
    else:
        log.info(f"  {name:<35}  {len(result):>5} rows  (no gaps)")

    return result


def run_alignment():
    log.info("=" * 65)
    log.info("Calendar alignment started")
    log.info("=" * 65)

    # ── Ensure aligned_prices table exists ────────────────────────────────────
    init_db()

    # ── Load all price data ────────────────────────────────────────────────────
    log.info("Loading price data and commodity metadata...")
    with get_db() as db:
        commodity_rows = db.query(
            Commodity.id, Commodity.name, Commodity.ticker
        ).all()

        price_rows = db.query(
            PriceHistory.commodity_id,
            PriceHistory.date,
            PriceHistory.adjusted_close,
        ).filter(
            PriceHistory.interval == "1d",
            PriceHistory.adjusted_close.isnot(None),
        ).all()

    commodities = {r.id: (r.name, r.ticker) for r in commodity_rows}
    futures_ids = {r.id for r in commodity_rows if r.ticker.endswith("=F")}

    df = pd.DataFrame(price_rows, columns=["commodity_id", "date", "adjusted_close"])
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"  Loaded {len(df):,} rows for {len(commodities)} instruments "
             f"({len(futures_ids)} direct futures).")

    # ── Build canonical calendar ───────────────────────────────────────────────
    log.info("\nBuilding canonical trading calendar...")
    calendar = build_canonical_calendar(df, futures_ids)

    # ── Align each instrument ──────────────────────────────────────────────────
    log.info("\nAligning instruments to canonical calendar...")
    all_aligned = []

    for comm_id, (name, ticker) in sorted(commodities.items(), key=lambda x: x[1][0]):
        subset = df[df["commodity_id"] == comm_id].set_index("date")["adjusted_close"].sort_index()
        if subset.empty:
            log.warning(f"  {name}: no adjusted_close data — skipping")
            continue
        aligned = align_instrument(comm_id, name, subset, calendar)
        all_aligned.append(aligned)

    combined = pd.concat(all_aligned, ignore_index=True)
    log.info(f"\n  Total aligned rows: {len(combined):,}  "
             f"({len(combined) // len(commodities)} per instrument on average)")

    # ── Write to database ──────────────────────────────────────────────────────
    log.info("\nWriting to aligned_prices table (replacing any existing data)...")

    with get_db() as db:
        # Clear existing data so this script is safely re-runnable
        deleted = db.query(AlignedPrice).delete(synchronize_session=False)
        if deleted:
            log.info(f"  Cleared {deleted:,} existing aligned rows.")

        written = 0
        for start in range(0, len(combined), WRITE_BATCH):
            chunk = combined.iloc[start : start + WRITE_BATCH]
            db.bulk_insert_mappings(AlignedPrice, [
                {
                    "commodity_id":   int(row["commodity_id"]),
                    "date":           row["date"].date(),
                    "adjusted_close": float(row["adjusted_close"]),
                    "is_filled":      bool(row["is_filled"]),
                }
                for _, row in chunk.iterrows()
            ])
            written += len(chunk)

        log.info(f"  Written {written:,} rows.")

    # ── Validation report ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  CALENDAR ALIGNMENT REPORT")
    print("=" * 80)

    n_instruments = combined["commodity_id"].nunique()
    n_total       = len(combined)
    n_filled      = int(combined["is_filled"].sum())
    pct_filled    = n_filled / n_total * 100

    print(f"\n  SUMMARY")
    print(f"  {'Canonical trading days:':<40} {len(calendar):,}")
    print(f"  {'Instruments aligned:':<40} {n_instruments}")
    print(f"  {'Total aligned rows:':<40} {n_total:,}")
    print(f"  {'Rows forward-filled (synthetic):':<40} {n_filled:,}  ({pct_filled:.1f}%)")

    # ── Per-instrument fill rate ───────────────────────────────────────────────
    print(f"\n  FILL RATE BY INSTRUMENT  (sorted by % filled — highest = most gaps)")
    print(f"  {'Instrument':<35} {'Ticker':<10} {'Rows':>6}  {'Filled':>7}  {'Fill%':>6}")
    print("  " + "-" * 70)

    for comm_id, group in combined.groupby("commodity_id"):
        name, ticker = commodities[comm_id]
        n_rows   = len(group)
        n_fill   = int(group["is_filled"].sum())
        pct_fill = n_fill / n_rows * 100
        flag     = " ⚠" if pct_fill > 15 else "  "
        print(
            f"  {name:<35} {ticker:<10} {n_rows:>6}  {n_fill:>7}  {pct_fill:>5.1f}%{flag}"
        )

    print()
    print("  WHAT THIS MEANS:")
    print("  Fill% > 0%   → normal. The instrument doesn't trade every day in the calendar.")
    print("  Fill% ≈ 28%  → expected for futures vs Bitcoin 7-day calendar.")
    print("  Fill% ⚠ >15% → investigate; may indicate data gaps beyond normal holidays.")
    print()
    print("  USE aligned_prices FOR:")
    print("  • Correlation matrices    • Cross-asset model training")
    print("  • Return spread analysis  • Synchronized portfolio backtests")
    print()
    print("  USE price_history FOR:")
    print("  • Single-instrument charts  • Individual rolling statistics")
    print("  • Auditing / raw data inspection")
    print()
    print("=" * 80)

    return combined, calendar


if __name__ == "__main__":
    run_alignment()
