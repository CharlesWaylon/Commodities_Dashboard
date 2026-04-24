"""
Step 2: Roll-adjusted prices — eliminate contract-roll discontinuities.

THE PROBLEM:
  Commodity futures expire every 1–3 months. When a contract expires, Yahoo
  Finance automatically rolls to the next front-month contract. The new
  contract almost always opens at a different price than the expiring one —
  not because the market moved, but because the two contracts were trading
  simultaneously at different prices (the new contract already priced in
  carry / storage costs / convenience yield for the extra months).

  This creates an ARTIFICIAL price jump in your time series that looks like
  a real market move but isn't. If you calculate daily returns over this raw
  series, the roll day shows a return of +10% or –10% that never happened.
  Every downstream calculation — volatility, Sharpe ratio, model training —
  is corrupted.

  Example  (WTI Crude):
    2024-10-31  raw close  =  $82.00  (November contract — last day)
    2024-11-01  raw close  =  $75.50  (December contract — first day)
    Raw return = (75.50 – 82.00) / 82.00 = –8.0% ← FICTIONAL; oil didn't crash

THE FIX — proportional backward adjustment:
  We work BACKWARD from the most recent date and, every time we cross a roll
  point, we rescale all PRIOR prices proportionally so the series is
  continuous at that date.

  Continuing the example:
    Roll ratio  = close_new / close_old = 75.50 / 82.00 = 0.9207
    All prices before 2024-11-01 are multiplied by 0.9207.
    The adjusted close on 2024-10-31 becomes 82.00 × 0.9207 = 75.50.
    Return on 2024-11-01 in the adjusted series = 75.50 / 75.50 = 0% ✓

  If there is a SECOND earlier roll, we multiply by the cumulative product
  of BOTH ratios, and so on. The most recent prices are never touched
  (adjustment_factor = 1.0); prices get progressively rescaled further back
  in time as more rolls accumulate.

HOW WE DETECT ROLLS:
  We compute log returns and z-score them per instrument.
    z = (log_return – μ) / σ
  A z-score > Z_THRESHOLD (5.0σ for most futures, 6.5σ for Natural Gas which
  has legitimately large moves) is treated as a roll artifact rather than a
  real market event.

  Why z-scores instead of a fixed % threshold?
    Natural Gas can legitimately move 15% in a day; Gold rarely moves 3%.
    A single fixed threshold would either miss Gas rolls or flag every large
    Gold day. Z-scores self-calibrate to each instrument's own volatility.

  Why 5.0σ?
    Under a normal distribution, |z| > 5.0 should occur in roughly 1 in
    3.5 million observations by chance. In our ~1,200-row series per
    instrument, genuine 5σ events are essentially impossible. Actual roll
    artifacts produce z-scores of 20–200+, so the threshold is conservative.

WHAT GETS ADJUSTED:
  - Direct futures (ticker ends with "=F"):  full proportional backward adj
  - ETF / equity proxies and Bitcoin:        adjusted_close = close,
                                             adjustment_factor = 1.0
    (These instruments don't roll — their prices are already continuous.)

OUTPUT:
  Every row in price_history gets:
    adjusted_close    — use this for any return calculation, chart, or model
    adjustment_factor — cumulative multiplier applied to close; 1.0 = no adj

RUN:
  python -m pipeline.roll_adjust
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_db
from database.models import Commodity, PriceHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
Z_DEFAULT  = 5.0   # |z| > this → roll artifact (most instruments)
Z_NATGAS   = 6.5   # Natural Gas legitimately spikes; use higher bar
NATGAS_TICKER = "NG=F"


def is_futures(ticker: str) -> bool:
    """Direct futures contract — needs roll adjustment."""
    return ticker.endswith("=F")


def detect_and_adjust(name: str, ticker: str, closes: pd.Series) -> pd.DataFrame:
    """
    Core algorithm.

    Parameters
    ----------
    name    : instrument display name (for logging)
    ticker  : Yahoo Finance ticker
    closes  : pd.Series of close prices, DatetimeIndex, sorted oldest → newest

    Returns
    -------
    DataFrame with columns: date, adjusted_close, adjustment_factor
    """
    n = len(closes)
    close_vals = closes.values.astype(float)
    dates      = closes.index

    # ── Non-futures: trivial pass-through ─────────────────────────────────────
    if not is_futures(ticker):
        return pd.DataFrame({
            "date":              dates,
            "adjusted_close":    close_vals,
            "adjustment_factor": np.ones(n),
        })

    threshold = Z_NATGAS if ticker == NATGAS_TICKER else Z_DEFAULT

    # ── Compute log returns and z-scores ──────────────────────────────────────
    log_returns = np.log(close_vals[1:] / close_vals[:-1])   # length n-1

    if len(log_returns) < 2 or np.std(log_returns) == 0:
        # Not enough data to compute z-scores; no adjustment
        log.warning(f"  {name}: insufficient data for z-score calculation — skipping")
        return pd.DataFrame({
            "date":              dates,
            "adjusted_close":    close_vals,
            "adjustment_factor": np.ones(n),
        })

    mu    = np.mean(log_returns)
    sigma = np.std(log_returns)
    z     = (log_returns - mu) / sigma   # length n-1; z[i] is return from i to i+1

    # roll_at[i+1] = True means the price jump from index i to i+1 is a roll
    # (z[i] corresponds to the return at index i+1)
    roll_set = set()
    for idx, z_val in enumerate(z):
        if abs(z_val) > threshold:
            roll_set.add(idx + 1)  # the spike lands at index idx+1

    # ── Proportional backward adjustment ──────────────────────────────────────
    # Start from the end (cumulative_factor = 1.0).
    # When crossing a roll point (going backward), update the cumulative factor
    # so all PRIOR prices are rescaled to eliminate the artificial gap.
    adj_close  = np.empty(n)
    adj_factor = np.empty(n)
    cum_factor = 1.0

    for i in range(n - 1, -1, -1):
        adj_close[i]  = close_vals[i] * cum_factor
        adj_factor[i] = cum_factor
        # If i is a roll date, the step from i-1 → i is artificial.
        # Before we process i-1 and earlier, update the factor:
        if i in roll_set:
            roll_ratio  = close_vals[i] / close_vals[i - 1]
            cum_factor *= roll_ratio

    return pd.DataFrame({
        "date":              dates,
        "adjusted_close":    adj_close,
        "adjustment_factor": adj_factor,
    })


def run_roll_adjust():
    """
    Load all instruments, compute adjustments, write back to the database,
    and print a validation report.
    """
    log.info("=" * 65)
    log.info("Roll adjustment started")
    log.info("=" * 65)

    # ── Load all prices ────────────────────────────────────────────────────────
    log.info("Loading price data from database...")
    with get_db() as db:
        rows = (
            db.query(
                Commodity.name,
                Commodity.ticker,
                PriceHistory.id,
                PriceHistory.date,
                PriceHistory.close,
            )
            .join(PriceHistory, Commodity.id == PriceHistory.commodity_id)
            .filter(PriceHistory.interval == "1d")
            .order_by(Commodity.name, PriceHistory.date)
            .all()
        )

    df = pd.DataFrame(rows, columns=["name", "ticker", "row_id", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"  Loaded {len(df):,} rows across {df['name'].nunique()} instruments.")

    # ── Process each instrument ────────────────────────────────────────────────
    all_results   = []
    summary_rows  = []

    for (name, ticker), group in df.groupby(["name", "ticker"]):
        series = group.set_index("date")["close"].sort_index()

        adj_df = detect_and_adjust(name, ticker, series)

        # Attach the database row IDs so we can UPDATE the right rows
        id_series = group.set_index("date")["row_id"].sort_index()
        adj_df["row_id"] = id_series.values

        all_results.append(adj_df)

        # ── Roll detection summary for report ─────────────────────────────────
        if is_futures(ticker):
            # Recompute roll indices to count and record them
            close_vals  = series.values.astype(float)
            log_returns = np.log(close_vals[1:] / close_vals[:-1])
            if len(log_returns) > 1 and np.std(log_returns) > 0:
                mu    = np.mean(log_returns)
                sigma = np.std(log_returns)
                z     = (log_returns - mu) / sigma
                thresh = Z_NATGAS if ticker == NATGAS_TICKER else Z_DEFAULT
                roll_mask = np.abs(z) > thresh
                roll_indices = np.where(roll_mask)[0] + 1  # index i+1

                worst_raw_z   = float(np.max(np.abs(z[roll_mask])))   if roll_mask.any() else 0.0
                worst_raw_pct = float(
                    (np.exp(log_returns[np.argmax(np.abs(z))]) - 1) * 100
                ) if len(log_returns) else 0.0

                # After adjustment: log returns of adjusted series
                adj_vals        = adj_df["adjusted_close"].values
                adj_log_returns = np.log(adj_vals[1:] / adj_vals[:-1])
                worst_adj_z     = float(np.max(np.abs(
                    (adj_log_returns - np.mean(adj_log_returns)) / np.std(adj_log_returns)
                ))) if np.std(adj_log_returns) > 0 else 0.0

                roll_dates = [series.index[i].date() for i in roll_indices]
            else:
                roll_indices = []
                roll_dates   = []
                worst_raw_z  = 0.0
                worst_raw_pct = 0.0
                worst_adj_z  = 0.0

            summary_rows.append({
                "name":           name,
                "ticker":         ticker,
                "type":           "futures",
                "rolls_detected": len(roll_indices),
                "worst_raw_z":    round(worst_raw_z, 1),
                "worst_raw_pct":  round(worst_raw_pct, 2),
                "worst_adj_z":    round(worst_adj_z, 1),
                "roll_dates":     roll_dates[:5],   # top 5 for display
            })
        else:
            summary_rows.append({
                "name":           name,
                "ticker":         ticker,
                "type":           "proxy/crypto",
                "rolls_detected": 0,
                "worst_raw_z":    0.0,
                "worst_raw_pct":  0.0,
                "worst_adj_z":    0.0,
                "roll_dates":     [],
            })

    # ── Write back to database ─────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    log.info(f"\nWriting {len(combined):,} adjusted rows to database...")

    BATCH = 500
    written = 0

    with get_db() as db:
        for start in range(0, len(combined), BATCH):
            chunk = combined.iloc[start : start + BATCH]
            for _, row in chunk.iterrows():
                db.query(PriceHistory).filter(PriceHistory.id == int(row["row_id"])).update(
                    {
                        "adjusted_close":    float(row["adjusted_close"]),
                        "adjustment_factor": float(row["adjustment_factor"]),
                    },
                    synchronize_session=False,
                )
                written += 1
            db.flush()   # send to DB; outer context manager commits on exit
        log.info(f"  {written:,} rows updated.")

    # ── Validation report ──────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("  ROLL ADJUSTMENT REPORT")
    print("=" * 90)

    futures_rows = [r for r in summary_rows if r["type"] == "futures"]
    proxy_rows   = [r for r in summary_rows if r["type"] != "futures"]

    total_rolls = sum(r["rolls_detected"] for r in futures_rows)
    print(f"\n  SUMMARY")
    print(f"  {'Futures instruments adjusted:':<40} {len(futures_rows)}")
    print(f"  {'Proxy/crypto (pass-through):':<40} {len(proxy_rows)}")
    print(f"  {'Total roll events detected & removed:':<40} {total_rolls}")
    print(f"  {'Total rows written:':<40} {written:,}")

    # ── Per-instrument futures table ────────────────────────────────────────────
    col = [32, 10, 7, 12, 12, 12]
    header = (
        f"\n  {'Instrument':<{col[0]}} {'Ticker':<{col[1]}} "
        f"{'Rolls':>{col[2]}} {'WorstRaw%':>{col[3]}} "
        f"{'WorstRawZ':>{col[4]}} {'WorstAdjZ':>{col[5]}}"
    )
    print(f"\n  FUTURES — roll detection and residual z-scores after adjustment")
    print(header)
    print("  " + "-" * 88)

    for r in sorted(futures_rows, key=lambda x: -x["rolls_detected"]):
        flag = " ⚠" if r["worst_adj_z"] > 5.0 else "  "
        print(
            f"  {r['name']:<{col[0]}} {r['ticker']:<{col[1]}} "
            f"{r['rolls_detected']:>{col[2]}} "
            f"{r['worst_raw_pct']:>+{col[3]}.1f}% "
            f"{r['worst_raw_z']:>{col[4]}.1f}σ "
            f"{r['worst_adj_z']:>{col[5]}.1f}σ{flag}"
        )

    # ── Roll date detail for instruments with most rolls ───────────────────────
    print(f"\n  DETECTED ROLL DATES  (top roll per instrument, sorted by z-score)")
    print(f"  {'Instrument':<32} {'Roll dates (up to 5 shown)'}")
    print("  " + "-" * 70)
    for r in sorted(futures_rows, key=lambda x: -x["rolls_detected"]):
        if r["roll_dates"]:
            dates_str = "  ".join(str(d) for d in r["roll_dates"])
            print(f"  {r['name']:<32} {dates_str}")

    # ── Proxy pass-through confirmation ───────────────────────────────────────
    print(f"\n  PROXY / CRYPTO  (adjusted_close = close, adjustment_factor = 1.0)")
    proxy_names = ", ".join(r["ticker"] for r in proxy_rows)
    print(f"  {proxy_names}")

    print("\n" + "=" * 90)
    print("  Adjustment complete. Use 'adjusted_close' for all return calculations.")
    print("=" * 90)

    return summary_rows


if __name__ == "__main__":
    run_roll_adjust()
