"""
Step 1: Data quality audit — measure before you fix.

WHAT THIS SCRIPT MEASURES:
  For each of the 41 instruments in the database, it catalogues:

  1. DATE GAPS
     How many calendar days between consecutive price records?
     Normal max gap = 4 days (Friday → Tuesday, spanning a US holiday).
     Gaps > 4 days are suspicious: they indicate missing data, a contract
     expiry without a replacement, or a trading halt.

  2. ZERO / NULL VOLUME
     Days where volume = 0 or NULL. This is normal for some commodity
     futures (OJ, rice — thin markets), but widespread zero-volume days
     on liquid contracts like WTI or Gold signal bad data.

  3. STALE PRICES
     Consecutive days with the exact same close price. A real market
     never closes at precisely the same price two days running. These
     are always a data error or a weekend/holiday mislabelled as a
     trading day.

  4. PRICE SPIKES (potential roll artifacts)
     Using z-scores on log returns rather than a fixed threshold.
     Why z-scores? Because Natural Gas can move 15% legitimately while
     Gold rarely moves 3%. A fixed 10% threshold would miss Gas rolls
     and incorrectly flag Gold's largest-ever single-day move.

     Method: compute full-series log return mean (μ) and std (σ).
     Flag any day where |log_return - μ| > Z_THRESHOLD × σ.
     Z_THRESHOLD = 4.0 catches the very extreme tail — at a normal
     distribution, this happens 1 in 31,560 observations by chance.
     In practice, futures roll artifacts produce z-scores of 20–200+,
     so they are unmistakably separated from real large moves.

  5. DATE RANGE ALIGNMENT
     Compares every instrument's start date to the earliest available
     date across all instruments. Bitcoin starts in 2021 with the rest,
     but physical ETFs and some proxies may start later.

  6. COVERAGE RELATIVE TO A GLOBAL CALENDAR
     "Global calendar" = the union of all dates any instrument has a
     record for. Any instrument missing more than 2% of global calendar
     days deserves investigation.

OUTPUT:
  - Rich terminal table (printed to stdout)
  - reports/audit_YYYYMMDD.csv  — machine-readable for later reference

RUN:
  python -m pipeline.audit
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_db
from database.models import Commodity, PriceHistory
from sqlalchemy import func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

Z_THRESHOLD   = 4.0    # flag returns further than 4σ from mean
GAP_THRESHOLD = 4      # flag calendar-day gaps larger than this
STALE_WINDOW  = 3      # flag N or more consecutive identical closes
REPORTS_DIR   = Path(__file__).parent.parent / "reports"


def load_all_prices() -> dict[str, pd.DataFrame]:
    """Load every instrument's full price series into a dict keyed by name."""
    log.info("Loading price data from database...")
    with get_db() as db:
        rows = (
            db.query(
                Commodity.name,
                Commodity.ticker,
                Commodity.sector,
                PriceHistory.date,
                PriceHistory.close,
                PriceHistory.volume,
            )
            .join(PriceHistory, Commodity.id == PriceHistory.commodity_id)
            .order_by(Commodity.name, PriceHistory.date)
            .all()
        )

    df = pd.DataFrame(rows, columns=["name", "ticker", "sector", "date", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])

    series = {}
    for name, group in df.groupby("name"):
        series[name] = group.set_index("date").sort_index()
    log.info(f"  Loaded {len(series)} instruments, {len(df):,} total rows.")
    return series


def build_global_calendar(series: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """
    Union of all dates any instrument has a record for.
    This is a conservative trading calendar: if ANY instrument has a record
    on day X, we treat X as a valid trading day.
    """
    all_dates = set()
    for df in series.values():
        all_dates.update(df.index.tolist())
    return pd.DatetimeIndex(sorted(all_dates))


def audit_instrument(name: str, df: pd.DataFrame, global_cal: pd.DatetimeIndex) -> dict:
    """Run all quality checks on one instrument. Returns a result dict."""
    close  = df["close"].dropna()
    volume = df["volume"]
    dates  = df.index

    result = {
        "name":             name,
        "ticker":           df["ticker"].iloc[0],
        "sector":           df["sector"].iloc[0],
        "rows":             len(df),
        "start":            dates.min().date(),
        "end":              dates.max().date(),
        "days_in_global":   len(global_cal[(global_cal >= dates.min()) & (global_cal <= dates.max())]),
        "missing_days":     0,
        "pct_coverage":     0.0,
        "max_gap_days":     0,
        "gap_dates":        [],       # list of (date, gap_size) for gaps > threshold
        "zero_volume_days": 0,
        "stale_price_runs": 0,        # number of stale-price episodes
        "spike_count":      0,        # number of flagged return spikes
        "spike_dates":      [],       # list of (date, return_pct, z_score)
        "price_zeros":      0,
    }

    # ── 1. Coverage ────────────────────────────────────────────────────────────
    expected = global_cal[(global_cal >= dates.min()) & (global_cal <= dates.max())]
    missing  = expected.difference(dates)
    result["missing_days"]  = len(missing)
    result["pct_coverage"]  = round(len(dates) / len(expected) * 100, 2) if len(expected) else 100.0

    # ── 2. Date gaps ──────────────────────────────────────────────────────────
    sorted_dates = sorted(dates)
    gaps = []
    for i in range(1, len(sorted_dates)):
        gap = (sorted_dates[i] - sorted_dates[i - 1]).days
        if gap > GAP_THRESHOLD:
            gaps.append((sorted_dates[i - 1].date(), gap))
    result["max_gap_days"] = max((g for _, g in gaps), default=0)
    result["gap_dates"]    = sorted(gaps, key=lambda x: -x[1])[:5]  # top 5 worst gaps

    # ── 3. Zero / null volume ─────────────────────────────────────────────────
    result["zero_volume_days"] = int((volume == 0).sum() + volume.isna().sum())

    # ── 4. Price = 0 ──────────────────────────────────────────────────────────
    result["price_zeros"] = int((close == 0).sum())

    # ── 5. Stale prices ───────────────────────────────────────────────────────
    stale_runs = 0
    run_len    = 1
    vals       = close.values
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            run_len += 1
            if run_len == STALE_WINDOW:
                stale_runs += 1
        else:
            run_len = 1
    result["stale_price_runs"] = stale_runs

    # ── 6. Price spikes (z-score on log returns) ──────────────────────────────
    if len(close) > 2:
        log_returns = np.log(close / close.shift(1)).dropna()
        mu    = log_returns.mean()
        sigma = log_returns.std()

        if sigma > 0:
            z_scores = (log_returns - mu) / sigma
            flagged  = z_scores[z_scores.abs() > Z_THRESHOLD]

            spikes = []
            for dt, z in flagged.items():
                ret_pct = (np.exp(log_returns.loc[dt]) - 1) * 100
                spikes.append((dt.date(), round(ret_pct, 2), round(z, 1)))

            result["spike_count"] = len(spikes)
            result["spike_dates"] = sorted(spikes, key=lambda x: -abs(x[2]))[:10]

    return result


def print_report(results: list[dict]):
    """Print a formatted terminal report."""
    SECTORS = ["Energy", "Metals", "Agriculture", "Livestock", "Digital Assets"]

    print()
    print("=" * 100)
    print("  DATA QUALITY AUDIT REPORT")
    print("=" * 100)

    # ── Overall summary ────────────────────────────────────────────────────────
    total_rows    = sum(r["rows"]             for r in results)
    total_missing = sum(r["missing_days"]     for r in results)
    total_spikes  = sum(r["spike_count"]      for r in results)
    total_zero_v  = sum(r["zero_volume_days"] for r in results)
    total_stale   = sum(r["stale_price_runs"] for r in results)
    total_zeros   = sum(r["price_zeros"]      for r in results)

    print(f"\n  OVERALL")
    print(f"  {'Instruments audited:':<35} {len(results)}")
    print(f"  {'Total price rows:':<35} {total_rows:,}")
    print(f"  {'Total missing trading days:':<35} {total_missing:,}")
    print(f"  {'Total flagged spikes (≥{Z_THRESHOLD}σ):':<35} {total_spikes:,}")
    print(f"  {'Total zero-volume days:':<35} {total_zero_v:,}")
    print(f"  {'Total stale-price runs:':<35} {total_stale:,}")
    print(f"  {'Total price=0 entries:':<35} {total_zeros:,}")

    # ── Per-sector detail ──────────────────────────────────────────────────────
    by_sector = {}
    for r in results:
        by_sector.setdefault(r["sector"], []).append(r)

    col_w = [32, 10, 7, 8, 8, 9, 8, 9, 10]
    header = (f"  {'Instrument':<{col_w[0]}} {'Ticker':<{col_w[1]}} "
              f"{'Rows':>{col_w[2]}} {'Start':<{col_w[3]}} {'End':<{col_w[4]}} "
              f"{'Cover%':>{col_w[5]}} {'Gaps':>{col_w[6]}} {'ZeroVol':>{col_w[7]}} "
              f"{'Spikes':>{col_w[8]}}")

    for sector in SECTORS:
        if sector not in by_sector:
            continue
        print(f"\n  ── {sector} {'─' * (90 - len(sector))}")
        print(header)
        print("  " + "-" * 98)
        for r in sorted(by_sector[sector], key=lambda x: x["name"]):
            cov_flag = "⚠️ " if r["pct_coverage"] < 98 else "  "
            print(
                f"  {r['name']:<{col_w[0]}} {r['ticker']:<{col_w[1]}} "
                f"{r['rows']:>{col_w[2]}} {str(r['start']):<{col_w[3]}} "
                f"{str(r['end']):<{col_w[4]}} "
                f"{cov_flag}{r['pct_coverage']:>{col_w[5]-2}.1f}% "
                f"{r['max_gap_days']:>{col_w[6]}} "
                f"{r['zero_volume_days']:>{col_w[7]}} "
                f"{r['spike_count']:>{col_w[8]}}"
            )

    # ── Spike detail ──────────────────────────────────────────────────────────
    print(f"\n\n  FLAGGED SPIKES  (|z| ≥ {Z_THRESHOLD} — probable roll artifacts or data errors)")
    print(f"  {'Instrument':<32} {'Date':<12} {'Return':>9}  {'Z-score':>8}  {'Likely cause'}")
    print("  " + "-" * 90)

    all_spikes = []
    for r in results:
        for dt, ret, z in r["spike_dates"]:
            all_spikes.append((r["name"], r["ticker"], dt, ret, z))

    # Sort by absolute z-score descending
    all_spikes.sort(key=lambda x: -abs(x[4]))

    is_futures = lambda t: t.endswith("=F")
    for name, ticker, dt, ret, z in all_spikes:
        cause = "roll artifact (futures)" if is_futures(ticker) else "data error or ETF event"
        print(f"  {name:<32} {str(dt):<12} {ret:>+8.2f}%  {z:>8.1f}σ  {cause}")

    if not all_spikes:
        print("  No spikes detected.")

    # ── Large gaps detail ─────────────────────────────────────────────────────
    print(f"\n\n  LARGE DATE GAPS  (> {GAP_THRESHOLD} calendar days between consecutive records)")
    print(f"  {'Instrument':<32} {'Date after gap':<16} {'Gap (days)'}")
    print("  " + "-" * 65)

    all_gaps = []
    for r in results:
        for dt, gap in r["gap_dates"]:
            all_gaps.append((r["name"], dt, gap))
    all_gaps.sort(key=lambda x: -x[2])

    for name, dt, gap in all_gaps[:30]:
        print(f"  {name:<32} {str(dt):<16} {gap}")

    if not all_gaps:
        print("  No large gaps detected.")

    print("\n" + "=" * 100)


def save_csv(results: list[dict], path: Path):
    """Save flat summary CSV to the reports directory."""
    flat = []
    for r in results:
        flat.append({
            "name":             r["name"],
            "ticker":           r["ticker"],
            "sector":           r["sector"],
            "rows":             r["rows"],
            "start":            r["start"],
            "end":              r["end"],
            "pct_coverage":     r["pct_coverage"],
            "missing_days":     r["missing_days"],
            "max_gap_days":     r["max_gap_days"],
            "zero_volume_days": r["zero_volume_days"],
            "stale_price_runs": r["stale_price_runs"],
            "spike_count":      r["spike_count"],
            "price_zeros":      r["price_zeros"],
            "top_spike_date":   r["spike_dates"][0][0]  if r["spike_dates"] else "",
            "top_spike_return": r["spike_dates"][0][1]  if r["spike_dates"] else "",
            "top_spike_z":      r["spike_dates"][0][2]  if r["spike_dates"] else "",
        })
    df = pd.DataFrame(flat)
    df.to_csv(path, index=False)
    log.info(f"Report saved to {path}")


def run_audit():
    series     = load_all_prices()
    global_cal = build_global_calendar(series)
    log.info(f"Global trading calendar: {len(global_cal)} unique dates  "
             f"({global_cal.min().date()} → {global_cal.max().date()})")

    results = []
    for name, df in sorted(series.items()):
        results.append(audit_instrument(name, df, global_cal))

    print_report(results)

    REPORTS_DIR.mkdir(exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    save_csv(results, REPORTS_DIR / f"audit_{today}.csv")

    return results


if __name__ == "__main__":
    run_audit()
