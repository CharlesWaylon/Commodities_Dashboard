"""
FRED ingestor — pulls key macro series and writes release-dated rows into the
point-in-time fundamental store, so macro-surprise signals read a clean,
release-dated macro state instead of re-deriving snapshots ad hoc.

Idempotent upsert; safe to schedule (daily for the market series, the monthly
indicators only change on their release days). Requires FRED_API_KEY.

Run manually:
    python -m services.fred_ingest
    python -m services.fred_ingest --start 2010-01-01 --series DGS10 CPIAUCSL

Gated by FUNDAMENTAL_FEEDS_ENABLED (default off).

PUBLICATION-LAG HONESTY (POINT-IN-TIME)
────────────────────────────────────────
FRED's plain REST helper returns the latest-vintage value keyed on the
*reference* date (e.g. CPIAUCSL for April is dated 2024-04-01), NOT the date it
was first published. True vintage timing needs the ALFRED realtime API — a flagged
future upgrade (see data/adapters/fred_adapter.py and MODEL_VERIFICATION_LOG).

Until then we approximate ``release_date = reference_date + PUBLICATION_LAG_BDAYS``
per series, deliberately rounding the lag UP. Erring late only costs a little
signal freshness; erring early would be look-ahead. The monthly indicators carry
the largest, most important lags (CPI/PPI ≈ 6 weeks after the reference month's
1st; employment ≈ 5 weeks). Daily market series get a 1-business-day lag (today's
close is usable next morning). Verified against published BLS/Fed release
calendars on 2026-06-04 (see MODEL_VERIFICATION_LOG).
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional

from data.adapters.fred_adapter import FredAdapter
from data.config import fundamental_feeds_enabled, load_env
from data import fundamental_store as store

logger = logging.getLogger(__name__)

# FRED series id -> human description. Market-wide macro state (no per-instrument
# mapping — the instrument column is intentionally NULL for FRED).
DEFAULT_SERIES = {
    # ── Inflation ──────────────────────────────────────────────────────────────
    "CPIAUCSL": "CPI, all urban consumers (monthly, SA)",
    "PPIACO": "PPI, all commodities (monthly)",
    "T10YIE": "10y breakeven inflation expectation (daily)",
    # ── Labor ──────────────────────────────────────────────────────────────────
    "UNRATE": "Unemployment rate (monthly)",
    "PAYEMS": "Total nonfarm payrolls (monthly)",
    # ── Real activity ──────────────────────────────────────────────────────────
    "INDPRO": "Industrial production index (monthly)",
    # ── Rates / curve ──────────────────────────────────────────────────────────
    "DGS10": "10y Treasury constant-maturity yield (daily)",
    "DGS2": "2y Treasury constant-maturity yield (daily)",
    "T10Y2Y": "10y-2y Treasury spread (daily)",
    "DFF": "Effective federal funds rate (daily)",
    # ── Risk / USD ─────────────────────────────────────────────────────────────
    "VIXCLS": "CBOE volatility index (daily)",
    "DTWEXBGS": "Broad trade-weighted USD index (daily)",
}

# Per-series publication lag in BUSINESS days (rounded up; see module docstring).
# Anything not listed defaults to DEFAULT_LAG_BDAYS.
DEFAULT_LAG_BDAYS = 1  # daily market series: today's close usable next morning
# Counted against the reference month's 1st and rounded UP past the latest
# plausible release so we never lead the real print (verified vs the May-2024 BLS
# calendar: April CPI ref 2024-04-01 released 2024-05-15 = 33 bdays, so 34 is safe).
PUBLICATION_LAG_BDAYS = {
    "CPIAUCSL": 34,   # BLS CPI ~10th-15th of the following month
    "PPIACO": 34,     # BLS PPI ~mid of the following month
    "INDPRO": 35,     # Fed G.17 ~15th-17th of the following month
    "UNRATE": 27,     # BLS employment situation, first Friday of following month
    "PAYEMS": 27,     # BLS employment situation, first Friday of following month
}


def run(series_ids: Optional[Iterable[str]] = None, start: Optional[str] = None) -> int:
    """Fetch FRED macro series and upsert release-dated rows; returns rows written."""
    load_env()
    if not fundamental_feeds_enabled():
        logger.info("fred_ingest: FUNDAMENTAL_FEEDS_ENABLED is off — skipping.")
        return 0

    ids = list(series_ids) if series_ids else list(DEFAULT_SERIES.keys())
    adapter = FredAdapter(
        publication_lag_bdays=DEFAULT_LAG_BDAYS,
        per_series_lag_bdays=PUBLICATION_LAG_BDAYS,
    )
    df = adapter.get_observations(ids, start=start)
    if df.empty:
        logger.warning("fred_ingest: no observations returned for %d series.", len(ids))
        return 0
    n = store.write_observations(df)
    logger.info("fred_ingest: wrote %d rows across %d series.", n, df["series_id"].nunique())
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest FRED macro series (release-dated).")
    ap.add_argument("--start", default=None, help="ISO start date (e.g. 2010-01-01).")
    ap.add_argument("--series", nargs="*", default=None, help="FRED series ids.")
    args = ap.parse_args()
    n = run(series_ids=args.series, start=args.start)
    print(f"fred_ingest: {n} rows written.")


if __name__ == "__main__":
    main()
