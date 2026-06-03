"""
COT ingestor — pulls CFTC Commitments-of-Traders managed-money net positioning
and writes release-dated rows into the point-in-time fundamental store.

Idempotent: re-running never duplicates (upsert keyed on
source/series_id/reference_date/release_date). Safe to schedule weekly via
launchd alongside the price ingest.

Run manually:
    python -m services.cot_ingest
    python -m services.cot_ingest --start 2020-01-01 --series 067651 001602

Gated by FUNDAMENTAL_FEEDS_ENABLED (default off) so the feed only runs once
explicitly enabled — pre-flag behaviour is a no-op, per the Evolution Rule.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional

from data.adapters.cftc_adapter import CftcAdapter
from data.config import fundamental_feeds_enabled, load_env
from data import fundamental_store as store

logger = logging.getLogger(__name__)

# Default contracts to track — CFTC contract-market codes for the liquid
# commodity futures in our universe. (Extend as needed; non-engineers can pass
# --series to override without touching code.)
DEFAULT_SERIES = {
    "067651": "WTI Crude Oil",       # CRUDE OIL, LIGHT SWEET - NYMEX
    "023651": "Natural Gas",         # NATURAL GAS - NYMEX (Henry Hub)
    "088691": "Gold",                # GOLD - COMMODITY EXCHANGE INC.
    "084691": "Silver",              # SILVER - COMMODITY EXCHANGE INC.
    "085692": "Copper",              # COPPER - #1 - COMMODITY EXCHANGE INC.
    "001602": "Corn",                # CORN - CHICAGO BOARD OF TRADE
    "005602": "Wheat",               # WHEAT-SRW - CHICAGO BOARD OF TRADE
    "007601": "Soybeans",            # SOYBEANS - CHICAGO BOARD OF TRADE
}


def run(series_ids: Optional[Iterable[str]] = None, start: Optional[str] = None) -> int:
    """Fetch COT net managed-money positioning and upsert; returns rows written."""
    load_env()
    if not fundamental_feeds_enabled():
        logger.info("cot_ingest: FUNDAMENTAL_FEEDS_ENABLED is off — skipping.")
        return 0

    codes = list(series_ids) if series_ids else list(DEFAULT_SERIES.keys())
    adapter = CftcAdapter()
    df = adapter.get_observations(codes, start=start)
    if df.empty:
        logger.warning("cot_ingest: no observations returned for %d series.", len(codes))
        return 0
    n = store.write_observations(df, default_unit="contracts_net")
    logger.info("cot_ingest: wrote %d rows across %d series.", n, df["series_id"].nunique())
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest CFTC COT managed-money positioning.")
    ap.add_argument("--start", default=None, help="ISO start date (e.g. 2020-01-01).")
    ap.add_argument("--series", nargs="*", default=None, help="CFTC contract-market codes.")
    args = ap.parse_args()
    n = run(series_ids=args.series, start=args.start)
    print(f"cot_ingest: {n} rows written.")


if __name__ == "__main__":
    main()
