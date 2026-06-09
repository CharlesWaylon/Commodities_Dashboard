"""
EIA ingestor — pulls EIA weekly petroleum & natural-gas stocks and writes
release-dated rows into the point-in-time fundamental store.

Idempotent upsert; safe to schedule weekly via launchd. Requires EIA_API_KEY.

Run manually:
    python -m services.eia_ingest
    python -m services.eia_ingest --start 2020-01-01

Gated by FUNDAMENTAL_FEEDS_ENABLED (default off).
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional

from data.adapters.eia_adapter import EiaAdapter
from data.config import fundamental_feeds_enabled, load_env
from data import fundamental_store as store

logger = logging.getLogger(__name__)

# EIA v2 series ids. Natural-gas storage publishes Thursday (lag 6); petroleum
# weekly publishes Wednesday (lag 5, the adapter default).
DEFAULT_SERIES = {
    "PET.WCESTUS1.W": "US crude oil ending stocks (weekly)",
    "PET.WGTSTUS1.W": "US total motor gasoline stocks (weekly)",
    "PET.WDISTUS1.W": "US distillate fuel oil stocks (weekly)",
    "NG.NW2_EPG0_SWO_R48_BCF.W": "US working natural gas in storage (weekly)",
}
NAT_GAS_LAG_DAYS = {"NG.NW2_EPG0_SWO_R48_BCF.W": 6}

# EIA series -> canonical instrument display name (data.universe). Distillate is
# the deliverable behind the Heating Oil / ULSD contract, so it maps there. This
# lets the inventory-surprise signal join straight onto price-panel columns.
SERIES_TO_INSTRUMENT = {
    "PET.WCESTUS1.W": "WTI Crude Oil",
    "PET.WGTSTUS1.W": "Gasoline (RBOB)",
    "PET.WDISTUS1.W": "Heating Oil",
    "NG.NW2_EPG0_SWO_R48_BCF.W": "Natural Gas",
}


def run(series_ids: Optional[Iterable[str]] = None, start: Optional[str] = None) -> int:
    load_env()
    if not fundamental_feeds_enabled():
        logger.info("eia_ingest: FUNDAMENTAL_FEEDS_ENABLED is off — skipping.")
        return 0

    ids = list(series_ids) if series_ids else list(DEFAULT_SERIES.keys())
    adapter = EiaAdapter(per_series_lag_days=NAT_GAS_LAG_DAYS)
    if not adapter._api_key:
        logger.warning("eia_ingest: EIA_API_KEY not set — skipping.")
        return 0
    df = adapter.get_observations(ids, start=start)
    if df.empty:
        logger.warning("eia_ingest: no observations returned for %d series.", len(ids))
        return 0
    df["instrument"] = df["series_id"].map(SERIES_TO_INSTRUMENT)
    n = store.write_observations(df)
    logger.info("eia_ingest: wrote %d rows across %d series.", n, df["series_id"].nunique())
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest EIA weekly petroleum / nat-gas stocks.")
    ap.add_argument("--start", default=None, help="ISO start date (e.g. 2020-01-01).")
    ap.add_argument("--series", nargs="*", default=None, help="EIA v2 series ids.")
    args = ap.parse_args()
    n = run(series_ids=args.series, start=args.start)
    print(f"eia_ingest: {n} rows written.")


if __name__ == "__main__":
    main()
