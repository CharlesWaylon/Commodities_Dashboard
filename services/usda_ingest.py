"""
USDA ingestor — pulls USDA NASS QuickStats agricultural fundamentals (ending
stocks for the grains/oilseeds in our universe) and writes release-dated rows
into the point-in-time fundamental store.

Idempotent upsert; schedule monthly via launchd (WASDE cadence). Requires
USDA_QUICKSTATS_KEY.

Run manually:
    python -m services.usda_ingest
    python -m services.usda_ingest --start 2018

Gated by FUNDAMENTAL_FEEDS_ENABLED (default off).
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional

from data.adapters.usda_adapter import UsdaAdapter
from data.config import fundamental_feeds_enabled
from data import fundamental_store as store

logger = logging.getLogger(__name__)

# series_id -> QuickStats query. Ending stocks (STOCKS, ENDING) at the national
# level, annual marketing year. Non-engineers can extend this map.
DEFAULT_QUERIES = {
    "CORN_ENDING_STOCKS": {
        "commodity_desc": "CORN",
        "statisticcat_desc": "STOCKS",
        "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
        "agg_level_desc": "NATIONAL",
    },
    "SOYBEANS_ENDING_STOCKS": {
        "commodity_desc": "SOYBEANS",
        "statisticcat_desc": "STOCKS",
        "short_desc": "SOYBEANS - STOCKS, MEASURED IN BU",
        "agg_level_desc": "NATIONAL",
    },
    "WHEAT_ENDING_STOCKS": {
        "commodity_desc": "WHEAT",
        "statisticcat_desc": "STOCKS",
        "short_desc": "WHEAT - STOCKS, MEASURED IN BU",
        "agg_level_desc": "NATIONAL",
    },
}


def run(series_ids: Optional[Iterable[str]] = None, start: Optional[str] = None) -> int:
    if not fundamental_feeds_enabled():
        logger.info("usda_ingest: FUNDAMENTAL_FEEDS_ENABLED is off — skipping.")
        return 0

    adapter = UsdaAdapter(series_queries=DEFAULT_QUERIES)
    if not adapter._api_key:
        logger.warning("usda_ingest: USDA_QUICKSTATS_KEY not set — skipping.")
        return 0
    ids = list(series_ids) if series_ids else list(DEFAULT_QUERIES.keys())
    df = adapter.get_observations(ids, start=start)
    if df.empty:
        logger.warning("usda_ingest: no observations returned for %d series.", len(ids))
        return 0
    n = store.write_observations(df, default_unit="bushels")
    logger.info("usda_ingest: wrote %d rows across %d series.", n, df["series_id"].nunique())
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ingest USDA NASS ending stocks.")
    ap.add_argument("--start", default=None, help="Start year (e.g. 2018).")
    ap.add_argument("--series", nargs="*", default=None, help="Series keys from DEFAULT_QUERIES.")
    args = ap.parse_args()
    n = run(series_ids=args.series, start=args.start)
    print(f"usda_ingest: {n} rows written.")


if __name__ == "__main__":
    main()
