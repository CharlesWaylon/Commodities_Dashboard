"""
COT ingestor — pulls CFTC Commitments-of-Traders managed-money net positioning
and writes release-dated rows into the point-in-time fundamental store.

Idempotent: re-running never duplicates (upsert keyed on
source/series_id/reference_date/release_date). Safe to schedule weekly via
launchd alongside the price ingest.

Run manually:
    python -m services.cot_ingest
    python -m services.cot_ingest --start 2020-01-01 --series 067651 002602

Gated by FUNDAMENTAL_FEEDS_ENABLED (default off) so the feed only runs once
explicitly enabled — pre-flag behaviour is a no-op, per the Evolution Rule.

CFTC CODE VERIFICATION (MODEL VERIFICATION RULE)
────────────────────────────────────────────────
Every contract-market code below was verified on 2026-06-04 against the live CFTC
Disaggregated Futures-Only catalog (resource 72hh-3qpy, ``contract_market_name`` /
``market_and_exchange_names`` for 2025+ reports). The prior version of this map had
the GRAIN codes wrong (001602 was labelled "Corn" but CFTC 001602 is WHEAT-SRW;
002602 is CORN; 005602 is SOYBEANS; 007601 is SOYBEAN OIL) and used instrument
names that did not match the price-panel display names — neither would have joined.
See MODEL_VERIFICATION_LOG.md (2026-06-04 entry) for the catalog audit.

The map VALUE is the exact ``data.universe`` / ``MODELING_COMMODITIES`` display
name, so the persisted ``instrument`` column joins straight onto price-panel
columns for cross-sectional COT signals.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, Optional

from data.adapters.cftc_adapter import CftcAdapter
from data.config import fundamental_feeds_enabled, load_env
from data import fundamental_store as store

logger = logging.getLogger(__name__)

# CFTC contract-market code -> canonical instrument display name (data.universe).
# Verified against the live CFTC catalog on 2026-06-04 (see module docstring).
# 27 liquid futures with managed-money COT — well above the gate's cross-sectional
# breadth floor. ETF/index proxies (carbon, LNG, coal, uranium, iron-ore, lithium,
# rare-earths, HRC steel) and crypto have no managed-money line in this report and
# are intentionally omitted.
DEFAULT_SERIES = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "067651": "WTI Crude Oil",        # CRUDE OIL, LIGHT SWEET-WTI - NYMEX
    "06765T": "Brent Crude Oil",      # BRENT LAST DAY - NYMEX
    "023651": "Natural Gas",          # NAT GAS NYME (Henry Hub) - NYMEX
    "111659": "Gasoline (RBOB)",      # GASOLINE RBOB - NYMEX
    "022651": "Heating Oil",          # NY HARBOR ULSD - NYMEX
    # ── Metals ────────────────────────────────────────────────────────────────
    "088691": "Gold (COMEX)",         # GOLD - COMEX
    "084691": "Silver (COMEX)",       # SILVER - COMEX
    "085692": "Copper (COMEX)",       # COPPER- #1 - COMEX
    "076651": "Platinum",             # PLATINUM - NYMEX
    "075651": "Palladium",            # PALLADIUM - NYMEX
    "191691": "Aluminum (COMEX)",     # ALUMINUM - COMEX
    # ── Agriculture (grains/oilseeds) ─────────────────────────────────────────
    "002602": "Corn (CBOT)",          # CORN - CBOT
    "001602": "Wheat (CBOT SRW)",     # WHEAT-SRW - CBOT
    "001612": "Wheat (KC HRW)",       # WHEAT-HRW - CBOT
    "005602": "Soybeans (CBOT)",      # SOYBEANS - CBOT
    "026603": "Soybean Meal",         # SOYBEAN MEAL - CBOT
    "007601": "Soybean Oil",          # SOYBEAN OIL - CBOT
    "004603": "Oats (CBOT)",          # OATS - CBOT
    "039601": "Rough Rice (CBOT)",    # ROUGH RICE - CBOT
    # ── Agriculture (softs) ───────────────────────────────────────────────────
    "083731": "Coffee",               # COFFEE C - ICE
    "073732": "Cocoa",                # COCOA - ICE
    "080732": "Sugar",                # SUGAR NO. 11 - ICE
    "033661": "Cotton",               # COTTON NO. 2 - ICE
    "040701": "Orange Juice (FCOJ-A)",  # FRZN CONCENTRATED ORANGE JUICE - ICE
    # ── Livestock ─────────────────────────────────────────────────────────────
    "057642": "Live Cattle",          # LIVE CATTLE - CME
    "061641": "Feeder Cattle",        # FEEDER CATTLE - CME
    "054642": "Lean Hogs",            # LEAN HOGS - CME
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

    # Attach the canonical instrument display name so cross-sectional COT signals
    # join straight onto price-panel columns. Codes outside the map (custom
    # --series) get a NULL instrument, which the store persists as None.
    df["instrument"] = df["series_id"].map(DEFAULT_SERIES)
    n_mapped = df.loc[df["instrument"].notna(), "series_id"].nunique()
    n = store.write_observations(df, default_unit="contracts_net")
    logger.info(
        "cot_ingest: wrote %d rows across %d series (%d mapped to instruments).",
        n, df["series_id"].nunique(), n_mapped,
    )
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
