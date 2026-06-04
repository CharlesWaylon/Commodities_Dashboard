"""
Deep-history backfill for the core long-history futures (RESEARCH data).

WHY
───
The production price panel only has ~5 years of COMMON history (gated by the
youngest ETF/crypto proxies), which is a single commodity-bull regime — too short
to fairly test multi-decade factors like value. The genuine futures, however, have
~24 years available on Yahoo. This script backfills that deep history for the core
futures so research backtests can span multiple regimes (2008 GFC, 2014-16 oil
crash, 2020 COVID, 2022 inflation).

ISOLATION (do not disturb production)
─────────────────────────────────────
Rows are written under ``interval='1d_deep'`` — a DISTINCT interval from the
production ``'1d'`` series. ``load_price_matrix_from_db`` filters ``interval='1d'``
and is therefore completely unaffected; only the new ``load_long_history_core_panel``
loader reads ``'1d_deep'``. Upserts are idempotent on (commodity_id, date,
interval), so re-running never duplicates.

NOTE
────
Yahoo ``=F`` series are continuous front-month (already back-adjusted by Yahoo),
internally consistent — exactly what value's multi-year price ratios need. We store
``close`` and mirror it into ``adjusted_close`` (no extra roll-adjust on this
research series). Network (yfinance) must be reachable; run on a machine with
access (the user's Mac, like the daily ingest).

Run:
    python -m services.deep_history_ingest
    python -m services.deep_history_ingest --start 2001-01-01
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from models.config import MODELING_COMMODITIES

logger = logging.getLogger(__name__)

DEEP_INTERVAL = "1d_deep"

# Core genuine futures with ~24y of Yahoo history (probed 2026-06-04). BZ=F (Brent,
# 2007) is omitted: it is redundant with WTI and would truncate the common window
# by six years. ETF/index proxies and crypto are excluded (no deep futures history).
CORE_TICKERS: List[str] = [
    "CL=F", "NG=F", "RB=F", "HO=F",                       # energy
    "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",               # metals
    "ZC=F", "ZW=F", "KE=F", "ZS=F", "ZM=F", "ZL=F",       # grains/oilseeds
    "ZO=F", "ZR=F", "KC=F", "CC=F", "SB=F", "CT=F", "OJ=F",  # grains/softs
    "LE=F", "GF=F", "HE=F",                               # livestock
]
DEFAULT_START = "2001-01-01"


def _ticker_to_commodity_id() -> Dict[str, int]:
    from sqlalchemy import text

    from database.db import get_engine

    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT id, ticker FROM commodities")).fetchall()
    return {t: i for i, t in rows}


def run(tickers: Optional[List[str]] = None, start: Optional[str] = None) -> int:
    """Fetch deep history for the core futures and upsert under 1d_deep. Returns rows written."""
    import yfinance as yf
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from database.db import get_engine, init_db
    from database.models import PriceHistory

    tickers = tickers or CORE_TICKERS
    start = start or DEFAULT_START
    tid = _ticker_to_commodity_id()
    missing = [t for t in tickers if t not in tid]
    if missing:
        logger.warning("deep_history_ingest: tickers absent from commodities table: %s", missing)
    tickers = [t for t in tickers if t in tid]
    if not tickers:
        logger.error("deep_history_ingest: no resolvable tickers.")
        return 0

    data = yf.download(tickers, start=start, progress=False, auto_adjust=False)["Close"]
    if isinstance(data, pd.Series):  # single ticker
        data = data.to_frame(tickers[0])
    if data.empty:
        logger.warning("deep_history_ingest: yfinance returned no data.")
        return 0

    init_db()
    now = datetime.now(timezone.utc)
    engine = get_engine()
    table = PriceHistory.__table__
    n = 0
    with engine.begin() as conn:
        for tk in tickers:
            if tk not in data.columns:
                continue
            s = data[tk].dropna()
            cid = tid[tk]
            for dt, close in s.items():
                rec = {
                    "commodity_id": cid, "date": pd.Timestamp(dt).date(),
                    "close": float(close), "adjusted_close": float(close),
                    "interval": DEEP_INTERVAL, "ingested_at": now,
                }
                stmt = pg_insert(table).values(**rec).on_conflict_do_update(
                    constraint="uq_commodity_date_interval",
                    set_={"close": rec["close"], "adjusted_close": rec["adjusted_close"],
                          "ingested_at": now},
                )
                conn.execute(stmt)
                n += 1
            logger.info("deep_history_ingest: %s -> %d rows (%s..%s)", tk, len(s),
                        s.index.min().date(), s.index.max().date())
    logger.info("deep_history_ingest: wrote %d rows across %d tickers under '%s'.", n, len(tickers), DEEP_INTERVAL)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Backfill deep core-futures history (research, 1d_deep).")
    ap.add_argument("--start", default=None, help="ISO start date (default 2001-01-01).")
    ap.add_argument("--tickers", nargs="*", default=None, help="override core ticker list.")
    args = ap.parse_args()
    n = run(tickers=args.tickers, start=args.start)
    print(f"deep_history_ingest: {n} rows written under '{DEEP_INTERVAL}'.")


if __name__ == "__main__":
    main()
