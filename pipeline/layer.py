"""
Step 4: Separate instruments into clean analytical layers by type.

THE PROBLEM:
  The database currently stores Peabody Energy (BTU — a coal mining stock) and
  WTI Crude Oil (CL=F — a direct commodity futures contract) as if they are the
  same kind of thing. They are not, and the difference matters enormously for
  any analysis you build on top of this data.

  When you include equity proxies in a correlation matrix alongside futures, you
  introduce "equity market beta" — the tendency of all stocks to rise and fall
  together with the broader market. If the S&P 500 drops 3% on a bad day, BTU,
  HCC, LNG, and GLNCY will all drop roughly in line with equities, making them
  appear correlated with each other and with commodity futures. But that
  correlation is driven by shared equity exposure, not by any real relationship
  between thermal coal prices and LNG export demand. You'd be measuring the
  wrong thing.

  The same problem applies to model training. An LSTM trained on a mix of
  commodity futures and equity proxies will learn patterns that are partly
  driven by equity market microstructure — patterns that have nothing to do
  with commodity supply and demand.

THE FIX — two parts:

  Part 1: Classify every instrument.
    Add an `instrument_type` column to the `commodities` table with one of:
      'futures'       — direct commodity futures contract, roll-adjusted
                        28 instruments (CL=F, GC=F, ZC=F, etc.)
      'etf_proxy'     — ETF holding a basket of related assets
                        8 instruments (URA, KRBN, SGOL, etc.)
      'equity_proxy'  — single company stock whose business approximates
                        exposure to one commodity
                        4 instruments (LNG, BTU, HCC, GLNCY)
      'crypto'        — digital asset, not a commodity in any legal/market sense
                        1 instrument (BTC-USD)

  Part 2: Create database views.
    A VIEW is a saved SELECT statement stored in the database. It behaves
    exactly like a table — you can query it with SELECT — but it contains
    no data of its own. Every time you query a view, the database runs the
    underlying SELECT and returns the current result.

    This means:
      - Zero additional storage (unlike a copied table)
      - Always up-to-date (when aligned_prices changes, the view reflects it
        immediately — no rebuild step needed)
      - Clean API for downstream code: `SELECT * FROM v_futures_aligned`
        is self-documenting and immune to accidental inclusion of proxies

    Views created:
      v_futures_aligned  — 28 futures × 1,258 days; use for commodity analysis
      v_proxies_aligned  — 12 ETF/equity proxies × 1,258 days; use separately
      v_crypto_aligned   — 1 crypto × 1,258 days
      v_all_typed        — all 41 instruments with instrument_type column attached

WHY NOT JUST FILTER IN PYTHON?
  You could filter in pandas every time:
    df = df[df["ticker"].str.endswith("=F")]
  But:
    1. It's easy to forget — a new team member queries aligned_prices without
       knowing about the proxy contamination problem.
    2. It doesn't scale — once you have 200 instruments, remembering which are
       proxies becomes error-prone.
    3. Views push the knowledge into the database itself, where it's enforced
       structurally rather than relying on everyone remembering a convention.

RUN:
  python -m pipeline.layer

SAFE TO RE-RUN:
  The script drops and recreates views every time — idempotent.
  It uses ALTER TABLE ADD COLUMN IF NOT EXISTS for the new column, so it
  won't fail if the column already exists.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.db import get_db, get_engine, init_db
from database.models import Commodity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Classification registry ────────────────────────────────────────────────────
# Tickers that are ETFs (hold a basket of assets)
ETF_TICKERS = {"URA", "KRBN", "SGOL", "SIVR", "SLX", "LIT", "REMX", "WOOD"}

# Tickers that are single-company equity proxies
EQUITY_TICKERS = {"LNG", "BTU", "HCC", "GLNCY"}

# Crypto — treated separately from both futures and equities
CRYPTO_TICKERS = {"BTC-USD"}


def classify(ticker: str) -> str:
    """Return the instrument_type string for a given ticker."""
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in ETF_TICKERS:
        return "etf_proxy"
    if ticker in EQUITY_TICKERS:
        return "equity_proxy"
    if ticker.endswith("=F"):
        return "futures"
    # Fallback — should never be reached with the current instrument set
    return "unknown"


# ── View definitions ───────────────────────────────────────────────────────────
# Each view JOINs aligned_prices with commodities to produce a self-contained
# result set that includes both price data and instrument metadata.
# All views share the same column layout so they're interchangeable in code.

_VIEW_SELECT = """
    SELECT
        ap.date,
        c.id          AS commodity_id,
        c.name,
        c.ticker,
        c.sector,
        c.unit,
        c.instrument_type,
        ap.adjusted_close,
        ap.is_filled
    FROM aligned_prices ap
    JOIN commodities c ON c.id = ap.commodity_id
"""

VIEWS = {
    "v_futures_aligned": f"{_VIEW_SELECT} WHERE c.instrument_type = 'futures'",
    "v_proxies_aligned": f"{_VIEW_SELECT} WHERE c.instrument_type IN ('etf_proxy', 'equity_proxy')",
    "v_crypto_aligned":  f"{_VIEW_SELECT} WHERE c.instrument_type = 'crypto'",
    "v_all_typed":       _VIEW_SELECT,   # no filter — all 41 instruments
}


def add_instrument_type_column(engine):
    """
    Add instrument_type column to the commodities table if it doesn't exist.
    Uses IF NOT EXISTS so it's safe to call repeatedly.
    PostgreSQL and SQLite both support this syntax.
    """
    with engine.connect() as conn:
        conn.execute(text(
            "ALTER TABLE commodities ADD COLUMN IF NOT EXISTS instrument_type VARCHAR(20)"
        ))
        conn.commit()
    log.info("  instrument_type column confirmed in commodities table.")


def classify_and_write(db):
    """Read every commodity row, classify it, write instrument_type back."""
    commodities = db.query(Commodity).all()
    counts = {"futures": 0, "etf_proxy": 0, "equity_proxy": 0, "crypto": 0}

    for c in commodities:
        itype = classify(c.ticker)
        c.instrument_type = itype
        counts[itype] = counts.get(itype, 0) + 1

    log.info(f"  Classified {len(commodities)} instruments: {counts}")
    return commodities, counts


def create_views(engine):
    """Drop and recreate all analytical views."""
    with engine.connect() as conn:
        for view_name, select_sql in VIEWS.items():
            conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
            conn.execute(text(f"CREATE VIEW {view_name} AS {select_sql}"))
            log.info(f"  Created view: {view_name}")
        conn.commit()


def run_layer():
    log.info("=" * 65)
    log.info("Instrument layer separation started")
    log.info("=" * 65)

    engine = get_engine()
    init_db()

    # ── Part 1: Add column and classify ───────────────────────────────────────
    log.info("\nStep 1: Adding instrument_type column to commodities...")
    add_instrument_type_column(engine)

    log.info("\nStep 2: Classifying instruments and writing types to DB...")
    with get_db() as db:
        _, counts = classify_and_write(db)
        # Read back as plain dicts while the session is still open —
        # avoids DetachedInstanceError when we access attributes after commit.
        classified = [
            {"name": c.name, "ticker": c.ticker,
             "sector": c.sector, "instrument_type": c.instrument_type}
            for c in db.query(Commodity).all()
        ]

    # ── Part 2: Create views ───────────────────────────────────────────────────
    log.info("\nStep 3: Creating analytical views...")
    create_views(engine)

    # ── Report ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  INSTRUMENT LAYER REPORT")
    print("=" * 80)

    type_order  = ["futures", "etf_proxy", "equity_proxy", "crypto"]
    type_labels = {
        "futures":       "Direct Futures",
        "etf_proxy":     "ETF Proxy",
        "equity_proxy":  "Equity Proxy",
        "crypto":        "Crypto",
    }
    type_notes = {
        "futures":      "Real commodity price, roll-adjusted. Use for commodity fundamentals analysis.",
        "etf_proxy":    "Basket of related assets. Carries fund management beta. Use separately.",
        "equity_proxy": "Single stock. Carries company-specific and equity market beta. Use separately.",
        "crypto":       "Digital asset. Non-commodity price dynamics. Use separately.",
    }

    for itype in type_order:
        label    = type_labels[itype]
        note     = type_notes[itype]
        members  = [c for c in classified if c["instrument_type"] == itype]
        if not members:
            continue

        print(f"\n  ── {label} ({len(members)} instruments) {'─' * (55 - len(label))}")
        print(f"  {note}")
        print(f"  {'Name':<35} {'Ticker':<12} {'Sector'}")
        print("  " + "-" * 70)
        for c in sorted(members, key=lambda x: (x["sector"], x["name"])):
            print(f"  {c['name']:<35} {c['ticker']:<12} {c['sector']}")

    print()
    print("  ── Views created ─────────────────────────────────────────────────────")
    view_descriptions = {
        "v_futures_aligned":  f"28 futures × calendar  →  use for commodity correlation / model training",
        "v_proxies_aligned":  f"12 proxies × calendar  →  equity/ETF analysis in isolation",
        "v_crypto_aligned":   f" 1 crypto  × calendar  →  Bitcoin standalone analysis",
        "v_all_typed":        f"41 total   × calendar  →  full dataset with instrument_type label",
    }
    for view, desc in view_descriptions.items():
        print(f"  {view:<25}  {desc}")

    print()
    print("  HOW TO USE THE VIEWS:")
    print("  All views share the same columns:")
    print("  date | commodity_id | name | ticker | sector | unit | instrument_type")
    print("  | adjusted_close | is_filled")
    print()
    print("  Example queries:")
    print("    SELECT * FROM v_futures_aligned WHERE name = 'WTI Crude Oil';")
    print("    SELECT name, COUNT(*) FROM v_proxies_aligned GROUP BY name;")
    print("    SELECT * FROM v_all_typed WHERE sector = 'Energy' ORDER BY date;")
    print()
    print("=" * 80)

    return classified, counts


if __name__ == "__main__":
    run_layer()
