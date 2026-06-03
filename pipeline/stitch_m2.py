"""
M2 stitcher — build genuine constant-maturity M1/M2 series from the dated
contract rows that pipeline/ingest_contracts.py downloaded.

WHAT THIS DOES
──────────────
For every commodity in config/futures_calendar.py:

  1. Read every dated-contract series stored in ``price_history`` under
     ``interval = <contract_code>`` (e.g. 'CLN26', 'CLQ26', …) for that commodity.
  2. For each trading day, ask the contract calendar (pipeline/contract_calendar.py)
     which *listed* contract is genuinely M1 (front) and which is M2 (second-nearby)
     on that date, and take that day's close from the matching contract series.
  3. Write the resulting two constant-maturity series back into ``price_history``:
       interval = '1d_m1_raw'   → the genuine front (raw, un-adjusted)
       interval = '1d_m2'       → the genuine second-nearby (raw, un-adjusted)
     Both legs are raw closes from the *same* contract universe, so
     basis = log(M1_raw / M2_raw) is a clean near-term calendar spread on every
     historical date — never a fixed far-dated contract masquerading as M2, and
     never a roll-adjusted-vs-raw mix.

WHY DELETE-THEN-INSERT
──────────────────────
The first M2 attempt stored a single fixed dated contract per commodity under
'1d_m2'.  Those rows are economically invalid for history.  This stitcher DELETEs
all existing '1d_m2' / '1d_m1_raw' rows per commodity before inserting the freshly
stitched series, so re-running it is idempotent AND it cleans out the old pollution.

PURE CORE / DB SHELL
────────────────────
``stitch_constant_maturity`` is pure (dict of series in → dict of series out) and
is unit-tested in the sandbox (pipeline/test_stitch_m2.py).  Only
``run_stitch`` touches Postgres, so the calendar→leg-selection logic is verified
without a database.

RUN (on the Mac — needs Postgres; run AFTER ingest_contracts):
  python -m pipeline.ingest_contracts
  python -m pipeline.stitch_m2
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy.exc import IntegrityError

from database.db import init_db, get_db
from database.models import Commodity, PriceHistory
from pipeline.contract_calendar import (
    listed_contracts,
    nearby_on,
    contract_code,
    CONTRACT_CODE_RE,
)
from config.futures_calendar import ContractSpec, FUTURES_CONTRACT_SPECS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Storage interval tags for the stitched outputs.
M1_RAW_INTERVAL = "1d_m1_raw"
M2_INTERVAL     = "1d_m2"


# ── Pure core ────────────────────────────────────────────────────────────────────

def stitch_constant_maturity(
    spec: ContractSpec,
    commodity: str,
    contract_series: dict[str, pd.Series],
    max_depth: int = 2,
) -> dict[int, pd.Series]:
    """
    Build constant-maturity nearby series from per-contract price series.

    Parameters
    ----------
    spec : ContractSpec for this commodity.
    commodity : display name (only used for calendar bookkeeping).
    contract_series : mapping ``contract_code -> Series(index=date, values=close)``.
        Index entries must be ``datetime.date`` objects.  These are the raw dated
        contract closes read from price_history.
    max_depth : how many nearby slots to stitch (2 → M1 and M2).

    Returns
    -------
    dict[int, pd.Series]
        ``{1: m1_series, 2: m2_series, …}`` — each indexed by ``date``, holding
        the close of whichever listed contract is the depth-th nearby on that day.
        Dates where the resolved contract has no observation are omitted (so the
        two legs can have slightly different coverage at the edges; the feature
        layer aligns them).
    """
    out: dict[int, dict[date, float]] = {d: {} for d in range(1, max_depth + 1)}

    # Union of every observation date across all contracts for this commodity.
    all_dates: set[date] = set()
    for s in contract_series.values():
        all_dates.update(s.index)
    if not all_dates:
        return {d: pd.Series(dtype=float) for d in range(1, max_depth + 1)}

    sorted_dates = sorted(all_dates)
    start, end = sorted_dates[0], sorted_dates[-1]

    # Build the listed-contract universe once and reuse it for every date.
    contracts = listed_contracts(spec, commodity, start, end)

    for d in sorted_dates:
        for depth in range(1, max_depth + 1):
            c = nearby_on(spec, commodity, d, depth, contracts)
            if c is None:
                continue
            code = contract_code(c.ticker)
            s = contract_series.get(code)
            if s is None:
                continue
            # Take the resolved contract's close on this exact date, if it traded.
            if d in s.index:
                val = s.loc[d]
                if pd.notna(val):
                    out[depth][d] = float(val)

    return {
        depth: pd.Series(vals, dtype=float).sort_index()
        for depth, vals in out.items()
    }


# ── DB helpers ─────────────────────────────────────────────────────────────────

def resolve_commodity_id(db, spec: ContractSpec, name: str) -> int | None:
    """
    Resolve a commodity's id, joining on the STABLE ``ticker`` column first.

    ``commodities.name`` was seeded inconsistently (some bare like 'Copper', some
    suffixed like 'Wheat (KC HRW)'), so matching on the display name silently
    skipped ~12 commodities.  ``commodities.ticker`` (e.g. 'CL=F') is
    ``unique=True`` and reliable, so we join on ``spec.yf_ticker`` and only fall
    back to the display name if the ticker is missing/unmatched.
    """
    row = None
    if spec.yf_ticker:
        row = db.query(Commodity).filter_by(ticker=spec.yf_ticker).first()
    if row is None:
        row = db.query(Commodity).filter_by(name=name).first()
    return row.id if row is not None else None


def _load_contract_series(db, commodity_id: int) -> dict[str, pd.Series]:
    """
    Read every dated-contract series for one commodity out of price_history.

    Picks only rows whose ``interval`` matches a contract code (CONTRACT_CODE_RE),
    so the '1d', '1wk', '1d_m2', '1d_m1_raw' series are never mistaken for
    contracts.  Uses raw ``close`` (dated contracts are not roll-adjusted).
    """
    rows = (
        db.query(PriceHistory.interval, PriceHistory.date, PriceHistory.close)
        .filter(PriceHistory.commodity_id == commodity_id)
        .all()
    )
    buckets: dict[str, dict[date, float]] = {}
    for interval, d, close in rows:
        if close is None:
            continue
        if not CONTRACT_CODE_RE.match(interval):
            continue
        buckets.setdefault(interval, {})[d] = float(close)

    return {
        code: pd.Series(vals, dtype=float).sort_index()
        for code, vals in buckets.items()
    }


def _replace_series(db, commodity_id: int, interval: str, series: pd.Series) -> int:
    """
    DELETE all existing rows for (commodity_id, interval), then INSERT the stitched
    series fresh.  Returns the number of rows inserted.

    Stores the stitched close in both ``close`` and ``adjusted_close`` with
    ``adjustment_factor = 1.0`` — these series are already the economically correct
    nearby legs; no roll adjustment is applied (roll_adjust.py ignores them, it
    only touches interval == '1d').
    """
    db.query(PriceHistory).filter(
        PriceHistory.commodity_id == commodity_id,
        PriceHistory.interval == interval,
    ).delete(synchronize_session=False)

    inserted = 0
    for d, val in series.items():
        if pd.isna(val):
            continue
        price_row = PriceHistory(
            commodity_id      = commodity_id,
            date              = d,
            close             = float(val),
            adjusted_close    = float(val),
            adjustment_factor = 1.0,
            interval          = interval,
        )
        try:
            with db.begin_nested():
                db.add(price_row)
            inserted += 1
        except IntegrityError:
            # Should not happen after the DELETE, but stay defensive.
            pass
    return inserted


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_stitch(max_depth: int = 2) -> tuple[int, int]:
    """
    Stitch constant-maturity M1/M2 series for every futures commodity and persist
    them to price_history under '1d_m1_raw' and '1d_m2'.

    Returns (m1_rows_written, m2_rows_written).
    """
    init_db()
    log.info("=" * 64)
    log.info("M2 stitch — building constant-maturity series from dated contracts")
    log.info("=" * 64)

    grand_m1 = grand_m2 = 0

    with get_db() as db:
        for name, spec in FUTURES_CONTRACT_SPECS.items():
            commodity_id = resolve_commodity_id(db, spec, name)
            if commodity_id is None:
                log.warning("skip — commodity not seeded (ticker=%s name=%r)",
                            spec.yf_ticker, name)
                continue

            contract_series = _load_contract_series(db, commodity_id)
            if not contract_series:
                log.warning("%-26s no dated-contract rows found — run ingest_contracts first",
                            name)
                continue

            stitched = stitch_constant_maturity(spec, name, contract_series, max_depth)
            m1 = stitched.get(1, pd.Series(dtype=float))
            m2 = stitched.get(2, pd.Series(dtype=float))

            n_m1 = _replace_series(db, commodity_id, M1_RAW_INTERVAL, m1)
            n_m2 = _replace_series(db, commodity_id, M2_INTERVAL, m2)
            grand_m1 += n_m1
            grand_m2 += n_m2

            span = ""
            if not m2.empty:
                span = f"  [{m2.index[0]} … {m2.index[-1]}]"
            log.info("%-26s contracts=%2d  M1=%4d  M2=%4d%s",
                     name, len(contract_series), n_m1, n_m2, span)

    log.info("=" * 64)
    log.info("Stitch complete.  M1 rows=%d  M2 rows=%d", grand_m1, grand_m2)
    log.info("Next: retrain (python -m models.daily_retrain --period 3y)")
    log.info("=" * 64)
    return grand_m1, grand_m2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stitch constant-maturity M1/M2 from dated contracts")
    p.add_argument("--max-depth", type=int, default=2)
    args = p.parse_args()
    run_stitch(max_depth=args.max_depth)
