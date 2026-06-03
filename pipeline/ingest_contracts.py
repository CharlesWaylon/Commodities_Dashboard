"""
Dated-contract ingest — download the individual futures contracts needed to
stitch a constant-maturity M1/M2 series.

WHAT THIS DOES
──────────────
For every commodity in config/futures_calendar.py:

  1. Resolve the set of dated contracts that the contract calendar needs to cover
     the M1 *and* M2 slots over a recent window (``required_tickers``).
  2. Download each dated contract from Yahoo (e.g. ``CLQ26.NYM``) once.
  3. Upsert it into the existing ``price_history`` table under
     ``interval = <contract_code>`` (e.g. ``CLQ26``).  The UniqueConstraint on
     (commodity_id, date, interval) makes this idempotent and keeps every
     contract series separate from the '1d' front series and the stitched
     '1d_m2' / '1d_m1_raw' outputs.

The stitcher (pipeline/stitch_m2.py) then reads these contract rows and builds
the genuine constant-maturity series.

WHY A SEPARATE SCRIPT (not folded into pipeline/ingest.py)
──────────────────────────────────────────────────────────
• It is slow (dozens of downloads per commodity) and only needs to run when the
  M2 stack is (re)built or refreshed — not on every 2:05pm daily ingest.
• It is rate-limit aware: Yahoo throttled the original single-pass M2 attempt
  (only 12 of 26 landed).  This script sleeps between requests and retries
  transient failures with backoff.

PRACTICAL DEPTH NOTE
────────────────────
Yahoo serves roughly the last ~12 months of history per *dated* contract.
Contracts that expired more than ~1y ago return empty and are skipped (logged,
non-fatal).  So the stitched series is ~1y deep today and grows forward as new
daily observations accrue — far better than the old fixed-contract approach,
which produced an economically invalid basis for all but the most recent weeks.

RUN (on the Mac — needs network + Postgres):
  python -m pipeline.ingest_contracts
  python -m pipeline.ingest_contracts --lookback-days 540 --sleep 1.5
"""

import argparse
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone

from database.db import init_db, get_db
from database.models import Commodity, IngestionLog
from pipeline.ingest import ingest_commodity
from pipeline.contract_calendar import required_tickers, contract_code
from pipeline.stitch_m2 import resolve_commodity_id
from config.futures_calendar import FUTURES_CONTRACT_SPECS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# yfinance logs expired/delisted contracts ("possibly delisted; no price data
# found") at ERROR level itself.  For dated-contract ingest that is EXPECTED and
# harmless noise — Yahoo simply doesn't serve contracts that rolled off long ago,
# and our code already classifies the empty result as a non-fatal 'empty'.  We
# both (a) pre-skip most doomed contracts via ``skip_rolled_before`` below and
# (b) raise yfinance's own logger to CRITICAL so any residual 404s don't bury the
# real run output in red.  This does NOT hide genuine network errors — those
# surface as our own 'error'/retry log lines, not via yfinance's logger.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Yahoo serves ~12 months of history per dated contract.  Resolve a slightly more
# generous ~13-month cutoff so we never query contracts that are guaranteed to
# 404, while keeping a small margin for contracts that lingered a bit longer.
YAHOO_CONTRACT_HISTORY_DAYS = 400


def _ingest_one_with_retry(
    db, commodity_id: int, ticker: str, name: str, run_id: str,
    sleep_s: float, max_retries: int,
) -> tuple[int, int, str, str | None]:
    """
    Call ingest_commodity for a single dated contract, retrying transient
    failures (rate limits / network blips) with linear backoff.

    'empty' is NOT retried — an expired contract legitimately returns no data.
    Stores under interval=<contract_code> (e.g. 'CLQ26'); backfill=True so the
    full ~12 months Yahoo holds per contract are pulled.
    """
    code = contract_code(ticker)
    last_status, last_err = "error", None
    for attempt in range(1, max_retries + 1):
        ins, sk, status, err, _ = ingest_commodity(
            db, commodity_id, ticker, name,
            backfill=True, run_id=run_id, interval=code,
        )
        last_status, last_err = status, err
        if status in ("ok", "empty", "circuit_breaker"):
            return ins, sk, status, err
        # transient error → back off and retry
        wait = sleep_s * attempt * 2
        log.warning("    %s attempt %d/%d failed (%s) — retrying in %.1fs",
                    ticker, attempt, max_retries, status, wait)
        time.sleep(wait)
    return 0, 0, last_status, last_err


def run_contract_ingestion(
    lookback_days: int = 540,
    buffer_days: int = 150,
    sleep_s: float = 1.5,
    max_retries: int = 3,
) -> tuple[int, int]:
    """
    Ingest every dated contract needed to stitch M1/M2 across the recent window.

    Parameters
    ----------
    lookback_days : how far back to resolve the M1/M2 schedule (Yahoo only holds
                    ~1y per contract, so older contracts simply return empty).
    buffer_days   : how far forward to resolve (covers the next roll's M2).
    sleep_s       : pause between Yahoo requests (rate-limit politeness).
    max_retries   : retries per contract on transient errors.
    """
    init_db()
    run_id = str(uuid.uuid4())
    start = date.today() - timedelta(days=lookback_days)
    end   = date.today() + timedelta(days=buffer_days)
    # Contracts that rolled off before this date are guaranteed to 404 on Yahoo —
    # skip them rather than wasting a download + retry cycle and printing red.
    skip_rolled_before = date.today() - timedelta(days=YAHOO_CONTRACT_HISTORY_DAYS)

    log.info("=" * 64)
    log.info("Dated-contract ingestion  window=[%s … %s]  run=%s",
             start, end, run_id[:8])
    log.info("=" * 64)

    grand_ins = grand_sk = 0

    with get_db() as db:
        # Resolve commodity_id by the STABLE ticker column (commodities.name is
        # seeded inconsistently — see resolve_commodity_id).
        name_to_id: dict[str, int] = {}
        for name, spec in FUTURES_CONTRACT_SPECS.items():
            cid = resolve_commodity_id(db, spec, name)
            if cid is None:
                log.warning("skip — commodity not seeded (ticker=%s name=%r)",
                            spec.yf_ticker, name)
                continue
            name_to_id[name] = cid

        for name, spec in FUTURES_CONTRACT_SPECS.items():
            commodity_id = name_to_id.get(name)
            if commodity_id is None:
                continue

            tickers = sorted(required_tickers(
                spec, name, start, end, max_depth=2,
                skip_rolled_before=skip_rolled_before,
            ))
            log.info("%-26s %2d contracts: %s",
                     name, len(tickers), ", ".join(tickers))

            c_ins = c_sk = c_ok = c_empty = c_err = 0
            for ticker in tickers:
                t0 = time.monotonic()
                try:
                    ins, sk, status, err = _ingest_one_with_retry(
                        db, commodity_id, ticker, name, run_id, sleep_s, max_retries,
                    )
                except Exception as exc:  # noqa: BLE001 — never let one contract kill the run
                    db.rollback()
                    ins, sk, status, err = 0, 0, "error", str(exc)[:500]
                    log.error("    %s ingest crashed (%s) — rolled back, continuing",
                              ticker, type(exc).__name__)
                dur = int((time.monotonic() - t0) * 1000)
                c_ins += ins
                c_sk += sk
                c_ok    += (status == "ok")
                c_empty += (status == "empty")
                c_err   += (status not in ("ok", "empty"))

                db.add(IngestionLog(
                    run_id=run_id, started_at=datetime.now(timezone.utc),
                    ticker=ticker, name=f"{name} [{contract_code(ticker)}]",
                    status=status, rows_inserted=ins, rows_skipped=sk,
                    error_msg=err, duration_ms=dur,
                ))
                # Commit per contract: releases the per-row SAVEPOINT subtransaction
                # XID locks (ingest_commodity wraps each insert in begin_nested()).
                # PostgreSQL holds those subxid locks until the TOP-level commit, so
                # accumulating them across 26×25 backfilled contracts in one giant
                # transaction exhausts max_locks_per_transaction shared memory
                # ("out of shared memory").  Committing here caps held locks at a
                # single contract (~1.3k rows) and preserves partial progress.
                try:
                    db.commit()
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    log.error("    %s commit failed (%s) — rolled back, continuing",
                              ticker, type(exc).__name__)
                time.sleep(sleep_s)

            grand_ins += c_ins
            grand_sk += c_sk
            log.info("  → %-24s ok=%d empty=%d err=%d  inserted=%d skipped=%d",
                     name, c_ok, c_empty, c_err, c_ins, c_sk)

    log.info("=" * 64)
    log.info("Contract ingestion complete.  Inserted=%d  Skipped=%d",
             grand_ins, grand_sk)
    log.info("Next: python -m pipeline.stitch_m2")
    log.info("=" * 64)
    return grand_ins, grand_sk


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest dated futures contracts for M1/M2 stitching")
    p.add_argument("--lookback-days", type=int, default=540)
    p.add_argument("--buffer-days",   type=int, default=150)
    p.add_argument("--sleep",         type=float, default=1.5, dest="sleep_s")
    p.add_argument("--max-retries",   type=int, default=3)
    args = p.parse_args()
    run_contract_ingestion(
        lookback_days=args.lookback_days,
        buffer_days=args.buffer_days,
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
    )
