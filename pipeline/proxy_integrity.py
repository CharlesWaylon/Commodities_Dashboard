"""
Cross-asset proxy-integrity check — the belt-and-suspenders backstop for scale
errors that the per-ticker sanity bands CANNOT see.

WHY THIS EXISTS:
  pipeline/price_validator.py catches scale errors by checking each ticker's price
  against an absolute sanity band. But a value can be rescaled to the WRONG number
  and still land *inside* its band — in which case the band check is blind to it.

  That is exactly what happened to SIVR in May–Jun 2026: real ~$73 values were
  rescaled ÷10 to ~$7.3 because they had (briefly) exceeded a stale ceiling, and
  ~$7.3 is a perfectly "sane" SIVR price, so nothing flagged it. The only thing
  that revealed it was the relationship to its underlying: a physically-backed
  ETF tracks its metal at a near-constant ratio, and that ratio suddenly broke by
  10×.

  This module formalises that cross-check. For each physically-backed proxy it
  compares the ETF close to its futures underlying close. The ratio is extremely
  stable (SGOL/GC=F ≈ 0.0096 ±2.5%, SIVR/SI=F ≈ 0.96 ±6%), so any deviation of
  more than RATIO_BREAK_FACTOR× from the robust historical median is almost
  certainly a unit/scale error, not real tracking drift.

SCOPE:
  Only PHYSICALLY-BACKED ETFs with a tight, mechanical link to a futures
  underlying belong here. Equity/sector proxies (URA, KRBN, SLX, LIT, REMX, WOOD,
  LNG, BTU, HCC, GLNCY) track baskets of companies, not the spot metal, so their
  ratio to any single future is noisy and NOT suitable for this check.

USAGE:
  • Detection runs automatically inside pipeline/alert_reporter.py after each
    ingest (non-destructive — it only flags).
  • Manual remediation:
        python -c "from pipeline.proxy_integrity import fix_proxy_ratio_breaks; \
                   print(fix_proxy_ratio_breaks('SIVR', dry_run=False))"
    then re-run roll_adjust + align_calendar.
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Physically-backed proxy → its futures underlying. ONLY tight mechanical links.
PROXY_UNDERLYING: dict[str, str] = {
    "SGOL": "GC=F",   # abrdn Physical Gold Shares  → COMEX Gold   (ratio ≈ 0.0096)
    "SIVR": "SI=F",   # abrdn Physical Silver Shares → COMEX Silver (ratio ≈ 0.96)
}

# Flag when the proxy/underlying ratio deviates from its robust median by more
# than this multiplicative factor. Real tracking noise is <~6%; the smallest
# scale error is 10×, so anything in (1.5, 5) cleanly separates the two. 2.0 is
# conservative — it never fires on tracking drift but always on an order-of-
# magnitude slip.
RATIO_BREAK_FACTOR = 2.0

# History used to establish the robust (median) ratio.
ROBUST_WINDOW_DAYS = 400
# Only rows newer than this are reported as fresh breaks (keeps the daily report
# from re-flagging an old, already-known break forever).
RECENT_DAYS = 5
# Minimum aligned observations before the median ratio is trustworthy.
MIN_HISTORY = 30


def _nearest_power_of_10(x: float) -> float:
    """Closest power of ten to x (e.g. 9.5 → 10, 0.11 → 0.1). None if non-finite/≤0."""
    if x is None or not math.isfinite(x) or x <= 0:
        return None
    return 10.0 ** round(math.log10(x))


def detect_ratio_breaks(
    proxy: pd.Series,
    underlying: pd.Series,
    *,
    recent_cutoff,
    break_factor: float = RATIO_BREAK_FACTOR,
    min_history: int = MIN_HISTORY,
) -> list[dict]:
    """
    Pure detector. Given a proxy close series and its underlying close series
    (each indexed by date), return one dict per recent day whose proxy/underlying
    ratio deviates from the robust median ratio by more than break_factor×.

    Each break dict carries the observed and expected ratio and a
    `suggested_factor` (the power of ten to MULTIPLY the proxy close by to restore
    tracking — e.g. 10.0 for the SIVR ÷10 corruption).
    """
    df = pd.concat([proxy.rename("p"), underlying.rename("u")], axis=1).dropna()
    df = df[df["u"] > 0]
    if len(df) < min_history:
        return []

    ratio = df["p"] / df["u"]
    median_ratio = float(ratio.median())
    if not math.isfinite(median_ratio) or median_ratio <= 0:
        return []

    breaks: list[dict] = []
    recent = ratio[ratio.index >= recent_cutoff]
    for dt, r in recent.items():
        if not math.isfinite(r) or r <= 0:
            continue
        rel = r / median_ratio
        if rel > break_factor or rel < 1.0 / break_factor:
            suggested = _nearest_power_of_10(median_ratio / r)
            breaks.append({
                "date":            dt.date() if hasattr(dt, "date") else dt,
                "proxy_close":     float(df.loc[dt, "p"]),
                "underlying_close": float(df.loc[dt, "u"]),
                "observed_ratio":  float(r),
                "expected_ratio":  median_ratio,
                "deviation_x":     float(rel if rel >= 1 else 1.0 / rel),
                "suggested_factor": suggested,
            })
    return breaks


# ── DB wrappers ──────────────────────────────────────────────────────────────

def _load_close(ticker: str, cutoff_date) -> pd.Series:
    from database.db import get_engine
    from sqlalchemy import text
    with get_engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT ph.date, ph.close
            FROM price_history ph JOIN commodities c ON c.id = ph.commodity_id
            WHERE c.ticker = :t AND ph.interval = '1d' AND ph.date >= :cutoff
              AND ph.close IS NOT NULL
            ORDER BY ph.date
        """), {"t": ticker, "cutoff": str(cutoff_date)}).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows})
    return s


def scan_proxy_ratio_breaks(recent_days: int = RECENT_DAYS) -> list[dict]:
    """
    Scan every PROXY_UNDERLYING pair for ratio breaks in the last `recent_days`.
    Returns a flat list of break dicts (each tagged with proxy/underlying tickers).
    Non-destructive — detection only. Safe to call with no DB (returns []).
    """
    try:
        robust_cutoff = (datetime.now(timezone.utc) - timedelta(days=ROBUST_WINDOW_DAYS)).date()
        recent_cutoff = pd.Timestamp(datetime.now() - timedelta(days=recent_days))
        out: list[dict] = []
        for proxy, underlying in PROXY_UNDERLYING.items():
            p = _load_close(proxy, robust_cutoff)
            u = _load_close(underlying, robust_cutoff)
            if p.empty or u.empty:
                continue
            for b in detect_ratio_breaks(p, u, recent_cutoff=recent_cutoff):
                b["proxy"] = proxy
                b["underlying"] = underlying
                out.append(b)
        return out
    except Exception as e:  # never let an audit check break the pipeline
        log.warning("scan_proxy_ratio_breaks failed (non-fatal): %s", e)
        return []


def fix_proxy_ratio_breaks(proxy_ticker: str, dry_run: bool = True,
                           lookback_days: int = ROBUST_WINDOW_DAYS) -> dict:
    """
    Manual remediation for a proxy whose recent rows were scale-corrupted. Applies
    the per-row `suggested_factor` (multiply close/OHLC) to each broken row.

    dry_run=True (default) reports what WOULD change without writing. Set
    dry_run=False to commit, then re-run roll_adjust + align_calendar.

    Examines a wide window (lookback_days) so an old break is also repairable, not
    just the last few days. Only rows that break the ratio test are touched.
    """
    underlying = PROXY_UNDERLYING.get(proxy_ticker)
    if underlying is None:
        return {"error": f"{proxy_ticker} is not a tracked physical proxy"}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    p = _load_close(proxy_ticker, cutoff)
    u = _load_close(underlying, cutoff)
    breaks = detect_ratio_breaks(
        p, u, recent_cutoff=pd.Timestamp(cutoff),  # whole window, not just recent
    )
    by_date = {b["date"]: b["suggested_factor"] for b in breaks
               if b["suggested_factor"]}

    if dry_run or not by_date:
        return {"proxy": proxy_ticker, "breaks": len(by_date),
                "dates": sorted(map(str, by_date)), "applied": False}

    from database.db import get_db
    from database.models import Commodity, PriceHistory
    applied = 0
    with get_db() as db:
        c = db.query(Commodity).filter_by(ticker=proxy_ticker).first()
        rows = (db.query(PriceHistory)
                  .filter(PriceHistory.commodity_id == c.id,
                          PriceHistory.interval == "1d")
                  .all())
        for r in rows:
            f = by_date.get(r.date)
            if not f:
                continue
            for attr in ("open", "high", "low", "close"):
                v = getattr(r, attr)
                if v is not None:
                    setattr(r, attr, v * f)
            applied += 1
    log.info("fix_proxy_ratio_breaks: %s — applied factor to %d row(s).",
             proxy_ticker, applied)
    return {"proxy": proxy_ticker, "breaks": len(by_date),
            "dates": sorted(map(str, by_date)), "applied": True, "rows": applied}
