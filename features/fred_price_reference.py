"""
FRED Energy Price Reference

Fetches authoritative EIA/ICE spot prices from FRED and cross-checks them
against the front-month futures prices that yfinance delivers.  A persistent
large deviation (> FLAG_THRESHOLD_PCT) signals a potential feed error, bad
contract roll, or instrument mapping problem.

FRED series
-----------
  DCOILWTICO   — WTI crude oil spot price (EIA/DOE, $/bbl, daily Mon–Fri)
  DCOILBRENTEU — Brent crude spot price  (ICE,     $/bbl, daily Mon–Fri)
  WTISPLC      — WTI spot, Platts methodology ($/bbl, daily) — different
                 delivery-point and timing assumptions from DCOILWTICO
  DHHNGSP      — Henry Hub nat gas spot (EIA/NYMEX, $/MMBtu, daily Mon–Fri)

Public API
----------
  fetch_fred_energy_prices(period)              → pd.DataFrame
  check_price_deviations(fred_prices, ...)      → pd.DataFrame
  get_cross_check_report(period, ...)           → dict
"""

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from features.macro_overlays import _fetch_fred_series, _period_to_start_date

log = logging.getLogger(__name__)

# ── Series definitions ─────────────────────────────────────────────────────────

# FRED series ID → output column name
_FRED_ENERGY: dict[str, str] = {
    "DCOILWTICO":   "wti_spot",      # WTI crude spot, EIA, $/bbl
    "DCOILBRENTEU": "brent_spot",    # Brent crude spot, ICE, $/bbl
    "WTISPLC":      "wti_platts",    # WTI spot, Platts, $/bbl
    "DHHNGSP":      "natgas_spot",   # Henry Hub nat gas spot, EIA, $/MMBtu
}

# yfinance front-month futures ticker for comparison (where applicable)
_YF_FUTURES: dict[str, str] = {
    "wti_spot":    "CL=F",   # WTI crude front-month
    "brent_spot":  "BZ=F",   # Brent crude front-month
    "natgas_spot": "NG=F",   # Henry Hub nat gas front-month
    # wti_platts compared against CL=F (same underlying, different price source)
    "wti_platts":  "CL=F",
}

# Instrument display names
_INSTRUMENT_NAMES: dict[str, str] = {
    "wti_spot":   "WTI Crude (EIA spot)",
    "brent_spot": "Brent Crude (ICE spot)",
    "wti_platts": "WTI Crude (Platts spot)",
    "natgas_spot": "Henry Hub Nat Gas (EIA spot)",
}

FLAG_THRESHOLD_PCT = 5.0   # flag if |spot vs futures| deviation exceeds this


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_fred_energy_prices(period: str = "1y") -> pd.DataFrame:
    """
    Fetch WTI (EIA), Brent, WTI (Platts), and Henry Hub spot prices from FRED.

    Returns
    -------
    pd.DataFrame
        Columns: wti_spot, brent_spot, wti_platts, natgas_spot.
        DatetimeIndex, business-day frequency, forward-filled up to 5 days.
    """
    start = _period_to_start_date(period)
    series: dict = {}

    for fred_id, col in _FRED_ENERGY.items():
        s = _fetch_fred_series(fred_id, start)
        if not s.empty:
            series[col] = s

    if not series:
        log.warning("fetch_fred_energy_prices: no FRED data retrieved — check FRED_API_KEY")
        return pd.DataFrame()

    df = pd.DataFrame(series).sort_index()
    for col in _FRED_ENERGY.values():
        if col not in df.columns:
            df[col] = np.nan

    return df[list(_FRED_ENERGY.values())].ffill(limit=5)


def _fetch_yf_latest(ticker: str) -> float:
    """Return the most recent adjusted close for a yfinance ticker."""
    try:
        raw = yf.download(ticker, period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return np.nan
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        return float(close.dropna().iloc[-1])
    except Exception:
        return np.nan


# ── Cross-check ───────────────────────────────────────────────────────────────

def check_price_deviations(
    fred_prices: pd.DataFrame,
    threshold_pct: float = FLAG_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    Compare latest FRED spot prices against live yfinance front-month futures.

    A non-zero deviation is always expected (spot vs futures basis), but a
    deviation above threshold_pct warrants investigation.

    Parameters
    ----------
    fred_prices   : output of fetch_fred_energy_prices()
    threshold_pct : flag level in percent (default 5 %)

    Returns
    -------
    pd.DataFrame  with columns:
        instrument, fred_col, yf_ticker,
        fred_price, yf_price, deviation_pct, flagged, note
    """
    if fred_prices.empty:
        return pd.DataFrame()

    fred_latest = fred_prices.ffill().dropna(how="all").iloc[-1]
    rows = []

    for fred_col, yf_ticker in _YF_FUTURES.items():
        fred_val = fred_latest.get(fred_col, np.nan)
        if np.isnan(fred_val):
            continue

        yf_val = _fetch_yf_latest(yf_ticker)
        if np.isnan(yf_val):
            continue

        dev_pct = (yf_val - fred_val) / fred_val * 100.0
        flagged = abs(dev_pct) > threshold_pct

        rows.append({
            "instrument":   _INSTRUMENT_NAMES.get(fred_col, fred_col),
            "fred_col":     fred_col,
            "yf_ticker":    yf_ticker,
            "fred_price":   round(fred_val, 4),
            "yf_price":     round(yf_val, 4),
            "deviation_pct": round(dev_pct, 2),
            "flagged":      flagged,
            "note": (
                f"FLAGGED: {dev_pct:+.1f}% exceeds ±{threshold_pct}% threshold"
                if flagged
                else f"OK — within normal basis ({dev_pct:+.1f}%)"
            ),
        })

    return pd.DataFrame(rows)


# ── Full report ───────────────────────────────────────────────────────────────

def get_cross_check_report(
    period: str = "1y",
    threshold_pct: float = FLAG_THRESHOLD_PCT,
    log_to_db: bool = True,
) -> dict:
    """
    Full pipeline: fetch FRED energy prices → cross-check yfinance → log flags.

    Returns
    -------
    dict
        "fred_prices"  : pd.DataFrame  full FRED history
        "deviations"   : pd.DataFrame  latest cross-check per instrument
        "n_flagged"    : int            number of flagged instruments
        "fetched_at"   : str            ISO 8601 UTC timestamp
    """
    fred_prices = fetch_fred_energy_prices(period=period)
    deviations  = check_price_deviations(fred_prices, threshold_pct=threshold_pct)

    n_flagged = int(deviations["flagged"].sum()) if not deviations.empty else 0

    if log_to_db and n_flagged > 0:
        _log_flags_to_db(deviations[deviations["flagged"]])
        log.warning(
            "FRED cross-check: %d instrument(s) flagged — logged to price_validation_log",
            n_flagged,
        )

    return {
        "fred_prices": fred_prices,
        "deviations":  deviations,
        "n_flagged":   n_flagged,
        "fetched_at":  datetime.now(timezone.utc).isoformat(),
    }


def _log_flags_to_db(flagged_rows: pd.DataFrame) -> None:
    """Write flagged deviations to price_validation_log for audit trail."""
    try:
        from database.db import get_db
        from database.models import PriceValidationLog

        run_id = str(uuid.uuid4())
        today  = datetime.now(timezone.utc).date()

        with get_db() as db:
            for _, row in flagged_rows.iterrows():
                db.add(PriceValidationLog(
                    run_id=run_id,
                    ticker=row["yf_ticker"],
                    name=row["instrument"],
                    date=today,
                    raw_close=row["yf_price"],
                    corrected_close=row["fred_price"],
                    reason_code="fred_spot_deviation",
                    action="quarantined",
                    details=(
                        f"FRED {row['fred_col']}={row['fred_price']:.4f} vs "
                        f"yfinance {row['yf_ticker']}={row['yf_price']:.4f} "
                        f"({row['deviation_pct']:+.2f}%)"
                    ),
                ))
    except Exception as exc:
        log.warning("Could not log FRED cross-check flags to DB: %s", exc)


# ── Historical spread analysis ────────────────────────────────────────────────

def compute_basis_history(
    fred_prices: pd.DataFrame,
    yf_ticker_map: dict | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Compute historical futures/spot basis for each FRED energy series.

    Fetches yfinance daily history and aligns it with FRED spot prices to
    produce a time-series of basis = (futures − spot) / spot × 100.

    Parameters
    ----------
    fred_prices   : output of fetch_fred_energy_prices()
    yf_ticker_map : override the default _YF_FUTURES map if needed
    period        : yfinance period for futures history fetch

    Returns
    -------
    pd.DataFrame  — columns named "<fred_col>_basis_pct", DatetimeIndex
    """
    if fred_prices.empty:
        return pd.DataFrame()

    ticker_map = yf_ticker_map or _YF_FUTURES
    start = _period_to_start_date(period)
    basis_cols: dict = {}

    for fred_col, yf_ticker in ticker_map.items():
        if fred_col not in fred_prices.columns:
            continue
        try:
            raw = yf.download(yf_ticker, start=start, interval="1d",
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close.index = pd.to_datetime(close.index).tz_localize(None)

            spot = fred_prices[fred_col].reindex(close.index).ffill(limit=5)
            basis_pct = (close - spot) / spot * 100.0
            basis_cols[f"{fred_col}_basis_pct"] = basis_pct
        except Exception as exc:
            log.debug("compute_basis_history skipped %s/%s: %s", fred_col, yf_ticker, exc)

    if not basis_cols:
        return pd.DataFrame()

    return pd.DataFrame(basis_cols).sort_index()
