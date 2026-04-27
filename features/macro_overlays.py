"""
Macro overlay features and commodity calendar event dummies.

DXY / VIX / TLT
  Three instruments, all free via yfinance, that proxy for three distinct
  macro regimes:
    DXY (^DXY) — dollar strength. Most commodity futures are USD-priced, so
      a strong dollar depresses commodity demand from non-USD buyers.
      Gold and oil are particularly sensitive; agricultural commodities slightly
      less so (driven more by local weather and supply).
    VIX (^VIX) — implied equity volatility (fear gauge). Above 20 = risk-off;
      above 30 = crisis. In risk-off regimes, commodity correlations converge
      and momentum signals weaken — this is when most systematic strategies fail.
    TLT — iShares 20+ Year Treasury ETF. TLT price inversely tracks long-term
      US interest rates. Rising rates = falling TLT = commodity headwind (higher
      discount rate on inventories, stronger USD carry, competition from fixed income).

WASDE calendar dummy
  The USDA World Agricultural Supply and Demand Estimates report is published
  monthly (typically the 2nd Tuesday, 12:00 noon ET). Agricultural futures
  (Corn, Wheat, Soybeans, Cotton, Sugar) experience predictable volatility
  spikes in the 1-2 days before and after release. This dummy encodes:
    - days_to_wasde: integer, negative = days before release
    - is_wasde_week: bool, True in the 7-day window around release
    - wasde_post5: bool, True for 5 trading days after release (surprise digestion)

OPEC+ meeting dummy
  OPEC+ typically meets twice yearly (June and November/December) to set
  production quotas. Energy futures exhibit elevated vol in the 2 weeks
  before a scheduled meeting. Since 2022, extraordinary meetings have become
  more common (production cut surprises). Hardcoded 2020-2027 meeting dates.

Usage
-----
    from features.macro_overlays import build_macro_overlay_features

    df = build_macro_overlay_features(start="2022-01-01", end="2026-04-25")
    # Columns: dxy_*, vix_*, tlt_*, days_to_wasde, is_wasde_week, wasde_post5,
    #          days_to_opec, is_opec_week, opec_post10
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from typing import Optional

# ── Macro tickers ──────────────────────────────────────────────────────────────
_MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",   # US Dollar Index (ICE, free via Yahoo)
    "vix": "^VIX",        # CBOE Volatility Index
    "tlt": "TLT",         # iShares 20+ Year Treasury ETF
}

# VIX regime thresholds (annualised basis points of expected S&P 500 vol)
VIX_RISK_OFF = 20.0
VIX_CRISIS   = 30.0

# WASDE rolling window features
_WASDE_PRE_WINDOW  = 7    # days before release to flag
_WASDE_POST_WINDOW = 5    # trading days after release to flag

# OPEC+ windows
_OPEC_PRE_WINDOW  = 14   # days before meeting
_OPEC_POST_WINDOW = 10   # trading days after meeting

# ── WASDE release dates ────────────────────────────────────────────────────────
# Source: USDA ERS, https://www.usda.gov/oce/commodity/wasde/
# Approximate: 2nd Tuesday of each month, 12:00 ET. Hardcoded for accuracy.
WASDE_DATES = [
    # 2022
    "2022-01-12", "2022-02-09", "2022-03-09", "2022-04-08", "2022-05-12",
    "2022-06-10", "2022-07-12", "2022-08-12", "2022-09-12", "2022-10-12",
    "2022-11-09", "2022-12-09",
    # 2023
    "2023-01-12", "2023-02-08", "2023-03-08", "2023-04-11", "2023-05-12",
    "2023-06-09", "2023-07-12", "2023-08-11", "2023-09-12", "2023-10-12",
    "2023-11-09", "2023-12-08",
    # 2024
    "2024-01-12", "2024-02-08", "2024-03-08", "2024-04-11", "2024-05-10",
    "2024-06-12", "2024-07-11", "2024-08-12", "2024-09-12", "2024-10-11",
    "2024-11-08", "2024-12-10",
    # 2025
    "2025-01-10", "2025-02-11", "2025-03-11", "2025-04-10", "2025-05-12",
    "2025-06-11", "2025-07-11", "2025-08-12", "2025-09-11", "2025-10-10",
    "2025-11-10", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11", "2026-03-10", "2026-04-09", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-09",
    "2026-11-10", "2026-12-09",
]

# ── OPEC+ meeting dates ────────────────────────────────────────────────────────
# Includes extraordinary meetings (marked with "*" in comments)
OPEC_MEETING_DATES = [
    # 2020
    "2020-03-05",  # extraordinary — Covid demand collapse
    "2020-04-09",  # extraordinary — historic production cut agreement
    "2020-06-06",  # regular
    "2020-12-03",  # regular
    # 2021
    "2021-03-04", "2021-04-01", "2021-05-27",
    "2021-07-01", "2021-09-01", "2021-11-04",
    # 2022
    "2022-02-02", "2022-03-02", "2022-04-05",
    "2022-06-02",
    "2022-10-05",  # extraordinary — 2mb/d production cut announcement
    "2022-12-04",
    # 2023
    "2023-02-01", "2023-04-03",
    "2023-06-04",
    "2023-11-26",  # extraordinary — additional Saudi voluntary cuts
    # 2024
    "2024-02-01", "2024-06-02",
    "2024-12-05",
    # 2025
    "2025-02-03", "2025-06-01",
    "2025-12-04",
    # 2026
    "2026-06-01",
    "2026-12-03",
]


def _nth_tuesday_of_month(year: int, month: int, n: int = 2) -> date:
    """Return the nth Tuesday of the given month (fallback for WASDE dates)."""
    first = date(year, month, 1)
    days_until_tue = (1 - first.weekday()) % 7   # 1 = Tuesday
    return first + timedelta(days=days_until_tue + (n - 1) * 7)


def _wasde_dates_range(start: date, end: date) -> list:
    """Return all WASDE dates within [start, end], extending algorithmically if needed."""
    known = {pd.Timestamp(d).date() for d in WASDE_DATES}
    result = [d for d in known if start <= d <= end]
    # Fill gaps via 2nd-Tuesday heuristic
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        approx = _nth_tuesday_of_month(y, m, n=2)
        if approx not in known and start <= approx <= end:
            result.append(approx)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return sorted(set(result))


def _opec_dates_range(start: date, end: date) -> list:
    known = [pd.Timestamp(d).date() for d in OPEC_MEETING_DATES]
    return [d for d in known if start <= d <= end]


# ── Macro price features ───────────────────────────────────────────────────────

def fetch_macro_prices(period: str = "2y") -> pd.DataFrame:
    """
    Download DXY, VIX, and TLT from yfinance.

    Returns
    -------
    pd.DataFrame
        Columns: dxy, vix, tlt. DatetimeIndex.
    """
    tickers = list(_MACRO_TICKERS.values())
    try:
        data = yf.download(tickers, period=period, interval="1d",
                           progress=False, auto_adjust=True)
        if data.empty:
            raise RuntimeError("No data returned")
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        close.index = pd.to_datetime(close.index).tz_localize(None)
        ticker_to_name = {v: k for k, v in _MACRO_TICKERS.items()}
        close = close.rename(columns=ticker_to_name)
        return close[["dxy", "vix", "tlt"]].ffill(limit=3)
    except Exception as e:
        warnings.warn(f"Macro price fetch failed: {e}")
        return pd.DataFrame()


def macro_features(period: str = "2y") -> pd.DataFrame:
    """
    Compute feature columns from DXY, VIX, TLT price history.

    Returns
    -------
    pd.DataFrame
        Columns:
          dxy, dxy_ret, dxy_zscore63, dxy_mom5, dxy_mom21
          vix, vix_ret5d, vix_risk_off, vix_crisis
          tlt, tlt_ret, tlt_yield_proxy, tlt_mom21
    """
    prices = fetch_macro_prices(period=period)
    if prices.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=prices.index)

    # ── DXY ──
    dxy = prices["dxy"]
    dxy_ret = np.log(dxy / dxy.shift(1))
    dxy_mu  = dxy.rolling(63).mean()
    dxy_sig = dxy.rolling(63).std().clip(lower=1e-9)
    result["dxy"]         = dxy
    result["dxy_ret"]     = dxy_ret
    result["dxy_zscore63"] = (dxy - dxy_mu) / dxy_sig
    result["dxy_mom5"]    = dxy_ret.rolling(5).sum()
    result["dxy_mom21"]   = dxy_ret.rolling(21).sum()

    # ── VIX ──
    vix = prices["vix"]
    result["vix"]         = vix
    result["vix_ret5d"]   = np.log(vix / vix.shift(5))
    result["vix_risk_off"] = (vix >= VIX_RISK_OFF).astype(float)
    result["vix_crisis"]   = (vix >= VIX_CRISIS).astype(float)
    result["vix_level_z"]  = (
        (vix - vix.rolling(252).mean()) /
        vix.rolling(252).std().clip(lower=1e-9)
    )

    # ── TLT ──
    tlt = prices["tlt"]
    tlt_ret = np.log(tlt / tlt.shift(1))
    result["tlt"]             = tlt
    result["tlt_ret"]         = tlt_ret
    result["tlt_yield_proxy"] = -tlt_ret   # bond price ↓ → yield ↑ → rates rising
    result["tlt_mom21"]       = tlt_ret.rolling(21).sum()

    return result


# ── Calendar event dummies ─────────────────────────────────────────────────────

def wasde_calendar_features(
    index: pd.DatetimeIndex,
    pre_window:  int = _WASDE_PRE_WINDOW,
    post_window: int = _WASDE_POST_WINDOW,
) -> pd.DataFrame:
    """
    Build WASDE release calendar dummy features aligned to `index`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Date range to generate features for. Typically the index of the
        full feature matrix.

    Returns
    -------
    pd.DataFrame
        Columns:
          days_to_wasde   — signed integer (negative = before release)
          is_wasde_window — True within [−pre_window, +post_window] days
          wasde_post5     — True for 5 calendar days after release
    """
    start = index.min().date()
    end   = index.max().date()
    releases = _wasde_dates_range(start, end)

    result = pd.DataFrame(index=index)
    result["days_to_wasde"]   = np.nan
    result["is_wasde_window"] = False
    result["wasde_post5"]     = False

    for t in index:
        d = t.date()
        # Days to nearest upcoming release
        future = [r for r in releases if r >= d]
        past   = [r for r in releases if r < d]
        dist_future = (future[0] - d).days if future else 999
        dist_past   = (d - past[-1]).days  if past   else 999

        if dist_future <= dist_past:
            result.loc[t, "days_to_wasde"]   = -dist_future
            result.loc[t, "is_wasde_window"] = dist_future <= pre_window
        else:
            result.loc[t, "days_to_wasde"]   = dist_past
            result.loc[t, "is_wasde_window"] = dist_past <= post_window

        result.loc[t, "wasde_post5"] = (0 <= dist_past <= post_window)

    result["days_to_wasde"] = result["days_to_wasde"].astype(float)
    return result


def opec_calendar_features(
    index: pd.DatetimeIndex,
    pre_window:  int = _OPEC_PRE_WINDOW,
    post_window: int = _OPEC_POST_WINDOW,
) -> pd.DataFrame:
    """
    Build OPEC+ meeting calendar dummy features aligned to `index`.

    Returns
    -------
    pd.DataFrame
        Columns:
          days_to_opec   — signed integer (negative = before meeting)
          is_opec_window — True within [−pre_window, +post_window] days
          opec_post10    — True for 10 calendar days after meeting
    """
    start = index.min().date()
    end   = index.max().date()
    meetings = _opec_dates_range(start, end)

    result = pd.DataFrame(index=index)
    result["days_to_opec"]   = np.nan
    result["is_opec_window"] = False
    result["opec_post10"]    = False

    for t in index:
        d = t.date()
        future = [m for m in meetings if m >= d]
        past   = [m for m in meetings if m < d]
        dist_future = (future[0] - d).days if future else 999
        dist_past   = (d - past[-1]).days  if past   else 999

        if dist_future <= dist_past:
            result.loc[t, "days_to_opec"]   = -dist_future
            result.loc[t, "is_opec_window"] = dist_future <= pre_window
        else:
            result.loc[t, "days_to_opec"]   = dist_past
            result.loc[t, "is_opec_window"] = dist_past <= post_window

        result.loc[t, "opec_post10"] = (0 <= dist_past <= post_window)

    result["days_to_opec"] = result["days_to_opec"].astype(float)
    return result


# ── Convenience runner ─────────────────────────────────────────────────────────

def build_macro_overlay_features(
    period: str = "2y",
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Assemble DXY/VIX/TLT features and WASDE/OPEC calendar dummies.

    Parameters
    ----------
    period : str
        yfinance period for macro price history.
    index : pd.DatetimeIndex, optional
        Date index to align calendar dummies to. If None, uses the price index.

    Returns
    -------
    pd.DataFrame
        All macro and calendar columns on a common DatetimeIndex.
    """
    macro = macro_features(period=period)
    if macro.empty:
        return pd.DataFrame()

    idx = index if index is not None else macro.index

    wasde = wasde_calendar_features(idx)
    opec  = opec_calendar_features(idx)

    combined = macro.reindex(idx).ffill(limit=3)
    combined = combined.join(wasde, how="left")
    combined = combined.join(opec,  how="left")

    return combined
