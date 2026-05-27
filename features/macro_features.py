"""
Shared macro feature surface — Step 1 of the macro trigger integration spec.

This module is the single source of truth for "what was the macro state on
date X?" Every downstream model (cascade_orchestrator, macro_router,
sector_model, meta_predictor, portfolio_optimizer, ripple) reads from here
rather than re-deriving its own macro snapshot. Without this, each model
drifts away from the others.

GAP — historical trigger coverage:
  trigger_events only contains rows since the macro-feed daemon began running.
  For training windows before that date, get_active_triggers() returns [] and
  regime_hint falls back to "neutral". FRED-derived surprise features
  (build_macro_surprise_features) are fully backfillable.

Public API
----------
  get_macro_state_at(date)              → dict (stable schema, see below)
  get_active_triggers(date, lookback)   → list[dict]  sorted by strength desc
  build_macro_surprise_features(date)   → dict of z-scored FRED deviations
"""

from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from features.macro_overlays import _fetch_fred_series

log = logging.getLogger(__name__)

# ── FRED + market series this module needs ────────────────────────────────────
# Kept here (rather than reused from fred_price_reference) so this module owns
# exactly the macro inputs it needs and is not coupled to the EIA cross-check.
_FRED_SERIES: dict[str, str] = {
    "DTWEXBGS":   "dxy",          # Broad trade-weighted dollar
    "VIXCLS":     "vix",          # CBOE volatility index
    "DGS20":      "tlt_yield",    # 20Y treasury yield (TLT-proxy yield)
    "CPIAUCSL":   "cpi",
    "UNRATE":     "unrate",
    "FEDFUNDS":   "fedfunds",
    "T10Y2Y":     "t10y2y",
    "DCOILWTICO": "wti",
}

# yfinance ticker for TLT itself — used for tlt_ret_5d only.
_YF_TLT = "TLT"

# Mapping from trigger family_name (live registry, config/trigger_registry.json)
# to the spec's regime buckets. Substring fallback at the bottom handles any
# future family that follows the documented prefixes.
_FAMILY_TO_REGIME: dict[str, str] = {
    "fomc_rate_decision": "rate_shock",
    "fed_tightening":     "rate_shock",
    "fed_chair_speech":   "rate_shock",
    "cpi_release":        "rate_shock",
    "ppi_release":        "rate_shock",
    "opec_action":         "commodity_shock",
    "eia_crude_inventory": "commodity_shock",
    "eia_gas_storage":     "commodity_shock",
    "usda_wasde_report":   "commodity_shock",
    "weather_shock":       "commodity_shock",
    "energy_transition":   "commodity_shock",
    "geopolitical_shock":  "commodity_shock",
    "nonfarm_payrolls":    "growth_shock",
    "recession_flag":      "growth_shock",
}

_STRENGTH_REGIME_OVERRIDE = 0.8   # spec: top trigger ≥ 0.8 → its family's regime


def family_to_regime(family: str) -> str:
    """
    Map a trigger family name to one of {"rate_shock", "commodity_shock",
    "growth_shock", "neutral"}. Public so other modules (cascade_orchestrator,
    macro_router, etc.) can derive a regime_hint from triggers they already
    fetched, without a second FRED roundtrip.
    """
    if family in _FAMILY_TO_REGIME:
        return _FAMILY_TO_REGIME[family]
    f = family.lower()
    if f.startswith(("fed_", "fomc_", "cpi_", "ppi_")):
        return "rate_shock"
    if f.startswith(("opec_", "weather_", "eia_", "usda_", "energy_", "geo")):
        return "commodity_shock"
    if f.startswith(("unemployment_", "gdp_", "nonfarm_", "recession_")):
        return "growth_shock"
    return "neutral"


def regime_hint_from_triggers(triggers: list[dict],
                              min_strength: float = _STRENGTH_REGIME_OVERRIDE) -> str:
    """
    Pure helper: derive the spec's regime_hint from an already-fetched list of
    trigger dicts. The top trigger (by strength) at or above ``min_strength``
    wins; otherwise "neutral".
    """
    for t in sorted(triggers, key=lambda d: d.get("strength", 0.0), reverse=True):
        if float(t.get("strength", 0.0)) >= min_strength:
            regime = family_to_regime(t.get("family", ""))
            if regime != "neutral":
                return regime
    return "neutral"


# Keep the old private alias so the internal cache function below keeps working.
_family_to_regime = family_to_regime


# ── History loader (cached) ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_macro_history(lookback_days: int = 365 * 5) -> pd.DataFrame:
    """
    Fetch all FRED series + TLT once and stitch into a daily DataFrame.

    Cached for the lifetime of the process — model training loops call
    get_macro_state_at() thousands of times and must not re-fetch.
    """
    start = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    cols: dict[str, pd.Series] = {}

    for fred_id, name in _FRED_SERIES.items():
        s = _fetch_fred_series(fred_id, start)
        if not s.empty:
            cols[name] = s

    # TLT close (yfinance — same source the rest of the dashboard uses).
    try:
        import yfinance as yf
        raw = yf.download(_YF_TLT, start=start, interval="1d",
                          progress=False, auto_adjust=True)
        if not raw.empty:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close.index = pd.to_datetime(close.index).tz_localize(None)
            cols["tlt"] = close.astype(float)
    except Exception as exc:
        warnings.warn(f"macro_features: TLT fetch failed ({exc}); tlt_ret_5d will be NaN")

    if not cols:
        return pd.DataFrame()

    df = pd.DataFrame(cols).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.ffill(limit=5)


def _asof_value(series: pd.Series, date: pd.Timestamp) -> float:
    """Latest observation in `series` at or before `date`, else NaN."""
    if series is None or series.empty:
        return float("nan")
    s = series.dropna()
    s = s[s.index <= date]
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _pct_return(series: pd.Series, date: pd.Timestamp, days: int) -> float:
    """Log-return of `series` over the last `days` business days ending at `date`."""
    if series is None or series.empty:
        return float("nan")
    s = series.dropna()
    s = s[s.index <= date]
    if len(s) < days + 1:
        return float("nan")
    end, start = s.iloc[-1], s.iloc[-(days + 1)]
    if start <= 0 or end <= 0:
        return float("nan")
    return float(np.log(end / start))


def _level_change(series: pd.Series, date: pd.Timestamp, days: int) -> float:
    """Absolute change in level over `days` business days ending at `date`."""
    if series is None or series.empty:
        return float("nan")
    s = series.dropna()
    s = s[s.index <= date]
    if len(s) < days + 1:
        return float("nan")
    return float(s.iloc[-1] - s.iloc[-(days + 1)])


def _rolling_zscore(series: pd.Series, date: pd.Timestamp, months: int = 24) -> float:
    """
    Z-score of the latest observation in `series` (as-of `date`) versus the
    trailing `months` of values. Returns NaN if the window is too short.
    """
    if series is None or series.empty:
        return float("nan")
    s = series.dropna()
    s = s[s.index <= date]
    if s.empty:
        return float("nan")
    window_start = s.index[-1] - pd.DateOffset(months=months)
    window = s[s.index >= window_start]
    if len(window) < 6:
        return float("nan")
    mean = window.mean()
    std = window.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float((window.iloc[-1] - mean) / std)


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _macro_state_cached(date_iso: str) -> dict:
    """Inner cache — keyed on YYYY-MM-DD string so dates with equal calendar days hit once."""
    date = pd.Timestamp(date_iso)
    hist = _load_macro_history()

    def col(name: str) -> pd.Series:
        return hist[name] if (not hist.empty and name in hist.columns) else pd.Series(dtype=float)

    state = {
        "dxy_ret_5d":       _pct_return(col("dxy"), date, 5),
        "vix_ret_5d":       _pct_return(col("vix"), date, 5),
        "tlt_ret_5d":       _pct_return(col("tlt"), date, 5),
        "tlt_yield_proxy":  _asof_value(col("tlt_yield"), date),
        "cpi_zscore":       _rolling_zscore(col("cpi"), date),
        "unrate_zscore":    _rolling_zscore(col("unrate"), date),
        "fedfunds_zscore":  _rolling_zscore(col("fedfunds"), date),
        "t10y2y_level":     _asof_value(col("t10y2y"), date),
        "t10y2y_change_5d": _level_change(col("t10y2y"), date, 5),
        "wti_ret_5d":       _pct_return(col("wti"), date, 5),
        "regime_hint":      _derive_regime_hint(date),
    }
    return state


def get_macro_state_at(date: pd.Timestamp) -> dict:
    """
    Snapshot of macro variables as-of `date` (no look-ahead).

    See module docstring for the stable return schema.
    """
    return _macro_state_cached(pd.Timestamp(date).strftime("%Y-%m-%d"))


def get_active_triggers(date: pd.Timestamp, lookback_days: int = 5) -> list[dict]:
    """
    Rows from `trigger_events` whose trigger_date falls in
    [date - lookback_days, date]. Sorted by strength descending.
    """
    # Imported lazily so unit tests can run without a configured DB.
    from database.db import get_db
    from database.models import TriggerEvent

    end = pd.Timestamp(date).normalize()
    start = end - pd.Timedelta(days=lookback_days)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    try:
        with get_db() as db:
            rows = (
                db.query(TriggerEvent)
                .filter(TriggerEvent.trigger_date >= start_str,
                        TriggerEvent.trigger_date <= end_str)
                .all()
            )
            out = [
                {
                    "family":           r.family,
                    "strength":         float(r.strength),
                    "trigger_date":     r.trigger_date,
                    "deviation_score":  float(r.strength),  # alias — strength IS the deviation score
                    "source":           "trigger_events",
                }
                for r in rows
            ]
    except Exception as exc:
        log.warning("get_active_triggers: DB query failed (%s); returning []", exc)
        return []

    out.sort(key=lambda d: d["strength"], reverse=True)
    return out


def build_macro_surprise_features(date: pd.Timestamp) -> dict:
    """
    Z-scored deviations vs trailing 24-month rolling mean for the five
    FRED series the spec calls out. Used directly as model feature columns.
    """
    date = pd.Timestamp(date)
    hist = _load_macro_history()

    def col(name: str) -> pd.Series:
        return hist[name] if (not hist.empty and name in hist.columns) else pd.Series(dtype=float)

    return {
        "cpi_surprise_z":       _rolling_zscore(col("cpi"), date),
        "unrate_surprise_z":    _rolling_zscore(col("unrate"), date),
        "fedfunds_surprise_z":  _rolling_zscore(col("fedfunds"), date),
        "t10y2y_surprise_z":    _rolling_zscore(col("t10y2y"), date),
        "wti_surprise_z":       _rolling_zscore(col("wti"), date),
    }


# ── Regime derivation ─────────────────────────────────────────────────────────

def _derive_regime_hint(date: pd.Timestamp) -> str:
    """Spec rule: top trigger with strength ≥ 0.8 in lookback wins. Else "neutral"."""
    return regime_hint_from_triggers(get_active_triggers(date, lookback_days=5))


# ── Cache management (for tests + retraining jobs) ────────────────────────────

def clear_caches() -> None:
    """Reset module-level caches. Call from tests or after a major data refresh."""
    _load_macro_history.cache_clear()
    _macro_state_cached.cache_clear()
