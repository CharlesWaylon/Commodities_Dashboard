"""
IC (Information Coefficient) Tracker — Phase 5 Part 2.

Purpose
───────
Measures and persists how well each model tier is actually forecasting.

The IC is the Spearman rank correlation between a tier's forecasted returns
and the realised next-day returns.  It is the standard quantitative-finance
accuracy metric for directional forecasting:

    IC ≥ 0.05   → tier is generating actionable signal  (GREEN)
    0 ≤ IC < 0.05 → marginal signal, monitor carefully  (AMBER)
    IC < 0      → tier is actively misleading           (RED)

Inputs
──────
Every BacktestRecord already contains:
  - tier_forecasts: {tier: forecast_return}  (from backtest_harness)
  - actual_return:  realised next-day log-return

So IC can be computed from any completed backtest run with zero extra model
fits.  This makes the tracker a pure post-processing step: call
`compute_ic_from_records(records)` right after BacktestHarness.run().

Flow
────
    BacktestHarness.run() → List[BacktestRecord]
                                    │
                                    ▼
                    compute_ic_from_records()
                                    │
                                    ▼
                          Dict[(commodity, tier), ICResult]
                                    │
                            log_ic_scores()
                                    │
                               ic_log (SQLite)
                                    │
                  ┌─────────────────┼───────────────────┐
                  ▼                 ▼                   ▼
            recent_ic_scores()  ic_summary()     ic_trend()
                  │                 │                   │
            Model Health tab   scorecard table    trend chart

Integration
───────────
Called automatically from daily_retrain.run_daily_retrain() after the
backtest completes — no manual step needed.

Can also be called standalone from any notebook:

    from models.ic_tracker import compute_ic_from_records, log_ic_scores
    records = harness.run(prices, macro, commodities)
    ic_results = compute_ic_from_records(records)
    log_ic_scores(ic_results)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

IC_THRESHOLD_ACTION  = 0.05   # ≥ this → actionable (green)
IC_THRESHOLD_NEUTRAL = 0.00   # 0 to threshold → marginal (amber), < 0 → red

# ── Commodity → Sector mapping ─────────────────────────────────────────────────
# Used by compute_sector_ic_from_records() to pool per-commodity records into
# sector buckets.  Any commodity not listed here is silently skipped at the
# sector-aggregation step (it still appears in per-commodity IC).

COMMODITY_SECTORS: Dict[str, str] = {
    # Energy — futures
    "WTI Crude Oil":            "Energy",
    "Brent Crude Oil":          "Energy",
    "Natural Gas":              "Energy",
    "Gasoline (RBOB)":          "Energy",
    "Heating Oil":              "Energy",
    # Energy — ETFs / equities
    "Carbon Credits*":          "Energy",
    "LNG / Intl Gas*":          "Energy",
    "Metallurgical Coal*":      "Energy",
    "Thermal Coal*":            "Energy",
    "Uranium*":                 "Energy",
    # Metals — futures
    "Gold (COMEX)":             "Metals",
    "Silver (COMEX)":           "Metals",
    "Copper (COMEX)":           "Metals",
    "Platinum":                 "Metals",
    "Palladium":                "Metals",
    "Aluminum (COMEX)":         "Metals",
    "HRC Steel":                "Metals",
    # Metals — ETFs / equities
    "Gold (Physical/London)*":  "Metals",
    "Silver (Physical)*":       "Metals",
    "Iron Ore / Steel*":        "Metals",
    "Lithium*":                 "Metals",
    "Rare Earths*":             "Metals",
    "Zinc & Cobalt*":           "Metals",
    # Agriculture — futures
    "Corn (CBOT)":              "Agriculture",
    "Wheat (CBOT SRW)":         "Agriculture",
    "Wheat (KC HRW)":           "Agriculture",
    "Soybeans (CBOT)":          "Agriculture",
    "Soybean Meal":             "Agriculture",
    "Soybean Oil":              "Agriculture",
    "Coffee":                   "Agriculture",
    "Cocoa":                    "Agriculture",
    "Sugar":                    "Agriculture",
    "Cotton":                   "Agriculture",
    "Orange Juice (FCOJ-A)":    "Agriculture",
    "Oats (CBOT)":              "Agriculture",
    "Rough Rice (CBOT)":        "Agriculture",
    "Lumber*":                  "Agriculture",
    # Livestock
    "Live Cattle":              "Livestock",
    "Feeder Cattle":            "Livestock",
    "Lean Hogs":                "Livestock",
    # Digital Assets
    "Bitcoin":                  "Digital Assets",
}

# Convenience set of all known sector names — used for filtering in query helpers
KNOWN_SECTORS: frozenset = frozenset(COMMODITY_SECTORS.values())

_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_DB  = _ROOT / "data" / "commodities.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ic_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at   TEXT NOT NULL,
    commodity     TEXT NOT NULL,
    tier          TEXT NOT NULL,
    ic_value      REAL NOT NULL,
    n_obs         INTEGER NOT NULL,
    window_start  TEXT,
    window_end    TEXT,
    regime        TEXT,
    inserted_at   TEXT NOT NULL,
    UNIQUE(computed_at, commodity, tier)
);
CREATE INDEX IF NOT EXISTS idx_ic_log_tier_date
    ON ic_log(tier, computed_at);
CREATE INDEX IF NOT EXISTS idx_ic_log_commodity
    ON ic_log(commodity, computed_at);
"""

# Migration: add regime column + its index to existing tables that pre-date this field.
# Each statement is run individually so we can swallow OperationalError on repeat runs.
_MIGRATIONS = [
    "ALTER TABLE ic_log ADD COLUMN regime TEXT;",
    "CREATE INDEX IF NOT EXISTS idx_ic_log_regime ON ic_log(regime, computed_at);",
]


# ── ICResult dataclass ─────────────────────────────────────────────────────────

@dataclass
class ICResult:
    """
    Spearman IC for one (commodity, tier) pairing over the backtest window.

    Attributes
    ──────────
    commodity    : commodity display name
    tier         : model tier ("statistical", "ml", "deep", "quantum")
    ic_value     : Spearman rank correlation ∈ [−1, +1]
    n_obs        : number of (forecast, actual) pairs used
    window_start : first date in the window (YYYY-MM-DD)
    window_end   : last date in the window  (YYYY-MM-DD)
    regime       : market regime label when IC was computed, e.g. "bull",
                   "bear", "high_vol", "crisis".  Empty string = regime-agnostic.
    computed_at  : UTC ISO timestamp of computation
    """
    commodity:    str
    tier:         str
    ic_value:     float
    n_obs:        int
    window_start: str
    window_end:   str
    regime:       str = ""
    computed_at:  str = ""

    def __post_init__(self):
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    @property
    def signal_strength(self) -> str:
        """Human-readable signal strength label."""
        if self.ic_value >= IC_THRESHOLD_ACTION:
            return "actionable"
        if self.ic_value >= IC_THRESHOLD_NEUTRAL:
            return "marginal"
        return "negative"

    @property
    def badge_color(self) -> str:
        """Hex colour for UI badges."""
        return {"actionable": "#66BB6A", "marginal": "#F59E0B",
                "negative": "#EF5350"}[self.signal_strength]


# ── Core computation ───────────────────────────────────────────────────────────

def compute_ic(
    forecasts: Sequence[float],
    actuals: Sequence[float],
) -> float:
    """
    Compute Spearman IC between forecast and actual return series.

    Parameters
    ----------
    forecasts : sequence of predicted returns
    actuals   : sequence of realised returns (same length, same order)

    Returns
    -------
    float in [−1, +1], or NaN if too few observations or all-constant input.
    """
    fc_arr = np.asarray(forecasts, dtype=float)
    ac_arr = np.asarray(actuals,   dtype=float)

    # Remove rows where either value is NaN
    valid = ~(np.isnan(fc_arr) | np.isnan(ac_arr))
    fc_arr, ac_arr = fc_arr[valid], ac_arr[valid]

    if len(fc_arr) < 5:
        return float("nan")

    # Constant input → no predictive signal; ranking would be arbitrary
    if fc_arr.std() < 1e-10 or ac_arr.std() < 1e-10:
        return float("nan")

    # Rank-based correlation (Spearman) — avoids scipy version mismatch
    def _rank(arr: np.ndarray) -> np.ndarray:
        temp = arr.argsort()
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr))
        return ranks

    rf = _rank(fc_arr).astype(float)
    ra = _rank(ac_arr).astype(float)

    # Pearson on ranks = Spearman
    rf -= rf.mean(); ra -= ra.mean()
    denom = (np.linalg.norm(rf) * np.linalg.norm(ra))
    if denom < 1e-12:
        return float("nan")

    return float(np.dot(rf, ra) / denom)


def compute_ic_from_records(
    records: list,   # List[BacktestRecord] — avoid circular import
    computed_at: Optional[str] = None,
) -> Dict[Tuple[str, str], ICResult]:
    """
    Compute Spearman IC for every (commodity, tier) pair in the backtest records.

    Parameters
    ----------
    records    : list of BacktestRecord (from BacktestHarness.run())
    computed_at: ISO timestamp to stamp all results with (defaults to now UTC).

    Returns
    -------
    dict mapping (commodity, tier) → ICResult.
    Empty dict if records is empty.
    """
    if not records:
        return {}

    ts = computed_at or datetime.now(timezone.utc).isoformat()

    # Collect (forecast, actual) by (commodity, tier)
    buckets: Dict[Tuple[str, str], Tuple[List[float], List[float], List]] = {}
    for rec in records:
        for tier, fc in rec.tier_forecasts.items():
            key = (rec.commodity, tier)
            if key not in buckets:
                buckets[key] = ([], [], [])
            buckets[key][0].append(fc)
            buckets[key][1].append(rec.actual_return)
            buckets[key][2].append(rec.date)

    results: Dict[Tuple[str, str], ICResult] = {}
    for (commodity, tier), (fc_list, ac_list, dates) in buckets.items():
        ic_val = compute_ic(fc_list, ac_list)
        if np.isnan(ic_val):
            continue

        sorted_dates = sorted(dates)
        results[(commodity, tier)] = ICResult(
            commodity=commodity,
            tier=tier,
            ic_value=round(ic_val, 6),
            n_obs=len(fc_list),
            window_start=pd.Timestamp(sorted_dates[0]).strftime("%Y-%m-%d"),
            window_end=pd.Timestamp(sorted_dates[-1]).strftime("%Y-%m-%d"),
            computed_at=ts,
        )
        log.info(
            "IC[%s, %s] = %.4f  (n=%d, %s → %s)",
            commodity, tier, ic_val, len(fc_list),
            results[(commodity, tier)].window_start,
            results[(commodity, tier)].window_end,
        )

    return results


def compute_sector_ic_from_records(
    records: list,   # List[BacktestRecord]
    computed_at: Optional[str] = None,
) -> Dict[Tuple[str, str], ICResult]:
    """
    Compute sector-level Spearman IC by pooling all commodity records that
    belong to the same sector (as defined by COMMODITY_SECTORS).

    Pooling gives more (forecast, actual) pairs per (sector, tier) bucket than
    any single commodity can, producing more stable IC estimates for rare tiers
    or shorter histories.

    Typical sectors: "Energy", "Metals", "Agriculture", "Livestock",
    "Digital Assets".

    Parameters
    ----------
    records    : list of BacktestRecord (from BacktestHarness.run())
    computed_at: ISO timestamp to stamp all results with (defaults to now UTC).

    Returns
    -------
    Dict[(sector_name, tier), ICResult]
    The ICResult.commodity field holds the sector name (e.g. "Energy") so
    results can be stored in the same ic_log table and queried via
    ic_sector_summary().  Commodities not in COMMODITY_SECTORS are skipped.
    """
    if not records:
        return {}

    ts = computed_at or datetime.now(timezone.utc).isoformat()

    # Pool (forecast, actual, date) by (sector, tier)
    buckets: Dict[Tuple[str, str], Tuple[List[float], List[float], List]] = {}
    for rec in records:
        sector = COMMODITY_SECTORS.get(rec.commodity)
        if sector is None:
            continue
        for tier, fc in rec.tier_forecasts.items():
            key = (sector, tier)
            if key not in buckets:
                buckets[key] = ([], [], [])
            buckets[key][0].append(fc)
            buckets[key][1].append(rec.actual_return)
            buckets[key][2].append(rec.date)

    results: Dict[Tuple[str, str], ICResult] = {}
    for (sector, tier), (fc_list, ac_list, dates) in buckets.items():
        ic_val = compute_ic(fc_list, ac_list)
        if np.isnan(ic_val):
            continue

        sorted_dates = sorted(dates)
        result = ICResult(
            commodity=sector,      # sector name stored as commodity field
            tier=tier,
            ic_value=round(ic_val, 6),
            n_obs=len(fc_list),
            window_start=pd.Timestamp(sorted_dates[0]).strftime("%Y-%m-%d"),
            window_end=pd.Timestamp(sorted_dates[-1]).strftime("%Y-%m-%d"),
            computed_at=ts,
        )
        results[(sector, tier)] = result
        log.info(
            "Sector IC[%s, %s] = %.4f  (n=%d, %s → %s)",
            sector, tier, ic_val, len(fc_list),
            result.window_start, result.window_end,
        )

    return results


# ── SQLite persistence ─────────────────────────────────────────────────────────

def _connect(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    p = Path(db_path) if db_path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(_SCHEMA_SQL)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # already applied — normal on all runs after the first
    return conn


def log_ic_scores(
    ic_results: Dict[Tuple[str, str], ICResult],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Persist IC results to the ic_log SQLite table.

    Uses INSERT OR REPLACE to handle re-runs on the same (computed_at,
    commodity, tier) triple without creating duplicates.

    Returns
    -------
    int : number of rows written.
    """
    if not ic_results:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for result in ic_results.values():
        rows.append((
            result.computed_at,
            result.commodity,
            result.tier,
            result.ic_value,
            result.n_obs,
            result.window_start,
            result.window_end,
            result.regime or None,
            now_iso,
        ))

    conn = _connect(db_path)
    conn.executemany(
        """
        INSERT OR REPLACE INTO ic_log
            (computed_at, commodity, tier, ic_value, n_obs,
             window_start, window_end, regime, inserted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    log.info("Persisted %d IC score(s) to %s", len(rows), db_path or DEFAULT_DB)
    return len(rows)


# ── Query helpers ──────────────────────────────────────────────────────────────

def recent_ic_scores(
    days: int = 90,
    commodity: Optional[str] = None,
    tier: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Return IC log entries from the last `days` days.

    Optionally filter by commodity and/or tier.

    Returns
    -------
    pd.DataFrame with columns:
        computed_at, commodity, tier, ic_value, n_obs,
        window_start, window_end
    Sorted by computed_at DESC.  Empty DataFrame if no data.
    """
    try:
        conn = _connect(db_path)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()

        conditions = ["computed_at >= ?"]
        params: List = [cutoff]
        if commodity:
            conditions.append("commodity = ?")
            params.append(commodity)
        if tier:
            conditions.append("tier = ?")
            params.append(tier)

        where = " AND ".join(conditions)
        df = pd.read_sql_query(
            f"""
            SELECT computed_at, commodity, tier, ic_value, n_obs,
                   window_start, window_end, regime
            FROM   ic_log
            WHERE  {where}
            ORDER  BY computed_at DESC
            """,
            conn,
            params=params,
        )
        conn.close()
        return df
    except Exception as exc:
        log.warning("recent_ic_scores failed: %s", exc)
        return pd.DataFrame()


def ic_summary(
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Latest IC score per (commodity, tier) — one row per pair.

    Useful for the scorecard table in the Model Health tab.

    Returns
    -------
    pd.DataFrame with columns:
        commodity, tier, ic_value, n_obs, window_end, computed_at, signal_strength
    Sorted by tier, then commodity.  Empty DataFrame if no data.
    """
    try:
        conn = _connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT   commodity, tier,
                     ic_value, n_obs, window_end, computed_at
            FROM     ic_log
            WHERE    (commodity, tier, computed_at) IN (
                         SELECT commodity, tier, MAX(computed_at)
                         FROM   ic_log
                         GROUP  BY commodity, tier
                     )
            ORDER    BY tier, commodity
            """,
            conn,
        )
        conn.close()
        if not df.empty:
            df["signal_strength"] = df["ic_value"].apply(
                lambda v: "actionable" if v >= IC_THRESHOLD_ACTION
                else ("marginal" if v >= IC_THRESHOLD_NEUTRAL else "negative")
            )
        return df
    except Exception as exc:
        log.warning("ic_summary failed: %s", exc)
        return pd.DataFrame()


def ic_sector_summary(
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Latest IC score per (sector, tier) — one row per pair.

    Filters ic_summary() to rows whose commodity field is a known sector name
    (i.e. rows written by compute_sector_ic_from_records()).

    Useful for a sector-level scorecard in the Model Health tab, showing
    whether the ml tier outperforms statistical on Energy vs Metals vs
    Agriculture independently.

    Returns
    -------
    pd.DataFrame with columns:
        commodity (= sector name), tier, ic_value, n_obs, window_end,
        computed_at, signal_strength
    Sorted by tier, then sector.  Empty DataFrame if no sector IC logged yet.
    """
    summary = ic_summary(db_path=db_path)
    if summary.empty:
        return summary
    return summary[summary["commodity"].isin(KNOWN_SECTORS)].copy()


def ic_commodity_summary(
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Latest IC per individual commodity (excludes sector-level rows).

    Complement to ic_sector_summary() — returns only rows whose commodity
    field is NOT a known sector name, i.e. the per-commodity IC scores.
    """
    summary = ic_summary(db_path=db_path)
    if summary.empty:
        return summary
    return summary[~summary["commodity"].isin(KNOWN_SECTORS)].copy()


def ic_trend(
    days: int = 180,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Mean IC per tier over time — averaged across all commodities per timestamp.

    Returns
    -------
    pd.DataFrame with columns: computed_at, tier, mean_ic, n_commodities
    Suitable for direct use in a Plotly line chart.
    Empty DataFrame if no data.
    """
    try:
        df = recent_ic_scores(days=days, db_path=db_path)
        if df.empty:
            return pd.DataFrame()
        grouped = (
            df.groupby(["computed_at", "tier"])["ic_value"]
            .agg(mean_ic="mean", n_commodities="count")
            .reset_index()
        )
        grouped["computed_at"] = pd.to_datetime(grouped["computed_at"])
        return grouped.sort_values("computed_at")
    except Exception as exc:
        log.warning("ic_trend failed: %s", exc)
        return pd.DataFrame()
