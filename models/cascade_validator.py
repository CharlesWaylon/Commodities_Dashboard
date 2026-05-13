"""
Cascade Validator — supporting types and persistence for BacktestHarness.validate_causal_chains().

Provides:
  • Dataclasses: SectorResult, ShockResult, LagResult, ValidationReport
  • SQLite schema and write/query helpers for the cascade_validation_log table
  • Utility: detect_macro_shocks() for falling back when trigger log is empty

This module is imported by backtest_harness.py — it must not import anything
from backtest_harness.py to avoid a circular reference.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text
from database.db import get_engine

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Sector → commodity map (kept here so the validator is self-contained) ─────
SECTOR_COMMODITIES: Dict[str, List[str]] = {
    "Energy":    ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas (Henry Hub)"],
    "Metals":    ["Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)"],
    "Grains":    ["Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)"],
    "Livestock": ["Feeder Cattle", "Lean Hogs"],
}

# Per-sector macro effect coefficients — same sign convention as 7_Macro_Market_Cascade.py.
# Effect in percentage-points per unit of normalised signal.
#   real_yields > 0 → rates rising
#   usd > 0         → dollar strengthening
#   risk_off > 0    → risk-off / VIX elevated
SECTOR_MACRO_EFFECTS: Dict[str, Dict[str, float]] = {
    "Energy":    {"real_yields": -0.20, "usd": -0.50, "risk_off": -0.40},
    "Metals":    {"real_yields": -0.35, "usd":  0.05, "risk_off":  0.25},
    "Grains":    {"real_yields":  0.00, "usd": -0.30, "risk_off": -0.25},
    "Livestock": {"real_yields":  0.00, "usd": -0.10, "risk_off": -0.30},
}

# Baseline: accuracy below which we flag the cascade as under-performing
CASCADE_ACCURACY_FLOOR = 0.50   # 50% directional accuracy = coin flip
CASCADE_LIFT_FLOOR     = 0.00   # lift must be non-negative vs. baseline


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class SectorResult:
    """Cascade vs. baseline comparison for one sector on one shock event."""
    sector: str
    cascade_forecast_pct: float       # cascade-predicted 1-week return (%)
    baseline_forecast_pct: float      # rolling-momentum baseline (%)
    actual_return_5d: float           # realised 5 trading-day return (%)
    actual_return_10d: float          # realised 10 trading-day return (%)
    cascade_dir_correct: bool         # cascade got direction right vs 5d actual
    baseline_dir_correct: bool        # baseline got direction right vs 5d actual
    mae_cascade: float                # |cascade_forecast - actual_5d|
    mae_baseline: float               # |baseline_forecast - actual_5d|
    lift: float                       # mae_baseline - mae_cascade (positive = better)

    def to_dict(self) -> dict:
        return {
            "sector":                self.sector,
            "cascade_forecast_pct":  round(self.cascade_forecast_pct, 4),
            "baseline_forecast_pct": round(self.baseline_forecast_pct, 4),
            "actual_return_5d":      round(self.actual_return_5d, 4),
            "actual_return_10d":     round(self.actual_return_10d, 4),
            "cascade_dir_correct":   int(self.cascade_dir_correct),
            "baseline_dir_correct":  int(self.baseline_dir_correct),
            "mae_cascade":           round(self.mae_cascade, 4),
            "mae_baseline":          round(self.mae_baseline, 4),
            "lift":                  round(self.lift, 4),
        }


@dataclass
class LagResult:
    """
    Energy → Metals lag-structure test for one shock event.

    Tests the hypothesis that Metals returns lag Energy returns by 1-2 days.
    Computed from the 10-day post-shock window of daily returns.
    """
    energy_contemp_corr: float     # corr(Energy[t], Metals[t])     lag=0
    energy_lag1_corr:    float     # corr(Energy[t], Metals[t+1])   lag=1
    energy_lag2_corr:    float     # corr(Energy[t], Metals[t+2])   lag=2
    max_lag: int                   # lag with highest absolute correlation
    confirmed: bool                # True if max_lag ∈ {1, 2} and |lag_k| > contemp

    def to_dict(self) -> dict:
        return {
            "energy_contemp_corr": round(self.energy_contemp_corr, 4),
            "energy_lag1_corr":    round(self.energy_lag1_corr, 4),
            "energy_lag2_corr":    round(self.energy_lag2_corr, 4),
            "max_lag":             self.max_lag,
            "confirmed":           int(self.confirmed),
        }


@dataclass
class ShockResult:
    """Full validation result for one macro shock event."""
    shock_date:     str
    shock_family:   str
    shock_strength: float
    sector_results: Dict[str, SectorResult]   # sector → SectorResult
    lag_result:     Optional[LagResult]

    @property
    def n_sectors_correct(self) -> int:
        return sum(1 for sr in self.sector_results.values() if sr.cascade_dir_correct)

    @property
    def n_sectors(self) -> int:
        return len(self.sector_results)

    @property
    def directional_accuracy(self) -> float:
        if not self.sector_results:
            return float("nan")
        return self.n_sectors_correct / self.n_sectors

    @property
    def avg_lift(self) -> float:
        lifts = [sr.lift for sr in self.sector_results.values()]
        return float(np.mean(lifts)) if lifts else 0.0

    def to_dict(self) -> dict:
        return {
            "shock_date":          self.shock_date,
            "shock_family":        self.shock_family,
            "shock_strength":      round(self.shock_strength, 4),
            "directional_accuracy": round(self.directional_accuracy, 4),
            "avg_lift":            round(self.avg_lift, 4),
            "sector_results":      {k: v.to_dict() for k, v in self.sector_results.items()},
            "lag_result":          self.lag_result.to_dict() if self.lag_result else None,
        }


@dataclass
class ValidationReport:
    """
    Weekly validation report summarising causal chain health.

    Generated by BacktestHarness.validate_causal_chains().
    """
    computed_at:       str
    shock_window_days: int                  # look-back window for shock events
    n_shock_events:    int
    shock_results:     List[ShockResult]    # one per event

    # Aggregated per-sector metrics (across all shocks)
    sector_directional_accuracy: Dict[str, float]   # sector → hit rate
    sector_baseline_accuracy:    Dict[str, float]   # sector → baseline hit rate
    sector_avg_lift:             Dict[str, float]   # sector → avg MAE lift

    # Lag structure
    lag_confirmed_pct: float    # % of shocks where E→M lag was confirmed
    avg_lag1_corr:     float    # mean Energy-lag-1 → Metals correlation

    # Health signals
    flags:          List[str]   # human-readable warnings
    overall_status: str         # "healthy" | "degraded" | "stale_coefficients"

    def to_dict(self) -> dict:
        return {
            "computed_at":                self.computed_at,
            "shock_window_days":          self.shock_window_days,
            "n_shock_events":             self.n_shock_events,
            "sector_directional_accuracy":self.sector_directional_accuracy,
            "sector_baseline_accuracy":   self.sector_baseline_accuracy,
            "sector_avg_lift":            self.sector_avg_lift,
            "lag_confirmed_pct":          round(self.lag_confirmed_pct, 4),
            "avg_lag1_corr":              round(self.avg_lag1_corr, 4),
            "flags":                      self.flags,
            "overall_status":             self.overall_status,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_text(self) -> str:
        """Human-readable summary for logging or a report email."""
        lines = [
            f"=== Cascade Validation Report ===",
            f"Run: {self.computed_at[:19]} UTC",
            f"Shock events analysed: {self.n_shock_events} (last {self.shock_window_days} days)",
            f"Overall status: {self.overall_status.upper()}",
            "",
            "Sector Directional Accuracy (cascade vs baseline):",
        ]
        for sector in SECTOR_COMMODITIES:
            ca = self.sector_directional_accuracy.get(sector, float("nan"))
            ba = self.sector_baseline_accuracy.get(sector, float("nan"))
            lf = self.sector_avg_lift.get(sector, float("nan"))
            ok = "✓" if (not math.isnan(ca) and ca >= CASCADE_ACCURACY_FLOOR) else "✗"
            lines.append(
                f"  {ok}  {sector:<12}  "
                f"cascade {ca*100:5.1f}%  baseline {ba*100:5.1f}%  "
                f"lift {lf:+.3f}pp"
            )
        lines += [
            "",
            f"Energy → Metals lag confirmed: {self.lag_confirmed_pct*100:.0f}% "
            f"of events  (mean lag-1 corr {self.avg_lag1_corr:+.3f})",
            "",
        ]
        if self.flags:
            lines.append("Flags:")
            lines += [f"  ⚠  {f}" for f in self.flags]
        else:
            lines.append("  No flags — all chains within healthy thresholds.")
        return "\n".join(lines)


# ── SQLAlchemy persistence helpers ────────────────────────────────────────────


def save_validation_report(
    report: ValidationReport,
) -> int:
    """
    Persist a ValidationReport to the cascade_validation_log and
    cascade_validation_summary tables.

    Returns the number of detail rows written.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    detail_rows = []

    for shock in report.shock_results:
        lag_ok = int(shock.lag_result.confirmed) if shock.lag_result else 0
        for sector, sr in shock.sector_results.items():
            detail_rows.append({
                "run_at":           report.computed_at,
                "shock_date":       shock.shock_date,
                "shock_family":     shock.shock_family,
                "shock_strength":   shock.shock_strength,
                "sector":           sector,
                "cascade_fcast":    sr.cascade_forecast_pct,
                "baseline_fcast":   sr.baseline_forecast_pct,
                "actual_5d":        sr.actual_return_5d,
                "actual_10d":       sr.actual_return_10d,
                "cascade_correct":  int(sr.cascade_dir_correct),
                "baseline_correct": int(sr.baseline_dir_correct),
                "mae_cascade":      sr.mae_cascade,
                "mae_baseline":     sr.mae_baseline,
                "lift":             sr.lift,
                "lag_confirmed":    lag_ok,
                "inserted_at":      now_iso,
            })

    summary_row = {
        "run_at":               report.computed_at,
        "shock_window_days":    report.shock_window_days,
        "n_shock_events":       report.n_shock_events,
        "sector_accuracy_json": json.dumps(report.sector_directional_accuracy),
        "sector_lift_json":     json.dumps(report.sector_avg_lift),
        "lag_confirmed_pct":    report.lag_confirmed_pct,
        "avg_lag1_corr":        report.avg_lag1_corr,
        "overall_status":       report.overall_status,
        "flags_json":           json.dumps(report.flags),
        "inserted_at":          now_iso,
    }

    detail_sql = text("""
        INSERT INTO cascade_validation_log
            (run_at, shock_date, shock_family, shock_strength, sector,
             cascade_fcast, baseline_fcast, actual_5d, actual_10d,
             cascade_correct, baseline_correct, mae_cascade, mae_baseline,
             lift, lag_confirmed, inserted_at)
        VALUES (:run_at, :shock_date, :shock_family, :shock_strength, :sector,
                :cascade_fcast, :baseline_fcast, :actual_5d, :actual_10d,
                :cascade_correct, :baseline_correct, :mae_cascade, :mae_baseline,
                :lift, :lag_confirmed, :inserted_at)
        ON CONFLICT (run_at, shock_date, shock_family, sector) DO UPDATE SET
            cascade_correct  = EXCLUDED.cascade_correct,
            baseline_correct = EXCLUDED.baseline_correct,
            mae_cascade      = EXCLUDED.mae_cascade,
            mae_baseline     = EXCLUDED.mae_baseline,
            lift             = EXCLUDED.lift,
            inserted_at      = EXCLUDED.inserted_at
    """)
    summary_sql = text("""
        INSERT INTO cascade_validation_summary
            (run_at, shock_window_days, n_shock_events,
             sector_accuracy_json, sector_lift_json,
             lag_confirmed_pct, avg_lag1_corr,
             overall_status, flags_json, inserted_at)
        VALUES (:run_at, :shock_window_days, :n_shock_events,
                :sector_accuracy_json, :sector_lift_json,
                :lag_confirmed_pct, :avg_lag1_corr,
                :overall_status, :flags_json, :inserted_at)
        ON CONFLICT (run_at) DO UPDATE SET
            n_shock_events       = EXCLUDED.n_shock_events,
            sector_accuracy_json = EXCLUDED.sector_accuracy_json,
            sector_lift_json     = EXCLUDED.sector_lift_json,
            overall_status       = EXCLUDED.overall_status,
            flags_json           = EXCLUDED.flags_json,
            inserted_at          = EXCLUDED.inserted_at
    """)

    with get_engine().connect() as conn:
        if detail_rows:
            conn.execute(detail_sql, detail_rows)
        conn.execute(summary_sql, summary_row)
        conn.commit()

    log.info(
        "Cascade validation: %d detail rows + 1 summary row written (status=%s)",
        len(detail_rows), report.overall_status,
    )
    return len(detail_rows)


def recent_validation_reports(n: int = 12) -> pd.DataFrame:
    """
    Return the last `n` validation summary rows as a DataFrame.

    Columns: run_at, shock_window_days, n_shock_events, overall_status,
             sector_accuracy_json (parsed dict), lag_confirmed_pct, flags_json.
    """
    try:
        sql = text(f"""
            SELECT run_at, shock_window_days, n_shock_events,
                   sector_accuracy_json, sector_lift_json,
                   lag_confirmed_pct, avg_lag1_corr,
                   overall_status, flags_json
            FROM   cascade_validation_summary
            ORDER  BY run_at DESC
            LIMIT  {int(n)}
        """)
        with get_engine().connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        for col in ("sector_accuracy_json", "sector_lift_json", "flags_json"):
            if col in df.columns:
                df[col] = df[col].apply(lambda s: json.loads(s) if s else {})
        return df
    except Exception as exc:
        log.warning("recent_validation_reports: %s", exc)
        return pd.DataFrame()


def validation_detail(
    run_at: Optional[str] = None,
    sector: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return event-level detail rows, optionally filtered by run timestamp or sector.
    Defaults to the most recent run.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            if run_at is None:
                row = conn.execute(
                    text("SELECT MAX(run_at) FROM cascade_validation_summary")
                ).fetchone()
                run_at = row[0] if row and row[0] else None
            if run_at is None:
                return pd.DataFrame()

            conditions = ["run_at = :run_at"]
            params: dict = {"run_at": run_at}
            if sector:
                conditions.append("sector = :sector")
                params["sector"] = sector

            sql = text(f"""
                SELECT shock_date, shock_family, shock_strength, sector,
                       cascade_fcast, baseline_fcast, actual_5d, actual_10d,
                       cascade_correct, baseline_correct, mae_cascade, mae_baseline,
                       lift, lag_confirmed
                FROM   cascade_validation_log
                WHERE  {' AND '.join(conditions)}
                ORDER  BY shock_date DESC, sector
            """)
            result = conn.execute(sql, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception as exc:
        log.warning("validation_detail: %s", exc)
        return pd.DataFrame()


# ── Shock detection ────────────────────────────────────────────────────────────

def detect_macro_shocks(
    macro_df: pd.DataFrame,
    min_days_apart: int = 5,
) -> List[Dict]:
    """
    Detect macro shock events from the macro overlay DataFrame when the
    trigger log is empty or unavailable.

    Identifies:
      • VIX rising above 20 (risk-off onset)
      • DXY z-score exceeding +1.5 (dollar strength spike)
      • TLT 21d momentum falling below −5% (rates rising sharply)

    Parameters
    ----------
    macro_df : pd.DataFrame
        From build_macro_overlay_features(). Needs vix, dxy_zscore63, tlt_mom21.
    min_days_apart : int
        Minimum calendar days between recorded events of the same type.
        Prevents the same sustained move from being counted multiple times.

    Returns
    -------
    List of dicts with keys: date, family, strength, description.
    """
    if macro_df is None or macro_df.empty:
        return []

    events = []
    last_dates: Dict[str, Optional[pd.Timestamp]] = {
        "vix_spike": None, "dxy_spike": None, "rates_rising": None
    }

    def _gap_ok(kind: str, dt: pd.Timestamp) -> bool:
        prev = last_dates[kind]
        if prev is None:
            return True
        return (dt - prev).days >= min_days_apart

    vix_s  = macro_df.get("vix", pd.Series(dtype=float)).dropna()
    dxy_s  = macro_df.get("dxy_zscore63", pd.Series(dtype=float)).dropna()
    tlt_s  = macro_df.get("tlt_mom21", pd.Series(dtype=float)).dropna()

    for i in range(1, len(vix_s)):
        dt = vix_s.index[i]
        if vix_s.iloc[i] >= 20 and vix_s.iloc[i - 1] < 20 and _gap_ok("vix_spike", dt):
            strength = min(1.0, (vix_s.iloc[i] - 20) / 20.0 + 0.5)
            events.append({
                "date":        dt,
                "family":      "vix_spike",
                "strength":    round(strength, 3),
                "description": f"VIX → {vix_s.iloc[i]:.1f}",
            })
            last_dates["vix_spike"] = dt

    for i in range(1, len(dxy_s)):
        dt = dxy_s.index[i]
        if dxy_s.iloc[i] >= 1.5 and dxy_s.iloc[i - 1] < 1.5 and _gap_ok("dxy_spike", dt):
            strength = min(1.0, dxy_s.iloc[i] / 3.0)
            events.append({
                "date":        dt,
                "family":      "dxy_spike",
                "strength":    round(strength, 3),
                "description": f"DXY z → {dxy_s.iloc[i]:.2f}",
            })
            last_dates["dxy_spike"] = dt

    for i in range(1, len(tlt_s)):
        dt = tlt_s.index[i]
        if tlt_s.iloc[i] <= -0.05 and tlt_s.iloc[i - 1] > -0.05 and _gap_ok("rates_rising", dt):
            strength = min(1.0, abs(tlt_s.iloc[i]) / 0.10)
            events.append({
                "date":        dt,
                "family":      "rates_rising",
                "strength":    round(strength, 3),
                "description": f"TLT mom → {tlt_s.iloc[i]*100:.1f}%",
            })
            last_dates["rates_rising"] = dt

    return sorted(events, key=lambda e: e["date"])


# ── Core computation helpers ───────────────────────────────────────────────────

def _macro_signals_at(macro_df: pd.DataFrame, dt: pd.Timestamp) -> Dict[str, float]:
    """Extract normalised macro channel signals at a given date."""
    if macro_df is None or macro_df.empty:
        return {"real_yields_signal": 0.0, "usd_signal": 0.0, "risk_off_signal": 0.0}

    aligned = macro_df.reindex(macro_df.index.union([dt])).ffill()
    if dt not in aligned.index:
        return {"real_yields_signal": 0.0, "usd_signal": 0.0, "risk_off_signal": 0.0}

    row   = aligned.loc[dt]
    dxy_z = float(row.get("dxy_zscore63", 0.0) or 0.0)
    vix   = float(row.get("vix", 18.0) or 18.0)
    tlt21 = float(row.get("tlt_mom21", 0.0) or 0.0)

    return {
        "real_yields_signal": max(-1.0, min(1.0, -tlt21 / 0.05)),
        "usd_signal":         max(-1.0, min(1.0,  dxy_z / 2.0)),
        "risk_off_signal":    max(0.0,  min(1.0,  (vix - 15.0) / 20.0)),
    }


def _sector_cascade_forecast(signals: Dict[str, float]) -> Dict[str, float]:
    """Compute cascade-predicted 1-week return (%) for each sector from macro signals."""
    ry  = signals["real_yields_signal"]
    usd = signals["usd_signal"]
    ro  = signals["risk_off_signal"]
    out = {}
    for sector, efx in SECTOR_MACRO_EFFECTS.items():
        out[sector] = (
            efx["real_yields"] * ry +
            efx["usd"]         * usd +
            efx["risk_off"]    * ro
        )
    return out


def _sector_baseline_forecast(prices: pd.DataFrame, dt: pd.Timestamp) -> Dict[str, float]:
    """
    Rolling 21-day mean log-return for each sector, expressed as
    a 1-week (5 trading day) forecast in percent.  Pure momentum baseline.
    """
    out = {}
    for sector, comms in SECTOR_COMMODITIES.items():
        avail   = [c for c in comms if c in prices.columns]
        if not avail:
            out[sector] = 0.0
            continue

        history = prices[avail].loc[:dt].tail(22)
        if len(history) < 5:
            out[sector] = 0.0
            continue

        log_rets = np.log(history / history.shift(1)).dropna()
        # Mean daily log-return → 5-day ahead in %
        out[sector] = float(log_rets.mean().mean() * 5 * 100)

    return out


def _sector_actual_returns(
    prices: pd.DataFrame,
    dt: pd.Timestamp,
    forward_days: int,
) -> Dict[str, float]:
    """
    Compute realised `forward_days`-ahead return (%) for each sector
    starting the trading day after `dt`.
    """
    out = {}
    for sector, comms in SECTOR_COMMODITIES.items():
        avail = [c for c in comms if c in prices.columns]
        if not avail:
            out[sector] = float("nan")
            continue

        idx = prices.index.searchsorted(dt)
        if idx >= len(prices) - forward_days:
            out[sector] = float("nan")
            continue

        p0 = prices[avail].iloc[idx]
        p1 = prices[avail].iloc[min(idx + forward_days, len(prices) - 1)]
        fwd = ((p1 / p0) - 1) * 100
        out[sector] = float(fwd.mean())

    return out


def _lag_test(prices: pd.DataFrame, dt: pd.Timestamp, window: int = 10) -> Optional[LagResult]:
    """
    Test whether Energy returns lead Metals returns in the `window`-day window
    following `dt`.  Computes daily cross-correlations at lag 0, 1, 2.
    """
    energy_comms = [c for c in SECTOR_COMMODITIES["Energy"] if c in prices.columns]
    metals_comms = [c for c in SECTOR_COMMODITIES["Metals"]  if c in prices.columns]

    if not energy_comms or not metals_comms:
        return None

    idx = prices.index.searchsorted(dt)
    end = min(idx + window + 2, len(prices))
    if end - idx < 5:
        return None

    slice_ = prices.iloc[idx:end]
    e_ret  = np.log(slice_[energy_comms] / slice_[energy_comms].shift(1)).dropna().mean(axis=1)
    m_ret  = np.log(slice_[metals_comms] / slice_[metals_comms].shift(1)).dropna().mean(axis=1)

    if len(e_ret) < 4 or len(m_ret) < 4:
        return None

    def _corr(x: pd.Series, y: pd.Series) -> float:
        aligned = pd.concat([x, y], axis=1).dropna()
        if len(aligned) < 3 or aligned.iloc[:, 0].std() < 1e-10 or aligned.iloc[:, 1].std() < 1e-10:
            return float("nan")
        return float(aligned.corr().iloc[0, 1])

    c0 = _corr(e_ret,      m_ret)
    c1 = _corr(e_ret[:-1], m_ret[1:])
    c2 = _corr(e_ret[:-2], m_ret[2:]) if len(e_ret) > 4 else float("nan")

    abs_lags = {0: abs(c0), 1: abs(c1), 2: abs(c2)}
    abs_lags  = {k: v for k, v in abs_lags.items() if not math.isnan(v)}
    if not abs_lags:
        return None

    max_lag = max(abs_lags, key=lambda k: abs_lags[k])
    confirmed = (
        max_lag in (1, 2)
        and not math.isnan(c0)
        and abs_lags[max_lag] > abs(c0)
    )

    return LagResult(
        energy_contemp_corr = c0 if not math.isnan(c0) else 0.0,
        energy_lag1_corr    = c1 if not math.isnan(c1) else 0.0,
        energy_lag2_corr    = c2 if not math.isnan(c2) else 0.0,
        max_lag             = max_lag,
        confirmed           = confirmed,
    )
