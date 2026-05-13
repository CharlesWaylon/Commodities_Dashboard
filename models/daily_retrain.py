"""
Daily Retraining Pipeline — Phase 5 Part 1 of the Model Integration Roadmap.

Purpose
───────
Closes the loop: until this script runs, MetaPredictor always falls back to
equal weights ("untrained"), making the Consensus tab and every Causal Chain
ensemble node meaningless.

This module:
  1. Loads prices + macro overlays from the local SQLite DB
  2. Runs BacktestHarness.train_meta_predictor() across all core commodities
  3. Saves the trained MetaPredictor to data/meta_predictor.pkl
  4. Writes a one-row audit entry to the `model_training_log` SQLite table
  5. Prints a human-readable summary

Usage
─────
  # From the project root (recommended after market close Mon–Fri):
  python -m models.daily_retrain

  # Programmatic:
  from models.daily_retrain import run_daily_retrain, RetrainConfig
  summary = run_daily_retrain()

  # With custom config:
  cfg = RetrainConfig(max_depth=4, min_samples_leaf=5)
  summary = run_daily_retrain(config=cfg)

Automation
──────────
Hook this into launchd (macOS) or cron so it runs daily after the 18:05 UTC
ingest completes — e.g. at 18:20 UTC Mon–Fri.

crontab example:
  20 18 * * 1-5  cd ~/Desktop/Future_of_Commodities/Commodities_Dashboard && \
                 python -m models.daily_retrain >> logs/retrain.log 2>&1

launchd: generate a .plist with the same command and set StartCalendarInterval.

Design notes
────────────
• Prices are loaded DB-first (services.data_contract) with yfinance fallback.
• Macro overlays are built fresh from the macro_overlays feature module.
• If the backtest produces < MIN_PAIRS training records, the run is aborted and
  the existing pkl (if any) is NOT overwritten.
• Each retraining run writes to model_training_log regardless of success/failure
  so you can diagnose regressions over time.
• All errors are caught and logged; the script always exits 0 so a cron job
  does not spam alerts on transient yfinance failures.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlalchemy import text
from database.db import get_engine

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_PKL  = _ROOT / "data" / "meta_predictor.pkl"

# ── Core commodity universe (mirrors pages/4_Models.py) ───────────────────────
CORE_TICKERS: Dict[str, str] = {
    "WTI Crude Oil":           "CL=F",
    "Brent Crude Oil":         "BZ=F",
    "Natural Gas (Henry Hub)": "NG=F",
    "Gasoline (RBOB)":         "RB=F",
    "Heating Oil":             "HO=F",
    "Gold (COMEX)":            "GC=F",
    "Silver (COMEX)":          "SI=F",
    "Copper (COMEX)":          "HG=F",
    "Corn (CBOT)":             "ZC=F",
    "Wheat (CBOT SRW)":        "ZW=F",
    "Soybeans (CBOT)":         "ZS=F",
}

# Minimum training pairs before the predictor is considered usable
MIN_PAIRS = 20


# Commodity groups for Granger causality (names must match prices.columns exactly)
_GRANGER_GROUPS: Dict[str, List[str]] = {
    "energy":       ["WTI Crude Oil", "Brent Crude Oil", "Natural Gas (Henry Hub)", "Heating Oil"],
    "metals":       ["Gold (COMEX)", "Silver (COMEX)", "Copper (COMEX)"],
    "grains":       ["Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)"],
    "cross_sector": ["WTI Crude Oil", "Gold (COMEX)", "Copper (COMEX)", "Corn (CBOT)"],
}

# Path to macro routes pkl (same location as macro_router.py uses)
_MACRO_ROUTES_PKL = Path(__file__).resolve().parent / "macro_routes.pkl"

# Alert thresholds
_CASCADE_ACCURACY_DROP_THRESHOLD = 0.10    # ≥10pp drop flags a sector
_ROUTE_SHIFT_THRESHOLD           = 0.20    # ≥20% relative change in routing slope
_SECTOR_IC_FLOOR                 = 0.0     # sector IC below zero triggers alert
_GRANGER_STALE_DAYS              = 7       # Granger refresh cadence (days)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrainConfig:
    """
    Parameters that control a retraining run.

    Attributes
    ──────────
    commodities     : list of commodity names to backtest. Defaults to all
                      CORE_TICKERS if None.
    prices_period   : how far back to fetch prices (e.g. "3y").
    macro_period    : how far back to fetch macro overlays.
    max_depth       : DecisionTree max_depth for MetaPredictor.
    min_samples_leaf: minimum leaf size (prevents over-fitting on small data).
    save_path       : where to save the pkl. None → DEFAULT_PKL.
    dry_run         : if True, trains but does NOT save the pkl or write the log.
    """
    commodities:      Optional[List[str]] = None
    prices_period:    str                 = "3y"
    macro_period:     str                 = "2y"
    max_depth:        int                 = 5
    min_samples_leaf: int                 = 10
    save_path:             Optional[Path] = None
    dry_run:               bool           = False
    skip_causal_monitoring: bool          = False
    force_granger:         bool           = False


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrainSummary:
    """
    Outcome of one retraining run.

    Attributes
    ──────────
    retrained_at        : UTC ISO timestamp when the run started.
    n_commodities       : number of commodities that contributed training records.
    n_training_pairs    : total (MetaFeatures, winning_tier) pairs generated.
    tier_distribution   : {tier: count} across all training pairs.
    tree_n_leaves       : number of leaves in the fitted decision tree (-1 if untrained).
    top_feature         : most important feature by Gini importance (or "" if untrained).
    save_path           : where the pkl was written (empty string if dry_run or failed).
    error               : non-empty if the run aborted with an exception.
    config              : the RetrainConfig used (for audit).
    success             : True if predictor was trained and saved (or dry_run succeeded).
    n_corr_pairs        : number of correlation pairs stored in correlation_snapshots.
    n_forecasts_logged  : number of forecast rows written to forecast_log.
    consistency_flags   : list of human-readable inconsistency messages from
                          check_forecast_consistency().
    n_cascade_forecasts : number of commodity forecasts written by the cascade.
    cascade_regime      : market regime detected by the cascade orchestrator.
    """
    retrained_at:       str               = ""
    n_commodities:      int               = 0
    n_training_pairs:   int               = 0
    tier_distribution:  Dict[str, int]    = field(default_factory=dict)
    tree_n_leaves:      int               = -1
    top_feature:        str               = ""
    save_path:          str               = ""
    error:              str               = ""
    config:             Optional[RetrainConfig] = None
    success:            bool              = False
    n_corr_pairs:       int               = 0
    n_forecasts_logged: int               = 0
    consistency_flags:  List[str]         = field(default_factory=list)
    n_cascade_forecasts:      int        = 0
    cascade_regime:           str        = ""
    granger_refresh_done:     bool       = False
    route_shift_alerts:       List[str]  = field(default_factory=list)
    cascade_validation_status: str       = ""
    n_monitoring_alerts:      int        = 0

    def pretty(self) -> str:
        """Multi-line human-readable summary for CLI output."""
        status = "SUCCESS" if self.success else f"FAILED — {self.error}"
        lines = [
            "=" * 60,
            "  MetaPredictor Daily Retraining Summary",
            "=" * 60,
            f"  Status          : {status}",
            f"  Retrained at    : {self.retrained_at}",
            f"  Commodities     : {self.n_commodities}",
            f"  Training pairs  : {self.n_training_pairs}",
        ]
        if self.tier_distribution:
            dist_str = "  |  ".join(
                f"{k}: {v}" for k, v in sorted(self.tier_distribution.items())
            )
            lines.append(f"  Tier dist       : {dist_str}")
        if self.tree_n_leaves > 0:
            lines.append(f"  Tree leaves     : {self.tree_n_leaves}")
        if self.top_feature:
            lines.append(f"  Top feature     : {self.top_feature}")
        if self.save_path:
            lines.append(f"  Saved to        : {self.save_path}")
        if self.n_corr_pairs:
            lines.append(f"  Corr pairs      : {self.n_corr_pairs}")
        if self.n_forecasts_logged:
            lines.append(f"  Forecasts logged: {self.n_forecasts_logged}")
        if self.n_cascade_forecasts:
            lines.append(f"  Cascade forecasts: {self.n_cascade_forecasts}  (regime={self.cascade_regime})")
        if self.consistency_flags:
            lines.append(f"  Consistency flags ({len(self.consistency_flags)}):")
            for msg in self.consistency_flags[:5]:
                lines.append(f"    ⚠  {msg}")
        if self.cascade_validation_status:
            lines.append(f"  Cascade health  : {self.cascade_validation_status}")
        if self.granger_refresh_done:
            lines.append("  Granger refresh : done (weekly)")
        if self.n_monitoring_alerts:
            lines.append(f"  Causal alerts   : {self.n_monitoring_alerts}")
            for alert in self.route_shift_alerts[:4]:
                lines.append(f"    🔔  {alert}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_prices(period: str, commodities: Optional[List[str]]) -> pd.DataFrame:
    """
    Load price matrix DB-first, yfinance fallback.
    Returns only the columns we need.
    """
    names = commodities or list(CORE_TICKERS.keys())
    tickers_needed = {n: CORE_TICKERS[n] for n in names if n in CORE_TICKERS}

    # DB-first
    try:
        from services.data_contract import fetch_price_matrix
        days = int(period[:-1]) * 365 if period.endswith("y") else 365
        df = fetch_price_matrix(names=names, days=days)
        if not df.empty:
            log.info("Prices loaded from DB: %d rows × %d cols", len(df), df.shape[1])
            return df
    except Exception as exc:
        log.warning("DB price load failed (%s); trying yfinance…", exc)

    # yfinance fallback
    import yfinance as yf
    raw = yf.download(
        list(tickers_needed.values()), period=period,
        interval="1d", progress=False, auto_adjust=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    prices = raw.rename(columns={v: k for k, v in tickers_needed.items()})
    log.info("Prices loaded from yfinance: %d rows × %d cols", len(prices), prices.shape[1])
    return prices.ffill().dropna()


def _load_macro(period: str, price_index: Optional[pd.DatetimeIndex]) -> pd.DataFrame:
    """Build macro overlay features; empty DataFrame on failure."""
    try:
        from features.macro_overlays import build_macro_overlay_features
        df = build_macro_overlay_features(period=period, index=price_index)
        log.info("Macro overlays loaded: %d rows × %d cols", len(df), df.shape[1])
        return df
    except Exception as exc:
        log.warning("Macro overlay load failed (%s); proceeding without macro.", exc)
        return pd.DataFrame()


def _load_climate(price_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    Fetch climate features (MEI/ENSO + PDSI if NOAA token present).

    MEI requires no key and is always attempted. PDSI/HDD require NOAA_CDO_TOKEN.
    Returns an empty DataFrame on failure so the retrain can continue without
    climate features — models degrade gracefully to price-only features.
    """
    import os
    try:
        from features.climate_weather import build_climate_features
        start = "2020-01-01"
        if price_index is not None and len(price_index) > 0:
            start = price_index.min().strftime("%Y-%m-%d")
        df = build_climate_features(
            noaa_token=os.environ.get("NOAA_CDO_TOKEN"),
            start_date=start,
        )
        if not df.empty:
            log.info("Climate features loaded: %d rows × %d cols", len(df), df.shape[1])
        else:
            log.info("Climate features: empty (NOAA token absent or fetch failed)")
        return df
    except Exception as exc:
        log.warning("Climate feature load failed (%s); proceeding without climate.", exc)
        return pd.DataFrame()


# ── SQLAlchemy audit log ───────────────────────────────────────────────────────

def _persist_training_log(summary: RetrainSummary) -> None:
    """Write one row to model_training_log. Silent on failure."""
    try:
        row = {
            "retrained_at":      summary.retrained_at,
            "n_commodities":     summary.n_commodities,
            "n_training_pairs":  summary.n_training_pairs,
            "tier_distribution": json.dumps(summary.tier_distribution),
            "tree_n_leaves":     summary.tree_n_leaves,
            "top_feature":       summary.top_feature,
            "save_path":         summary.save_path,
            "error":             summary.error,
            "config_json":       json.dumps(
                asdict(summary.config),
                default=lambda v: str(v),
            ) if summary.config else "{}",
            "inserted_at":       datetime.now(timezone.utc).isoformat(),
        }
        sql = text("""
            INSERT INTO model_training_log
                (retrained_at, n_commodities, n_training_pairs,
                 tier_distribution, tree_n_leaves, top_feature,
                 save_path, error, config_json, inserted_at)
            VALUES (:retrained_at, :n_commodities, :n_training_pairs,
                    :tier_distribution, :tree_n_leaves, :top_feature,
                    :save_path, :error, :config_json, :inserted_at)
        """)
        with get_engine().connect() as conn:
            conn.execute(sql, row)
            conn.commit()
        log.info("Training log persisted.")
    except Exception as exc:
        log.warning("Could not write training log: %s", exc)


def recent_training_runs(n: int = 10) -> pd.DataFrame:
    """
    Return the last `n` training log entries as a DataFrame.
    Useful for the dashboard Model Health tab (Phase 5 Part 2).
    Returns empty DataFrame if the log table doesn't exist yet.
    """
    try:
        sql = text(f"""
            SELECT retrained_at, n_commodities, n_training_pairs,
                   tier_distribution, tree_n_leaves, top_feature,
                   save_path, error
            FROM   model_training_log
            ORDER  BY retrained_at DESC
            LIMIT  {int(n)}
        """)
        with get_engine().connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception:
        return pd.DataFrame()


# ── Causal monitoring helpers ─────────────────────────────────────────────────

def _granger_is_stale() -> bool:
    """True if Granger has not been refreshed within the last _GRANGER_STALE_DAYS."""
    try:
        sql = text(
            "SELECT logged_at FROM causal_monitoring_log "
            "WHERE granger_refreshed=1 ORDER BY logged_at DESC LIMIT 1"
        )
        with get_engine().connect() as conn:
            row = conn.execute(sql).fetchone()
        if not row:
            return True
        last_dt  = datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - last_dt).days
        return age_days >= _GRANGER_STALE_DAYS
    except Exception:
        return True


def _run_granger_refresh(
    prices:   pd.DataFrame,
    force:    bool = False,
    max_lag:  int  = 5,
    lookback: int  = 504,
) -> Dict:
    """
    VAR Granger causality on each commodity group; returns:
      refreshed : bool
      summary   : {group: {caused: {cause: p_value}}}
      error     : str
    Only runs when stale (>7 days) or force=True.
    """
    if not force and not _granger_is_stale():
        return {"refreshed": False, "summary": {}, "error": ""}

    from models.statistical.var_vecm import CommodityVAR

    summary: Dict[str, Dict] = {}
    errors:  List[str]       = []
    prices_sub = prices.iloc[-lookback:] if len(prices) > lookback else prices

    for group_name, members in _GRANGER_GROUPS.items():
        avail = [m for m in members if m in prices_sub.columns]
        if len(avail) < 2:
            log.debug(
                "Granger group %r: only %d column(s) available — skipping.",
                group_name, len(avail),
            )
            continue
        try:
            var = CommodityVAR(group="custom", custom_cols=avail, use_returns=True)
            var.fit(prices_sub[avail].dropna())
            gc_df = var.granger_causality(max_lag=max_lag)
            summary[group_name] = {
                caused: {
                    cause: float(gc_df.loc[caused, cause])
                    for cause in gc_df.columns
                    if cause != caused and not np.isnan(gc_df.loc[caused, cause])
                }
                for caused in gc_df.index
            }
            log.info(
                "Granger causality refreshed for group %r (%d commodities).",
                group_name, len(avail),
            )
        except Exception as exc:
            errors.append(f"{group_name}: {exc}")
            log.warning("Granger refresh failed for %r: %s", group_name, exc)

    return {
        "refreshed": bool(summary),
        "summary":   summary,
        "error":     "; ".join(errors),
    }


def _load_last_monitoring_field(field_name: str) -> Dict:
    """Generic helper: load one JSON field from the most recent monitoring row."""
    try:
        sql = text(
            f"SELECT {field_name} FROM causal_monitoring_log "
            f"WHERE {field_name} IS NOT NULL ORDER BY logged_at DESC LIMIT 1"
        )
        with get_engine().connect() as conn:
            row = conn.execute(sql).fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception:
        pass
    return {}


def _detect_route_shifts() -> Dict:
    """
    Load current macro_routes.pkl → compare routing_matrix to previous run.
    Returns: coefficients, shift_pcts, alerts.
    Alert threshold: |pct_shift| ≥ _ROUTE_SHIFT_THRESHOLD.
    """
    result: Dict = {"coefficients": {}, "shift_pcts": {}, "alerts": []}
    try:
        import pickle
        if not _MACRO_ROUTES_PKL.exists():
            return result
        with open(_MACRO_ROUTES_PKL, "rb") as f:
            routes = pickle.load(f)
        routing_matrix: Dict = routes.get("routing_matrix", {})

        # Flatten: {label → {sector: slope}}
        current_coefs: Dict[str, Dict[str, float]] = {
            label: {s: float(v) for s, v in sd.items()}
            for label, sd in routing_matrix.items()
        }
        result["coefficients"] = current_coefs

        prev_coefs = _load_last_monitoring_field("route_coefficients_json")
        shift_pcts: Dict[str, Dict[str, float]] = {}
        alerts:     List[str]                   = []

        for mv, sectors in current_coefs.items():
            prev_mv = prev_coefs.get(mv, {})
            shift_pcts[mv] = {}
            for sector, slope in sectors.items():
                prev_slope = prev_mv.get(sector)
                if prev_slope is None:
                    continue
                denom = max(abs(float(prev_slope)), 1e-4)
                pct   = (slope - float(prev_slope)) / denom
                shift_pcts[mv][sector] = round(pct, 4)
                if abs(pct) >= _ROUTE_SHIFT_THRESHOLD:
                    alerts.append(
                        f"ROUTE SHIFT: {mv}→{sector} shifted {pct:+.0%} "
                        f"({float(prev_slope):.4f} → {slope:.4f}) — manual review advised."
                    )

        result["shift_pcts"] = shift_pcts
        result["alerts"]     = alerts

    except Exception as exc:
        log.warning("Route shift detection failed: %s", exc)

    return result


def _check_sector_ic_alerts(
    current_ic: Dict[str, float],
    prev_ic:    Dict[str, float],
) -> List[str]:
    """Return alert strings for IC below floor or ≥10pp drop vs prior run."""
    alerts: List[str] = []
    for sector, ic in current_ic.items():
        if ic < _SECTOR_IC_FLOOR:
            alerts.append(
                f"SECTOR IC BELOW FLOOR: {sector} IC={ic:.4f} "
                f"(threshold={_SECTOR_IC_FLOOR}) — signal may be noise."
            )
        if sector in prev_ic:
            drop = float(prev_ic[sector]) - ic
            if drop >= 0.10:
                alerts.append(
                    f"SECTOR IC DROP: {sector} IC dropped "
                    f"{float(prev_ic[sector]):.4f} → {ic:.4f} (−{drop:.4f})."
                )
    return alerts


def _run_cascade_validation_step(
    harness,
    prices:   pd.DataFrame,
    macro:    pd.DataFrame,
    dry_run:  bool = False,
) -> Dict:
    """
    Call harness.validate_causal_chains() and build alert list.
    Returns: status, accuracy {sector: float}, n_events, alerts.
    """
    result: Dict = {
        "status":   "insufficient_data",
        "accuracy": {},
        "n_events": 0,
        "alerts":   [],
    }
    try:
        report = harness.validate_causal_chains(
            prices,
            macro,
            shock_window_days      = 365,
            forward_days_primary   = 5,
            forward_days_secondary = 10,
            save                   = not dry_run,
        )
        result["status"]   = report.overall_status
        result["accuracy"] = {k: round(v, 4) for k, v in report.sector_directional_accuracy.items()}
        result["n_events"] = report.n_shock_events

        # Compare accuracy to previous run
        prev_acc = _load_last_monitoring_field("cascade_accuracy_json")
        for sector, acc in result["accuracy"].items():
            prev = prev_acc.get(sector)
            if prev is not None:
                drop = float(prev) - acc
                if drop >= _CASCADE_ACCURACY_DROP_THRESHOLD:
                    result["alerts"].append(
                        f"CASCADE ACCURACY DROP: {sector} dropped "
                        f"{float(prev):.0%} → {acc:.0%} (−{drop:.0%}) "
                        f"— cascade coefficients may need recalibration."
                    )

        # Surface harness health flags as lower-severity alerts
        for flag in (report.flags or []):
            result["alerts"].append(f"CASCADE FLAG: {flag}")

        log.info(
            "Cascade validation: status=%s, %d shock events, %d sector(s).",
            result["status"], result["n_events"], len(result["accuracy"]),
        )

    except Exception as exc:
        log.warning("Cascade validation step failed: %s", exc)
        result["status"] = "error"

    return result


def _persist_causal_monitoring(
    logged_at:  str,
    run_type:   str,
    granger:    Dict,
    routes:     Dict,
    sector_ic:  Dict[str, float],
    cascade:    Dict,
    all_alerts: List[str],
) -> None:
    """Write one row to causal_monitoring_log. Silent on failure."""
    try:
        row = {
            "logged_at":               logged_at,
            "run_type":                run_type,
            "granger_refreshed":       1 if granger.get("refreshed") else 0,
            "granger_summary_json":    json.dumps(granger.get("summary", {})),
            "route_coefficients_json": json.dumps(routes.get("coefficients", {})),
            "route_shift_pct_json":    json.dumps(routes.get("shift_pcts", {})),
            "sector_ic_json":          json.dumps(sector_ic),
            "cascade_status":          cascade.get("status", ""),
            "cascade_accuracy_json":   json.dumps(cascade.get("accuracy", {})),
            "cascade_n_events":        cascade.get("n_events", 0),
            "alerts_json":             json.dumps(all_alerts),
            "n_alerts":                len(all_alerts),
        }
        sql = text("""
            INSERT INTO causal_monitoring_log
                (logged_at, run_type, granger_refreshed, granger_summary_json,
                 route_coefficients_json, route_shift_pct_json,
                 sector_ic_json, cascade_status, cascade_accuracy_json,
                 cascade_n_events, alerts_json, n_alerts)
            VALUES (:logged_at, :run_type, :granger_refreshed, :granger_summary_json,
                    :route_coefficients_json, :route_shift_pct_json,
                    :sector_ic_json, :cascade_status, :cascade_accuracy_json,
                    :cascade_n_events, :alerts_json, :n_alerts)
        """)
        with get_engine().connect() as conn:
            conn.execute(sql, row)
            conn.commit()
        log.info(
            "Causal monitoring logged: %d alert(s), cascade=%s, granger_refreshed=%s.",
            len(all_alerts), cascade.get("status"), granger.get("refreshed"),
        )
    except Exception as exc:
        log.warning("Could not write causal monitoring log: %s", exc)


def run_causal_monitoring(
    harness,
    prices:            pd.DataFrame,
    macro:             pd.DataFrame,
    cfg:               "RetrainConfig",
    logged_at:         str,
    sector_ic_current: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Orchestrate all causal-architecture health checks after a successful retrain.

      (a) Granger causality refresh — weekly (staleness-gated like macro routes)
      (b) Macro route coefficient shift detection — flags >20% shift
      (c) Sector IC monitoring — flags below-floor and >10pp drops
      (d) Cascade validation — directional accuracy, lag confirmation

    Returns a summary dict: alerts, n_alerts, granger_refreshed, cascade_status.
    Every sub-step is non-fatal; failures are logged and skipped.
    """
    all_alerts:      List[str] = []
    granger_result  = {"refreshed": False, "summary": {}, "error": ""}
    routes_result   = {"coefficients": {}, "shift_pcts": {}, "alerts": []}
    cascade_result  = {"status": "insufficient_data", "accuracy": {}, "n_events": 0, "alerts": []}
    sector_ic       = sector_ic_current or {}

    # (a) Granger causality refresh (weekly)
    try:
        granger_result = _run_granger_refresh(
            prices, force=getattr(cfg, "force_granger", False)
        )
        if granger_result.get("error"):
            log.warning("Granger refresh partial error: %s", granger_result["error"])
    except Exception as exc:
        log.warning("Granger causality step skipped: %s", exc)

    # (b) Route coefficient shift detection (daily — uses freshly-built pkl)
    try:
        routes_result = _detect_route_shifts()
        all_alerts.extend(routes_result.get("alerts", []))
        if routes_result["alerts"]:
            log.warning(
                "%d route-shift alert(s): %s",
                len(routes_result["alerts"]),
                routes_result["alerts"][0],
            )
    except Exception as exc:
        log.warning("Route shift detection skipped: %s", exc)

    # (c) Sector IC monitoring
    try:
        prev_ic    = _load_last_monitoring_field("sector_ic_json")
        ic_alerts  = _check_sector_ic_alerts(sector_ic, prev_ic)
        all_alerts.extend(ic_alerts)
    except Exception as exc:
        log.warning("Sector IC monitoring skipped: %s", exc)

    # (d) Cascade validation
    try:
        cascade_result = _run_cascade_validation_step(
            harness, prices, macro,
            dry_run = cfg.dry_run,
        )
        all_alerts.extend(cascade_result.get("alerts", []))
    except Exception as exc:
        log.warning("Cascade validation skipped: %s", exc)

    # Log all alerts
    if all_alerts:
        log.warning(
            "[CAUSAL MONITORING] %d alert(s) raised this run.", len(all_alerts)
        )
        for alert in all_alerts:
            log.warning("[CAUSAL ALERT] %s", alert)

    # Persist to DB
    if not cfg.dry_run:
        run_type = "weekly" if granger_result.get("refreshed") else "daily"
        _persist_causal_monitoring(
            logged_at, run_type,
            granger_result, routes_result, sector_ic, cascade_result,
            all_alerts,
        )

    return {
        "alerts":            all_alerts,
        "n_alerts":          len(all_alerts),
        "granger_refreshed": granger_result.get("refreshed", False),
        "cascade_status":    cascade_result.get("status", ""),
        "route_shifts":      len(routes_result.get("alerts", [])),
    }


def recent_causal_monitoring_runs(n: int = 10) -> pd.DataFrame:
    """
    Return the last `n` causal monitoring rows as a DataFrame.
    Useful for the dashboard Health tab.
    """
    try:
        sql = text(f"""
            SELECT logged_at, run_type, granger_refreshed, cascade_status,
                   n_alerts, alerts_json, cascade_n_events,
                   cascade_accuracy_json, sector_ic_json
            FROM   causal_monitoring_log
            ORDER  BY logged_at DESC
            LIMIT  {int(n)}
        """)
        with get_engine().connect() as conn:
            result = conn.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df
    except Exception:
        return pd.DataFrame()


# ── Core retraining function ───────────────────────────────────────────────────

def run_daily_retrain(
    config: Optional[RetrainConfig] = None,
    prices_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
) -> RetrainSummary:
    """
    Run one full retraining cycle.

    Parameters
    ----------
    config     : RetrainConfig (defaults used if None).
    prices_df  : pre-loaded price DataFrame (skip DB fetch if provided).
    macro_df   : pre-loaded macro DataFrame (skip feature build if provided).

    Returns
    -------
    RetrainSummary
        Always returned — check `.success` and `.error` fields.
    """
    cfg = config or RetrainConfig()
    summary = RetrainSummary(
        retrained_at=datetime.now(timezone.utc).isoformat(),
        config=cfg,
    )

    try:
        # ── 1. Load data ───────────────────────────────────────────────────────
        log.info("Loading prices (period=%s)…", cfg.prices_period)
        prices = prices_df if prices_df is not None else _load_prices(
            cfg.prices_period, cfg.commodities
        )
        if prices.empty:
            raise RuntimeError("Price matrix is empty — cannot retrain.")

        log.info("Loading macro overlays (period=%s)…", cfg.macro_period)
        macro = macro_df if macro_df is not None else _load_macro(
            cfg.macro_period, prices.index
        )

        log.info("Loading climate features (MEI + PDSI)…")
        climate = _load_climate(price_index=prices.index)

        # ── 1b. Store daily correlation snapshot ──────────────────────────────
        # Reads all 41 commodities from aligned_prices in the DB (not just the
        # 11 CORE_TICKERS used for the backtest) so the correlation matrix is
        # as wide as possible for the consistency checker and prior propagation.
        if not cfg.dry_run:
            try:
                from models.cross_asset import (
                    store_correlation_snapshot,
                    store_covariance_snapshot,
                )
                n_corr = store_correlation_snapshot()
                summary.n_corr_pairs = n_corr
                log.info("Stored %d correlation pairs in correlation_snapshots.", n_corr)
                n_cov = store_covariance_snapshot()
                log.info("Stored %d covariance entries in covariance_snapshots.", n_cov)
            except Exception as exc:
                log.warning("Correlation/covariance snapshot skipped: %s", exc)

        # ── 2. Determine commodity list ────────────────────────────────────────
        commodities = cfg.commodities or [
            c for c in CORE_TICKERS if c in prices.columns
        ]
        commodities = [c for c in commodities if c in prices.columns]
        if not commodities:
            raise RuntimeError("No valid commodities in price matrix.")

        log.info("Running backtest on %d commodities…", len(commodities))

        # ── 3. Run backtest harness ────────────────────────────────────────────
        from models.backtest_harness import BacktestHarness, DEFAULT_ADAPTERS
        harness = BacktestHarness(
            adapters=list(DEFAULT_ADAPTERS),
            climate_df=climate if not climate.empty else None,
        )
        records = harness.run(prices, macro, commodities)

        if not records:
            raise RuntimeError(
                "Backtest returned zero records. "
                "Prices may be too short or macro overlap insufficient."
            )

        # ── 4. Compute tier distribution ───────────────────────────────────────
        tier_dist: Dict[str, int] = {}
        for r in records:
            tier_dist[r.winning_tier] = tier_dist.get(r.winning_tier, 0) + 1

        # Count distinct commodities that contributed
        n_commodities = len({r.commodity for r in records})

        summary.n_commodities    = n_commodities
        summary.n_training_pairs = len(records)
        summary.tier_distribution = tier_dist
        log.info(
            "Backtest done: %d records from %d commodities. Tier dist: %s",
            len(records), n_commodities, tier_dist,
        )

        # ── 4b. Compute and log IC scores (per-commodity + sector-level) ─────────
        _sector_ic_for_monitoring: Dict[str, float] = {}   # captured for step 10
        if not cfg.dry_run:
            try:
                from models.ic_tracker import (
                    compute_ic_from_records,
                    compute_sector_ic_from_records,
                    log_ic_scores,
                )
                ic_results = compute_ic_from_records(
                    records, computed_at=summary.retrained_at
                )
                sector_ic = compute_sector_ic_from_records(
                    records, computed_at=summary.retrained_at
                )
                # Capture sector IC values for causal monitoring (step 10)
                _sector_ic_for_monitoring = {
                    k: float(v.ic_value)
                    for k, v in sector_ic.items()
                    if hasattr(v, "ic_value")
                }
                # Merge both dicts; sector rows use sector name as commodity field
                combined = {**ic_results, **sector_ic}
                n_ic = log_ic_scores(combined)
                log.info(
                    "Logged %d IC score(s) to ic_log "
                    "(%d commodity-level, %d sector-level).",
                    n_ic, len(ic_results), len(sector_ic),
                )
            except Exception as exc:
                log.warning("IC logging skipped: %s", exc)

        # ── 4c. Log forecasts, realize yesterday's actuals, consistency check ──
        if not cfg.dry_run and records:
            try:
                from models.cross_asset import (
                    log_forecasts,
                    realize_forecasts,
                    check_forecast_consistency,
                    load_correlation_matrix,
                )

                # Extract the most recent date's forecasts from the backtest
                latest_date = max(r.date for r in records)
                latest_records = [r for r in records if r.date == latest_date]

                def _regime(mf) -> str:
                    if getattr(mf, "vix_crisis", 0):
                        return "crisis"
                    if getattr(mf, "vix_risk_off", 0):
                        return "high_vol"
                    return "bull"

                forecasts_to_log: Dict[str, Dict] = {}
                for rec in latest_records:
                    regime_label = _regime(rec.meta_features)
                    for tier, fc_return in rec.tier_forecasts.items():
                        key = f"{rec.commodity}|{tier}"
                        forecasts_to_log[key] = {
                            "model_name":      tier,
                            "tier":            tier,
                            "forecast_return": fc_return,
                            "regime":          regime_label,
                            "confidence":      None,
                        }
                        # Use commodity as the unique key for log_forecasts;
                        # store one row per (commodity, tier) by encoding tier in model_name
                        forecasts_to_log[key]["_commodity"] = rec.commodity

                # Rekey by commodity+tier for log_forecasts format
                fc_dict = {
                    f"{v['_commodity']}|{v['tier']}": {
                        k: vv for k, vv in v.items() if k != "_commodity"
                    }
                    for v in forecasts_to_log.values()
                }
                n_fc = log_forecasts(
                    {k: v for k, v in fc_dict.items()},
                    forecast_date=latest_date.date(),
                )
                summary.n_forecasts_logged = n_fc
                log.info("Logged %d forecast rows for %s", n_fc, latest_date.date())

                # Back-fill actual returns for the day before latest_date
                log_returns = np.log(prices / prices.shift(1)).dropna()
                if latest_date in log_returns.index:
                    actuals = {
                        col: float(log_returns.loc[latest_date, col])
                        for col in log_returns.columns
                        if not np.isnan(log_returns.loc[latest_date, col])
                    }
                    realize_forecasts(
                        actuals,
                        forecast_date=latest_date.date(),
                    )

                # Consistency check on the latest ensemble forecast
                ensemble_fc = {
                    rec.commodity: float(np.mean(list(rec.tier_forecasts.values())))
                    for rec in latest_records
                    if rec.tier_forecasts
                }
                corr_mat = load_correlation_matrix()
                flags = check_forecast_consistency(
                    ensemble_fc, corr_matrix=corr_mat
                )
                summary.consistency_flags = [f.message for f in flags]
                if flags:
                    log.warning(
                        "%d cross-commodity consistency flag(s): %s",
                        len(flags),
                        "; ".join(f.message for f in flags[:3]),
                    )

            except Exception as exc:
                log.warning("Forecast logging / consistency check skipped: %s", exc)

        # ── 5. Guard: too few pairs → don't overwrite existing pkl ────────────
        if len(records) < MIN_PAIRS:
            raise RuntimeError(
                f"Only {len(records)} training pairs generated (minimum {MIN_PAIRS}). "
                "Refusing to overwrite existing MetaPredictor."
            )

        # ── 6. Fit MetaPredictor ───────────────────────────────────────────────
        from models.meta_predictor import MetaPredictor
        pairs = [r.to_training_pair() for r in records]
        meta = MetaPredictor()
        meta.fit(
            pairs,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
        )

        # Capture tree stats
        if meta.is_trained and meta._tree is not None:
            summary.tree_n_leaves = int(meta._tree.get_n_leaves())
        summary.top_feature = meta._top_feature()
        log.info(
            "MetaPredictor fitted: %d leaves, top feature=%r",
            summary.tree_n_leaves, summary.top_feature,
        )

        # ── 7. Save ───────────────────────────────────────────────────────────
        if not cfg.dry_run:
            save_path = Path(cfg.save_path) if cfg.save_path else DEFAULT_PKL
            saved = meta.save(save_path)
            summary.save_path = str(saved)
            log.info("MetaPredictor saved to %s", saved)
        else:
            log.info("Dry run — MetaPredictor NOT saved.")
            summary.save_path = ""

        summary.success = True

    except Exception as exc:
        summary.error = str(exc)
        log.error("Retraining failed: %s", exc, exc_info=True)

    # ── 8. Weekly macro route refresh (non-fatal, staleness-gated) ────────────
    # Runs at most once every 7 days.  summary.success=True guarantees that
    # `prices` and `macro` were assigned in the try block above, so we can
    # pass them directly to avoid redundant network calls.
    if not cfg.dry_run and summary.success:
        try:
            from models.macro_router import _is_stale, build_macro_routes
            if _is_stale(max_age_days=7):
                log.info("macro_routes.pkl is stale (>7 days) — rebuilding…")
                build_macro_routes(
                    prices   = prices,   # already loaded in step 1
                    macro_df = macro,    # already loaded in step 1
                    save     = True,
                )
                log.info("macro_routes.pkl refreshed.")
            else:
                log.info("macro_routes.pkl is fresh — skipping macro route rebuild.")
        except Exception as exc:
            log.warning("Macro route refresh skipped (non-fatal): %s", exc)

    # ── 9. Cascade sector forecasts (non-fatal, post-macro) ───────────────────
    # Runs only after a successful retrain so prices/macro are guaranteed loaded.
    # Fits SectorSpecificModel for every available commodity, runs Energy →
    # Metals → Agriculture → Livestock → Digital in causal order, and stores
    # per-commodity forecasts with breakdowns in cascade_forecasts table.
    if not cfg.dry_run and summary.success:
        try:
            from models.cascade_orchestrator import run_cascade
            cascade_result = run_cascade(
                prices   = prices,
                macro_df = macro,
                dry_run  = False,
            )
            summary.n_cascade_forecasts = len(cascade_result.commodities)
            summary.cascade_regime      = cascade_result.regime
            if cascade_result.success:
                log.info(
                    "Cascade complete: %d forecasts, regime=%s, %d rows written.",
                    len(cascade_result.commodities),
                    cascade_result.regime,
                    cascade_result.n_written,
                )
            else:
                errs = cascade_result.errors
                log.warning("Cascade ran with errors: %s", list(errs.values())[:2])
        except Exception as exc:
            log.warning("Cascade orchestrator skipped (non-fatal): %s", exc)

    # ── 10. Causal monitoring (Granger + route shifts + cascade validation) ─────
    # Runs after cascade (prices/macro/harness guaranteed available when success=True).
    # Non-fatal; failures are caught and logged at WARNING level.
    if not cfg.dry_run and summary.success and not cfg.skip_causal_monitoring:
        try:
            monitoring = run_causal_monitoring(
                harness            = harness,
                prices             = prices,
                macro              = macro,
                cfg                = cfg,
                logged_at          = summary.retrained_at,
                sector_ic_current  = _sector_ic_for_monitoring,
            )
            summary.granger_refresh_done      = monitoring.get("granger_refreshed", False)
            summary.cascade_validation_status = monitoring.get("cascade_status", "")
            summary.n_monitoring_alerts       = monitoring.get("n_alerts", 0)
            summary.route_shift_alerts        = monitoring.get("alerts", [])
            log.info(
                "Causal monitoring complete: %d alert(s), cascade=%s.",
                summary.n_monitoring_alerts, summary.cascade_validation_status,
            )
        except Exception as exc:
            log.warning("Causal monitoring skipped (non-fatal): %s", exc)

    # ── 11. Persist audit log (always, even on failure) ────────────────────────
    if not cfg.dry_run:
        _persist_training_log(summary)

    return summary


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point for `python -m models.daily_retrain`.

    Flags
    ─────
    --dry-run   : train but do not save pkl or write log
    --depth N   : set max_depth for the decision tree (default 5)
    --leaf N    : set min_samples_leaf (default 10)
    --period P  : price history period, e.g. "3y" (default "3y")
    --verbose   : show DEBUG logging
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrain the MetaPredictor on today's data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run",      action="store_true",
                        help="Train but do not save pkl or write log.")
    parser.add_argument("--depth",        type=int, default=5,
                        help="Decision tree max_depth (default 5).")
    parser.add_argument("--leaf",         type=int, default=10,
                        help="min_samples_leaf (default 10).")
    parser.add_argument("--period",       type=str, default="3y",
                        help="Price history period (default '3y').")
    parser.add_argument("--verbose",      action="store_true",
                        help="Show DEBUG logging.")
    parser.add_argument("--skip-causal",  action="store_true",
                        help="Skip causal monitoring (Granger + route shift + cascade validation).")
    parser.add_argument("--force-granger", action="store_true",
                        help="Force Granger causality refresh even if not stale.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    cfg = RetrainConfig(
        prices_period          = args.period,
        max_depth              = args.depth,
        min_samples_leaf       = args.leaf,
        dry_run                = args.dry_run,
        skip_causal_monitoring = args.skip_causal,
        force_granger          = args.force_granger,
    )

    summary = run_daily_retrain(config=cfg)
    print(summary.pretty())

    # Exit 1 only on hard failure (for shell error-checking)
    if not summary.success and summary.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
