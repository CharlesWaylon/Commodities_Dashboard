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
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
DEFAULT_DB   = _ROOT / "data" / "commodities.db"
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

# SQLite schema for the training audit log
_TRAINING_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_training_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    retrained_at        TEXT NOT NULL,
    n_commodities       INTEGER NOT NULL,
    n_training_pairs    INTEGER NOT NULL,
    tier_distribution   TEXT,
    tree_n_leaves       INTEGER,
    top_feature         TEXT,
    save_path           TEXT,
    error               TEXT,
    config_json         TEXT,
    inserted_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_log_retrained_at
    ON model_training_log(retrained_at);
"""


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
    db_path         : SQLite path for the audit log. None → DEFAULT_DB.
    dry_run         : if True, trains but does NOT save the pkl or write the log.
    """
    commodities:      Optional[List[str]] = None
    prices_period:    str                 = "3y"
    macro_period:     str                 = "2y"
    max_depth:        int                 = 5
    min_samples_leaf: int                 = 10
    save_path:        Optional[Path]      = None
    db_path:          Optional[Path]      = None
    dry_run:          bool                = False


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrainSummary:
    """
    Outcome of one retraining run.

    Attributes
    ──────────
    retrained_at       : UTC ISO timestamp when the run started.
    n_commodities      : number of commodities that contributed training records.
    n_training_pairs   : total (MetaFeatures, winning_tier) pairs generated.
    tier_distribution  : {tier: count} across all training pairs.
    tree_n_leaves      : number of leaves in the fitted decision tree (-1 if untrained).
    top_feature        : most important feature by Gini importance (or "" if untrained).
    save_path          : where the pkl was written (empty string if dry_run or failed).
    error              : non-empty if the run aborted with an exception.
    config             : the RetrainConfig used (for audit).
    success            : True if predictor was trained and saved (or dry_run succeeded).
    """
    retrained_at:     str               = ""
    n_commodities:    int               = 0
    n_training_pairs: int               = 0
    tier_distribution: Dict[str, int]   = field(default_factory=dict)
    tree_n_leaves:    int               = -1
    top_feature:      str               = ""
    save_path:        str               = ""
    error:            str               = ""
    config:           Optional[RetrainConfig] = None
    success:          bool              = False

    def pretty(self) -> str:
        """Multi-line human-readable summary for CLI output."""
        status = "✅ SUCCESS" if self.success else f"❌ FAILED — {self.error}"
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


# ── SQLite audit log ───────────────────────────────────────────────────────────

def _connect_log(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    p = Path(db_path) if db_path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(_TRAINING_LOG_SCHEMA)
    return conn


def _persist_training_log(
    summary: RetrainSummary,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write one row to model_training_log. Silent on failure."""
    try:
        conn = _connect_log(db_path)
        conn.execute(
            """
            INSERT INTO model_training_log
                (retrained_at, n_commodities, n_training_pairs,
                 tier_distribution, tree_n_leaves, top_feature,
                 save_path, error, config_json, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.retrained_at,
                summary.n_commodities,
                summary.n_training_pairs,
                json.dumps(summary.tier_distribution),
                summary.tree_n_leaves,
                summary.top_feature,
                summary.save_path,
                summary.error,
                json.dumps(
                    asdict(summary.config),
                    default=lambda v: str(v),   # Path → str, any other non-serializable
                ) if summary.config else "{}",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        log.info("Training log persisted to %s", db_path or DEFAULT_DB)
    except Exception as exc:
        log.warning("Could not write training log: %s", exc)


def recent_training_runs(
    n: int = 10,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Return the last `n` training log entries as a DataFrame.
    Useful for the dashboard Model Health tab (Phase 5 Part 2).
    Returns empty DataFrame if the log table doesn't exist yet.
    """
    try:
        conn = _connect_log(db_path)
        df = pd.read_sql_query(
            f"""
            SELECT retrained_at, n_commodities, n_training_pairs,
                   tier_distribution, tree_n_leaves, top_feature,
                   save_path, error
            FROM   model_training_log
            ORDER  BY retrained_at DESC
            LIMIT  {int(n)}
            """,
            conn,
        )
        conn.close()
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
        harness = BacktestHarness(adapters=list(DEFAULT_ADAPTERS))
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
                # Merge both dicts; sector rows use sector name as commodity field
                combined = {**ic_results, **sector_ic}
                n_ic = log_ic_scores(combined, db_path=cfg.db_path)
                log.info(
                    "Logged %d IC score(s) to ic_log "
                    "(%d commodity-level, %d sector-level).",
                    n_ic, len(ic_results), len(sector_ic),
                )
            except Exception as exc:
                log.warning("IC logging skipped: %s", exc)

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

    # ── 8. Persist audit log (always, even on failure) ─────────────────────────
    if not cfg.dry_run:
        _persist_training_log(summary, cfg.db_path)

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
    parser.add_argument("--dry-run",  action="store_true",
                        help="Train but do not save pkl or write log.")
    parser.add_argument("--depth",    type=int, default=5,
                        help="Decision tree max_depth (default 5).")
    parser.add_argument("--leaf",     type=int, default=10,
                        help="min_samples_leaf (default 10).")
    parser.add_argument("--period",   type=str, default="3y",
                        help="Price history period (default '3y').")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    cfg = RetrainConfig(
        prices_period=args.period,
        max_depth=args.depth,
        min_samples_leaf=args.leaf,
        dry_run=args.dry_run,
    )

    summary = run_daily_retrain(config=cfg)
    print(summary.pretty())

    # Exit 1 only on hard failure (for shell error-checking)
    if not summary.success and summary.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
