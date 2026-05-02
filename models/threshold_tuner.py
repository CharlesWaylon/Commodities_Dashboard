"""
Trigger Threshold Auto-Tuner — Phase 5 Part 3.

Purpose
───────
Each trigger detector emits events with a strength value in [0, 1].
Right now every event fires regardless of its strength.  This module asks:

    "At what minimum strength does a trigger actually predict
     N-day forward price moves?"

It finds the threshold per family that maximises the Spearman IC between
the binary "above-threshold" signal and the subsequent commodity return.

How it works
────────────
1. Simulate a detection history by replaying every detector over the
   last `lookback_days` of macro data — one row at a time.  This gives
   a sequence of (date, family, strength) events without relying on the
   (possibly sparse) trigger log.

2. For each detection date look up the actual N-day forward return of the
   family's primary affected commodity.

3. Grid-search thresholds [0.1, 0.2, ..., 0.9].  For each threshold T:
       signal  = 1 where strength ≥ T, else 0
       IC(T)   = Spearman(signal, forward_return)
   Require ≥ `min_events_above` events above the threshold.

4. Optimal threshold = argmax IC(T).  Saved to the `threshold_config`
   SQLite table alongside the full IC grid so you can see the curve.

5. `load_optimal_thresholds()` returns a {family → float} dict that
   the dashboard (and future versions of `detect_all`) can use to filter
   low-confidence signals.

Design notes
────────────
• The simulation can fire the OPEC detector every day inside the OPEC
  window — that's intentional.  Each row is an independent (strength,
  return) pair for IC estimation.
• A family with continuous_ic < 0 means the detector itself is not
  predictive on this data.  The tuner flags it but still picks the
  least-bad threshold rather than crashing.
• All errors are isolated per family; one broken detector never aborts
  the whole run.
• `dry_run=True` computes and returns results without writing to SQLite.

Extension
─────────
When `weather_shock` and `energy_transition` detectors are wired up,
add them to DETECTORS in features/trigger_detectors.py — the tuner
picks them up automatically via that registry.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from models.ic_tracker import compute_ic   # reuse rank-IC implementation

log = logging.getLogger(__name__)

_ROOT      = Path(__file__).resolve().parent.parent
DEFAULT_DB = _ROOT / "data" / "commodities.db"

# Default grid: 9 evenly spaced strength thresholds
DEFAULT_THRESHOLD_GRID: List[float] = [round(t, 2) for t in np.arange(0.1, 1.0, 0.1)]

# Fallback threshold when not enough data or IC is negative everywhere
FALLBACK_THRESHOLD = 0.50

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS threshold_config (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    family                  TEXT NOT NULL,
    optimal_threshold       REAL NOT NULL,
    best_ic                 REAL,
    continuous_ic           REAL,
    n_events_total          INTEGER,
    n_events_at_threshold   INTEGER,
    lookback_days           INTEGER,
    forward_days            INTEGER,
    grid_results            TEXT,
    evaluated_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threshold_config_family
    ON threshold_config(family, evaluated_at);
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class TunerConfig:
    """
    Configuration for a threshold-tuning run.

    Attributes
    ──────────
    families          : trigger family names to tune. None → all registered.
    lookback_days     : days of macro history to replay for simulation.
    forward_days      : N-day forward return horizon for IC calculation.
    threshold_grid    : list of strength values to evaluate.
    min_events_above  : minimum events above a threshold to count it valid.
    save_to_db        : whether to persist results to SQLite.
    db_path           : override default DB path.
    dry_run           : compute but do NOT save.
    """
    families:          Optional[List[str]]  = None
    lookback_days:     int                  = 180
    forward_days:      int                  = 5
    threshold_grid:    List[float]          = field(default_factory=lambda: list(DEFAULT_THRESHOLD_GRID))
    min_events_above:  int                  = 8
    save_to_db:        bool                 = True
    db_path:           Optional[Path]       = None
    dry_run:           bool                 = False


@dataclass
class TuneResult:
    """
    Outcome for one trigger family.

    Attributes
    ──────────
    family                : trigger family name (e.g. "opec_action")
    optimal_threshold     : the threshold that maximised IC
    best_ic               : Spearman IC at optimal threshold
    continuous_ic         : IC using raw strength values (no binary threshold)
    n_events_total        : total simulated detections in the lookback window
    n_events_at_threshold : events where strength ≥ optimal_threshold
    grid_results          : {threshold: ic} for the full grid search
    evaluated_at          : UTC ISO timestamp
    lookback_days         : days of history used
    forward_days          : forward return horizon used
    insufficient_data     : True if not enough events to be reliable
    """
    family:                str
    optimal_threshold:     float
    best_ic:               float
    continuous_ic:         float
    n_events_total:        int
    n_events_at_threshold: int
    grid_results:          Dict[float, float] = field(default_factory=dict)
    evaluated_at:          str = ""
    lookback_days:         int = 180
    forward_days:          int = 5
    insufficient_data:     bool = False

    def __post_init__(self):
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def signal_label(self) -> str:
        """Human-readable label for the IC quality at optimal threshold."""
        if self.insufficient_data:
            return "⚪ Insufficient data"
        if self.best_ic >= 0.05:
            return "🟢 Actionable"
        if self.best_ic >= 0.0:
            return "🟡 Marginal"
        return "🔴 Negative"

    def summary_line(self) -> str:
        return (
            f"{self.family}: threshold={self.optimal_threshold:.2f}  "
            f"IC={self.best_ic:+.4f}  ({self.signal_label()})  "
            f"n={self.n_events_at_threshold}/{self.n_events_total}"
        )


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _connect(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    p = Path(db_path) if db_path else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(_SCHEMA_SQL)
    return conn


def save_tune_results(
    results: Dict[str, TuneResult],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Persist tuning results to threshold_config.  Returns rows written."""
    if not results:
        return 0
    conn = _connect(db_path)
    rows = []
    for r in results.values():
        rows.append((
            r.family,
            r.optimal_threshold,
            r.best_ic,
            r.continuous_ic,
            r.n_events_total,
            r.n_events_at_threshold,
            r.lookback_days,
            r.forward_days,
            json.dumps({str(k): v for k, v in r.grid_results.items()}),
            r.evaluated_at,
        ))
    conn.executemany(
        """
        INSERT INTO threshold_config
            (family, optimal_threshold, best_ic, continuous_ic,
             n_events_total, n_events_at_threshold,
             lookback_days, forward_days, grid_results, evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    log.info("Saved %d threshold result(s) to DB.", len(rows))
    return len(rows)


def load_optimal_thresholds(
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Return the most recent optimal threshold per trigger family.

    Returns
    -------
    dict mapping family → optimal_threshold.
    Falls back to FALLBACK_THRESHOLD for any family not in the table.
    Empty dict if the table has no rows yet.
    """
    try:
        conn = _connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT   family, optimal_threshold
            FROM     threshold_config
            WHERE    (family, evaluated_at) IN (
                         SELECT family, MAX(evaluated_at)
                         FROM   threshold_config
                         GROUP  BY family
                     )
            """,
            conn,
        )
        conn.close()
        return dict(zip(df["family"], df["optimal_threshold"]))
    except Exception as exc:
        log.warning("load_optimal_thresholds failed: %s", exc)
        return {}


def recent_tune_history(
    n: int = 10,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Last n tuning runs per family — useful for the Health tab."""
    try:
        conn = _connect(db_path)
        df = pd.read_sql_query(
            f"""
            SELECT family, optimal_threshold, best_ic, continuous_ic,
                   n_events_total, n_events_at_threshold,
                   lookback_days, forward_days, evaluated_at
            FROM   threshold_config
            ORDER  BY evaluated_at DESC
            LIMIT  {int(n)}
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ── Simulation helpers ─────────────────────────────────────────────────────────

def simulate_detections(
    macro_df: pd.DataFrame,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    Replay all registered detectors over the last `lookback_days` rows
    of macro_df, one row at a time.

    Returns
    -------
    pd.DataFrame with columns: date, family, strength, affected_commodities
    One row per (date, family) event that fired.  Empty DataFrame if nothing fires.
    """
    from features.trigger_detectors import DETECTORS

    if macro_df is None or macro_df.empty:
        return pd.DataFrame(columns=["date", "family", "strength",
                                      "affected_commodities"])

    # Slice to the lookback window
    tail = macro_df.tail(lookback_days)
    if tail.empty:
        return pd.DataFrame(columns=["date", "family", "strength",
                                      "affected_commodities"])

    rows = []
    for i in range(len(tail)):
        # Slice up to and including row i (gives detector its rolling context)
        slice_df = macro_df.loc[: tail.index[i]]
        date = tail.index[i]

        for fn in DETECTORS:
            try:
                event = fn(slice_df)
                if event is not None:
                    rows.append({
                        "date":                date,
                        "family":              event.family,
                        "strength":            float(event.strength),
                        "affected_commodities": list(event.affected_commodities or []),
                    })
            except Exception as exc:
                log.debug("Detector %s failed on %s: %s", fn.__name__, date, exc)

    if not rows:
        return pd.DataFrame(columns=["date", "family", "strength",
                                      "affected_commodities"])
    return pd.DataFrame(rows)


def _primary_commodity(affected: list, prices: pd.DataFrame) -> Optional[str]:
    """Return the first affected commodity that's in prices.columns."""
    for name in affected:
        if name in prices.columns:
            return name
    # fallback: first column
    return prices.columns[0] if len(prices.columns) > 0 else None


def _forward_returns(
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    commodity: str,
    forward_days: int,
) -> pd.Series:
    """
    Compute mean log-return over [date+1, date+forward_days] for each date.

    Returns pd.Series indexed by the input dates.  NaN where insufficient
    future data exists.
    """
    series = prices[commodity]
    log_ret = np.log(series / series.shift(1))

    result = {}
    for d in dates:
        try:
            pos = series.index.searchsorted(d)
            end = min(pos + forward_days + 1, len(series))
            fwd_slice = log_ret.iloc[pos + 1: end]
            result[d] = float(fwd_slice.sum()) if len(fwd_slice) > 0 else float("nan")
        except Exception:
            result[d] = float("nan")

    return pd.Series(result)


# ── Core tuner ─────────────────────────────────────────────────────────────────

class ThresholdTuner:
    """
    Finds the optimal strength threshold for each trigger family.

    Usage
    ─────
        from models.threshold_tuner import ThresholdTuner, TunerConfig

        tuner = ThresholdTuner()
        results = tuner.tune_all(macro_df=macro_df, prices=prices)

        # Inspect results
        for family, r in results.items():
            print(r.summary_line())

        # Load saved thresholds for use in detect_all
        from models.threshold_tuner import load_optimal_thresholds
        thresholds = load_optimal_thresholds()
        # → {"opec_action": 0.6, "fed_tightening": 0.4, ...}
    """

    def __init__(self, config: Optional[TunerConfig] = None):
        self.config = config or TunerConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def tune_all(
        self,
        macro_df: pd.DataFrame,
        prices: pd.DataFrame,
        config: Optional[TunerConfig] = None,
    ) -> Dict[str, TuneResult]:
        """
        Simulate detections and tune thresholds for all configured families.

        Parameters
        ----------
        macro_df : pd.DataFrame  — macro overlay features
        prices   : pd.DataFrame  — price matrix (n_days × n_commodities)
        config   : TunerConfig, optional — overrides self.config

        Returns
        -------
        dict mapping family → TuneResult
        """
        cfg = config or self.config

        log.info(
            "Simulating detections over last %d days…", cfg.lookback_days
        )
        events_df = simulate_detections(macro_df, cfg.lookback_days)

        if events_df.empty:
            log.warning("No trigger events detected in lookback window.")
            return {}

        # Determine which families to tune
        detected_families = events_df["family"].unique().tolist()
        families = cfg.families or detected_families
        families = [f for f in families if f in detected_families]

        if not families:
            log.warning("None of the requested families had detections.")
            return {}

        results: Dict[str, TuneResult] = {}
        for family in families:
            try:
                fam_events = events_df[events_df["family"] == family]
                r = self._tune_family(fam_events, prices, cfg)
                results[family] = r
                log.info(r.summary_line())
            except Exception as exc:
                log.warning("Tuning failed for %r: %s", family, exc)

        # Persist unless dry_run
        if not cfg.dry_run and cfg.save_to_db:
            save_tune_results(results, db_path=cfg.db_path)

        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _tune_family(
        self,
        fam_events: pd.DataFrame,
        prices: pd.DataFrame,
        cfg: TunerConfig,
    ) -> TuneResult:
        """Tune one family given its simulated events and price matrix."""
        family = fam_events["family"].iloc[0]
        n_total = len(fam_events)

        # Determine primary commodity for this family
        aff_lists = fam_events["affected_commodities"].tolist()
        aff_flat  = [c for row in aff_lists for c in row]
        primary   = _primary_commodity(aff_flat, prices)

        if primary is None:
            log.warning("No matching commodity for family %r — skipping.", family)
            return TuneResult(
                family=family,
                optimal_threshold=FALLBACK_THRESHOLD,
                best_ic=float("nan"),
                continuous_ic=float("nan"),
                n_events_total=n_total,
                n_events_at_threshold=0,
                insufficient_data=True,
                lookback_days=cfg.lookback_days,
                forward_days=cfg.forward_days,
            )

        # Compute forward returns for every detection date
        dates  = pd.DatetimeIndex(fam_events["date"].values)
        fwd    = _forward_returns(dates, prices, primary, cfg.forward_days)
        strengths = fam_events.set_index("date")["strength"]

        # Align: only rows with a valid forward return
        valid_idx = fwd.dropna().index
        fwd_valid = fwd.loc[valid_idx]
        str_valid = strengths.reindex(valid_idx).fillna(0.0)

        if len(fwd_valid) < cfg.min_events_above:
            return TuneResult(
                family=family,
                optimal_threshold=FALLBACK_THRESHOLD,
                best_ic=float("nan"),
                continuous_ic=float("nan"),
                n_events_total=n_total,
                n_events_at_threshold=0,
                insufficient_data=True,
                lookback_days=cfg.lookback_days,
                forward_days=cfg.forward_days,
            )

        # Continuous IC (raw strength as signal)
        cont_ic = compute_ic(str_valid.values, fwd_valid.values)

        # Grid search
        grid_ic: Dict[float, float] = {}
        for t in cfg.threshold_grid:
            binary = (str_valid >= t).astype(float)
            n_above = int(binary.sum())
            if n_above < cfg.min_events_above:
                continue  # not enough events above this threshold
            ic_t = compute_ic(binary.values, fwd_valid.values)
            if not np.isnan(ic_t):
                grid_ic[t] = ic_t

        if not grid_ic:
            # No threshold met the min_events requirement
            return TuneResult(
                family=family,
                optimal_threshold=FALLBACK_THRESHOLD,
                best_ic=cont_ic if not np.isnan(cont_ic) else float("nan"),
                continuous_ic=cont_ic,
                n_events_total=n_total,
                n_events_at_threshold=0,
                grid_results={},
                insufficient_data=True,
                lookback_days=cfg.lookback_days,
                forward_days=cfg.forward_days,
            )

        # Optimal threshold = argmax IC
        optimal_t = max(grid_ic, key=lambda k: grid_ic[k])
        best_ic   = grid_ic[optimal_t]
        n_at_opt  = int((str_valid >= optimal_t).sum())

        return TuneResult(
            family=family,
            optimal_threshold=optimal_t,
            best_ic=best_ic,
            continuous_ic=cont_ic if not np.isnan(cont_ic) else float("nan"),
            n_events_total=n_total,
            n_events_at_threshold=n_at_opt,
            grid_results=grid_ic,
            lookback_days=cfg.lookback_days,
            forward_days=cfg.forward_days,
        )


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point for `python -m models.threshold_tuner`.

    Flags
    ─────
    --lookback N   : days of macro history to replay (default 180)
    --forward N    : forward return horizon in trading days (default 5)
    --min-events N : minimum events above threshold (default 8)
    --dry-run      : compute but do not save to DB
    --verbose      : show DEBUG logging
    """
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Auto-tune trigger strength thresholds."
    )
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--forward",  type=int, default=5)
    parser.add_argument("--min-events", type=int, default=8, dest="min_events")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Load data
    log.info("Loading prices and macro overlays…")
    try:
        from services.data_contract import fetch_price_matrix
        prices = fetch_price_matrix(days=args.lookback + 30)
    except Exception:
        import yfinance as yf
        from models.daily_retrain import CORE_TICKERS
        raw = yf.download(list(CORE_TICKERS.values()), period="2y",
                          interval="1d", progress=False, auto_adjust=True)
        prices = raw["Close"].rename(columns={v: k for k, v in CORE_TICKERS.items()})
        prices = prices.ffill().dropna()

    try:
        from features.macro_overlays import build_macro_overlay_features
        macro = build_macro_overlay_features(period="2y", index=prices.index)
    except Exception:
        log.error("Could not load macro overlays.")
        sys.exit(1)

    cfg = TunerConfig(
        lookback_days=args.lookback,
        forward_days=args.forward,
        min_events_above=args.min_events,
        dry_run=args.dry_run,
    )

    tuner = ThresholdTuner(config=cfg)
    results = tuner.tune_all(macro_df=macro, prices=prices)

    print("\n" + "=" * 60)
    print("  Trigger Threshold Tuning Results")
    print("=" * 60)
    if not results:
        print("  No results — not enough events in lookback window.")
    for r in results.values():
        print(f"  {r.summary_line()}")
        if r.grid_results:
            best = max(r.grid_results.items(), key=lambda x: x[1])
            print(f"    IC curve: " + "  ".join(
                f"t={t:.1f}→{ic:+.3f}" for t, ic in sorted(r.grid_results.items())
            ))
    print("=" * 60)
    if args.dry_run:
        print("  (dry-run — results NOT saved to DB)")
    else:
        print("  Results saved to threshold_config table.")
    print()


if __name__ == "__main__":
    main()
