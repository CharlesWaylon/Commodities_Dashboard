"""
Cascade Forecaster — Cross-Sector Prediction Orchestrator
===========================================================
Coordinates SectorSpecificModel predictions across all commodity sectors
in causal dependency order:

    Energy → Metals → Agriculture → Livestock → Digital

Energy is modelled using macro context only (DXY / VIX / TLT).
Each subsequent sector receives the finalized forecasts of all upstream
sectors as upstream_shocks, in addition to the shared macro state.

    Energy    : macro only
    Metals    : macro + Energy forecasts
    Agriculture: macro + Energy forecasts
    Livestock : macro + Agriculture forecasts
    Digital   : macro only (BTC is macro-driven, not commodity-chain-driven)

Pipeline integration
--------------------
`run_cascade()` is the single public function.  It accepts pre-loaded
DataFrames (to avoid redundant DB / yfinance calls) and writes per-commodity
forecast rows to a `cascade_forecasts` SQLite table.  Call it as a non-fatal
step from daily_retrain.py after the macro route refresh.

CLI
---
    python -m models.cascade_orchestrator [--dry-run] [--verbose]
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from database.db import get_engine

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# ── Cascade topology ───────────────────────────────────────────────────────────
# Order in which sectors are predicted; earlier sectors feed later ones.
SECTOR_ORDER: List[str] = [
    "energy",
    "metals",
    "agriculture",
    "livestock",
    "digital",
]

# Maps each sector to the set of upstream sectors whose forecasts it receives.
# Order within each list must respect SECTOR_ORDER (no backward edges).
UPSTREAM_MAP: Dict[str, List[str]] = {
    "energy":       [],                                    # macro only
    "metals":       ["energy"],                            # energy drives mining & smelting costs
    "agriculture":  ["energy", "metals"],                  # energy = fertilizer+fuel; metals = equipment & irrigation
    "livestock":    ["agriculture", "energy"],             # feed-grain prices primary; energy for facilities & transport
    "digital":      ["energy", "metals", "agriculture"],   # electricity cost; hardware metals; food-CPI inflation signal
}

# ── Macro variable column names in macro_df ───────────────────────────────────
_MACRO_RETURN_COLS = ["dxy_ret", "vix_ret5d", "tlt_ret", "tlt_yield_proxy"]




# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class CommodityForecast:
    """Final per-commodity output from the cascade run."""
    commodity:           str
    sector:              str
    sector_rank:         int         # 1=energy, 2=metals, …
    regime:              str
    base_forecast:       float
    macro_adjustment:    float
    upstream_adjustment: float
    final_forecast:      float
    confidence:          float
    macro_detail:        Dict[str, float]
    upstream_detail:     Dict[str, float]


@dataclass
class CascadeResult:
    """Full output of one cascade run."""
    run_at:          str                                    = ""
    forecast_date:   Optional[date]                        = None
    regime:          str                                    = "neutral"
    macro_snapshot:  Dict[str, float]                      = field(default_factory=dict)
    commodities:     Dict[str, CommodityForecast]          = field(default_factory=dict)
    n_written:       int                                    = 0
    errors:          Dict[str, str]                        = field(default_factory=dict)
    success:         bool                                   = False

    def pretty(self) -> str:
        lines = [
            "=" * 60,
            "  Cascade Forecaster Summary",
            "=" * 60,
            f"  Run at         : {self.run_at}",
            f"  Forecast date  : {self.forecast_date}",
            f"  Regime         : {self.regime}",
            f"  Commodities    : {len(self.commodities)}",
            f"  DB rows written: {self.n_written}",
        ]
        if self.errors:
            lines.append(f"  Errors         : {len(self.errors)}")
            for c, msg in list(self.errors.items())[:3]:
                lines.append(f"    [{c}] {msg}")
        for sector in SECTOR_ORDER:
            sector_rows = [
                v for v in self.commodities.values() if v.sector == sector
            ]
            if not sector_rows:
                continue
            mean_fc = np.mean([r.final_forecast for r in sector_rows])
            lines.append(
                f"  {sector.capitalize():<14}: "
                f"{len(sector_rows)} commodities  mean_fc={mean_fc:+.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Core orchestrator class ────────────────────────────────────────────────────

class CascadeForecaster:
    """
    Fits and coordinates SectorSpecificModel instances across all sectors,
    running predictions in causal order and propagating upstream forecasts.

    Parameters
    ----------
    min_price_rows : int
        Minimum non-null price rows required to include a commodity.
    """

    def __init__(self, min_price_rows: int = 252):
        self.min_price_rows = min_price_rows
        self._models:         Dict[str, object]          = {}   # commodity → SectorSpecificModel
        self._prices:         Optional[pd.DataFrame]     = None
        self._macro_snapshot: Dict[str, float]           = {}
        self._regime:         str                        = "neutral"
        self._sector_members: Dict[str, List[str]]       = {}   # sector → [commodity, …]

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_sector_members(self, prices: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Intersect COMMODITY_SECTORS with the price matrix columns, filtered
        to commodities that have enough price history.
        """
        from models.config import COMMODITY_SECTORS
        members: Dict[str, List[str]] = {s: [] for s in SECTOR_ORDER}
        for commodity, sector in COMMODITY_SECTORS.items():
            if sector not in members:
                continue
            if commodity not in prices.columns:
                continue
            n_valid = int(prices[commodity].notna().sum())
            if n_valid < self.min_price_rows:
                log.debug("Skipping %s — only %d price rows.", commodity, n_valid)
                continue
            members[sector].append(commodity)
        return members

    def _extract_macro_snapshot(self, macro_df: pd.DataFrame) -> Dict[str, float]:
        """
        Pull the most recent macro return values as a flat dict.
        Derives tlt_yield_proxy = −tlt_ret if not directly available.
        """
        if macro_df is None or macro_df.empty:
            return {}

        # Try to get the last non-null row for each column
        snap: Dict[str, float] = {}
        for col in _MACRO_RETURN_COLS:
            if col in macro_df.columns:
                series = macro_df[col].dropna()
                if not series.empty:
                    snap[col] = float(series.iloc[-1])

        # Derive proxy if missing
        if "tlt_yield_proxy" not in snap and "tlt_ret" in snap:
            snap["tlt_yield_proxy"] = -snap["tlt_ret"]

        return snap

    def _infer_regime(self, macro_df: pd.DataFrame) -> str:
        """Classify today's regime from the macro DataFrame."""
        try:
            from models.macro_router import get_current_regime
            return get_current_regime(macro_df)
        except Exception:
            return "neutral"

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        prices:   pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> "CascadeForecaster":
        """
        Fit one SectorSpecificModel per commodity, in sector order.

        Parameters
        ----------
        prices   : full price matrix from load_price_matrix_from_db().
        macro_df : macro overlay DataFrame (for regime detection).
        """
        from models.sector_model import SectorSpecificModel
        from models.features import build_target

        self._prices         = prices
        self._sector_members = self._build_sector_members(prices)
        self._macro_snapshot = self._extract_macro_snapshot(macro_df)
        self._regime         = self._infer_regime(macro_df) if macro_df is not None else "neutral"

        log.info(
            "Cascade fit: regime=%s  macro_snapshot=%s",
            self._regime,
            {k: f"{v:+.6f}" for k, v in self._macro_snapshot.items()},
        )

        for sector in SECTOR_ORDER:
            commodities = self._sector_members.get(sector, [])
            log.info("Fitting %d %s model(s)…", len(commodities), sector)
            for commodity in commodities:
                try:
                    model = SectorSpecificModel(sector=sector, commodity=commodity)
                    target = build_target(prices, commodity)
                    model.fit(prices, target)
                    self._models[commodity] = model
                    log.info(
                        "  ✓ %s  IC=%.4f  features=%d",
                        commodity, model.ic_score(), len(model.sector_feature_names()),
                    )
                except Exception as exc:
                    log.warning("  ✗ %s fit failed: %s", commodity, exc)

        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self) -> Dict[str, CommodityForecast]:
        """
        Run cascade predictions in sector order, propagating upstream forecasts.

        Returns
        -------
        dict mapping commodity name → CommodityForecast
        """
        if not self._models:
            raise RuntimeError("Call fit() before predict().")

        # Accumulated: sector → list of (commodity, final_forecast) tuples
        sector_outputs: Dict[str, List[Tuple[str, float]]] = {s: [] for s in SECTOR_ORDER}
        results:        Dict[str, CommodityForecast]        = {}

        for rank, sector in enumerate(SECTOR_ORDER, start=1):
            commodities = self._sector_members.get(sector, [])
            if not commodities:
                continue

            # Build upstream_shocks from all previously-run upstream sectors
            upstream_shocks: Dict[str, float] = {}
            for upstream_sector in UPSTREAM_MAP.get(sector, []):
                for commodity, final_fc in sector_outputs.get(upstream_sector, []):
                    upstream_shocks[commodity] = final_fc

            log.info(
                "Predicting %s sector (%d commodities, %d upstream shocks)…",
                sector, len(commodities), len(upstream_shocks),
            )

            for commodity in commodities:
                model = self._models.get(commodity)
                if model is None:
                    log.warning("No fitted model for %s — skipping.", commodity)
                    continue
                try:
                    result = model.predict_with_context(
                        self._prices,
                        macro_state=self._macro_snapshot,
                        upstream_shocks=upstream_shocks,
                        regime=self._regime,
                    )
                    bd = result["breakdown"]
                    cf = CommodityForecast(
                        commodity=commodity,
                        sector=sector,
                        sector_rank=rank,
                        regime=bd["regime"],
                        base_forecast=bd["base_forecast"],
                        macro_adjustment=bd["macro_adjustment"],
                        upstream_adjustment=bd["upstream_adjustment"],
                        final_forecast=bd["final_forecast"],
                        confidence=bd["final_confidence"],
                        macro_detail=bd["macro_detail"],
                        upstream_detail=bd["upstream_detail"],
                    )
                    results[commodity] = cf
                    sector_outputs[sector].append((commodity, cf.final_forecast))
                    log.info(
                        "  %s  base=%+.4f  macro=%+.4f  upstream=%+.4f  final=%+.4f  conf=%.3f",
                        commodity, cf.base_forecast, cf.macro_adjustment,
                        cf.upstream_adjustment, cf.final_forecast, cf.confidence,
                    )
                except Exception as exc:
                    log.warning("Predict failed for %s: %s", commodity, exc)

        return results


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _write_cascade_forecasts(
    forecasts:     Dict[str, CommodityForecast],
    forecast_date: date,
) -> int:
    """
    Upsert cascade forecast rows into cascade_forecasts.
    Returns the number of rows inserted/replaced.
    """
    now    = datetime.now(timezone.utc).isoformat()
    date_s = str(forecast_date)
    rows_data = []

    for cf in forecasts.values():
        rows_data.append({
            "forecast_date":       date_s,
            "commodity":           cf.commodity,
            "sector":              cf.sector,
            "sector_rank":         cf.sector_rank,
            "regime":              cf.regime,
            "base_forecast":       cf.base_forecast,
            "macro_adjustment":    cf.macro_adjustment,
            "upstream_adjustment": cf.upstream_adjustment,
            "final_forecast":      cf.final_forecast,
            "confidence":          cf.confidence,
            "macro_detail":        json.dumps(cf.macro_detail),
            "upstream_detail":     json.dumps(cf.upstream_detail),
            "inserted_at":         now,
        })

    if not rows_data:
        return 0

    sql = text("""
        INSERT INTO cascade_forecasts (
            forecast_date, commodity, sector, sector_rank, regime,
            base_forecast, macro_adjustment, upstream_adjustment,
            final_forecast, confidence, macro_detail, upstream_detail,
            inserted_at
        ) VALUES (
            :forecast_date, :commodity, :sector, :sector_rank, :regime,
            :base_forecast, :macro_adjustment, :upstream_adjustment,
            :final_forecast, :confidence, :macro_detail, :upstream_detail,
            :inserted_at
        )
        ON CONFLICT (forecast_date, commodity) DO UPDATE SET
            base_forecast       = EXCLUDED.base_forecast,
            macro_adjustment    = EXCLUDED.macro_adjustment,
            upstream_adjustment = EXCLUDED.upstream_adjustment,
            final_forecast      = EXCLUDED.final_forecast,
            confidence          = EXCLUDED.confidence,
            macro_detail        = EXCLUDED.macro_detail,
            upstream_detail     = EXCLUDED.upstream_detail,
            inserted_at         = EXCLUDED.inserted_at
    """)

    try:
        with get_engine().connect() as conn:
            conn.execute(sql, rows_data)
            conn.commit()
        return len(rows_data)
    except Exception as exc:
        log.warning("cascade_forecasts DB write failed: %s", exc)
        return 0


def load_cascade_forecasts(
    forecast_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load cascade forecasts from the DB.

    Parameters
    ----------
    forecast_date : specific date to load; defaults to the most recent run.

    Returns
    -------
    pd.DataFrame with columns matching cascade_forecasts schema.
    Empty DataFrame if no rows exist yet.
    """
    engine = get_engine()
    if forecast_date is not None:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM cascade_forecasts WHERE forecast_date = :fd ORDER BY sector_rank, commodity"),
                {"fd": str(forecast_date)},
            )
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    else:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM cascade_forecasts
                WHERE  forecast_date = (SELECT MAX(forecast_date) FROM cascade_forecasts)
                ORDER  BY sector_rank, commodity
            """))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
    for col in ("macro_detail", "upstream_detail"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.loads(v) if v else {})
    return df


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_prices() -> pd.DataFrame:
    """Load full price matrix from DB."""
    try:
        from models.data_loader import load_price_matrix_from_db
        df = load_price_matrix_from_db()
        log.info("Cascade prices: %d rows × %d cols", len(df), df.shape[1])
        return df
    except Exception as exc:
        log.warning("Cascade price load failed: %s", exc)
        return pd.DataFrame()


def _load_macro(price_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Load macro overlay features from yfinance."""
    try:
        from features.macro_overlays import macro_features
        df = macro_features(period="2y")
        if price_index is not None and not df.empty:
            df = df.reindex(price_index, method="ffill")
        log.info("Cascade macro: %d rows × %d cols", len(df), df.shape[1])
        return df
    except Exception as exc:
        log.warning("Cascade macro load failed: %s", exc)
        return pd.DataFrame()


# ── Public pipeline entry point ────────────────────────────────────────────────

def run_cascade(
    prices:   Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    dry_run:  bool                   = False,
) -> CascadeResult:
    """
    Full cascade pipeline: fit → predict → store.

    Parameters
    ----------
    prices   : pre-loaded price matrix.  Loaded from DB if None.
    macro_df : pre-loaded macro DataFrame.  Fetched from yfinance if None.
    db_path  : SQLite path.  Defaults to data/commodities.db.
    dry_run  : if True, fit and predict but do NOT write to the DB.

    Returns
    -------
    CascadeResult (always returned; check .success and .errors).
    """
    result = CascadeResult(
        run_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        # ── Load data ──────────────────────────────────────────────────────────
        if prices is None or prices.empty:
            prices = _load_prices()
        if prices.empty:
            raise RuntimeError("Price matrix is empty — cannot run cascade.")

        if macro_df is None or macro_df.empty:
            macro_df = _load_macro(price_index=prices.index)

        # ── Determine forecast date ────────────────────────────────────────────
        forecast_date = prices.index[-1].date() if hasattr(prices.index[-1], "date") else date.today()
        result.forecast_date = forecast_date

        # ── Fit ────────────────────────────────────────────────────────────────
        log.info("Cascade fit starting (forecast_date=%s)…", forecast_date)
        cascade = CascadeForecaster()
        cascade.fit(prices, macro_df)

        result.regime        = cascade._regime
        result.macro_snapshot = cascade._macro_snapshot

        if not cascade._models:
            raise RuntimeError("No commodity models could be fitted.")

        # ── Predict in causal order ────────────────────────────────────────────
        log.info("Cascade predict starting…")
        forecasts = cascade.predict()

        result.commodities = forecasts
        log.info("Cascade predict complete: %d commodity forecasts.", len(forecasts))

        # ── Store ──────────────────────────────────────────────────────────────
        if not dry_run and forecasts:
            n = _write_cascade_forecasts(forecasts, forecast_date)
            result.n_written = n
            log.info("Cascade: wrote %d rows to cascade_forecasts (date=%s).", n, forecast_date)
        elif dry_run:
            log.info("Dry run — cascade forecasts NOT written to DB.")

        result.success = True

    except Exception as exc:
        result.errors["_pipeline"] = str(exc)
        log.error("Cascade pipeline failed: %s", exc, exc_info=True)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """python -m models.cascade_orchestrator [--dry-run] [--verbose]"""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Run cascade sector forecasts.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Fit and predict but do not write to DB.")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show DEBUG log output.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = run_cascade(dry_run=args.dry_run)
    print(result.pretty())

    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
