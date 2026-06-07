"""
portfolio/cascade_allocator.py — cascade-forecast μ-substitution allocator.

Re-casts the legacy ``models.portfolio_optimizer.CascadePortfolioOptimizer`` — which
substituted cascade-derived expected returns into the QAOA selection — as a
first-class Layer-3 ``Allocator``. Same MV-selection machinery as every other
selection allocator (so the bake-off is apples-to-apples), but the cross-sectional
view of expected return comes from the cascade orchestrator's per-instrument
``final_forecast`` instead of the signal's own forecast column.

Honest data caveat
──────────────────
``cascade_forecasts`` is a LIVE, daily-produced table — it accumulates day by day
and has only a short historical tail (weeks, not years). It therefore CANNOT
compete fairly in a multi-year backtest: on any date before its coverage starts,
this allocator has no cascade view and FALLS BACK to the base allocator's signal
forecast. That fallback is the design: it lets the same allocator run continuously
through history without crashes, but the bake-off will naturally penalise cascade
on long panels (it is starving for data). The right home for cascade is the LIVE
production page (gap 2), where it has same-day data.

POINT-IN-TIME
─────────────
Reads ``forecast_date <= asof`` from the ``cascade_forecasts`` table — the same
PIT discipline as every other Layer-3 read. The asof is taken from
``risk_model.asof``, which the backtest already sets correctly. Reads nothing
after ``asof``.

LAYER DISCIPLINE
────────────────
numpy / pandas / sqlalchemy. May read from the data layer (the cascade table).
No streamlit / pages / app (enforced by .importlinter).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from portfolio.allocators import MeanVarianceSelectAllocator, _SelectAllocator
from portfolio.risk import RiskModel

logger = logging.getLogger(__name__)


def load_cascade_view(asof) -> pd.Series:
    """
    Cascade ``final_forecast`` per instrument as of ``asof`` (latest forecast_date
    ≤ asof). Empty Series on failure / no coverage — never raises.
    """
    try:
        from sqlalchemy import text

        from database.db import get_engine
    except Exception:  # pragma: no cover - environment-dependent
        return pd.Series(dtype=float)

    asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
    sql = text("""
        SELECT commodity, final_forecast
          FROM cascade_forecasts
         WHERE forecast_date <= :asof_str
           AND (commodity, forecast_date) IN (
             SELECT commodity, max(forecast_date)
               FROM cascade_forecasts
              WHERE forecast_date <= :asof_str
              GROUP BY commodity
           )
    """)
    try:
        with get_engine().connect() as conn:
            df = pd.DataFrame(conn.execute(sql, {"asof_str": asof_str}).fetchall(),
                              columns=["instrument", "final_forecast"])
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("instrument")["final_forecast"].astype(float)


class CascadeAugmentedAllocator(_SelectAllocator):
    """
    Cardinality-constrained mean-variance selection on cascade-substituted μ.

    Identical math to ``MeanVarianceSelectAllocator``; the only difference is the
    forecast view: when cascade data exists at ``risk_model.asof``, it replaces
    the signal forecast for instruments cascade covers; the rest fall back to the
    signal forecast. This is the legacy "cascade μ" substitution made
    bake-off-compatible.
    """

    name = "cascade"

    def __init__(self, *args, fallback: Optional[_SelectAllocator] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fallback = fallback or MeanVarianceSelectAllocator(
            k=self.k, lam=self.lam, n_universe=self.n_universe,
            target_vol=self.target_vol, periods_per_year=self.periods_per_year,
            max_leverage=self.max_leverage,
        )
        self._warned_no_cascade = False

    def _cascade_blend(self, forecasts: pd.Series, asof) -> pd.Series:
        """Replace signal forecast with cascade view where available; preserve the rest.

        Both signal and cascade are converted to percentile ranks across the FULL
        signal universe before substitution, so scales match (cascade covering only
        a few names would otherwise be crushed by the signal's full-range scale).
        Names not covered by cascade keep their signal rank → the cross-section is
        whole and comparable.
        """
        view = load_cascade_view(asof)
        if view.empty:
            if not self._warned_no_cascade:
                logger.info("CascadeAugmentedAllocator: no cascade view at %s — using signal forecast.", asof)
                self._warned_no_cascade = True
            return forecasts
        sig = forecasts.dropna().astype(float)
        if sig.empty:
            return forecasts
        overlap = view.index.intersection(sig.index)
        if len(overlap) == 0:
            return forecasts
        # Rank EVERYTHING on the same percentile scale so cascade and signal are
        # comparable across the full universe regardless of cascade coverage.
        blended_rank = (sig.rank(pct=True) - 0.5).copy()
        cas_rank = view.reindex(overlap).rank(pct=True) - 0.5
        blended_rank.loc[overlap] = cas_rank.values
        return blended_rank

    def allocate(self, forecasts: pd.Series, risk_model: RiskModel) -> pd.Series:
        asof = getattr(risk_model, "asof", None)
        if asof is None:
            # Without asof we cannot read PIT cascade — defer to the fallback.
            return self._fallback.allocate(forecasts, risk_model)
        substituted = self._cascade_blend(forecasts, asof)
        cand, mu, cov = self._candidates(substituted, risk_model)
        if not cand or len(cand) < self.k:
            return pd.Series(dtype=float)
        idx = self._select(mu, cov)
        if not idx:
            return pd.Series(dtype=float)
        sel = [cand[i] for i in idx]
        w = pd.Series(1.0 / len(sel), index=sel)
        from portfolio.allocators import _vol_target

        return _vol_target(w, risk_model, self.target_vol, self.periods_per_year, self.max_leverage)

    def _select(self, mu: np.ndarray, cov: np.ndarray):
        # Same exact MV optimum as the classical baseline — cascade differs only in μ.
        from itertools import combinations

        n = len(mu)
        best_obj, best = None, None
        for c in combinations(range(n), self.k):
            x = np.zeros(n)
            x[list(c)] = 1.0
            obj = float(x @ cov @ x - self.lam * (mu @ x))
            if best_obj is None or obj < best_obj:
                best_obj, best = obj, list(c)
        return best or []
