"""
Low volatility — the betting-against-vol / low-risk anomaly.

ECONOMIC RATIONALE
──────────────────
Low-risk assets earn higher RISK-ADJUSTED returns than high-risk assets — the
opposite of what the CAPM predicts. Leverage- and lottery-constrained investors
crowd into high-volatility / high-beta names (paying up for the chance of a big
move), which depresses their forward returns, while shunned low-vol names are left
cheap. Going long low-volatility and short high-volatility instruments harvests the
gap (Frazzini-Pedersen 2014 "Betting Against Beta"; Baker-Bradley-Wurgler 2011;
Blitz-de Groot for commodities). It is a risk-based, slow-moving edge — almost
orthogonal to return-direction signals like momentum and reversal, and to the COT
risk premium — exactly the kind of independent breadth the ensemble needs.

CONSTRUCTION
────────────
Rank instruments by trailing realised volatility of daily log returns; the forecast
is the NEGATIVE cross-sectional z-score of that volatility (low vol → high score →
long). Volatility is persistent, so turnover is naturally low.

POINT-IN-TIME
─────────────
``compute(asof, panel)`` slices ``panel.loc[:asof]`` and reads nothing after it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, register_signal


@register_signal
class LowVolatility(Signal):
    """Long low-volatility / short high-volatility instruments (low-risk anomaly)."""

    name = "low_vol"
    economic_rationale = (
        "The low-risk anomaly: low-volatility instruments earn higher risk-adjusted "
        "returns than high-volatility ones, because leverage-constrained investors "
        "overpay for high-vol exposure (Frazzini-Pedersen 2014; Baker-Bradley-"
        "Wurgler 2011). Forecast = -z(trailing realised volatility) — long the calm "
        "names, short the wild ones. Slow-moving and near-orthogonal to momentum."
    )

    def __init__(self, vol_window: int = 63):
        self.vol_window = int(vol_window)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        hist = panel.loc[: pd.Timestamp(asof)]
        if len(hist) < self.vol_window + 1:
            return self._empty_frame(panel.columns)

        daily_ret = np.log(hist).diff()
        vol = daily_ret.iloc[-self.vol_window :].std()
        # need a reasonable count of observations behind each vol estimate
        valid = daily_ret.iloc[-self.vol_window :].count() >= (self.vol_window // 2)
        vol = vol[valid].replace(0.0, np.nan).dropna()
        if vol.empty:
            return self._empty_frame(panel.columns)

        sigma = vol.std()
        if not np.isfinite(sigma) or sigma == 0:
            return self._empty_frame(panel.columns)
        score = -(vol - vol.mean()) / sigma  # low vol -> positive score (long)
        score.index.name = "instrument"
        confidence = score.abs().clip(upper=3.0) / 3.0

        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]], names=["horizon", "field"]
        )
        out = pd.DataFrame(index=score.index, columns=cols, dtype=float)
        out.index.name = "instrument"
        for h in self.horizons:
            out[(h, FORECAST_FIELD)] = score
            out[(h, CONFIDENCE_FIELD)] = confidence
        return out
