"""
Time-series momentum (trend) — the own-history complement to cross-sectional
momentum.

ECONOMIC RATIONALE
──────────────────
Distinct from cross-sectional momentum (which ranks instruments against each
other), time-series momentum bets each instrument on the SIGN of its OWN trailing
return: an instrument trending up tends to keep trending up, one trending down
keeps falling. Moskowitz, Ooi & Pedersen (2012, "Time Series Momentum") document
a significant ~12-month trend premium across 58 futures including commodities,
attributed to under-reaction to information and demand from hedgers/CTAs. It is
the canonical trend-following edge and is largely orthogonal to the
cross-sectional version — exactly the kind of independent, economically-grounded
signal the ensemble wants for breadth (Edge = IC × √breadth).

The per-instrument score is the trailing return scaled by its own volatility, so
high- and low-vol instruments contribute comparably. The harness scores it
cross-sectionally (Spearman IC + long-short book), so sign and rank are what
matter.

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
class TimeSeriesMomentum(Signal):
    """12-month time-series (trend) momentum, vol-scaled, per instrument."""

    name = "trend_ts"
    economic_rationale = (
        "Time-series momentum: each instrument's own trailing ~12-month return "
        "(skipping the most recent month) predicts the sign of its next-period "
        "return. A robust trend-following premium across futures incl. commodities "
        "(Moskowitz-Ooi-Pedersen 2012), driven by under-reaction to information and "
        "hedger/CTA demand. Long up-trends / short down-trends, vol-scaled."
    )

    def __init__(self, lookback: int = 252, skip: int = 21, vol_window: int = 63):
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.vol_window = int(vol_window)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        hist = panel.loc[: pd.Timestamp(asof)]
        if len(hist) < self.lookback + 1:
            return self._empty_frame(panel.columns)

        daily_ret = np.log(hist).diff()

        # Trailing return over the 12-1 window (ends `skip` days before asof).
        window = daily_ret.iloc[-self.lookback : -self.skip] if self.skip > 0 else daily_ret.iloc[-self.lookback :]
        trailing = window.sum(min_count=max(1, (self.lookback - self.skip) // 2))

        # Volatility scaling: trend strength per unit of own risk. Unlike the
        # cross-sectional signal we do NOT demean — the absolute sign carries the
        # time-series view (an all-up market should be net long).
        vol = daily_ret.iloc[-self.vol_window :].std()
        vol = vol.replace(0.0, np.nan)
        score = (trailing / (vol * np.sqrt(self.vol_window))).dropna()
        if score.empty:
            return self._empty_frame(panel.columns)

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
