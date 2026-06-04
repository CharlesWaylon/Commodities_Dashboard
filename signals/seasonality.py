"""
Seasonality — calendar effects with a genuine physical basis.

ECONOMIC RATIONALE
──────────────────
Commodities have real, recurring seasonal demand/supply cycles: heating-oil and
natural-gas demand peaks in winter, gasoline in the summer driving season, grains
swing around the planting/harvest calendar, and these patterns are well documented
(e.g. Sorensen 2002 on agricultural seasonality; the energy seasonality literature).
Unlike a data-mined calendar quirk, the cause is physical and repeats, so a
seasonal expectation has a fundamental reason to exist.

CONSTRUCTION
────────────
For each instrument we estimate the mean daily return conditional on calendar
month using ONLY history available at ``asof``. The horizon-``h`` forecast is the
expected cumulative return over the next ``h`` business days, obtained by summing
those monthly means over the forward calendar window. This makes the signal
genuinely multi-horizon (a 5-day vs 21-day forward window can straddle different
months) and lets the gate discover the horizon at which seasonality pays.

POINT-IN-TIME
─────────────
Monthly means use only ``panel.loc[:asof]``. The forward window is pure calendar
arithmetic (future business *dates*, not future *data*), so no look-ahead: a
property test confirms appending future rows cannot change the output.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, register_signal


@register_signal
class Seasonality(Signal):
    """Forward-window expected return from historical calendar-month seasonal means."""

    name = "seasonality"
    economic_rationale = (
        "Commodities have recurring physical seasonal cycles (winter heating "
        "demand for nat-gas/heating oil, summer gasoline driving season, "
        "planting/harvest cycles for grains; Sorensen 2002 and the energy "
        "seasonality literature). The horizon-h forecast is the expected return "
        "over the forward calendar window from month-conditional historical means."
    )

    def __init__(self, min_history: int = 504, min_month_obs: int = 8):
        # ~2y minimum so each calendar month is seen at least twice; require a
        # handful of observations per month before trusting its mean.
        self.min_history = int(min_history)
        self.min_month_obs = int(min_month_obs)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        asof_ts = pd.Timestamp(asof)
        hist = panel.loc[:asof_ts]
        if len(hist) < self.min_history:
            return self._empty_frame(panel.columns)

        daily_ret = np.log(hist).diff()
        months = daily_ret.index.month

        # Monthly mean daily return per instrument (only past data). Months with
        # too few observations are left NaN so they don't pollute the forward sum.
        monthly_mean = daily_ret.groupby(months).mean()
        monthly_count = daily_ret.groupby(months).count()
        monthly_mean = monthly_mean.where(monthly_count >= self.min_month_obs)

        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]], names=["horizon", "field"]
        )
        out = pd.DataFrame(index=pd.Index(panel.columns, name="instrument"), columns=cols, dtype=float)

        for h in self.horizons:
            # Forward business-day window — calendar arithmetic only, no data read.
            fwd_dates = pd.bdate_range(start=asof_ts + pd.tseries.offsets.BDay(1), periods=h)
            fwd_months = pd.Index(fwd_dates.month)
            # Expected cumulative return = Σ monthly_mean[month] over the window.
            contrib = monthly_mean.reindex(fwd_months)  # rows=forward days, cols=instrument
            expected = contrib.sum(min_count=max(1, h // 2))  # NaN if too sparse
            score = expected.dropna()
            if score.empty:
                continue
            denom = score.abs().max() or np.nan
            confidence = (score.abs() / denom).clip(upper=1.0) if np.isfinite(denom) else score.abs() * 0.0
            out.loc[score.index, (h, FORECAST_FIELD)] = score
            out.loc[score.index, (h, CONFIDENCE_FIELD)] = confidence

        if out.dropna(how="all").empty:
            return self._empty_frame(panel.columns)
        return out
