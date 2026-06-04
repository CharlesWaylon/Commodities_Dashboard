"""
Carry proxy — a stand-in for true term-structure carry until the stitched-M2 basis
has the depth to drive a signal on its own.

ECONOMIC RATIONALE
──────────────────
The carry (roll-yield) premium is the most fundamentally-grounded commodity edge:
a market in backwardation (front richer than deferred) pays a positive roll yield
to a long, and historically backwardated commodities outperform contangoed ones
(Erb-Harvey 2006; Gorton-Rouwenhorst 2006; Koijen et al. 2018 "Carry"). The clean
measure is the futures-curve slope, which needs a reliable deferred-contract series
(the stitched-M2 basis, still accruing coverage behind its gate).

Until that depth arrives, this signal uses a documented PROXY: short-horizon
risk-adjusted momentum ``mom(10) / vol(21)``. Backwardation tends to coincide with
firm, low-volatility front-month strength, so risk-adjusted short momentum is a
noisy but economically-aligned read on carry. It is explicitly a placeholder — it
promotes itself out the moment the true basis clears its coverage gate — and is
logged as such (inconclusive-by-construction, not a claim of a distinct edge).

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
class CarryProxy(Signal):
    """Risk-adjusted short-horizon momentum mom(10)/vol(21) as a carry proxy."""

    name = "carry_proxy"
    economic_rationale = (
        "Proxy for the futures-curve carry premium (backwardation outperforms "
        "contango: Erb-Harvey 2006, Gorton-Rouwenhorst 2006, Koijen et al. 2018) "
        "until the stitched-M2 basis has coverage depth. Uses risk-adjusted short "
        "momentum mom(10)/vol(21), since backwardation coincides with firm "
        "low-vol front strength. Placeholder; promotes to true basis carry later."
    )

    def __init__(self, mom_window: int = 10, vol_window: int = 21):
        self.mom_window = int(mom_window)
        self.vol_window = int(vol_window)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        hist = panel.loc[: pd.Timestamp(asof)]
        min_rows = max(self.mom_window, self.vol_window) + 1
        if len(hist) < min_rows:
            return self._empty_frame(panel.columns)

        daily_ret = np.log(hist).diff()
        mom = daily_ret.iloc[-self.mom_window :].sum(min_count=max(1, self.mom_window // 2))
        vol = daily_ret.iloc[-self.vol_window :].std().replace(0.0, np.nan)
        score = (mom / (vol * np.sqrt(self.vol_window))).dropna()
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
