"""
Short-term reversal — the liquidity-provision complement to momentum.

ECONOMIC RATIONALE
──────────────────
Over short horizons (about a week to a month) returns tend to REVERSE rather than
persist: an instrument that has jumped recently tends to give some of it back, and
a recent laggard tends to bounce. The accepted cause is liquidity provision and
overreaction — traders demanding immediacy push prices temporarily away from fair
value, and liquidity providers earn the snap-back (Jegadeesh 1990; Lehmann 1990;
the short-term reversal factor of Khandani-Lo and AQR). This is distinct from, and
empirically opposite to, the 12-1 momentum window, so it is a near-orthogonal
source of breadth for the ensemble.

This construction is also pre-validated by THIS project's own gate output: the
`carry_proxy` signal (risk-adjusted short momentum mom10/vol21) came back with a
significantly NEGATIVE information coefficient — i.e. short-horizon momentum
reverses here — so a sign-correct reversal factor is the right reading of that
finding. We use a 1-month (21-day) window (the standard short-term-reversal factor,
lower turnover than a 1-week version, and NOT identical to carry_proxy's 10-day
window — a fresh test rather than a mechanical sign-flip).

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
class ShortTermReversal(Signal):
    """1-month short-term reversal: short the recent winners, long the recent losers."""

    name = "reversal_st"
    economic_rationale = (
        "Short-horizon returns reverse: recent winners give back and recent losers "
        "bounce, as liquidity providers are paid for absorbing immediacy demand "
        "(Jegadeesh 1990; Lehmann 1990). Forecast = -z(trailing 1-month vol-scaled "
        "return). Near-orthogonal to 12-1 momentum and pre-validated by this "
        "project's carry_proxy negative-IC finding."
    )

    def __init__(self, lookback: int = 21, vol_window: int = 63):
        self.lookback = int(lookback)
        self.vol_window = int(vol_window)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        hist = panel.loc[: pd.Timestamp(asof)]
        if len(hist) < self.vol_window + 1:
            return self._empty_frame(panel.columns)

        daily_ret = np.log(hist).diff()
        trailing = daily_ret.iloc[-self.lookback :].sum(min_count=max(1, self.lookback // 2))
        vol = daily_ret.iloc[-self.vol_window :].std().replace(0.0, np.nan)
        scaled = (trailing / (vol * np.sqrt(self.vol_window))).dropna()
        if scaled.empty:
            return self._empty_frame(panel.columns)

        sigma = scaled.std()
        if not np.isfinite(sigma) or sigma == 0:
            return self._empty_frame(panel.columns)
        score = -(scaled - scaled.mean()) / sigma  # reversal: recent strength -> short
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
