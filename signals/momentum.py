"""
Cross-sectional momentum — the gate's first real, economically-grounded signal.

ECONOMIC RATIONALE
──────────────────
Commodity futures exhibit a robust cross-sectional momentum premium: instruments
that have outperformed their peers over the past several months tend to keep
outperforming over the next few weeks, and laggards keep lagging. The standard
academic construction (Jegadeesh-Titman 1993 for equities; Erb-Harvey 2006 and
Miffre-Rallis 2007 for commodities) ranks instruments by their trailing 12-month
return *skipping the most recent month* (the "12-1" window) to avoid the short-term
reversal that contaminates the most recent weeks. Going long the winners and short
the losers harvests the premium. The fundamental drivers are slow diffusion of
supply/demand information and producer hedging pressure — a real reason to exist,
not a data-mined pattern.

This is a RELATIVE-VALUE signal: the per-instrument number is a cross-sectional
score (rank-demeaned, vol-scaled trailing return), not a calibrated expected
return. The harness uses it cross-sectionally (Spearman IC + long-short PnL), so
the score's sign and rank are what matter.

POINT-IN-TIME
─────────────
``compute(asof, panel)`` slices ``panel.loc[:asof]`` first and touches nothing
after ``asof``. Appending future rows cannot change the output.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from signals.base import (
    CONFIDENCE_FIELD,
    FORECAST_FIELD,
    Signal,
    register_signal,
)


@register_signal
class CrossSectionalMomentum(Signal):
    """12-1 cross-sectional momentum across the instrument universe."""

    name = "momentum_xs"
    economic_rationale = (
        "Commodity futures show a persistent cross-sectional momentum premium: "
        "winners over the trailing ~12 months (skipping the most recent month to "
        "avoid short-term reversal) keep outperforming laggards over the next few "
        "weeks. Driven by slow information diffusion and producer hedging pressure "
        "(Erb-Harvey 2006, Miffre-Rallis 2007). Long winners / short losers."
    )

    def __init__(self, lookback: int = 252, skip: int = 21, vol_window: int = 63):
        # lookback/skip default to the classic 12-1 window in trading days.
        # vol_window scales the trailing return by its own volatility so the
        # cross-section is comparable across high- and low-vol instruments.
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.vol_window = int(vol_window)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        # ── point-in-time slice: nothing after asof is ever read ──────────────
        hist = panel.loc[:pd.Timestamp(asof)]

        min_rows = self.lookback + 1
        if len(hist) < min_rows:
            return self._empty_frame(panel.columns)

        log_px = np.log(hist)
        daily_ret = log_px.diff()

        # Trailing 12-1 return: sum of daily log returns over the window that ends
        # `skip` days before asof and starts `lookback` days before asof.
        window = daily_ret.iloc[-self.lookback : -self.skip] if self.skip > 0 else daily_ret.iloc[-self.lookback :]
        trailing = window.sum(min_count=max(1, (self.lookback - self.skip) // 2))

        # Volatility scaling — normalise each instrument's signal by its own
        # recent daily vol so the cross-section is risk-comparable.
        vol = daily_ret.iloc[-self.vol_window :].std()
        vol = vol.replace(0.0, np.nan)
        scaled = trailing / (vol * np.sqrt(self.vol_window))

        scaled = scaled.dropna()
        if scaled.empty:
            return self._empty_frame(panel.columns)

        # Cross-sectional z-score → a dollar-neutral relative score.
        mu = scaled.mean()
        sigma = scaled.std()
        if not np.isfinite(sigma) or sigma == 0:
            return self._empty_frame(panel.columns)
        score = (scaled - mu) / sigma

        # Confidence: conviction grows with |z|, capped at 1. (A name ranked far
        # from the cross-sectional mean is a higher-conviction bet.)
        confidence = score.abs().clip(upper=3.0) / 3.0

        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]],
            names=["horizon", "field"],
        )
        out = pd.DataFrame(index=score.index, columns=cols, dtype=float)
        out.index.name = "instrument"
        # Cross-sectional momentum is a relative-value view; the same score is the
        # forecast at every horizon. The harness discovers WHICH horizon it best
        # predicts — momentum typically strengthens with horizon.
        for h in self.horizons:
            out[(h, FORECAST_FIELD)] = score
            out[(h, CONFIDENCE_FIELD)] = confidence
        return out
