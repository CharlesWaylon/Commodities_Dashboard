"""
Value — commodity long-horizon mean reversion (the value/momentum complement).

ECONOMIC RATIONALE
──────────────────
Commodities that are cheap relative to their own multi-year history tend to
outperform, and richly-priced ones underperform, as prices mean-revert toward the
marginal cost of production over multi-year horizons. This is the commodity "value"
factor of Asness-Moskowitz-Pedersen (2013, "Value and Momentum Everywhere"), who
define value as (roughly) the negative of the long-horizon return — long what has
fallen, short what has risen over several years. Crucially, value is NEGATIVELY
correlated with momentum, so the two combine far better than either alone (the
canonical value+momentum pairing) — exactly the orthogonal breadth the ensemble
needs.

HISTORY CONSTRAINT (honesty note)
─────────────────────────────────
The textbook horizon is ~5 years, but the aligned price panel only has ~5 years of
COMMON history (its start is gated by the youngest instruments — newer ETF proxies
and crypto), and is calendar-day aligned (~365 rows/yr). A 5-year reference would
leave almost no evaluation window, so the reference here is the average log-price
over a window roughly 1.4-2.75 years back. This is shorter than canonical value but
is the deepest mean-reversion the data supports; it remains distinct from the
1-month reversal and ~8-month momentum windows. Revisit with a longer reference
once the panel's common history deepens.

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
class Value(Signal):
    """Long-horizon mean reversion: long the multi-year cheap, short the multi-year rich."""

    name = "value"
    economic_rationale = (
        "Commodity value (Asness-Moskowitz-Pedersen 2013): prices mean-revert toward "
        "the cost of production over multi-year horizons, so instruments cheap vs "
        "their own multi-year price are expected to outperform. Forecast = "
        "+z(reference log-price − current log-price) using a reference ~1.4-2.75y "
        "back. Negatively correlated with momentum by construction."
    )

    def __init__(self, gap: int = 504, span: int = 504):
        # reference window = the `span` rows ending `gap` rows before asof
        # (≈ 1.4y wide, ending ≈ 1.4y ago → spanning ~1.4-2.75y back).
        self.gap = int(gap)
        self.span = int(span)

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        hist = panel.loc[: pd.Timestamp(asof)]
        if len(hist) < self.gap + self.span + 1:
            return self._empty_frame(panel.columns)

        log_px = np.log(hist)
        ref_window = log_px.iloc[-(self.gap + self.span) : -self.gap]
        # per-instrument reference mean, requiring enough observations behind it
        reference = ref_window.mean().where(ref_window.count() >= max(1, self.span // 2))
        current = log_px.iloc[-1]

        value = (reference - current).dropna()  # cheap (current below ref) -> positive
        if value.empty:
            return self._empty_frame(panel.columns)

        sigma = value.std()
        if not np.isfinite(sigma) or sigma == 0:
            return self._empty_frame(panel.columns)
        score = (value - value.mean()) / sigma
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
