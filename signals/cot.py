"""
COT positioning-reversal — the first FUNDAMENTAL (non-price) signal through the gate.

ECONOMIC RATIONALE
──────────────────
The CFTC Commitments-of-Traders report shows how net-long "managed money"
(speculators) are in each futures market. Two competing theories link positioning
to forward returns, and they disagree on SIGN:

  • Hedging-pressure / risk-premium (Cootner 1960; De Roon-Nijman-Veld 2000;
    Basu-Miffre 2013): speculators are paid to absorb hedgers' risk, so positioning
    LEVEL is return-PREDICTIVE in the same direction — when specs are net long
    (hedgers short), the market is backwardated and longs earn a premium.
  • Positioning-EXTREME reversal (the practitioner "COT index"; Sanders-Irwin-Merrin
    2009 on the limits of crowded specs): at extremes the marginal speculative buyer
    is exhausted and crowded positions unwind, so extreme net-long predicts
    mean-reversion DOWN and extreme net-short predicts a bounce UP.

This signal implements the REVERSAL hypothesis: it scores each instrument by how
stretched its current net managed-money position is versus its OWN recent history
(a trailing z-score) and forecasts the NEGATIVE of that — short the crowded longs,
long the crowded shorts. The two theories are in genuine tension, so the GATE
adjudicates: if the out-of-sample IC comes back negative, positioning here behaves
as a risk-premium (momentum-aligned) signal rather than a reversal one, which is
itself a reportable finding (flip the sign / switch to the hedging-pressure
construction) — not a silent failure.

POINT-IN-TIME
─────────────
COT is released Friday for the prior Tuesday (a release-date lag baked into the
store at ingest). ``compute(asof, panel)`` reads positioning via the fundamental
store's raw rows filtered to ``release_date <= asof`` (latest vintage per report
date), so a decision on date ``t`` only ever sees reports public by ``t``. The raw
rows are loaded ONCE and the PIT filter is replayed in memory per date for speed.
``panel`` is used only for its instrument set (no price look-ahead).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, register_signal


class _CotPositioning(Signal):
    """
    Shared COT positioning machinery. NOT registered — concrete subclasses set
    ``name``, ``economic_rationale`` and ``SIGN`` (+1 risk-premium / -1 reversal).

    Both subclasses score the SAME quantity — how stretched current net
    managed-money positioning is versus the instrument's own trailing history (a
    z-score) — and differ only in the sign of the forecast, i.e. in which of two
    competing economic theories they bet on. Keeping them as one code path means
    the gate compares two pre-registered hypotheses on identical inputs.
    """

    #: +1 => forecast follows positioning (risk-premium); -1 => contrarian.
    SIGN: int = 0

    def __init__(self, lookback_weeks: int = 156, min_weeks: int = 52):
        # ~3y trailing window to define "stretched"; require ~1y of reports before
        # an instrument is scorable so the z-score has a stable reference.
        self.lookback_weeks = int(lookback_weeks)
        self.min_weeks = int(min_weeks)
        self._raw: Optional[pd.DataFrame] = None

    def _cot(self) -> pd.DataFrame:
        """Load raw COT rows once (every vintage), cached on the instance."""
        if self._raw is None:
            from data import fundamental_store as store

            raw = store.load_raw(source="cftc")
            # Only instrument-mapped rows can join the cross-section.
            self._raw = raw[raw["instrument"].notna()].copy() if not raw.empty else raw
        return self._raw

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        raw = self._cot()
        if raw is None or raw.empty:
            return self._empty_frame(panel.columns)

        asof_ts = pd.Timestamp(asof)
        # ── point-in-time: only reports released by asof, latest vintage each ──
        pit = raw[raw["release_date"] <= asof_ts]
        if pit.empty:
            return self._empty_frame(panel.columns)
        pit = (
            pit.sort_values("release_date")
            .groupby(["instrument", "reference_date"], as_index=False)
            .tail(1)
        )

        scores = {}
        for inst, g in pit.groupby("instrument"):
            s = g.sort_values("reference_date")["value"]
            if len(s) < self.min_weeks:
                continue
            window = s.iloc[-self.lookback_weeks :]
            mu = window.mean()
            sigma = window.std()
            if not np.isfinite(sigma) or sigma == 0:
                continue
            z = (s.iloc[-1] - mu) / sigma
            scores[inst] = self.SIGN * float(z)

        score = pd.Series(scores, dtype=float).dropna()
        if score.empty:
            return self._empty_frame(panel.columns)
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


@register_signal
class CotPositioningReversal(_CotPositioning):
    """Contrarian score on stretched managed-money net positioning (CFTC COT)."""

    name = "cot_reversal"
    SIGN = -1
    economic_rationale = (
        "CFTC managed-money net positioning, scored as a contrarian reversal: an "
        "instrument whose speculative net-long is stretched far above its own "
        "trailing norm tends to mean-revert down (crowded longs unwind), and a "
        "stretched net-short tends to bounce. Forecast = -z(net positioning). "
        "Competes with hedging-pressure risk-premium theory (which would be "
        "same-signed); the gate adjudicates the sign out-of-sample."
    )


@register_signal
class CotPositioningRiskPremium(_CotPositioning):
    """Hedging-pressure risk-premium reading of the SAME positioning z-score."""

    name = "cot_risk_premium"
    SIGN = +1
    economic_rationale = (
        "CFTC managed-money net positioning as a hedging-pressure risk premium: "
        "speculators are paid to absorb hedgers' risk, so a high net-long (hedgers "
        "short, market backwardated) predicts POSITIVE forward returns and a high "
        "net-short predicts negative — forecast = +z(net positioning) "
        "(Cootner 1960; De Roon-Nijman-Veld 2000; Basu-Miffre 2013). The opposite "
        "sign of cot_reversal; the gate decides which theory holds in this universe."
    )
