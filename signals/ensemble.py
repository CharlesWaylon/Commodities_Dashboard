"""
Ensemble — the parameter-free composite of the right-signed sub-threshold edges.

WHY THIS EXISTS
───────────────
Edge = IC × √breadth. Several signals came back through the gate economically
confirmed and right-signed but too NOISY to promote alone (IC IR ~0.10-0.19 vs the
0.30 bar): cross-sectional momentum, time-series momentum, and the COT
hedging-pressure risk premium. The textbook way to convert near-orthogonal,
sub-threshold edges into a promotable one is to COMBINE them — averaging
diversifies away idiosyncratic signal noise while keeping the shared directional
edge, lifting the information ratio.

CONSTRUCTION (deliberately parameter-free)
──────────────────────────────────────────
For each horizon we take each component's cross-sectional forecast, standardise it
(z-score across the instruments that have a view), and average the standardised
scores across whichever components cover each instrument, then re-standardise. No
weights are fitted — equal weight is the honest default because fitting component
weights on the same history the gate scores would be in-sample optimisation (the
overfitting failure mode this whole rebuild exists to avoid). IC- or risk-weighted
blending is a later step that must itself be justified out-of-sample.

Because each component already enforces the point-in-time contract, the ensemble
inherits it — it reads nothing but its components' ``compute(asof, panel)`` output.
The look-ahead property test and the contract test cover it via ``list_signals()``.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, get_signal, register_signal


@register_signal
class EnsembleComposite(Signal):
    """Equal-weight composite of right-signed, sub-threshold component signals."""

    name = "ensemble_v1"
    #: Right-signed, gate-confirmed-but-sub-threshold, mutually DISTINCT edges
    #: (2026-06-04). Pairwise cross-sectional rank-corr (H10): momentum/cot +0.59,
    #: momentum/reversal -0.10, cot/reversal -0.28 — reversal_st adds genuinely
    #: orthogonal (indeed anti-correlated) breadth.
    #: Excluded: trend_ts (corr +1.000 with momentum_xs under ranking — demeaning
    #: doesn't change ranks, so it would only double-weight momentum); low_vol
    #: (near-zero IC in this universe — orthogonal but a null, would add noise).
    COMPONENTS = ("momentum_xs", "cot_risk_premium", "reversal_st")
    economic_rationale = (
        "Equal-weight composite of economically-confirmed, right-signed but "
        "individually sub-threshold and mutually distinct edges — cross-sectional "
        "momentum, the COT hedging-pressure risk premium, and short-term reversal. "
        "Their low / negative mutual correlation (momentum/cot +0.59, "
        "momentum/reversal -0.10, cot/reversal -0.28) raises the information ratio "
        "(Edge = IC × √breadth) without fitting any in-sample weights."
    )

    def __init__(self):
        self._components: Optional[List[Signal]] = None

    def _comps(self) -> List[Signal]:
        if self._components is None:
            self._components = [get_signal(n) for n in self.COMPONENTS]
        return self._components

    @staticmethod
    def _zscore(s: pd.Series) -> Optional[pd.Series]:
        s = s.dropna()
        if s.empty:
            return None
        sd = s.std()
        if not np.isfinite(sd) or sd == 0:
            return None
        return (s - s.mean()) / sd

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        # Collect each component's standardised per-horizon forecast.
        per_h: dict[int, List[pd.Series]] = {h: [] for h in self.horizons}
        for sig in self._comps():
            try:
                out = sig.compute(asof, panel)
            except Exception:  # a broken component must not sink the ensemble
                continue
            if out is None or out.dropna(how="all").empty:
                continue
            for h in self.horizons:
                if (h, FORECAST_FIELD) not in out.columns:
                    continue
                z = self._zscore(out[(h, FORECAST_FIELD)])
                if z is not None:
                    per_h[h].append(z)

        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]], names=["horizon", "field"]
        )
        out = pd.DataFrame(index=pd.Index(panel.columns, name="instrument"), columns=cols, dtype=float)

        produced = False
        for h in self.horizons:
            comps = per_h[h]
            if not comps:
                continue
            # instruments × components, then mean across available components.
            # Restrict to the panel universe (a component may score instruments
            # outside this panel, e.g. COT reads all instruments in the store).
            mat = pd.concat(comps, axis=1)
            combined = mat.mean(axis=1, skipna=True).dropna()
            combined = combined[combined.index.isin(panel.columns)]
            z = self._zscore(combined)
            if z is None:
                continue
            out.loc[z.index, (h, FORECAST_FIELD)] = z
            out.loc[z.index, (h, CONFIDENCE_FIELD)] = z.abs().clip(upper=3.0) / 3.0
            produced = True

        if not produced:
            return self._empty_frame(panel.columns)
        return out


@register_signal
class EnsembleV2MultiRegime(EnsembleComposite):
    """Multi-regime composite: value + short-term reversal + COT risk premium.

    Built for the deep ~21y ``long_core`` panel, where ``value`` PROMOTES (H21 IC IR
    0.348) and cross-sectional ``momentum_xs`` flips wrong-signed over the full cycle
    — so momentum is dropped and the gate-clearing multi-regime value factor leads.
    On long_core (H21) the three are well-diversified: value/reversal +0.18,
    value/cot -0.21, reversal/cot -0.22. Inherits the equal-weight combine logic;
    only the component set differs.

    HONESTY CAVEAT: the component set was chosen partly from long_core results
    (include value because it promoted there; drop momentum because it is
    wrong-signed there). That is in-sample component selection — legitimate model
    selection among economically-motivated, individually-validated edges, but a
    LIVE promotion still requires nested / out-of-sample component selection. Until
    then this is a research composite, not a promoted live signal.
    """

    name = "ensemble_v2"
    COMPONENTS = ("value", "reversal_st", "cot_risk_premium")
    economic_rationale = (
        "Equal-weight multi-regime composite of value (long-horizon mean reversion; "
        "gate-clearing on the 21y panel), short-term reversal, and the COT "
        "hedging-pressure risk premium. Low/negative mutual correlation (value/"
        "reversal +0.18, value/cot -0.21, reversal/cot -0.22) gives genuine breadth; "
        "cross-sectional momentum is excluded as wrong-signed over the full cycle. "
        "Equal weight, no in-sample weight fitting."
    )
