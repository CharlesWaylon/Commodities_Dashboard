"""
Macro-surprise — cross-sectional macro factor exposures × recent macro moves.

ECONOMIC RATIONALE
──────────────────
A scalar macro state ("rates rose", "the dollar fell") cannot, on its own, RANK a
cross-section — it is the same number for every instrument. What ranks commodities
is their differing EXPOSURE to macro factors: gold falls when real rates and the
dollar rise; industrial metals and energy rally on growth/inflation; risk-off
(spiking volatility) punishes the cyclical complex. So the signal is each
instrument's BETA to a handful of macro factors multiplied by the factor's RECENT
move:

    forecast_i = Σ_f  β_{i,f} · s_f

where β_{i,f} is instrument i's trailing sensitivity to factor f and s_f is the
standardised recent change in factor f. The edge exists because macro shocks
diffuse into the commodity cross-section with a lag (not all instruments reprice
instantly), so beta × recent-shock carries cross-sectional information for the next
few weeks. This is the macro analogue of factor momentum (Moskowitz) routed
through asset-specific betas.

FACTORS (daily, market-priced — chosen so the "surprise" is continuous and not
already fully absorbed on a monthly release day, the failure mode that sank the
monthly hard-data and inventory constructions):
  • T10YIE   — 10y breakeven inflation expectations
  • DGS10    — 10y nominal Treasury yield (rates; with T10YIE separates real rate)
  • DTWEXBGS — broad trade-weighted USD
  • VIXCLS   — equity volatility (risk-on/off)

POINT-IN-TIME
─────────────
Daily market series are not revised, so each value's first print is its only
vintage. The factor panel is built once, stamped by ``release_date`` (= reference
+ 1 bday at ingest), and sliced to ``release_date <= asof`` — a decision on ``t``
sees only macro data public by ``t``. Betas use only the trailing window of
returns/factors up to ``asof``; ``panel`` returns are sliced ``<= asof``.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, register_signal

_FACTORS = ("T10YIE", "DGS10", "DTWEXBGS", "VIXCLS")


@register_signal
class MacroSurprise(Signal):
    """Cross-sectional macro factor betas × recent standardised macro moves."""

    name = "macro_surprise"
    economic_rationale = (
        "Macro shocks transmit into the commodity cross-section through differing "
        "factor exposures (gold short the dollar/real rates; energy & metals long "
        "growth/inflation; risk-off hurts cyclicals) and diffuse with a lag. "
        "Forecast = Σ β_{i,f}·s_f: each instrument's trailing beta to inflation "
        "expectations, 10y rates, the broad USD and equity vol, times the recent "
        "standardised move in each factor."
    )

    def __init__(self, beta_window: int = 252, surprise_window: int = 10, min_obs: int = 150):
        self.beta_window = int(beta_window)
        self.surprise_window = int(surprise_window)
        self.min_obs = int(min_obs)
        self._panel: Optional[pd.DataFrame] = None       # factor levels, index=reference_date
        self._usable_by: Optional[pd.Series] = None      # reference_date -> release_date

    def _macro(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Build the factor level panel once (daily series → 1 vintage each)."""
        if self._panel is None:
            from data import fundamental_store as store

            raw = store.load_raw(source="fred", series_ids=list(_FACTORS))
            if raw.empty:
                self._panel = pd.DataFrame()
                self._usable_by = pd.Series(dtype="datetime64[ns]")
                return self._panel, self._usable_by
            # latest vintage per (series_id, reference_date) — a no-op for unrevised
            # daily series, but correct if a revision ever appears.
            raw = raw.sort_values("release_date").groupby(["series_id", "reference_date"], as_index=False).tail(1)
            self._panel = raw.pivot(index="reference_date", columns="series_id", values="value").sort_index()
            # A reference_date is usable only once ALL its factors are released.
            self._usable_by = raw.groupby("reference_date")["release_date"].max()
        return self._panel, self._usable_by

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        levels, usable_by = self._macro()
        if levels is None or levels.empty:
            return self._empty_frame(panel.columns)

        asof_ts = pd.Timestamp(asof)
        # ── point-in-time slice of the factor panel ───────────────────────────
        usable_refs = usable_by.index[usable_by <= asof_ts]
        L = levels.loc[levels.index.isin(usable_refs), [c for c in _FACTORS if c in levels.columns]]
        if L.shape[1] < 2 or len(L) < self.min_obs + self.surprise_window:
            return self._empty_frame(panel.columns)

        chg = L.diff()

        # Instrument returns, PIT.
        ret = np.log(panel.loc[:asof_ts]).diff()

        # Trailing estimation window: factor rows complete and present in returns.
        win0 = chg.iloc[-(self.beta_window + 1):]
        common = win0.index.intersection(ret.index)
        win = win0.loc[common].dropna(how="any")
        if len(win) < self.min_obs:
            return self._empty_frame(panel.columns)

        mu = win.mean()
        sd = win.std().replace(0.0, np.nan)
        if sd.isna().any():
            return self._empty_frame(panel.columns)

        # Standardised factor changes (T×K), demeaned for an intercept-free OLS.
        Z = (win - mu) / sd
        Xz = Z.to_numpy()
        Xz = Xz - Xz.mean(axis=0)

        # Returns over the same window, instruments with COMPLETE history only.
        R = ret.loc[win.index]
        R = R.dropna(axis=1, how="any")
        if R.shape[1] < 2:
            return self._empty_frame(panel.columns)
        Rd = R.to_numpy()
        Rd = Rd - Rd.mean(axis=0)

        # Vectorised multivariate betas for the WHOLE cross-section at once:
        # B = (X'X)^-1 X'R  (K×N). Tiny ridge for numerical stability.
        K = Xz.shape[1]
        XtX = Xz.T @ Xz + 1e-8 * np.eye(K)
        try:
            B = np.linalg.solve(XtX, Xz.T @ Rd)  # K×N
        except np.linalg.LinAlgError:
            return self._empty_frame(panel.columns)

        # Recent standardised macro move per factor (same std units as Z).
        recent = chg.iloc[-self.surprise_window:].mean()
        s = ((recent - mu) / sd).reindex(Z.columns).to_numpy()
        if not np.isfinite(s).all():
            return self._empty_frame(panel.columns)

        raw_forecast = s @ B  # (N,)
        score = pd.Series(raw_forecast, index=R.columns, dtype=float).dropna()
        if score.empty or not np.isfinite(score.std()) or score.std() == 0:
            return self._empty_frame(panel.columns)

        # Cross-sectional z-score → dollar-neutral relative view.
        score = (score - score.mean()) / score.std()
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
