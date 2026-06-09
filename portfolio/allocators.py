"""
portfolio/allocators.py — forecast → position mapping (Layer 3, step 3.2).

Turns a signal's cross-sectional forecast into risk-aware target weights using the
``portfolio.risk.RiskModel``. Four ingredients, all standard institutional practice:

1. RISK-PARITY-AWARE INVERSE-VOL SIZING. A given signal strength on a calm
   instrument should command more capital than the same strength on a wild one, so
   each raw bet is scaled by 1/σ_i. This equalises risk contributions across names
   (the risk-parity principle) instead of equalising dollars.

2. CONCENTRATION CAPS. A per-instrument cap and a per-sector cap (as fractions of
   gross exposure) prevent any single name or sector (energy, metals, ags,
   livestock) from dominating — applied by iterated water-filling so the book stays
   normalised.

3. PORTFOLIO-VOL TARGETING. The whole book is scaled so its EX-ANTE annualised
   volatility — sqrt(wᵀ Σ w) from the risk model's covariance — hits a target
   (default 10%). Gross leverage is the OUTPUT of this, capped for safety.

4. HORIZON SLEEVES. The {5,10,21}-day forecasts are treated as three separate books
   — a short, medium and slow sleeve — each capped independently, then blended with
   tunable sleeve weights. This makes turnover and signal-decay explicit: the fast
   sleeve trades often, the slow sleeve barely moves, and their mix is a dial.

POINT-IN-TIME / LAYER DISCIPLINE
────────────────────────────────
Pure function of the forecast + risk model (both already as-of). May import data
(for the sector map) and signals; never streamlit/pages/app (import-linter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from portfolio import Allocator
from portfolio.risk import RiskModel

FORECAST_FIELD = "forecast"


# ── configuration ────────────────────────────────────────────────────────────
@dataclass
class AllocatorConfig:
    target_vol: float = 0.10          # annualised ex-ante portfolio vol target
    name_cap: float = 0.10            # max |weight| per instrument (fraction of gross)
    sector_cap: float = 0.35          # max gross exposure per sector
    max_leverage: float = 3.0         # cap on gross leverage after vol-targeting
    periods_per_year: float = 252.0
    dollar_neutral: bool = True       # demean weights (long-short relative value)
    sleeve_weights: Dict[int, float] = field(
        default_factory=lambda: {5: 1 / 3, 10: 1 / 3, 21: 1 / 3}
    )


@dataclass
class AllocationResult:
    """Target weights plus diagnostics for one allocation."""

    weights: pd.Series                       # final blended target weights
    sleeves: Dict[int, pd.Series]            # per-horizon capped, vol-targeted books
    gross_leverage: float
    net_exposure: float
    ex_ante_vol: float                       # annualised, of the final book
    n_names: int


# ── sector map ───────────────────────────────────────────────────────────────
def _default_sectors() -> pd.Series:
    """instrument display name -> sector, from the canonical universe registry."""
    try:
        from data.universe import INSTRUMENTS

        return pd.Series({name: inst.sector for name, inst in INSTRUMENTS.items()})
    except Exception:
        return pd.Series(dtype=object)


# ── core sizing helpers ──────────────────────────────────────────────────────
def _raw_risk_scaled(score: pd.Series, vol: pd.Series, dollar_neutral: bool) -> pd.Series:
    """Inverse-vol scaled, optionally dollar-neutral, gross-normalised raw weights."""
    common = score.dropna().index.intersection(vol.dropna().index)
    s = score.reindex(common).astype(float)
    v = vol.reindex(common).astype(float).replace(0.0, np.nan)
    w = (s / v).dropna()
    if w.empty:
        return w
    if dollar_neutral:
        w = w - w.mean()
    gross = w.abs().sum()
    return w / gross if gross > 0 else w


def _apply_caps(
    w: pd.Series,
    sectors: pd.Series,
    name_cap: float,
    sector_cap: float,
    dollar_neutral: bool = True,
    iters: int = 100,
) -> pd.Series:
    """
    Iterated water-filling: enforce per-name and per-sector gross caps (gross→1),
    re-imposing dollar-neutrality each pass when requested (centering preserves the
    zero-sum under the subsequent global vol-target scaling).
    """
    if w.empty:
        return w
    w = w / w.abs().sum()
    sec = sectors.reindex(w.index).fillna("unknown")
    for _ in range(iters):
        prev = w.copy()
        # per-name cap (in gross-normalised space)
        w = np.sign(w) * w.abs().clip(upper=name_cap)
        g = w.abs().sum()
        if g > 0:
            w = w / g
        # per-sector cap
        sec_gross = w.abs().groupby(sec).sum()
        over = sec_gross[sec_gross > sector_cap + 1e-12]
        for s_name, sg in over.items():
            mask = (sec == s_name)
            w[mask] = w[mask] * (sector_cap / sg)
        # re-impose dollar-neutrality (centering keeps sum=0 through later scaling)
        if dollar_neutral:
            w = w - w.mean()
        g = w.abs().sum()
        if g > 0:
            w = w / g
        if np.allclose(w.reindex(prev.index).fillna(0.0), prev, atol=1e-9):
            break
    return w


def _vol_target(
    w: pd.Series, risk_model: RiskModel, target_vol: float, ppy: float, max_leverage: float
) -> pd.Series:
    """Scale weights so ex-ante annualised vol == target_vol; cap gross leverage."""
    if w.empty:
        return w
    common = w.index.intersection(risk_model.cov.index)
    if len(common) == 0:
        return w * 0.0
    wv = w.reindex(common).fillna(0.0).to_numpy()
    cov = risk_model.cov.reindex(index=common, columns=common).to_numpy()
    var = float(wv @ cov @ wv)
    if var <= 0:
        return w * 0.0
    ann_vol = np.sqrt(var) * np.sqrt(ppy)
    scale = target_vol / ann_vol if ann_vol > 0 else 0.0
    out = w.reindex(common) * scale
    gross = out.abs().sum()
    if gross > max_leverage and gross > 0:
        out = out * (max_leverage / gross)
    return out


def _ex_ante_vol(w: pd.Series, risk_model: RiskModel, ppy: float) -> float:
    common = w.index.intersection(risk_model.cov.index)
    if len(common) == 0:
        return 0.0
    wv = w.reindex(common).fillna(0.0).to_numpy()
    cov = risk_model.cov.reindex(index=common, columns=common).to_numpy()
    var = float(wv @ cov @ wv)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(ppy))


# ── single-horizon allocator (implements the Phase-0 ABC) ────────────────────
class RiskScaledAllocator(Allocator):
    """Vol-targeted, risk-parity-aware, concentration-capped single-horizon allocator."""

    def __init__(self, config: Optional[AllocatorConfig] = None, sectors: Optional[pd.Series] = None):
        self.config = config or AllocatorConfig()
        self._sectors = sectors

    def sectors(self) -> pd.Series:
        if self._sectors is None:
            self._sectors = _default_sectors()
        return self._sectors

    def capped_book(self, forecasts: pd.Series, risk_model: RiskModel) -> pd.Series:
        """Capped, gross-normalised book WITHOUT vol-targeting (for sleeve blending)."""
        cfg = self.config
        raw = _raw_risk_scaled(forecasts, risk_model.vol, cfg.dollar_neutral)
        return _apply_caps(raw, self.sectors(), cfg.name_cap, cfg.sector_cap, cfg.dollar_neutral)

    def allocate(self, forecasts: pd.Series, risk_model: RiskModel) -> pd.Series:
        """Full single-horizon pipeline → vol-targeted target weights."""
        cfg = self.config
        book = self.capped_book(forecasts, risk_model)
        return _vol_target(book, risk_model, cfg.target_vol, cfg.periods_per_year, cfg.max_leverage)


# ── cardinality-constrained selection allocators (QAOA & its classical rival) ──
class _SelectAllocator(Allocator):
    """
    Base for long-only, cardinality-constrained mean-variance SELECTION allocators
    — the paradigm of the legacy QAOA optimizer (pick k assets minimising
    ``xᵀΣx − λ·μᵀx``, equal-weight the chosen, then vol-target).

    Subclasses differ ONLY in ``_select`` (which indices to hold): a classical exact
    solver vs the quantum QAOA approximation. Same μ, Σ, k and post-processing, so
    the backtest comparison is apples-to-apples.

    μ is built from the signal forecast as an expected DAILY return,
    ``μ_i = forecast_i · σ_i`` (Grinold: a +1σ score ⇒ +1 own-vol of expected
    return), putting μ in the same units as the daily covariance so λ trades off
    risk and return meaningfully.
    """

    def __init__(self, k: int = 5, lam: float = 2.0, n_universe: int = 12,
                 target_vol: float = 0.10, periods_per_year: float = 252.0, max_leverage: float = 3.0):
        self.k = int(k)
        self.lam = float(lam)
        self.n_universe = int(n_universe)
        self.target_vol = float(target_vol)
        self.periods_per_year = float(periods_per_year)
        self.max_leverage = float(max_leverage)

    def _candidates(self, forecasts: pd.Series, risk_model: RiskModel):
        common = forecasts.dropna().index.intersection(risk_model.vol.dropna().index)
        f = forecasts.reindex(common).astype(float)
        if f.empty:
            return [], None, None
        cand = f.nlargest(min(self.n_universe, len(f))).index.tolist()  # long candidates
        mu = (f.reindex(cand) * risk_model.vol.reindex(cand)).to_numpy()  # daily-return units
        cov = risk_model.cov.reindex(index=cand, columns=cand).to_numpy()
        return cand, mu, cov

    def _select(self, mu: np.ndarray, cov: np.ndarray) -> list:
        raise NotImplementedError

    def allocate(self, forecasts: pd.Series, risk_model: RiskModel) -> pd.Series:
        cand, mu, cov = self._candidates(forecasts, risk_model)
        if not cand or len(cand) < self.k:
            return pd.Series(dtype=float)
        idx = self._select(mu, cov)
        if not idx:
            return pd.Series(dtype=float)
        sel = [cand[i] for i in idx]
        w = pd.Series(1.0 / len(sel), index=sel)  # equal-weight long-only
        return _vol_target(w, risk_model, self.target_vol, self.periods_per_year, self.max_leverage)


class MeanVarianceSelectAllocator(_SelectAllocator):
    """Classical EXACT cardinality-constrained mean-variance selection (QAOA's rival).

    Enumerates all C(n, k) subsets and returns the global minimiser of
    ``xᵀΣx − λ·μᵀx``. At n≈12 this is ~hundreds of combinations — instant and exact,
    so it is the strict upper bound the QAOA approximation is measured against.
    """

    def _select(self, mu, cov):
        from itertools import combinations

        n = len(mu)
        best_obj, best = None, None
        for c in combinations(range(n), self.k):
            x = np.zeros(n)
            x[list(c)] = 1.0
            obj = float(x @ cov @ x - self.lam * (mu @ x))
            if best_obj is None or obj < best_obj:
                best_obj, best = obj, list(c)
        return best or []


class SingleHorizonFrameAllocator:
    """
    Adapt a single-horizon ``Allocator`` to the frame → ``AllocationResult``
    interface the backtest uses, by selecting one horizon's forecast column. Lets
    selection allocators (MV-select, QAOA) compete in the same backtest as the
    multi-horizon SleeveAllocator.
    """

    def __init__(self, base: Allocator, horizon: int = 21, periods_per_year: float = 252.0):
        self.base = base
        self.horizon = int(horizon)
        self.ppy = float(periods_per_year)

    def allocate(self, forecast_frame: pd.DataFrame, risk_model: RiskModel) -> AllocationResult:
        col = (self.horizon, FORECAST_FIELD)
        if col not in forecast_frame.columns:
            return AllocationResult(pd.Series(dtype=float), {}, 0.0, 0.0, 0.0, 0)
        f = forecast_frame[col].dropna()
        if f.empty:
            return AllocationResult(pd.Series(dtype=float), {}, 0.0, 0.0, 0.0, 0)
        w = self.base.allocate(f, risk_model)
        w = w[w != 0.0] if not w.empty else w
        return AllocationResult(
            weights=w,
            sleeves={self.horizon: w},
            gross_leverage=float(w.abs().sum()),
            net_exposure=float(w.sum()),
            ex_ante_vol=_ex_ante_vol(w, risk_model, self.ppy),
            n_names=int((w != 0.0).sum()),
        )


# ── multi-horizon sleeve allocator ───────────────────────────────────────────
class SleeveAllocator:
    """Blend per-horizon sleeve books into one vol-targeted portfolio."""

    def __init__(self, config: Optional[AllocatorConfig] = None, sectors: Optional[pd.Series] = None):
        self.config = config or AllocatorConfig()
        self.base = RiskScaledAllocator(self.config, sectors)

    def allocate(self, forecast_frame: pd.DataFrame, risk_model: RiskModel) -> AllocationResult:
        """
        Parameters
        ----------
        forecast_frame : DataFrame with MultiIndex columns (horizon, field), as
            produced by ``Signal.compute`` — index = instrument.
        risk_model : RiskModel as-of the same date.
        """
        cfg = self.config
        sleeves: Dict[int, pd.Series] = {}
        capped: Dict[int, pd.Series] = {}
        for h, sw in cfg.sleeve_weights.items():
            if sw == 0 or (h, FORECAST_FIELD) not in forecast_frame.columns:
                continue
            f = forecast_frame[(h, FORECAST_FIELD)].dropna()
            if f.empty:
                continue
            book = self.base.capped_book(f, risk_model)
            if book.empty:
                continue
            capped[h] = book
            # per-sleeve vol-targeted view (diagnostics / standalone use)
            sleeves[h] = _vol_target(book, risk_model, cfg.target_vol, cfg.periods_per_year, cfg.max_leverage)

        if not capped:
            empty = pd.Series(dtype=float)
            return AllocationResult(empty, {}, 0.0, 0.0, 0.0, 0)

        # Blend the capped (un-vol-targeted) books with sleeve weights. Blending can
        # shrink gross via cross-sleeve cancellation, which would inflate per-name /
        # per-sector fractions, so RE-APPLY the caps to the blend (the final book is
        # what must satisfy the constraints), then vol-target the blend once.
        all_names = sorted(set().union(*[b.index for b in capped.values()]))
        blended = pd.Series(0.0, index=all_names)
        wsum = sum(cfg.sleeve_weights[h] for h in capped)
        for h, book in capped.items():
            blended = blended.add((cfg.sleeve_weights[h] / wsum) * book.reindex(all_names).fillna(0.0), fill_value=0.0)

        blended = _apply_caps(blended, self.base.sectors(), cfg.name_cap, cfg.sector_cap, cfg.dollar_neutral)
        final = _vol_target(blended, risk_model, cfg.target_vol, cfg.periods_per_year, cfg.max_leverage)
        final = final[final != 0.0]

        return AllocationResult(
            weights=final,
            sleeves=sleeves,
            gross_leverage=float(final.abs().sum()),
            net_exposure=float(final.sum()),
            ex_ante_vol=_ex_ante_vol(final, risk_model, cfg.periods_per_year),
            n_names=int((final != 0.0).sum()),
        )
