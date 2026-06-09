"""
Tests for the forecast → position allocator (portfolio/allocators.py, step 3.2):
inverse-vol sizing, dollar-neutrality, name + sector caps, portfolio-vol targeting,
and horizon-sleeve blending.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.allocators import (
    AllocatorConfig,
    RiskScaledAllocator,
    SleeveAllocator,
)
from portfolio.risk import estimate_risk_model

HORIZONS = (5, 10, 21)
FORECAST = "forecast"


def _risk_model(n_assets=12, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.03, n_assets)
    c = 0.25 * np.ones((n_assets, n_assets))
    np.fill_diagonal(c, 1.0)
    cov = np.outer(vols, vols) * c
    L = np.linalg.cholesky(cov)
    rets = rng.standard_normal((400, n_assets)) @ L.T
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2015-01-01", periods=400)
    panel = pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])
    return estimate_risk_model(panel, panel.index[-1], lookback=252)


def _forecast_frame(instruments, seed=1):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product([HORIZONS, [FORECAST]], names=["horizon", "field"])
    out = pd.DataFrame(index=pd.Index(instruments, name="instrument"), columns=cols, dtype=float)
    for h in HORIZONS:
        s = pd.Series(rng.standard_normal(len(instruments)), index=instruments)
        out[(h, FORECAST)] = (s - s.mean()) / s.std()
    return out


def _sectors(instruments):
    # round-robin a few sectors so sector caps are exercised
    names = ["energy", "metals", "agriculture", "livestock"]
    return pd.Series({a: names[i % len(names)] for i, a in enumerate(instruments)})


def test_sleeve_allocation_respects_all_constraints():
    rm = _risk_model()
    instruments = list(rm.vol.index)
    fc = _forecast_frame(instruments)
    sectors = _sectors(instruments)
    cfg = AllocatorConfig(target_vol=0.10, name_cap=0.10, sector_cap=0.35)
    res = SleeveAllocator(cfg, sectors=sectors).allocate(fc, rm)

    w = res.weights
    assert not w.empty
    # dollar-neutral
    assert abs(res.net_exposure) < 1e-6
    # name cap (as fraction of gross)
    assert (w.abs() / res.gross_leverage).max() <= cfg.name_cap + 1e-6
    # sector cap
    sec_frac = w.abs().groupby(sectors.reindex(w.index)).sum() / res.gross_leverage
    assert (sec_frac <= cfg.sector_cap + 1e-6).all()
    # portfolio vol target hit
    assert res.ex_ante_vol == pytest.approx(cfg.target_vol, abs=1e-3)
    # all three sleeves produced
    assert set(res.sleeves.keys()) == set(HORIZONS)


def test_inverse_vol_sizing_favours_low_vol_names():
    # equal-magnitude scores, different vols → lower-vol name gets larger weight.
    instruments = ["L1", "H1", "L2", "H2"]
    cov = pd.DataFrame(np.diag([0.01**2, 0.04**2, 0.01**2, 0.04**2]),
                       index=instruments, columns=instruments)
    from portfolio.risk import RiskModel
    rm = RiskModel(cov=cov, vol=pd.Series(np.sqrt(np.diag(cov)), index=instruments),
                   shrinkage=0.0, instruments=tuple(instruments))
    score = pd.Series([1.0, 1.0, -1.0, -1.0], index=instruments)
    cfg = AllocatorConfig(name_cap=1.0, sector_cap=1.0)  # caps off
    alloc = RiskScaledAllocator(cfg, sectors=pd.Series({a: "x" for a in instruments}))
    book = alloc.capped_book(score, rm)
    assert abs(book["L1"]) > abs(book["H1"])   # same +score, L1 lower vol → bigger
    assert abs(book["L2"]) > abs(book["H2"])   # same -score, L2 lower vol → bigger


def test_name_cap_binds_when_tight():
    rm = _risk_model(n_assets=8, seed=3)
    instruments = list(rm.vol.index)
    fc = _forecast_frame(instruments, seed=4)
    cfg = AllocatorConfig(name_cap=0.20, sector_cap=1.0)
    res = SleeveAllocator(cfg, sectors=pd.Series({a: "x" for a in instruments})).allocate(fc, rm)
    assert (res.weights.abs() / res.gross_leverage).max() <= cfg.name_cap + 1e-6


def test_sleeve_weights_change_the_blend():
    rm = _risk_model(seed=5)
    instruments = list(rm.vol.index)
    fc = _forecast_frame(instruments, seed=6)
    sectors = _sectors(instruments)
    only_fast = SleeveAllocator(
        AllocatorConfig(sleeve_weights={5: 1.0, 10: 0.0, 21: 0.0}), sectors=sectors
    ).allocate(fc, rm).weights
    only_slow = SleeveAllocator(
        AllocatorConfig(sleeve_weights={5: 0.0, 10: 0.0, 21: 1.0}), sectors=sectors
    ).allocate(fc, rm).weights
    common = only_fast.index.intersection(only_slow.index)
    # different horizon forecasts → different books
    assert not np.allclose(only_fast.reindex(common), only_slow.reindex(common), atol=1e-6)


def test_vol_target_scaling_is_proportional():
    rm = _risk_model(seed=7)
    instruments = list(rm.vol.index)
    fc = _forecast_frame(instruments, seed=8)
    sectors = _sectors(instruments)
    lo = SleeveAllocator(AllocatorConfig(target_vol=0.05), sectors=sectors).allocate(fc, rm)
    hi = SleeveAllocator(AllocatorConfig(target_vol=0.20), sectors=sectors).allocate(fc, rm)
    assert lo.ex_ante_vol == pytest.approx(0.05, abs=1e-3)
    assert hi.ex_ante_vol == pytest.approx(0.20, abs=1e-3)
    # 4x vol target → ~4x gross leverage (same direction, scaled)
    assert hi.gross_leverage == pytest.approx(4 * lo.gross_leverage, rel=0.05)


def test_leverage_cap_binds():
    rm = _risk_model(seed=9)
    instruments = list(rm.vol.index)
    fc = _forecast_frame(instruments, seed=10)
    sectors = _sectors(instruments)
    # absurd vol target would demand huge leverage → capped
    cfg = AllocatorConfig(target_vol=5.0, max_leverage=2.0)
    res = SleeveAllocator(cfg, sectors=sectors).allocate(fc, rm)
    assert res.gross_leverage <= 2.0 + 1e-6


def test_empty_forecast_returns_empty():
    rm = _risk_model(seed=11)
    cols = pd.MultiIndex.from_product([HORIZONS, [FORECAST]], names=["horizon", "field"])
    empty = pd.DataFrame(index=pd.Index(list(rm.vol.index), name="instrument"), columns=cols, dtype=float)
    res = SleeveAllocator().allocate(empty, rm)
    assert res.weights.empty and res.n_names == 0
