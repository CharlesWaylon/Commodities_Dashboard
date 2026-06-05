"""
Tests for the allocator bake-off (portfolio/compete.py):
- MeanVarianceSelectAllocator is exact (it FINDS the global optimum the gate is
  measuring QAOA against),
- SingleHorizonFrameAllocator adapts a single-horizon Allocator into the backtest's
  frame interface,
- run_bakeoff runs all three (or a subset), produces a sorted table and a verdict,
- production_allocator implements the "ships only where it wins" policy
  (QAOA only when its flag is on AND it beat the best classical baseline).
"""

import itertools
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from portfolio.allocators import (
    AllocatorConfig,
    MeanVarianceSelectAllocator,
    SingleHorizonFrameAllocator,
)
from portfolio.backtest import BacktestConfig
from portfolio.compete import (
    Bakeoff,
    QUANTUM,
    production_allocator,
    run_bakeoff,
)
from portfolio.risk import RiskModel
from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD


HORIZONS = (5, 10, 21)


def _risk_model(n=8, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.01, 0.03, n)
    c = 0.25 * np.ones((n, n))
    np.fill_diagonal(c, 1.0)
    cov_arr = np.outer(vols, vols) * c
    cols = [f"A{i}" for i in range(n)]
    cov = pd.DataFrame(cov_arr, index=cols, columns=cols)
    vol = pd.Series(np.sqrt(np.diag(cov_arr)), index=cols)
    return RiskModel(cov=cov, vol=vol, shrinkage=0.0, instruments=tuple(cols))


def test_mv_select_returns_global_optimum():
    rm = _risk_model(n=8, seed=1)
    forecasts = pd.Series([2.5, 1.0, 0.7, 0.4, 0.3, 0.1, -0.2, -0.5], index=list(rm.vol.index))
    k = 3
    alloc = MeanVarianceSelectAllocator(k=k, lam=2.0, n_universe=6, target_vol=0.10)
    w = alloc.allocate(forecasts, rm)
    assert not w.empty and len(w) == k

    # reproduce the optimisation independently and confirm the SAME subset wins
    cand = forecasts.nlargest(6).index.tolist()
    mu = (forecasts.reindex(cand) * rm.vol.reindex(cand)).to_numpy()
    cov = rm.cov.reindex(index=cand, columns=cand).to_numpy()
    best, best_obj = None, None
    for c in itertools.combinations(range(len(cand)), k):
        x = np.zeros(len(cand)); x[list(c)] = 1.0
        obj = float(x @ cov @ x - 2.0 * (mu @ x))
        if best_obj is None or obj < best_obj:
            best_obj, best = obj, [cand[i] for i in c]
    assert set(w.index) == set(best)


def test_single_horizon_frame_adapter_selects_correct_horizon():
    rm = _risk_model(n=6, seed=2)
    instruments = list(rm.vol.index)
    cols = pd.MultiIndex.from_product([HORIZONS, [FORECAST_FIELD, CONFIDENCE_FIELD]],
                                       names=["horizon", "field"])
    frame = pd.DataFrame(index=pd.Index(instruments, name="instrument"), columns=cols, dtype=float)
    # Only horizon 21 has a usable view; others NaN
    frame[(21, FORECAST_FIELD)] = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    frame[(21, CONFIDENCE_FIELD)] = 0.5

    adapter = SingleHorizonFrameAllocator(MeanVarianceSelectAllocator(k=2, n_universe=4), horizon=21)
    res = adapter.allocate(frame, rm)
    assert res.n_names == 2 and not res.weights.empty
    # picking from the top of the H21 column → the highest-forecast names dominate
    assert "A0" in res.weights.index

    # asking for an absent horizon → empty
    adapter5 = SingleHorizonFrameAllocator(MeanVarianceSelectAllocator(k=2, n_universe=4), horizon=5)
    res5 = adapter5.allocate(frame, rm)
    assert res5.weights.empty and res5.n_names == 0


def _toy_panel(n_days=260, n_assets=8, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.01, 0.03, n_assets)
    c = 0.2 * np.ones((n_assets, n_assets)); np.fill_diagonal(c, 1.0)
    L = np.linalg.cholesky(np.outer(vols, vols) * c)
    rets = rng.standard_normal((n_days, n_assets)) @ L.T
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


class _TrendSignal:
    """20-day trend z-score; just a stand-in to drive run_backtest in tests."""
    name = "fake_trend"
    horizons = HORIZONS

    def compute(self, asof, panel):
        hist = panel.loc[:pd.Timestamp(asof)]
        if len(hist) < 21:
            return None
        trailing = np.log(hist).diff().iloc[-20:].sum()
        z = (trailing - trailing.mean()) / (trailing.std() or 1.0)
        cols = pd.MultiIndex.from_product([HORIZONS, [FORECAST_FIELD, CONFIDENCE_FIELD]],
                                          names=["horizon", "field"])
        out = pd.DataFrame(index=z.index, columns=cols, dtype=float)
        for h in HORIZONS:
            out[(h, FORECAST_FIELD)] = z
            out[(h, CONFIDENCE_FIELD)] = 0.5
        return out


def test_bakeoff_runs_classical_only_and_ranks():
    """The classical pair must run without quantum deps and be ranked head-to-head."""
    panel = _toy_panel(seed=3)
    bt = BacktestConfig(warmup=80, risk_lookback=80, rebalance_days=21,
                        allocator=AllocatorConfig(target_vol=0.10))
    bake = run_bakeoff(_TrendSignal(), panel, bt, horizon=21, k=3, n_universe=6,
                       include=["classical_mv", "risk_parity"])
    assert set(bake.results) == {"classical_mv", "risk_parity"}
    assert list(bake.table["allocator"])[0] == bake.winner
    # sorted by net Sharpe descending
    sharpes = bake.table["net_sharpe"].tolist()
    assert sharpes == sorted(sharpes, reverse=True)


def test_production_policy_blocks_qaoa_when_it_loses():
    """If QAOA underperforms, production_allocator MUST return the classical winner."""
    # synthesise a Bakeoff where QAOA loses
    mock_res = {
        "classical_mv": mock.Mock(sharpe=0.6),
        "risk_parity":  mock.Mock(sharpe=0.5),
        QUANTUM:        mock.Mock(sharpe=0.2),
    }
    bake = Bakeoff(results=mock_res, table=pd.DataFrame(), winner="classical_mv", quantum_wins=False)
    os.environ["QAOA_ALLOCATOR_ENABLED"] = "true"  # even with flag on, must not ship
    try:
        name, _ = production_allocator(bake, horizon=21, k=5, n_universe=12, target_vol=0.10)
    finally:
        os.environ.pop("QAOA_ALLOCATOR_ENABLED", None)
    assert name == "classical_mv"


def test_production_policy_promotes_qaoa_when_it_wins_and_flag_is_on():
    mock_res = {
        "classical_mv": mock.Mock(sharpe=0.4),
        "risk_parity":  mock.Mock(sharpe=0.3),
        QUANTUM:        mock.Mock(sharpe=0.7),
    }
    bake = Bakeoff(results=mock_res, table=pd.DataFrame(), winner="qaoa", quantum_wins=True)

    # flag OFF → still classical (flag governs production eligibility)
    os.environ.pop("QAOA_ALLOCATOR_ENABLED", None)
    name_off, _ = production_allocator(bake, horizon=21, k=5, n_universe=12, target_vol=0.10)
    assert name_off in {"classical_mv", "risk_parity"}

    # flag ON and quantum wins → ships
    os.environ["QAOA_ALLOCATOR_ENABLED"] = "true"
    try:
        name_on, _ = production_allocator(bake, horizon=21, k=5, n_universe=12, target_vol=0.10)
    finally:
        os.environ.pop("QAOA_ALLOCATOR_ENABLED", None)
    assert name_on == QUANTUM
