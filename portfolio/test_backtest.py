"""
Tests for the portfolio backtest (portfolio/backtest.py, step 3.3):
runs end-to-end, costs reduce returns monotonically, a flat signal yields a flat
book, and rebalance frequency drives total turnover.
"""

import numpy as np
import pandas as pd

from portfolio.backtest import BacktestConfig, run_backtest
from portfolio.allocators import AllocatorConfig
from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD

HORIZONS = (5, 10, 21)


def _panel(n_days=500, n_assets=10, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.01, 0.03, n_assets)
    c = 0.2 * np.ones((n_assets, n_assets))
    np.fill_diagonal(c, 1.0)
    L = np.linalg.cholesky(np.outer(vols, vols) * c)
    rets = rng.standard_normal((n_days, n_assets)) @ L.T
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


class _TrendSignal:
    """Minimal Signal-like: cross-sectional z-score of the trailing 20-day return."""

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
        out.index.name = "instrument"
        for h in HORIZONS:
            out[(h, FORECAST_FIELD)] = z
            out[(h, CONFIDENCE_FIELD)] = 0.5
        return out


class _FlatSignal:
    """Genuine no-view: all-NaN forecasts (a constant forecast is NOT no-view under
    inverse-vol sizing — const/σ still varies — so 'no view' must be empty)."""

    name = "fake_flat"
    horizons = HORIZONS

    def compute(self, asof, panel):
        cols = pd.MultiIndex.from_product([HORIZONS, [FORECAST_FIELD, CONFIDENCE_FIELD]],
                                          names=["horizon", "field"])
        out = pd.DataFrame(np.nan, index=panel.columns, columns=cols, dtype=float)
        out.index.name = "instrument"
        return out


def _cfg(**kw):
    base = dict(warmup=80, risk_lookback=80, rebalance_days=21, cost_bps=10.0)
    base.update(kw)
    alloc = base.pop("allocator", AllocatorConfig())
    return BacktestConfig(allocator=alloc, **base)


def test_backtest_runs_and_reports_finite_metrics():
    panel = _panel()
    res = run_backtest(_TrendSignal(), panel, _cfg())
    assert len(res.equity) > 0
    assert res.equity.iloc[0] > 0
    assert np.isfinite(res.sharpe)
    assert np.isfinite(res.max_drawdown) and res.max_drawdown <= 0
    assert res.n_rebalances > 0


def test_higher_cost_reduces_net_return():
    panel = _panel(seed=1)
    cheap = run_backtest(_TrendSignal(), panel, _cfg(cost_bps=1.0))
    dear = run_backtest(_TrendSignal(), panel, _cfg(cost_bps=100.0))
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert dear.sharpe < cheap.sharpe


def test_flat_signal_gives_flat_book():
    panel = _panel(seed=2)
    res = run_backtest(_FlatSignal(), panel, _cfg())
    # no cross-sectional view => no positions => equity stays at 1.0
    assert res.n_rebalances == 0
    assert np.allclose(res.equity.values, 1.0)


def test_more_frequent_rebalancing_raises_total_turnover():
    panel = _panel(seed=3)
    slow = run_backtest(_TrendSignal(), panel, _cfg(rebalance_days=60))
    fast = run_backtest(_TrendSignal(), panel, _cfg(rebalance_days=5))
    assert fast.turnover.sum() > slow.turnover.sum()
    assert fast.n_rebalances > slow.n_rebalances


def test_realized_vol_in_reasonable_range_of_target():
    panel = _panel(seed=4)
    res = run_backtest(_TrendSignal(), panel, _cfg(allocator=AllocatorConfig(target_vol=0.10)))
    # ex-ante targeting won't be exact out-of-sample, but should be the right order
    assert 0.03 < res.realized_vol < 0.30
