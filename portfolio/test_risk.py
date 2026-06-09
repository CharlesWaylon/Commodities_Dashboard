"""
Tests for the Layer-3 risk model (portfolio/risk.py).

Covers the estimator's structural guarantees (symmetry, PSD, variance-preservation,
intensity bounds), the conditioning benefit of shrinkage, and the point-in-time
contract (future rows cannot change an as-of estimate).
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.risk import (
    RiskModel,
    estimate_risk_model,
    ledoit_wolf_constant_correlation,
    sample_covariance,
)


def _synth_prices(n_days=400, n_assets=12, seed=0):
    """Correlated geometric random-walk price panel."""
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.03, n_assets)
    c = 0.3 * np.ones((n_assets, n_assets))
    np.fill_diagonal(c, 1.0)
    cov = np.outer(vols, vols) * c
    L = np.linalg.cholesky(cov)
    rets = rng.standard_normal((n_days, n_assets)) @ L.T
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_estimate_returns_riskmodel_with_sane_structure():
    panel = _synth_prices()
    rm = estimate_risk_model(panel, panel.index[-1], lookback=252)
    assert isinstance(rm, RiskModel)
    C = rm.cov.values
    assert np.allclose(C, C.T), "covariance must be symmetric"
    assert np.linalg.eigvalsh(C).min() > -1e-12, "covariance must be PSD"
    assert 0.0 <= rm.shrinkage <= 1.0
    # vol is the sqrt of the diagonal
    assert np.allclose(rm.vol.values, np.sqrt(np.diag(C)))
    assert list(rm.cov.index) == list(rm.cov.columns) == list(rm.vol.index)


def test_lw_preserves_diagonal_variances():
    panel = _synth_prices(seed=1)
    rets = np.log(panel).diff().dropna()
    S = sample_covariance(rets, ddof=0)
    cov, delta = ledoit_wolf_constant_correlation(rets)
    # any convex combo of sample and the constant-corr target keeps the diagonal
    assert np.allclose(np.diag(cov.values), np.diag(S.values))
    assert 0.0 <= delta <= 1.0


def test_shrinkage_improves_conditioning_when_ill_posed():
    # N close to T -> sample covariance is ill-conditioned; shrinkage should help.
    panel = _synth_prices(n_days=80, n_assets=25, seed=2)
    rets = np.log(panel).diff().dropna()
    S = sample_covariance(rets, ddof=0)
    cov, delta = ledoit_wolf_constant_correlation(rets)
    assert delta > 0.0
    assert np.linalg.cond(cov.values) < np.linalg.cond(S.values)


def test_sample_method_has_zero_shrinkage():
    panel = _synth_prices(seed=3)
    rm = estimate_risk_model(panel, panel.index[-1], lookback=200, method="sample")
    assert rm.shrinkage == 0.0
    rets = np.log(panel.loc[: panel.index[-1]]).diff().iloc[-200:].dropna(axis=1, how="any")
    assert np.allclose(rm.cov.values, rets.cov(ddof=0).values)


def test_point_in_time_future_rows_do_not_change_estimate():
    panel = _synth_prices(n_days=400, seed=4)
    asof = panel.index[300]
    rm_truncated = estimate_risk_model(panel.loc[:asof], asof, lookback=150)
    rm_full = estimate_risk_model(panel, asof, lookback=150)
    assert rm_truncated is not None and rm_full is not None
    assert np.allclose(rm_truncated.cov.values, rm_full.cov.values)
    assert np.allclose(rm_truncated.vol.values, rm_full.vol.values)
    assert rm_truncated.shrinkage == rm_full.shrinkage


def test_insufficient_history_returns_none():
    panel = _synth_prices(n_days=30, seed=5)
    assert estimate_risk_model(panel, panel.index[-1], lookback=252, min_obs=60) is None


def test_correlation_has_unit_diagonal():
    panel = _synth_prices(seed=6)
    rm = estimate_risk_model(panel, panel.index[-1])
    corr = rm.correlation()
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-8)
    assert (corr.values <= 1.0 + 1e-8).all() and (corr.values >= -1.0 - 1e-8).all()


def test_annualized_vol_scales():
    panel = _synth_prices(seed=7)
    rm = estimate_risk_model(panel, panel.index[-1])
    assert np.allclose(rm.annualized_vol(252).values, rm.vol.values * np.sqrt(252))
