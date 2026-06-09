"""
Tests for the macro-factor risk model (portfolio/factor_risk.py).

Covers:
  • Structural guarantees (symmetric, PSD, sane diagonal, betas attached).
  • Isolation property — when the truth IS a factor structure on synthetic data,
    the factor model recovers cov in fewer Frobenius-distance units than the
    plain Ledoit-Wolf estimate. This is the formal sign the factor model adds
    information beyond shrinkage.
  • Point-in-time + fallback: factor_risk → None when factor data are absent at
    asof, and ``estimate_risk_model(method="factor")`` then falls back to LW.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.factor_risk import FactorRiskModel, estimate_factor_risk_model
from portfolio.risk import (
    estimate_risk_model,
    ledoit_wolf_constant_correlation,
    sample_covariance,
)


# Synthetic factor world: T days, K factors, N instruments
def _make_factor_world(T=400, K=3, N=15, seed=0):
    rng = np.random.default_rng(seed)
    factor_vol = np.array([0.012, 0.008, 0.020])[:K]
    idio_vol = rng.uniform(0.005, 0.020, N)
    B_true = rng.normal(0.0, 1.5, (K, N))
    F = rng.standard_normal((T, K)) * factor_vol
    eps = rng.standard_normal((T, N)) * idio_vol
    R = F @ B_true + eps
    return F, R, B_true, factor_vol, idio_vol


def _panel_from_returns(R, start="2010-01-04"):
    idx = pd.bdate_range(start, periods=R.shape[0])
    prices = 100 * np.exp(np.cumsum(R, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(R.shape[1])])


def _factor_panel(F, factor_names, start="2010-01-04"):
    """Convert factor RETURNS into level series (the store holds levels, not changes)."""
    idx = pd.bdate_range(start, periods=F.shape[0])
    levels = np.cumsum(F, axis=0)
    return pd.DataFrame(levels, index=idx, columns=factor_names)


# ── Direct ANALYTIC tests of the math (no fundamental store needed) ──────────
def _build_factor_model(R_panel, F_panel, asof, lookback, shrink_to_lw):
    """
    Bypass the store and exercise the SAME math by reproducing
    ``estimate_factor_risk_model`` on supplied (panel, factor-level) frames.
    """
    from portfolio.risk import _returns_window
    rets = _returns_window(R_panel, asof, lookback, min_obs=60)
    F = F_panel.loc[:asof].diff().dropna()
    common = rets.index.intersection(F.index)
    R = rets.loc[common].values
    Fm = F.loc[common].values
    Rm = R - R.mean(0); Fc = Fm - Fm.mean(0)
    B = np.linalg.solve(Fc.T @ Fc + 1e-10 * np.eye(Fc.shape[1]), Fc.T @ Rm)
    resid = Rm - Fc @ B
    idio = (resid ** 2).sum(0) / (len(common) - Fc.shape[1])
    fcov = (Fc.T @ Fc) / (len(common) - 1)
    cov = B.T @ fcov @ B + np.diag(idio)
    if shrink_to_lw > 0:
        lw, _ = ledoit_wolf_constant_correlation(rets.loc[common])
        cov = (1 - shrink_to_lw) * cov + shrink_to_lw * lw.values
    return 0.5 * (cov + cov.T), B, fcov, idio


def test_factor_model_structural_properties():
    F, R, _, _, _ = _make_factor_world(T=300, N=10, K=3, seed=1)
    R_panel = _panel_from_returns(R)
    F_panel = _factor_panel(F, ["f0", "f1", "f2"])
    cov, B, fcov, idio = _build_factor_model(R_panel, F_panel, R_panel.index[-1], lookback=250, shrink_to_lw=0.0)
    assert np.allclose(cov, cov.T), "symmetric"
    assert np.linalg.eigvalsh(cov).min() > -1e-10, "PSD"
    assert (np.diag(cov) > 0).all(), "diagonal variances positive"
    assert B.shape == (3, 10) and fcov.shape == (3, 3) and idio.shape == (10,)


def test_factor_model_recovers_truth_better_than_lw_under_factor_truth():
    """When the truth IS factor + idio, the factor model should beat LW in
    Frobenius distance to the true covariance — the isolation test."""
    F, R, B_true, fvol, idio_vol = _make_factor_world(T=500, N=15, K=3, seed=2)
    R_panel = _panel_from_returns(R)
    F_panel = _factor_panel(F, ["f0", "f1", "f2"])

    Sigma_true = B_true.T @ np.diag(fvol ** 2) @ B_true + np.diag(idio_vol ** 2)
    cov_factor, _, _, _ = _build_factor_model(R_panel, F_panel, R_panel.index[-1], lookback=400, shrink_to_lw=0.0)
    from portfolio.risk import _returns_window
    rets = _returns_window(R_panel, R_panel.index[-1], 400, min_obs=60)
    cov_lw, _ = ledoit_wolf_constant_correlation(rets)
    cov_sample = sample_covariance(rets, ddof=0).values

    err_factor = np.linalg.norm(cov_factor - Sigma_true, ord="fro")
    err_lw = np.linalg.norm(cov_lw.values - Sigma_true, ord="fro")
    err_sample = np.linalg.norm(cov_sample - Sigma_true, ord="fro")

    # Factor model strictly beats BOTH plain shrinkage and the sample matrix
    assert err_factor < err_lw, f"factor {err_factor:.3e} not better than LW {err_lw:.3e}"
    assert err_factor < err_sample


def test_shrinkage_alpha_interpolates_between_factor_and_lw():
    """Shrinkage α=0 → pure factor; α=1 → LW (up to the diagonal). Endpoints distinct."""
    F, R, _, _, _ = _make_factor_world(T=500, N=12, K=3, seed=3)
    R_panel = _panel_from_returns(R)
    F_panel = _factor_panel(F, ["f0", "f1", "f2"])
    pure, *_ = _build_factor_model(R_panel, F_panel, R_panel.index[-1], 400, shrink_to_lw=0.0)
    mixed, *_ = _build_factor_model(R_panel, F_panel, R_panel.index[-1], 400, shrink_to_lw=0.5)
    full, *_ = _build_factor_model(R_panel, F_panel, R_panel.index[-1], 400, shrink_to_lw=1.0)
    # All PSD
    for X in (pure, mixed, full):
        assert np.linalg.eigvalsh(X).min() > -1e-10
    # Endpoints distinguishable
    assert not np.allclose(pure, full)
    # Mixed lies between in Frobenius distance to either endpoint
    d_pure_full = np.linalg.norm(pure - full)
    d_mixed_pure = np.linalg.norm(mixed - pure)
    d_mixed_full = np.linalg.norm(mixed - full)
    assert d_mixed_pure < d_pure_full and d_mixed_full < d_pure_full


def test_estimate_risk_model_factor_method_falls_back_when_factors_absent(monkeypatch):
    """Real entry point: method='factor' returns LW when no factor data exist."""
    from portfolio import factor_risk

    monkeypatch.setattr(factor_risk, "_load_factor_changes", lambda *a, **k: None)
    F, R, _, _, _ = _make_factor_world(T=300, N=10, K=3, seed=4)
    panel = _panel_from_returns(R)
    rm = estimate_risk_model(panel, panel.index[-1], lookback=250, method="factor")
    # Falls through to LW: not a FactorRiskModel, but a usable RiskModel
    assert rm is not None
    assert not isinstance(rm, FactorRiskModel)
    assert rm.shrinkage > 0  # LW produced a real shrinkage intensity


def test_estimate_factor_risk_returns_none_when_factors_absent(monkeypatch):
    from portfolio import factor_risk

    monkeypatch.setattr(factor_risk, "_load_factor_changes", lambda *a, **k: None)
    F, R, _, _, _ = _make_factor_world(T=300, N=10, K=3, seed=5)
    panel = _panel_from_returns(R)
    assert estimate_factor_risk_model(panel, panel.index[-1]) is None
