"""
portfolio/risk.py — the Layer-3 risk model (covariance + volatility).

Every allocator needs a risk model: a covariance matrix to control portfolio risk
and a per-instrument volatility vector to size positions. A raw SAMPLE covariance
is a poor estimator when the number of instruments (N) is not small relative to the
window length (T) — it is ill-conditioned and its extreme eigenvalues are badly
biased, which mean-variance optimisers then exploit, producing unstable,
error-maximising weights.

The fix is SHRINKAGE: pull the noisy sample covariance toward a structured target.
This module implements the Ledoit-Wolf (2004, "Honey, I Shrunk the Sample
Covariance Matrix", J. Portfolio Management) CONSTANT-CORRELATION estimator — the
finance-standard choice — with the analytically optimal shrinkage intensity (no
cross-validation, no free parameter). The result is well-conditioned, PSD, and
closer in expectation to the true covariance than either the sample matrix or the
target alone.

POINT-IN-TIME
─────────────
``estimate_risk_model(panel, asof, ...)`` uses only ``panel.loc[:asof]`` — never a
row after the decision date. Appending future rows cannot change the estimate.

LAYER DISCIPLINE
────────────────
numpy/pandas only. No streamlit / pages / app imports (enforced by .importlinter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskModel:
    """A point-in-time risk model: covariance + volatility for a cross-section.

    Attributes
    ----------
    cov : pd.DataFrame
        N×N covariance of per-period (daily) returns, instruments on both axes.
        Symmetric, PSD, Ledoit-Wolf shrunk.
    vol : pd.Series
        Per-instrument volatility (sqrt of the cov diagonal), same period as cov.
    shrinkage : float
        Applied Ledoit-Wolf intensity δ ∈ [0, 1] (0 = pure sample, 1 = pure target).
    asof, lookback, n_obs : metadata for audit.
    """

    cov: pd.DataFrame
    vol: pd.Series
    shrinkage: float
    asof: Optional[pd.Timestamp] = None
    lookback: int = 0
    n_obs: int = 0
    instruments: Tuple[str, ...] = field(default_factory=tuple)

    def annualized_vol(self, periods_per_year: float = 252.0) -> pd.Series:
        """Volatility scaled to annual units (cov/vol are stored per-period)."""
        return self.vol * np.sqrt(periods_per_year)

    def correlation(self) -> pd.DataFrame:
        """Correlation matrix implied by the shrunk covariance."""
        d = np.sqrt(np.diag(self.cov.values))
        denom = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, self.cov.values / denom, 0.0)
        return pd.DataFrame(corr, index=self.cov.index, columns=self.cov.columns)


def sample_covariance(returns: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    """Plain sample covariance (ddof=0 matches the Ledoit-Wolf 1/T convention)."""
    return returns.cov(ddof=ddof)


def ledoit_wolf_constant_correlation(returns: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf (2004) shrinkage toward the constant-correlation target.

    Parameters
    ----------
    returns : pd.DataFrame
        T×N matrix of (already windowed) returns; columns = instruments.

    Returns
    -------
    (cov_shrunk, delta) : (pd.DataFrame, float)
        Shrunk covariance and the applied intensity δ ∈ [0, 1].

    Notes
    -----
    Implements the estimator of Ledoit & Wolf, "Honey, I Shrunk the Sample
    Covariance Matrix" (2004): target F has the sample variances on the diagonal
    and the AVERAGE sample correlation imposed on every off-diagonal; the optimal
    intensity is δ* = (π − ρ) / γ, clipped to [0, 1] after dividing by T. Variable
    names follow the paper.
    """
    cols = list(returns.columns)
    X = returns.to_numpy(dtype=float)
    T, N = X.shape
    if T < 2 or N < 1:
        S = returns.cov(ddof=0)
        return S, 0.0
    if N == 1:
        S = returns.cov(ddof=0)
        return S, 0.0

    # Demean; sample covariance with the 1/T convention used throughout the paper.
    Xm = X - X.mean(axis=0, keepdims=True)
    S = (Xm.T @ Xm) / T
    var = np.diag(S)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_std > 0, S / outer_std, 0.0)

    # Average off-diagonal correlation (rbar) and the constant-correlation target F.
    if N > 1:
        rbar = (corr.sum() - N) / (N * (N - 1))
    else:
        rbar = 0.0
    F = rbar * outer_std
    np.fill_diagonal(F, var)

    # π : sum of asymptotic variances of the sample covariance entries.
    Xm2 = Xm ** 2
    pi_mat = (Xm2.T @ Xm2) / T - S ** 2
    pi_hat = pi_mat.sum()

    # ρ : asymptotic covariances between the target and the sample estimates.
    #   diagonal part = Σ π_ii ; off-diagonal part uses the θ terms below.
    rho_diag = np.diag(pi_mat).sum()
    # Building block: cube[i,j] = (1/T) Σ_t x_it^3 x_jt.
    cube = (Xm ** 3).T @ Xm / T
    # θ_ii,ij = (1/T)Σ(x_i^2 - s_ii)(x_i x_j - s_ij) = cube[i,j] - var_i * s_ij.
    theta_ii_ij = cube - var[:, None] * S
    # θ_jj,ij = (1/T)Σ(x_j^2 - s_jj)(x_i x_j - s_ij) = (1/T)Σ x_i x_j^3 - var_j*s_ij;
    # note (1/T)Σ x_i x_j^3 == cube.T[i,j].
    theta_jj_ij = cube.T - var[None, :] * S
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ji = np.where(outer_std > 0, np.outer(1.0 / np.where(std > 0, std, np.nan), std), 0.0)
        ratio_ij = np.where(outer_std > 0, np.outer(std, 1.0 / np.where(std > 0, std, np.nan)), 0.0)
    rho_off = (rbar * (ratio_ji * theta_ii_ij + ratio_ij * theta_jj_ij)).copy()
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = rho_diag + np.nansum(rho_off)

    # γ : Frobenius distance between target and sample covariance.
    gamma_hat = float(((F - S) ** 2).sum())

    # Optimal shrinkage intensity.
    if gamma_hat <= 0:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / T, 0.0, 1.0))

    cov = delta * F + (1.0 - delta) * S
    cov = 0.5 * (cov + cov.T)  # enforce exact symmetry
    return pd.DataFrame(cov, index=cols, columns=cols), delta


def _returns_window(
    panel: pd.DataFrame, asof: date, lookback: int, min_obs: int
) -> pd.DataFrame:
    """PIT log-return window: rows <= asof, last ``lookback`` returns, complete columns."""
    hist = panel.loc[: pd.Timestamp(asof)]
    if len(hist) < min_obs + 1:
        return pd.DataFrame()
    rets = np.log(hist).diff().iloc[-lookback:]
    # keep instruments with a full window (covariance needs aligned, complete data)
    rets = rets.dropna(axis=1, how="any")
    if rets.shape[0] < min_obs or rets.shape[1] < 1:
        return pd.DataFrame()
    return rets


def estimate_risk_model(
    panel: pd.DataFrame,
    asof: date,
    lookback: int = 252,
    min_obs: int = 60,
    method: str = "lw_cc",
) -> Optional[RiskModel]:
    """
    Estimate a point-in-time RiskModel from a wide price ``panel`` as of ``asof``.

    method = "lw_cc"  -> Ledoit-Wolf constant-correlation shrinkage (default).
    method = "sample" -> plain sample covariance (baseline / comparison).
    method = "factor" -> macro-factor structural model Bᵀ Σ_f B + D; falls back
                        to Ledoit-Wolf when factor data are unavailable at asof
                        (e.g. dates before FRED ingest coverage starts).

    Returns None if there is insufficient history for a usable estimate.
    """
    if method == "factor":
        from portfolio.factor_risk import estimate_factor_risk_model

        fm = estimate_factor_risk_model(panel, asof, lookback=lookback, min_obs=min_obs)
        if fm is not None:
            return fm
        # Fall through to LW so the backtest stays continuous through pre-2010.
        method = "lw_cc"

    rets = _returns_window(panel, asof, lookback, min_obs)
    if rets.empty:
        return None

    if method == "sample":
        cov = sample_covariance(rets, ddof=0)
        delta = 0.0
    elif method == "lw_cc":
        cov, delta = ledoit_wolf_constant_correlation(rets)
    else:
        raise ValueError(f"unknown risk method {method!r} (use 'lw_cc' / 'sample' / 'factor')")

    vol = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index, name="vol")
    return RiskModel(
        cov=cov,
        vol=vol,
        shrinkage=delta,
        asof=pd.Timestamp(asof),
        lookback=int(lookback),
        n_obs=int(rets.shape[0]),
        instruments=tuple(cov.index),
    )
