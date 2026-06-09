"""
portfolio/factor_risk.py — macro-factor risk model (Layer 3).

Cashes in the verified ``signals.macro`` finding: macro-factor betas ARE
economically sound (gold short USD/rates; cyclicals long inflation/growth;
nat-gas macro-neutral) but they are NOT alpha — they belong in the RISK layer.
A factor risk model decomposes covariance as

    Σ = Bᵀ · Σ_f · B + D

where B is K×N betas to a few macro factors, Σ_f is the K×K factor covariance,
and D is the diagonal of idiosyncratic residual variances. This structure has two
well-known wins over a plain sample / Ledoit-Wolf estimate when the factor story
is true:
  • the cross-asset off-diagonal is denoised — most of it comes from K=4 shared
    factor exposures, not from 25·24/2 = 300 noisy pairwise correlations;
  • the cross-asset off-diagonal is INTERPRETABLE — "gold and silver co-move
    because they share a −USD beta", not "the sample says so".
(Connor-Korajczyk 1986; Ledoit-Wolf 2003 single-factor shrinkage; Fan-Liao-
Mincheva 2008 POET — the canonical high-N factor + sparse-residual story.)

The macro factor universe is the 4 daily, market-priced FRED series already
ingested: T10YIE (inflation), DGS10 (rates), DTWEXBGS (broad USD), VIXCLS (risk).
This is identical to ``signals/macro.py`` so the same betas the alpha test
verified flow into risk.

POINT-IN-TIME / FALLBACK
────────────────────────
Reads factors from ``data.fundamental_store`` filtered to ``release_date <=
asof``; daily market series so vintages are no-ops. Factor data starts ~2010-01,
so dates before that have no factor view → the estimator returns ``None`` and the
caller falls back to the Ledoit-Wolf model. No fake factors, no silent degradation.

LAYER DISCIPLINE
────────────────
numpy / pandas / data store only. No streamlit / pages / app (enforced by
.importlinter — note that portfolio MAY read from data, like the other
data-consuming layers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.risk import RiskModel, _returns_window, ledoit_wolf_constant_correlation

DEFAULT_FACTORS: Tuple[str, ...] = ("T10YIE", "DGS10", "DTWEXBGS", "VIXCLS")


@dataclass
class FactorRiskModel(RiskModel):
    """A ``RiskModel`` whose covariance is the structural Bᵀ Σ_f B + D decomposition.

    Adds the interpretable ingredients on top of the base ``RiskModel`` API:
      • ``betas`` (K×N)   — each instrument's factor loadings (a +1 unit move in
        factor f changes daily return by β_{f,i}).
      • ``factor_cov`` (K×K) — covariance of factor CHANGES (daily units).
      • ``idiosyncratic`` (N,) — residual variance left after factors explain.
      • ``r2`` (N,)       — fraction of each instrument's variance explained by
        the factor block (a diagnostic, not used by the allocator).
    """

    betas: pd.DataFrame = field(default_factory=pd.DataFrame)          # K × N
    factor_cov: pd.DataFrame = field(default_factory=pd.DataFrame)     # K × K
    idiosyncratic: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    r2: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    factors: Tuple[str, ...] = ()


def _load_factor_changes(asof: date, factors: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """PIT panel of factor CHANGES (daily diffs of levels), keyed by reference date."""
    from data import fundamental_store as store

    raw = store.load_raw(source="fred", series_ids=list(factors))
    if raw.empty:
        return None
    asof_ts = pd.Timestamp(asof)
    pit = raw[raw["release_date"] <= asof_ts]
    if pit.empty:
        return None
    pit = (
        pit.sort_values("release_date")
        .groupby(["series_id", "reference_date"], as_index=False)
        .tail(1)
    )
    levels = pit.pivot(index="reference_date", columns="series_id", values="value").sort_index()
    have = [c for c in factors if c in levels.columns]
    if len(have) < 2:
        return None
    return levels[have].diff().dropna(how="any")


def estimate_factor_risk_model(
    panel: pd.DataFrame,
    asof: date,
    factors: Tuple[str, ...] = DEFAULT_FACTORS,
    lookback: int = 252,
    min_obs: int = 60,
    shrink_to_lw: float = 0.5,
) -> Optional[FactorRiskModel]:
    """
    Estimate a macro-factor RiskModel as-of ``asof`` from ``panel`` and the PIT
    FRED factor series. Returns ``None`` when factor data are missing for the
    requested window — caller falls back to a non-factor model.

    ``shrink_to_lw`` ∈ [0, 1] blends the pure factor covariance with a Ledoit-Wolf
    sample covariance: ``Σ = (1-α) · Bᵀ Σ_f B + α · Σ_LW``. This is the standard
    Ledoit-Wolf 2003 / Fan-Liao-Mincheva (POET 2008) remedy for the (typical)
    case where a small factor block does not explain ALL co-movement: the factor
    block contributes the interpretable, shared-risk component; the LW block
    contributes residual co-movement structure the factors missed. ``α = 0`` is
    the pure factor model; ``α = 1`` is plain LW. Default 0.5 — equal weight, no
    in-sample tuning. Diagonal idiosyncratic noise is always preserved on top.
    """
    rets = _returns_window(panel, asof, lookback, min_obs)
    if rets.empty:
        return None

    f_chg = _load_factor_changes(asof, factors)
    if f_chg is None or f_chg.empty:
        return None

    # Align returns and factor changes on the trading days both have. Tiny ridge
    # on Fᵀ F for numerical stability (irrelevant at full rank, helps if a factor
    # is degenerate in a thin window).
    common = rets.index.intersection(f_chg.index)
    if len(common) < min_obs:
        return None
    R = rets.loc[common]
    F = f_chg.loc[common]
    K = F.shape[1]
    Fm = F.values - F.values.mean(axis=0, keepdims=True)
    Rm = R.values - R.values.mean(axis=0, keepdims=True)
    FtF = Fm.T @ Fm + 1e-10 * np.eye(K)
    B = np.linalg.solve(FtF, Fm.T @ Rm)        # K × N
    resid = Rm - Fm @ B                         # T × N
    idio = (resid ** 2).sum(axis=0) / (len(common) - K)
    factor_cov_arr = (Fm.T @ Fm) / (len(common) - 1)

    # Bᵀ Σ_f B  (N × N) + diag(idio): the pure factor decomposition.
    factor_part = B.T @ factor_cov_arr @ B
    cov_arr = factor_part + np.diag(idio)

    # Optional shrinkage toward a Ledoit-Wolf estimate (POET-style; the standard
    # remedy when factor R² is moderate, as it is here at ~10%).
    alpha = float(np.clip(shrink_to_lw, 0.0, 1.0))
    if alpha > 0.0:
        lw_cov, _ = ledoit_wolf_constant_correlation(R)
        cov_arr = (1.0 - alpha) * cov_arr + alpha * lw_cov.values
    cov_arr = 0.5 * (cov_arr + cov_arr.T)       # exact symmetry

    # R² per instrument (variance explained by factors / total).
    total_var = (Rm ** 2).sum(axis=0) / (len(common) - 1)
    explained = np.diag(B.T @ factor_cov_arr @ B)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total_var > 0, np.clip(explained / total_var, 0.0, 1.0), 0.0)

    instruments = list(R.columns)
    factor_names = list(F.columns)
    cov = pd.DataFrame(cov_arr, index=instruments, columns=instruments)
    vol = pd.Series(np.sqrt(np.diag(cov_arr)), index=instruments, name="vol")
    return FactorRiskModel(
        cov=cov,
        vol=vol,
        shrinkage=0.0,                          # structural model, not a shrinkage estimator
        asof=pd.Timestamp(asof),
        lookback=int(lookback),
        n_obs=int(len(common)),
        instruments=tuple(instruments),
        betas=pd.DataFrame(B, index=factor_names, columns=instruments),
        factor_cov=pd.DataFrame(factor_cov_arr, index=factor_names, columns=factor_names),
        idiosyncratic=pd.Series(idio, index=instruments),
        r2=pd.Series(r2, index=instruments),
        factors=tuple(factor_names),
    )
