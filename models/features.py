"""
Feature engineering for commodity price models.

Takes a raw price matrix (output of data_loader.load_price_matrix) and
produces a feature matrix X and target vector y ready for model training.

Why these features?
  - Log returns:     stationary (unlike raw prices), comparable across commodities
  - Rolling vol:     captures regime changes — high-vol markets behave differently
  - Momentum:        short-term trend signal with academic backing (cross-sectional)
  - Z-score:         normalises each series relative to its recent history,
                     removing slow-moving drift while preserving mean-reversion signals
  - Cross-corr:      pairwise rolling correlation captures how commodity relationships
                     shift during macro events (e.g. oil-gold decoupling in crises)

The quantum circuits expect exactly N_QUBITS features per sample, so
build_quantum_features() performs a final selection + normalisation step.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.config import (
    DEFAULT_TARGET,
    RETURN_LAGS,
    ROLLING_VOL_WINDOW,
    ROLLING_MOM_WINDOW,
    ZSCORE_WINDOW,
    N_QUBITS,
)


# ── Core feature builders ──────────────────────────────────────────────────────

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def rolling_volatility(returns: pd.DataFrame, window: int = ROLLING_VOL_WINDOW) -> pd.DataFrame:
    """Rolling annualised volatility (std dev of log-returns × √252)."""
    return returns.rolling(window).std() * np.sqrt(252)


def rolling_momentum(returns: pd.DataFrame, window: int = ROLLING_MOM_WINDOW) -> pd.DataFrame:
    """Cumulative log-return over the past `window` days."""
    return returns.rolling(window).sum()


def rolling_zscore(prices: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """
    (P_t - rolling_mean) / rolling_std over `window` days.
    Captures how far the current price sits from its recent norm.
    """
    mu  = prices.rolling(window).mean()
    sig = prices.rolling(window).std()
    return (prices - mu) / sig


# ── Full feature matrix ────────────────────────────────────────────────────────

def build_feature_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a wide feature DataFrame from a price matrix.

    Input
    -----
    prices : pd.DataFrame
        Shape (n_days, n_commodities). Columns are display names.

    Output
    ------
    pd.DataFrame
        One row per trading day. Columns:
          {commodity}_ret_{lag}d    — lagged log-return
          {commodity}_vol21         — rolling 21-day realised volatility
          {commodity}_mom10         — rolling 10-day cumulative return
          {commodity}_zscore        — rolling 63-day z-score
    """
    ret = log_returns(prices)
    vol = rolling_volatility(ret)
    mom = rolling_momentum(ret)
    zsc = rolling_zscore(prices)

    parts = []

    # Lagged returns (1d, 2d, 5d) — these become predictive signals
    for lag in RETURN_LAGS:
        lagged = ret.shift(lag)
        lagged.columns = [f"{c}_ret_{lag}d" for c in ret.columns]
        parts.append(lagged)

    vol.columns = [f"{c}_vol21" for c in vol.columns]
    mom.columns = [f"{c}_mom10" for c in mom.columns]
    zsc.columns = [f"{c}_zscore" for c in zsc.columns]

    parts += [vol, mom, zsc]

    feature_df = pd.concat(parts, axis=1)
    return feature_df


def build_target(prices: pd.DataFrame, target_name: str = DEFAULT_TARGET) -> pd.Series:
    """
    Next-day log-return of the target commodity.

    We shift by -1 so that each row in X aligns with *tomorrow's* return —
    the value we are trying to forecast.
    """
    ret = log_returns(prices)
    return ret[target_name].shift(-1).rename("target")


# ── Quantum-ready feature set ─────────────────────────────────────────────────

def build_quantum_features(
    prices: pd.DataFrame,
    target_name: str = DEFAULT_TARGET,
    n_features: int = N_QUBITS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a compact (n_samples, n_features) feature matrix scaled to [-π, π]
    for angle encoding into quantum circuits, paired with a target vector.

    Why angle encoding?
        RY(θ) and RZ(θ²) gates expect angles in radians. Feeding raw financial
        numbers (e.g. returns of 0.01, vol of 0.25) maps to very small rotations
        and under-utilises the Hilbert space. Scaling to [-π, π] uses the full
        range of each qubit's rotation.

    Steps
    1. Build full feature matrix
    2. Drop NaN rows (rolling windows need a warm-up period)
    3. Align X and y to same index, drop rows where y is NaN
    4. Select the `n_features` highest-variance columns (data-driven reduction)
    5. StandardScaler → scale to [-π, π]

    Returns
    -------
    (X, y) — both numpy arrays, aligned row-by-row.
    """
    feat_df = build_feature_matrix(prices)
    y_series = build_target(prices, target_name)

    # Align on common index, drop any NaN in either
    combined = pd.concat([feat_df, y_series], axis=1).dropna()
    X_raw = combined.drop(columns=["target"])
    y = combined["target"].values

    # Select top-N by variance (avoids arbitrary column picks)
    variances = X_raw.var()
    top_cols = variances.nlargest(n_features).index
    X_selected = X_raw[top_cols].values

    # Scale to [-π, π] for quantum angle encoding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_angle  = np.clip(X_scaled * np.pi, -np.pi, np.pi)

    return X_angle, y
