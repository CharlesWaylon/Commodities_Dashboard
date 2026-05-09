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

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.config import (
    COMMODITY_SECTORS,
    DEFAULT_TARGET,
    RETURN_LAGS,
    ROLLING_VOL_WINDOW,
    ROLLING_MOM_WINDOW,
    ZSCORE_WINDOW,
    N_QUBITS,
)

# Commodities that receive PDSI (Corn Belt drought) features during training.
PDSI_COMMODITIES: frozenset = frozenset([
    "Corn (CBOT)",
    "Soybeans (CBOT)",
    "Wheat (CBOT SRW)",
])

# Commodities that receive ENSO (MEI / phase) features during training.
# Corn Belt grains benefit from both PDSI and ENSO; soft commodities get ENSO only.
ENSO_COMMODITIES: frozenset = frozenset([
    "Corn (CBOT)",
    "Soybeans (CBOT)",
    "Wheat (CBOT SRW)",
    "Coffee (Arabica)",
    "Cocoa (ICE)",
    "Sugar (Raw #11)",
])

# ── Sector helpers ─────────────────────────────────────────────────────────────

# Inverted sector map: sector_name → frozenset of display-name commodity strings
_SECTOR_MEMBERS: dict[str, frozenset] = {}
for _c, _s in COMMODITY_SECTORS.items():
    _SECTOR_MEMBERS.setdefault(_s, set()).add(_c)
_SECTOR_MEMBERS = {k: frozenset(v) for k, v in _SECTOR_MEMBERS.items()}

# Keyword patterns for external signal columns (columns in prices that are NOT
# commodity prices). Matching is case-insensitive substring.
_ALWAYS_KEEP_SIGNALS = ("dxy", "vix", "tlt")          # macro — all sectors
_AGRICULTURE_SIGNALS = ("pdsi", "mei", "enso")         # climate — agriculture only
_ENERGY_SIGNALS      = ("slope", "spread", "curve")    # futures curve — energy only


def _filter_prices_to_sector(prices: pd.DataFrame, sector: str) -> pd.DataFrame:
    """
    Split prices columns into commodity columns and external-signal columns,
    then return only the sector's commodity columns plus permitted signal columns.

    Commodity columns are those present in COMMODITY_SECTORS (known display names).
    Unknown columns are treated as external signals and filtered by sector rules.
    """
    known_commodities = set(COMMODITY_SECTORS.keys())
    sector_members    = _SECTOR_MEMBERS.get(sector, frozenset())

    commodity_cols = [c for c in prices.columns if c in known_commodities]
    external_cols  = [c for c in prices.columns if c not in known_commodities]

    # Sector commodity columns that actually exist in prices
    keep_commodity = [c for c in commodity_cols if c in sector_members]

    # External signal filtering
    keep_external: list[str] = []
    for col in external_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in _ALWAYS_KEEP_SIGNALS):
            keep_external.append(col)
        elif sector == "agriculture" and any(kw in col_lower for kw in _AGRICULTURE_SIGNALS):
            keep_external.append(col)
        elif sector == "energy" and any(kw in col_lower for kw in _ENERGY_SIGNALS):
            keep_external.append(col)
        # Unknown external signals not matching any rule are silently dropped
        # when a sector is active — avoids adding spurious features.

    keep = keep_commodity + keep_external
    if not keep:
        raise ValueError(
            f"build_feature_matrix: sector '{sector}' matched no columns in prices. "
            f"Available columns: {list(prices.columns[:10])}…"
        )
    return prices[keep]


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

def build_feature_matrix(
    prices: pd.DataFrame,
    sector: str | None = None,
) -> pd.DataFrame:
    """
    Construct a wide feature DataFrame from a price matrix.

    Parameters
    ----------
    prices : pd.DataFrame
        Shape (n_days, n_commodities[+external_signals]). Columns are display
        names drawn from COMMODITY_SECTORS, plus any macro/climate columns the
        caller has merged in (e.g. dxy_ret, vix_ret5d, pdsi_cornbelt).

    sector : str or None, optional
        When supplied, restricts features to the commodities in that sector
        (one of 'energy', 'metals', 'agriculture', 'livestock', 'digital').
        External-signal columns are also filtered by sector rules:
          • All sectors  : any column whose name contains 'dxy', 'vix', or 'tlt'
          • agriculture  : additionally keeps 'pdsi', 'mei', 'enso' columns
          • energy       : additionally keeps 'slope', 'spread', 'curve' columns
        When None (default), all columns are used — identical to the original
        behaviour, so all existing call sites remain unaffected.

    Output
    ------
    pd.DataFrame
        One row per trading day. Columns:
          {commodity}_ret_{lag}d    — lagged log-return (lags from RETURN_LAGS)
          {commodity}_vol21         — rolling 21-day realised volatility
          {commodity}_mom10         — rolling 10-day cumulative return
          {commodity}_zscore        — rolling 63-day z-score
        External-signal columns that survive filtering are left unchanged
        (no renaming) and appended at the end.

    Raises
    ------
    ValueError
        If *sector* is specified but matches no columns in *prices*.
    """
    if sector is not None:
        prices = _filter_prices_to_sector(prices, sector)

    # Separate commodity columns from any external signals that survived filtering.
    # External signals get passed through as-is; commodity columns get the full
    # lag/vol/mom/zscore treatment.
    known_commodities = set(COMMODITY_SECTORS.keys())
    commodity_cols = [c for c in prices.columns if c in known_commodities]
    external_cols  = [c for c in prices.columns if c not in known_commodities]

    price_data = prices[commodity_cols] if commodity_cols else prices

    ret = log_returns(price_data)
    vol = rolling_volatility(ret)
    mom = rolling_momentum(ret)
    zsc = rolling_zscore(price_data)

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

    # Append any external signal columns (DXY, VIX, TLT, PDSI, etc.) unchanged.
    # They were already filtered to sector-appropriate signals above.
    if external_cols:
        parts.append(prices[external_cols])

    feature_df = pd.concat(parts, axis=1)
    return feature_df


def augment_with_climate(
    feat_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    commodity: str,
) -> pd.DataFrame:
    """
    Join PDSI and/or ENSO columns into feat_df for climate-sensitive commodities.

    PDSI columns (pdsi_cornbelt, pdsi_zscore) are added for PDSI_COMMODITIES.
    ENSO columns (mei, mei_lag3m, mei_lag6m, enso_phase) are added for ENSO_COMMODITIES.
    Non-climate commodities pass through unchanged, so this is safe to call unconditionally.

    Parameters
    ----------
    feat_df     : output of build_feature_matrix() — rows = trading days
    climate_df  : output of build_climate_features() — same or wider date range
    commodity   : display name used to select the correct feature subset

    Returns
    -------
    pd.DataFrame with climate columns appended (left-joined on feat_df.index).
    """
    if climate_df is None or climate_df.empty:
        return feat_df

    cols: list[str] = []
    if commodity in PDSI_COMMODITIES:
        cols += [c for c in ("pdsi_cornbelt", "pdsi_zscore") if c in climate_df.columns]
    if commodity in ENSO_COMMODITIES:
        cols += [c for c in ("mei", "mei_lag3m", "mei_lag6m", "enso_phase") if c in climate_df.columns]

    if not cols:
        return feat_df

    climate_aligned = climate_df[cols].reindex(feat_df.index).ffill(limit=5)
    return feat_df.join(climate_aligned, how="left")


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
