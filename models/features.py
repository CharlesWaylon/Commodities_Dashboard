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
    FORECAST_HORIZON,
    RETURN_LAGS,
    ROLLING_VOL_WINDOW,
    ROLLING_MOM_WINDOW,
    ROLLING_MOM_WINDOW_LONG,
    ZSCORE_WINDOW,
    N_QUBITS,
    MIN_BASIS_COVERAGE,
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


def sharpe_signal(returns: pd.DataFrame, mom_window: int = ROLLING_MOM_WINDOW,
                  vol_window: int = ROLLING_VOL_WINDOW) -> pd.DataFrame:
    """
    Risk-adjusted momentum: rolling cumulative return divided by rolling vol.

    Gorton–Rouwenhorst (2006) show that commodity return predictability comes
    partly from carry/roll yield.  In the absence of term-structure data (second
    contract), this signal proxies carry: a commodity in backwardation tends to
    show positive momentum backed by low vol, giving a high Sharpe signal.
    Vol-scaling also prevents high-vol instruments (crypto, nat gas) from
    dominating the cross-sectional ranking.
    """
    mom = returns.rolling(mom_window).sum()
    vol = returns.rolling(vol_window).std().replace(0, np.nan)
    return (mom / vol).fillna(0.0)


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
    second_contract_prices: pd.DataFrame | None = None,
    front_raw_prices: pd.DataFrame | None = None,
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

    second_contract_prices : pd.DataFrame or None, optional
        Next-maturity (M2) price matrix — same column names as ``prices``.
        When supplied, ``build_term_structure_features()`` is called and its
        ``{c}_basis``, ``{c}_basis_zscore``, and ``{c}_roll_yield`` columns are
        left-joined onto the feature matrix (after the vol/mom/zscore block).
        Columns present in M2 but absent from the sector-filtered ``prices`` are
        silently ignored.  NaN rows during the z-score warmup are expected and
        handled identically to other rolling features.
        When None (the default), output is identical to the pre-M2 behaviour, so
        all existing call sites remain unaffected.

    front_raw_prices : pd.DataFrame or None, optional
        Genuine front (M1) RAW price matrix from pipeline/stitch_m2.py
        (interval='1d_m1_raw'), same column names as ``prices``.  When supplied
        alongside ``second_contract_prices``, the basis is computed as
        log(M1_raw / M2_raw) — both legs from the same dated-contract universe on
        the same dates (the only economically valid construction).  When None, the
        roll-adjusted continuous front in ``prices`` is used as the M1 leg
        (legacy behaviour), which mixes an adjusted front with a raw M2.

    Output
    ------
    pd.DataFrame
        One row per trading day. Columns:
          {commodity}_ret_{lag}d    — lagged log-return (lags from RETURN_LAGS)
          {commodity}_vol21         — rolling 21-day realised volatility
          {commodity}_mom10         — rolling 10-day cumulative return (short trend)
          {commodity}_mom21         — rolling 21-day cumulative return (medium trend)
          {commodity}_sharpe        — risk-adjusted momentum: mom10 / vol21 (carry proxy)
          {commodity}_zscore        — rolling 63-day z-score
          {commodity}_basis         — log(front/M2); only when second_contract_prices given
          {commodity}_basis_zscore  — rolling z-score of basis; only with M2 data
          {commodity}_roll_yield    — annualised basis (basis × 252); only with M2 data
        External-signal columns that survive filtering are left unchanged.

    Raises
    ------
    ValueError
        If *sector* is specified but matches no columns in *prices*.
    """
    if sector is not None:
        prices = _filter_prices_to_sector(prices, sector)

    # Separate commodity columns from any external signals that survived filtering.
    known_commodities = set(COMMODITY_SECTORS.keys())
    commodity_cols = [c for c in prices.columns if c in known_commodities]
    external_cols  = [c for c in prices.columns if c not in known_commodities]

    price_data = prices[commodity_cols] if commodity_cols else prices

    ret      = log_returns(price_data)
    vol      = rolling_volatility(ret)
    mom      = rolling_momentum(ret)
    mom_long = rolling_momentum(ret, ROLLING_MOM_WINDOW_LONG)
    shp      = sharpe_signal(ret)
    zsc      = rolling_zscore(price_data)

    parts = []

    # Lagged returns (1d, 2d, 5d, 10d) — 10d lag aligns with FORECAST_HORIZON
    for lag in RETURN_LAGS:
        lagged = ret.shift(lag)
        lagged.columns = [f"{c}_ret_{lag}d" for c in ret.columns]
        parts.append(lagged)

    vol.columns      = [f"{c}_vol21"   for c in vol.columns]
    mom.columns      = [f"{c}_mom10"   for c in mom.columns]
    mom_long.columns = [f"{c}_mom21"   for c in mom_long.columns]
    shp.columns      = [f"{c}_sharpe"  for c in shp.columns]
    zsc.columns      = [f"{c}_zscore"  for c in zsc.columns]

    parts += [vol, mom, mom_long, shp, zsc]

    # ── Term-structure features (basis / roll yield) ───────────────────────────
    # Only produced when second_contract_prices is supplied.  The M2 matrix is
    # pre-filtered to the sector-active columns so basis is only computed for
    # futures that have real term structure (ETFs/proxies are absent from M2).
    if second_contract_prices is not None and not second_contract_prices.empty:
        # Restrict M2 to columns that survived the sector filter on prices
        m2_filtered = second_contract_prices.reindex(
            columns=[c for c in second_contract_prices.columns if c in price_data.columns]
        )
        if not m2_filtered.empty:
            # Restrict the raw-front matrix to the same sector-active columns so
            # basis = log(M1_raw / M2_raw) is computed leg-consistently.
            m1_raw_filtered = None
            if front_raw_prices is not None and not front_raw_prices.empty:
                m1_raw_filtered = front_raw_prices.reindex(
                    columns=[c for c in front_raw_prices.columns if c in price_data.columns]
                )
            ts_features = build_term_structure_features(
                price_data, m2_filtered, front_raw_prices=m1_raw_filtered,
            )
            # Coverage gate: stitched M2/M1-raw series are shallow at first and
            # are ~99% NaN over a multi-year training window.  Admitting such a
            # column would make `feat.join(target).dropna()` wipe EVERY row (each
            # has a NaN basis), zeroing out the ML tiers.  So keep only columns
            # whose non-NaN coverage over the feature window reaches
            # MIN_BASIS_COVERAGE; drop the rest (the model then trains exactly as
            # it did pre-M2).  As the stitched series deepens, columns switch on
            # automatically.  MIN_BASIS_COVERAGE = 0.0 disables the gate.
            if not ts_features.empty:
                n_rows = len(price_data.index)
                if n_rows > 0 and MIN_BASIS_COVERAGE > 0.0:
                    min_obs = MIN_BASIS_COVERAGE * n_rows
                    keep = [
                        c for c in ts_features.columns
                        if ts_features[c].notna().sum() >= min_obs
                    ]
                    ts_features = ts_features[keep]
                if not ts_features.empty:
                    parts.append(ts_features)

    # Append any external signal columns (DXY, VIX, TLT, PDSI, etc.) unchanged.
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


def build_term_structure_features(
    prices: pd.DataFrame,
    second_contract_prices: pd.DataFrame | None = None,
    front_raw_prices: pd.DataFrame | None = None,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Basis and roll-yield features from front-to-next-maturity spreads.

    These are the Gorton–Hayashi–Rouwenhorst carry signals that have the most
    documented forecasting power for commodity returns at monthly horizons.
    Positive roll yield = backwardation → long position earns carry.
    Negative roll yield = contango → carry hurts longs.

    Parameters
    ----------
    prices : pd.DataFrame
        Continuous front-contract price matrix (roll-adjusted) — same as passed
        to build_feature_matrix().  Used as the M1 leg ONLY as a fallback for
        commodities that have no stitched raw-front series.
    second_contract_prices : pd.DataFrame or None
        Genuine second-nearby (M2) RAW prices from pipeline/stitch_m2.py
        (interval='1d_m2'), same column names as prices.  When None/empty an empty
        DataFrame is returned so callers are unaffected.
    front_raw_prices : pd.DataFrame or None
        Genuine front (M1) RAW prices from pipeline/stitch_m2.py
        (interval='1d_m1_raw'), same column names as prices.  When supplied, the
        basis for each shared column is log(M1_raw / M2_raw) — both legs drawn
        from the SAME dated-contract universe on the SAME dates, which is the only
        economically valid construction.  Columns absent here fall back to the
        roll-adjusted continuous front in ``prices`` (legacy behaviour).
    zscore_window : int
        Window for normalising the basis series.

    Returns
    -------
    pd.DataFrame
        Columns per commodity (when second_contract_prices is supplied):
          {c}_basis         — log(front / M2) = log spread
          {c}_basis_zscore  — rolling z-score of basis
          {c}_roll_yield    — annualised basis (basis * 252)
        Empty DataFrame when second_contract_prices is None.
    """
    if second_contract_prices is None or second_contract_prices.empty:
        return pd.DataFrame(index=prices.index)

    shared = [c for c in prices.columns if c in second_contract_prices.columns]
    if not shared:
        return pd.DataFrame(index=prices.index)

    p2 = second_contract_prices[shared]

    # M1 leg: prefer the stitched RAW front (same universe/dates as M2); fall back
    # to the roll-adjusted continuous front for any column with no raw-front series.
    if front_raw_prices is not None and not front_raw_prices.empty:
        p1 = pd.DataFrame(index=p2.index)
        for col in shared:
            if col in front_raw_prices.columns:
                p1[col] = front_raw_prices[col].reindex(p2.index)
            else:
                p1[col] = prices[col].reindex(p2.index)
    else:
        p1 = prices[shared].reindex(p2.index)

    basis = np.log(p1 / p2)

    mu  = basis.rolling(zscore_window).mean()
    sig = basis.rolling(zscore_window).std().replace(0, np.nan)
    basis_z = (basis - mu) / sig

    roll_yield = basis * 252  # annualise: assumes ~1 calendar day per bar

    parts = []
    for col in shared:
        df = pd.DataFrame({
            f"{col}_basis":        basis[col],
            f"{col}_basis_zscore": basis_z[col],
            f"{col}_roll_yield":   roll_yield[col],
        })
        parts.append(df)

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=prices.index)


def build_target(
    prices: pd.DataFrame,
    target_name: str = DEFAULT_TARGET,
    horizon: int = FORECAST_HORIZON,
) -> pd.Series:
    """
    Forward cumulative log-return of the target commodity over `horizon` days.

    At horizon=1 this is the next-day return (legacy behaviour).
    At horizon=10 (the default, matching FORECAST_HORIZON) the target is the
    2-week cumulative log-return — a partly-forecastable quantity where momentum
    and carry signals have documented predictive power (Gorton–Rouwenhorst 2006).

    Overlapping windows
    -------------------
    Consecutive H-day targets share H-1 return observations.  This serial
    correlation reduces effective sample size but does NOT introduce look-ahead
    bias — the shift(-H) ensures row t only sees returns from t+1 onward.
    Models that split train from validation must embargo H rows at the boundary
    to prevent correlated targets from inflating held-out loss estimates.
    See XGBoostForecaster.fit() and BacktestHarness._run_commodity().
    """
    ret = log_returns(prices)
    if horizon == 1:
        return ret[target_name].shift(-1).rename("target")
    # rolling(H).sum() at row i = sum(ret[i-H+1 : i+1])
    # shift(-H) at row i       = that sum at row i+H = sum(ret[i+1 : i+H+1])
    return ret[target_name].rolling(horizon).sum().shift(-horizon).rename("target")


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
    y_series = build_target(prices, target_name)  # uses FORECAST_HORIZON by default

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
