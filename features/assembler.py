"""
Feature assembler — combines all external signals into one aligned DataFrame.

This is the single entry point for the augmented feature matrix. It joins
the four external signal modules onto a common DatetimeIndex and produces
a DataFrame that can be concatenated with the price-history features from
models/features.py to create the full ML training matrix.

Column groups
-------------
  energy_transition:   uranium_spread_zscore, battery_demand_index,
                       ets_stress_zscore                        (3 columns)
  macro_overlays:      dxy_zscore63, dxy_mom5, dxy_mom21,
                       vix, vix_risk_off, vix_crisis, vix_level_z,
                       tlt_yield_proxy, tlt_mom21,
                       days_to_wasde, is_wasde_window, wasde_post5,
                       days_to_opec, is_opec_window, opec_post10  (15 columns)
  climate_weather:     mei, mei_lag3m, mei_lag6m, enso_phase,
                       pdsi_cornbelt (if token), pdsi_zscore (if token),
                       hdd_deviation_pct, cdd_deviation_pct (if token)
  sentiment:           sentiment_* per commodity (if FinBERT),
                       eia_crude_surprise, eia_crude_surprise_z (if EIA key)

Compatibility
-------------
The output DataFrame is designed to be appended to build_feature_matrix()
from models/features.py. Both produce DatetimeIndex DataFrames of float
columns with no look-ahead bias — all external signals are either
contemporaneous (prices), backward-looking (rolling computations), or
calendar-based (WASDE/OPEC dummies whose future dates are publicly known
in advance and carry no predictive information about outcomes).

Usage
-----
    from models.features import build_feature_matrix
    from models.data_loader import load_price_matrix
    from features.assembler import build_augmented_features, feature_coverage

    prices = load_price_matrix()

    # Augmented feature matrix (all external signals)
    ext_df = build_augmented_features(
        prices,
        noaa_token=os.environ.get("NOAA_CDO_TOKEN"),
        eia_key=os.environ.get("EIA_API_KEY"),
        run_sentiment=False,   # True if FinBERT and GPU available
    )

    # Price-history features from existing models/features.py
    price_features = build_feature_matrix(prices)

    # Full training matrix
    full_features = price_features.join(ext_df, how="left").ffill(limit=3).dropna()

    print(feature_coverage(ext_df))   # which columns have data
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional

from features.energy_transition import build_energy_transition_features
from features.macro_overlays     import build_macro_overlay_features
from features.climate_weather    import build_climate_features
from features.sentiment          import SentimentFeatures

# Columns to keep from each module (trim raw prices / intermediate values)
_ENERGY_COLS = [
    "uranium_spread_zscore",
    "battery_demand_index", "battery_demand_index_21d_ma",
    "ets_stress_zscore",
]
_MACRO_COLS = [
    "dxy_zscore63", "dxy_mom5", "dxy_mom21",
    "vix", "vix_risk_off", "vix_crisis", "vix_level_z",
    "tlt_yield_proxy", "tlt_mom21",
    "days_to_wasde", "is_wasde_window", "wasde_post5",
    "days_to_opec", "is_opec_window", "opec_post10",
]
_CLIMATE_COLS = [
    "mei", "mei_lag3m", "mei_lag6m", "enso_phase",
    "pdsi_cornbelt", "pdsi_zscore",
    "hdd_deviation_pct", "cdd_deviation_pct", "hdd_dev_21d",
]


def _keep_available(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return only the columns that exist in df."""
    return df[[c for c in cols if c in df.columns]]


def build_augmented_features(
    prices: pd.DataFrame,
    noaa_token:    Optional[str] = None,
    eia_key:       Optional[str] = None,
    run_sentiment: bool = False,
    headlines_df:  Optional[pd.DataFrame] = None,
    period:        str = "2y",
) -> pd.DataFrame:
    """
    Build the full external feature matrix and align it to `prices`.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices from load_price_matrix(). Used only for its DatetimeIndex.
    noaa_token : str, optional
        NOAA CDO API token for PDSI and HDD/CDD climate features.
        Falls back to NOAA_CDO_TOKEN env var. Missing → NaN climate columns.
    eia_key : str, optional
        EIA API key for inventory surprise factor.
        Falls back to EIA_API_KEY env var. Missing → no EIA columns.
    run_sentiment : bool
        If True, runs FinBERT inference on `headlines_df`. Requires
        `transformers` and 5-50s per batch. False = load from cache only.
    headlines_df : pd.DataFrame, optional
        Headlines from services.news_data.fetch_news(). Required if
        run_sentiment=True and no cache exists.
    period : str
        yfinance period passed to energy/macro fetchers.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex aligned to `prices.index`. All columns are float.
        Missing values are forward-filled up to 3 bars; remainder = NaN.
    """
    target_idx = prices.index
    start_date = target_idx.min().strftime("%Y-%m-%01")

    parts = []

    # ── Energy transition ──────────────────────────────────────────────────────
    try:
        et_raw = build_energy_transition_features(period=period)
        if not et_raw.empty:
            et = _keep_available(et_raw, _ENERGY_COLS)
            parts.append(et)
    except Exception as e:
        warnings.warn(f"Energy transition features failed: {e}")

    # ── Macro overlays ─────────────────────────────────────────────────────────
    try:
        macro = build_macro_overlay_features(period=period, index=target_idx)
        if not macro.empty:
            macro = _keep_available(macro, _MACRO_COLS)
            parts.append(macro)
    except Exception as e:
        warnings.warn(f"Macro overlay features failed: {e}")

    # ── Climate / weather ──────────────────────────────────────────────────────
    try:
        climate = build_climate_features(
            noaa_token=noaa_token,
            start_date=start_date,
        )
        if not climate.empty:
            climate = _keep_available(climate, _CLIMATE_COLS)
            parts.append(climate)
    except Exception as e:
        warnings.warn(f"Climate features failed: {e}")

    # ── Sentiment + EIA ────────────────────────────────────────────────────────
    try:
        sf = SentimentFeatures(eia_key=eia_key)
        if run_sentiment:
            sentiment_df = sf.build_all(
                headlines_df=headlines_df,
                rolling_window=5,
            )
        else:
            # Load from cache + EIA only (no FinBERT inference)
            sentiment_df = sf.build_all(headlines_df=None, rolling_window=5)
        if not sentiment_df.empty:
            parts.append(sentiment_df)
    except Exception as e:
        warnings.warn(f"Sentiment/EIA features failed: {e}")

    if not parts:
        return pd.DataFrame(index=target_idx)

    # Outer-join all parts, then align to target_idx
    combined = parts[0]
    for df in parts[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.reindex(target_idx).ffill(limit=3)
    return combined.astype(float, errors="ignore")


def feature_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report data coverage for each column in the augmented feature matrix.

    Useful for diagnosing which external API keys are missing and what
    fraction of the date range each signal covers.

    Returns
    -------
    pd.DataFrame
        Columns: feature, n_non_nan, pct_coverage, first_date, last_date.
        Sorted by pct_coverage ascending (missing features at the top).
    """
    rows = []
    for col in df.columns:
        valid = df[col].dropna()
        rows.append({
            "feature":      col,
            "n_non_nan":    len(valid),
            "pct_coverage": round(100 * len(valid) / max(len(df), 1), 1),
            "first_date":   str(valid.index.min().date()) if len(valid) > 0 else "—",
            "last_date":    str(valid.index.max().date()) if len(valid) > 0 else "—",
        })
    return (
        pd.DataFrame(rows)
        .sort_values("pct_coverage")
        .reset_index(drop=True)
    )


def augment_model_features(
    prices: pd.DataFrame,
    noaa_token: Optional[str] = None,
    eia_key:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: build the full augmented feature matrix that is
    ready to concatenate with models/features.build_feature_matrix().

    Returns a DataFrame with only columns that have ≥ 50% data coverage,
    so sparse external signals don't introduce excessive NaN into the
    ML training matrix.

    Usage
    -----
        from models.features import build_feature_matrix
        from features.assembler import augment_model_features

        prices = load_price_matrix()
        price_feat  = build_feature_matrix(prices)
        ext_feat    = augment_model_features(prices)
        full_matrix = price_feat.join(ext_feat, how="left")
    """
    ext = build_augmented_features(
        prices,
        noaa_token=noaa_token,
        eia_key=eia_key,
        run_sentiment=False,
    )
    if ext.empty:
        return ext

    # Drop columns with < 50% coverage (typically those requiring missing API keys)
    coverage = feature_coverage(ext)
    keep_cols = coverage[coverage["pct_coverage"] >= 50]["feature"].tolist()
    return ext[keep_cols]
