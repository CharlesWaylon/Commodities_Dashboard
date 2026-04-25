"""
Energy-transition feature signals.

Standard momentum and roll-adjusted price features completely misread the
instruments in the energy transition universe (Uranium ETF, Lithium, Carbon
Credits, Rare Earths) because their price dynamics are driven by policy cycles,
battery chemistry shifts, and decarbonisation commitments — not by traditional
supply/demand seasonality.

Three signals:

1. Uranium spread proxy
   Uranium's price is split between the spot market (immediate delivery) and
   long-term contracts (3-10 year utility supply agreements). When spot > LT,
   utilities are scrambling for material — a bullish signal 4-8 weeks ahead.
   Free proxy: URA ETF price relative to its 90-day rolling mean (fast money
   vs. long-duration capital). Positive divergence → spot market stress.

2. Battery metals demand index
   EV adoption drives demand for Li, Co, and rare earths simultaneously but
   with different battery chemistry ratios (LFP = no cobalt; NMC = 30%+ cobalt).
   We construct a weighted composite of LIT (lithium chain), GLNCY (cobalt/zinc),
   and REMX (rare earths) normalised daily returns, then extract the first
   principal component as the "demand regime" index. PC1 typically explains
   60-70% of the variance — it is the shared macro EV demand factor.

3. EU ETS policy-stress indicator
   Carbon credit volume spikes when utilities hedge aggressively ahead of
   compliance deadlines or when new regulation surprises the market. We compute
   volume-weighted return: KRBN_return × (today_volume / 21d_avg_volume).
   A large positive number means a policy shock is being priced in right now.

Usage
-----
    from features.energy_transition import build_energy_transition_features

    prices = load_price_matrix(commodities={
        "Uranium*": "URA", "Lithium*": "LIT",
        "Zinc & Cobalt*": "GLNCY", "Rare Earths*": "REMX",
        "Carbon Credits*": "KRBN",
    }, period="2y")

    df = build_energy_transition_features(prices, volume_df)
    # Returns DataFrame with columns:
    #   uranium_spot_stress, uranium_spread_zscore
    #   battery_demand_index (PC1), li_ret, co_ret, re_ret
    #   ets_stress, ets_vol_ratio
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from typing import Optional

# ── Tickers ────────────────────────────────────────────────────────────────────
_URANIUM_TICKER   = "URA"
_BATTERY_TICKERS  = {"Lithium*": "LIT", "Zinc & Cobalt*": "GLNCY", "Rare Earths*": "REMX"}
_CARBON_TICKER    = "KRBN"

# Uranium spread proxy: divergence of price from 90-day SMA
_URANIUM_LT_WINDOW = 90   # days — approximates long-term contract pricing horizon
_URANIUM_Z_WINDOW  = 63   # z-score normalisation window

# Battery index: PCA window
_PCA_WINDOW = 252   # use full-year covariance structure for PC decomposition

# ETS stress: volume ratio window
_ETS_VOL_WINDOW = 21


def _fetch_ohlcv(tickers: list, period: str = "2y") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Volume for a list of tickers from yfinance."""
    data = yf.download(tickers, period=period, interval="1d",
                       progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    if isinstance(data.columns, pd.MultiIndex):
        close  = data["Close"]
        volume = data["Volume"]
    else:
        close  = data[["Close"]].rename(columns={"Close": tickers[0]})
        volume = data[["Volume"]].rename(columns={"Volume": tickers[0]})

    close.index  = pd.to_datetime(close.index).tz_localize(None)
    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return close.ffill(limit=3), volume.ffill(limit=3)


# ── 1. Uranium spread proxy ────────────────────────────────────────────────────

def uranium_spread_proxy(period: str = "2y") -> pd.DataFrame:
    """
    Compute a proxy for the uranium spot-vs-long-term contract spread.

    Returns
    -------
    pd.DataFrame
        Columns:
          uranium_price        — raw URA ETF close
          uranium_lt_proxy     — 90-day SMA (proxy for LT contract level)
          uranium_spot_stress  — price − SMA (positive = spot above LT)
          uranium_spread_zscore — z-score of the spread over 63-day window
    """
    try:
        close, _ = _fetch_ohlcv([_URANIUM_TICKER], period=period)
        ura = close[_URANIUM_TICKER].dropna()
    except Exception as e:
        warnings.warn(f"Uranium data unavailable: {e}")
        return pd.DataFrame()

    lt_proxy = ura.rolling(_URANIUM_LT_WINDOW).mean()
    stress   = ura - lt_proxy

    mu  = stress.rolling(_URANIUM_Z_WINDOW).mean()
    sig = stress.rolling(_URANIUM_Z_WINDOW).std()
    zscore = ((stress - mu) / sig.clip(lower=1e-9)).rename("uranium_spread_zscore")

    return pd.DataFrame({
        "uranium_price":         ura,
        "uranium_lt_proxy":      lt_proxy,
        "uranium_spot_stress":   stress,
        "uranium_spread_zscore": zscore,
    })


# ── 2. Battery metals demand index ────────────────────────────────────────────

def battery_metals_index(period: str = "2y") -> pd.DataFrame:
    """
    First principal component of {LIT, GLNCY, REMX} daily log-returns.

    PC1 captures the shared EV-demand macro factor. The loadings show which
    metal is leading the current demand cycle.

    Returns
    -------
    pd.DataFrame
        Columns:
          li_ret, co_ret, re_ret       — individual log returns
          battery_demand_index         — PC1 (standardised)
          battery_demand_index_21d_ma  — smoothed signal (21-day MA)
          pc1_loading_li, pc1_loading_co, pc1_loading_re — factor loadings
    """
    tickers = list(_BATTERY_TICKERS.values())
    try:
        close, _ = _fetch_ohlcv(tickers, period=period)
    except Exception as e:
        warnings.warn(f"Battery metals data unavailable: {e}")
        return pd.DataFrame()

    # Log returns
    ret = np.log(close / close.shift(1)).dropna()
    ret.columns = ["li_ret", "co_ret", "re_ret"]

    # PCA via SVD on the return matrix (no sklearn for simplicity)
    scaler = StandardScaler()
    R = scaler.fit_transform(ret.values)
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    pc1_loadings = Vt[0]               # first right singular vector
    pc1_scores   = R @ pc1_loadings    # project onto PC1

    # Flip sign so that positive = demand expansion (positive Li/Co/Re returns)
    if pc1_loadings.mean() < 0:
        pc1_loadings = -pc1_loadings
        pc1_scores   = -pc1_scores

    idx = pd.Series(pc1_scores, index=ret.index, name="battery_demand_index")
    result = ret.copy()
    result["battery_demand_index"]        = idx
    result["battery_demand_index_21d_ma"] = idx.rolling(21).mean()
    result["pc1_loading_li"]              = pc1_loadings[0]
    result["pc1_loading_co"]              = pc1_loadings[1]
    result["pc1_loading_re"]              = pc1_loadings[2]

    return result


# ── 3. EU ETS policy-stress indicator ─────────────────────────────────────────

def ets_policy_stress(period: str = "2y") -> pd.DataFrame:
    """
    Volume-weighted return of KRBN (KraneShares Global Carbon ETF).

    Large positive values mean: carbon prices are moving sharply on heavy
    volume — a policy-shock event is being priced in (new regulation,
    compliance deadline, or cap-tightening announcement).

    Returns
    -------
    pd.DataFrame
        Columns:
          krbn_close       — raw ETF close
          krbn_return      — daily log-return
          krbn_vol_ratio   — today_volume / 21d_avg_volume
          ets_stress       — krbn_return × krbn_vol_ratio (policy-stress signal)
          ets_stress_zscore — z-score of ets_stress over 63-day window
    """
    try:
        close, volume = _fetch_ohlcv([_CARBON_TICKER], period=period)
        krbn_close  = close[_CARBON_TICKER].dropna()
        krbn_volume = volume[_CARBON_TICKER].reindex(krbn_close.index).fillna(0)
    except Exception as e:
        warnings.warn(f"Carbon credits data unavailable: {e}")
        return pd.DataFrame()

    krbn_ret    = np.log(krbn_close / krbn_close.shift(1))
    vol_ratio   = krbn_volume / (krbn_volume.rolling(_ETS_VOL_WINDOW).mean().clip(lower=1))
    stress      = krbn_ret * vol_ratio

    mu  = stress.rolling(63).mean()
    sig = stress.rolling(63).std()
    stress_z = (stress - mu) / sig.clip(lower=1e-9)

    return pd.DataFrame({
        "krbn_close":       krbn_close,
        "krbn_return":      krbn_ret,
        "krbn_vol_ratio":   vol_ratio,
        "ets_stress":       stress,
        "ets_stress_zscore": stress_z,
    })


# ── Convenience runner ─────────────────────────────────────────────────────────

def build_energy_transition_features(period: str = "2y") -> pd.DataFrame:
    """
    Assemble all three energy-transition signals into one aligned DataFrame.

    Parameters
    ----------
    period : str
        yfinance period string passed to each sub-fetcher.

    Returns
    -------
    pd.DataFrame
        All columns from uranium_spread_proxy(), battery_metals_index(), and
        ets_policy_stress(), joined on a common DatetimeIndex. Missing rows
        are forward-filled up to 3 bars, then left as NaN.
    """
    parts = []

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for fn in [uranium_spread_proxy, battery_metals_index, ets_policy_stress]:
            try:
                df = fn(period=period)
                if not df.empty:
                    parts.append(df)
            except Exception as e:
                warnings.warn(f"{fn.__name__} failed: {e}")

    if not parts:
        return pd.DataFrame()

    combined = parts[0]
    for df in parts[1:]:
        combined = combined.join(df, how="outer")

    return combined.ffill(limit=3).sort_index()
