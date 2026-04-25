"""
Kalman filter for dynamic hedge ratio estimation on cointegrated pairs.

Classical approach: model the hedge ratio β as a random walk (Kalman's "local
level" model). The filter updates β_t on each new observation, weighting recent
data more heavily when the ratio is drifting and trusting prior estimates more
when the spread is stable. This adaptive smoothing is what a rolling-window OLS
hedge ratio fundamentally cannot do.

State-space formulation
-----------------------
    Observation:  y_t = x_t · β_t + ε_t,     ε_t ~ N(0, R)
    Transition:   β_t = β_{t-1} + η_t,        η_t ~ N(0, Q)

  y_t  = price of the "y" leg (e.g. WTI)
  x_t  = price of the "x" leg (e.g. Brent)
  β_t  = time-varying hedge ratio
  R    = observation noise variance (estimated from residuals)
  Q    = state noise variance (controls how fast β can change)

The signal-to-noise ratio Q/R is the key tuning parameter. Higher Q/R lets
β_t track more aggressively; lower Q/R smooths more. We estimate Q via a
simple variance-of-OLS-residuals heuristic and let the user override.

Usage
-----
    from models.statistical.kalman import KalmanHedgeRatio, run_all_pairs

    prices = load_price_matrix(period="2y")
    kf = KalmanHedgeRatio(pair="WTI/Brent")
    kf.fit(y=prices["WTI Crude Oil"], x=prices["Brent Crude Oil"])

    ratios = kf.hedge_ratios()      # pd.Series of β_t
    spread = kf.spread()            # pd.Series of y_t - β_t * x_t
    z      = kf.spread_zscore()     # standardised spread (mean-reversion signal)

    all_pairs = run_all_pairs(prices)
"""

import numpy as np
import pandas as pd
from typing import Optional

# Pre-defined pairs (display names matching MODELING_COMMODITIES or full registry)
HEDGE_PAIRS = {
    "WTI/Brent":       ("WTI Crude Oil",           "Brent Crude Oil"),
    "Corn/Soybeans":   ("Corn (CBOT)",              "Soybeans (CBOT)"),
    "Gold/Silver":     ("Gold (COMEX)",             "Silver (COMEX)"),
    "Wheat/Corn":      ("Wheat (CBOT SRW)",         "Corn (CBOT)"),
}

# How many days of history to use for the initial OLS estimate of R
INIT_WINDOW = 60


class KalmanHedgeRatio:
    """
    Univariate Kalman filter tracking the dynamic hedge ratio β for a pair.

    Parameters
    ----------
    pair : str
        Descriptive label (e.g. "WTI/Brent"). For display only.
    delta : float
        State transition noise scaling factor. Controls how fast β can drift.
        Larger delta → β tracks more aggressively. Typical range: 1e-5 – 1e-3.
    """

    def __init__(self, pair: str = "WTI/Brent", delta: float = 1e-4):
        self.pair = pair
        self.delta = delta
        self._beta: Optional[pd.Series] = None
        self._spread: Optional[pd.Series] = None
        self._y: Optional[pd.Series] = None
        self._x: Optional[pd.Series] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _run_filter(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Core Kalman recursion. Returns array of filtered β_t estimates.
        """
        n = len(y)
        beta = np.zeros(n)        # filtered state estimates
        P = np.zeros(n)           # state error covariance

        # Initialise from OLS on first INIT_WINDOW observations
        w = min(INIT_WINDOW, n // 4)
        beta[0] = np.cov(y[:w], x[:w])[0, 1] / np.var(x[:w])
        P[0] = 1.0

        # Q = process noise variance (how fast β is allowed to drift)
        # R = observation noise variance (calibrated from OLS residuals)
        Q = self.delta / (1 - self.delta)
        resid_init = y[:w] - beta[0] * x[:w]
        R = np.var(resid_init) if np.var(resid_init) > 0 else 1.0

        for t in range(1, n):
            # Predict
            beta_pred = beta[t - 1]
            P_pred = P[t - 1] + Q

            # Update (Kalman gain)
            K = P_pred * x[t] / (x[t] ** 2 * P_pred + R)
            innovation = y[t] - beta_pred * x[t]

            beta[t] = beta_pred + K * innovation
            P[t] = (1 - K * x[t]) * P_pred

            # Adapt R online from squared innovation (fading memory)
            R = 0.95 * R + 0.05 * innovation ** 2

        return beta

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, y: pd.Series, x: pd.Series) -> "KalmanHedgeRatio":
        """
        Run the Kalman filter on a price pair.

        Parameters
        ----------
        y, x : pd.Series
            Price series for the two legs. Must share a DatetimeIndex.
            Typical convention: y is the instrument you hold long,
            x is the one you hedge (e.g. y=WTI, x=Brent).
        """
        idx = y.index.intersection(x.index)
        y_aligned = y.loc[idx]
        x_aligned = x.loc[idx]

        self._y = y_aligned
        self._x = x_aligned

        beta_arr = self._run_filter(y_aligned.values, x_aligned.values)
        self._beta = pd.Series(beta_arr, index=idx, name=f"beta_{self.pair}")
        self._spread = y_aligned - beta_arr * x_aligned
        self._spread.name = f"spread_{self.pair}"

        return self

    def hedge_ratios(self) -> pd.Series:
        """Time-varying hedge ratio β_t. Long 1 unit of y, short β_t units of x."""
        if self._beta is None:
            raise RuntimeError("Call fit() first.")
        return self._beta

    def spread(self) -> pd.Series:
        """Dynamic spread: y_t - β_t * x_t."""
        if self._spread is None:
            raise RuntimeError("Call fit() first.")
        return self._spread

    def spread_zscore(self, window: int = 63) -> pd.Series:
        """
        Rolling z-score of the spread. Values beyond ±2 indicate
        a potential mean-reversion entry; beyond ±3 suggest a breakout.
        """
        if self._spread is None:
            raise RuntimeError("Call fit() first.")
        mu = self._spread.rolling(window).mean()
        sigma = self._spread.rolling(window).std()
        return ((self._spread - mu) / sigma).rename(f"zscore_{self.pair}")

    def summary(self) -> dict:
        """Return a concise summary dict for display."""
        if self._beta is None:
            raise RuntimeError("Call fit() first.")
        beta = self._beta
        spread = self._spread
        return {
            "pair":              self.pair,
            "current_beta":      round(float(beta.iloc[-1]), 4),
            "beta_mean":         round(float(beta.mean()), 4),
            "beta_std":          round(float(beta.std()), 4),
            "spread_mean":       round(float(spread.mean()), 4),
            "spread_std":        round(float(spread.std()), 4),
            "current_zscore":    round(float(self.spread_zscore().iloc[-1]), 3),
        }


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all_pairs(prices: pd.DataFrame) -> dict:
    """
    Fit Kalman hedge ratio for every pair defined in HEDGE_PAIRS
    that has both legs present in `prices`.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices (columns = display names). From load_price_matrix().

    Returns
    -------
    dict
        Keys are pair labels ("WTI/Brent", etc.).
        Values are dicts with keys: kf (KalmanHedgeRatio), summary, beta, spread, zscore.
    """
    results = {}
    for label, (y_name, x_name) in HEDGE_PAIRS.items():
        if y_name not in prices.columns or x_name not in prices.columns:
            continue
        kf = KalmanHedgeRatio(pair=label)
        kf.fit(prices[y_name], prices[x_name])
        results[label] = {
            "kf":      kf,
            "summary": kf.summary(),
            "beta":    kf.hedge_ratios(),
            "spread":  kf.spread(),
            "zscore":  kf.spread_zscore(),
        }
    return results
