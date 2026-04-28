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
    kf = KalmanHedgeRatio(pair="WTI ↔ Brent")
    kf.fit(y=prices["WTI Crude Oil"], x=prices["Brent Crude Oil"])

    ratios  = kf.hedge_ratios()          # pd.Series of β_t
    lo, hi  = kf.beta_ci()              # 95% confidence bands on β_t
    spread  = kf.spread()               # pd.Series of y_t - β_t * x_t
    z       = kf.spread_zscore()        # standardised spread (mean-reversion signal)
    p_val   = kf.cointegration_pvalue() # Engle-Granger p-value

    all_pairs = run_all_pairs(prices)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

# Pre-defined pairs with strong cointegration rationale.
# Display names must match COMMODITY_TICKERS keys in services/price_data.py exactly.
HEDGE_PAIRS = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI ↔ Brent":              ("WTI Crude Oil",           "Brent Crude Oil"),
    "Gasoline ↔ WTI":           ("Gasoline (RBOB)",         "WTI Crude Oil"),
    "Heating Oil ↔ Nat Gas":    ("Heating Oil (No.2)",      "Natural Gas (Henry Hub)"),
    # ── Precious Metals ───────────────────────────────────────────────────────
    "Gold ↔ Silver":            ("Gold (COMEX)",            "Silver (COMEX)"),
    "Gold ↔ Platinum":          ("Gold (COMEX)",            "Platinum"),
    "Platinum ↔ Palladium":     ("Platinum",                "Palladium"),
    # ── Base / Industrial Metals ──────────────────────────────────────────────
    "Copper ↔ Aluminum":        ("Copper (COMEX)",          "Aluminum (COMEX)"),
    # ── Grains & Oilseeds ─────────────────────────────────────────────────────
    "Corn ↔ Soybeans":          ("Corn (CBOT)",             "Soybeans (CBOT)"),
    "Wheat ↔ Corn":             ("Wheat (CBOT SRW)",        "Corn (CBOT)"),
    "SRW ↔ HRW Wheat":          ("Wheat (CBOT SRW)",        "Wheat (KC HRW)"),
    "Soy Oil ↔ Soybeans":       ("Soybean Oil",             "Soybeans (CBOT)"),
    "Soy Meal ↔ Soybeans":      ("Soybean Meal",            "Soybeans (CBOT)"),
    # ── Livestock ─────────────────────────────────────────────────────────────
    "Live ↔ Feeder Cattle":     ("Live Cattle",             "Feeder Cattle"),
    "Cattle ↔ Corn":            ("Live Cattle",             "Corn (CBOT)"),
    "Hogs ↔ Corn":              ("Lean Hogs",               "Corn (CBOT)"),
    # ── Softs ─────────────────────────────────────────────────────────────────
    "Coffee ↔ Sugar":           ("Coffee (Arabica)",        "Sugar (Raw #11)"),
}

# Days used for the initial OLS estimate of β and R.
# 120 days (~6 months) gives a more stable prior than the original 60,
# especially for seasonal pairs (grains, livestock) where a shorter window
# may land entirely in one weather/demand regime.
INIT_WINDOW = 120


class KalmanHedgeRatio:
    """
    Univariate Kalman filter tracking the dynamic hedge ratio β for a pair.

    Parameters
    ----------
    pair : str
        Descriptive label (e.g. "WTI ↔ Brent"). For display only.
    delta : float
        State transition noise scaling factor. Controls how fast β can drift.
        Larger delta → β tracks more aggressively. Typical range: 1e-5 – 1e-3.
    """

    def __init__(self, pair: str = "WTI ↔ Brent", delta: float = 1e-4):
        self.pair  = pair
        self.delta = delta
        self._beta:   Optional[pd.Series] = None
        self._P:      Optional[pd.Series] = None   # state error covariance P_t
        self._spread: Optional[pd.Series] = None
        self._y:      Optional[pd.Series] = None
        self._x:      Optional[pd.Series] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _run_filter(
        self, y: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core Kalman recursion. Returns (beta, P) arrays.

        P[t] is the posterior state error covariance — a direct measure of
        how uncertain the filter is about β_t. The 95% CI on β_t is
        β_t ± 1.96 * sqrt(P[t]).
        """
        n    = len(y)
        beta = np.zeros(n)
        P    = np.zeros(n)

        # Initialise from OLS on first INIT_WINDOW observations.
        w        = min(INIT_WINDOW, n // 4)
        beta[0]  = np.cov(y[:w], x[:w])[0, 1] / np.var(x[:w])
        P[0]     = 1.0

        Q            = self.delta / (1 - self.delta)
        resid_init   = y[:w] - beta[0] * x[:w]
        R            = np.var(resid_init) if np.var(resid_init) > 0 else 1.0

        for t in range(1, n):
            # Predict
            beta_pred = beta[t - 1]
            P_pred    = P[t - 1] + Q

            # Update
            K           = P_pred * x[t] / (x[t] ** 2 * P_pred + R)
            innovation  = y[t] - beta_pred * x[t]
            beta[t]     = beta_pred + K * innovation
            P[t]        = (1 - K * x[t]) * P_pred

            # Adapt R online with a faster fading memory (half-life ≈ 7 days).
            # The original 0.95 decay was too slow to respond to volatility shocks;
            # 0.90 lets R track regime changes without becoming noise-driven.
            R = 0.90 * R + 0.10 * innovation ** 2

        return beta, P

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
        idx        = y.index.intersection(x.index)
        y_aligned  = y.loc[idx]
        x_aligned  = x.loc[idx]

        self._y = y_aligned
        self._x = x_aligned

        beta_arr, P_arr = self._run_filter(y_aligned.values, x_aligned.values)

        self._beta   = pd.Series(beta_arr, index=idx, name=f"beta_{self.pair}")
        self._P      = pd.Series(P_arr,    index=idx, name=f"P_{self.pair}")
        self._spread = (y_aligned - beta_arr * x_aligned).rename(f"spread_{self.pair}")

        return self

    def cointegration_pvalue(self) -> float:
        """
        Engle-Granger cointegration test p-value on the fitted pair.

        Returns a float in [0, 1]. Values below 0.05 indicate the pair is
        statistically cointegrated at the 5% level — a prerequisite for the
        Kalman spread to mean-revert reliably.

        Returns np.nan if statsmodels is not available or fit() hasn't run.
        """
        if self._y is None or self._x is None:
            raise RuntimeError("Call fit() first.")
        try:
            from statsmodels.tsa.stattools import coint
            _, p_value, _ = coint(self._y.values, self._x.values)
            return float(p_value)
        except ImportError:
            return float("nan")

    def hedge_ratios(self) -> pd.Series:
        """Time-varying hedge ratio β_t. Long 1 unit of y, short β_t units of x."""
        if self._beta is None:
            raise RuntimeError("Call fit() first.")
        return self._beta

    def beta_ci(self, z: float = 1.96) -> Tuple[pd.Series, pd.Series]:
        """
        95% confidence interval on β_t derived from the Kalman state covariance P_t.

        Returns (lower, upper) pd.Series. A wide band means the filter is uncertain
        about the current hedge ratio — treat the signal with caution.
        """
        if self._P is None:
            raise RuntimeError("Call fit() first.")
        std   = np.sqrt(self._P.clip(lower=0))
        lower = (self._beta - z * std).rename(f"beta_lo_{self.pair}")
        upper = (self._beta + z * std).rename(f"beta_hi_{self.pair}")
        return lower, upper

    def spread(self) -> pd.Series:
        """Dynamic spread: y_t - β_t * x_t."""
        if self._spread is None:
            raise RuntimeError("Call fit() first.")
        return self._spread

    def spread_zscore(self, window: int = 63) -> pd.Series:
        """
        Rolling z-score of the spread.

        Parameters
        ----------
        window : int
            Lookback for the rolling mean and std. Default 63 (one quarter).
            Shorter windows are more sensitive to recent regime; longer windows
            are more stable but slower to signal.

        Values beyond ±2 indicate a potential mean-reversion entry;
        beyond ±3 suggest a breakout / failed cointegration.
        """
        if self._spread is None:
            raise RuntimeError("Call fit() first.")
        mu    = self._spread.rolling(window).mean()
        sigma = self._spread.rolling(window).std()
        return ((self._spread - mu) / sigma).rename(f"zscore_{self.pair}")

    def summary(self) -> dict:
        """Return a concise summary dict for display."""
        if self._beta is None:
            raise RuntimeError("Call fit() first.")
        beta   = self._beta
        spread = self._spread
        return {
            "pair":           self.pair,
            "current_beta":   round(float(beta.iloc[-1]), 4),
            "beta_mean":      round(float(beta.mean()), 4),
            "beta_std":       round(float(beta.std()), 4),
            "beta_min":       round(float(beta.min()), 4),
            "beta_max":       round(float(beta.max()), 4),
            "spread_mean":    round(float(spread.mean()), 4),
            "spread_std":     round(float(spread.std()), 4),
            "current_zscore": round(float(self.spread_zscore().iloc[-1]), 3),
        }

    def hedge_info(self) -> dict:
        """Alias for summary()."""
        return self.summary()


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all_pairs(prices: pd.DataFrame) -> dict:
    """
    Fit Kalman hedge ratio for every pair defined in HEDGE_PAIRS
    that has both legs present in `prices`.

    Returns
    -------
    dict
        Keys are pair labels. Values are dicts with keys:
        kf, summary, beta, beta_ci, spread, zscore, coint_pvalue.
    """
    results = {}
    for label, (y_name, x_name) in HEDGE_PAIRS.items():
        if y_name not in prices.columns or x_name not in prices.columns:
            continue
        kf = KalmanHedgeRatio(pair=label)
        kf.fit(prices[y_name], prices[x_name])
        lo, hi = kf.beta_ci()
        results[label] = {
            "kf":           kf,
            "summary":      kf.summary(),
            "beta":         kf.hedge_ratios(),
            "beta_ci":      (lo, hi),
            "spread":       kf.spread(),
            "zscore":       kf.spread_zscore(),
            "coint_pvalue": kf.cointegration_pvalue(),
        }
    return results
