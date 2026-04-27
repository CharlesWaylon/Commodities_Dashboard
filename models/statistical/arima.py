"""
ARIMA / SARIMA forecaster for per-commodity next-day return prediction.

Order selection is done by AIC grid search over p ∈ [0,3], d ∈ [0,1], q ∈ [0,3].
Natural Gas and agricultural commodities automatically receive the seasonal
extension (SARIMA) with a weekly seasonal period (m=5 trading days).

Usage
-----
    from models.statistical.arima import ARIMAForecaster, run_all

    prices = load_price_matrix()
    returns = np.log(prices / prices.shift(1)).dropna()

    fc = ARIMAForecaster(commodity="WTI Crude Oil")
    fc.fit(returns["WTI Crude Oil"])
    result = fc.forecast(steps=1)
    # {"forecast": float, "lower_95": float, "upper_95": float, "order": (p,d,q), "aic": float}

    all_results = run_all(prices)   # dict keyed by commodity display name
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Optional

from models.config import MODELING_COMMODITIES

# Commodities that warrant a seasonal component (weekly cycle, m=5)
SEASONAL_COMMODITIES = {
    "Natural Gas (Henry Hub)",
    "Corn (CBOT)",
    "Wheat (CBOT SRW)",
    "Soybeans (CBOT)",
    "Live Cattle",
}

# Grid bounds for AIC search
P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)
SEASONAL_ORDER = (1, 0, 1, 5)   # (P,D,Q,m) fixed seasonal component


class ARIMAForecaster:
    """
    AIC-optimal ARIMA (or SARIMA) forecaster for a single commodity.

    Parameters
    ----------
    commodity : str
        Display name — used to decide whether to apply seasonal terms.
    max_p, max_d, max_q : int
        Upper bounds for the AIC grid search.
    """

    def __init__(
        self,
        commodity: str = "WTI Crude Oil",
        max_p: int = 3,
        max_d: int = 1,
        max_q: int = 3,
    ):
        self.commodity = commodity
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self._model_fit = None
        self.best_order: Optional[tuple] = None
        self.best_aic: Optional[float] = None

    # ── private helpers ────────────────────────────────────────────────────────

    def _is_seasonal(self) -> bool:
        return self.commodity in SEASONAL_COMMODITIES

    def _fit_one(self, series: pd.Series, order: tuple, seasonal_order: tuple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False, maxiter=200)

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """
        Grid-search (p,d,q) space by AIC and store the best model.

        Parameters
        ----------
        series : pd.Series
            Daily log returns for this commodity (DatetimeIndex).
        """
        best_aic = np.inf
        best_fit = None
        best_order = (1, 0, 0)

        s_order = SEASONAL_ORDER if self._is_seasonal() else (0, 0, 0, 0)

        for p, d, q in itertools.product(
            range(self.max_p + 1),
            range(self.max_d + 1),
            range(self.max_q + 1),
        ):
            if p == 0 and q == 0:
                continue   # degenerate white-noise model
            try:
                fit = self._fit_one(series, (p, d, q), s_order)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit
                    best_order = (p, d, q)
            except Exception:
                continue

        if best_fit is None:
            # Fallback to AR(1) if grid search fails entirely
            best_fit = self._fit_one(series, (1, 0, 0), (0, 0, 0, 0))
            best_order = (1, 0, 0)
            best_aic = best_fit.aic

        self._model_fit = best_fit
        self.best_order = best_order
        self.best_aic = best_aic
        return self

    def forecast(self, steps: int = 1) -> dict:
        """
        Generate h-step-ahead point forecast with 95 % prediction interval.

        Returns
        -------
        dict with keys:
            forecast  — point estimate (list of length `steps`)
            lower_95  — lower bound of 95 % PI
            upper_95  — upper bound of 95 % PI
            order     — (p, d, q) tuple selected by AIC
            aic       — AIC of the fitted model
        """
        if self._model_fit is None:
            raise RuntimeError("Call fit() before forecast().")

        pred = self._model_fit.get_forecast(steps=steps)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)

        return {
            "commodity":  self.commodity,
            "forecast":   mean.tolist(),
            "lower_95":   ci.iloc[:, 0].tolist(),
            "upper_95":   ci.iloc[:, 1].tolist(),
            "order":      self.best_order,
            "seasonal":   self._is_seasonal(),
            "aic":        round(self.best_aic, 2),
        }

    def summary(self) -> str:
        if self._model_fit is None:
            raise RuntimeError("Model not fitted.")
        return str(self._model_fit.summary())


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
    steps: int = 1,
) -> dict:
    """
    Fit ARIMA/SARIMA for every commodity in `prices` and return forecasts.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices (columns = display names). From load_price_matrix().
    commodities : dict, optional
        Subset override. Defaults to MODELING_COMMODITIES.
    steps : int
        Forecast horizon in trading days.

    Returns
    -------
    dict
        Keys are commodity display names. Values are forecast dicts from
        ARIMAForecaster.forecast().
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    returns = np.log(prices / prices.shift(1)).dropna()
    results = {}

    for name in commodities:
        if name not in returns.columns:
            continue
        fc = ARIMAForecaster(commodity=name)
        fc.fit(returns[name])
        results[name] = fc.forecast(steps=steps)

    return results
