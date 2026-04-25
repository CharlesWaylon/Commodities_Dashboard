"""
GARCH / GJR-GARCH conditional volatility forecaster.

Standard GARCH(1,1) captures volatility clustering (large moves follow large moves).
GJR-GARCH adds an asymmetric leverage term — negative shocks amplify variance more
than positive shocks of equal magnitude. This asymmetry is especially pronounced in
energy: an unexpected supply disruption causes disproportionate vol spikes compared to
a supply-glut of the same size. Always prefer GJR-GARCH for energy commodities.

Usage
-----
    from models.statistical.garch import GARCHForecaster, run_all

    prices = load_price_matrix()
    returns = 100 * np.log(prices / prices.shift(1)).dropna()   # percent returns

    fc = GARCHForecaster(commodity="WTI Crude Oil", model="GJR-GARCH")
    fc.fit(returns["WTI Crude Oil"])

    vol_df = fc.forecast_volatility(horizon=10)
    # DataFrame: columns ["date", "conditional_vol_ann", "realized_vol_ann"]

    summary = fc.parameter_summary()
    # DataFrame with omega, alpha, gamma (leverage), beta, nu (t-dof if StudentT)

    all_results = run_all(prices)
"""

import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Literal

from models.config import MODELING_COMMODITIES, ROLLING_VOL_WINDOW

# Energy commodities get GJR-GARCH by default (leverage term needed)
ENERGY_COMMODITIES = {
    "WTI Crude Oil",
    "Natural Gas (Henry Hub)",
}


class GARCHForecaster:
    """
    Wrapper around the arch library's GARCH / GJR-GARCH estimator.

    Parameters
    ----------
    commodity : str
        Display name — auto-selects GJR when in ENERGY_COMMODITIES.
    model : {"GARCH", "GJR-GARCH", "auto"}
        "auto" selects GJR-GARCH for energy, GARCH otherwise.
    dist : {"normal", "t"}
        Innovation distribution. Student-t captures heavy tails in commodities.
    p, q : int
        ARCH(p) and GARCH(q) lag orders. (1,1) is almost always sufficient.
    """

    def __init__(
        self,
        commodity: str = "WTI Crude Oil",
        model: Literal["GARCH", "GJR-GARCH", "auto"] = "auto",
        dist: Literal["normal", "t"] = "t",
        p: int = 1,
        q: int = 1,
    ):
        self.commodity = commodity
        self.dist = dist
        self.p = p
        self.q = q

        if model == "auto":
            self.model = "GJR-GARCH" if commodity in ENERGY_COMMODITIES else "GARCH"
        else:
            self.model = model

        self._fit = None
        self._returns: Optional[pd.Series] = None

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, returns: pd.Series) -> "GARCHForecaster":
        """
        Estimate model parameters via MLE.

        Parameters
        ----------
        returns : pd.Series
            Percent log-returns (i.e. 100 × log(P_t/P_{t-1})). DatetimeIndex.
        """
        self._returns = returns.copy()

        vol_type = "EGARCH" if self.model == "GJR-GARCH" else "GARCH"
        # arch_model uses "GARCH" vol type; leverage (o=1) activates GJR term
        o = 1 if self.model == "GJR-GARCH" else 0

        dist_map = {"normal": "Normal", "t": "StudentsT"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                returns,
                vol="GARCH",
                p=self.p,
                o=o,
                q=self.q,
                dist=dist_map[self.dist],
                rescale=False,
            )
            self._fit = am.fit(disp="off", show_warning=False)

        return self

    def forecast_volatility(self, horizon: int = 10) -> pd.DataFrame:
        """
        Forward-looking conditional volatility forecast (annualized %).

        Parameters
        ----------
        horizon : int
            Number of trading days ahead to forecast.

        Returns
        -------
        pd.DataFrame
            Columns: conditional_vol_ann (GARCH forecast),
                     realized_vol_ann (trailing 21d rolling vol for overlay).
            Index: datetime — last in-sample date + forecast dates.
        """
        if self._fit is None:
            raise RuntimeError("Call fit() before forecast_volatility().")

        fc = self._fit.forecast(horizon=horizon, reindex=False)
        # variance is in (percent return)^2 units
        cond_var = fc.variance.iloc[-1].values           # shape (horizon,)
        cond_vol_ann = np.sqrt(cond_var * 252)           # annualised %

        last_date = self._returns.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

        realized = self.realized_vol(window=ROLLING_VOL_WINDOW)

        df = pd.DataFrame(
            {
                "conditional_vol_ann": cond_vol_ann,
                "realized_vol_ann": np.nan,
            },
            index=future_dates[:horizon],
        )

        # Append in-sample conditional vol for full overlay
        in_sample_var = self._fit.conditional_volatility           # in % units
        in_sample_ann = in_sample_var * np.sqrt(252)
        realized_aligned = realized.reindex(in_sample_ann.index)
        if hasattr(realized_aligned, "squeeze"):
            realized_aligned = realized_aligned.squeeze()
        in_sample_df = pd.DataFrame(
            {
                "conditional_vol_ann": np.asarray(in_sample_ann).ravel(),
                "realized_vol_ann": np.asarray(realized_aligned).ravel(),
            },
            index=in_sample_ann.index,
        )

        return pd.concat([in_sample_df, df]).sort_index()

    def realized_vol(self, window: int = ROLLING_VOL_WINDOW) -> pd.Series:
        """
        Rolling realized volatility (annualized %) for comparison overlay.
        """
        if self._returns is None:
            raise RuntimeError("Call fit() first.")
        return self._returns.rolling(window).std() * np.sqrt(252)

    def parameter_summary(self) -> pd.DataFrame:
        """
        Return fitted parameters as a tidy DataFrame for display.

        Columns: param, coef, std_err, t_stat, p_value
        """
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        summary = self._fit.params.rename("coef").to_frame()
        summary["std_err"] = self._fit.std_err
        summary["t_stat"] = self._fit.tvalues
        summary["p_value"] = self._fit.pvalues
        summary.index.name = "param"
        return summary.round(6)

    def aic(self) -> float:
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        return float(self._fit.aic)

    def leverage_effect(self) -> Optional[float]:
        """
        Return the GJR leverage coefficient (gamma). None for plain GARCH.

        A positive gamma means negative shocks amplify variance more than
        positive shocks — the asymmetric response at the heart of GJR-GARCH.
        """
        if self.model != "GJR-GARCH" or self._fit is None:
            return None
        params = self._fit.params
        gamma_keys = [k for k in params.index if k.startswith("gamma")]
        return float(params[gamma_keys[0]]) if gamma_keys else None


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
    horizon: int = 10,
) -> dict:
    """
    Fit GARCH/GJR-GARCH for every commodity and return volatility forecasts.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices (columns = display names). From load_price_matrix().
    horizon : int
        Forecast horizon in trading days.

    Returns
    -------
    dict
        Keys are commodity display names. Values are dicts with:
            model       — "GARCH" or "GJR-GARCH"
            vol_df      — DataFrame from forecast_volatility()
            params      — DataFrame from parameter_summary()
            leverage    — gamma coefficient (None for non-energy)
            aic         — float
    """
    if commodities is None:
        commodities = MODELING_COMMODITIES

    # percent log returns (arch library convention)
    returns = 100 * np.log(prices / prices.shift(1)).dropna()
    results = {}

    for name in commodities:
        if name not in returns.columns:
            continue
        fc = GARCHForecaster(commodity=name)
        fc.fit(returns[name])
        results[name] = {
            "commodity": name,
            "model":     fc.model,
            "vol_df":    fc.forecast_volatility(horizon=horizon),
            "params":    fc.parameter_summary(),
            "leverage":  fc.leverage_effect(),
            "aic":       fc.aic(),
        }

    return results
