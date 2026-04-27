"""
VAR / VECM for joint modeling of commodity complexes.

Vector Autoregression (VAR) treats a group of commodities as a system: each
series depends on its own lags and the lags of all other series. This captures
lead-lag relationships that single-equation models miss — e.g. a WTI spike
that ripples through Brent and Heating Oil over several days.

If Johansen's cointegration test finds long-run equilibrium relationships
among the series, we switch to VECM (Vector Error Correction Model), which
augments the VAR with error-correction terms that pull prices back toward
the cointegrating relationship.

Key outputs
-----------
  1. Granger causality table — p-values for "does X Granger-cause Y?". A low
     p-value means X's lagged values carry predictive information about Y
     beyond Y's own history. Good for club discussion ("does Corn Granger-cause
     Soybeans?").

  2. Impulse Response Functions (IRFs) — how a one-standard-deviation shock to
     one commodity propagates through the system over N days. The canonical
     diagnostic for energy complex dynamics.

  3. Forecast Error Variance Decomposition (FEVD) — what fraction of commodity
     Y's forecast error at horizon h is attributable to shocks originating in
     commodity X. Complements the IRF.

Usage
-----
    from models.statistical.var_vecm import CommodityVAR, run_energy_system

    prices = load_price_matrix()

    var = CommodityVAR(group="energy")
    var.fit(prices)

    gc   = var.granger_causality()         # DataFrame of p-values
    irf  = var.impulse_response(steps=10)  # DataFrame of IRF paths
    fevd = var.fevd(steps=10)              # DataFrame of variance decomposition

    # Full convenience runner (fits both energy and grains groups)
    results = run_energy_system(prices)
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.stattools import adfuller
from typing import Optional, Literal

# Pre-defined commodity groups for joint modeling
COMMODITY_GROUPS = {
    "energy": [
        "WTI Crude Oil",
        "Natural Gas (Henry Hub)",
    ],
    "grains": [
        "Corn (CBOT)",
        "Wheat (CBOT SRW)",
        "Soybeans (CBOT)",
    ],
    "metals": [
        "Gold (COMEX)",
        "Silver (COMEX)",
        "Copper (COMEX)",
    ],
}

MAX_LAGS = 10          # upper bound for VAR lag-order selection (BIC)
IRF_ORTH = True        # use orthogonalised (Cholesky) IRFs


class CommodityVAR:
    """
    VAR (or VECM) on a commodity group.

    Parameters
    ----------
    group : str
        One of "energy", "grains", "metals" — or "custom".
    custom_cols : list[str], optional
        Column names to use when group="custom".
    use_returns : bool
        If True, model log returns (stationary). If False, model log prices
        (potentially non-stationary — needed for VECM).
    """

    def __init__(
        self,
        group: str = "energy",
        custom_cols: Optional[list] = None,
        use_returns: bool = True,
    ):
        self.group = group
        self.custom_cols = custom_cols
        self.use_returns = use_returns

        self._var_fit = None
        self._vecm_fit = None
        self._data: Optional[pd.DataFrame] = None
        self._cols: Optional[list] = None
        self._lag_order: Optional[int] = None
        self._coint_rank: Optional[int] = None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _select_columns(self, prices: pd.DataFrame) -> list:
        if self.group == "custom" and self.custom_cols:
            return [c for c in self.custom_cols if c in prices.columns]
        cols = COMMODITY_GROUPS.get(self.group, [])
        return [c for c in cols if c in prices.columns]

    def _is_stationary(self, series: pd.Series, significance: float = 0.05) -> bool:
        result = adfuller(series.dropna(), autolag="AIC")
        return result[1] < significance

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame) -> "CommodityVAR":
        """
        Fit VAR (on returns) or VECM (on log prices when cointegrated).

        Lag order is selected by BIC up to MAX_LAGS.

        Parameters
        ----------
        prices : pd.DataFrame
            Closing prices. Columns must include this group's commodities.
        """
        cols = self._select_columns(prices)
        if len(cols) < 2:
            raise ValueError(
                f"Need ≥2 commodities for {self.group}. Found: {cols}"
            )
        self._cols = cols

        log_prices = np.log(prices[cols]).dropna()

        if self.use_returns:
            self._data = log_prices.diff().dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                var = VAR(self._data)
                sel = var.select_order(maxlags=MAX_LAGS)
                self._lag_order = max(sel.bic, 1)
                self._var_fit = var.fit(self._lag_order, ic=None, trend="c")
        else:
            # Test for cointegration on log prices
            self._data = log_prices
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rank_result = select_coint_rank(
                    log_prices, det_order=0, k_ar_diff=2, signif=0.05
                )
                self._coint_rank = int(rank_result.rank)

            if self._coint_rank > 0:
                # VECM
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._vecm_fit = VECM(
                        log_prices,
                        k_ar_diff=2,
                        coint_rank=self._coint_rank,
                        deterministic="co",
                    ).fit()
            else:
                # No cointegration — fall back to VAR on returns
                returns = log_prices.diff().dropna()
                self._data = returns
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    var = VAR(returns)
                    sel = var.select_order(maxlags=MAX_LAGS)
                    self._lag_order = max(sel.bic, 1)
                    self._var_fit = var.fit(self._lag_order, ic=None, trend="c")

        return self

    def model_type(self) -> str:
        if self._vecm_fit is not None:
            return f"VECM(rank={self._coint_rank})"
        return f"VAR(lag={self._lag_order})"

    def granger_causality(self, max_lag: int = 5) -> pd.DataFrame:
        """
        Pairwise Granger causality p-value matrix.

        Entry [i, j] = p-value for H₀: "column j does NOT Granger-cause column i".
        A small p-value (< 0.05) means column j's lags improve forecasts of i.

        Parameters
        ----------
        max_lag : int
            Number of lags to include in each causality test.

        Returns
        -------
        pd.DataFrame
            n×n matrix of p-values. Columns and index = commodity names.
        """
        if self._var_fit is None:
            raise RuntimeError("Granger causality requires a VAR model (use_returns=True).")

        from statsmodels.tsa.stattools import grangercausalitytests

        cols = self._cols
        pmat = pd.DataFrame(np.nan, index=cols, columns=cols)

        for caused in cols:
            for cause in cols:
                if caused == cause:
                    pmat.loc[caused, cause] = np.nan
                    continue
                xy = self._data[[caused, cause]].dropna()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc_res = grangercausalitytests(xy, maxlag=max_lag, verbose=False)
                    # Take the p-value at max_lag from the F-test
                    p = gc_res[max_lag][0]["ssr_ftest"][1]
                    pmat.loc[caused, cause] = round(p, 4)

        return pmat

    def impulse_response(
        self,
        steps: int = 10,
        shock_commodity: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Orthogonalised Impulse Response Functions over `steps` periods.

        A one-standard-deviation shock to `shock_commodity` (or to every
        commodity if None) and the response in all other commodities.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (shocked_commodity, responding_commodity).
            Index: period 0 … steps.
        """
        if self._var_fit is None:
            raise RuntimeError("IRF requires a fitted VAR model.")

        irf = self._var_fit.irf(steps)
        orth_irfs = irf.orth_irfs    # shape (steps+1, n, n): [step, response, shock]

        cols = self._cols
        records = {}
        shocked_cols = [shock_commodity] if shock_commodity else cols

        for shock_idx, shock_name in enumerate(cols):
            if shock_name not in shocked_cols:
                continue
            for resp_idx, resp_name in enumerate(cols):
                key = (shock_name, resp_name)
                records[key] = orth_irfs[:, resp_idx, shock_idx]

        df = pd.DataFrame(records, index=range(steps + 1))
        df.index.name = "period"
        df.columns = pd.MultiIndex.from_tuples(
            df.columns, names=["shock", "response"]
        )
        return df

    def fevd(self, steps: int = 10) -> pd.DataFrame:
        """
        Forecast Error Variance Decomposition at `steps` horizon.

        Entry [response, shock] = fraction of response's forecast error variance
        at horizon `steps` attributable to shocks in `shock`.

        Returns
        -------
        pd.DataFrame
            n×n DataFrame. Rows = responding commodity, columns = shock source.
            Values sum to 1.0 across each row.
        """
        if self._var_fit is None:
            raise RuntimeError("FEVD requires a fitted VAR model.")

        fevd_result = self._var_fit.fevd(steps)
        # fevd_result.decomp shape: (n, steps+1, n) → [variable, step, shock]
        decomp = fevd_result.decomp[:, -1, :]    # at horizon = steps

        return pd.DataFrame(
            decomp.round(4),
            index=self._cols,
            columns=self._cols,
        )

    def cointegration_summary(self) -> Optional[dict]:
        """
        Return VECM cointegrating vectors and adjustment speeds if fitted.
        None if VAR was used instead.
        """
        if self._vecm_fit is None:
            return None

        return {
            "rank":              self._coint_rank,
            "alpha":             pd.DataFrame(
                                     self._vecm_fit.alpha,
                                     index=self._cols,
                                     columns=[f"EC{i+1}" for i in range(self._coint_rank)],
                                 ).round(6),
            "beta":              pd.DataFrame(
                                     self._vecm_fit.beta,
                                     index=self._cols,
                                     columns=[f"EC{i+1}" for i in range(self._coint_rank)],
                                 ).round(6),
        }

    def forecast(self, steps: int = 5) -> pd.DataFrame:
        """
        h-step-ahead point forecast for all commodities in the group.

        Returns
        -------
        pd.DataFrame
            Columns = commodity names. Index = periods 1 … steps.
        """
        if self._var_fit is None:
            raise RuntimeError("Forecast requires a fitted VAR model.")

        lag = self._lag_order
        y_input = self._data.values[-lag:]
        fc = self._var_fit.forecast(y_input, steps=steps)
        return pd.DataFrame(fc, columns=self._cols, index=range(1, steps + 1))


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_energy_system(prices: pd.DataFrame) -> dict:
    """
    Fit VAR on the energy complex and return the full suite of outputs.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices. Must include "WTI Crude Oil" and "Natural Gas (Henry Hub)".

    Returns
    -------
    dict
        Keys: var (CommodityVAR), model_type, granger, irf, fevd, forecast.
    """
    var = CommodityVAR(group="energy", use_returns=True)
    var.fit(prices)

    return {
        "var":         var,
        "model_type":  var.model_type(),
        "granger":     var.granger_causality(max_lag=5),
        "irf":         var.impulse_response(steps=10),
        "fevd":        var.fevd(steps=10),
        "forecast":    var.forecast(steps=5),
    }


def run_grains_system(prices: pd.DataFrame) -> dict:
    """Same as run_energy_system but for the grains complex."""
    var = CommodityVAR(group="grains", use_returns=True)
    var.fit(prices)

    return {
        "var":         var,
        "model_type":  var.model_type(),
        "granger":     var.granger_causality(max_lag=5),
        "irf":         var.impulse_response(steps=10),
        "fevd":        var.fevd(steps=10),
        "forecast":    var.forecast(steps=5),
    }
