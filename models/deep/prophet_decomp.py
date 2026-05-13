"""
Meta Prophet trend + seasonality decomposer for commodity prices.

Prophet fits an additive (or multiplicative) model:

    y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)

Trend is piecewise linear with automatic changepoint detection. Seasonality
is represented as a Fourier series. This makes Prophet highly interpretable:
you can read the decomposed components as a story about the commodity's
structural drivers.

Why per-commodity seasonal configuration?
  Agricultural commodities (Corn, Wheat, Soybeans) have strong yearly cycles
  driven by planting/growing/harvest calendars. Natural Gas has both yearly
  (heating vs. cooling seasons) and weekly (industrial demand drops on weekends)
  seasonality. Energy and metals are more trend-driven and need fewer seasonal
  components — over-parameterising their seasonality leads to nonsense cycles.

Changepoint detection
  Prophet identifies dates where the trend slope changes significantly. These
  correspond to structural breaks — supply shocks, policy reversals, demand
  collapses. The `changepoint_summary()` method matches detected changepoints to
  a catalog of known macro events, making them immediately interpretable for
  a club discussion ("Prophet found a break in WTI on 2022-02-28 — that's the
  week oil markets priced in the Ukraine invasion").

Fitting on log prices (not returns)
  Prophet decomposes levels, not differences. Log prices are stationary enough
  for Prophet's piecewise trend to handle and preserve the trend/seasonality
  interpretation that raw returns destroy.

Usage
-----
    from models.deep.prophet_decomp import ProphetDecomposer, run_all

    prices = load_price_matrix()

    dc = ProphetDecomposer(commodity="Natural Gas (Henry Hub)")
    dc.fit(prices)

    forecast = dc.forecast(periods=30)         # pd.DataFrame: ds, yhat, bands
    comps    = dc.components()                 # dict: trend, weekly, yearly, residual
    cps      = dc.changepoint_summary()        # matched to macro events
    regime   = dc.trend_regime()               # "accelerating" / "decelerating" / "stable"

    all_results = run_all(prices)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import spearmanr

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

from models.config import MODELING_COMMODITIES, RANDOM_SEED, TEST_FRACTION


def _require_prophet():
    if not _PROPHET_AVAILABLE:
        raise ImportError(
            "prophet is required for ProphetDecomposer.  "
            "Install with: pip install prophet"
        )


# ── Seasonal profiles ──────────────────────────────────────────────────────────

# Fourier order controls how many harmonics are used for each seasonal component.
# Higher order → more flexible seasonality, more risk of overfitting.
_SEASONAL_CONFIG = {
    # Agricultural: strong yearly harvest/planting cycle + mild weekly
    "Corn (CBOT)":              {"yearly": True, "weekly": True,  "yearly_order": 8,  "mode": "multiplicative"},
    "Wheat (CBOT SRW)":        {"yearly": True, "weekly": True,  "yearly_order": 8,  "mode": "multiplicative"},
    "Soybeans (CBOT)":         {"yearly": True, "weekly": True,  "yearly_order": 8,  "mode": "multiplicative"},
    "Live Cattle":             {"yearly": True, "weekly": False, "yearly_order": 5,  "mode": "additive"},
    # Natural Gas: both yearly (heating/cooling degree days) and weekly
    "Natural Gas (Henry Hub)": {"yearly": True, "weekly": True,  "yearly_order": 10, "mode": "multiplicative"},
    # Energy & metals: trend-driven, minimal seasonality
    "WTI Crude Oil":           {"yearly": False, "weekly": False, "yearly_order": 3,  "mode": "additive"},
    "Gold (COMEX)":            {"yearly": False, "weekly": False, "yearly_order": 3,  "mode": "additive"},
    "Silver (COMEX)":          {"yearly": False, "weekly": False, "yearly_order": 3,  "mode": "additive"},
    "Copper (COMEX)":          {"yearly": False, "weekly": False, "yearly_order": 3,  "mode": "additive"},
}

_DEFAULT_CONFIG = {"yearly": False, "weekly": False, "yearly_order": 3, "mode": "additive"}

# ── Known macro events for changepoint matching ────────────────────────────────
MACRO_EVENTS = {
    "2020-03-20": "Covid demand collapse (WTI/Ag)",
    "2020-04-20": "WTI negative price event",
    "2021-11-26": "Omicron variant discovery",
    "2022-02-24": "Russia invasion of Ukraine",
    "2022-03-08": "LME nickel short squeeze",
    "2022-03-16": "Fed first rate hike (2022 cycle)",
    "2022-06-15": "Fed 75bp hike — peak hawkishness signal",
    "2022-10-05": "OPEC+ 2mb/d production cut",
    "2023-03-10": "SVB collapse / banking stress",
    "2023-05-04": "Fed final hike of 2022-23 cycle",
}

# Match a detected changepoint to the nearest macro event within this window
MATCH_WINDOW_DAYS = 30


class ProphetDecomposer:
    """
    Prophet decomposer for a single commodity's price history.

    Parameters
    ----------
    commodity : str
        Display name from MODELING_COMMODITIES.
    changepoint_prior_scale : float
        Controls flexibility of the trend. Higher → more changepoints detected.
        0.05 is Prophet's default; 0.01 is more conservative for noisy series.
    """

    def __init__(
        self,
        commodity: str = "Natural Gas (Henry Hub)",
        changepoint_prior_scale: float = 0.05,
    ):
        _require_prophet()
        self.commodity = commodity
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model: Optional[Prophet] = None
        self._forecast: Optional[pd.DataFrame] = None
        self._history: Optional[pd.DataFrame] = None

    # ── private ────────────────────────────────────────────────────────────────

    def _build_prophet_df(self, prices: pd.DataFrame) -> pd.DataFrame:
        series = np.log(prices[self.commodity]).dropna()
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        return df

    def _make_model(self) -> "Prophet":
        cfg = _SEASONAL_CONFIG.get(self.commodity, _DEFAULT_CONFIG)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=cfg["mode"],
                yearly_seasonality=False,   # add manually below for control
                weekly_seasonality=cfg["weekly"],
                daily_seasonality=False,
                interval_width=0.95,
            )
            if cfg["yearly"]:
                m.add_seasonality(
                    name="yearly",
                    period=365.25,
                    fourier_order=cfg["yearly_order"],
                    mode=cfg["mode"],
                )
        return m

    # ── public interface ───────────────────────────────────────────────────────

    def fit(self, prices: pd.DataFrame) -> "ProphetDecomposer":
        """
        Fit Prophet to the log-price history of this commodity.

        Parameters
        ----------
        prices : pd.DataFrame
            Closing prices from load_price_matrix().
        """
        df = self._build_prophet_df(prices)
        self._history = df
        self._model = self._make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(df)
        return self

    def forecast(self, periods: int = 30) -> pd.DataFrame:
        """
        Generate an in-sample + `periods`-day forward forecast.

        Returns
        -------
        pd.DataFrame
            Columns: ds, yhat (log-price), yhat_lower, yhat_upper,
                     yhat_exp (exp(yhat) ≈ price level), trend, plus
                     any seasonality components.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        future = self._model.make_future_dataframe(periods=periods, freq="B")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._model.predict(future)
        fc["yhat_exp"]       = np.exp(fc["yhat"])
        fc["yhat_lower_exp"] = np.exp(fc["yhat_lower"])
        fc["yhat_upper_exp"] = np.exp(fc["yhat_upper"])
        self._forecast = fc
        return fc

    def components(self) -> dict:
        """
        Decomposed time-series components from the fitted model.

        Returns
        -------
        dict with keys:
            trend   — pd.Series of the fitted trend component
            weekly  — pd.Series of weekly seasonality (if fitted)
            yearly  — pd.Series of yearly seasonality (if fitted)
            residual — pd.Series of (actual log-price − fitted yhat)
        """
        if self._forecast is None:
            self.forecast(periods=0)

        fc = self._forecast
        hist = self._history

        result = {"trend": pd.Series(fc["trend"].values, index=fc["ds"])}

        if "weekly" in fc.columns:
            result["weekly"] = pd.Series(fc["weekly"].values, index=fc["ds"])

        if "yearly" in fc.columns:
            result["yearly"] = pd.Series(fc["yearly"].values, index=fc["ds"])

        # Residual only for in-sample dates
        in_sample = fc[fc["ds"].isin(hist["ds"])]
        actual_y  = hist.set_index("ds")["y"]
        fitted_y  = in_sample.set_index("ds")["yhat"]
        residual  = (actual_y - fitted_y).dropna()
        result["residual"] = residual

        return result

    def changepoint_summary(self) -> pd.DataFrame:
        """
        Detected trend changepoints, matched to the nearest macro event.

        Returns
        -------
        pd.DataFrame
            Columns: changepoint_date, delta (slope change), magnitude_rank,
                     nearest_macro_event, days_from_event.
            Sorted by |delta| descending.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        cps = self._model.changepoints
        if len(cps) == 0:
            return pd.DataFrame()

        deltas = self._model.params["delta"].mean(axis=0)
        # changepoints may be fewer than deltas (Prophet pads with zeros)
        n = min(len(cps), len(deltas))
        cp_dates = pd.to_datetime(cps[:n].values).normalize()
        delta_vals = deltas[:n]

        macro_dates = {pd.Timestamp(k): v for k, v in MACRO_EVENTS.items()}

        rows = []
        for cp, dv in zip(cp_dates, delta_vals):
            nearest_label, nearest_days = "—", None
            for md, label in macro_dates.items():
                diff = abs((cp - md).days)
                if nearest_days is None or diff < nearest_days:
                    nearest_days = diff
                    nearest_label = label if diff <= MATCH_WINDOW_DAYS else "—"

            rows.append({
                "changepoint_date":  cp.date(),
                "delta":             round(float(dv), 5),
                "nearest_macro":     nearest_label,
                "days_from_event":   nearest_days,
            })

        df = pd.DataFrame(rows)
        df["magnitude_rank"] = df["delta"].abs().rank(ascending=False).astype(int)
        return df.sort_values("magnitude_rank").reset_index(drop=True)

    def trend_regime(self) -> str:
        """
        Classify the current trend as "accelerating", "decelerating", or "stable"
        based on the second derivative of the trend component over the last 90 days.
        """
        if self._forecast is None:
            self.forecast(periods=0)
        trend = self._forecast.set_index("ds")["trend"].iloc[-90:]
        d2 = np.gradient(np.gradient(trend.values))
        mean_d2 = d2.mean()
        if mean_d2 > 1e-6:
            return "accelerating"
        if mean_d2 < -1e-6:
            return "decelerating"
        return "stable"

    def seasonal_strength(self) -> dict:
        """
        Ratio of seasonal variance to total variance — measures how much
        of the price variation is explained by calendar patterns.

        Returns
        -------
        dict
            Keys: weekly_strength, yearly_strength (each in [0, 1]).
            Values near 1 mean the season dominates; near 0 means it's noise.
        """
        comps = self.components()
        total_var = comps["residual"].var() + 1e-10
        result = {}
        for key in ("weekly", "yearly"):
            if key in comps:
                result[f"{key}_strength"] = round(
                    float(comps[key].var() / (comps[key].var() + total_var)), 4
                )
        return result


# ── Optuna hyperparameter tuner ────────────────────────────────────────────────

def tune_prophet(
    prices: pd.DataFrame,
    commodity: str,
    n_trials: int = 20,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Optuna search over Prophet hyperparameters for a single commodity.

    Objective: Spearman IC of implied 1-day log-return forecasts on the
    held-out 20% test period.

    Parameters
    ----------
    prices : pd.DataFrame
        Full closing price matrix.
    commodity : str
        Column name in prices to tune.
    n_trials : int
        Number of Optuna trials.
    seed : int
        Reproducibility seed for TPESampler.

    Returns
    -------
    dict
        best_params, best_ic, n_trials, best_trial.
    """
    _require_prophet()
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    # Build log-price DataFrame in Prophet format
    series = np.log(prices[commodity]).dropna()
    df = series.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.reset_index(drop=True)

    split = int(len(df) * (1 - TEST_FRACTION))
    df_train = df.iloc[:split].copy()
    df_test  = df.iloc[split:].copy()

    cfg = _SEASONAL_CONFIG.get(commodity, _DEFAULT_CONFIG)

    def objective(trial: "optuna.Trial") -> float:
        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        sps = trial.suggest_float("seasonality_prior_scale", 0.1, 20.0, log=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                changepoint_prior_scale=cps,
                seasonality_prior_scale=sps,
                seasonality_mode=cfg["mode"],
                yearly_seasonality=False,
                weekly_seasonality=cfg["weekly"],
                daily_seasonality=False,
                interval_width=0.95,
            )
            if cfg["yearly"]:
                m.add_seasonality(
                    name="yearly",
                    period=365.25,
                    fourier_order=cfg["yearly_order"],
                    mode=cfg["mode"],
                )
            m.fit(df_train)
            future = m.make_future_dataframe(periods=len(df_test), freq="B")
            fc = m.predict(future)

        # Align forecast to test dates, compute 1-day return IC
        fc_indexed = fc.set_index("ds")["yhat"]
        actual_indexed = df_test.set_index("ds")["y"]
        aligned = pd.concat(
            [fc_indexed.rename("pred"), actual_indexed.rename("actual")], axis=1
        ).dropna()
        if len(aligned) < 10:
            return 0.0

        implied_ret = aligned["pred"].diff().dropna()
        actual_ret  = aligned["actual"].diff().dropna()
        min_len = min(len(implied_ret), len(actual_ret))
        if min_len < 5:
            return 0.0

        ic, _ = spearmanr(implied_ret.iloc[:min_len].values,
                          actual_ret.iloc[:min_len].values)
        return float(ic) if not np.isnan(ic) else 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"prophet_{commodity}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_ic":     round(study.best_value, 4),
        "n_trials":    n_trials,
        "best_trial":  study.best_trial.number,
    }


# ── Convenience runner ─────────────────────────────────────────────────────────

def run_all(
    prices: pd.DataFrame,
    commodities: Optional[dict] = None,
    forecast_periods: int = 30,
) -> dict:
    """
    Fit Prophet decomposer for every commodity with seasonal configuration.

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices from load_price_matrix().
    forecast_periods : int
        Days ahead to forecast beyond the last price date.

    Returns
    -------
    dict
        Keys = commodity display names.
        Values = dicts: decomposer, forecast (df), components (dict),
                        changepoints (df), trend_regime (str),
                        seasonal_strength (dict).
    """
    _require_prophet()
    if commodities is None:
        commodities = MODELING_COMMODITIES

    results = {}
    for name in commodities:
        if name not in prices.columns:
            continue
        dc = ProphetDecomposer(commodity=name)
        dc.fit(prices)
        results[name] = {
            "decomposer":        dc,
            "forecast":          dc.forecast(periods=forecast_periods),
            "components":        dc.components(),
            "changepoints":      dc.changepoint_summary(),
            "trend_regime":      dc.trend_regime(),
            "seasonal_strength": dc.seasonal_strength(),
        }

    return results
