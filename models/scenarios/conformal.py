"""
Split-conformal prediction intervals for the project's ML forecasters.

Background
----------
Conformal prediction takes any point-prediction model and turns it into a
calibrated interval predictor. The "split" variant is the cheapest form:

  1. Train the model on a training fold.
  2. On a held-out calibration fold, score signed residuals e_i = y_i − ŷ_i.
  3. The prediction interval at quantiles (q_lo, q_hi) is
        [ŷ_new + quantile(e, q_lo),  ŷ_new + quantile(e, q_hi)].

Under exchangeability the interval has finite-sample coverage at least
(q_hi − q_lo). For financial time series strict exchangeability is violated
(non-stationarity), so we treat empirical coverage on a rolling holdout as
the source of truth and surface it via models.scenarios.calibration.

Multi-step horizons
-------------------
The project's ML models are 1-step predictors. To draw a fan chart spanning
several days we widen the conformal interval by √h (random-walk scaling on
the residuals). This keeps the band shape comparable to ARIMA/VAR fans
while making no claim about a multi-step joint distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConformalCalibration:
    """Cached split-conformal calibration for a single point predictor."""

    residuals: np.ndarray              # signed residuals on calibration fold
    point_forecast: float              # latest 1-step point prediction (log return)

    def interval(self, bear_q: float, bull_q: float) -> tuple[float, float, float]:
        """Return (bear, mean, bull) at the requested quantiles for h=1."""
        if len(self.residuals) < 20:
            raise ValueError(
                f"Need ≥20 calibration residuals; got {len(self.residuals)}."
            )
        bear_off = float(np.quantile(self.residuals, bear_q))
        bull_off = float(np.quantile(self.residuals, bull_q))
        return (
            self.point_forecast + bear_off,
            self.point_forecast,
            self.point_forecast + bull_off,
        )

    def horizon_bands(
        self,
        horizon: int,
        bear_q: float,
        bull_q: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand the 1-step conformal interval to ``horizon`` steps using a
        √h random-walk widening on the residual offsets.

        Returns
        -------
        (mean, bear, bull)  — each np.ndarray of length ``horizon``.
        """
        bear_off = float(np.quantile(self.residuals, bear_q))
        bull_off = float(np.quantile(self.residuals, bull_q))
        steps = np.arange(1, horizon + 1)
        sqrt_h = np.sqrt(steps)
        mean = np.full(horizon, self.point_forecast)
        bear = self.point_forecast + bear_off * sqrt_h
        bull = self.point_forecast + bull_off * sqrt_h
        return mean, bear, bull


def calibrate_from_model(
    point_preds_test: pd.Series,
    realized_test: pd.Series,
    next_point_forecast: float,
) -> ConformalCalibration:
    """
    Build a ConformalCalibration from a model's out-of-sample predictions.

    Parameters
    ----------
    point_preds_test : pd.Series
        Model's predictions on its holdout (e.g. TEST_FRACTION) split.
    realized_test : pd.Series
        Realized log returns aligned to ``point_preds_test``.
    next_point_forecast : float
        The model's prediction for the next forecast step (used as the band centre).

    Returns
    -------
    ConformalCalibration
    """
    df = pd.concat(
        [point_preds_test.rename("pred"), realized_test.rename("y")], axis=1
    ).dropna()
    if df.empty:
        raise ValueError("calibrate_from_model: no overlapping test predictions / realized.")
    residuals = (df["y"] - df["pred"]).to_numpy()
    return ConformalCalibration(
        residuals=residuals,
        point_forecast=float(next_point_forecast),
    )
