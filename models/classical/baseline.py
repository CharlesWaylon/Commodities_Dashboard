"""
Classical baseline models for benchmarking.

Before evaluating any fancy model (quantum or otherwise), you need a
baseline to beat. Two simple baselines here:

  1. PersistenceModel — predicts tomorrow's return = today's return.
     The "naive" benchmark. If your model can't beat this, it's not useful.

  2. RollingMeanModel — predicts tomorrow's return = mean of the last N returns.
     Slightly smoother than persistence. Common in quant research as the
     "dumb" baseline before introducing any signal.

Both expose the same fit/predict/score interface as QuantumHybridRegressor
so they can be compared directly in experiment scripts.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from models.config import ROLLING_MEAN_BASELINE_WINDOW


class PersistenceModel:
    """
    Predicts next-step return = current-step return (lag-1 carry).

    No fitting required — this is a zero-parameter baseline.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        self._last_train_return = float(y[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # The first column of X is expected to be the 1-day lagged return
        # (matches features.py column ordering after variance selection).
        # We use the first feature column as a proxy for "yesterday's return."
        return X[:, 0]

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(r2_score(y_true, self.predict(X)))


class RollingMeanModel:
    """
    Predicts next-step return = mean of the last `window` returns in training data.

    Parameters
    ----------
    window : int
        How many trailing returns to average. Default from config.
    """

    def __init__(self, window: int = ROLLING_MEAN_BASELINE_WINDOW):
        self.window = window
        self._forecast = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RollingMeanModel":
        # Compute rolling mean over the last `window` training targets
        self._forecast = float(np.mean(y[-self.window:]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._forecast is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(len(X), self._forecast)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        return float(r2_score(y_true, self.predict(X)))
