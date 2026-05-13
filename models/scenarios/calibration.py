"""
Calibration — empirical coverage audit for scenario bands.

For each day in a holdout window, refit the provider on data up to that
day and check whether the realized 1-step log return fell inside [bear,
bull]. Empirical coverage should approach (bull_q - bear_q); large
deviations mean the band is mis-calibrated and the visible fan chart
overstates or understates true uncertainty.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd


def empirical_coverage(
    returns: pd.Series,
    build_band: Callable[[pd.Series], "ScenarioBand"],
    window: int = 60,
    min_train: int = 250,
) -> dict:
    """
    Rolling-origin 1-step coverage test.

    Parameters
    ----------
    returns : pd.Series
        Full log-return series, DatetimeIndex ascending.
    build_band : callable
        Function that takes a returns series and returns a 1-step ScenarioBand.
        Will be called ``window`` times (slow — keep window small for ARIMA).
    window : int
        Number of trailing observations to evaluate against.
    min_train : int
        Minimum training length to require before evaluation starts.

    Returns
    -------
    dict
        {
            "expected": float,        # nominal coverage = bull_q - bear_q
            "empirical": float,       # fraction of realized returns inside [bear, bull]
            "n_eval": int,
            "hits": np.ndarray,       # boolean array of length n_eval
            "realized": np.ndarray,
            "bear": np.ndarray,
            "bull": np.ndarray,
        }
    """
    returns = returns.dropna()
    n = len(returns)
    start = max(min_train, n - window)
    if start >= n - 1:
        raise ValueError(
            f"Not enough data: have {n} observations, need > {min_train + 1}."
        )

    hits = []
    realized = []
    bear_arr = []
    bull_arr = []
    expected = None

    for t in range(start, n - 1):
        train = returns.iloc[:t + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                band = build_band(train)
            except Exception:
                continue
        if band is None:
            continue
        expected = band.bull_q - band.bear_q
        b, u = float(band.bear[0]), float(band.bull[0])
        y = float(returns.iloc[t + 1])
        hits.append(b <= y <= u)
        realized.append(y)
        bear_arr.append(b)
        bull_arr.append(u)

    if not hits:
        return {"expected": expected, "empirical": float("nan"), "n_eval": 0,
                "hits": np.array([]), "realized": np.array([]),
                "bear": np.array([]), "bull": np.array([])}

    hits_np = np.asarray(hits)
    return {
        "expected": expected,
        "empirical": float(hits_np.mean()),
        "n_eval": int(len(hits_np)),
        "hits": hits_np,
        "realized": np.asarray(realized),
        "bear": np.asarray(bear_arr),
        "bull": np.asarray(bull_arr),
    }
