"""
ScenarioBand — the common contract every model family produces.

A band carries three log-return paths over a forecast horizon: the bear
(low quantile), the average (median / point forecast), and the bull (high
quantile). Helpers convert those return paths into price paths anchored to
the most recent observed close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class ScenarioBand:
    """A bear/avg/bull envelope of log-return forecasts."""

    commodity: str
    source_model: str
    asof: pd.Timestamp
    horizon: int
    bear_q: float
    bull_q: float
    mean: np.ndarray
    bear: np.ndarray
    bull: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.bear = np.asarray(self.bear, dtype=float)
        self.bull = np.asarray(self.bull, dtype=float)
        if not (len(self.mean) == len(self.bear) == len(self.bull) == self.horizon):
            raise ValueError(
                f"{self.source_model}: band arrays must all have length {self.horizon}"
            )
        # Enforce ordering: bear ≤ mean ≤ bull at every step
        if np.any(self.bear > self.mean + 1e-9) or np.any(self.mean > self.bull + 1e-9):
            # Don't raise — just clip. Some providers can produce slight
            # inversions on tiny samples; clipping preserves a usable band.
            self.bear = np.minimum(self.bear, self.mean)
            self.bull = np.maximum(self.bull, self.mean)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"bear": self.bear, "mean": self.mean, "bull": self.bull},
            index=pd.RangeIndex(1, self.horizon + 1, name="step"),
        )


def returns_to_price_paths(
    last_price: float,
    band: ScenarioBand,
) -> pd.DataFrame:
    """
    Compound a ScenarioBand's log-return paths into price paths.

    Each column is a cumulative product anchored at ``last_price``. The
    returned frame includes step 0 (= last_price) so the fan chart connects
    to the observed series.
    """
    cum_mean = np.exp(np.cumsum(band.mean))
    cum_bear = np.exp(np.cumsum(band.bear))
    cum_bull = np.exp(np.cumsum(band.bull))

    prices = pd.DataFrame(
        {
            "bear": np.concatenate([[1.0], cum_bear]) * last_price,
            "mean": np.concatenate([[1.0], cum_mean]) * last_price,
            "bull": np.concatenate([[1.0], cum_bull]) * last_price,
        },
        index=pd.RangeIndex(0, band.horizon + 1, name="step"),
    )
    return prices


def attach_business_dates(
    price_paths: pd.DataFrame,
    anchor: pd.Timestamp,
) -> pd.DataFrame:
    """Re-index a price-paths frame on business days starting at ``anchor``."""
    n = len(price_paths)
    idx = pd.bdate_range(start=anchor, periods=n)
    out = price_paths.copy()
    out.index = idx
    return out


def collapse_to_tail(band: "ScenarioBand", side: str = "bear") -> "ScenarioBand":
    """
    Force a band into a deterministic tail scenario.

    The bear (or bull) path is copied into all three slots so downstream
    machinery — especially the causal-ripple engine — treats the chosen
    tail as the assumed truth. Useful for "stress-test mode": *if* the
    bear case plays out, what does the rest of the complex do?
    """
    if side not in ("bear", "bull"):
        raise ValueError("side must be 'bear' or 'bull'")
    path = band.bear if side == "bear" else band.bull
    return ScenarioBand(
        commodity=band.commodity,
        source_model=f"{band.source_model}[stress:{side}]",
        asof=band.asof,
        horizon=band.horizon,
        bear_q=band.bear_q,
        bull_q=band.bull_q,
        mean=path.copy(),
        bear=path.copy(),
        bull=path.copy(),
        diagnostics={**band.diagnostics, "stress_side": side},
    )
