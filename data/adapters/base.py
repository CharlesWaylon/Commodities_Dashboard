"""
Source-adapter interfaces — the "free now, paid later" hinge.

Every data source (free or paid) is hidden behind one of two ABCs. Callers depend
on the ABC, never on the concrete class, so swapping a free feed for a paid vendor
curve later is a one-line wiring change with zero impact on the layers above.

Two shapes:

  PriceAdapter        — returns a wide price PANEL (DatetimeIndex × instrument).
  FundamentalAdapter  — returns a long, RELEASE-DATED observation table. Every row
                        carries BOTH a reference_date (the period the datum
                        describes) and a release_date (when it became public).
                        This is the raw material of the point-in-time store: a
                        signal as-of date t may only read rows with
                        release_date <= t (publication-lag-correct). See
                        data/fundamental_store.py.

OBSERVATION SCHEMA (FundamentalAdapter.get_observations)
─────────────────────────────────────────────────────────
A DataFrame with columns:
    source         str    — adapter source_name
    series_id      str    — vendor series identifier
    reference_date date   — the date/period the value describes
    release_date   date   — when the value was first published (publication lag)
    value          float
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, List, Optional

import pandas as pd

OBSERVATION_COLUMNS: List[str] = [
    "source",
    "series_id",
    "reference_date",
    "release_date",
    "value",
]


class PriceAdapter(ABC):
    """Returns price panels keyed by instrument ticker."""

    source_name: str = ""

    @abstractmethod
    def get_prices(
        self,
        tickers: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Wide close-price panel: DatetimeIndex rows, one column per ticker."""
        raise NotImplementedError


class FundamentalAdapter(ABC):
    """Returns release-dated fundamental observations (the PIT raw material)."""

    source_name: str = ""

    @abstractmethod
    def get_observations(
        self,
        series_ids: Iterable[str],
        start: Optional[str] = None,
    ) -> pd.DataFrame:
        """Long observation table with OBSERVATION_COLUMNS (see module docstring)."""
        raise NotImplementedError

    @staticmethod
    def empty_observations() -> pd.DataFrame:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
