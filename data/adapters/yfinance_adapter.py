"""
YFinanceAdapter — price supply from Yahoo Finance / the local DB mirror.

Wraps the existing, battle-tested fetchers in ``models.data_loader`` rather than
re-implementing yfinance plumbing, so there is exactly one place that knows how to
talk to Yahoo. (Transitional lateral import; when the data layer fully owns
ingestion, models.data_loader will be the thin caller instead.)

``prefer_db=True`` reads the already-ingested local Postgres mirror (fast, works
offline, the default the harness uses); ``prefer_db=False`` hits Yahoo live.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from data.adapters.base import PriceAdapter
from data.universe import name_for_ticker


class YFinanceAdapter(PriceAdapter):
    source_name = "yfinance"

    def __init__(self, prefer_db: bool = True):
        self.prefer_db = prefer_db

    def get_prices(
        self,
        tickers: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        tickers = list(tickers)
        # Build the {display_name: ticker} sub-mapping the loaders expect.
        commodities = {}
        for t in tickers:
            name = name_for_ticker(t) or t
            commodities[name] = t

        if self.prefer_db:
            from models.data_loader import load_price_matrix_from_db

            try:
                panel = load_price_matrix_from_db(commodities)
            except Exception:
                panel = self._fetch_live(commodities)
        else:
            panel = self._fetch_live(commodities)

        if start is not None:
            panel = panel.loc[pd.Timestamp(start):]
        if end is not None:
            panel = panel.loc[:pd.Timestamp(end)]
        return panel

    @staticmethod
    def _fetch_live(commodities: dict) -> pd.DataFrame:
        from models.data_loader import load_price_matrix

        return load_price_matrix(commodities)
