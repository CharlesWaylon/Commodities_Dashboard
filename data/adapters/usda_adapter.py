"""
UsdaAdapter — USDA NASS QuickStats agricultural fundamentals (ending stocks,
production, grain stocks) as release-dated fundamental observations.

WHAT IT EMITS
─────────────
One observation per requested ``series_ids`` entry, where a series id is an
opaque key the caller maps to a QuickStats query via ``series_queries`` (a dict of
series_id -> query-param dict, e.g. commodity/statisticcat/agg-level filters).
``value`` is the reported figure; ``reference_date`` is the end of the reference
period (year/marketing-year).

RELEASE TIMING (point-in-time correctness)
──────────────────────────────────────────
USDA reports (WASDE, Grain Stocks, Crop Production) are released on scheduled
dates well after the reference period closes. QuickStats exposes
``load_time`` / report metadata but not a clean first-publish date per row, so we
approximate ``release_date = reference_date + publication_lag_days`` (default 30;
override per series). Documented approximation, flagged in MODEL_VERIFICATION_LOG;
a later upgrade to the published WASDE release calendar is an explicit task.

DATA SOURCE
───────────
USDA NASS QuickStats API (https://quickstats.nass.usda.gov/api/api_GET/).
Requires the free ``USDA_QUICKSTATS_KEY`` env var. Network/key failures return an
empty frame rather than raising.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS, FundamentalAdapter

_DEFAULT_LAG_DAYS = 30


class UsdaAdapter(FundamentalAdapter):
    source_name = "usda"

    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    def __init__(
        self,
        series_queries: Optional[Dict[str, dict]] = None,
        publication_lag_days: int = _DEFAULT_LAG_DAYS,
        per_series_lag_days: Optional[Dict[str, int]] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.series_queries = dict(series_queries or {})
        self.publication_lag_days = int(publication_lag_days)
        self.per_series_lag_days = dict(per_series_lag_days or {})
        self._api_key = api_key if api_key is not None else os.getenv("USDA_QUICKSTATS_KEY", "")
        self.timeout = int(timeout)

    def _lag(self, series_id: str) -> int:
        return int(self.per_series_lag_days.get(series_id, self.publication_lag_days))

    def _reference_date(self, rec: dict) -> Optional[pd.Timestamp]:
        """End-of-period date from QuickStats fields (year, optional end-code)."""
        end_code = rec.get("reference_period_desc") or rec.get("end_code")
        year = rec.get("year")
        if year is None:
            return None
        try:
            yr = int(year)
        except (ValueError, TypeError):
            return None
        # Default to calendar year-end; QuickStats marketing-year rows are
        # year-stamped, and the lag-based release handles the publication offset.
        return pd.Timestamp(year=yr, month=12, day=31)

    # ── pure transform (testable core) ────────────────────────────────────────
    def _shape(self, series_id: str, data_rows: List[dict]) -> pd.DataFrame:
        lag = self._lag(series_id)
        rows = []
        for rec in data_rows:
            ref = self._reference_date(rec)
            raw_val = rec.get("Value")
            if ref is None or raw_val in (None, "", "(D)", "(NA)", "(Z)"):
                continue
            try:
                val = float(str(raw_val).replace(",", ""))
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "source": self.source_name,
                    "series_id": series_id,
                    "reference_date": ref.date(),
                    "release_date": (ref + timedelta(days=lag)).date(),
                    "value": val,
                }
            )
        if not rows:
            return self.empty_observations()
        return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)

    # ── best-effort fetch ─────────────────────────────────────────────────────
    def get_observations(
        self,
        series_ids: Iterable[str],
        start: Optional[str] = None,
    ) -> pd.DataFrame:
        import requests

        if not self._api_key:
            return self.empty_observations()

        frames: List[pd.DataFrame] = []
        for sid in series_ids:
            query = self.series_queries.get(sid)
            if not query:
                continue
            params = {"key": self._api_key, "format": "JSON", **query}
            if start:
                params.setdefault("year__GE", pd.Timestamp(start).year)
            try:
                resp = requests.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                shaped = self._shape(str(sid), resp.json().get("data", []))
            except Exception:
                continue
            if not shaped.empty:
                frames.append(shaped)

        if not frames:
            return self.empty_observations()
        return pd.concat(frames, ignore_index=True)
