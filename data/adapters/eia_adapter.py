"""
EiaAdapter — U.S. Energy Information Administration weekly stocks (petroleum &
natural gas) as release-dated fundamental observations.

WHAT IT EMITS
─────────────
One observation per series per period. ``series_ids`` are EIA v2 series ids
(e.g. ``PET.WCESTUS1.W`` = weekly U.S. crude oil ending stocks). ``value`` is the
reported level; ``reference_date`` is the period the figure describes.

RELEASE TIMING (point-in-time correctness)
──────────────────────────────────────────
EIA weekly reports lag their reference week:
  • Weekly Petroleum Status Report — week ending Friday, published the following
    Wednesday (~5 calendar days).
  • Weekly Natural Gas Storage Report — week ending Friday, published Thursday
    (~6 calendar days).
The EIA API does not return the publish timestamp, so we approximate
``release_date = reference_date + publication_lag_days`` (default 5; pass
``per_series_lag_days`` for natural-gas = 6). Documented approximation, flagged in
MODEL_VERIFICATION_LOG.

DATA SOURCE
───────────
EIA API v2 (https://api.eia.gov/v2/seriesid/{id}?api_key=...). Requires the free
``EIA_API_KEY`` env var. Network/key failures return an empty frame rather than
raising, so a scheduled ingest never crashes.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS, FundamentalAdapter

_DEFAULT_LAG_DAYS = 5


class EiaAdapter(FundamentalAdapter):
    source_name = "eia"

    base_url = "https://api.eia.gov/v2/seriesid/"

    def __init__(
        self,
        publication_lag_days: int = _DEFAULT_LAG_DAYS,
        per_series_lag_days: Optional[Dict[str, int]] = None,
        api_key: Optional[str] = None,
        timeout: int = 20,
    ):
        self.publication_lag_days = int(publication_lag_days)
        self.per_series_lag_days = dict(per_series_lag_days or {})
        self._api_key = api_key if api_key is not None else os.getenv("EIA_API_KEY", "")
        self.timeout = int(timeout)

    def _lag(self, series_id: str) -> int:
        return int(self.per_series_lag_days.get(series_id, self.publication_lag_days))

    # ── pure transform (testable core) ────────────────────────────────────────
    def _shape(self, series_id: str, data_rows: List[dict]) -> pd.DataFrame:
        lag = self._lag(series_id)
        rows = []
        for rec in data_rows:
            period = rec.get("period")
            value = rec.get("value")
            if period is None or value in (None, ".", ""):
                continue
            try:
                ref = pd.Timestamp(period)
                val = float(value)
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
            params = {"api_key": self._api_key, "out": "json"}
            if start:
                params["start"] = start
            try:
                resp = requests.get(f"{self.base_url}{sid}", params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json().get("response", {})
                shaped = self._shape(str(sid), payload.get("data", []))
            except Exception:
                continue
            if not shaped.empty:
                frames.append(shaped)

        if not frames:
            return self.empty_observations()
        return pd.concat(frames, ignore_index=True)
