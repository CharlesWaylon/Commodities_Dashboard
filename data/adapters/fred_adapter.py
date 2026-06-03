"""
FredAdapter — FRED (Federal Reserve Economic Data) series as release-dated
fundamental observations.

Wraps the existing ``features.macro_overlays._fetch_fred_series`` helper (one
place that knows the FRED REST call + FRED_API_KEY) and reshapes the result into
the long, release-dated observation table the point-in-time store expects.

PUBLICATION LAG (honesty note)
───────────────────────────────
True vintage-correct release timing requires FRED's ALFRED realtime API
(realtime_start / realtime_end), which returns the exact first-print date of each
observation. The simple REST helper does not, so this adapter approximates the
release_date as ``reference_date + publication_lag_bdays`` (default 0 for daily
market series; pass per-series overrides for laggy macro series, e.g. CPI ≈ 2
weeks). This is a documented approximation, not vintage truth — flagged here and
in MODEL_VERIFICATION_LOG so a later upgrade to ALFRED is an explicit task, and so
no signal silently assumes more precision than we have.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Iterable, Optional

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS, FundamentalAdapter


class FredAdapter(FundamentalAdapter):
    source_name = "fred"

    def __init__(
        self,
        publication_lag_bdays: int = 0,
        per_series_lag_bdays: Optional[Dict[str, int]] = None,
    ):
        self.publication_lag_bdays = int(publication_lag_bdays)
        self.per_series_lag_bdays = dict(per_series_lag_bdays or {})

    def _lag(self, series_id: str) -> int:
        return int(self.per_series_lag_bdays.get(series_id, self.publication_lag_bdays))

    def get_observations(
        self,
        series_ids: Iterable[str],
        start: Optional[str] = None,
    ) -> pd.DataFrame:
        from features.macro_overlays import _fetch_fred_series, _period_to_start_date

        start = start or _period_to_start_date("5y")
        rows = []
        for sid in series_ids:
            series = _fetch_fred_series(sid, start)
            if series is None or series.empty:
                continue
            lag = self._lag(sid)
            for ref_date, value in series.items():
                ref = pd.Timestamp(ref_date)
                release = (ref + pd.tseries.offsets.BDay(lag)) if lag else ref
                rows.append(
                    {
                        "source": self.source_name,
                        "series_id": sid,
                        "reference_date": ref.date(),
                        "release_date": release.date(),
                        "value": float(value),
                    }
                )
        if not rows:
            return self.empty_observations()
        return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
