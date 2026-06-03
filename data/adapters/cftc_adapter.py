"""
CftcAdapter — CFTC Commitments of Traders (COT) "managed money" net positioning
as release-dated fundamental observations.

WHAT IT EMITS
─────────────
For each requested contract (``series_ids`` = CFTC contract-market codes) one
observation per weekly report: ``value = managed_money_long − managed_money_short``
(net managed-money contracts), the speculative-flow fundamental used downstream.

RELEASE TIMING (point-in-time correctness)
──────────────────────────────────────────
The COT report references positions held on **Tuesday** but is published the
following **Friday** at 15:30 ET (a 3-calendar-day lag; longer around federal
holidays, but the public dataset does not carry the actual publish timestamp).
We therefore set ``release_date = reference_date + 3 days`` so a signal as-of date
``t`` only sees a report once it was actually public. This is a documented
approximation (holiday weeks can slip a day); flagged here and in
MODEL_VERIFICATION_LOG so a later upgrade to exact publish dates is an explicit
task.

DATA SOURCE
───────────
CFTC's public Socrata API (no API key required, courtesy rate limits apply):
the Disaggregated Futures-Only report. Network is best-effort — any failure
returns an empty observation frame rather than raising, matching the other
adapters, so a scheduled ingest never crashes the box.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable, List, Optional

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS, FundamentalAdapter

_RELEASE_LAG_DAYS = 3  # Tuesday reference -> Friday publish


class CftcAdapter(FundamentalAdapter):
    source_name = "cftc"

    # Disaggregated Futures-Only Reports (Socrata). Overridable for tests / future
    # dataset moves without touching call sites.
    base_url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    code_field = "cftc_contract_market_code"
    date_field = "report_date_as_yyyy_mm_dd"
    long_field = "m_money_positions_long_all"
    short_field = "m_money_positions_short_all"

    def __init__(self, timeout: int = 20, page_limit: int = 5000):
        self.timeout = int(timeout)
        self.page_limit = int(page_limit)

    # ── pure transform (the testable core; no network) ───────────────────────
    def _shape(self, records: List[dict]) -> pd.DataFrame:
        rows = []
        for rec in records:
            code = rec.get(self.code_field)
            raw_date = rec.get(self.date_field)
            if code is None or raw_date is None:
                continue
            try:
                ref = pd.Timestamp(raw_date)
                long_ = float(rec.get(self.long_field) or 0.0)
                short_ = float(rec.get(self.short_field) or 0.0)
            except (ValueError, TypeError):
                continue
            release = ref + timedelta(days=_RELEASE_LAG_DAYS)
            rows.append(
                {
                    "source": self.source_name,
                    "series_id": str(code),
                    "reference_date": ref.date(),
                    "release_date": release.date(),
                    "value": long_ - short_,
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

        codes = [str(s) for s in series_ids]
        if not codes:
            return self.empty_observations()

        frames: List[pd.DataFrame] = []
        for code in codes:
            params = {
                "$limit": self.page_limit,
                "$order": f"{self.date_field} ASC",
                self.code_field: code,
            }
            if start:
                params["$where"] = f"{self.date_field} >= '{start}'"
            try:
                resp = requests.get(self.base_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                shaped = self._shape(resp.json())
            except Exception:
                continue
            if not shaped.empty:
                frames.append(shaped)

        if not frames:
            return self.empty_observations()
        return pd.concat(frames, ignore_index=True)
