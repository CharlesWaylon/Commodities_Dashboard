"""
Macro Data Ingestion Service

Polls three sources on configurable intervals and normalizes all releases into
MacroEvent objects, then pushes them to a local MacroQueue.

Sources
-------
  FRED               — Federal Reserve Economic Data (series observations)
  Alpha Vantage      — Economic indicator time series (CPI, NFP, GDP, …)
  Economic Calendar  — Alpha Vantage ECONOMIC_CALENDAR (expected vs actual)

Required env vars
-----------------
  FRED_API_KEY        — fred.stlouisfed.org/docs/api/api_key.html  (free)
  ALPHA_VANTAGE_KEY   — alphavantage.co  (free tier: 25 req/day, 5 req/min)

Optional env vars
-----------------
  MACRO_QUEUE_PATH    — JSONL sink path  (default: logs/macro_events.jsonl)

Quick start
-----------
  from services.macro_ingestion import MacroIngestionService

  svc = MacroIngestionService.from_env()
  svc.start()

  for event in svc.queue.stream():
      print(event)

  svc.stop()
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from pathlib import Path
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import numpy as np
import requests

from services.macro_queue import MacroQueue

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Standard event schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MacroEvent:
    trigger_id: str            # UUID4 — unique per emission
    source: str                # "FRED" | "alpha_vantage" | "economic_calendar"
    event_type: str            # "CPI" | "NFP" | "Fed_Funds_Rate" | …
    expected_value: Optional[float]
    actual_value: Optional[float]
    release_timestamp: str     # ISO 8601 UTC
    deviation_score: float     # σ from rolling baseline; or surprise / baseline_std

    country: str = "US"
    impact: str = "medium"     # "low" | "medium" | "high"
    unit: str = ""
    previous_value: Optional[float] = None
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# FRED Poller
# ──────────────────────────────────────────────────────────────────────────────

FRED_SERIES: Dict[str, dict] = {
    "CPIAUCSL":     {"name": "CPI",                  "impact": "high",   "unit": "index"},
    "UNRATE":       {"name": "Unemployment",          "impact": "high",   "unit": "%"},
    "FEDFUNDS":     {"name": "Fed_Funds_Rate",        "impact": "high",   "unit": "%"},
    "T10Y2Y":       {"name": "Yield_Curve_10Y2Y",     "impact": "medium", "unit": "bp"},
    "DFII10":       {"name": "Real_Yield_10Y",        "impact": "high",   "unit": "%"},
    "INDPRO":       {"name": "Industrial_Production", "impact": "medium", "unit": "index"},
    "PPIACO":       {"name": "PPI_All_Commodities",   "impact": "medium", "unit": "index"},
    "M2SL":         {"name": "M2_Money_Supply",       "impact": "low",    "unit": "billions_USD"},
    "DEXUSEU":      {"name": "USD_EUR",               "impact": "medium", "unit": "USD"},
    "DCOILWTICO":   {"name": "WTI_Spot",              "impact": "high",   "unit": "USD_per_bbl"},
}

_ZSCORE_WINDOW = 20  # rolling observations for baseline mean/std
# How far back to fetch on each poll. ~2.7y so monthly series (CPI, UNRATE,
# INDPRO, …) return ≥ _ZSCORE_WINDOW observations and yield a real z-score on
# the first poll — not 0.0 until an in-memory buffer warms up across restarts.
_FRED_HISTORY_DAYS = 1000
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
# Pacing between back-to-back series requests. FRED's per-IP burst limiter
# 429s on tight bursts even well under the 120 req/min ceiling.
_FRED_INTER_REQUEST_SLEEP = 0.5
# On a 429, sleep this long and retry once before giving up for the cycle.
_FRED_BACKOFF_ON_429 = 5.0


class FREDPoller:
    """
    Fetches FRED series observations via the REST API. On each poll() it
    compares the latest observation date against the last-seen date and
    emits a MacroEvent only for genuinely new releases.

    Deviation score = z-score of the latest value relative to the prior
    _ZSCORE_WINDOW observations.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_seen: Dict[str, str] = {}          # series_id → last obs date
        self._history: Dict[str, List[float]] = {s: [] for s in FRED_SERIES}

    def seed_last_seen(self, days: int) -> int:
        """
        Backfill _last_seen from trigger_events so a restart does not re-emit
        releases the DB already has. For each FRED series, finds the most
        recent trigger_date (family=meta['name']) within the lookback window
        and seeds _last_seen[series_id] to it. Series with no rows in the
        window get seeded to `today - days` so we never replay ancient data.

        Returns the number of series seeded from DB (vs floor).
        """
        from datetime import date as _date_type
        floor_date = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        seeded_from_db = 0

        try:
            from database.db import get_db
            from database.models import TriggerEvent as DBTriggerEvent
            from sqlalchemy import func

            with get_db() as session:
                for series_id, meta in FRED_SERIES.items():
                    family = meta["name"]
                    most_recent = (
                        session.query(func.max(DBTriggerEvent.trigger_date))
                        .filter(DBTriggerEvent.family == family)
                        .filter(DBTriggerEvent.trigger_date >= floor_date)
                        .scalar()
                    )
                    if most_recent:
                        # trigger_date is stored as ISO date string ("YYYY-MM-DD").
                        seed = most_recent if isinstance(most_recent, str) else most_recent.isoformat()
                        self._last_seen[series_id] = seed
                        seeded_from_db += 1
                    else:
                        self._last_seen[series_id] = floor_date
        except Exception as exc:
            logger.warning("FRED seed_last_seen: DB read failed (%s) — falling back to floor", exc)
            for series_id in FRED_SERIES:
                self._last_seen.setdefault(series_id, floor_date)

        logger.info(
            "FRED backfill: seeded %d/%d series from DB; floor=%s (lookback=%dd)",
            seeded_from_db, len(FRED_SERIES), floor_date, days,
        )
        return seeded_from_db

    def poll(self) -> List[MacroEvent]:
        # Fetch a long window so even monthly series (CPI, UNRATE, …) come back
        # with enough observations to build a real z-score baseline on the very
        # first poll — daily series are capped to _ZSCORE_WINDOW below anyway.
        start = (datetime.now(timezone.utc) - timedelta(days=_FRED_HISTORY_DAYS)).strftime("%Y-%m-%d")
        events: List[MacroEvent] = []

        for idx, (series_id, meta) in enumerate(FRED_SERIES.items()):
            if idx > 0:
                time.sleep(_FRED_INTER_REQUEST_SLEEP)
            params = {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start,
                "sort_order": "asc",
            }
            obs = None
            for attempt in range(2):
                try:
                    resp = requests.get(_FRED_BASE, params=params, timeout=15)
                    if resp.status_code == 429 and attempt == 0:
                        logger.info(
                            "FRED %s: 429 — backing off %.1fs before retry",
                            series_id, _FRED_BACKOFF_ON_429,
                        )
                        time.sleep(_FRED_BACKOFF_ON_429)
                        continue
                    resp.raise_for_status()
                    obs = resp.json().get("observations", [])
                    break
                except Exception as exc:
                    logger.warning("FRED %s: fetch failed — %s", series_id, exc)
                    break
            if obs is None:
                continue

            # Parse the full returned series into floats (ascending by date).
            # Computing the baseline from THIS list — not an in-memory buffer —
            # is what makes deviation work on the first poll after any restart.
            valid = [o for o in obs if o.get("value") not in (".", "")]
            numeric: List[float] = []
            valid_obs: List[dict] = []
            for o in valid:
                try:
                    numeric.append(float(o["value"]))
                    valid_obs.append(o)
                except (ValueError, KeyError):
                    continue
            if not numeric:
                continue

            latest_value = numeric[-1]
            latest_date  = valid_obs[-1]["date"]

            # Emit only when the FRED publish-date is strictly newer than the
            # last release we've already counted (seeded from trigger_events on
            # startup so restarts don't replay history).
            prev = self._last_seen.get(series_id)
            if prev is not None and latest_date <= prev:
                continue
            self._last_seen[series_id] = latest_date

            # Keep the rolling buffer updated for any other consumers, but the
            # deviation below is derived from the API window, not this buffer.
            hist = self._history[series_id]
            hist.append(latest_value)
            if len(hist) > _ZSCORE_WINDOW + 1:
                hist.pop(0)

            # Baseline = up to _ZSCORE_WINDOW observations immediately preceding
            # the latest, taken straight from the API response.
            baseline = numeric[-(_ZSCORE_WINDOW + 1):-1]
            if len(baseline) >= 3:
                mean = float(np.mean(baseline))
                std = float(np.std(baseline, ddof=1))
                deviation = (latest_value - mean) / (std + 1e-8)
            else:
                deviation = 0.0

            previous = numeric[-2] if len(numeric) >= 2 else None

            rel_ts = (
                datetime.strptime(latest_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .isoformat()
            )

            events.append(MacroEvent(
                trigger_id=str(uuid.uuid4()),
                source="FRED",
                event_type=meta["name"],
                expected_value=None,
                actual_value=latest_value,
                release_timestamp=rel_ts,
                deviation_score=round(deviation, 4),
                impact=meta["impact"],
                unit=meta["unit"],
                previous_value=previous,
                metadata={"series_id": series_id},
            ))

        return events


# ──────────────────────────────────────────────────────────────────────────────
# Alpha Vantage Economic Indicators Poller
# ──────────────────────────────────────────────────────────────────────────────

# Free-tier AV functions that return time series (newest data point first)
AV_INDICATORS: Dict[str, dict] = {
    "REAL_GDP":           {"name": "Real_GDP",        "interval": "quarterly", "impact": "high",   "unit": "billions_USD"},
    "CPI":                {"name": "CPI_AV",          "interval": "monthly",   "impact": "high",   "unit": "index"},
    "INFLATION":          {"name": "Inflation_YoY",   "interval": "annual",    "impact": "medium", "unit": "%"},
    "RETAIL_SALES":       {"name": "Retail_Sales",    "interval": "monthly",   "impact": "high",   "unit": "millions_USD"},
    "DURABLES":           {"name": "Durable_Goods",   "interval": "monthly",   "impact": "medium", "unit": "millions_USD"},
    "UNEMPLOYMENT":       {"name": "Unemployment_AV", "interval": "monthly",   "impact": "high",   "unit": "%"},
    "NONFARM_PAYROLL":    {"name": "NFP",             "interval": "monthly",   "impact": "high",   "unit": "thousands"},
    "FEDERAL_FUNDS_RATE": {"name": "Fed_Funds_AV",   "interval": "monthly",   "impact": "high",   "unit": "%"},
    "TREASURY_YIELD":     {"name": "Treasury_10Y",    "interval": "monthly",   "impact": "medium", "unit": "%"},
}

_AV_BASE = "https://www.alphavantage.co/query"

# Free tier: 5 req/min, 25 req/day. The token bucket below enforces both
# limits across all AV pollers sharing this limiter instance.
_AV_PER_MINUTE = 5
_AV_PER_DAY    = 25
_AV_BACKOFF_LADDER_SECONDS = (60, 300, 900)  # 1m → 5m → 15m


class AVRateLimiter:
    """
    Thread-safe token bucket shared by every AV-backed poller.

    Enforces both the per-minute (5) and per-day (25) caps via sliding-window
    accounting. On a 429 / "Note: API call frequency" response, callers should
    invoke .register_throttled() to walk up the backoff ladder; .register_ok()
    resets the ladder on the next successful call.
    """

    def __init__(
        self,
        per_minute: int = _AV_PER_MINUTE,
        per_day:    int = _AV_PER_DAY,
        ladder:     tuple = _AV_BACKOFF_LADDER_SECONDS,
    ):
        self._per_minute   = per_minute
        self._per_day      = per_day
        self._ladder       = ladder
        self._lock         = threading.Lock()
        self._cond         = threading.Condition(self._lock)
        self._calls: List[float] = []          # unix timestamps of recent acquires
        self._backoff_step = -1                # -1 = no active backoff
        self._backoff_until_ts = 0.0           # unix timestamp; no calls before this

    def acquire(self, label: str = "AV") -> bool:
        """
        Block until a token is available. Returns True on success, False if
        the daily cap has been exhausted (caller should give up this cycle).
        """
        with self._cond:
            while True:
                now = time.time()
                # Purge events older than 24h so the day window slides.
                cutoff_day = now - 86_400
                self._calls = [t for t in self._calls if t >= cutoff_day]

                if len(self._calls) >= self._per_day:
                    # Hard cap: wait until the oldest call falls off the day window.
                    wait_s = (self._calls[0] + 86_400) - now
                    logger.warning(
                        "%s: daily cap (%d) hit — sleeping %.0fs until window resets",
                        label, self._per_day, max(wait_s, 0),
                    )
                    self._cond.wait(timeout=max(wait_s, 1.0))
                    continue

                if now < self._backoff_until_ts:
                    wait_s = self._backoff_until_ts - now
                    logger.warning(
                        "%s: in backoff — sleeping %.0fs", label, wait_s,
                    )
                    self._cond.wait(timeout=wait_s)
                    continue

                cutoff_minute = now - 60.0
                in_last_minute = sum(1 for t in self._calls if t >= cutoff_minute)
                if in_last_minute >= self._per_minute:
                    oldest_in_min = min(t for t in self._calls if t >= cutoff_minute)
                    wait_s = (oldest_in_min + 60.0) - now
                    self._cond.wait(timeout=max(wait_s, 0.5))
                    continue

                # Token granted.
                self._calls.append(now)
                return True

    def register_ok(self) -> None:
        """Reset the backoff ladder after a clean response."""
        with self._cond:
            self._backoff_step    = -1
            self._backoff_until_ts = 0.0

    def register_throttled(self, label: str = "AV") -> None:
        """Walk up the backoff ladder (1m → 5m → 15m, capped at last rung)."""
        with self._cond:
            self._backoff_step = min(self._backoff_step + 1, len(self._ladder) - 1)
            penalty = self._ladder[self._backoff_step]
            self._backoff_until_ts = time.time() + penalty
            logger.warning(
                "%s: throttled — backing off %ds (step %d/%d)",
                label, penalty, self._backoff_step + 1, len(self._ladder),
            )
            self._cond.notify_all()

    def calls_in_last(self, seconds: float) -> int:
        """Diagnostic helper: how many tokens acquired in the last N seconds."""
        with self._lock:
            cutoff = time.time() - seconds
            return sum(1 for t in self._calls if t >= cutoff)


def _is_throttled_response(resp) -> bool:
    """True if AV signaled rate-limiting via 429 or the textual 'Note' field."""
    if resp.status_code == 429:
        return True
    body = (resp.text or "")[:512].lower()
    # AV returns 200 with a JSON {"Note": "...call frequency..."} on free-tier throttle.
    return ("api call frequency" in body) or ('"note"' in body and "frequency" in body)


class AlphaVantagePoller:
    """
    Polls Alpha Vantage economic indicator endpoints. AV returns newest-first
    lists; we compare the latest date against last-seen and emit on new data.

    Uses the same rolling z-score deviation as FREDPoller when no forecast
    consensus is available.
    """

    def __init__(self, api_key: str, limiter: Optional[AVRateLimiter] = None):
        self._api_key = api_key
        self._limiter = limiter or AVRateLimiter()
        self._last_seen: Dict[str, str] = {}
        self._history: Dict[str, List[float]] = {fn: [] for fn in AV_INDICATORS}

    def seed_last_seen(self, days: int) -> int:
        """
        Same contract as FREDPoller.seed_last_seen — for AV_INDICATORS.
        Keys are AV function names; values are the latest observed date.
        """
        floor_date = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        seeded_from_db = 0
        try:
            from database.db import get_db
            from database.models import TriggerEvent as DBTriggerEvent
            from sqlalchemy import func

            with get_db() as session:
                for function, meta in AV_INDICATORS.items():
                    family = meta["name"]
                    most_recent = (
                        session.query(func.max(DBTriggerEvent.trigger_date))
                        .filter(DBTriggerEvent.family == family)
                        .filter(DBTriggerEvent.trigger_date >= floor_date)
                        .scalar()
                    )
                    if most_recent:
                        seed = most_recent if isinstance(most_recent, str) else most_recent.isoformat()
                        self._last_seen[function] = seed
                        seeded_from_db += 1
                    else:
                        self._last_seen[function] = floor_date
        except Exception as exc:
            logger.warning("AV seed_last_seen: DB read failed (%s) — falling back to floor", exc)
            for function in AV_INDICATORS:
                self._last_seen.setdefault(function, floor_date)

        logger.info(
            "AV backfill: seeded %d/%d functions from DB; floor=%s (lookback=%dd)",
            seeded_from_db, len(AV_INDICATORS), floor_date, days,
        )
        return seeded_from_db

    def _fetch(self, function: str, interval: str) -> Optional[List[dict]]:
        params: dict = {"function": function, "apikey": self._api_key}
        if interval in ("monthly", "quarterly", "annual", "daily"):
            params["interval"] = interval
        if function == "TREASURY_YIELD":
            params["maturity"] = "10year"

        # Token-bucket gate: blocks until a per-minute slot is free; aborts
        # if the daily 25-call cap is already exhausted.
        self._limiter.acquire(label=f"AV/{function}")
        try:
            resp = requests.get(_AV_BASE, params=params, timeout=20)
            if _is_throttled_response(resp):
                self._limiter.register_throttled(label=f"AV/{function}")
                return None
            resp.raise_for_status()
            self._limiter.register_ok()
            return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("AV %s: fetch failed — %s", function, exc)
            return None

    def poll(self) -> List[MacroEvent]:
        events: List[MacroEvent] = []

        for function, meta in AV_INDICATORS.items():
            records = self._fetch(function, meta["interval"])

            if not records:
                continue

            valid = [r for r in records if r.get("value") not in (".", "", None)]
            if not valid:
                continue

            latest = valid[0]
            latest_date = latest.get("date", "")
            try:
                latest_value = float(latest["value"])
            except (ValueError, KeyError):
                continue

            prev = self._last_seen.get(function)
            if prev is not None and latest_date <= prev:
                continue
            self._last_seen[function] = latest_date

            hist = self._history[function]
            hist.append(latest_value)
            if len(hist) > _ZSCORE_WINDOW + 1:
                hist.pop(0)

            baseline = hist[:-1]
            previous = float(valid[1]["value"]) if len(valid) >= 2 else None

            if len(baseline) >= 3:
                mean = float(np.mean(baseline))
                std = float(np.std(baseline, ddof=1))
                deviation = (latest_value - mean) / (std + 1e-8)
            elif previous is not None:
                deviation = (latest_value - previous) / (abs(previous) + 1e-8)
            else:
                deviation = 0.0

            try:
                rel_ts = (
                    datetime.strptime(latest_date, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .isoformat()
                )
            except ValueError:
                rel_ts = datetime.now(timezone.utc).isoformat()

            events.append(MacroEvent(
                trigger_id=str(uuid.uuid4()),
                source="alpha_vantage",
                event_type=meta["name"],
                expected_value=None,
                actual_value=latest_value,
                release_timestamp=rel_ts,
                deviation_score=round(deviation, 4),
                impact=meta["impact"],
                unit=meta["unit"],
                previous_value=previous,
                metadata={"function": function, "interval": meta["interval"]},
            ))

        return events


# ──────────────────────────────────────────────────────────────────────────────
# Economic Calendar Poller (Alpha Vantage ECONOMIC_CALENDAR)
# ──────────────────────────────────────────────────────────────────────────────

_IMPACT_MAP = {
    "high": "high", "medium": "medium", "low": "low",
    "3": "high", "2": "medium", "1": "low",
}


class EconomicCalendarPoller:
    """
    Polls AV ECONOMIC_CALENDAR (CSV format) for events that have been released.
    Emits one MacroEvent per released event not previously seen.

    Deviation score = (actual − expected) / rolling_std(surprises) when
    a consensus estimate is available; else (actual − previous) / |previous|.
    """

    def __init__(
        self,
        api_key: str,
        horizon: str = "3month",
        limiter: Optional[AVRateLimiter] = None,
    ):
        self._api_key = api_key
        self._horizon = horizon
        self._limiter = limiter or AVRateLimiter()
        self._seen: Set[str] = set()
        self._surprise_history: Dict[str, List[float]] = {}

    @staticmethod
    def _dedup_key(row: dict) -> str:
        return f"{row.get('date', '')}|{row.get('event', '')}"

    @staticmethod
    def _parse_float(s: str) -> Optional[float]:
        try:
            return float(s.replace("%", "").replace(",", "").strip())
        except (ValueError, AttributeError):
            return None

    def poll(self) -> List[MacroEvent]:
        self._limiter.acquire(label="AV/ECONOMIC_CALENDAR")
        try:
            resp = requests.get(
                _AV_BASE,
                params={
                    "function": "ECONOMIC_CALENDAR",
                    "horizon": self._horizon,
                    "apikey": self._api_key,
                },
                timeout=20,
            )
            if _is_throttled_response(resp):
                self._limiter.register_throttled(label="AV/ECONOMIC_CALENDAR")
                return []
            resp.raise_for_status()
            self._limiter.register_ok()
        except Exception as exc:
            logger.warning("Economic calendar fetch failed: %s", exc)
            return []

        try:
            reader = csv.DictReader(io.StringIO(resp.text))
            rows = list(reader)
        except Exception as exc:
            logger.warning("Economic calendar CSV parse failed: %s", exc)
            return []

        events: List[MacroEvent] = []

        for row in rows:
            actual_str = (row.get("actual") or "").strip()
            if not actual_str:
                continue  # Not yet released

            dedup = self._dedup_key(row)
            if dedup in self._seen:
                continue
            self._seen.add(dedup)

            actual = self._parse_float(actual_str)
            if actual is None:
                continue

            expected = self._parse_float(row.get("estimate", ""))
            previous = self._parse_float(row.get("previous", ""))
            event_name = (row.get("event") or "UNKNOWN").strip()

            # Deviation score
            hist = self._surprise_history.setdefault(event_name, [])
            if expected is not None:
                surprise = actual - expected
                hist.append(surprise)
                if len(hist) > 12:
                    hist.pop(0)
                std = float(np.std(hist, ddof=1)) if len(hist) >= 3 else abs(expected) + 1e-8
                deviation = surprise / (std + 1e-8)
            elif previous is not None:
                deviation = (actual - previous) / (abs(previous) + 1e-8)
            else:
                deviation = 0.0

            # Parse release timestamp
            date_str = (row.get("date") or "").strip()
            time_str = (row.get("time") or "").strip()
            try:
                if time_str and time_str.lower() != "tentative":
                    rel_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                else:
                    rel_dt = datetime.strptime(date_str, "%Y-%m-%d")
                rel_ts = rel_dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                rel_ts = datetime.now(timezone.utc).isoformat()

            impact = _IMPACT_MAP.get((row.get("impact") or "medium").strip().lower(), "medium")
            country = (row.get("country") or "US").strip()

            events.append(MacroEvent(
                trigger_id=str(uuid.uuid4()),
                source="economic_calendar",
                event_type=event_name.replace(" ", "_"),
                expected_value=expected,
                actual_value=actual,
                release_timestamp=rel_ts,
                deviation_score=round(deviation, 4),
                country=country,
                impact=impact,
                unit=(row.get("unit") or "").strip(),
                previous_value=previous,
                metadata={
                    "currency": (row.get("currency") or "").strip(),
                    "raw_event": event_name,
                },
            ))

        return events


# ──────────────────────────────────────────────────────────────────────────────
# Service orchestrator
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    "fred": {
        "enabled": True,
        "poll_interval_minutes": 60,
    },
    "alpha_vantage_indicators": {
        "enabled": True,
        "poll_interval_minutes": 360,  # AV free tier: 25 calls/day
    },
    "economic_calendar": {
        "enabled": True,
        "poll_interval_minutes": 30,
    },
    # Write events to trigger_events DB table so downstream cascade/routing
    # models pick them up automatically. Defaults are deliberately permissive
    # (medium impact, ≥0.5σ deviation) — earlier high/1.0σ defaults produced
    # zero rows from the 10 FRED series. Override via MACRO_MIN_DEV /
    # MACRO_MIN_IMPACT env vars when running pipeline/run_macro_feed.py.
    "db_write_enabled": True,
    "db_write_min_deviation": float(os.getenv("MACRO_MIN_DEV", "0.5")),
    "db_write_min_impact": os.getenv("MACRO_MIN_IMPACT", "medium"),
}


class MacroIngestionService:
    """
    Background service that polls macro data sources and emits normalized
    MacroEvent objects to a MacroQueue.

    Each source runs in its own daemon thread. The service can be stopped
    cleanly via stop().

    DB integration
    --------------
    When db_write_enabled=True, events meeting the impact/deviation thresholds
    are upserted into the trigger_events table so downstream cascade and routing
    models consume them without additional wiring.
    """

    def __init__(
        self,
        fred_api_key: str,
        av_api_key: str,
        config: Optional[dict] = None,
        out_queue: Optional[MacroQueue] = None,
    ):
        self._config: dict = {**DEFAULT_CONFIG, **(config or {})}
        self.queue: MacroQueue = out_queue or MacroQueue(
            persist_path=os.getenv("MACRO_QUEUE_PATH", "logs/macro_events.jsonl")
        )
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        # Liveness heartbeat. The pollers only write to the events JSONL when a
        # NEW release is emitted, so during quiet periods (nights/weekends,
        # between releases) that file goes stale even though the daemon is
        # perfectly healthy. The heartbeat is touched on a fixed timer
        # regardless of events, giving the dashboard a true "is the daemon
        # alive?" signal distinct from "when did the last event arrive?".
        self._heartbeat_path = Path(
            os.getenv("MACRO_HEARTBEAT_PATH", "logs/macro_feed.heartbeat")
        )
        self._heartbeat_interval = int(os.getenv("MACRO_HEARTBEAT_INTERVAL", "60"))
        self._heartbeat_thread: Optional[threading.Thread] = None

        # One token bucket shared by both AV-backed pollers so the 5/min
        # and 25/day caps are enforced across the whole service, not per-poller.
        self._av_limiter = AVRateLimiter() if av_api_key else None

        self._fred = FREDPoller(fred_api_key) if fred_api_key else None
        self._av = (
            AlphaVantagePoller(av_api_key, limiter=self._av_limiter)
            if av_api_key else None
        )
        self._calendar = (
            EconomicCalendarPoller(av_api_key, limiter=self._av_limiter)
            if av_api_key else None
        )

        if not fred_api_key:
            logger.warning("FRED_API_KEY not set — FRED poller disabled.")
        if not av_api_key:
            logger.warning("ALPHA_VANTAGE_KEY not set — AV pollers disabled.")

    @classmethod
    def from_env(cls, config: Optional[dict] = None) -> "MacroIngestionService":
        return cls(
            fred_api_key=os.getenv("FRED_API_KEY", ""),
            av_api_key=os.getenv("ALPHA_VANTAGE_KEY", ""),
            config=config,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all enabled pollers in daemon threads."""
        self._stop_event.clear()
        cfg = self._config

        specs = [
            (self._fred,     cfg["fred"],                      "FRED"),
            (self._av,       cfg["alpha_vantage_indicators"],  "AV-Indicators"),
            (self._calendar, cfg["economic_calendar"],         "EconCalendar"),
        ]
        for poller, poller_cfg, label in specs:
            if not poller_cfg.get("enabled") or poller is None:
                continue
            interval_s = poller_cfg["poll_interval_minutes"] * 60
            t = threading.Thread(
                target=self._run_poller,
                args=(poller, interval_s, label),
                name=f"MacroPoller-{label}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
            logger.info(
                "MacroIngestionService: %s started (interval=%dm)",
                label, poller_cfg["poll_interval_minutes"],
            )

        # Liveness heartbeat thread — runs as long as the service is up,
        # independent of whether any poller emits events. Tracked separately
        # from _threads so it doesn't count toward is_running() (which gates
        # the "no pollers started" check in the daemon entrypoint).
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat,
            name="MacroHeartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Signal all pollers to stop and join their threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None
        logger.info("MacroIngestionService: stopped.")

    def _run_heartbeat(self) -> None:
        """
        Touch the heartbeat file every _heartbeat_interval seconds while the
        service is running. The dashboard reads this file's mtime to show
        LIVE/OFFLINE — a signal that stays fresh even when no events are being
        emitted (so quiet markets don't trigger a false OFFLINE).
        """
        self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop_event.is_set():
            try:
                self._heartbeat_path.write_text(
                    datetime.now(timezone.utc).isoformat() + "\n"
                )
            except Exception as exc:
                logger.warning("Heartbeat write failed: %s", exc)
            self._stop_event.wait(self._heartbeat_interval)

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def backfill(self, days: int = 7) -> None:
        """
        Seed each poller's _last_seen cache from the trigger_events table so
        the first poll cycle after a restart does not re-emit releases the
        DB already has. Call before start().
        """
        if self._fred is not None:
            self._fred.seed_last_seen(days)
        if self._av is not None:
            self._av.seed_last_seen(days)
        # EconomicCalendarPoller maintains its own dedup set keyed by
        # "date|event_name" — there's no clean reverse mapping from the
        # underscore-normalized family back to that key, so the calendar
        # is deliberately left to self-deduplicate on its first poll.

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_poller(self, poller, interval_seconds: int, label: str) -> None:
        while not self._stop_event.is_set():
            try:
                events = poller.poll()
                for evt in events:
                    self.queue.put(evt)
                    logger.info(
                        "[%s] %s: actual=%.4g dev=%.2fσ impact=%s",
                        evt.source, evt.event_type,
                        evt.actual_value or 0, evt.deviation_score, evt.impact,
                    )
                    if self._config.get("db_write_enabled"):
                        self._maybe_write_db(evt)
            except Exception as exc:
                logger.error("%s poller error: %s", label, exc, exc_info=True)
            self._stop_event.wait(interval_seconds)

    def _maybe_write_db(self, event: MacroEvent) -> None:
        """
        Upsert into trigger_events if the event clears the impact/deviation
        thresholds. Uses the family+trigger_date unique constraint to merge
        same-day re-fires.
        """
        min_dev = self._config.get("db_write_min_deviation", 0.5)
        min_impact = self._config.get("db_write_min_impact", "medium")
        impact_rank = {"low": 0, "medium": 1, "high": 2}

        if impact_rank.get(event.impact, 0) < impact_rank.get(min_impact, 1):
            logger.info(
                "below threshold (dev=%.2f, impact=%s) — %s/%s skipped (min_dev=%.2f, min_impact=%s)",
                event.deviation_score, event.impact,
                event.source, event.event_type, min_dev, min_impact,
            )
            return
        if abs(event.deviation_score) < min_dev:
            logger.info(
                "below threshold (dev=%.2f, impact=%s) — %s/%s skipped (min_dev=%.2f, min_impact=%s)",
                event.deviation_score, event.impact,
                event.source, event.event_type, min_dev, min_impact,
            )
            return

        try:
            from database.db import get_db
            from database.models import TriggerEvent as DBTriggerEvent

            strength = min(abs(event.deviation_score) / 3.0, 1.0)
            trigger_date = event.release_timestamp[:10]
            meta_json = json.dumps({
                "source": event.source,
                "unit": event.unit,
                "actual": event.actual_value,
                "expected": event.expected_value,
                "deviation_score": event.deviation_score,
                "trigger_id": event.trigger_id,
            })

            with get_db() as session:
                existing = (
                    session.query(DBTriggerEvent)
                    .filter_by(family=event.event_type, trigger_date=trigger_date)
                    .first()
                )
                if existing:
                    existing.strength = strength
                    existing.trigger_metadata = meta_json
                    action = "updated"
                else:
                    session.add(DBTriggerEvent(
                        detected_at=datetime.now(timezone.utc).isoformat(),
                        trigger_date=trigger_date,
                        family=event.event_type,
                        strength=strength,
                        rationale=(
                            f"{event.source}: {event.event_type} "
                            f"dev={event.deviation_score:.2f}σ"
                        ),
                        affected_commodities=json.dumps([]),
                        trigger_metadata=meta_json,
                        inserted_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    action = "inserted"
            logger.info(
                "wrote to DB (%s) — %s/%s dev=%+.2fσ impact=%s strength=%.2f date=%s",
                action, event.source, event.event_type,
                event.deviation_score, event.impact, strength, trigger_date,
            )
        except Exception as exc:
            logger.warning("DB write for MacroEvent failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import signal
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-24s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Macro Data Ingestion Service")
    parser.add_argument(
        "--config",
        default=None,
        help='JSON string overriding DEFAULT_CONFIG keys. '
             'Example: \'{"fred": {"poll_interval_minutes": 30}}\'',
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable DB writes (stream to console only)",
    )
    args = parser.parse_args()

    config_override: dict = {}
    if args.config:
        config_override = json.loads(args.config)
    if args.no_db:
        config_override["db_write_enabled"] = False

    service = MacroIngestionService.from_env(config=config_override)
    service.start()

    if not service.is_running():
        print("ERROR: No pollers started. Check FRED_API_KEY and ALPHA_VANTAGE_KEY env vars.")
        sys.exit(1)

    print("MacroIngestionService running. Streaming events (Ctrl+C to stop):\n")

    def _handle_sigint(sig, frame):
        print("\nShutting down…")
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    for event in service.queue.stream():
        print(
            f"[{event.release_timestamp[:16]}] "
            f"{event.source:<20} {event.event_type:<30} "
            f"actual={event.actual_value!r:<12} "
            f"expected={event.expected_value!r:<12} "
            f"dev={event.deviation_score:+.2f}σ  [{event.impact}]"
        )
