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
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


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

    def poll(self) -> List[MacroEvent]:
        start = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
        events: List[MacroEvent] = []

        for series_id, meta in FRED_SERIES.items():
            try:
                resp = requests.get(
                    _FRED_BASE,
                    params={
                        "series_id": series_id,
                        "api_key": self._api_key,
                        "file_type": "json",
                        "observation_start": start,
                        "sort_order": "asc",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
            except Exception as exc:
                logger.warning("FRED %s: fetch failed — %s", series_id, exc)
                continue

            valid = [o for o in obs if o.get("value") not in (".", "")]
            if not valid:
                continue

            latest = valid[-1]
            latest_date = latest["date"]
            try:
                latest_value = float(latest["value"])
            except ValueError:
                continue

            if self._last_seen.get(series_id) == latest_date:
                continue
            self._last_seen[series_id] = latest_date

            hist = self._history[series_id]
            hist.append(latest_value)
            if len(hist) > _ZSCORE_WINDOW + 1:
                hist.pop(0)

            baseline = hist[:-1]
            if len(baseline) >= 3:
                mean = float(np.mean(baseline))
                std = float(np.std(baseline, ddof=1))
                deviation = (latest_value - mean) / (std + 1e-8)
            else:
                deviation = 0.0

            previous = float(valid[-2]["value"]) if len(valid) >= 2 else None

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
_AV_CALL_DELAY = 13.0  # seconds between calls — free tier: 5 req/min


class AlphaVantagePoller:
    """
    Polls Alpha Vantage economic indicator endpoints. AV returns newest-first
    lists; we compare the latest date against last-seen and emit on new data.

    Uses the same rolling z-score deviation as FREDPoller when no forecast
    consensus is available.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._last_seen: Dict[str, str] = {}
        self._history: Dict[str, List[float]] = {fn: [] for fn in AV_INDICATORS}

    def _fetch(self, function: str, interval: str) -> Optional[List[dict]]:
        params: dict = {"function": function, "apikey": self._api_key}
        if interval in ("monthly", "quarterly", "annual", "daily"):
            params["interval"] = interval
        if function == "TREASURY_YIELD":
            params["maturity"] = "10year"
        try:
            resp = requests.get(_AV_BASE, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as exc:
            logger.warning("AV %s: fetch failed — %s", function, exc)
            return None

    def poll(self) -> List[MacroEvent]:
        events: List[MacroEvent] = []

        for function, meta in AV_INDICATORS.items():
            records = self._fetch(function, meta["interval"])
            time.sleep(_AV_CALL_DELAY)  # rate-limit guard before next call

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

            if self._last_seen.get(function) == latest_date:
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

    def __init__(self, api_key: str, horizon: str = "3month"):
        self._api_key = api_key
        self._horizon = horizon
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
            resp.raise_for_status()
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
    # Write high-impact events (|deviation| >= 1σ) to trigger_events DB table
    # so downstream cascade/routing models pick them up automatically.
    "db_write_enabled": True,
    "db_write_min_deviation": 1.0,
    "db_write_min_impact": "high",
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

        self._fred = FREDPoller(fred_api_key) if fred_api_key else None
        self._av = AlphaVantagePoller(av_api_key) if av_api_key else None
        self._calendar = EconomicCalendarPoller(av_api_key) if av_api_key else None

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

    def stop(self) -> None:
        """Signal all pollers to stop and join their threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        logger.info("MacroIngestionService: stopped.")

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

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
        min_dev = self._config.get("db_write_min_deviation", 1.0)
        min_impact = self._config.get("db_write_min_impact", "high")
        impact_rank = {"low": 0, "medium": 1, "high": 2}

        if impact_rank.get(event.impact, 0) < impact_rank.get(min_impact, 2):
            return
        if abs(event.deviation_score) < min_dev:
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
