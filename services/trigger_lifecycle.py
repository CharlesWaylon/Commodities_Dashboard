"""
Trigger Lifecycle Manager  (Mission 6)
========================================
Tracks active ``TriggerSignal`` objects through their decay windows, enforces
commodity scope (primary-shock elevation vs. secondary-transmission awareness),
and emits ``ResetSignal`` objects when a decay window expires.

Architecture
------------
The trigger bus (Mission 4) broadcasts every signal to all models.  The
lifecycle manager adds a second layer: it tracks *which commodities are
currently elevated* by marking only the **primary-shock** cluster for each
trigger type as "active."  Secondary transmission commodities are informed by
the bus but not elevated by the lifecycle manager.

This lets the dashboard answer: "Is Corn currently elevated?" (No — FOMC fires
everywhere, but Corn's primary elevation requires a WASDE or WEATHER trigger.)

Primary cluster mapping
-----------------------
Derived from the "Primary shock:" lines in each trigger's registry description:

  FOMC_RATE_DECISION         → metals, energy
  CPI_RELEASE                → metals, energy
  NONFARM_PAYROLLS           → metals
  OPEC_PRODUCTION_DECISION   → energy
  EIA_CRUDE_INVENTORY        → energy
  EIA_GAS_STORAGE            → energy
  USDA_WASDE_REPORT          → agriculture
  FED_CHAIR_SPEECH           → metals
  DOLLAR_SHOCK               → metals, energy
  GEOPOLITICAL_SHOCK         → energy, metals
  RECESSION_FLAG             → all five clusters (systemic)
  WEATHER_SHOCK              → agriculture, energy
  ENERGY_TRANSITION_SIGNAL   → metals, energy
  PPI_RELEASE                → energy, metals

Decay
-----
``TriggerSignal.decay_expires_at`` (set by the classifier, M3) is the
authoritative expiry timestamp.  ``tick()`` compares it against the injected
clock so tests can fast-forward time without sleeping.

Public API
----------
  TriggerLifecycleManager(clock=None, tick_interval_seconds=30)
    .activate(signal)           → ActiveTrigger
    .tick()                     → List[ResetSignal]   # call manually or via background thread
    .force_expire(trigger_type) → Optional[ResetSignal]
    .active_triggers()          → List[ActiveTrigger]
    .is_elevated(commodity)     → bool
    .elevated_by(commodity)     → List[ActiveTrigger]
    .remaining_ttl(trigger_type)→ Optional[timedelta]
    .state_snapshot()           → dict   # dashboard-ready
    .register_reset_handler(fn) → None
    .start() / .stop()          # background ticker thread

  LIFECYCLE                   — module-level singleton (clock = utcnow)
  activate_signal(signal)     — convenience wrapper
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from models.config import COMMODITY_SECTORS
from services.trigger_classifier import TriggerSignal

log = logging.getLogger(__name__)

# ── Primary-cluster scope map ──────────────────────────────────────────────────
# Controls which clusters are *elevated* (not just informed) by each trigger.
# All five clusters still receive bus notifications; only primary clusters
# are marked active in the lifecycle manager's scope enforcement.

_PRIMARY_CLUSTERS: Dict[str, FrozenSet[str]] = {
    "FOMC_RATE_DECISION":         frozenset({"metals", "energy"}),
    "CPI_RELEASE":                frozenset({"metals", "energy"}),
    "NONFARM_PAYROLLS":           frozenset({"metals"}),
    "OPEC_PRODUCTION_DECISION":   frozenset({"energy"}),
    "EIA_CRUDE_INVENTORY":        frozenset({"energy"}),
    "EIA_GAS_STORAGE":            frozenset({"energy"}),
    "USDA_WASDE_REPORT":          frozenset({"agriculture"}),
    "FED_CHAIR_SPEECH":           frozenset({"metals"}),
    "DOLLAR_SHOCK":               frozenset({"metals", "energy"}),
    "GEOPOLITICAL_SHOCK":         frozenset({"energy", "metals"}),
    "RECESSION_FLAG":             frozenset({"energy", "metals", "agriculture", "livestock", "digital"}),
    "WEATHER_SHOCK":              frozenset({"agriculture", "energy"}),
    "ENERGY_TRANSITION_SIGNAL":   frozenset({"metals", "energy"}),
    "PPI_RELEASE":                frozenset({"energy", "metals"}),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(s: str) -> datetime:
    """Parse an ISO 8601 UTC string (handles both 'Z' and '+00:00' suffixes)."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _resolve_primary_commodities(
    trigger_type: str,
    all_commodities: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Return the subset of commodities that fall in the primary-shock clusters."""
    primary_clusters = _PRIMARY_CLUSTERS.get(trigger_type)
    if not primary_clusters:
        # Unknown trigger type: treat all affected as primary
        return all_commodities
    all_set = set(all_commodities)
    return tuple(
        name
        for name, sector in COMMODITY_SECTORS.items()
        if sector in primary_clusters and name in all_set
    )


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ActiveTrigger:
    """One live trigger currently tracked by the lifecycle manager."""
    signal:               TriggerSignal
    activated_at:         datetime                  # when activate() was called
    expires_at:           datetime                  # parsed from signal.decay_expires_at
    primary_clusters:     FrozenSet[str]            # sectors being elevated
    primary_commodities:  Tuple[str, ...]           # commodities being elevated
    all_commodities:      Tuple[str, ...]           # full notification scope (from signal)

    def remaining_ttl(self, now: Optional[datetime] = None) -> timedelta:
        if now is None:
            now = _utcnow()
        return self.expires_at - now

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        return self.remaining_ttl(now).total_seconds() <= 0


@dataclass
class ResetSignal:
    """
    Emitted by the lifecycle manager when a trigger's decay window expires.

    Consumers (cascade handlers, dashboard) should treat this as the end of the
    elevated state for the listed primary commodities.  The ``all_commodities``
    field covers the full notification scope so models can also roll back any
    secondary-transmission adjustments they applied.
    """
    signal_id:           str                        # UUID4 — unique per ResetSignal
    origin_signal_id:    str                        # TriggerSignal.signal_id that expired
    trigger_type:        str
    family_name:         str
    expired_at:          str                        # ISO 8601 UTC
    primary_commodities: Tuple[str, ...]
    all_commodities:     Tuple[str, ...]
    model_targets:       Tuple[str, ...]
    cascade_priority:    int
    metadata:            Dict = field(default_factory=dict)


# ── Lifecycle Manager ──────────────────────────────────────────────────────────

ResetHandler = Callable[[ResetSignal], None]


class TriggerLifecycleManager:
    """
    Tracks active triggers, enforces commodity scope, and emits reset signals.

    Thread-safe: all mutations of ``_active`` are protected by ``_lock``.
    Reset handlers are called outside the lock so they can safely call back
    into the manager.

    The ``clock`` parameter accepts any ``() -> datetime`` callable.
    Inject a fake clock in tests to fast-forward time without sleeping.
    """

    def __init__(
        self,
        clock: Optional[Callable[[], datetime]] = None,
        tick_interval_seconds: float = 30.0,
    ) -> None:
        self._clock = clock or _utcnow
        self._tick_interval = tick_interval_seconds

        # trigger_type → ActiveTrigger (at most one active instance per type)
        self._active: Dict[str, ActiveTrigger] = {}
        # Rolling log of expired triggers (most-recent first)
        self._expired_log: deque[ActiveTrigger] = deque(maxlen=500)
        # Registered reset callbacks
        self._reset_handlers: List[ResetHandler] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None

    # ── Activation ────────────────────────────────────────────────────────────

    def activate(self, signal: TriggerSignal) -> ActiveTrigger:
        """
        Register a live TriggerSignal.

        If the same trigger_type is already active, the new signal replaces it
        (the freshest market event is authoritative).
        """
        primary_clusters = _PRIMARY_CLUSTERS.get(
            signal.trigger_type, frozenset(signal.affected_clusters)
        )
        primary_commodities = _resolve_primary_commodities(
            signal.trigger_type, signal.affected_commodities
        )

        entry = ActiveTrigger(
            signal=signal,
            activated_at=self._clock(),
            expires_at=_parse_dt(signal.decay_expires_at),
            primary_clusters=primary_clusters,
            primary_commodities=primary_commodities,
            all_commodities=signal.affected_commodities,
        )

        with self._lock:
            self._active[signal.trigger_type] = entry

        log.debug(
            "Trigger activated: %s | expires=%s | primary_clusters=%s | "
            "primary_commodities=%d",
            signal.trigger_type,
            entry.expires_at.isoformat(),
            sorted(primary_clusters),
            len(primary_commodities),
        )
        return entry

    # ── Decay tick ────────────────────────────────────────────────────────────

    def tick(self) -> List[ResetSignal]:
        """
        Check all active triggers for expiry.

        Returns a list of ResetSignal objects for every trigger that expired
        since the last call.  Expired triggers are removed from the active set
        and appended to the expired log.  Reset handlers are called after the
        lock is released.
        """
        now = self._clock()
        expired_entries: List[ActiveTrigger] = []

        with self._lock:
            expired_types = [
                ttype
                for ttype, entry in self._active.items()
                if entry.is_expired(now)
            ]
            for ttype in expired_types:
                entry = self._active.pop(ttype)
                self._expired_log.appendleft(entry)
                expired_entries.append(entry)

        resets: List[ResetSignal] = [self._make_reset(e, now) for e in expired_entries]

        for reset in resets:
            log.info("Trigger expired: %s → reset emitted", reset.trigger_type)
            self._dispatch_reset(reset)

        return resets

    # ── Force expiry ──────────────────────────────────────────────────────────

    def force_expire(self, trigger_type: str) -> Optional[ResetSignal]:
        """
        Immediately expire an active trigger regardless of its decay window.

        Useful for dashboard "dismiss" buttons or manual overrides.
        Returns None if the trigger_type is not currently active.
        """
        now = self._clock()
        with self._lock:
            entry = self._active.pop(trigger_type, None)
            if entry is None:
                return None
            self._expired_log.appendleft(entry)

        reset = self._make_reset(entry, now)
        log.info("Trigger force-expired: %s", trigger_type)
        self._dispatch_reset(reset)
        return reset

    # ── Queries ───────────────────────────────────────────────────────────────

    def active_triggers(self) -> List[ActiveTrigger]:
        """Snapshot of all currently active (non-expired) triggers."""
        with self._lock:
            return list(self._active.values())

    def expired_log(self) -> List[ActiveTrigger]:
        """All triggers that have expired (most-recent first, capped at 500)."""
        with self._lock:
            return list(self._expired_log)

    def is_elevated(self, commodity: str) -> bool:
        """
        True if ``commodity`` is in the primary-shock scope of any active trigger.

        Commodities in secondary transmission scope are NOT elevated — they
        receive bus notifications but are not marked active here.
        """
        with self._lock:
            return any(
                commodity in entry.primary_commodities
                for entry in self._active.values()
            )

    def elevated_by(self, commodity: str) -> List[ActiveTrigger]:
        """Return all active triggers that are elevating ``commodity``."""
        with self._lock:
            return [
                entry
                for entry in self._active.values()
                if commodity in entry.primary_commodities
            ]

    def remaining_ttl(self, trigger_type: str) -> Optional[timedelta]:
        """
        Remaining time until the named trigger expires.

        Returns None if the trigger is not active.  The returned timedelta may
        be negative if the trigger has logically expired but ``tick()`` has not
        been called yet.
        """
        with self._lock:
            entry = self._active.get(trigger_type)
        if entry is None:
            return None
        return entry.remaining_ttl(self._clock())

    def state_snapshot(self) -> dict:
        """
        Dashboard-ready snapshot of the full lifecycle state.

        Structure::

            {
              "snapshot_at": "2026-05-15T14:05:00Z",
              "active_count": 2,
              "elevated_commodities": ["WTI Crude Oil", "Gold (COMEX)", ...],
              "triggers": [
                {
                  "trigger_type": "GEOPOLITICAL_SHOCK",
                  "display_name": "Geopolitical Supply Shock",
                  "severity": "HIGH",
                  "magnitude_score": 0.73,
                  "direction": "downside_surprise",
                  "primary_clusters": ["energy", "metals"],
                  "primary_commodity_count": 23,
                  "all_commodity_count": 48,
                  "activated_at": "...",
                  "expires_at": "...",
                  "remaining_ttl_seconds": 86400.0,
                  "cascade_priority": 1,
                  "is_repeat": false,
                },
                ...
              ],
            }
        """
        now = self._clock()
        with self._lock:
            entries = list(self._active.values())

        elevated: set[str] = set()
        trigger_rows = []
        for e in entries:
            elevated.update(e.primary_commodities)
            ttl = e.remaining_ttl(now)
            trigger_rows.append({
                "trigger_type":          e.signal.trigger_type,
                "display_name":          e.signal.config.display_name,
                "severity":              e.signal.severity.label(),
                "magnitude_score":       round(e.signal.magnitude_score, 4),
                "direction":             e.signal.direction,
                "primary_clusters":      sorted(e.primary_clusters),
                "primary_commodity_count": len(e.primary_commodities),
                "all_commodity_count":   len(e.all_commodities),
                "activated_at":          e.activated_at.isoformat(),
                "expires_at":            e.expires_at.isoformat(),
                "remaining_ttl_seconds": round(ttl.total_seconds(), 1),
                "cascade_priority":      e.signal.cascade_priority,
                "is_repeat":             e.signal.is_repeat,
            })

        # Sort by cascade_priority then remaining TTL (most urgent first)
        trigger_rows.sort(key=lambda r: (r["cascade_priority"], r["remaining_ttl_seconds"]))

        return {
            "snapshot_at":          now.isoformat(),
            "active_count":         len(entries),
            "elevated_commodities": sorted(elevated),
            "triggers":             trigger_rows,
        }

    # ── Reset handler registration ─────────────────────────────────────────────

    def register_reset_handler(self, handler: ResetHandler) -> None:
        """Register a callback to be called whenever a trigger's decay expires."""
        self._reset_handlers.append(handler)

    # ── Background thread ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background decay-check thread."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()
        self._bg_thread = threading.Thread(
            target=self._run, daemon=True, name="TriggerLifecycle"
        )
        self._bg_thread.start()
        log.info("TriggerLifecycleManager background thread started (interval=%ss)",
                 self._tick_interval)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background thread gracefully."""
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=timeout)
            self._bg_thread = None
        log.info("TriggerLifecycleManager background thread stopped")

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._tick_interval):
            try:
                self.tick()
            except Exception:
                log.exception("Unhandled error in lifecycle tick")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_reset(self, entry: ActiveTrigger, now: datetime) -> ResetSignal:
        return ResetSignal(
            signal_id=str(uuid.uuid4()),
            origin_signal_id=entry.signal.signal_id,
            trigger_type=entry.signal.trigger_type,
            family_name=entry.signal.family_name,
            expired_at=now.isoformat(),
            primary_commodities=entry.primary_commodities,
            all_commodities=entry.all_commodities,
            model_targets=entry.signal.model_targets,
            cascade_priority=entry.signal.cascade_priority,
            metadata={
                "original_magnitude":  entry.signal.magnitude_score,
                "original_severity":   entry.signal.severity.label(),
                "original_direction":  entry.signal.direction,
                "decay_window_minutes": entry.signal.config.decay_window_minutes,
            },
        )

    def _dispatch_reset(self, reset: ResetSignal) -> None:
        for handler in self._reset_handlers:
            try:
                handler(reset)
            except Exception:
                log.exception(
                    "Reset handler raised for trigger %s", reset.trigger_type
                )


# ── Module-level singleton ─────────────────────────────────────────────────────

LIFECYCLE = TriggerLifecycleManager()


def activate_signal(signal: TriggerSignal) -> ActiveTrigger:
    """Convenience wrapper — activates via the module-level LIFECYCLE manager."""
    return LIFECYCLE.activate(signal)
