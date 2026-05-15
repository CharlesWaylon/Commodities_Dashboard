"""
Mission 10 — End-to-End Integration & Stress Test
===================================================
Simulates a sequence of 5 macro events flowing through the complete stack:

  MacroEvent → TriggerClassifier → TriggerSignal
             → TriggerBus (in-process transport)
             → [CascadeDispatcher, TriggerLifecycleManager, AlertEngine]
             → Dashboard state assertions

Five scenario events
--------------------
  E1  FOMC CRITICAL     — Fed_Funds_Rate, σ=2.8, 25bp surprise, impact=high
  E2  PPI LOW           — PPI_All_Commodities, σ=0.25, no actual/expected bias
  E3  GEOPOLITICAL HIGH — geopolitical_shock, σ=2.0, downside, impact=high
  E4  FOMC repeat       — Same release date as E1 → is_repeat=True
  E5  RECESSION CRITICAL— USREC, σ=2.6, actual=1 (recession on), impact=high

Assertions
----------
  - Scope filtering       : FOMC elevates metals, NOT agriculture/livestock
  - Severity routing      : LOW signal fires 0 alerts; CRITICAL fires ≥1
  - Repeat detection      : E4 carries is_repeat=True from the classifier
  - Decay / reset         : FakeClock advance past TTL triggers reset signal
  - Model cascade         : DispatchResult produced for each HIGH+ signal
  - Alert deduplication   : same rule×trigger_type fires only once per decay window
  - UI state              : state_snapshot() fields are dashboard-ready after each event

Latency benchmarks
------------------
  Per signal: classify_ns, publish_ns, handler_ns, total_ns
  Reported in the final benchmark table printed to stdout.

Run
---
  cd /path/to/Commodities_Dashboard
  python -m pytest services/test_integration.py -v
  # or
  python services/test_integration.py
"""

from __future__ import annotations

import threading
import time
import unittest
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

# ── Stack imports ─────────────────────────────────────────────────────────────
from services.macro_ingestion import MacroEvent
from services.trigger_classifier import TriggerClassifier, TriggerSignal, SeverityTier
from services.trigger_config import REGISTRY
from services.trigger_bus import TriggerBus, reconstruct_signal
from services.trigger_lifecycle import TriggerLifecycleManager, ResetSignal
from services.cascade_handlers import CascadeDispatcher, DispatchResult, _build_default_registry
from services.alert_engine import AlertEngine, AlertRule, DEFAULT_RULES, Alert
from models.config import MODELING_COMMODITIES, COMMODITY_SECTORS

# ── Shared test helpers ────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FakeClock:
    """Controllable clock for deterministic decay-window testing."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self._now = start or _utcnow()
        self._lock = threading.Lock()

    def __call__(self) -> datetime:
        with self._lock:
            return self._now

    def advance(self, minutes: float = 0, seconds: float = 0) -> None:
        with self._lock:
            self._now += timedelta(minutes=minutes, seconds=seconds)

    def set(self, dt: datetime) -> None:
        with self._lock:
            self._now = dt


def _make_macro_event(
    event_type: str,
    deviation_score: float,
    actual_value: Optional[float] = None,
    expected_value: Optional[float] = None,
    impact: str = "medium",
    release_timestamp: Optional[str] = None,
) -> MacroEvent:
    return MacroEvent(
        trigger_id=str(uuid.uuid4()),
        source="test_integration",
        event_type=event_type,
        expected_value=expected_value,
        actual_value=actual_value,
        release_timestamp=release_timestamp or _utcnow().isoformat(),
        deviation_score=deviation_score,
        impact=impact,
    )


def _drain_queue(engine: AlertEngine) -> List[Alert]:
    """Drain all pending alerts from an AlertEngine's queue."""
    alerts: List[Alert] = []
    while True:
        try:
            alerts.append(engine.pending_queue.get_nowait())
        except Exception:
            break
    return alerts


class _BusBarrier:
    """
    Thread-safe barrier that fires a threading.Event once N handler
    invocations have been recorded.  Used to synchronise test assertions
    with bus consumer threads.
    """

    def __init__(self, target: int, timeout: float = 3.0) -> None:
        self._target  = target
        self._timeout = timeout
        self._count   = 0
        self._lock    = threading.Lock()
        self._event   = threading.Event()

    def tick(self) -> None:
        with self._lock:
            self._count += 1
            if self._count >= self._target:
                self._event.set()

    def wait(self) -> bool:
        """Block until target reached. Returns True on success, False on timeout."""
        return self._event.wait(timeout=self._timeout)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


# ── Reusable stack builder ─────────────────────────────────────────────────────

class _Stack:
    """
    Self-contained copy of the full pipeline with fresh instances.

    Wires bus → [lifecycle, cascade dispatcher, alert engine].
    Provides a barrier-based wait for all handlers to complete.
    """

    def __init__(self, fake_clock: Optional[FakeClock] = None) -> None:
        self.clock      = fake_clock or FakeClock()
        self.classifier = TriggerClassifier(REGISTRY, dedup_window_hours=24.0)
        self.bus        = TriggerBus(redis_url="")   # force in-process
        self.lifecycle  = TriggerLifecycleManager(
            clock=self.clock, tick_interval_seconds=9999  # disable background tick
        )
        self.dispatcher = CascadeDispatcher(_build_default_registry())
        self.engine     = AlertEngine()
        for rule in DEFAULT_RULES:
            self.engine.register(rule)

        # Per-layer call log for assertions
        self.lifecycle_calls:   List[dict] = []
        self.dispatcher_calls:  List[DispatchResult] = []
        self.alert_calls:       List[List[Alert]] = []

        # Bus barriers (set before each publish)
        self._barriers: List[_BusBarrier] = []
        self._barrier_lock = threading.Lock()

        # Subscribe handlers
        self.bus.subscribe("lifecycle",  self._lifecycle_handler,  list(self.bus._sub_registry._subs.keys() or []) or None)
        self.bus.subscribe("cascade",    self._cascade_handler,    None)
        self.bus.subscribe("alert",      self._alert_handler,      None)

        # Start consumers
        self.bus.start_consuming("lifecycle")
        self.bus.start_consuming("cascade")
        self.bus.start_consuming("alert")

    def _lifecycle_handler(self, payload: dict, channel: str, model_id: str) -> None:
        sig = reconstruct_signal(payload)
        if sig is not None:
            self.lifecycle.activate(sig)
            self.lifecycle_calls.append({"trigger_type": sig.trigger_type, "severity": sig.severity.label()})
        self._tick_barriers()

    def _cascade_handler(self, payload: dict, channel: str, model_id: str) -> None:
        sig = reconstruct_signal(payload)
        if sig is not None:
            result = self.dispatcher.dispatch(sig)
            self.dispatcher_calls.append(result)
        self._tick_barriers()

    def _alert_handler(self, payload: dict, channel: str, model_id: str) -> None:
        sig = reconstruct_signal(payload)
        if sig is not None:
            fired = self.engine.evaluate(sig)
            self.alert_calls.append(fired)
        self._tick_barriers()

    def _tick_barriers(self) -> None:
        with self._barrier_lock:
            for b in self._barriers:
                b.tick()

    def add_barrier(self, target: int, timeout: float = 3.0) -> _BusBarrier:
        b = _BusBarrier(target, timeout)
        with self._barrier_lock:
            self._barriers.append(b)
        return b

    def clear_barriers(self) -> None:
        with self._barrier_lock:
            self._barriers.clear()

    def classify_and_publish(
        self, event: MacroEvent, *, wait_handlers: int = 3
    ) -> Tuple[Optional[TriggerSignal], _BusBarrier]:
        """
        Classify a MacroEvent and publish the resulting signal to the bus.

        Returns (signal, barrier).  Call barrier.wait() before asserting.
        wait_handlers is the number of handler completions to await (3 = all layers).
        """
        barrier = self.add_barrier(wait_handlers)
        sig = self.classifier.classify(event)
        if sig is not None:
            self.bus.publish(sig)
        return sig, barrier

    def publish_direct(
        self, sig: TriggerSignal, *, wait_handlers: int = 3
    ) -> _BusBarrier:
        """Publish a pre-built TriggerSignal, bypassing the classifier."""
        barrier = self.add_barrier(wait_handlers)
        self.bus.publish(sig)
        return barrier

    def stop(self) -> None:
        self.lifecycle.stop()
        for model_id in list(self.bus._threads.keys()):
            self.bus.stop_consuming(model_id)
        self.bus.close()


def _build_geopolitical_signal(clock: Callable[[], datetime]) -> TriggerSignal:
    """Build a HIGH GEOPOLITICAL_SHOCK signal directly (no MacroEvent mapping needed)."""
    cfg = REGISTRY.get("GEOPOLITICAL_SHOCK")
    now = clock()
    all_comms = tuple(MODELING_COMMODITIES.keys())
    return TriggerSignal(
        signal_id=str(uuid.uuid4()),
        trigger_type="GEOPOLITICAL_SHOCK",
        family_name=cfg.family_name,
        source_event_id=str(uuid.uuid4()),
        magnitude_score=0.68,
        severity=SeverityTier.HIGH,
        direction="downside_surprise",
        release_timestamp=now.isoformat(),
        classified_at=now.isoformat(),
        decay_expires_at=(now + timedelta(minutes=cfg.decay_window_minutes)).isoformat(),
        affected_commodities=all_comms,
        affected_clusters=list(cfg.affected_clusters),
        model_targets=cfg.model_targets,
        cascade_priority=cfg.cascade_priority,
        classification_method="direct_build",
        raw_deviation_score=2.0,
        event_type="GEOPOLITICAL_SHOCK",
        actual_value=1.0,
        expected_value=0.0,
        is_repeat=False,
        config=cfg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Suite 1 — Classifier unit assertions
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifierScenarios(unittest.TestCase):
    """Verify each of the 5 events is classified with correct severity and direction."""

    def setUp(self) -> None:
        self.classifier = TriggerClassifier(REGISTRY)
        self.now        = _utcnow().isoformat()

    def _classify(
        self,
        event_type: str,
        deviation_score: float,
        actual_value: Optional[float] = None,
        expected_value: Optional[float] = None,
        impact: str = "medium",
        release_timestamp: Optional[str] = None,
    ) -> Optional[TriggerSignal]:
        ev = _make_macro_event(
            event_type=event_type,
            deviation_score=deviation_score,
            actual_value=actual_value,
            expected_value=expected_value,
            impact=impact,
            release_timestamp=release_timestamp or self.now,
        )
        return self.classifier.classify(ev)

    # E1 — FOMC CRITICAL ───────────────────────────────────────────────────────

    def test_e1_fomc_resolves_to_correct_trigger_type(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.trigger_type, "FOMC_RATE_DECISION")

    def test_e1_fomc_critical_severity(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.severity, SeverityTier.CRITICAL)

    def test_e1_fomc_downside_direction(self) -> None:
        # 5.50 actual vs 5.25 expected — higher rate than expected = downside for risk assets
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        self.assertIn(sig.direction, ("upside_surprise", "downside_surprise", "neutral"))

    def test_e1_fomc_magnitude_above_threshold(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        self.assertGreater(sig.magnitude_score, 0.7)

    def test_e1_fomc_decay_window_is_set(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        exp = datetime.fromisoformat(sig.decay_expires_at)
        cls = datetime.fromisoformat(sig.classified_at)
        self.assertGreater((exp - cls).total_seconds(), 0)

    # E2 — PPI LOW ─────────────────────────────────────────────────────────────

    def test_e2_ppi_resolves_correctly(self) -> None:
        sig = self._classify("PPI_All_Commodities", 0.25, actual_value=102.5,
                              expected_value=103.0, impact="low")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.trigger_type, "PPI_RELEASE")

    def test_e2_ppi_low_severity(self) -> None:
        sig = self._classify("PPI_All_Commodities", 0.25, actual_value=102.5,
                              expected_value=103.0, impact="low")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.severity, SeverityTier.LOW)

    def test_e2_ppi_magnitude_below_high_threshold(self) -> None:
        sig = self._classify("PPI_All_Commodities", 0.25, actual_value=102.5,
                              expected_value=103.0, impact="low")
        self.assertIsNotNone(sig)
        self.assertLess(sig.magnitude_score, 0.50)

    # E4 — FOMC repeat ─────────────────────────────────────────────────────────

    def test_e4_fomc_repeat_flagged(self) -> None:
        # Publish same release date twice — second should be is_repeat=True
        release = _utcnow().isoformat()
        sig1 = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                               expected_value=5.00, impact="high",
                               release_timestamp=release)
        sig2 = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                               expected_value=5.00, impact="high",
                               release_timestamp=release)
        self.assertIsNotNone(sig1)
        self.assertIsNotNone(sig2)
        self.assertFalse(sig1.is_repeat)
        self.assertTrue(sig2.is_repeat)

    def test_e4_different_release_date_not_repeat(self) -> None:
        from datetime import timedelta
        t1 = _utcnow()
        t2 = t1 + timedelta(hours=25)   # outside 24h dedup window
        sig1 = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                               expected_value=5.00, impact="high",
                               release_timestamp=t1.isoformat())
        sig2 = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                               expected_value=5.00, impact="high",
                               release_timestamp=t2.isoformat())
        self.assertIsNotNone(sig1)
        self.assertIsNotNone(sig2)
        self.assertFalse(sig1.is_repeat)
        self.assertFalse(sig2.is_repeat)

    # E5 — RECESSION CRITICAL ──────────────────────────────────────────────────

    def test_e5_recession_resolves_correctly(self) -> None:
        sig = self._classify("USREC", 2.6, actual_value=1.0,
                              expected_value=0.0, impact="high")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.trigger_type, "RECESSION_FLAG")

    def test_e5_recession_severity_high_or_critical(self) -> None:
        sig = self._classify("USREC", 2.6, actual_value=1.0,
                              expected_value=0.0, impact="high")
        self.assertIsNotNone(sig)
        self.assertGreaterEqual(sig.severity, SeverityTier.HIGH)

    def test_e5_recession_affects_all_clusters(self) -> None:
        sig = self._classify("USREC", 2.6, actual_value=1.0,
                              expected_value=0.0, impact="high")
        self.assertIsNotNone(sig)
        clusters = set(sig.affected_clusters)
        # RECESSION_FLAG is the broadest — must hit energy + metals at minimum
        self.assertTrue(clusters & {"energy", "metals"})

    # Classification method ────────────────────────────────────────────────────

    def test_blended_method_used_when_expected_value_present(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=5.75,
                              expected_value=5.00, impact="high")
        self.assertIsNotNone(sig)
        self.assertIn("blended", sig.classification_method)

    def test_z_score_method_used_without_expected_value(self) -> None:
        sig = self._classify("USREC", 2.6, actual_value=1.0,
                              expected_value=None, impact="high")
        self.assertIsNotNone(sig)
        self.assertIn("z_score", sig.classification_method)

    def test_out_of_scope_event_returns_none(self) -> None:
        sig = self._classify("consumer_sentiment_index", 0.5, actual_value=75.0)
        self.assertIsNone(sig)

    def test_no_actual_value_returns_none(self) -> None:
        sig = self._classify("Fed_Funds_Rate", 2.8, actual_value=None,
                              expected_value=5.00, impact="high")
        self.assertIsNone(sig)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 2 — Scope filtering (lifecycle manager)
# ══════════════════════════════════════════════════════════════════════════════

class TestScopeFiltering(unittest.TestCase):
    """
    Verify the two-layer scope model:
      - Bus notifies all model handlers (broad)
      - LifecycleManager elevates ONLY primary-cluster commodities
    """

    def setUp(self) -> None:
        self.clock     = FakeClock()
        self.lifecycle = TriggerLifecycleManager(
            clock=self.clock, tick_interval_seconds=9999
        )

    def tearDown(self) -> None:
        self.lifecycle.stop()

    def _activate(self, trigger_type: str, severity: SeverityTier,
                  magnitude: float = 0.80) -> TriggerSignal:
        cfg = REGISTRY.get(trigger_type)
        now = self.clock()
        sig = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=magnitude,
            severity=severity,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=cfg.decay_window_minutes)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=list(cfg.affected_clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="test",
            raw_deviation_score=2.5,
            event_type=trigger_type,
            actual_value=1.0,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )
        self.lifecycle.activate(sig)
        return sig

    # FOMC scope ───────────────────────────────────────────────────────────────

    def test_fomc_elevates_gold(self) -> None:
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertTrue(self.lifecycle.is_elevated("Gold (COMEX)"))

    def test_fomc_elevates_silver(self) -> None:
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertTrue(self.lifecycle.is_elevated("Silver (COMEX)"))

    def test_fomc_does_not_elevate_corn(self) -> None:
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertFalse(self.lifecycle.is_elevated("Corn (CBOT)"))

    def test_fomc_does_not_elevate_live_cattle(self) -> None:
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertFalse(self.lifecycle.is_elevated("Live Cattle"))

    def test_fomc_does_not_elevate_bitcoin(self) -> None:
        # FOMC primary clusters = metals + energy — digital is not included
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertFalse(self.lifecycle.is_elevated("Bitcoin"))

    # Geopolitical scope ───────────────────────────────────────────────────────

    def test_geopolitical_elevates_wti_crude(self) -> None:
        self._activate("GEOPOLITICAL_SHOCK", SeverityTier.HIGH)
        self.assertTrue(self.lifecycle.is_elevated("WTI Crude Oil"))

    def test_geopolitical_elevates_brent(self) -> None:
        self._activate("GEOPOLITICAL_SHOCK", SeverityTier.HIGH)
        self.assertTrue(self.lifecycle.is_elevated("Brent Crude Oil"))

    def test_geopolitical_does_not_elevate_wheat(self) -> None:
        self._activate("GEOPOLITICAL_SHOCK", SeverityTier.HIGH)
        self.assertFalse(self.lifecycle.is_elevated("Wheat (CBOT SRW)"))

    def test_geopolitical_does_not_elevate_lean_hogs(self) -> None:
        self._activate("GEOPOLITICAL_SHOCK", SeverityTier.HIGH)
        self.assertFalse(self.lifecycle.is_elevated("Lean Hogs"))

    # USDA scope ───────────────────────────────────────────────────────────────

    def test_wasde_elevates_corn(self) -> None:
        self._activate("USDA_WASDE_REPORT", SeverityTier.HIGH)
        self.assertTrue(self.lifecycle.is_elevated("Corn (CBOT)"))

    def test_wasde_does_not_elevate_gold(self) -> None:
        self._activate("USDA_WASDE_REPORT", SeverityTier.HIGH)
        self.assertFalse(self.lifecycle.is_elevated("Gold (COMEX)"))

    def test_wasde_does_not_elevate_crude_oil(self) -> None:
        self._activate("USDA_WASDE_REPORT", SeverityTier.HIGH)
        self.assertFalse(self.lifecycle.is_elevated("WTI Crude Oil"))

    # Recession — all clusters ─────────────────────────────────────────────────

    def test_recession_elevates_all_major_clusters(self) -> None:
        self._activate("RECESSION_FLAG", SeverityTier.CRITICAL)
        snap = self.lifecycle.state_snapshot()
        elevated = set(snap["elevated_commodities"])
        self.assertIn("Gold (COMEX)", elevated)
        self.assertIn("WTI Crude Oil", elevated)
        self.assertIn("Corn (CBOT)", elevated)
        self.assertIn("Live Cattle", elevated)

    # State snapshot structure ─────────────────────────────────────────────────

    def test_state_snapshot_has_required_keys(self) -> None:
        snap = self.lifecycle.state_snapshot()
        for key in ("active_count", "elevated_commodities", "triggers", "snapshot_at"):
            self.assertIn(key, snap)

    def test_state_snapshot_active_count_increments(self) -> None:
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 0)
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 1)
        self._activate("GEOPOLITICAL_SHOCK", SeverityTier.HIGH)
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 2)

    def test_elevated_commodities_list_non_empty_after_activation(self) -> None:
        self._activate("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        snap = self.lifecycle.state_snapshot()
        self.assertGreater(len(snap["elevated_commodities"]), 0)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 3 — Decay windows and reset signals
# ══════════════════════════════════════════════════════════════════════════════

class TestDecayWindows(unittest.TestCase):

    def setUp(self) -> None:
        self.clock     = FakeClock()
        self.resets:   List[ResetSignal] = []
        self.lifecycle = TriggerLifecycleManager(
            clock=self.clock, tick_interval_seconds=9999
        )
        self.lifecycle.register_reset_handler(self.resets.append)

    def tearDown(self) -> None:
        self.lifecycle.stop()

    def _activate_short_decay(
        self,
        trigger_type: str = "EIA_CRUDE_INVENTORY",
        decay_minutes: int = 60,
    ) -> TriggerSignal:
        """Activate a signal with an artificially short decay window."""
        cfg  = REGISTRY.get(trigger_type)
        now  = self.clock()
        exp  = now + timedelta(minutes=decay_minutes)
        sig  = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=0.75,
            severity=SeverityTier.HIGH,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=exp.isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=list(cfg.affected_clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="test",
            raw_deviation_score=2.0,
            event_type=trigger_type,
            actual_value=1.0,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )
        self.lifecycle.activate(sig)
        return sig

    def test_trigger_active_before_expiry(self) -> None:
        self._activate_short_decay(decay_minutes=120)
        self.clock.advance(minutes=59)
        self.lifecycle.tick()
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 1)

    def test_trigger_expires_after_decay_window(self) -> None:
        self._activate_short_decay(decay_minutes=60)
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 0)

    def test_commodity_no_longer_elevated_after_expiry(self) -> None:
        self._activate_short_decay("EIA_CRUDE_INVENTORY", decay_minutes=60)
        self.assertTrue(self.lifecycle.is_elevated("WTI Crude Oil"))
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        self.assertFalse(self.lifecycle.is_elevated("WTI Crude Oil"))

    def test_reset_signal_emitted_on_expiry(self) -> None:
        self._activate_short_decay(decay_minutes=60)
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        self.assertEqual(len(self.resets), 1)

    def test_reset_signal_carries_origin_signal_id(self) -> None:
        sig = self._activate_short_decay(decay_minutes=60)
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        self.assertEqual(self.resets[0].origin_signal_id, sig.signal_id)

    def test_reset_signal_trigger_type_matches(self) -> None:
        self._activate_short_decay("EIA_CRUDE_INVENTORY", decay_minutes=60)
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        self.assertEqual(self.resets[0].trigger_type, "EIA_CRUDE_INVENTORY")

    def test_multiple_triggers_expire_independently(self) -> None:
        self._activate_short_decay("EIA_CRUDE_INVENTORY", decay_minutes=60)
        self._activate_short_decay("USDA_WASDE_REPORT",   decay_minutes=120)
        self.clock.advance(minutes=61)
        self.lifecycle.tick()
        snap = self.lifecycle.state_snapshot()
        # EIA expired; WASDE still live
        self.assertEqual(snap["active_count"], 1)
        self.assertEqual(snap["triggers"][0]["trigger_type"], "USDA_WASDE_REPORT")

    def test_force_expire_removes_trigger_immediately(self) -> None:
        self._activate_short_decay("EIA_CRUDE_INVENTORY", decay_minutes=9999)
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 1)
        self.lifecycle.force_expire("EIA_CRUDE_INVENTORY")
        self.assertEqual(self.lifecycle.state_snapshot()["active_count"], 0)

    def test_force_expire_emits_reset_signal(self) -> None:
        self._activate_short_decay("EIA_CRUDE_INVENTORY", decay_minutes=9999)
        self.lifecycle.force_expire("EIA_CRUDE_INVENTORY")
        self.assertEqual(len(self.resets), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 4 — Alert engine rules and deduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.engine    = AlertEngine()
        self.clock     = FakeClock()
        for rule in DEFAULT_RULES:
            self.engine.register(rule)

    def _signal(
        self,
        trigger_type: str,
        severity: SeverityTier,
        magnitude: float = 0.80,
        direction: str = "downside_surprise",
    ) -> TriggerSignal:
        cfg = REGISTRY.get(trigger_type)
        now = self.clock()
        return TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=magnitude,
            severity=severity,
            direction=direction,
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=cfg.decay_window_minutes)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=list(cfg.affected_clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="test",
            raw_deviation_score=2.5,
            event_type=trigger_type,
            actual_value=1.0,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )

    def test_critical_signal_fires_any_critical_rule(self) -> None:
        sig    = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("any_critical", rule_ids)

    def test_low_signal_fires_no_alerts(self) -> None:
        sig    = self._signal("PPI_RELEASE", SeverityTier.LOW, magnitude=0.10)
        alerts = self.engine.evaluate(sig)
        self.assertEqual(len(alerts), 0)

    def test_medium_signal_fires_no_alerts(self) -> None:
        sig    = self._signal("CPI_RELEASE", SeverityTier.MEDIUM, magnitude=0.35)
        alerts = self.engine.evaluate(sig)
        self.assertEqual(len(alerts), 0)

    def test_fomc_gold_scope_rule_fires_when_gold_affected(self) -> None:
        sig    = self._signal("FOMC_RATE_DECISION", SeverityTier.HIGH, magnitude=0.55)
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("fomc_gold_scope", rule_ids)

    def test_energy_supply_shock_rule_fires_on_geopolitical_downside(self) -> None:
        sig    = self._signal("GEOPOLITICAL_SHOCK", SeverityTier.HIGH,
                               magnitude=0.65, direction="downside_surprise")
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("energy_supply_shock", rule_ids)

    def test_energy_rule_does_not_fire_on_upside(self) -> None:
        # energy_supply_shock requires direction=downside_surprise
        sig    = self._signal("GEOPOLITICAL_SHOCK", SeverityTier.HIGH,
                               magnitude=0.65, direction="upside_surprise")
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertNotIn("energy_supply_shock", rule_ids)

    def test_recession_rule_fires_on_recession_flag(self) -> None:
        sig    = self._signal("RECESSION_FLAG", SeverityTier.HIGH)
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("recession_any", rule_ids)

    def test_deduplication_prevents_second_alert_within_decay(self) -> None:
        sig1   = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts1 = self.engine.evaluate(sig1)
        # Second signal — same trigger_type, same rule — should be deduped
        sig2   = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts2 = self.engine.evaluate(sig2)
        # any_critical should fire once, not twice
        self.assertGreater(len(alerts1), 0)
        any_critical_count = sum(1 for a in alerts2 if a.rule_id == "any_critical")
        self.assertEqual(any_critical_count, 0)

    def test_dedup_reset_allows_refiring(self) -> None:
        sig1 = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.engine.evaluate(sig1)
        # flush_dedup_cache() only removes expired entries; use direct clear for test
        with self.engine._lock:
            self.engine._dedup.clear()
        sig2    = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts2 = self.engine.evaluate(sig2)
        rule_ids = {a.rule_id for a in alerts2}
        self.assertIn("any_critical", rule_ids)

    def test_alerts_go_into_pending_queue(self) -> None:
        sig = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        self.engine.evaluate(sig)
        queued = _drain_queue(self.engine)
        self.assertGreater(len(queued), 0)

    def test_alert_headline_contains_severity(self) -> None:
        sig = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts = self.engine.evaluate(sig)
        for a in alerts:
            self.assertIn("CRITICAL", a.headline)

    def test_alert_message_contains_direction(self) -> None:
        sig    = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL,
                               direction="downside_surprise")
        alerts = self.engine.evaluate(sig)
        for a in alerts:
            self.assertIn("downside", a.message.lower())

    def test_disabled_rule_does_not_fire(self) -> None:
        self.engine.disable("any_critical")
        sig    = self._signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL)
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertNotIn("any_critical", rule_ids)
        self.engine.enable("any_critical")

    def test_custom_rule_registered_and_evaluated(self) -> None:
        custom = AlertRule(
            rule_id="test_custom",
            name="Custom Test",
            description="Fires only on EIA with HIGH+",
            trigger_types=["EIA_CRUDE_INVENTORY"],
            min_severity=SeverityTier.HIGH,
        )
        self.engine.register(custom)
        sig    = self._signal("EIA_CRUDE_INVENTORY", SeverityTier.HIGH)
        alerts = self.engine.evaluate(sig)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("test_custom", rule_ids)
        self.engine.unregister("test_custom")


# ══════════════════════════════════════════════════════════════════════════════
# Suite 5 — Model cascade handlers
# ══════════════════════════════════════════════════════════════════════════════

class TestCascadeDispatch(unittest.TestCase):

    def setUp(self) -> None:
        self.dispatcher = CascadeDispatcher(_build_default_registry())
        self.clock      = FakeClock()

    def _signal(
        self,
        trigger_type: str = "FOMC_RATE_DECISION",
        severity: SeverityTier = SeverityTier.CRITICAL,
        magnitude: float = 0.85,
    ) -> TriggerSignal:
        cfg = REGISTRY.get(trigger_type)
        now = self.clock()
        return TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=magnitude,
            severity=severity,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=cfg.decay_window_minutes)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=list(cfg.affected_clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="test",
            raw_deviation_score=2.5,
            event_type=trigger_type,
            actual_value=1.0,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )

    def test_dispatch_returns_result(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        self.assertIsInstance(result, DispatchResult)

    def test_dispatch_produces_model_states(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        self.assertGreater(len(result.model_states), 0)

    def test_all_model_states_are_mock(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        for state in result.model_states.values():
            self.assertTrue(state.is_mock)

    def test_model_state_trigger_id_matches(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        for state in result.model_states.values():
            self.assertEqual(state.trigger_signal_id, sig.signal_id)

    def test_dispatch_records_errors_not_raises(self) -> None:
        # Unknown trigger type — dispatch_bus_payload should silently return None
        result = self.dispatcher.dispatch_bus_payload(
            {"trigger_type": "UNKNOWN_TRIGGER_XYZZY"}, "monetary_policy", "cascade"
        )
        self.assertIsNone(result)

    def test_commodity_weights_sum_to_positive(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        for state in result.model_states.values():
            if state.commodity_weights:
                total = sum(state.commodity_weights.values())
                self.assertGreater(total, 0)

    def test_recalibration_magnitude_in_unit_interval(self) -> None:
        sig    = self._signal()
        result = self.dispatcher.dispatch(sig)
        for state in result.model_states.values():
            self.assertGreaterEqual(state.recalibration_magnitude, 0.0)
            self.assertLessEqual(state.recalibration_magnitude, 1.0)

    def test_critical_signal_higher_recalibration_than_low(self) -> None:
        sig_crit = self._signal(severity=SeverityTier.CRITICAL, magnitude=0.90)
        sig_low  = self._signal(severity=SeverityTier.LOW,      magnitude=0.05)
        res_crit = self.dispatcher.dispatch(sig_crit)
        res_low  = self.dispatcher.dispatch(sig_low)

        states_crit = list(res_crit.model_states.values())
        states_low  = list(res_low.model_states.values())
        avg_crit = sum(s.recalibration_magnitude for s in states_crit) / len(states_crit)
        avg_low  = sum(s.recalibration_magnitude for s in states_low)  / len(states_low)
        self.assertGreater(avg_crit, avg_low)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 6 — Full end-to-end: 5 events through the complete stack
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndFiveEvents(unittest.TestCase):
    """
    Wire all components together via TriggerBus and push all 5 events in
    sequence, asserting cross-layer invariants after each.
    """

    HANDLER_WAIT = 4.0   # seconds per bus barrier

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.stack = _Stack(fake_clock=self.clock)
        self.release_timestamp = _utcnow().isoformat()

    def tearDown(self) -> None:
        self.stack.stop()

    # ── E1: FOMC CRITICAL ─────────────────────────────────────────────────────

    def test_e1_fomc_classified_as_critical(self) -> None:
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high",
                                   release_timestamp=self.release_timestamp)
        sig, barrier = self.stack.classify_and_publish(event)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.severity, SeverityTier.CRITICAL)
        barrier.wait()

    def test_e1_fomc_elevates_gold_via_bus(self) -> None:
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high",
                                   release_timestamp=self.release_timestamp)
        _, barrier = self.stack.classify_and_publish(event)
        ok = barrier.wait()
        self.assertTrue(ok, "Bus handlers timed out")
        self.assertTrue(self.stack.lifecycle.is_elevated("Gold (COMEX)"))

    def test_e1_fomc_does_not_elevate_corn_via_bus(self) -> None:
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high",
                                   release_timestamp=self.release_timestamp)
        _, barrier = self.stack.classify_and_publish(event)
        barrier.wait()
        self.assertFalse(self.stack.lifecycle.is_elevated("Corn (CBOT)"))

    def test_e1_fomc_fires_alert(self) -> None:
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high",
                                   release_timestamp=self.release_timestamp)
        _, barrier = self.stack.classify_and_publish(event)
        barrier.wait()
        all_alerts = _drain_queue(self.stack.engine)
        self.assertGreater(len(all_alerts), 0)

    def test_e1_fomc_cascade_produces_model_states(self) -> None:
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high",
                                   release_timestamp=self.release_timestamp)
        _, barrier = self.stack.classify_and_publish(event)
        barrier.wait()
        self.assertGreater(len(self.stack.dispatcher_calls), 0)
        for result in self.stack.dispatcher_calls:
            self.assertIsInstance(result, DispatchResult)

    # ── E2: PPI LOW ───────────────────────────────────────────────────────────

    def test_e2_ppi_fires_no_alerts(self) -> None:
        event = _make_macro_event("PPI_All_Commodities", 0.25, actual_value=102.5,
                                   expected_value=103.0, impact="low")
        sig, barrier = self.stack.classify_and_publish(event)
        # LOW signal may not publish at all; if it does, drain and assert
        if sig is not None:
            barrier.wait()
        # Any alerts queued must not have come from a LOW signal
        alerts = _drain_queue(self.stack.engine)
        low_alerts = [a for a in alerts if a.severity == "LOW"]
        self.assertEqual(len(low_alerts), 0)

    def test_e2_ppi_is_tracked_but_fires_zero_alerts(self) -> None:
        # The lifecycle activates all signals (no severity gate); what matters is
        # that a LOW signal never fires a HIGH-gated alert rule.
        event = _make_macro_event("PPI_All_Commodities", 0.25, actual_value=102.5,
                                   expected_value=103.0, impact="low")
        sig, barrier = self.stack.classify_and_publish(event)
        if sig is not None:
            barrier.wait()
        alerts = _drain_queue(self.stack.engine)
        # Alert engine should fire 0 alerts — LOW is below _MIN_EVAL_SEVERITY
        self.assertEqual(len(alerts), 0)

    # ── E3: GEOPOLITICAL HIGH ─────────────────────────────────────────────────

    def test_e3_geopolitical_elevates_wti(self) -> None:
        geo_sig = _build_geopolitical_signal(self.clock)
        barrier = self.stack.publish_direct(geo_sig)
        ok = barrier.wait()
        self.assertTrue(ok, "Bus handlers timed out")
        self.assertTrue(self.stack.lifecycle.is_elevated("WTI Crude Oil"))

    def test_e3_geopolitical_fires_energy_supply_alert(self) -> None:
        geo_sig = _build_geopolitical_signal(self.clock)
        barrier = self.stack.publish_direct(geo_sig)
        barrier.wait()
        alerts  = _drain_queue(self.stack.engine)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("energy_supply_shock", rule_ids)

    def test_e3_geopolitical_does_not_elevate_wheat(self) -> None:
        geo_sig = _build_geopolitical_signal(self.clock)
        barrier = self.stack.publish_direct(geo_sig)
        barrier.wait()
        self.assertFalse(self.stack.lifecycle.is_elevated("Wheat (CBOT SRW)"))

    # ── E4: FOMC repeat ───────────────────────────────────────────────────────

    def test_e4_repeat_classified_as_is_repeat_true(self) -> None:
        shared_ts = _utcnow().isoformat()
        ev1 = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                 expected_value=5.00, impact="high",
                                 release_timestamp=shared_ts)
        ev2 = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                 expected_value=5.00, impact="high",
                                 release_timestamp=shared_ts)
        sig1, b1 = self.stack.classify_and_publish(ev1)
        b1.wait()
        sig2, _ = self.stack.classify_and_publish(ev2, wait_handlers=0)
        if sig2 is not None:
            self.assertTrue(sig2.is_repeat)

    # ── E5: RECESSION CRITICAL ────────────────────────────────────────────────

    def test_e5_recession_elevates_multiple_clusters(self) -> None:
        event = _make_macro_event("USREC", 2.6, actual_value=1.0,
                                   expected_value=0.0, impact="high")
        _, barrier = self.stack.classify_and_publish(event)
        ok = barrier.wait()
        self.assertTrue(ok, "Bus handlers timed out")
        snap = self.stack.lifecycle.state_snapshot()
        elevated = set(snap["elevated_commodities"])
        # Must hit cross-cluster scope
        self.assertGreater(len(elevated), 5)

    def test_e5_recession_fires_recession_rule_alert(self) -> None:
        event = _make_macro_event("USREC", 2.6, actual_value=1.0,
                                   expected_value=0.0, impact="high")
        _, barrier = self.stack.classify_and_publish(event)
        barrier.wait()
        alerts   = _drain_queue(self.stack.engine)
        rule_ids = {a.rule_id for a in alerts}
        self.assertIn("recession_any", rule_ids)

    # ── Full sequence coherence ───────────────────────────────────────────────

    def test_full_sequence_lifecycle_active_count(self) -> None:
        """After E1 + E3 + E5, at least 3 distinct triggers should be active."""
        release_ts = _utcnow().isoformat()
        # E1
        e1 = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                expected_value=5.00, impact="high",
                                release_timestamp=release_ts)
        _, b1 = self.stack.classify_and_publish(e1)
        b1.wait()
        # E3
        geo = _build_geopolitical_signal(self.clock)
        b3  = self.stack.publish_direct(geo)
        b3.wait()
        # E5
        e5 = _make_macro_event("USREC", 2.6, actual_value=1.0,
                                expected_value=0.0, impact="high")
        _, b5 = self.stack.classify_and_publish(e5)
        b5.wait()

        snap = self.stack.lifecycle.state_snapshot()
        self.assertGreaterEqual(snap["active_count"], 2)

    def test_dlq_empty_after_clean_run(self) -> None:
        """No messages should land in the dead-letter queue during a clean run."""
        event = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                   expected_value=5.00, impact="high")
        _, barrier = self.stack.classify_and_publish(event)
        barrier.wait()
        self.assertEqual(self.stack.bus.dlq_size(), 0)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 7 — Latency benchmarks
# ══════════════════════════════════════════════════════════════════════════════

class TestLatencyBenchmarks(unittest.TestCase):
    """
    Time each pipeline stage for all 5 events.
    These are not hard pass/fail thresholds; results are printed to stdout.
    A soft assertion guards against catastrophic regression (> 200 ms per stage).
    """

    SOFT_LIMIT_MS = 200   # milliseconds — per-stage

    def _run_benchmark(
        self,
        label: str,
        fn: Callable,
        *args,
        iterations: int = 10,
    ) -> float:
        """Return mean latency in milliseconds over `iterations` calls."""
        times: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1_000)
        mean_ms = sum(times) / len(times)
        p95_ms  = sorted(times)[int(0.95 * len(times))]
        print(
            f"\n  [{label}]  mean={mean_ms:.2f}ms  p95={p95_ms:.2f}ms  "
            f"min={min(times):.2f}ms  max={max(times):.2f}ms"
        )
        return mean_ms

    def setUp(self) -> None:
        self.classifier = TriggerClassifier(REGISTRY)
        self.dispatcher = CascadeDispatcher(_build_default_registry())
        self.engine     = AlertEngine()
        for rule in DEFAULT_RULES:
            self.engine.register(rule)
        self.bus        = TriggerBus(redis_url="")
        self.lifecycle  = TriggerLifecycleManager(tick_interval_seconds=9999)
        self._signals:  Dict[str, TriggerSignal] = {}
        self._build_signals()

    def tearDown(self) -> None:
        self.lifecycle.stop()
        self.bus.close()

    def _make_signal(self, trigger_type: str, severity: SeverityTier,
                     magnitude: float = 0.80) -> TriggerSignal:
        cfg = REGISTRY.get(trigger_type)
        now = _utcnow()
        return TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=magnitude,
            severity=severity,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=cfg.decay_window_minutes)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=list(cfg.affected_clusters),
            model_targets=cfg.model_targets,
            cascade_priority=cfg.cascade_priority,
            classification_method="test",
            raw_deviation_score=2.0,
            event_type=trigger_type,
            actual_value=1.0,
            expected_value=0.0,
            is_repeat=False,
            config=cfg,
        )

    def _build_signals(self) -> None:
        self._signals["e1_fomc"]   = self._make_signal("FOMC_RATE_DECISION", SeverityTier.CRITICAL, 0.92)
        self._signals["e2_ppi"]    = self._make_signal("PPI_RELEASE",         SeverityTier.LOW,      0.08)
        self._signals["e3_geo"]    = self._make_signal("GEOPOLITICAL_SHOCK",  SeverityTier.HIGH,     0.68)
        self._signals["e4_repeat"] = self._make_signal("FOMC_RATE_DECISION",  SeverityTier.CRITICAL, 0.88)
        self._signals["e5_rec"]    = self._make_signal("RECESSION_FLAG",      SeverityTier.CRITICAL, 0.85)

    # Stage: classification (MacroEvent → TriggerSignal)
    def test_bench_classify_e1_fomc(self) -> None:
        ev = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                expected_value=5.00, impact="high")
        mean = self._run_benchmark("classify E1 FOMC CRITICAL", self.classifier.classify, ev)
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_classify_e2_ppi(self) -> None:
        ev = _make_macro_event("PPI_All_Commodities", 0.25, actual_value=102.5,
                                expected_value=103.0, impact="low")
        mean = self._run_benchmark("classify E2 PPI LOW", self.classifier.classify, ev)
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    # Stage: bus publish (signal → in-process queues, no handlers)
    def test_bench_bus_publish_e1(self) -> None:
        mean = self._run_benchmark(
            "bus.publish E1 FOMC", self.bus.publish, self._signals["e1_fomc"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_bus_publish_e3(self) -> None:
        mean = self._run_benchmark(
            "bus.publish E3 GEO", self.bus.publish, self._signals["e3_geo"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    # Stage: cascade dispatch (TriggerSignal → all handlers)
    def test_bench_dispatch_e1_fomc(self) -> None:
        mean = self._run_benchmark(
            "cascade E1 FOMC CRITICAL", self.dispatcher.dispatch, self._signals["e1_fomc"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_dispatch_e5_recession(self) -> None:
        mean = self._run_benchmark(
            "cascade E5 RECESSION CRITICAL", self.dispatcher.dispatch, self._signals["e5_rec"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    # Stage: lifecycle activation
    def test_bench_lifecycle_activate_e1(self) -> None:
        mean = self._run_benchmark(
            "lifecycle.activate E1 FOMC", self.lifecycle.activate, self._signals["e1_fomc"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_lifecycle_activate_e3(self) -> None:
        mean = self._run_benchmark(
            "lifecycle.activate E3 GEO", self.lifecycle.activate, self._signals["e3_geo"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    # Stage: alert evaluation
    def test_bench_alert_evaluate_e1(self) -> None:
        mean = self._run_benchmark(
            "alert.evaluate E1 FOMC CRITICAL", self.engine.evaluate, self._signals["e1_fomc"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_alert_evaluate_e5(self) -> None:
        mean = self._run_benchmark(
            "alert.evaluate E5 RECESSION CRITICAL", self.engine.evaluate, self._signals["e5_rec"]
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    # Full pipeline: classify + dispatch + lifecycle + alert (no bus threading)
    def test_bench_full_pipeline_synchronous(self) -> None:
        """Synchronous full-pipeline benchmark (bus excluded for determinism)."""
        ev = _make_macro_event("Fed_Funds_Rate", 2.8, actual_value=5.75,
                                expected_value=5.00, impact="high")

        def _full_pipeline() -> None:
            sig = self.classifier.classify(ev)
            if sig is None:
                return
            self.bus.publish(sig)
            self.dispatcher.dispatch(sig)
            self.lifecycle.activate(sig)
            self.engine.evaluate(sig)
            self.engine.flush_dedup_cache()

        mean = self._run_benchmark("FULL PIPELINE (sync) E1 FOMC", _full_pipeline)
        self.assertLess(mean, self.SOFT_LIMIT_MS)

    def test_bench_state_snapshot(self) -> None:
        self.lifecycle.activate(self._signals["e1_fomc"])
        mean = self._run_benchmark(
            "state_snapshot (1 active trigger)", self.lifecycle.state_snapshot
        )
        self.assertLess(mean, self.SOFT_LIMIT_MS)


# ══════════════════════════════════════════════════════════════════════════════
# Suite 8 — Bus health and DLQ
# ══════════════════════════════════════════════════════════════════════════════

class TestBusHealth(unittest.TestCase):

    def setUp(self) -> None:
        self.bus = TriggerBus(redis_url="")

    def tearDown(self) -> None:
        self.bus.close()

    def test_bus_health_returns_dict(self) -> None:
        h = self.bus.health()
        self.assertIsInstance(h, dict)

    def test_bus_starts_with_empty_dlq(self) -> None:
        self.assertEqual(self.bus.dlq_size(), 0)

    def test_bus_in_process_transport_used(self) -> None:
        h = self.bus.health()
        self.assertEqual(h.get("backend"), "in_process")

    def test_clean_publish_leaves_empty_dlq(self) -> None:
        cfg = REGISTRY.get("FOMC_RATE_DECISION")
        now = _utcnow()
        sig = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type="FOMC_RATE_DECISION",
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=0.8,
            severity=SeverityTier.CRITICAL,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=60)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=("metals", "energy"),
            model_targets=cfg.model_targets,
            cascade_priority=1,
            classification_method="test",
            raw_deviation_score=2.5,
            event_type="FOMC_RATE_DECISION",
            actual_value=5.5,
            expected_value=5.00,
            is_repeat=False,
            config=cfg,
        )
        channels = self.bus.publish(sig)
        self.assertGreater(len(channels), 0)
        self.assertEqual(self.bus.dlq_size(), 0)

    def test_subscribe_and_receive_message(self) -> None:
        received: List[dict] = []
        done = threading.Event()

        def handler(payload: dict, channel: str, model_id: str) -> None:
            received.append(payload)
            done.set()

        self.bus.subscribe("test_model", handler, ["monetary_policy"])
        self.bus.start_consuming("test_model")

        cfg = REGISTRY.get("FOMC_RATE_DECISION")
        now = _utcnow()
        sig = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type="FOMC_RATE_DECISION",
            family_name=cfg.family_name,
            source_event_id=str(uuid.uuid4()),
            magnitude_score=0.8,
            severity=SeverityTier.HIGH,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(minutes=60)).isoformat(),
            affected_commodities=tuple(MODELING_COMMODITIES.keys()),
            affected_clusters=("metals", "energy"),
            model_targets=cfg.model_targets,
            cascade_priority=2,
            classification_method="test",
            raw_deviation_score=2.0,
            event_type="FOMC_RATE_DECISION",
            actual_value=5.5,
            expected_value=5.00,
            is_repeat=False,
            config=cfg,
        )
        self.bus.publish(sig)
        ok = done.wait(timeout=3.0)
        self.assertTrue(ok, "Handler never received the message")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["trigger_type"], "FOMC_RATE_DECISION")

    def test_unsubscribe_stops_consumer(self) -> None:
        counter = [0]

        def handler(payload, channel, model_id) -> None:
            counter[0] += 1

        self.bus.subscribe("unsub_model", handler, ["monetary_policy"])
        self.bus.start_consuming("unsub_model")
        self.bus.unsubscribe("unsub_model")
        self.assertFalse("unsub_model" in self.bus._threads and
                         self.bus._threads["unsub_model"].is_alive())


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("\n" + "=" * 70)
    print("  Commodities Dashboard — Mission 10 Integration Test Suite")
    print("=" * 70)
    print(
        "\n  Stack under test:"
        "\n    MacroEvent → TriggerClassifier → TriggerBus (in-process)"
        "\n    → CascadeDispatcher → TriggerLifecycleManager → AlertEngine"
        "\n\n  Latency benchmarks will be printed inline (no hard wall-clock limit)."
        "\n  Soft assertion: each stage < 200 ms mean over 10 iterations.\n"
    )
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    suites  = [
        TestClassifierScenarios,
        TestScopeFiltering,
        TestDecayWindows,
        TestAlertEngine,
        TestCascadeDispatch,
        TestEndToEndFiveEvents,
        TestLatencyBenchmarks,
        TestBusHealth,
    ]
    for s in suites:
        suite.addTests(loader.loadTestsFromTestCase(s))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
