"""
Test suite — Trigger Lifecycle Manager  (Mission 6)
=====================================================
Tests for:
  - Activation and primary-cluster scope resolution
  - Decay / expiry via injected fake clock
  - Scope enforcement (is_elevated, elevated_by)
  - Reset signal emission and handler dispatch
  - force_expire
  - state_snapshot structure
  - Background thread start/stop
  - Thread safety (multiple simultaneous active triggers)

Run with:
    python -m pytest services/test_trigger_lifecycle.py -v
"""

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from models.config import COMMODITY_SECTORS, MODELING_COMMODITIES
from services.trigger_classifier import SeverityTier, TriggerSignal
from services.trigger_config import ThresholdBand, TriggerConfig
from services.trigger_lifecycle import (
    ActiveTrigger,
    ResetSignal,
    TriggerLifecycleManager,
    _PRIMARY_CLUSTERS,
    _resolve_primary_commodities,
)


# ── Fake clock ─────────────────────────────────────────────────────────────────

class FakeClock:
    """Controllable clock for testing decay without sleeping."""

    def __init__(self, initial: datetime) -> None:
        self._now = initial

    def __call__(self) -> datetime:
        return self._now

    def advance(self, minutes: float = 0.0, seconds: float = 0.0) -> None:
        self._now += timedelta(minutes=minutes, seconds=seconds)


# ── Fixtures ───────────────────────────────────────────────────────────────────

_START = datetime(2026, 5, 15, 14, 0, 0, tzinfo=timezone.utc)

_SAMPLE_THRESHOLDS = {
    "mild":     ThresholdBand(min=0.15, max=0.40, label="mild"),
    "moderate": ThresholdBand(min=0.40, max=0.70, label="moderate"),
    "severe":   ThresholdBand(min=0.70, max=1.00, label="severe"),
}

_ALL_COMMODITIES = tuple(MODELING_COMMODITIES.keys())


def _make_config(
    trigger_type: str = "GEOPOLITICAL_SHOCK",
    decay_window_minutes: int = 60,
    model_targets: tuple = ("macro_router", "xgboost"),
    cascade_priority: int = 1,
) -> TriggerConfig:
    return TriggerConfig(
        trigger_type=trigger_type,
        family_name=trigger_type.lower(),
        display_name=trigger_type.replace("_", " ").title(),
        source="calendar",
        description="",
        thresholds=_SAMPLE_THRESHOLDS,
        affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
        affected_commodities=_ALL_COMMODITIES,
        cascade_priority=cascade_priority,
        decay_window_minutes=decay_window_minutes,
        model_targets=model_targets,
    )


def _make_signal(
    trigger_type: str = "GEOPOLITICAL_SHOCK",
    magnitude: float = 0.75,
    severity: SeverityTier = SeverityTier.HIGH,
    direction: str = "downside_surprise",
    decay_window_minutes: int = 60,
    is_repeat: bool = False,
    clock: FakeClock = None,
    config: TriggerConfig = None,
) -> TriggerSignal:
    now = clock() if clock else _START
    cfg = config or _make_config(trigger_type, decay_window_minutes)
    expires = now + timedelta(minutes=decay_window_minutes)
    return TriggerSignal(
        signal_id=str(uuid.uuid4()),
        trigger_type=trigger_type,
        family_name=trigger_type.lower(),
        source_event_id=str(uuid.uuid4()),
        magnitude_score=magnitude,
        severity=severity,
        direction=direction,
        release_timestamp=now.isoformat(),
        classified_at=now.isoformat(),
        decay_expires_at=expires.isoformat(),
        affected_commodities=_ALL_COMMODITIES,
        affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
        model_targets=cfg.model_targets,
        cascade_priority=cfg.cascade_priority,
        classification_method="z_score",
        raw_deviation_score=2.1,
        event_type=trigger_type,
        actual_value=105.0,
        expected_value=100.0,
        is_repeat=is_repeat,
        config=cfg,
    )


def _fresh(clock: FakeClock = None) -> TriggerLifecycleManager:
    return TriggerLifecycleManager(clock=clock or FakeClock(_START))


# ── Activation ─────────────────────────────────────────────────────────────────

class TestActivation:

    def test_activate_stores_trigger(self):
        mgr = _fresh()
        sig = _make_signal()
        mgr.activate(sig)
        assert len(mgr.active_triggers()) == 1

    def test_activate_returns_active_trigger(self):
        mgr = _fresh()
        sig = _make_signal()
        entry = mgr.activate(sig)
        assert isinstance(entry, ActiveTrigger)
        assert entry.signal is sig

    def test_activate_resolves_primary_clusters_geopolitical(self):
        mgr = _fresh()
        sig = _make_signal("GEOPOLITICAL_SHOCK")
        entry = mgr.activate(sig)
        # GEOPOLITICAL_SHOCK primary = energy + metals
        assert entry.primary_clusters == frozenset({"energy", "metals"})

    def test_activate_primary_commodities_subset_of_all(self):
        mgr = _fresh()
        sig = _make_signal("GEOPOLITICAL_SHOCK")
        entry = mgr.activate(sig)
        all_set = set(entry.all_commodities)
        for c in entry.primary_commodities:
            assert c in all_set

    def test_activate_primary_commodities_match_primary_sectors(self):
        mgr = _fresh()
        sig = _make_signal("USDA_WASDE_REPORT")
        entry = mgr.activate(sig)
        # WASDE primary = agriculture only
        for c in entry.primary_commodities:
            assert COMMODITY_SECTORS[c] == "agriculture", \
                f"{c} sector {COMMODITY_SECTORS[c]} is not agriculture"

    def test_activate_replaces_same_trigger_type(self):
        mgr = _fresh()
        s1 = _make_signal(magnitude=0.30)
        s2 = _make_signal(magnitude=0.80)
        mgr.activate(s1)
        mgr.activate(s2)
        active = mgr.active_triggers()
        assert len(active) == 1
        assert active[0].signal.magnitude_score == 0.80

    def test_multiple_different_triggers_can_coexist(self):
        mgr = _fresh()
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        mgr.activate(_make_signal("CPI_RELEASE", config=_make_config("CPI_RELEASE")))
        assert len(mgr.active_triggers()) == 2


# ── Scope enforcement ──────────────────────────────────────────────────────────

class TestScopeEnforcement:

    def test_is_elevated_true_for_primary_commodity(self):
        mgr = _fresh()
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        # GEOPOLITICAL primary = energy + metals; WTI Crude Oil is energy
        assert mgr.is_elevated("WTI Crude Oil") is True

    def test_is_elevated_true_for_metal_in_geopolitical(self):
        mgr = _fresh()
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        assert mgr.is_elevated("Gold (COMEX)") is True

    def test_is_elevated_false_for_non_primary_commodity(self):
        mgr = _fresh()
        # GEOPOLITICAL_SHOCK primary = energy + metals; Corn is agriculture
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        assert mgr.is_elevated("Corn (CBOT)") is False

    def test_is_elevated_false_for_livestock_on_fomc(self):
        mgr = _fresh()
        # FOMC primary = metals + energy; Live Cattle is livestock
        mgr.activate(_make_signal("FOMC_RATE_DECISION",
                                  config=_make_config("FOMC_RATE_DECISION")))
        assert mgr.is_elevated("Live Cattle") is False

    def test_is_elevated_false_when_no_active_triggers(self):
        mgr = _fresh()
        assert mgr.is_elevated("Gold (COMEX)") is False

    def test_elevated_by_returns_matching_triggers(self):
        mgr = _fresh()
        geo = mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        by = mgr.elevated_by("WTI Crude Oil")
        assert len(by) == 1
        assert by[0].signal.trigger_type == "GEOPOLITICAL_SHOCK"

    def test_elevated_by_commodity_in_two_triggers(self):
        mgr = _fresh()
        # Gold is elevated by both FOMC (metals) and GEOPOLITICAL (metals)
        mgr.activate(_make_signal("FOMC_RATE_DECISION",
                                  config=_make_config("FOMC_RATE_DECISION")))
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        by = mgr.elevated_by("Gold (COMEX)")
        assert len(by) == 2

    def test_recession_flag_elevates_all_clusters(self):
        mgr = _fresh()
        cfg = _make_config("RECESSION_FLAG")
        mgr.activate(_make_signal("RECESSION_FLAG", config=cfg))
        # RECESSION_FLAG primary = all 5 clusters
        assert mgr.is_elevated("Corn (CBOT)")     # agriculture
        assert mgr.is_elevated("Gold (COMEX)")    # metals
        assert mgr.is_elevated("WTI Crude Oil")   # energy
        assert mgr.is_elevated("Live Cattle")     # livestock
        assert mgr.is_elevated("Bitcoin")         # digital


# ── Decay and expiry ───────────────────────────────────────────────────────────

class TestDecayAndExpiry:

    def test_tick_no_expiry_within_window(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        clk.advance(minutes=30)
        resets = mgr.tick()
        assert resets == []
        assert len(mgr.active_triggers()) == 1

    def test_tick_expires_trigger_after_window(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        clk.advance(minutes=61)
        resets = mgr.tick()
        assert len(resets) == 1
        assert resets[0].trigger_type == "GEOPOLITICAL_SHOCK"

    def test_tick_removes_expired_from_active(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        clk.advance(minutes=61)
        mgr.tick()
        assert len(mgr.active_triggers()) == 0

    def test_tick_moves_to_expired_log(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        clk.advance(minutes=61)
        mgr.tick()
        assert len(mgr.expired_log()) == 1

    def test_is_elevated_false_after_expiry(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK", decay_window_minutes=60, clock=clk))
        assert mgr.is_elevated("WTI Crude Oil") is True
        clk.advance(minutes=61)
        mgr.tick()
        assert mgr.is_elevated("WTI Crude Oil") is False

    def test_remaining_ttl_positive_during_window(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        clk.advance(minutes=20)
        ttl = mgr.remaining_ttl("GEOPOLITICAL_SHOCK")
        assert ttl is not None
        assert ttl.total_seconds() > 0

    def test_remaining_ttl_none_for_unknown_type(self):
        mgr = _fresh()
        assert mgr.remaining_ttl("OPEC_PRODUCTION_DECISION") is None

    def test_zero_decay_window_expires_immediately(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=0, clock=clk))
        clk.advance(seconds=1)
        resets = mgr.tick()
        assert len(resets) == 1


# ── Reset signal ───────────────────────────────────────────────────────────────

class TestResetSignal:

    def test_reset_handler_called_on_expiry(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        received: List[ResetSignal] = []
        mgr.register_reset_handler(received.append)
        sig = _make_signal(decay_window_minutes=60, clock=clk)
        mgr.activate(sig)
        clk.advance(minutes=61)
        mgr.tick()
        assert len(received) == 1

    def test_reset_signal_origin_id_matches(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        received: List[ResetSignal] = []
        mgr.register_reset_handler(received.append)
        sig = _make_signal(decay_window_minutes=60, clock=clk)
        mgr.activate(sig)
        clk.advance(minutes=61)
        mgr.tick()
        assert received[0].origin_signal_id == sig.signal_id

    def test_reset_signal_has_primary_commodities(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        received: List[ResetSignal] = []
        mgr.register_reset_handler(received.append)
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK", decay_window_minutes=10, clock=clk))
        clk.advance(minutes=11)
        mgr.tick()
        reset = received[0]
        # Primary commodities = energy + metals
        for c in reset.primary_commodities:
            assert COMMODITY_SECTORS[c] in {"energy", "metals"}

    def test_multiple_reset_handlers_all_called(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        calls_a: List[ResetSignal] = []
        calls_b: List[ResetSignal] = []
        mgr.register_reset_handler(calls_a.append)
        mgr.register_reset_handler(calls_b.append)
        mgr.activate(_make_signal(decay_window_minutes=5, clock=clk))
        clk.advance(minutes=6)
        mgr.tick()
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_failing_reset_handler_does_not_block_others(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        calls: List[ResetSignal] = []

        def bad_handler(r):
            raise RuntimeError("handler exploded")

        mgr.register_reset_handler(bad_handler)
        mgr.register_reset_handler(calls.append)
        mgr.activate(_make_signal(decay_window_minutes=1, clock=clk))
        clk.advance(minutes=2)
        mgr.tick()
        # Second handler should still have been called
        assert len(calls) == 1


# ── force_expire ───────────────────────────────────────────────────────────────

class TestForceExpire:

    def test_force_expire_removes_trigger(self):
        mgr = _fresh()
        mgr.activate(_make_signal(decay_window_minutes=120))
        mgr.force_expire("GEOPOLITICAL_SHOCK")
        assert len(mgr.active_triggers()) == 0

    def test_force_expire_returns_reset_signal(self):
        mgr = _fresh()
        mgr.activate(_make_signal(decay_window_minutes=120))
        reset = mgr.force_expire("GEOPOLITICAL_SHOCK")
        assert isinstance(reset, ResetSignal)
        assert reset.trigger_type == "GEOPOLITICAL_SHOCK"

    def test_force_expire_calls_handlers(self):
        mgr = _fresh()
        received: List[ResetSignal] = []
        mgr.register_reset_handler(received.append)
        mgr.activate(_make_signal(decay_window_minutes=120))
        mgr.force_expire("GEOPOLITICAL_SHOCK")
        assert len(received) == 1

    def test_force_expire_returns_none_for_unknown(self):
        mgr = _fresh()
        result = mgr.force_expire("UNKNOWN_TRIGGER_TYPE")
        assert result is None


# ── State snapshot ─────────────────────────────────────────────────────────────

class TestStateSnapshot:

    def test_snapshot_empty_when_no_active_triggers(self):
        mgr = _fresh()
        snap = mgr.state_snapshot()
        assert snap["active_count"] == 0
        assert snap["elevated_commodities"] == []
        assert snap["triggers"] == []

    def test_snapshot_active_count_correct(self):
        mgr = _fresh()
        mgr.activate(_make_signal("GEOPOLITICAL_SHOCK"))
        mgr.activate(_make_signal("CPI_RELEASE", config=_make_config("CPI_RELEASE")))
        snap = mgr.state_snapshot()
        assert snap["active_count"] == 2

    def test_snapshot_elevated_commodities_union(self):
        mgr = _fresh()
        # FOMC elevates metals+energy; WASDE elevates agriculture
        mgr.activate(_make_signal("FOMC_RATE_DECISION",
                                  config=_make_config("FOMC_RATE_DECISION")))
        mgr.activate(_make_signal("USDA_WASDE_REPORT",
                                  config=_make_config("USDA_WASDE_REPORT")))
        snap = mgr.state_snapshot()
        elevated = set(snap["elevated_commodities"])
        # Must include metals, energy, and agriculture
        assert any(COMMODITY_SECTORS[c] == "metals"      for c in elevated)
        assert any(COMMODITY_SECTORS[c] == "energy"      for c in elevated)
        assert any(COMMODITY_SECTORS[c] == "agriculture" for c in elevated)

    def test_snapshot_trigger_row_keys(self):
        mgr = _fresh()
        mgr.activate(_make_signal())
        snap = mgr.state_snapshot()
        row = snap["triggers"][0]
        required = {
            "trigger_type", "display_name", "severity", "magnitude_score",
            "direction", "primary_clusters", "primary_commodity_count",
            "all_commodity_count", "activated_at", "expires_at",
            "remaining_ttl_seconds", "cascade_priority", "is_repeat",
        }
        assert required.issubset(row.keys())

    def test_snapshot_remaining_ttl_positive(self):
        clk = FakeClock(_START)
        mgr = _fresh(clk)
        mgr.activate(_make_signal(decay_window_minutes=60, clock=clk))
        snap = mgr.state_snapshot()
        assert snap["triggers"][0]["remaining_ttl_seconds"] > 0


# ── Background thread ──────────────────────────────────────────────────────────

class TestBackgroundThread:

    def test_start_and_stop_cleanly(self):
        mgr = TriggerLifecycleManager(tick_interval_seconds=0.05)
        mgr.start()
        assert mgr._bg_thread is not None
        assert mgr._bg_thread.is_alive()
        thread_ref = mgr._bg_thread
        mgr.stop(timeout=2.0)
        assert not thread_ref.is_alive()

    def test_double_start_does_not_spawn_extra_thread(self):
        mgr = TriggerLifecycleManager(tick_interval_seconds=0.1)
        mgr.start()
        t1 = mgr._bg_thread
        mgr.start()
        t2 = mgr._bg_thread
        assert t1 is t2
        mgr.stop(timeout=2.0)

    def test_background_thread_expires_triggers(self):
        """Real-time integration: background thread fires and expires a trigger."""
        received: List[ResetSignal] = []
        # Use a real clock — signal expires in 0.05 seconds, tick runs every 0.02 s
        now = datetime.now(timezone.utc)
        cfg = _make_config("GEOPOLITICAL_SHOCK", decay_window_minutes=0)
        sig = TriggerSignal(
            signal_id=str(uuid.uuid4()),
            trigger_type="GEOPOLITICAL_SHOCK",
            family_name="geopolitical_shock",
            source_event_id=str(uuid.uuid4()),
            magnitude_score=0.75,
            severity=SeverityTier.HIGH,
            direction="downside_surprise",
            release_timestamp=now.isoformat(),
            classified_at=now.isoformat(),
            decay_expires_at=(now + timedelta(milliseconds=50)).isoformat(),
            affected_commodities=_ALL_COMMODITIES,
            affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
            model_targets=("macro_router",),
            cascade_priority=1,
            classification_method="z_score",
            raw_deviation_score=2.1,
            event_type="GEOPOLITICAL_SHOCK",
            actual_value=105.0,
            expected_value=100.0,
            is_repeat=False,
            config=cfg,
        )
        mgr = TriggerLifecycleManager(tick_interval_seconds=0.02)
        mgr.register_reset_handler(received.append)
        mgr.start()
        mgr.activate(sig)
        # Wait for up to 1 s for the background thread to expire it
        deadline = time.monotonic() + 1.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.02)
        mgr.stop(timeout=2.0)
        assert len(received) == 1, "Background thread did not emit reset within 1 s"


# ── Helper function ────────────────────────────────────────────────────────────

class TestHelpers:

    def test_resolve_primary_commodities_geopolitical(self):
        primary = _resolve_primary_commodities("GEOPOLITICAL_SHOCK", _ALL_COMMODITIES)
        sectors = {COMMODITY_SECTORS[c] for c in primary}
        assert sectors <= {"energy", "metals"}

    def test_resolve_primary_commodities_unknown_type(self):
        # Unknown type falls back to all provided commodities
        primary = _resolve_primary_commodities("TOTALLY_UNKNOWN", _ALL_COMMODITIES)
        assert set(primary) == set(_ALL_COMMODITIES)

    def test_all_registry_trigger_types_have_primary_mapping(self):
        from services.trigger_config import REGISTRY
        for ttype in REGISTRY.all():
            assert ttype.trigger_type in _PRIMARY_CLUSTERS, \
                f"{ttype.trigger_type} is missing from _PRIMARY_CLUSTERS"
