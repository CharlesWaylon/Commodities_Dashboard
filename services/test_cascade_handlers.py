"""
Test suite — Model Cascade Handlers  (Mission 5)
=================================================
25 tests, covering:
  - CorrelationHandler regime scalar accumulation and weight bounds
  - MomentumHandler dampening, direction bias, and weight non-negativity
  - SentimentHandler ensemble-weight adjustment (news vs. data triggers)
  - AttentionHandler softmax invariant, entropy direction, and concentration
  - CascadeDispatcher routing, "all" target, alias deduplication, error capture
  - ModelState contract: is_mock, severity label, regime override, history
  - StateHistory push/query

Run with:
    python -m pytest services/test_cascade_handlers.py -v
"""

import math
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple

import pytest

from models.config import MODELING_COMMODITIES
from services.cascade_handlers import (
    AttentionHandler,
    CascadeDispatcher,
    CorrelationHandler,
    DispatchResult,
    MomentumHandler,
    SentimentHandler,
    _NEWS_DRIVEN,
    _DATA_DRIVEN,
)
from services.model_state import ModelState, StateHistory
from services.trigger_classifier import SeverityTier, TriggerSignal
from services.trigger_config import ThresholdBand, TriggerConfig


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

_ALL_COMMODITIES: Tuple[str, ...] = tuple(MODELING_COMMODITIES.keys())

_SAMPLE_THRESHOLDS = {
    "mild":     ThresholdBand(min=0.15, max=0.40, label="mild"),
    "moderate": ThresholdBand(min=0.40, max=0.70, label="moderate"),
    "severe":   ThresholdBand(min=0.70, max=1.00, label="severe"),
}

_SAMPLE_CONFIG = TriggerConfig(
    trigger_type="FOMC_RATE_DECISION",
    family_name="fomc_rate_decision",
    display_name="FOMC Rate Decision",
    source="calendar",
    description="Test trigger",
    thresholds=_SAMPLE_THRESHOLDS,
    affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
    affected_commodities=_ALL_COMMODITIES,
    cascade_priority=1,
    decay_window_minutes=5760,
    model_targets=("macro_router", "xgboost", "random_forest", "meta_predictor"),
)


def _make_signal(
    trigger_type: str = "FOMC_RATE_DECISION",
    family_name: str = "fomc_rate_decision",
    magnitude: float = 0.50,
    severity: SeverityTier = SeverityTier.MEDIUM,
    direction: str = "downside_surprise",
    model_targets: Tuple[str, ...] = ("macro_router", "xgboost"),
    affected_commodities: Tuple[str, ...] = _ALL_COMMODITIES,
    cascade_priority: int = 1,
    config: TriggerConfig = _SAMPLE_CONFIG,
    is_repeat: bool = False,
) -> TriggerSignal:
    now = datetime.now(timezone.utc)
    return TriggerSignal(
        signal_id=str(uuid.uuid4()),
        trigger_type=trigger_type,
        family_name=family_name,
        source_event_id=str(uuid.uuid4()),
        magnitude_score=magnitude,
        severity=severity,
        direction=direction,
        release_timestamp=now.isoformat(),
        classified_at=now.isoformat(),
        decay_expires_at=(now + timedelta(minutes=5760)).isoformat(),
        affected_commodities=affected_commodities,
        affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
        model_targets=model_targets,
        cascade_priority=cascade_priority,
        classification_method="z_score",
        raw_deviation_score=1.5,
        event_type=trigger_type,
        actual_value=105.0,
        expected_value=100.0,
        is_repeat=is_repeat,
        config=config,
    )


def _make_config(
    trigger_type: str,
    model_targets: Tuple[str, ...],
    decay_window_minutes: int = 1440,
    cascade_priority: int = 2,
) -> TriggerConfig:
    return TriggerConfig(
        trigger_type=trigger_type,
        family_name=trigger_type.lower(),
        display_name=trigger_type,
        source="calendar",
        description="",
        thresholds=_SAMPLE_THRESHOLDS,
        affected_clusters=("energy", "metals", "agriculture", "livestock", "digital"),
        affected_commodities=_ALL_COMMODITIES,
        cascade_priority=cascade_priority,
        decay_window_minutes=decay_window_minutes,
        model_targets=model_targets,
    )


# ── CorrelationHandler ─────────────────────────────────────────────────────────

class TestCorrelationHandler:

    def test_low_signal_decays_regime_scalar(self):
        handler = CorrelationHandler()
        handler._regime_scalar = 0.30
        sig = _make_signal(severity=SeverityTier.LOW, magnitude=0.20)
        handler.on_trigger(sig)
        assert handler._regime_scalar < 0.30, "LOW should decay regime scalar"

    def test_critical_signal_raises_regime_scalar(self):
        handler = CorrelationHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=0.90)
        handler.on_trigger(sig)
        assert handler._regime_scalar > 0.30, "CRITICAL should raise regime scalar significantly"

    def test_regime_scalar_clamps_at_one(self):
        handler = CorrelationHandler()
        handler._regime_scalar = 0.90
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0)
        handler.on_trigger(sig)
        assert handler._regime_scalar <= 1.0

    def test_regime_scalar_clamps_at_zero(self):
        handler = CorrelationHandler()
        handler._regime_scalar = 0.0
        sig = _make_signal(severity=SeverityTier.LOW, magnitude=0.10)
        handler.on_trigger(sig)
        assert handler._regime_scalar >= 0.0

    def test_weights_in_valid_range(self):
        handler = CorrelationHandler()
        sig = _make_signal(severity=SeverityTier.HIGH, magnitude=0.75)
        state = handler.on_trigger(sig)
        for commodity, w in state.commodity_weights.items():
            assert 0.0 <= w <= 1.0, f"{commodity} weight {w} out of [0, 1]"

    def test_weights_cover_all_commodities(self):
        handler = CorrelationHandler()
        sig = _make_signal()
        state = handler.on_trigger(sig)
        assert set(state.commodity_weights.keys()) == set(MODELING_COMMODITIES.keys())

    def test_implied_correlation_in_metadata(self):
        handler = CorrelationHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0)
        state = handler.on_trigger(sig)
        assert "implied_correlation" in state.metadata
        corr = state.metadata["implied_correlation"]
        assert 0.0 <= corr <= 1.0


# ── MomentumHandler ────────────────────────────────────────────────────────────

class TestMomentumHandler:

    def test_critical_signal_high_dampener(self):
        handler = MomentumHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0)
        handler.on_trigger(sig)
        for commodity in sig.affected_commodities:
            assert handler._dampener[commodity] > 0.0

    def test_low_signal_minimal_dampener(self):
        handler = MomentumHandler()
        sig = _make_signal(severity=SeverityTier.LOW, magnitude=0.20)
        handler.on_trigger(sig)
        for commodity in sig.affected_commodities:
            assert handler._dampener[commodity] < 0.10

    def test_weights_non_negative(self):
        handler = MomentumHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0)
        state = handler.on_trigger(sig)
        for commodity, w in state.commodity_weights.items():
            assert w >= 0.0, f"{commodity} weight {w} is negative"

    def test_downside_negative_direction_bias(self):
        handler = MomentumHandler()
        sig = _make_signal(direction="downside_surprise", magnitude=0.80,
                           severity=SeverityTier.HIGH)
        handler.on_trigger(sig)
        biases = [handler._direction_bias[c] for c in sig.affected_commodities]
        assert all(b < 0 for b in biases), "Downside surprise should produce negative bias"

    def test_upside_positive_direction_bias(self):
        handler = MomentumHandler()
        sig = _make_signal(direction="upside_surprise", magnitude=0.80,
                           severity=SeverityTier.HIGH)
        handler.on_trigger(sig)
        biases = [handler._direction_bias[c] for c in sig.affected_commodities]
        assert all(b > 0 for b in biases)

    def test_metadata_keys_present(self):
        handler = MomentumHandler()
        sig = _make_signal(severity=SeverityTier.MEDIUM)
        state = handler.on_trigger(sig)
        assert "applied_dampener" in state.metadata
        assert "max_dampener_for_severity" in state.metadata
        assert "mean_affected_dampener" in state.metadata


# ── SentimentHandler ───────────────────────────────────────────────────────────

class TestSentimentHandler:

    def test_news_driven_raises_weight(self):
        handler = SentimentHandler()
        base = handler._ensemble_weight
        news_trigger = next(iter(_NEWS_DRIVEN))
        cfg = _make_config(news_trigger, model_targets=("meta_predictor",))
        sig = _make_signal(trigger_type=news_trigger, severity=SeverityTier.HIGH,
                           magnitude=0.80, config=cfg)
        handler.on_trigger(sig)
        assert handler._ensemble_weight > base

    def test_data_driven_lowers_weight(self):
        handler = SentimentHandler()
        base = handler._ensemble_weight
        data_trigger = next(iter(_DATA_DRIVEN))
        cfg = _make_config(data_trigger, model_targets=("meta_predictor",))
        sig = _make_signal(trigger_type=data_trigger, severity=SeverityTier.HIGH,
                           magnitude=0.80, config=cfg)
        handler.on_trigger(sig)
        assert handler._ensemble_weight < base

    def test_weight_clamped_to_max(self):
        handler = SentimentHandler()
        handler._ensemble_weight = SentimentHandler.MAX_WEIGHT - 0.01
        news_trigger = next(iter(_NEWS_DRIVEN))
        cfg = _make_config(news_trigger, model_targets=("meta_predictor",))
        sig = _make_signal(trigger_type=news_trigger, severity=SeverityTier.CRITICAL,
                           magnitude=1.0, config=cfg)
        handler.on_trigger(sig)
        assert handler._ensemble_weight <= SentimentHandler.MAX_WEIGHT

    def test_weight_clamped_to_min(self):
        handler = SentimentHandler()
        handler._ensemble_weight = SentimentHandler.MIN_WEIGHT + 0.01
        data_trigger = next(iter(_DATA_DRIVEN))
        cfg = _make_config(data_trigger, model_targets=("meta_predictor",))
        sig = _make_signal(trigger_type=data_trigger, severity=SeverityTier.CRITICAL,
                           magnitude=1.0, config=cfg)
        handler.on_trigger(sig)
        assert handler._ensemble_weight >= SentimentHandler.MIN_WEIGHT

    def test_category_recorded_in_metadata(self):
        handler = SentimentHandler()
        news_trigger = next(iter(_NEWS_DRIVEN))
        cfg = _make_config(news_trigger, model_targets=("meta_predictor",))
        sig = _make_signal(trigger_type=news_trigger, config=cfg)
        state = handler.on_trigger(sig)
        assert state.metadata.get("trigger_category") == "news_driven"


# ── AttentionHandler ───────────────────────────────────────────────────────────

class TestAttentionHandler:

    def test_attention_sums_to_one_after_trigger(self):
        handler = AttentionHandler()
        sig = _make_signal(severity=SeverityTier.HIGH, magnitude=0.70)
        handler.on_trigger(sig)
        total = sum(handler._attention.values())
        assert abs(total - 1.0) < 1e-6, f"Attention sums to {total}, not 1.0"

    def test_attention_sums_to_one_after_multiple_triggers(self):
        handler = AttentionHandler()
        for _ in range(5):
            sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=0.90)
            handler.on_trigger(sig)
        total = sum(handler._attention.values())
        assert abs(total - 1.0) < 1e-6

    def test_affected_commodities_gain_attention(self):
        handler = AttentionHandler()
        pre = {k: v for k, v in handler._attention.items()}
        # Use a small subset to make the gain measurable
        subset = _ALL_COMMODITIES[:3]
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0,
                           affected_commodities=subset)
        handler.on_trigger(sig)
        for c in subset:
            assert handler._attention[c] > pre[c], f"{c} should have gained attention"

    def test_entropy_decreases_on_high_severity(self):
        handler = AttentionHandler()
        # Initial entropy is maximal (uniform)
        initial_entropy = -sum(v * math.log(v + 1e-12) for v in handler._attention.values())
        sig = _make_signal(severity=SeverityTier.CRITICAL, magnitude=1.0,
                           affected_commodities=_ALL_COMMODITIES[:5])
        handler.on_trigger(sig)
        post_entropy = -sum(v * math.log(v + 1e-12) for v in handler._attention.values())
        assert post_entropy < initial_entropy, "CRITICAL trigger should concentrate attention"

    def test_metadata_entropy_in_range(self):
        handler = AttentionHandler()
        sig = _make_signal(severity=SeverityTier.MEDIUM, magnitude=0.50)
        state = handler.on_trigger(sig)
        entropy = state.metadata["attention_entropy"]
        assert 0.0 <= entropy <= 1.0


# ── ModelState contract ────────────────────────────────────────────────────────

class TestModelStateContract:

    def test_is_mock_true(self):
        handler = CorrelationHandler()
        state = handler.on_trigger(_make_signal())
        assert state.is_mock is True

    def test_severity_label_matches_signal(self):
        handler = MomentumHandler()
        sig = _make_signal(severity=SeverityTier.HIGH)
        state = handler.on_trigger(sig)
        assert state.severity_applied == "HIGH"

    def test_regime_override_critical_downside(self):
        handler = CorrelationHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, direction="downside_surprise")
        state = handler.on_trigger(sig)
        assert state.regime_override == "crisis"

    def test_regime_override_critical_upside(self):
        handler = AttentionHandler()
        sig = _make_signal(severity=SeverityTier.CRITICAL, direction="upside_surprise")
        state = handler.on_trigger(sig)
        assert state.regime_override == "risk_on"

    def test_regime_override_low_is_none(self):
        handler = SentimentHandler()
        sig = _make_signal(severity=SeverityTier.LOW, direction="downside_surprise")
        state = handler.on_trigger(sig)
        assert state.regime_override is None

    def test_recalibration_magnitude_in_range(self):
        for handler in [CorrelationHandler(), MomentumHandler(), SentimentHandler(), AttentionHandler()]:
            sig = _make_signal(severity=SeverityTier.HIGH, magnitude=0.80)
            state = handler.on_trigger(sig)
            assert 0.0 <= state.recalibration_magnitude <= 1.0, \
                f"{handler.model_id}: recal_mag {state.recalibration_magnitude} out of range"

    def test_state_history_accumulates(self):
        handler = CorrelationHandler()
        for i in range(3):
            handler.on_trigger(_make_signal())
        assert len(handler.history) == 3

    def test_state_history_latest_is_most_recent(self):
        handler = MomentumHandler()
        s1 = handler.on_trigger(_make_signal(magnitude=0.10))
        s2 = handler.on_trigger(_make_signal(magnitude=0.90))
        assert handler.history.latest().trigger_signal_id == s2.trigger_signal_id

    def test_state_history_for_trigger_filter(self):
        handler = SentimentHandler()
        handler.on_trigger(_make_signal(trigger_type="FOMC_RATE_DECISION"))
        handler.on_trigger(_make_signal(trigger_type="CPI_RELEASE",
                                        config=_make_config("CPI_RELEASE", ("meta_predictor",))))
        fomc_states = handler.history.for_trigger("FOMC_RATE_DECISION")
        assert len(fomc_states) == 1
        assert fomc_states[0].trigger_type == "FOMC_RATE_DECISION"


# ── CascadeDispatcher ──────────────────────────────────────────────────────────

class TestCascadeDispatcher:

    def _fresh(self) -> CascadeDispatcher:
        """Dispatcher with four independent handler instances."""
        return CascadeDispatcher()

    def test_dispatch_returns_dispatch_result(self):
        d = self._fresh()
        sig = _make_signal(model_targets=("macro_router",))
        result = d.dispatch(sig)
        assert isinstance(result, DispatchResult)

    def test_dispatch_all_target_invokes_all_handlers(self):
        d = self._fresh()
        sig = _make_signal(
            model_targets=("all",),
            config=_make_config("GEOPOLITICAL_SHOCK", model_targets=("all",)),
        )
        result = d.dispatch(sig)
        assert result.n_handlers_invoked == len(d.handlers)
        assert len(result.model_states) == len(d.handlers)

    def test_dispatch_specific_targets_only_fires_those(self):
        d = self._fresh()
        sig = _make_signal(model_targets=("macro_router",))
        result = d.dispatch(sig)
        # macro_router aliases to correlation_engine
        assert "correlation_engine" in result.model_states
        assert "momentum_model" not in result.model_states

    def test_alias_deduplication_xgboost_and_random_forest(self):
        """xgboost and random_forest both map to momentum_model — should fire once."""
        d = self._fresh()
        sig = _make_signal(model_targets=("xgboost", "random_forest"))
        result = d.dispatch(sig)
        assert result.n_handlers_invoked == 1
        assert "momentum_model" in result.model_states

    def test_unknown_target_ignored_gracefully(self):
        d = self._fresh()
        sig = _make_signal(model_targets=("nonexistent_model",))
        result = d.dispatch(sig)
        assert result.n_handlers_invoked == 0
        assert result.errors == {}

    def test_handler_error_captured_not_raised(self):
        """A handler that raises should not crash the dispatcher."""

        class BrokenHandler(CorrelationHandler):
            model_id = "broken_handler"
            def on_trigger(self, signal):
                raise RuntimeError("simulated model failure")

        d = self._fresh()
        d.register("broken", BrokenHandler())
        sig = _make_signal(model_targets=("broken",))
        result = d.dispatch(sig)
        assert "broken_handler" in result.errors
        assert "RuntimeError" in result.errors["broken_handler"]

    def test_dispatch_result_signal_id_matches(self):
        d = self._fresh()
        sig = _make_signal(model_targets=("xgboost",))
        result = d.dispatch(sig)
        assert result.signal_id == sig.signal_id

    def test_dispatch_result_trigger_type_matches(self):
        d = self._fresh()
        sig = _make_signal(trigger_type="FOMC_RATE_DECISION", model_targets=("xgboost",))
        result = d.dispatch(sig)
        assert result.trigger_type == "FOMC_RATE_DECISION"

    def test_dispatcher_history_accessible_by_model_id(self):
        d = self._fresh()
        sig = _make_signal(model_targets=("macro_router",))
        d.dispatch(sig)
        hist = d.history("correlation_engine")
        assert hist is not None
        assert len(hist) == 1

    def test_dispatcher_history_none_for_unknown_model(self):
        d = self._fresh()
        assert d.history("does_not_exist") is None
