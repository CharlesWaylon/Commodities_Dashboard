"""
Model Cascade Handlers  (Mission 5)
====================================
Four ``TriggerHandler`` implementations — one per model archetype — plus a
``CascadeDispatcher`` that routes ``TriggerSignal`` objects to the right
handlers based on each signal's ``model_targets`` field.

Handler archetypes
------------------
  CorrelationHandler  — statistical cross-asset correlation engine
  MomentumHandler     — ML momentum model (XGBoost / Random Forest)
  SentimentHandler    — sentiment overlay (FinBERT + Alpha Vantage)
  AttentionHandler    — deep-learning attention model (LSTM / Transformer)

All handlers use mock internal state.  The interface contract — ``on_trigger``
accepting a ``TriggerSignal`` and returning a ``ModelState`` — is what matters
at this stage; real model objects are wired in at the integration layer.

Public API
----------
  DISPATCHER                — module-level CascadeDispatcher singleton
  dispatch_signal(signal)   — convenience wrapper
"""

from __future__ import annotations

import abc
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.config import MODELING_COMMODITIES
from services.model_state import ModelState, StateHistory
from services.trigger_classifier import SeverityTier, TriggerSignal


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _regime_from_signal(signal: TriggerSignal) -> Optional[str]:
    """Map (severity, direction) → regime override string."""
    if signal.severity == SeverityTier.CRITICAL:
        return "crisis" if signal.direction == "downside_surprise" else "risk_on"
    if signal.severity == SeverityTier.HIGH:
        return "risk_off" if signal.direction == "downside_surprise" else "risk_on"
    if signal.severity == SeverityTier.MEDIUM and signal.direction == "downside_surprise":
        return "defensive"
    return None


# ── Base interface ─────────────────────────────────────────────────────────────

class TriggerHandler(abc.ABC):
    """
    Contract every model-side cascade handler must satisfy.

    Subclasses maintain their own internal mock state and update it each time
    ``on_trigger`` is called.  The returned ``ModelState`` is a complete
    snapshot — it does not require diffing against the previous state.
    """

    model_id: str       # unique identifier, matches trigger_bus subscriber ids
    handler_type: str   # "correlation" | "momentum" | "sentiment" | "attention"

    @abc.abstractmethod
    def on_trigger(self, signal: TriggerSignal) -> ModelState:
        """Ingest a signal, update internal state, return a ModelState snapshot."""
        ...

    @property
    def history(self) -> StateHistory:
        return self._history  # type: ignore[attr-defined]

    def _make_state(
        self,
        signal: TriggerSignal,
        commodity_weights: Dict[str, float],
        recalibration_magnitude: float,
        reasoning: List[str],
        metadata: Optional[Dict] = None,
    ) -> ModelState:
        return ModelState(
            model_id=self.model_id,
            handler_type=self.handler_type,
            trigger_signal_id=signal.signal_id,
            trigger_type=signal.trigger_type,
            updated_at=_now_iso(),
            commodity_weights=commodity_weights,
            severity_applied=signal.severity.label(),
            recalibration_magnitude=min(1.0, max(0.0, recalibration_magnitude)),
            regime_override=_regime_from_signal(signal),
            affected_commodities=signal.affected_commodities,
            cascade_priority=signal.cascade_priority,
            reasoning=reasoning,
            metadata=metadata or {},
            is_mock=True,
        )


# ── 1. Correlation Handler ─────────────────────────────────────────────────────

class CorrelationHandler(TriggerHandler):
    """
    Statistical cross-asset correlation engine.

    Tracks a ``_regime_scalar`` in [0, 1] that measures how far the market is
    from the normal (historical) correlation baseline toward the crisis regime
    (all assets moving together).  CRITICAL events push it toward 1.0; LOW
    events let it decay back toward 0.

    Per-commodity weights represent the correlation-adjusted signal strength.
    A weight above 0.5 means the commodity is being over-weighted relative to
    the normal correlation structure; below 0.5 means under-weighted.
    """

    model_id = "correlation_engine"
    handler_type = "correlation"

    _CRISIS_TARGET   = 0.85
    _NORMAL_BASELINE = 0.35

    # Severity → regime scalar step size (additive, then clamped to [0, 1])
    _STEP = {
        SeverityTier.CRITICAL: +0.40,
        SeverityTier.HIGH:     +0.20,
        SeverityTier.MEDIUM:   +0.08,
        SeverityTier.LOW:      -0.05,
    }

    def __init__(self) -> None:
        self._regime_scalar: float = 0.0
        self._commodity_bias: Dict[str, float] = {n: 0.0 for n in MODELING_COMMODITIES}
        self._history = StateHistory(self.model_id)

    def on_trigger(self, signal: TriggerSignal) -> ModelState:
        mag = signal.magnitude_score
        sev = signal.severity

        # Update regime scalar
        step = self._STEP[sev]
        self._regime_scalar = max(0.0, min(1.0, self._regime_scalar + step))

        direction_sign = -1.0 if signal.direction == "downside_surprise" else 1.0
        affected = set(signal.affected_commodities)
        weights: Dict[str, float] = {}

        for commodity in MODELING_COMMODITIES:
            if commodity in affected:
                # Shift bias proportionally to magnitude and current regime
                delta = direction_sign * mag * (0.6 + 0.4 * self._regime_scalar)
                self._commodity_bias[commodity] = max(
                    -1.0, min(1.0, self._commodity_bias[commodity] + delta * 0.30)
                )
            else:
                # Non-affected commodities revert slowly toward zero bias
                self._commodity_bias[commodity] *= 0.95

            weights[commodity] = round(0.5 + 0.5 * self._commodity_bias[commodity], 6)

        implied_corr = (
            self._NORMAL_BASELINE * (1.0 - self._regime_scalar)
            + self._CRISIS_TARGET * self._regime_scalar
        )
        recal_mag = min(1.0, mag * self._regime_scalar + mag * 0.30)

        reasoning = [
            f"Correlation regime scalar → {self._regime_scalar:.3f}",
            f"Implied cross-asset correlation → {implied_corr:.3f}",
            f"Severity {sev.label()}: direction={signal.direction}",
            f"{len(affected)} commodities re-weighted",
        ]

        state = self._make_state(
            signal, weights, recal_mag, reasoning,
            metadata={
                "regime_scalar":      round(self._regime_scalar, 4),
                "implied_correlation": round(implied_corr, 4),
            },
        )
        self._history.push(state)
        return state


# ── 2. Momentum Handler ────────────────────────────────────────────────────────

class MomentumHandler(TriggerHandler):
    """
    ML momentum model (XGBoost / Random Forest).

    Maintains a per-commodity ``_dampener`` in [0, 1] that suppresses the raw
    momentum signal.  High-severity events increase dampening (trend reversals
    become likely); LOW events let the dampener decay back toward zero so
    momentum runs freely again.

    A ``_direction_bias`` in [-1, 1] tilts the remaining momentum signal toward
    the shock direction — negative on downside surprises, positive on upside.
    """

    model_id = "momentum_model"
    handler_type = "momentum"

    # Severity → maximum dampener fraction applied per trigger
    _MAX_DAMP = {
        SeverityTier.CRITICAL: 0.85,
        SeverityTier.HIGH:     0.55,
        SeverityTier.MEDIUM:   0.25,
        SeverityTier.LOW:      0.05,
    }

    def __init__(self) -> None:
        self._dampener:       Dict[str, float] = {n: 0.0 for n in MODELING_COMMODITIES}
        self._direction_bias: Dict[str, float] = {n: 0.0 for n in MODELING_COMMODITIES}
        self._history = StateHistory(self.model_id)

    def on_trigger(self, signal: TriggerSignal) -> ModelState:
        mag = signal.magnitude_score
        sev = signal.severity
        applied_damp = self._MAX_DAMP[sev] * mag
        dir_sign = -1.0 if signal.direction == "downside_surprise" else 1.0

        affected = set(signal.affected_commodities)
        weights: Dict[str, float] = {}

        for commodity in MODELING_COMMODITIES:
            if commodity in affected:
                self._dampener[commodity] = min(
                    1.0, self._dampener[commodity] + applied_damp * 0.5
                )
                self._direction_bias[commodity] = max(
                    -1.0, min(1.0,
                        self._direction_bias[commodity] * 0.7 + dir_sign * mag * 0.3
                    )
                )
            else:
                self._dampener[commodity]       = max(0.0, self._dampener[commodity] - 0.02)
                self._direction_bias[commodity] *= 0.90

            momentum_scale = 1.0 - self._dampener[commodity]
            weights[commodity] = round(
                max(0.0, momentum_scale * (1.0 + 0.2 * self._direction_bias[commodity])), 6
            )

        mean_damp = sum(self._dampener[c] for c in affected) / max(len(affected), 1)
        recal_mag  = min(1.0, applied_damp + mean_damp * 0.3)

        reasoning = [
            f"Momentum dampener applied: {applied_damp:.3f} (sev={sev.label()}, mag={mag:.3f})",
            f"Direction bias: {dir_sign:+.0f} ({signal.direction})",
            f"Mean dampener across affected commodities: {mean_damp:.3f}",
            f"Suppression active on {len(affected)} commodities",
        ]

        state = self._make_state(
            signal, weights, recal_mag, reasoning,
            metadata={
                "applied_dampener":          round(applied_damp, 4),
                "max_dampener_for_severity": self._MAX_DAMP[sev],
                "mean_affected_dampener":    round(mean_damp, 4),
            },
        )
        self._history.push(state)
        return state


# ── 3. Sentiment Handler ───────────────────────────────────────────────────────

# Triggers where headline sentiment is more informative than hard data
_NEWS_DRIVEN = frozenset({
    "FED_CHAIR_SPEECH", "GEOPOLITICAL_SHOCK", "WEATHER_SHOCK",
    "ENERGY_TRANSITION_SIGNAL", "OPEC_PRODUCTION_DECISION",
})
# Triggers where the published number supersedes narrative
_DATA_DRIVEN = frozenset({
    "CPI_RELEASE", "NONFARM_PAYROLLS", "PPI_RELEASE",
    "EIA_CRUDE_INVENTORY", "EIA_GAS_STORAGE", "USDA_WASDE_REPORT",
    "RECESSION_FLAG", "FOMC_RATE_DECISION",
})


class SentimentHandler(TriggerHandler):
    """
    Sentiment overlay (FinBERT + Alpha Vantage ensemble).

    Controls how much weight the sentiment signal contributes to the ensemble
    forecast.  News-driven macro events raise sentiment influence; hard data
    releases lower it (the published number should dominate).

    A per-commodity ``_sentiment_score`` in [-1, 1] is also maintained —
    negative for downside surprises, positive for upside — and modulates the
    per-commodity weight output.
    """

    model_id = "sentiment_overlay"
    handler_type = "sentiment"

    BASE_WEIGHT = 0.20
    MAX_WEIGHT  = 0.60
    MIN_WEIGHT  = 0.05

    def __init__(self) -> None:
        self._ensemble_weight: float = self.BASE_WEIGHT
        self._sentiment_scores: Dict[str, float] = {n: 0.0 for n in MODELING_COMMODITIES}
        self._history = StateHistory(self.model_id)

    def on_trigger(self, signal: TriggerSignal) -> ModelState:
        mag  = signal.magnitude_score
        sev  = signal.severity
        ttype = signal.trigger_type

        # Adjust ensemble weight based on trigger category
        if ttype in _NEWS_DRIVEN:
            boost = mag * 0.30 * (sev.value / 4.0)
            self._ensemble_weight = min(self.MAX_WEIGHT, self._ensemble_weight + boost)
            category = "news_driven"
        elif ttype in _DATA_DRIVEN:
            reduction = mag * 0.10 * (sev.value / 4.0)
            self._ensemble_weight = max(self.MIN_WEIGHT, self._ensemble_weight - reduction)
            category = "data_driven"
        else:
            category = "structural"

        dir_sign = -1.0 if signal.direction == "downside_surprise" else 1.0
        affected  = set(signal.affected_commodities)
        weights: Dict[str, float] = {}

        for commodity in MODELING_COMMODITIES:
            if commodity in affected:
                delta = dir_sign * mag * 0.40 * (sev.value / 4.0)
                self._sentiment_scores[commodity] = max(
                    -1.0, min(1.0, self._sentiment_scores[commodity] * 0.6 + delta)
                )
            else:
                self._sentiment_scores[commodity] *= 0.85

            score_factor = 1.0 + 0.3 * self._sentiment_scores[commodity]
            weights[commodity] = round(max(0.0, self._ensemble_weight * score_factor), 6)

        weight_range = self.MAX_WEIGHT - self.MIN_WEIGHT
        recal_mag = abs(self._ensemble_weight - self.BASE_WEIGHT) / weight_range

        reasoning = [
            f"Sentiment ensemble weight → {self._ensemble_weight:.3f} ({category})",
            f"Trigger: {ttype} | severity: {sev.label()} | direction: {signal.direction}",
            f"Sentiment score shift applied to {len(affected)} commodities",
        ]

        state = self._make_state(
            signal, weights, min(1.0, recal_mag), reasoning,
            metadata={
                "ensemble_weight": round(self._ensemble_weight, 4),
                "trigger_category": category,
            },
        )
        self._history.push(state)
        return state


# ── 4. Attention Handler ───────────────────────────────────────────────────────

class AttentionHandler(TriggerHandler):
    """
    Deep-learning attention model (LSTM / Transformer).

    Simulates redistribution of attention-head budget across commodity "tokens".
    A fraction of the total attention budget is redirected from unaffected
    commodities to the shocked set; the result is softmax-normalised so that
    all weights always sum to 1.0.

    Attention entropy (Shannon H / log N) tracks concentration: 1.0 = uniform,
    0.0 = all attention on a single commodity.  Recalibration magnitude is
    1 − normalised_entropy so that more concentrated states score higher.
    """

    model_id = "attention_model"
    handler_type = "attention"

    # Severity → fraction of total attention budget redirected to shocked cluster
    _REDIRECT = {
        SeverityTier.CRITICAL: 0.65,
        SeverityTier.HIGH:     0.45,
        SeverityTier.MEDIUM:   0.25,
        SeverityTier.LOW:      0.10,
    }

    def __init__(self) -> None:
        n = len(MODELING_COMMODITIES)
        uniform = 1.0 / n
        self._attention: Dict[str, float] = {name: uniform for name in MODELING_COMMODITIES}
        self._history = StateHistory(self.model_id)

    def on_trigger(self, signal: TriggerSignal) -> ModelState:
        mag = signal.magnitude_score
        sev = signal.severity
        redirect_fraction = self._REDIRECT[sev] * mag

        affected     = set(signal.affected_commodities)
        n_affected   = max(len(affected), 1)
        n_unaffected = max(len(MODELING_COMMODITIES) - n_affected, 1)

        per_affected_gain = redirect_fraction / n_affected
        per_unaffected_loss = redirect_fraction / n_unaffected

        new_attn: Dict[str, float] = {}
        for commodity in MODELING_COMMODITIES:
            if commodity in affected:
                new_attn[commodity] = self._attention[commodity] + per_affected_gain
            else:
                new_attn[commodity] = max(1e-9, self._attention[commodity] - per_unaffected_loss)

        # Softmax-normalize so weights sum to exactly 1.0
        total = sum(new_attn.values())
        self._attention = {k: v / total for k, v in new_attn.items()}

        weights = {k: round(v, 8) for k, v in self._attention.items()}

        # Shannon entropy (normalized to [0, 1])
        entropy = -sum(v * math.log(v + 1e-12) for v in self._attention.values())
        max_entropy = math.log(len(MODELING_COMMODITIES))
        normalized_entropy = entropy / max_entropy
        recal_mag = 1.0 - normalized_entropy

        reasoning = [
            f"Attention redirected: {redirect_fraction:.3f} → {n_affected} affected commodities",
            f"Per-affected gain: {per_affected_gain:.5f}, per-unaffected loss: {per_unaffected_loss:.5f}",
            f"Attention entropy: {normalized_entropy:.3f} (0=concentrated, 1=uniform)",
            f"Severity {sev.label()} with magnitude {mag:.3f}",
        ]

        state = self._make_state(
            signal, weights, min(1.0, max(0.0, recal_mag)), reasoning,
            metadata={
                "redirect_fraction":    round(redirect_fraction, 4),
                "attention_entropy":    round(normalized_entropy, 4),
                "n_affected":           n_affected,
            },
        )
        self._history.push(state)
        return state


# ── Cascade Dispatcher ─────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """Summary of one signal dispatched to all matching handlers."""
    signal_id:           str
    trigger_type:        str
    triggered_at:        str
    model_states:        Dict[str, ModelState]   # model_id → ModelState
    n_handlers_invoked:  int
    errors:              Dict[str, str] = field(default_factory=dict)


def _build_default_registry() -> Dict[str, "TriggerHandler"]:
    correlation = CorrelationHandler()
    momentum    = MomentumHandler()
    sentiment   = SentimentHandler()
    attention   = AttentionHandler()

    return {
        # model_target strings from trigger_registry.json
        "macro_router":    correlation,
        "xgboost":         momentum,
        "random_forest":   momentum,
        "lstm":            attention,
        "quantum":         attention,
        "meta_predictor":  sentiment,
        # named aliases (direct lookup)
        "correlation_engine": correlation,
        "momentum_model":     momentum,
        "sentiment_overlay":  sentiment,
        "attention_model":    attention,
    }


class CascadeDispatcher:
    """
    Routes ``TriggerSignal`` objects to registered ``TriggerHandler`` instances.

    A signal's ``model_targets`` tuple drives routing.  The special target
    ``"all"`` dispatches to every unique handler.  Alias targets (e.g.
    ``"xgboost"`` and ``"random_forest"`` both map to the same
    ``MomentumHandler``) are deduplicated so each underlying handler fires
    exactly once per signal.

    ``dispatch_bus_payload`` provides the ``Handler`` signature expected by
    ``TriggerBus.subscribe()`` so the dispatcher can be wired directly into the
    bus without an adapter.
    """

    def __init__(self, handlers: Optional[Dict[str, TriggerHandler]] = None) -> None:
        self._handlers: Dict[str, TriggerHandler] = handlers or _build_default_registry()
        # Deduplicated view: model_id → handler (prevents double-firing aliases)
        self._unique: Dict[str, TriggerHandler] = {
            h.model_id: h for h in self._handlers.values()
        }

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, target: str, handler: TriggerHandler) -> None:
        """Add or replace a target→handler mapping."""
        self._handlers[target] = handler
        self._unique[handler.model_id] = handler

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(self, signal: TriggerSignal) -> DispatchResult:
        """Fire all handlers matching model_targets; collect ModelState outputs."""
        targets     = set(signal.model_targets)
        send_to_all = "all" in targets

        to_invoke: Dict[str, TriggerHandler] = {}
        if send_to_all:
            to_invoke = dict(self._unique)
        else:
            for target in targets:
                handler = self._handlers.get(target)
                if handler is not None:
                    to_invoke[handler.model_id] = handler

        model_states: Dict[str, ModelState] = {}
        errors: Dict[str, str] = {}

        for model_id, handler in to_invoke.items():
            try:
                model_states[model_id] = handler.on_trigger(signal)
            except Exception as exc:  # noqa: BLE001
                errors[model_id] = f"{type(exc).__name__}: {exc}"

        return DispatchResult(
            signal_id=signal.signal_id,
            trigger_type=signal.trigger_type,
            triggered_at=_now_iso(),
            model_states=model_states,
            n_handlers_invoked=len(to_invoke),
            errors=errors,
        )

    def dispatch_bus_payload(self, payload: dict, channel: str, model_id: str) -> None:
        """TriggerBus.Handler-compatible callback — reconstruct signal and dispatch."""
        from services.trigger_bus import reconstruct_signal  # local import avoids circular
        signal = reconstruct_signal(payload)
        if signal is not None:
            self.dispatch(signal)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def handlers(self) -> Dict[str, TriggerHandler]:
        return dict(self._unique)

    def history(self, model_id: str) -> Optional[StateHistory]:
        h = self._unique.get(model_id)
        return h.history if h is not None else None


# ── Module-level singleton ─────────────────────────────────────────────────────

DISPATCHER = CascadeDispatcher()


def dispatch_signal(signal: TriggerSignal) -> DispatchResult:
    """Convenience wrapper — dispatches via the module-level DISPATCHER."""
    return DISPATCHER.dispatch(signal)
