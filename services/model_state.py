"""
Model State
===========
Output contract for TriggerHandler implementations.

``ModelState`` is the single, structured artefact that every cascade handler
emits after processing a ``TriggerSignal``.  It captures *what changed* in the
model's internal weights/calibration, *why* (audit trail), and *when*, so the
orchestrator layer can sequence downstream re-forecasts deterministically.

``StateHistory`` is a per-handler rolling buffer of ``ModelState`` objects
used for introspection, dashboards, and replay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelState:
    """
    Snapshot of one model handler's internal state after processing a trigger.

    All numeric weights are expressed relative to the handler's own scale —
    callers should not compare weights across different handler types.
    ``is_mock=True`` on every state until real model objects are wired in.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_id: str                    # e.g. "correlation_engine"
    handler_type: str                # "correlation" | "momentum" | "sentiment" | "attention"
    trigger_signal_id: str           # links back to TriggerSignal.signal_id
    trigger_type: str                # e.g. "FOMC_RATE_DECISION"
    updated_at: str                  # ISO 8601 UTC

    # ── Recalibrated weights ──────────────────────────────────────────────────
    commodity_weights: Dict[str, float]   # commodity_name → adjusted weight

    # ── Severity response ─────────────────────────────────────────────────────
    severity_applied: str                 # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recalibration_magnitude: float        # [0, 1] — how much state shifted this cycle

    # ── Regime ───────────────────────────────────────────────────────────────
    regime_override: Optional[str]        # None | "risk_off" | "risk_on" | "defensive" | "crisis"

    # ── Scope ─────────────────────────────────────────────────────────────────
    affected_commodities: Tuple[str, ...]
    cascade_priority: int

    # ── Audit ─────────────────────────────────────────────────────────────────
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    is_mock: bool = True


class StateHistory:
    """Rolling buffer of ModelState snapshots for a single handler."""

    def __init__(self, model_id: str, maxlen: int = 100) -> None:
        self.model_id = model_id
        self._states: deque[ModelState] = deque(maxlen=maxlen)

    def push(self, state: ModelState) -> None:
        self._states.appendleft(state)

    def latest(self) -> Optional[ModelState]:
        return self._states[0] if self._states else None

    def all(self) -> List[ModelState]:
        return list(self._states)

    def for_trigger(self, trigger_type: str) -> List[ModelState]:
        return [s for s in self._states if s.trigger_type == trigger_type]

    def __len__(self) -> int:
        return len(self._states)
