"""
SignalRouter — applies macro trigger context to model signals.

When a TriggerEvent fires for a family, every ModelSignal whose commodity is
in that family's affected list gets a confidence multiplier. Multiple active
triggers compound multiplicatively, capped at confidence == 1.0.

This is the central choke point where the ecosystem wires up:

    Models emit signals  →  Router reweights via active triggers  →  Dashboard

The router does NOT detect triggers itself — that lives in features/triggers/
(chunk 2B). It only consumes TriggerEvent objects and reweights signals.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import List, Dict, Optional

from models.model_signal import ModelSignal
from models.triggers import TriggerEvent, get_trigger_family


class SignalRouter:
    """
    Reweight ModelSignals based on active TriggerEvents.

    Boost formula:
        confidence_new = min(1.0, confidence_old * Π(1 + k * strength_i))

    where k = AMPLIFICATION_COEF (default 0.5). At full strength=1.0,
    a single trigger adds +50% to confidence. Multiple triggers compound.

    Cap at 1.0 keeps confidence in the valid [0, 1] range required by ModelSignal.
    """

    AMPLIFICATION_COEF = 0.5

    def __init__(self, amplification_coef: float = AMPLIFICATION_COEF):
        self.k = amplification_coef

    def route(
        self,
        signals: List[ModelSignal],
        events: List[TriggerEvent],
    ) -> List[ModelSignal]:
        """
        Return new signals with confidence reweighted per active triggers.
        Original signals are deep-copied — never mutated.

        Each routed signal records its multiplier and trigger trail in
        metadata['routed_triggers'] for transparency in the dashboard.
        """
        if not events:
            return [deepcopy(s) for s in signals]

        out: List[ModelSignal] = []
        for sig in signals:
            new_sig = deepcopy(sig)
            applied: List[Dict] = []
            multiplier = 1.0

            for ev in events:
                if sig.commodity in ev.affected_commodities:
                    boost = 1.0 + self.k * ev.strength
                    multiplier *= boost
                    applied.append({
                        "family": ev.family,
                        "strength": round(ev.strength, 4),
                        "boost": round(boost, 4),
                        "rationale": ev.rationale,
                    })

            if applied:
                new_sig.confidence = min(1.0, new_sig.confidence * multiplier)
                new_sig.metadata = dict(new_sig.metadata)
                new_sig.metadata["routed_triggers"] = applied
                new_sig.metadata["confidence_multiplier"] = round(multiplier, 4)

            out.append(new_sig)

        return out

    @staticmethod
    def make_event(
        family: str,
        strength: float,
        rationale: str = "",
        metadata: Optional[Dict] = None,
    ) -> TriggerEvent:
        """
        Convenience constructor — looks up family commodities, clamps strength,
        attaches a UTC timestamp.
        """
        fam = get_trigger_family(family)
        if fam is None:
            raise ValueError(
                f"Unknown trigger family: {family!r}. "
                f"Register it via models.triggers.register_trigger_family()."
            )
        clamped = max(0.0, min(1.0, strength))
        return TriggerEvent(
            family=family,
            strength=clamped,
            detected_at=datetime.now(timezone.utc).isoformat(),
            affected_commodities=list(fam.affected_commodities),
            rationale=rationale,
            metadata=metadata or {},
        )
