"""
Research extension — one-call hook for proprietary causal-research signals.

Why this module exists
──────────────────────
The long-term vision for this dashboard is an ecosystem where bespoke,
patentable research signals (quantified headlines, policy z-scores, supply-
chain disruption indices, etc.) ripple through every downstream model the
same way the built-in `opec_action` and `fed_tightening` triggers do.

Rather than asking research code to touch four different files
(`models/triggers.py`, `features/trigger_detectors.py`, the router, the
dashboard), this module exposes a single function:

    register_research_signal(family, detector)

It does two things atomically:
  1. Registers the TriggerFamily so the router knows which commodities it
     affects.
  2. Appends the detector callable to `features.trigger_detectors.DETECTORS`
     so `detect_all()` will run it on every dashboard refresh.

End-to-end example
──────────────────
    from models.triggers import TriggerFamily
    from models.router import SignalRouter
    from models.research_extension import register_research_signal

    # 1. Define the family — which commodities does this signal touch?
    headline_family = TriggerFamily(
        name="opec_headline_sentiment",
        description="Proprietary OPEC+ headline z-score (research)",
        affected_commodities=("WTI Crude Oil", "Brent Crude Oil"),
        source="research",
    )

    # 2. Write the detector — it gets the macro DataFrame, returns event-or-None
    def detect_opec_headlines(macro_df):
        if "opec_headline_z" not in macro_df.columns:
            return None
        z = float(macro_df["opec_headline_z"].iloc[-1])
        if abs(z) < 1.5:
            return None
        return SignalRouter.make_event(
            family="opec_headline_sentiment",
            strength=min(1.0, abs(z) / 3.0),
            rationale=f"Headline z={z:+.2f}",
            metadata={"z": z},
        )

    # 3. One call wires it into the live dashboard
    register_research_signal(headline_family, detect_opec_headlines)

After that single call, the next dashboard refresh will:
  • Run `detect_opec_headlines` alongside the built-in detectors
  • Boost confidence on WTI / Brent model signals when it fires
  • Persist the event to the SQLite trigger log for backtests
  • Display it in the live triggers panel — no UI code change required

Safety
──────
- Re-registering the same family name overwrites silently (idempotent —
  safe for Streamlit reruns).
- A buggy detector cannot crash the dashboard: `detect_all()` swallows
  exceptions per-detector.
"""

from typing import Callable, Optional

import pandas as pd

from models.triggers import TriggerFamily, TriggerEvent, register_trigger_family
import features.trigger_detectors as _td


DetectorFn = Callable[[pd.DataFrame], Optional[TriggerEvent]]


def register_research_signal(
    family: TriggerFamily,
    detector: DetectorFn,
) -> None:
    """
    Register a proprietary research signal end-to-end.

    Parameters
    ----------
    family : TriggerFamily
        Defines the canonical name, description, and affected commodities.
        Convention: `source="research"` for proprietary signals.
    detector : Callable[[pd.DataFrame], Optional[TriggerEvent]]
        A pure function that inspects the macro overlay DataFrame and returns
        a `TriggerEvent` when the signal fires, or `None` otherwise. The
        event's `family` field must equal `family.name`.

    Notes
    -----
    Idempotent — calling twice with the same `family.name` and the same
    detector instance will not stack duplicate detectors. Calling twice with
    a *different* detector for the same family replaces the prior detector.
    """
    if family is None or detector is None:
        raise ValueError("family and detector are both required")
    if not callable(detector):
        raise TypeError("detector must be callable")

    # 1. Register / refresh the family definition.
    register_trigger_family(family)

    # 2. Replace any existing detector that targets the same family,
    #    otherwise append.
    target_name = family.name
    new_registry = []
    replaced = False
    for fn in _td.DETECTORS:
        fn_target = getattr(fn, "_research_family", None)
        if fn_target == target_name:
            new_registry.append(detector)
            replaced = True
        else:
            new_registry.append(fn)
    if not replaced:
        new_registry.append(detector)

    # Tag the detector so future re-registrations can find & replace it.
    try:
        detector._research_family = target_name  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass  # built-in / C-level callable; tagging is best-effort

    _td.DETECTORS[:] = new_registry


def unregister_research_signal(family_name: str) -> bool:
    """
    Remove a research detector by family name. Useful in tests and when
    swapping out a research module live.

    Returns True if a detector was removed, False otherwise. Note: this does
    not remove the TriggerFamily from the registry (families are cheap and
    keeping them avoids breaking historical log rows).
    """
    before = len(_td.DETECTORS)
    _td.DETECTORS[:] = [
        fn for fn in _td.DETECTORS
        if getattr(fn, "_research_family", None) != family_name
    ]
    return len(_td.DETECTORS) < before
