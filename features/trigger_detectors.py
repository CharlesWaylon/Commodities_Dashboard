"""
Trigger detectors — read macro overlay features and emit TriggerEvents.

These functions sit between the macro feature pipeline (features/macro_overlays.py)
and the SignalRouter. Each detector inspects the most recent row of the macro
DataFrame and returns either a TriggerEvent (with strength in [0, 1]) or None.

Add a new detector
──────────────────
1. Write a function `detect_<name>(macro_df) -> Optional[TriggerEvent]`
2. Append it to `DETECTORS` below
3. Make sure the corresponding TriggerFamily is registered in models/triggers.py

For proprietary research detectors (quantified headlines, policy z-scores, etc.):
    - Register a new family via `models.triggers.register_trigger_family(...)`
    - Add the detector function to `DETECTORS` (or a separate registry that
      research code imports and extends)

The dashboard calls `detect_all()` once per refresh; failures in one detector
do not break the others.
"""

from typing import List, Optional, Callable

import pandas as pd

from models.triggers import TriggerEvent
from models.router import SignalRouter


# ── OPEC+ meeting proximity ───────────────────────────────────────────────────

OPEC_STRENGTH_SCALE_DAYS = 14  # outside ±14 days → strength 0


def detect_opec_action(macro_df: pd.DataFrame) -> Optional[TriggerEvent]:
    """
    Fire when the latest macro row sits inside the OPEC+ meeting window.
    Strength scales linearly with proximity: meeting day = 1.0, edge = ~0.

    Requires columns: `days_to_opec`, `is_opec_window`.
    """
    if "days_to_opec" not in macro_df.columns or "is_opec_window" not in macro_df.columns:
        return None

    last = macro_df.iloc[-1]
    if not bool(last.get("is_opec_window", False)):
        return None

    days = last["days_to_opec"]
    if pd.isna(days):
        return None
    days = float(days)

    strength = max(0.0, 1.0 - abs(days) / OPEC_STRENGTH_SCALE_DAYS)
    if days < 0:
        timing = f"{int(abs(days))} days before meeting"
    elif days > 0:
        timing = f"{int(days)} days after meeting"
    else:
        timing = "meeting day"

    return SignalRouter.make_event(
        family="opec_action",
        strength=strength,
        rationale=f"OPEC+ window — {timing}",
        metadata={"days_to_opec": days},
    )


# ── Fed tightening / DXY shock ────────────────────────────────────────────────

DXY_Z_THRESHOLD = 2.0           # z-score above this counts as "elevated"
DXY_Z_CONSECUTIVE_DAYS = 3      # need this many in a row to fire


def detect_fed_tightening(
    macro_df: pd.DataFrame,
    threshold: float = DXY_Z_THRESHOLD,
    n_consecutive: int = DXY_Z_CONSECUTIVE_DAYS,
) -> Optional[TriggerEvent]:
    """
    Fire when DXY z-score exceeds `threshold` for `n_consecutive` consecutive days.
    Strength = clipped (current_z - threshold) / threshold; at z=2*threshold = 1.0.

    Requires column: `dxy_zscore63`.
    """
    if "dxy_zscore63" not in macro_df.columns:
        return None

    z = macro_df["dxy_zscore63"].dropna()
    if len(z) < n_consecutive:
        return None

    last_n = z.iloc[-n_consecutive:]
    if not (last_n > threshold).all():
        return None

    current = float(z.iloc[-1])
    strength = max(0.0, min(1.0, (current - threshold) / threshold))

    return SignalRouter.make_event(
        family="fed_tightening",
        strength=strength,
        rationale=(
            f"DXY z-score {current:.2f} > {threshold} for "
            f"{n_consecutive}+ consecutive days"
        ),
        metadata={
            "dxy_zscore_current": current,
            "threshold": threshold,
            "consecutive_days": n_consecutive,
        },
    )


# ── Stubs for future detectors ────────────────────────────────────────────────
# Wire these up in chunk 2D / Phase 2.5 when the underlying features land.
#
# def detect_weather_shock(macro_df, climate_df) -> Optional[TriggerEvent]:
#     """PDSI / HDD anomaly threshold — needs features/climate_weather.py extensions."""
#     ...
#
# def detect_energy_transition(macro_df, et_df) -> Optional[TriggerEvent]:
#     """Battery PC1 changepoint — needs features/energy_transition.py PC1 column."""
#     ...


# ── Detector registry ─────────────────────────────────────────────────────────

DETECTORS: List[Callable[[pd.DataFrame], Optional[TriggerEvent]]] = [
    detect_opec_action,
    detect_fed_tightening,
    # detect_weather_shock,        # see stub above
    # detect_energy_transition,    # see stub above
]


def detect_all(macro_df: pd.DataFrame) -> List[TriggerEvent]:
    """
    Run every registered detector and return the events that fired.
    A failing detector is logged silently and skipped — one broken detector
    must never take down the live dashboard.
    """
    if macro_df is None or macro_df.empty:
        return []

    events: List[TriggerEvent] = []
    for fn in DETECTORS:
        try:
            ev = fn(macro_df)
            if ev is not None:
                events.append(ev)
        except Exception:
            # Silent skip — dashboard resilience over detector correctness
            continue
    return events
