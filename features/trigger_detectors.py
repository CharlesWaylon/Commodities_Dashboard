"""
Trigger detectors — read macro overlay features and emit TriggerEvents.

These functions sit between the macro feature pipeline (features/macro_overlays.py)
and the SignalRouter. Each detector inspects the most recent row of the macro
DataFrame and returns either a TriggerEvent (with strength in [0, 1]) or None.

The macro DataFrame passed to `detect_all()` should be pre-enriched with any
extra feature columns that individual detectors require.  The dashboard does
this by merging ENSO (features/climate_weather.fetch_mei) and energy-transition
(features/energy_transition.build_energy_transition_features) DataFrames into
_macro_for_triggers before calling detect_all().

Add a new detector
──────────────────
1. Write a function `detect_<name>(macro_df) -> Optional[TriggerEvent]`
2. Append it to `DETECTORS` below
3. Make sure the corresponding TriggerFamily is registered in models/triggers.py

Graceful degradation
────────────────────
Every detector must return None (not raise) when its required columns are
absent.  The dashboard may run without a NOAA token or yfinance connection,
and the ecosystem should degrade to fewer triggers rather than breaking.

For proprietary research detectors (quantified headlines, policy z-scores, etc.):
    - Register a new family via `models.triggers.register_trigger_family(...)`
    - Add the detector function to `DETECTORS` (or a separate registry that
      research code imports and extends)

The dashboard calls `detect_all()` once per refresh; failures in one detector
do not break the others.
"""

import math
from typing import List, Optional, Callable

import numpy as np
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

DXY_Z_THRESHOLD = 1.5           # z-score above this counts as "elevated"
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


# ── Weather shock — ENSO + HDD anomaly ────────────────────────────────────────

# Multivariate ENSO Index thresholds
ENSO_MIN_SIGNAL  = 1.0   # |MEI| below this → no signal (neutral zone)
ENSO_MAX_SIGNAL  = 3.0   # |MEI| at this level → strength = 1.0
# HDD (heating degree days) deviation threshold — secondary signal for NG/oil
HDD_DEV_THRESHOLD = 0.30  # 21-day average HDD 30% above normal triggers add-on


def detect_weather_shock(macro_df: pd.DataFrame) -> Optional[TriggerEvent]:
    """
    Fire when ENSO (MEI) enters a meaningful El Niño or La Niña phase,
    optionally amplified by an HDD heating anomaly.

    ENSO disrupts agricultural output (Corn Belt drought, Brazil soy stress,
    Australian wheat) and US natural gas demand through temperature extremes.

    Strength scaling:
        |MEI| = 1.0  →  strength = 0.00  (just crossed threshold)
        |MEI| = 2.0  →  strength = 0.50
        |MEI| = 3.0+ →  strength = 1.00  (strong ENSO event)

    If `hdd_dev_21d` is present (requires NOAA CDO token) and the 21-day HDD
    deviation exceeds 30%, strength is boosted by up to +0.20 and a temperature
    note is appended to the rationale.

    Requires columns (at least one): `mei`, `enso_index`.
    Optional columns: `enso_phase`, `hdd_dev_21d`.

    Affected commodities: Corn (CBOT), Wheat (CBOT SRW), Soybeans (CBOT).
    (Natural Gas is indirectly affected; its own HMM + trigger chain captures it.)
    """
    # Accept either column name — 'mei' is from fetch_mei(); 'enso_index' is
    # used in tests and synthetic DataFrames.
    mei_col = None
    for candidate in ("mei", "enso_index"):
        if candidate in macro_df.columns:
            mei_col = candidate
            break

    if mei_col is None:
        return None

    last = macro_df.iloc[-1]
    mei  = float(last[mei_col]) if not pd.isna(last[mei_col]) else float("nan")

    if math.isnan(mei) or abs(mei) < ENSO_MIN_SIGNAL:
        return None

    # Base strength from ENSO magnitude
    strength = min(1.0, (abs(mei) - ENSO_MIN_SIGNAL) / (ENSO_MAX_SIGNAL - ENSO_MIN_SIGNAL))

    # Phase label
    if mei > 0:
        phase_name  = "El Niño"
        phase_note  = "drier SE Asia / wetter US Gulf — watch soy & wheat"
    else:
        phase_name  = "La Niña"
        phase_note  = "wetter SE Asia, drier S. America — watch Brazil soy & corn"

    rationale = f"{phase_name} active (MEI {mei:+.2f}) — {phase_note}"

    # Optional HDD boost (available only when NOAA CDO token is configured)
    hdd_boost = 0.0
    if "hdd_dev_21d" in macro_df.columns:
        hdd_dev = last.get("hdd_dev_21d", float("nan"))
        if not pd.isna(hdd_dev):
            hdd_dev = float(hdd_dev)
            if abs(hdd_dev) >= HDD_DEV_THRESHOLD:
                hdd_boost = min(0.20, abs(hdd_dev) * 0.40)
                direction = "above" if hdd_dev > 0 else "below"
                rationale += (
                    f"; HDD {abs(hdd_dev):.0%} {direction} normal "
                    f"— elevated NG/heating-oil demand signal"
                )

    final_strength = min(1.0, strength + hdd_boost)

    return SignalRouter.make_event(
        family="weather_shock",
        strength=final_strength,
        rationale=rationale,
        metadata={
            "mei": mei,
            "enso_phase": float(last.get("enso_phase", np.sign(mei))),
            "hdd_boost": hdd_boost,
        },
    )


# ── Energy transition — battery metals + ETS policy stress ────────────────────

# Battery metals PC1 (battery_demand_index is already a z-score from PCA)
BATTERY_DEMAND_THRESHOLD = 1.0   # |PC1| above this = meaningful demand shift
BATTERY_DEMAND_MAX       = 3.0   # |PC1| at this → strength = 1.0

# ETS (Emissions Trading Scheme) carbon price stress z-score
ETS_STRESS_THRESHOLD = 1.5       # ets_stress_zscore above this fires
ETS_STRESS_MAX       = 3.0

# Uranium spread z-score (proxy for nuclear-sector sentiment)
URANIUM_Z_THRESHOLD  = 1.5
URANIUM_Z_MAX        = 3.0


def detect_energy_transition(macro_df: pd.DataFrame) -> Optional[TriggerEvent]:
    """
    Fire when battery-metals demand (LIT/GLNCY/REMX PC1), carbon-price stress
    (KRBN ETS z-score), or uranium spread pressure signals a structural
    energy-transition event.

    These signals affect base metals through electrification demand:
      • High battery demand index → bullish Copper, Silver (EV + solar)
      • High ETS stress → bearish fossil-fuel complex, bullish transition metals
      • Uranium spread stress → bullish nuclear sentiment

    Strength logic (each sub-signal is scaled 0→1; composite = weighted max):
        battery : weight 0.50   (direct EV/solar demand signal)
        ets     : weight 0.30   (carbon policy shock)
        uranium : weight 0.20   (nuclear narrative)

    Falls back gracefully if only one or two sub-signals are available.

    Requires at least one of:
        `battery_demand_index`, `ets_stress_zscore`, `uranium_spread_zscore`.
    """
    last = macro_df.iloc[-1]

    def _scaled(val: float, threshold: float, max_val: float) -> float:
        """Scale a raw z-score to [0, 1] using threshold as the zero point."""
        if math.isnan(val) or abs(val) < threshold:
            return 0.0
        return min(1.0, (abs(val) - threshold) / (max_val - threshold))

    # ── Sub-signals ───────────────────────────────────────────────────────────
    battery_raw = float(last.get("battery_demand_index", float("nan")))
    ets_raw     = float(last.get("ets_stress_zscore",    float("nan")))
    uranium_raw = float(last.get("uranium_spread_zscore", float("nan")))

    battery_s  = _scaled(battery_raw,  BATTERY_DEMAND_THRESHOLD, BATTERY_DEMAND_MAX)
    ets_s      = _scaled(ets_raw,      ETS_STRESS_THRESHOLD,     ETS_STRESS_MAX)
    uranium_s  = _scaled(uranium_raw,  URANIUM_Z_THRESHOLD,      URANIUM_Z_MAX)

    # Require at least one column to exist
    has_any = any(
        col in macro_df.columns
        for col in ("battery_demand_index", "ets_stress_zscore", "uranium_spread_zscore")
    )
    if not has_any:
        return None

    # Composite strength (weighted; must exceed a minimum to fire)
    composite = battery_s * 0.50 + ets_s * 0.30 + uranium_s * 0.20
    if composite < 0.05:
        return None

    # Build rationale from active sub-signals
    parts = []
    if battery_s > 0:
        direction = "expansion" if battery_raw > 0 else "contraction"
        parts.append(
            f"battery-metals demand {direction} "
            f"(PC1 {battery_raw:+.2f})"
        )
    if ets_s > 0:
        parts.append(f"ETS carbon-price stress (z={ets_raw:+.2f})")
    if uranium_s > 0:
        parts.append(f"uranium spread pressure (z={uranium_raw:+.2f})")

    if not parts:
        return None

    rationale = "Energy-transition signal: " + "; ".join(parts)

    return SignalRouter.make_event(
        family="energy_transition",
        strength=composite,
        rationale=rationale,
        metadata={
            "battery_demand_index": battery_raw,
            "ets_stress_zscore":    ets_raw,
            "uranium_spread_zscore": uranium_raw,
            "battery_sub_signal":   battery_s,
            "ets_sub_signal":       ets_s,
            "uranium_sub_signal":   uranium_s,
        },
    )


# ── Detector registry ─────────────────────────────────────────────────────────

DETECTORS: List[Callable[[pd.DataFrame], Optional[TriggerEvent]]] = [
    detect_opec_action,
    detect_fed_tightening,
    detect_weather_shock,
    detect_energy_transition,
]


def detect_all(macro_df: pd.DataFrame) -> List[TriggerEvent]:
    """
    Run every registered detector and return the events that fired.

    The macro_df should be pre-enriched with any extra feature columns
    that individual detectors require (ENSO, battery demand, ETS stress).
    See the module docstring for how the dashboard does this.

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
