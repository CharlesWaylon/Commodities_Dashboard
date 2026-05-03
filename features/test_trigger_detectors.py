"""
Tests for features/trigger_detectors.py — covering all four detectors.

Coverage
────────
 1.  detect_opec_action — fires inside window
 2.  detect_opec_action — silent outside window
 3.  detect_opec_action — missing columns → None
 4.  detect_fed_tightening — fires when DXY z-score elevated for 3+ days
 5.  detect_fed_tightening — silent when below threshold
 6.  detect_fed_tightening — silent when only 2 consecutive elevated days
 7.  detect_fed_tightening — missing column → None
 8.  detect_weather_shock — fires on El Niño (positive MEI)
 9.  detect_weather_shock — fires on La Niña (negative MEI)
10.  detect_weather_shock — silent in ENSO neutral zone (|MEI| < 1.0)
11.  detect_weather_shock — accepts 'enso_index' as fallback column name
12.  detect_weather_shock — HDD boost adds strength when column present
13.  detect_weather_shock — missing MEI columns → None
14.  detect_energy_transition — fires on battery demand expansion
15.  detect_energy_transition — fires on ETS carbon stress alone
16.  detect_energy_transition — fires on uranium z-score alone
17.  detect_energy_transition — composite weighting (all three signals)
18.  detect_energy_transition — sub-threshold composite → None
19.  detect_energy_transition — missing all columns → None
20.  detect_all — all four detectors return results when conditions met
21.  detect_all — empty DataFrame → empty list
22.  detect_all — individual detector failure does not break detect_all
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.trigger_detectors import (
    DETECTORS,
    detect_all,
    detect_energy_transition,
    detect_fed_tightening,
    detect_opec_action,
    detect_weather_shock,
    ENSO_MIN_SIGNAL,
    BATTERY_DEMAND_THRESHOLD,
    ETS_STRESS_THRESHOLD,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _opec_df(in_window: bool = True, days: float = 0.0, n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "is_opec_window": [in_window] * n,
        "days_to_opec":   [days] * n,
        "dxy_zscore63":   [0.5] * n,
    }, index=idx)


def _dxy_df(z_values: list) -> pd.DataFrame:
    n = len(z_values)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"dxy_zscore63": z_values}, index=idx)


def _enso_df(mei: float, n: int = 5, col: str = "mei",
             hdd_dev: float = None) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    data = {
        col:          [mei] * n,
        "enso_phase": [float(np.sign(mei))] * n,
        "dxy_zscore63": [0.5] * n,
    }
    if hdd_dev is not None:
        data["hdd_dev_21d"] = [hdd_dev] * n
    return pd.DataFrame(data, index=idx)


def _et_df(battery: float = float("nan"),
           ets: float = float("nan"),
           uranium: float = float("nan"),
           n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    data: dict = {}
    if not math.isnan(battery):
        data["battery_demand_index"]  = [battery] * n
    if not math.isnan(ets):
        data["ets_stress_zscore"]     = [ets] * n
    if not math.isnan(uranium):
        data["uranium_spread_zscore"] = [uranium] * n
    return pd.DataFrame(data, index=idx)


def _check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
        raise AssertionError(f"Test failed: {label}. {detail}")


# ── Tests ──────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    print("=" * 60)
    print("TRIGGER DETECTORS — TEST SUITE")
    print("=" * 60)

    # ── 1. detect_opec_action fires inside window ─────────────────────────────
    ev1 = detect_opec_action(_opec_df(in_window=True, days=3.0))
    _check("detect_opec_action — fires inside window",
           ev1 is not None and ev1.family == "opec_action",
           f"event={ev1}")

    # ── 2. detect_opec_action silent outside window ───────────────────────────
    ev2 = detect_opec_action(_opec_df(in_window=False, days=20.0))
    _check("detect_opec_action — silent outside window", ev2 is None, str(ev2))

    # ── 3. detect_opec_action missing columns → None ──────────────────────────
    ev3 = detect_opec_action(pd.DataFrame({"vix": [15.0]},
                              index=pd.date_range("2025-01-01", periods=1)))
    _check("detect_opec_action — missing columns → None", ev3 is None)

    # ── 4. detect_fed_tightening fires ────────────────────────────────────────
    ev4 = detect_fed_tightening(_dxy_df([2.5, 2.6, 2.7, 2.8, 3.0]))
    _check("detect_fed_tightening — fires when z elevated 3+ days",
           ev4 is not None and ev4.family == "fed_tightening",
           f"event={ev4}")
    _check("detect_fed_tightening — strength in (0, 1]",
           0 < ev4.strength <= 1.0, f"strength={ev4.strength}")

    # ── 5. detect_fed_tightening silent below threshold ───────────────────────
    ev5 = detect_fed_tightening(_dxy_df([0.5, 0.6, 0.7, 0.8, 0.9]))
    _check("detect_fed_tightening — silent below threshold", ev5 is None)

    # ── 6. detect_fed_tightening needs n_consecutive days ────────────────────
    # Last 2 days elevated, third below → should NOT fire
    ev6 = detect_fed_tightening(_dxy_df([0.5, 0.6, 1.0, 2.5, 2.6]))
    _check("detect_fed_tightening — only 2 consecutive elevated → None",
           ev6 is None, f"event={ev6}")

    # ── 7. detect_fed_tightening missing column → None ───────────────────────
    ev7 = detect_fed_tightening(pd.DataFrame({"vix": [15.0]},
                                 index=pd.date_range("2025-01-01", periods=1)))
    _check("detect_fed_tightening — missing column → None", ev7 is None)

    # ── 8. detect_weather_shock — El Niño ─────────────────────────────────────
    ev8 = detect_weather_shock(_enso_df(mei=2.0))
    _check("detect_weather_shock — El Niño fires (MEI=2.0)",
           ev8 is not None and ev8.family == "weather_shock",
           f"event={ev8}")
    _check("detect_weather_shock — El Niño strength 0.5",
           abs(ev8.strength - 0.5) < 1e-9, f"strength={ev8.strength}")
    _check("detect_weather_shock — rationale mentions El Niño",
           "El Niño" in ev8.rationale, ev8.rationale)

    # ── 9. detect_weather_shock — La Niña ─────────────────────────────────────
    ev9 = detect_weather_shock(_enso_df(mei=-2.0))
    _check("detect_weather_shock — La Niña fires (MEI=-2.0)",
           ev9 is not None, f"event={ev9}")
    _check("detect_weather_shock — La Niña rationale",
           "La Niña" in ev9.rationale, ev9.rationale)

    # ── 10. detect_weather_shock silent in neutral zone ───────────────────────
    ev10 = detect_weather_shock(_enso_df(mei=0.3))
    _check("detect_weather_shock — neutral |MEI|<1.0 → None", ev10 is None)

    # ── 11. detect_weather_shock accepts 'enso_index' column name ────────────
    ev11 = detect_weather_shock(_enso_df(mei=2.5, col="enso_index"))
    _check("detect_weather_shock — accepts 'enso_index' column",
           ev11 is not None, f"event={ev11}")

    # ── 12. HDD boost increases strength ─────────────────────────────────────
    ev12_base = detect_weather_shock(_enso_df(mei=2.0))
    ev12_hdd  = detect_weather_shock(_enso_df(mei=2.0, hdd_dev=0.40))
    _check("detect_weather_shock — HDD boost increases strength",
           ev12_hdd.strength > ev12_base.strength,
           f"base={ev12_base.strength:.3f} hdd={ev12_hdd.strength:.3f}")
    _check("detect_weather_shock — HDD rationale mention",
           "HDD" in ev12_hdd.rationale, ev12_hdd.rationale)

    # ── 13. detect_weather_shock missing MEI columns → None ──────────────────
    ev13 = detect_weather_shock(pd.DataFrame({"vix": [15.0]},
                                 index=pd.date_range("2025-01-01", periods=1)))
    _check("detect_weather_shock — no MEI columns → None", ev13 is None)

    # ── 14. detect_energy_transition fires on battery demand ─────────────────
    ev14 = detect_energy_transition(_et_df(battery=2.0))
    _check("detect_energy_transition — fires on battery demand (PC1=2.0)",
           ev14 is not None and ev14.family == "energy_transition",
           f"event={ev14}")
    _check("detect_energy_transition — battery rationale present",
           "battery" in ev14.rationale.lower(), ev14.rationale)

    # ── 15. detect_energy_transition fires on ETS stress alone ───────────────
    ev15 = detect_energy_transition(_et_df(ets=2.0))
    _check("detect_energy_transition — fires on ETS stress alone",
           ev15 is not None, f"event={ev15}")
    _check("detect_energy_transition — ETS rationale present",
           "ETS" in ev15.rationale or "carbon" in ev15.rationale.lower(),
           ev15.rationale)

    # ── 16. detect_energy_transition fires on uranium alone ──────────────────
    ev16 = detect_energy_transition(_et_df(uranium=2.0))
    _check("detect_energy_transition — fires on uranium stress alone",
           ev16 is not None, f"event={ev16}")

    # ── 17. Composite weighting — all three signals ───────────────────────────
    ev17_all   = detect_energy_transition(_et_df(battery=2.0, ets=2.0, uranium=2.0))
    ev17_batt  = detect_energy_transition(_et_df(battery=2.0))
    _check("detect_energy_transition — all-signals composite > battery-only",
           ev17_all.strength >= ev17_batt.strength,
           f"all={ev17_all.strength:.3f} battery-only={ev17_batt.strength:.3f}")

    # ── 18. Sub-threshold composite → None ───────────────────────────────────
    ev18 = detect_energy_transition(_et_df(battery=0.5, ets=0.5, uranium=0.5))
    _check("detect_energy_transition — sub-threshold → None", ev18 is None)

    # ── 19. Missing all columns → None ───────────────────────────────────────
    ev19 = detect_energy_transition(pd.DataFrame({"vix": [15.0]},
                                     index=pd.date_range("2025-01-01", periods=1)))
    _check("detect_energy_transition — no columns → None", ev19 is None)

    # ── 20. detect_all — all four detectors fire ──────────────────────────────
    full_df = pd.DataFrame({
        # OPEC
        "is_opec_window": [True],
        "days_to_opec":   [2.0],
        # DXY (3 consecutive rows needed — extend to 5)
        "dxy_zscore63":   [2.5],
        # ENSO
        "mei":            [2.0],
        "enso_phase":     [1.0],
        # Battery / ETS / Uranium
        "battery_demand_index":  [2.0],
        "ets_stress_zscore":     [2.0],
        "uranium_spread_zscore": [2.0],
    }, index=pd.date_range("2025-01-01", periods=1))
    # Need 3 consecutive rows for DXY — replicate
    full_df = pd.concat([full_df] * 5).reset_index(drop=True)
    full_df.index = pd.date_range("2025-01-01", periods=5, freq="B")

    all_events20 = detect_all(full_df)
    families20   = {ev.family for ev in all_events20}
    _check("detect_all — opec_action fires",        "opec_action"       in families20, str(families20))
    _check("detect_all — fed_tightening fires",     "fed_tightening"    in families20, str(families20))
    _check("detect_all — weather_shock fires",      "weather_shock"     in families20, str(families20))
    _check("detect_all — energy_transition fires",  "energy_transition" in families20, str(families20))

    # ── 21. detect_all — empty DataFrame → empty list ────────────────────────
    _check("detect_all — empty DataFrame → []",
           detect_all(pd.DataFrame()) == [], "non-empty result")

    # ── 22. detect_all — broken detector does not break others ───────────────
    import features.trigger_detectors as _td
    _orig = _td.DETECTORS[:]
    def _broken(_df):
        raise RuntimeError("intentional failure")
    _td.DETECTORS.insert(0, _broken)
    try:
        ev22 = detect_all(_opec_df(in_window=True, days=2.0))
        _check("detect_all — broken detector skipped silently",
               any(e.family == "opec_action" for e in ev22),
               f"events={[e.family for e in ev22]}")
    finally:
        _td.DETECTORS[:] = _orig

    print()
    print("=" * 60)
    print("ALL 22 ASSERTIONS PASSED")
    print("weather_shock + energy_transition detectors ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
