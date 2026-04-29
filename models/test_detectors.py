"""
Tests for features/trigger_detectors.py.
Uses synthetic macro DataFrames so no network calls are required.
Run: python models/test_detectors.py
"""

import sys
import traceback

import numpy as np
import pandas as pd

from features.trigger_detectors import (
    detect_opec_action,
    detect_fed_tightening,
    detect_all,
    OPEC_STRENGTH_SCALE_DAYS,
    DXY_Z_THRESHOLD,
)

PASS = 0
FAIL = 0


def _ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg, _err=None):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {msg}")
    traceback.print_exc()


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-04-01", periods=n, freq="B")


# ── OPEC detector ─────────────────────────────────────────────────────────────

def test_opec_inside_window_pre_meeting():
    print("1. OPEC — inside window, pre-meeting")
    try:
        df = pd.DataFrame({
            "days_to_opec":   [-7, -6, -5, -4, -3],
            "is_opec_window": [True] * 5,
        }, index=_idx(5))
        ev = detect_opec_action(df)
        assert ev is not None, "should fire"
        assert ev.family == "opec_action"
        expected = 1.0 - 3 / OPEC_STRENGTH_SCALE_DAYS  # last row days=-3
        assert abs(ev.strength - expected) < 1e-9
        assert "before meeting" in ev.rationale
        _ok(f"strength={ev.strength:.4f} (expected {expected:.4f})")
    except Exception as e:
        _fail("opec pre-meeting", e)


def test_opec_meeting_day():
    print("2. OPEC — meeting day, strength 1.0")
    try:
        df = pd.DataFrame({
            "days_to_opec":   [-2, -1, 0],
            "is_opec_window": [True, True, True],
        }, index=_idx(3))
        ev = detect_opec_action(df)
        assert ev is not None
        assert abs(ev.strength - 1.0) < 1e-9
        assert "meeting day" in ev.rationale
        _ok("strength=1.0 on meeting day")
    except Exception as e:
        _fail("opec meeting day", e)


def test_opec_outside_window_returns_none():
    print("3. OPEC — outside window, no fire")
    try:
        df = pd.DataFrame({
            "days_to_opec":   [-50, -49, -48],
            "is_opec_window": [False, False, False],
        }, index=_idx(3))
        ev = detect_opec_action(df)
        assert ev is None
        _ok("returns None when is_opec_window=False")
    except Exception as e:
        _fail("opec outside window", e)


def test_opec_missing_columns_safe():
    print("4. OPEC — missing columns, safe None")
    try:
        df = pd.DataFrame({"unrelated": [1, 2, 3]}, index=_idx(3))
        ev = detect_opec_action(df)
        assert ev is None
        _ok("returns None when columns missing")
    except Exception as e:
        _fail("opec missing columns", e)


# ── Fed tightening detector ───────────────────────────────────────────────────

def test_fed_fires_on_3_consecutive_above():
    print("5. Fed tightening — 3 consecutive days above threshold")
    try:
        df = pd.DataFrame({
            "dxy_zscore63": [1.5, 1.8, 2.1, 2.4, 2.5],
        }, index=_idx(5))
        ev = detect_fed_tightening(df)
        assert ev is not None
        assert ev.family == "fed_tightening"
        # last z=2.5, strength = (2.5 - 2.0) / 2.0 = 0.25
        assert abs(ev.strength - 0.25) < 1e-9
        assert ev.metadata["dxy_zscore_current"] == 2.5
        _ok(f"strength=0.25 at z=2.5; rationale: {ev.rationale}")
    except Exception as e:
        _fail("fed fires", e)


def test_fed_does_not_fire_on_break():
    print("6. Fed tightening — non-consecutive breaks should NOT fire")
    try:
        df = pd.DataFrame({
            "dxy_zscore63": [2.5, 2.5, 1.5, 2.5, 2.5],  # break in middle
        }, index=_idx(5))
        # last 3: [1.5, 2.5, 2.5] — not all above 2.0
        ev = detect_fed_tightening(df)
        assert ev is None
        _ok("returns None when last 3 days include a break")
    except Exception as e:
        _fail("fed non-consecutive", e)


def test_fed_caps_at_strength_one():
    print("7. Fed tightening — strength caps at 1.0")
    try:
        df = pd.DataFrame({"dxy_zscore63": [4.5, 4.6, 4.7]}, index=_idx(3))
        ev = detect_fed_tightening(df)
        assert ev is not None
        assert ev.strength == 1.0  # (4.7 - 2.0) / 2.0 = 1.35 → capped
        _ok("z=4.7 → strength capped at 1.0")
    except Exception as e:
        _fail("fed cap", e)


def test_fed_below_threshold():
    print("8. Fed tightening — below threshold, no fire")
    try:
        df = pd.DataFrame({"dxy_zscore63": [1.0, 1.5, 1.9]}, index=_idx(3))
        ev = detect_fed_tightening(df)
        assert ev is None
        _ok("returns None when z below threshold")
    except Exception as e:
        _fail("fed below threshold", e)


def test_fed_missing_column_safe():
    print("9. Fed tightening — missing column, safe None")
    try:
        df = pd.DataFrame({"vix": [15, 16, 17]}, index=_idx(3))
        ev = detect_fed_tightening(df)
        assert ev is None
        _ok("returns None when dxy_zscore63 missing")
    except Exception as e:
        _fail("fed missing col", e)


# ── detect_all() integration ──────────────────────────────────────────────────

def test_detect_all_both_fire():
    print("10. detect_all — both detectors fire")
    try:
        df = pd.DataFrame({
            "days_to_opec":   [-7] * 5,
            "is_opec_window": [True] * 5,
            "dxy_zscore63":   [2.5] * 5,
        }, index=_idx(5))
        events = detect_all(df)
        families = {e.family for e in events}
        assert families == {"opec_action", "fed_tightening"}
        _ok(f"both detectors fired: {sorted(families)}")
    except Exception as e:
        _fail("detect_all both", e)


def test_detect_all_only_one_fires():
    print("11. detect_all — only OPEC fires when DXY column missing")
    try:
        df = pd.DataFrame({
            "days_to_opec":   [-3] * 3,
            "is_opec_window": [True] * 3,
        }, index=_idx(3))
        events = detect_all(df)
        assert len(events) == 1
        assert events[0].family == "opec_action"
        _ok("only opec_action fires (resilient to missing data)")
    except Exception as e:
        _fail("detect_all partial", e)


def test_detect_all_empty_df():
    print("12. detect_all — empty df returns []")
    try:
        events = detect_all(pd.DataFrame())
        assert events == []
        events = detect_all(None)
        assert events == []
        _ok("empty / None df → no events")
    except Exception as e:
        _fail("detect_all empty", e)


def test_detect_all_no_columns():
    print("13. detect_all — irrelevant df returns []")
    try:
        df = pd.DataFrame({"foo": [1, 2, 3]}, index=_idx(3))
        events = detect_all(df)
        assert events == []
        _ok("irrelevant columns → no events")
    except Exception as e:
        _fail("detect_all irrelevant", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TRIGGER DETECTORS — TEST SUITE")
    print("=" * 60)
    print()

    for fn in [
        test_opec_inside_window_pre_meeting,
        test_opec_meeting_day,
        test_opec_outside_window_returns_none,
        test_opec_missing_columns_safe,
        test_fed_fires_on_3_consecutive_above,
        test_fed_does_not_fire_on_break,
        test_fed_caps_at_strength_one,
        test_fed_below_threshold,
        test_fed_missing_column_safe,
        test_detect_all_both_fire,
        test_detect_all_only_one_fires,
        test_detect_all_empty_df,
        test_detect_all_no_columns,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print("Detector layer ready. Chunk 2C will surface live triggers in the dashboard.")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
