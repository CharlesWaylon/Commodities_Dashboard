"""
Tests for models/trigger_log.py and models/research_extension.py.
Uses temporary SQLite files so the live dashboard DB is never touched.
Run: python -m models.test_trigger_log
"""

import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models.router import SignalRouter
from models.triggers import TriggerFamily, TriggerEvent, get_trigger_family
from models.trigger_log import log_trigger_events, recent_trigger_events
from models.research_extension import (
    register_research_signal,
    unregister_research_signal,
)
import features.trigger_detectors as _td
from features.trigger_detectors import detect_all

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


def _tmpdb() -> Path:
    """Fresh temp DB path per test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return Path(tmp.name)


def _make_event(family: str, strength: float, when: str = None) -> TriggerEvent:
    ev = SignalRouter.make_event(
        family=family,
        strength=strength,
        rationale=f"test fire {family} @ {strength}",
        metadata={"unit_test": True},
    )
    if when is not None:
        ev.detected_at = when
    return ev


# ── trigger_log.py ────────────────────────────────────────────────────────────

def test_log_and_read_roundtrip():
    print("1. trigger_log — log + read roundtrip")
    try:
        db = _tmpdb()
        events = [
            _make_event("opec_action", 0.7),
            _make_event("fed_tightening", 0.4),
        ]
        n = log_trigger_events(events, db_path=db)
        assert n == 2, f"expected 2 rows, got {n}"

        df = recent_trigger_events(days=7, db_path=db)
        assert len(df) == 2
        assert set(df["family"]) == {"opec_action", "fed_tightening"}
        # JSON columns should parse back into containers
        assert isinstance(df["affected_commodities"].iloc[0], list)
        assert isinstance(df["metadata"].iloc[0], dict)
        _ok(f"wrote and read {len(df)} rows; JSON cols parsed")
    except Exception as e:
        _fail("roundtrip", e)


def test_log_empty_list():
    print("2. trigger_log — empty list returns 0")
    try:
        db = _tmpdb()
        n = log_trigger_events([], db_path=db)
        assert n == 0
        df = recent_trigger_events(days=30, db_path=db)
        assert df.empty
        _ok("empty input → 0 rows, empty read")
    except Exception as e:
        _fail("empty list", e)


def test_log_dedup_same_day():
    print("3. trigger_log — re-fire same family/day overwrites")
    try:
        db = _tmpdb()
        log_trigger_events([_make_event("opec_action", 0.3)], db_path=db)
        log_trigger_events([_make_event("opec_action", 0.9)], db_path=db)
        df = recent_trigger_events(days=7, db_path=db)
        assert len(df) == 1, f"expected dedup to 1 row, got {len(df)}"
        assert abs(df["strength"].iloc[0] - 0.9) < 1e-9
        _ok("dedup keyed on (family, trigger_date) — keeps latest strength")
    except Exception as e:
        _fail("dedup", e)


def test_recent_filter_by_family():
    print("4. trigger_log — family filter narrows results")
    try:
        db = _tmpdb()
        log_trigger_events(
            [_make_event("opec_action", 0.5), _make_event("fed_tightening", 0.6)],
            db_path=db,
        )
        df = recent_trigger_events(days=7, family="fed_tightening", db_path=db)
        assert len(df) == 1
        assert df["family"].iloc[0] == "fed_tightening"
        _ok("family= filter returns only matching rows")
    except Exception as e:
        _fail("family filter", e)


def test_recent_excludes_old():
    print("5. trigger_log — events outside lookback window are excluded")
    try:
        db = _tmpdb()
        # Manually craft an old detected_at (60 days ago)
        old_when = (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=60)
        ).isoformat()
        log_trigger_events([_make_event("opec_action", 0.5, when=old_when)], db_path=db)

        df_short = recent_trigger_events(days=30, db_path=db)
        assert df_short.empty, "60-day-old event should not appear in 30d window"

        df_long = recent_trigger_events(days=90, db_path=db)
        assert len(df_long) == 1
        _ok("lookback window correctly excludes ancient events")
    except Exception as e:
        _fail("recent excludes old", e)


# ── research_extension.py ─────────────────────────────────────────────────────

def test_register_research_signal_end_to_end():
    print("6. research_extension — register, fire, unregister")
    try:
        # Define a synthetic research family + detector
        fam = TriggerFamily(
            name="unit_test_research",
            description="synthetic test signal",
            affected_commodities=("WTI Crude Oil",),
            source="research",
        )

        def my_detector(df):
            if "fake_research_z" not in df.columns:
                return None
            z = float(df["fake_research_z"].iloc[-1])
            if abs(z) < 1.0:
                return None
            return SignalRouter.make_event(
                family="unit_test_research",
                strength=min(1.0, abs(z) / 3.0),
                rationale=f"fake z={z}",
                metadata={"z": z},
            )

        baseline = len(_td.DETECTORS)
        register_research_signal(fam, my_detector)

        # 1. Family is in the registry
        assert get_trigger_family("unit_test_research") is not None
        # 2. Detector got appended
        assert len(_td.DETECTORS) == baseline + 1

        # 3. detect_all picks it up when the trigger column is present
        macro = pd.DataFrame(
            {"fake_research_z": [2.5]},
            index=pd.date_range("2026-04-29", periods=1, freq="D"),
        )
        events = detect_all(macro)
        families = {e.family for e in events}
        assert "unit_test_research" in families, f"got {families}"

        # 4. Re-register with a DIFFERENT detector replaces, doesn't duplicate
        def my_detector_v2(df):
            return None
        register_research_signal(fam, my_detector_v2)
        assert len(_td.DETECTORS) == baseline + 1, "re-register should replace"

        # 5. Unregister cleans up
        removed = unregister_research_signal("unit_test_research")
        assert removed is True
        assert len(_td.DETECTORS) == baseline

        _ok("register → detect_all fires → re-register replaces → unregister removes")
    except Exception as e:
        _fail("research_extension end-to-end", e)
        # Best-effort cleanup so other tests aren't polluted
        unregister_research_signal("unit_test_research")


def test_register_research_signal_validation():
    print("7. research_extension — input validation")
    try:
        try:
            register_research_signal(None, lambda df: None)
            assert False, "should raise"
        except ValueError:
            pass

        fam = TriggerFamily(
            name="bad_test",
            description="x",
            affected_commodities=(),
            source="research",
        )
        try:
            register_research_signal(fam, "not a callable")  # type: ignore[arg-type]
            assert False, "should raise"
        except TypeError:
            pass

        _ok("rejects None / non-callable detectors")
    except Exception as e:
        _fail("validation", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("TRIGGER LOG + RESEARCH EXTENSION — TEST SUITE")
    print("=" * 60)
    print()

    for fn in [
        test_log_and_read_roundtrip,
        test_log_empty_list,
        test_log_dedup_same_day,
        test_recent_filter_by_family,
        test_recent_excludes_old,
        test_register_research_signal_end_to_end,
        test_register_research_signal_validation,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print("Phase 2 complete: detect → route → log → extend (research-ready).")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
