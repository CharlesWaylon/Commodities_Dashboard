"""
Test suite — Trigger Classification Engine

Covers:
  - Normal classification (known event types, severity tiers)
  - Edge cases: missing actual/expected values, zero deviation, extreme z-scores
  - Repeat-event detection (dedup window)
  - Direction inference (upside / downside / neutral)
  - Batch classification
  - Impact floor for high-impact events
  - Magnitude clamping at 1.0
  - Event-type resolution (exact map, keyword scan, family-name lookup)
  - Decay window (decay_expires_at is in the future)
  - SeverityTier ordering (IntEnum comparison)
  - All four severity tiers reachable

Run:
    python -m services.test_trigger_classifier
"""

import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# ── Shared counters ───────────────────────────────────────────────────────────
PASS = 0
FAIL = 0


def _ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    suffix = f"\n         detail: {detail}" if detail else ""
    print(f"  FAIL  {msg}{suffix}")


def _assert(cond: bool, msg: str, detail: str = "") -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg, detail)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


def make_event(**kwargs):
    """Build a MacroEvent with sensible defaults; kwargs override."""
    from services.macro_ingestion import MacroEvent

    defaults = dict(
        trigger_id        = _uuid(),
        source            = "FRED",
        event_type        = "CPI",
        expected_value    = 310.0,
        actual_value      = 312.5,
        release_timestamp = "2026-05-14T12:30:00+00:00",
        deviation_score   = 2.0,
        impact            = "high",
        unit              = "index",
        previous_value    = 309.0,
    )
    defaults.update(kwargs)
    return MacroEvent(**defaults)


def fresh_classifier(**kwargs):
    """Return a TriggerClassifier with an empty dedup cache."""
    from services.trigger_classifier import TriggerClassifier
    from services.trigger_config import load_trigger_registry

    reg = load_trigger_registry()
    return TriggerClassifier(reg, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cpi_normal_high_deviation():
    """CPI with deviation_score=2.0 should resolve to CPI_RELEASE with HIGH+ severity."""
    print("test_cpi_normal_high_deviation")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=2.0,
                     expected_value=310.0, actual_value=312.5)
    sig = clf.classify(ev)

    _assert(sig is not None,           "signal emitted")
    _assert(sig.trigger_type == "CPI_RELEASE", f"trigger_type={sig.trigger_type!r}")
    from services.trigger_classifier import SeverityTier
    _assert(sig.severity >= SeverityTier.HIGH, f"severity={sig.severity.label()}")
    _assert(0.0 < sig.magnitude_score <= 1.0,  f"magnitude={sig.magnitude_score}")
    _assert(sig.cascade_priority == 1,          f"priority={sig.cascade_priority}")
    _assert("macro_router" in sig.model_targets, "model_targets includes macro_router")
    _assert(not sig.is_repeat,                   "first emission not a repeat")


def test_severity_tiers_all_reachable():
    """Every tier (LOW, MEDIUM, HIGH, CRITICAL) must be achievable."""
    print("test_severity_tiers_all_reachable")
    from services.trigger_classifier import SeverityTier, TriggerClassifier
    from services.trigger_config import load_trigger_registry

    clf = TriggerClassifier(load_trigger_registry())

    # Use expected_value=None for HIGH and CRITICAL so the pure z-score path is
    # taken (blending a tiny pct_surprise would drag the score below the band).
    cases = [
        ("LOW",      "CPI", 0.0,  None,  310.0, 310.0,  "low"),
        ("MEDIUM",   "CPI", 0.5,  None,  310.0, None,   "medium"),
        ("HIGH",     "CPI", 1.5,  None,  310.0, None,   "high"),
        ("CRITICAL", "CPI", 3.0,  None,  310.0, None,   "high"),
    ]

    expected_tiers = {
        "LOW":      SeverityTier.LOW,
        "MEDIUM":   SeverityTier.MEDIUM,
        "HIGH":     SeverityTier.HIGH,
        "CRITICAL": SeverityTier.CRITICAL,
    }

    for label, etype, z, expected, actual, prev, impact in cases:
        ev = make_event(
            trigger_id=_uuid(), event_type=etype,
            deviation_score=z,
            expected_value=expected,
            actual_value=actual,
            previous_value=prev,
            impact=impact,
        )
        sig = clf.classify(ev)
        if sig is None:
            _fail(f"tier={label}: signal is None (z={z})")
            continue
        _assert(
            sig.severity == expected_tiers[label],
            f"tier={label}: got {sig.severity.label()} (mag={sig.magnitude_score:.3f}, z={z})",
        )


def test_missing_actual_value_returns_none():
    """actual_value=None means no observable signal — classifier returns None."""
    print("test_missing_actual_value_returns_none")
    clf = fresh_classifier()
    ev  = make_event(actual_value=None)
    sig = clf.classify(ev)
    _assert(sig is None, "returns None when actual_value is absent")


def test_missing_expected_falls_back_to_z_score():
    """FRED poller emits expected_value=None. Classification should use z-score method."""
    print("test_missing_expected_falls_back_to_z_score")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", expected_value=None,
                     actual_value=313.1, deviation_score=2.0)
    sig = clf.classify(ev)
    _assert(sig is not None,                  "signal emitted")
    _assert("z_score" in sig.classification_method,
            f"method={sig.classification_method!r}")
    _assert(sig.magnitude_score > 0,          f"magnitude={sig.magnitude_score}")


def test_unknown_event_type_returns_none():
    """An event_type with no registry mapping should return None (not raise)."""
    print("test_unknown_event_type_returns_none")
    clf = fresh_classifier()
    ev  = make_event(event_type="TOTALLY_UNKNOWN_EVENT_XYZZY_42")
    sig = clf.classify(ev)
    _assert(sig is None, "unknown event_type → None")


def test_repeat_event_detection():
    """Second arrival of the same (trigger_type, date) within the window → is_repeat=True."""
    print("test_repeat_event_detection")
    clf = fresh_classifier(dedup_window_hours=24)

    ev1 = make_event(event_type="CPI", deviation_score=2.0, trigger_id=_uuid())
    sig1 = clf.classify(ev1)
    _assert(sig1 is not None,   "first signal emitted")
    _assert(not sig1.is_repeat, "first emission: is_repeat=False")

    ev2 = make_event(event_type="CPI", deviation_score=1.9, trigger_id=_uuid())
    sig2 = clf.classify(ev2)
    _assert(sig2 is not None,   "repeat signal emitted (not suppressed)")
    _assert(sig2.is_repeat,     "second emission: is_repeat=True")


def test_repeat_expired_resets():
    """After the dedup window expires the same pair is no longer a repeat."""
    print("test_repeat_expired_resets")
    import time
    from services.trigger_classifier import TriggerClassifier
    from services.trigger_config import load_trigger_registry

    # 0.00001 h = 0.036 s; sleep 0.1 s to guarantee expiry
    clf = TriggerClassifier(load_trigger_registry(), dedup_window_hours=0.00001)

    ev1 = make_event(event_type="CPI", deviation_score=2.0, trigger_id=_uuid())
    clf.classify(ev1)

    time.sleep(0.10)

    ev2 = make_event(event_type="CPI", deviation_score=2.1, trigger_id=_uuid())
    sig2 = clf.classify(ev2)
    _assert(sig2 is not None,    "second signal emitted")
    _assert(not sig2.is_repeat,  "after window expiry: is_repeat=False")


def test_negative_deviation_direction():
    """Negative deviation_score → downside_surprise direction."""
    print("test_negative_deviation_direction")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=-2.5,
                     actual_value=307.0, expected_value=310.0)
    sig = clf.classify(ev)
    _assert(sig is not None,                    "signal emitted")
    _assert(sig.direction == "downside_surprise",
            f"direction={sig.direction!r}")
    _assert(sig.magnitude_score > 0,            "magnitude is positive despite negative deviation")


def test_zero_deviation_low_severity():
    """deviation_score=0.0 and matching expected → LOW severity, magnitude≈0."""
    print("test_zero_deviation_low_severity")
    from services.trigger_classifier import SeverityTier
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=0.0,
                     actual_value=310.0, expected_value=310.0, impact="low")
    sig = clf.classify(ev)
    _assert(sig is not None,                  "signal emitted even at zero deviation")
    _assert(sig.severity == SeverityTier.LOW, f"severity={sig.severity.label()}")
    _assert(sig.magnitude_score < 0.20,       f"magnitude={sig.magnitude_score}")


def test_extreme_z_score_clamped_to_one():
    """deviation_score=100 must be clamped to magnitude=1.0 (no overflow)."""
    print("test_extreme_z_score_clamped_to_one")
    from services.trigger_classifier import SeverityTier
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=100.0, expected_value=None)
    sig = clf.classify(ev)
    _assert(sig is not None,                       "signal emitted")
    _assert(sig.magnitude_score == 1.0,            f"magnitude={sig.magnitude_score} (expected 1.0)")
    _assert(sig.severity == SeverityTier.CRITICAL, f"severity={sig.severity.label()}")


def test_high_impact_floor():
    """High-impact events with near-zero deviation should receive the impact floor."""
    print("test_high_impact_floor")
    from services.trigger_classifier import IMPACT_FLOOR
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=0.001,
                     actual_value=310.001, expected_value=310.0, impact="high")
    sig = clf.classify(ev)
    _assert(sig is not None,                   "signal emitted")
    _assert(sig.magnitude_score >= IMPACT_FLOOR,
            f"magnitude={sig.magnitude_score} < floor={IMPACT_FLOOR}")
    _assert("impact_floor" in sig.classification_method,
            f"method={sig.classification_method!r}")


def test_zero_expected_value_no_exception():
    """expected_value=0.0 should not cause ZeroDivisionError; falls back to z-score."""
    print("test_zero_expected_value_no_exception")
    clf = fresh_classifier()
    try:
        ev  = make_event(event_type="CPI", expected_value=0.0,
                         actual_value=0.3, deviation_score=1.5)
        sig = clf.classify(ev)
        _assert(sig is not None, "signal emitted without error")
    except ZeroDivisionError:
        _fail("ZeroDivisionError raised for expected_value=0.0")
    except Exception as exc:
        _fail("unexpected exception", str(exc))


def test_batch_classification():
    """classify_batch drops None results and returns only valid signals."""
    print("test_batch_classification")
    from services.trigger_classifier import TriggerClassifier
    from services.trigger_config import load_trigger_registry

    clf = TriggerClassifier(load_trigger_registry())
    events = [
        make_event(event_type="CPI",    deviation_score=2.5, trigger_id=_uuid()),
        make_event(event_type="NFP",    deviation_score=1.8, trigger_id=_uuid(),
                   expected_value=200.0, actual_value=280.0),
        make_event(event_type="TOTALLY_UNKNOWN_XYZZY", trigger_id=_uuid()),
        make_event(actual_value=None,   trigger_id=_uuid()),       # no observable signal
    ]
    signals = clf.classify_batch(events)
    types   = {s.trigger_type for s in signals}
    _assert(len(signals) >= 1,             f"at least 1 signal emitted, got {len(signals)}")
    _assert("CPI_RELEASE" in types,        f"CPI_RELEASE in {types}")
    _assert("NONFARM_PAYROLLS" in types,   f"NONFARM_PAYROLLS in {types}")
    _assert(len(signals) <= 2,             f"unknown and None events dropped (got {len(signals)})")


def test_fomc_event_type_resolution():
    """Fed_Funds_Rate and FOMC_Statement should both resolve to FOMC_RATE_DECISION."""
    print("test_fomc_event_type_resolution")
    clf = fresh_classifier()
    for etype in ("Fed_Funds_Rate", "FOMC_Statement", "federal_funds_rate", "fomc_rate_decision"):
        ev  = make_event(event_type=etype, deviation_score=2.0, trigger_id=_uuid())
        sig = clf.classify(ev)
        _assert(
            sig is not None and sig.trigger_type == "FOMC_RATE_DECISION",
            f"event_type={etype!r} → {sig.trigger_type if sig else None!r}",
        )


def test_keyword_scan_fallback():
    """Strings not in _EXACT_MAP but matching a keyword rule should still resolve."""
    print("test_keyword_scan_fallback")
    clf = fresh_classifier()
    cases = [
        ("FOMC_Press_Conference_Extended", "FED_CHAIR_SPEECH"),
        ("EIA_Crude_Oil_Inventories_Weekly", "EIA_CRUDE_INVENTORY"),
        ("Natural_Gas_Storage_Change", "EIA_GAS_STORAGE"),
        ("USDA_WASDE_Monthly_Report", "USDA_WASDE_REPORT"),
    ]
    for etype, expected_tt in cases:
        ev  = make_event(event_type=etype, deviation_score=1.5, trigger_id=_uuid())
        sig = clf.classify(ev)
        _assert(
            sig is not None and sig.trigger_type == expected_tt,
            f"{etype!r} → {sig.trigger_type if sig else None!r} (expected {expected_tt!r})",
        )


def test_family_name_resolution():
    """Existing snake_case family names should resolve via _EXACT_MAP."""
    print("test_family_name_resolution")
    clf = fresh_classifier()
    cases = [
        ("opec_action",      "OPEC_PRODUCTION_DECISION"),
        ("fed_tightening",   "DOLLAR_SHOCK"),
        ("weather_shock",    "WEATHER_SHOCK"),
        ("energy_transition","ENERGY_TRANSITION_SIGNAL"),
    ]
    for etype, expected_tt in cases:
        ev  = make_event(event_type=etype, deviation_score=1.0, trigger_id=_uuid())
        sig = clf.classify(ev)
        _assert(
            sig is not None and sig.trigger_type == expected_tt,
            f"family={etype!r} → {sig.trigger_type if sig else None!r}",
        )


def test_recession_flag_full_universe():
    """RECESSION_FLAG should affect all five clusters."""
    print("test_recession_flag_full_universe")
    clf = fresh_classifier()
    ev  = make_event(event_type="USREC", deviation_score=1.0, impact="high")
    sig = clf.classify(ev)
    _assert(sig is not None,                            "signal emitted")
    _assert(sig.trigger_type == "RECESSION_FLAG",       f"trigger_type={sig.trigger_type!r}")
    _assert(sig.cascade_priority == 1,                  f"priority={sig.cascade_priority}")
    for cluster in ("energy", "metals", "agriculture", "livestock", "digital"):
        _assert(cluster in sig.affected_clusters,       f"{cluster} in affected_clusters")


def test_decay_expires_in_future():
    """decay_expires_at must always be strictly after classified_at."""
    print("test_decay_expires_in_future")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=2.0)
    sig = clf.classify(ev)
    _assert(sig is not None, "signal emitted")

    expires = datetime.fromisoformat(sig.decay_expires_at)
    now     = datetime.now(timezone.utc)
    _assert(expires > now,
            f"decay_expires_at={sig.decay_expires_at} should be > now={now.isoformat()}")


def test_severity_ordering():
    """SeverityTier is an IntEnum — LOW < MEDIUM < HIGH < CRITICAL."""
    print("test_severity_ordering")
    from services.trigger_classifier import SeverityTier
    tiers = [SeverityTier.LOW, SeverityTier.MEDIUM, SeverityTier.HIGH, SeverityTier.CRITICAL]
    for i in range(len(tiers) - 1):
        _assert(tiers[i] < tiers[i + 1],
                f"{tiers[i].label()} < {tiers[i+1].label()}")


def test_signal_is_live_property():
    """TriggerSignal.is_live should be True immediately after classification."""
    print("test_signal_is_live_property")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=2.0)
    sig = clf.classify(ev)
    _assert(sig is not None,  "signal emitted")
    _assert(sig.is_live,      "signal.is_live is True immediately after classification")


def test_opec_long_decay():
    """OPEC_PRODUCTION_DECISION has a 7-day decay — decay_expires_at should be ~7 days out."""
    print("test_opec_long_decay")
    clf = fresh_classifier()
    ev  = make_event(event_type="opec_action", deviation_score=1.5, impact="high")
    sig = clf.classify(ev)
    _assert(sig is not None, "signal emitted")
    expires = datetime.fromisoformat(sig.decay_expires_at)
    now     = datetime.now(timezone.utc)
    days_out = (expires - now).total_seconds() / 86400
    _assert(days_out > 6.9,  f"OPEC decay should be ~7 days; got {days_out:.2f}d")


def test_blended_surprise_method():
    """When expected_value is provided classification_method should contain 'blended_surprise'."""
    print("test_blended_surprise_method")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", expected_value=310.0,
                     actual_value=313.1, deviation_score=2.0, impact="low")
    sig = clf.classify(ev)
    _assert(sig is not None,                            "signal emitted")
    _assert("blended_surprise" in sig.classification_method,
            f"method={sig.classification_method!r}")


def test_flush_dedup_cache():
    """flush_dedup_cache removes stale entries; subsequent same-key events are fresh."""
    print("test_flush_dedup_cache")
    import time
    from services.trigger_classifier import TriggerClassifier
    from services.trigger_config import load_trigger_registry

    # 0.00001 h = 0.036 s window; wait 0.1 s before flush so entry is stale
    clf = TriggerClassifier(load_trigger_registry(), dedup_window_hours=0.00001)
    ev1 = make_event(event_type="CPI", deviation_score=2.0, trigger_id=_uuid())
    clf.classify(ev1)
    _assert(len(clf._seen) == 1, f"one entry in dedup cache, got {len(clf._seen)}")

    time.sleep(0.10)          # let window expire so entry is considered stale
    clf.flush_dedup_cache()
    _assert(len(clf._seen) == 0, f"cache cleared after flush, got {len(clf._seen)}")


def test_ppi_release_resolution():
    """PPI event_type variants should resolve to PPI_RELEASE."""
    print("test_ppi_release_resolution")
    clf = fresh_classifier()
    for etype in ("PPI_All_Commodities", "PPI_m/m", "PPI_y/y", "Core_PPI_m/m"):
        ev  = make_event(event_type=etype, deviation_score=1.5, trigger_id=_uuid())
        sig = clf.classify(ev)
        _assert(
            sig is not None and sig.trigger_type == "PPI_RELEASE",
            f"{etype!r} → {sig.trigger_type if sig else None!r}",
        )


def test_direction_neutral_near_zero():
    """Very small |deviation_score| (< 0.10) should give direction='neutral'."""
    print("test_direction_neutral_near_zero")
    clf = fresh_classifier()
    ev  = make_event(event_type="CPI", deviation_score=0.05,
                     actual_value=310.0, expected_value=309.95)
    sig = clf.classify(ev)
    _assert(sig is not None,               "signal emitted")
    _assert(sig.direction == "neutral",    f"direction={sig.direction!r}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_cpi_normal_high_deviation,
        test_severity_tiers_all_reachable,
        test_missing_actual_value_returns_none,
        test_missing_expected_falls_back_to_z_score,
        test_unknown_event_type_returns_none,
        test_repeat_event_detection,
        test_repeat_expired_resets,
        test_negative_deviation_direction,
        test_zero_deviation_low_severity,
        test_extreme_z_score_clamped_to_one,
        test_high_impact_floor,
        test_zero_expected_value_no_exception,
        test_batch_classification,
        test_fomc_event_type_resolution,
        test_keyword_scan_fallback,
        test_family_name_resolution,
        test_recession_flag_full_universe,
        test_decay_expires_in_future,
        test_severity_ordering,
        test_signal_is_live_property,
        test_opec_long_decay,
        test_blended_surprise_method,
        test_flush_dedup_cache,
        test_ppi_release_resolution,
        test_direction_neutral_near_zero,
    ]

    print()
    print("=" * 64)
    print("TRIGGER CLASSIFICATION ENGINE — TEST SUITE")
    print("=" * 64)

    for fn in tests:
        print()
        try:
            fn()
        except Exception:
            _fail(fn.__name__, traceback.format_exc())

    print()
    print("=" * 64)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 64)
    print()
