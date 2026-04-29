"""
Test suite for trigger registry + SignalRouter.
Run: python models/test_router.py
"""

import sys
import traceback

from models.model_signal import ModelSignal
from models.triggers import (
    TriggerFamily,
    register_trigger_family,
    get_trigger_family,
    all_trigger_families,
)
from models.router import SignalRouter

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


def _make_signal(commodity: str, confidence: float = 0.10) -> ModelSignal:
    return ModelSignal(
        commodity=commodity,
        model_type="ml",
        model_name="XGBoostForecaster",
        timestamp="2026-04-28T18:30:00+00:00",
        forecast_return=0.01,
        confidence=confidence,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_default_registry():
    print("1. Default trigger registry")
    try:
        names = {f.name for f in all_trigger_families()}
        for expected in ("opec_action", "fed_tightening", "weather_shock", "energy_transition"):
            assert expected in names, f"missing default family: {expected}"
        _ok(f"4 default families registered: {sorted(n for n in names if not n.startswith('test_'))}")
    except Exception as e:
        _fail("default registry", e)


def test_extensibility_hook():
    print("2. Research extension hook")
    try:
        custom = TriggerFamily(
            name="patented_research_a",
            description="Hypothetical proprietary causal signal",
            affected_commodities=("WTI Crude Oil", "Brent Crude Oil"),
            source="research",
        )
        register_trigger_family(custom)
        retrieved = get_trigger_family("patented_research_a")
        assert retrieved is not None
        assert retrieved.source == "research"
        assert retrieved.affected_commodities == ("WTI Crude Oil", "Brent Crude Oil")
        _ok("custom research family registered & retrieved cleanly")
    except Exception as e:
        _fail("extensibility", e)


def test_router_no_events():
    print("3. SignalRouter — no events (passthrough)")
    try:
        sigs = [_make_signal("WTI Crude Oil", 0.08)]
        router = SignalRouter()
        out = router.route(sigs, events=[])
        assert len(out) == 1
        assert out[0].confidence == 0.08
        assert out[0] is not sigs[0], "should deep-copy"
        assert "routed_triggers" not in out[0].metadata
        _ok("no events → confidence unchanged, signals deep-copied")
    except Exception as e:
        _fail("router no-op", e)


def test_router_single_trigger():
    print("4. SignalRouter — single trigger boost")
    try:
        router = SignalRouter()
        sigs = [
            _make_signal("WTI Crude Oil", 0.10),
            _make_signal("Gold (COMEX)",  0.10),
        ]
        ev = router.make_event("opec_action", strength=0.8, rationale="OPEC+ cut 1.3M bpd")
        out = router.route(sigs, [ev])

        # WTI is in opec_action — boosted by 1 + 0.5*0.8 = 1.4
        wti = out[0]
        expected_mult = 1.4
        assert abs(wti.confidence - 0.10 * expected_mult) < 1e-6
        assert wti.metadata["confidence_multiplier"] == round(expected_mult, 4)
        assert wti.metadata["routed_triggers"][0]["family"] == "opec_action"
        assert wti.metadata["routed_triggers"][0]["rationale"] == "OPEC+ cut 1.3M bpd"

        # Gold is NOT in opec_action — unchanged
        gold = out[1]
        assert gold.confidence == 0.10
        assert "routed_triggers" not in gold.metadata

        _ok(f"WTI 0.10 → {wti.confidence:.4f} (×{expected_mult}); Gold untouched")
    except Exception as e:
        _fail("single trigger", e)


def test_router_multi_trigger_compound():
    print("5. SignalRouter — multiple triggers compound")
    try:
        # Add a custom family that overlaps with fed_tightening on Gold
        register_trigger_family(TriggerFamily(
            name="test_overlap_gold",
            description="test-only overlap family",
            affected_commodities=("Gold (COMEX)",),
            source="research",
        ))
        router = SignalRouter()
        sigs = [_make_signal("Gold (COMEX)", 0.10)]
        ev1 = router.make_event("fed_tightening",   strength=0.6)
        ev2 = router.make_event("test_overlap_gold", strength=0.4)
        out = router.route(sigs, [ev1, ev2])

        m1, m2 = 1.0 + 0.5 * 0.6, 1.0 + 0.5 * 0.4   # 1.3, 1.2
        expected = 0.10 * m1 * m2                    # 0.156
        assert abs(out[0].confidence - expected) < 1e-6
        assert len(out[0].metadata["routed_triggers"]) == 2
        _ok(f"Gold compound: 0.10 × {m1} × {m2} = {expected:.4f}")
    except Exception as e:
        _fail("multi-trigger compound", e)


def test_router_caps_at_one():
    print("6. SignalRouter — confidence capped at 1.0")
    try:
        router = SignalRouter()
        sigs = [_make_signal("WTI Crude Oil", 0.95)]
        ev = router.make_event("opec_action", strength=1.0)
        out = router.route(sigs, [ev])
        # Raw: 0.95 * 1.5 = 1.425 → capped to 1.0
        assert out[0].confidence == 1.0
        _ok("0.95 × 1.5 = 1.425 → capped to 1.0")
    except Exception as e:
        _fail("cap", e)


def test_make_event_validation():
    print("7. make_event — strength clamping & unknown family")
    try:
        router = SignalRouter()
        ev_hi = router.make_event("opec_action", strength=1.5)
        ev_lo = router.make_event("opec_action", strength=-0.2)
        assert ev_hi.strength == 1.0
        assert ev_lo.strength == 0.0
        _ok("strength clamped to [0, 1]")

        try:
            router.make_event("nonexistent_family", strength=0.5)
        except ValueError:
            _ok("unknown family raises ValueError")
        else:
            _fail("unknown family should have raised")
    except Exception as e:
        _fail("make_event validation", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("ROUTER & TRIGGER REGISTRY — TEST SUITE")
    print("=" * 60)
    print()

    for fn in [
        test_default_registry,
        test_extensibility_hook,
        test_router_no_events,
        test_router_single_trigger,
        test_router_multi_trigger_compound,
        test_router_caps_at_one,
        test_make_event_validation,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print("Router foundation ready for chunk 2B (real trigger detection).")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
