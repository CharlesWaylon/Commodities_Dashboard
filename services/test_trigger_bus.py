"""
Test suite — Trigger Bus (Pub/Sub Layer)

All tests use the in-process transport (no Redis required).
A fresh TriggerBus is constructed per test to guarantee isolation.

Covers:
  - Channel routing (trigger_type → channel(s))
  - Multi-channel fan-out (_MULTI_CHANNEL secondaries)
  - publish() delivers to subscribed handler
  - publish() does NOT deliver to unsubscribed model
  - Multiple subscribers on the same channel all receive
  - publish_raw() sends arbitrary dicts
  - Unsubscribe stops delivery
  - Dead-letter queue: failed handler → DLQ entry
  - dlq_size() / dlq_peek() / dlq_clear()
  - dlq_retry() re-dispatches and returns success count
  - dlq_retry() re-queues entries that fail again
  - dlq_retry() discards after max_retries
  - Serialization / deserialization roundtrip
  - reconstruct_signal() produces TriggerSignal
  - health() returns correct structure
  - start_consuming() + stop_consuming() thread lifecycle
  - Consumer thread dispatches messages to handler
  - Subscription registry: register / unregister / subscribers_for
  - DEFAULT_SUBSCRIPTIONS covers all known model_targets
  - ALL_CHANNELS contains all mapped channel names
  - CHANNEL_MAP covers all registry trigger_types

Run:
    python -m services.test_trigger_bus
"""

import sys
import time
import threading
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone

PASS = 0
FAIL = 0


def _ok(msg: str) -> None:
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    suffix = f"\n         {detail}" if detail else ""
    print(f"  FAIL  {msg}{suffix}")


def _assert(cond: bool, msg: str, detail: str = "") -> None:
    if cond:
        _ok(msg)
    else:
        _fail(msg, detail)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_bus():
    """Return a TriggerBus using the in-process transport (redis_url=None)."""
    from services.trigger_bus import TriggerBus
    return TriggerBus(redis_url=None)


def make_signal(trigger_type: str = "CPI_RELEASE", deviation: float = 2.0,
                severity: str = "HIGH"):
    """Construct a minimal TriggerSignal suitable for bus tests."""
    from services.trigger_classifier import TriggerClassifier, SeverityTier
    from services.trigger_config import load_trigger_registry
    from services.macro_ingestion import MacroEvent

    reg = load_trigger_registry()
    clf = TriggerClassifier(reg)

    # Map trigger_type → an event_type the classifier knows
    _event_map = {
        "CPI_RELEASE":              "CPI",
        "FOMC_RATE_DECISION":       "Fed_Funds_Rate",
        "NONFARM_PAYROLLS":         "NFP",
        "OPEC_PRODUCTION_DECISION": "opec_action",
        "EIA_CRUDE_INVENTORY":      "EIA_Crude_Oil_Inventories",
        "EIA_GAS_STORAGE":          "EIA_Natural_Gas_Storage",
        "USDA_WASDE_REPORT":        "WASDE",
        "FED_CHAIR_SPEECH":         "FOMC_Press_Conference",
        "DOLLAR_SHOCK":             "fed_tightening",
        "GEOPOLITICAL_SHOCK":       "geopolitical_shock",
        "RECESSION_FLAG":           "USREC",
        "WEATHER_SHOCK":            "weather_shock",
        "ENERGY_TRANSITION_SIGNAL": "energy_transition",
        "PPI_RELEASE":              "PPI_All_Commodities",
    }
    etype = _event_map.get(trigger_type, "CPI")
    ev = MacroEvent(
        trigger_id        = str(uuid.uuid4()),
        source            = "FRED",
        event_type        = etype,
        expected_value    = None,
        actual_value      = 313.0,
        release_timestamp = datetime.now(timezone.utc).isoformat(),
        deviation_score   = deviation,
        impact            = "high",
    )
    sig = clf.classify(ev)
    assert sig is not None, f"make_signal: classifier returned None for {trigger_type}"
    return sig


def wait_for(condition, timeout=1.0, interval=0.02) -> bool:
    """Poll condition() until True or timeout. Returns whether it became True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_channel_routing_primary():
    """Each trigger_type maps to the expected primary channel."""
    print("test_channel_routing_primary")
    from services.trigger_bus import route_signal, CHANNEL_MAP

    cases = {
        "CPI_RELEASE":              "inflation",
        "FOMC_RATE_DECISION":       "monetary_policy",
        "NONFARM_PAYROLLS":         "employment",
        "OPEC_PRODUCTION_DECISION": "energy_supply",
        "EIA_CRUDE_INVENTORY":      "energy_supply",
        "EIA_GAS_STORAGE":          "energy_supply",
        "USDA_WASDE_REPORT":        "agriculture",
        "FED_CHAIR_SPEECH":         "monetary_policy",
        "DOLLAR_SHOCK":             "monetary_policy",
        "GEOPOLITICAL_SHOCK":       "geopolitical",
        "RECESSION_FLAG":           "macro_regime",
        "WEATHER_SHOCK":            "weather",
        "ENERGY_TRANSITION_SIGNAL": "structural",
        "PPI_RELEASE":              "inflation",
    }
    for tt, expected_ch in cases.items():
        sig = make_signal(tt)
        channels = route_signal(sig)
        _assert(
            channels[0] == expected_ch,
            f"{tt} → primary={channels[0]!r}",
            f"expected {expected_ch!r}",
        )


def test_multi_channel_fan_out():
    """Trigger types in _MULTI_CHANNEL broadcast to both primary and secondary."""
    print("test_multi_channel_fan_out")
    from services.trigger_bus import route_signal, _MULTI_CHANNEL

    for tt, secondary in _MULTI_CHANNEL.items():
        sig = make_signal(tt)
        channels = route_signal(sig)
        _assert(len(channels) >= 2, f"{tt}: {len(channels)} channels (expected ≥ 2)")
        _assert(secondary in channels, f"{tt}: secondary {secondary!r} in {channels}")


def test_publish_delivers_to_subscriber():
    """A subscribed model receives the message on its channel."""
    print("test_publish_delivers_to_subscriber")
    bus      = fresh_bus()
    received = []

    bus.subscribe("xgboost", lambda data, ch, mid: received.append((data, ch, mid)),
                  channels=["inflation"])
    bus.start_consuming("xgboost")

    sig = make_signal("CPI_RELEASE", deviation=2.0)
    bus.publish(sig)

    ok = wait_for(lambda: len(received) >= 1)
    bus.stop_consuming("xgboost")
    bus.close()

    _assert(ok, "message received within 1 s")
    if received:
        data, ch, mid = received[0]
        _assert(data["trigger_type"] == "CPI_RELEASE", f"trigger_type={data.get('trigger_type')!r}")
        _assert(ch == "inflation",                      f"channel={ch!r}")
        _assert(mid == "xgboost",                       f"model_id={mid!r}")


def test_non_subscriber_does_not_receive():
    """A model subscribed only to 'agriculture' must not receive CPI_RELEASE.
    CPI_RELEASE fans out to inflation + monetary_policy — never agriculture."""
    print("test_non_subscriber_does_not_receive")
    bus      = fresh_bus()
    received = []

    bus.subscribe("xgboost", lambda d, c, m: received.append(d),
                  channels=["agriculture"])   # CPI_RELEASE never reaches agriculture
    bus.start_consuming("xgboost")

    bus.publish(make_signal("CPI_RELEASE"))
    time.sleep(0.20)

    bus.stop_consuming("xgboost")
    bus.close()

    _assert(len(received) == 0, f"no message on unsubscribed channel (got {len(received)})")


def test_multiple_subscribers_same_channel():
    """All subscribers on 'inflation' receive a CPI_RELEASE message."""
    print("test_multiple_subscribers_same_channel")
    bus = fresh_bus()
    counts: dict = {"a": 0, "b": 0, "c": 0}

    for mid in ("model_a", "model_b", "model_c"):
        key = mid.split("_")[1]
        bus.subscribe(mid, lambda d, c, m, k=key: counts.__setitem__(k, counts[k] + 1),
                      channels=["inflation"])
        bus.start_consuming(mid)

    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: all(v >= 1 for v in counts.values()))

    for mid in ("model_a", "model_b", "model_c"):
        bus.stop_consuming(mid)
    bus.close()

    for k, v in counts.items():
        _assert(v >= 1, f"model_{k} received {v} message(s)")


def test_publish_raw_delivers():
    """publish_raw() delivers an arbitrary dict on a valid channel."""
    print("test_publish_raw_delivers")
    bus      = fresh_bus()
    received = []

    bus.subscribe("macro_router", lambda d, c, m: received.append(d),
                  channels=["macro_regime"])
    bus.start_consuming("macro_router")

    bus.publish_raw("macro_regime", {"test": True, "value": 42})
    wait_for(lambda: len(received) >= 1)

    bus.stop_consuming("macro_router")
    bus.close()

    _assert(len(received) >= 1, "raw message received")
    if received:
        _assert(received[0].get("value") == 42, f"payload intact: {received[0]}")


def test_publish_raw_invalid_channel():
    """publish_raw() raises ValueError for unknown channels."""
    print("test_publish_raw_invalid_channel")
    bus = fresh_bus()
    try:
        bus.publish_raw("nonexistent_channel", {"x": 1})
        _fail("expected ValueError — not raised")
    except ValueError:
        _ok("ValueError raised for unknown channel")
    finally:
        bus.close()


def test_unsubscribe_stops_delivery():
    """After unsubscribe(), the model receives no further messages."""
    print("test_unsubscribe_stops_delivery")
    bus      = fresh_bus()
    received = []

    bus.subscribe("xgboost", lambda d, c, m: received.append(d), channels=["inflation"])
    bus.start_consuming("xgboost")

    # First message — should arrive
    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: len(received) >= 1)

    bus.unsubscribe("xgboost")
    before = len(received)

    # Second message — should NOT arrive
    bus.publish(make_signal("CPI_RELEASE"))
    time.sleep(0.15)
    bus.close()

    _assert(len(received) == before, f"no new messages after unsubscribe (got {len(received) - before} extra)")


def test_failed_handler_goes_to_dlq():
    """An exception in the handler pushes the message to the DLQ."""
    print("test_failed_handler_goes_to_dlq")
    bus = fresh_bus()

    def bad_handler(data, ch, mid):
        raise RuntimeError("deliberate test failure")

    bus.subscribe("xgboost", bad_handler, channels=["inflation"])
    bus.start_consuming("xgboost")

    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: bus.dlq_size() >= 1)

    bus.stop_consuming("xgboost")
    bus.close()

    _assert(bus.dlq_size() >= 1, f"DLQ has {bus.dlq_size()} entry(ies)")
    entries = bus.dlq_peek(1)
    if entries:
        _assert(entries[0]["model_id"] == "xgboost", f"model_id={entries[0].get('model_id')!r}")
        _assert("RuntimeError" in entries[0]["error_class"],
                f"error_class={entries[0].get('error_class')!r}")


def test_dlq_size_and_peek():
    """dlq_size() and dlq_peek() return consistent results."""
    print("test_dlq_size_and_peek")
    bus = fresh_bus()

    def bad_handler(data, ch, mid):
        raise ValueError("test dlq")

    bus.subscribe("macro_router", bad_handler, channels=["monetary_policy"])
    bus.start_consuming("macro_router")

    for _ in range(3):
        bus.publish(make_signal("FOMC_RATE_DECISION"))
    wait_for(lambda: bus.dlq_size() >= 3)

    bus.stop_consuming("macro_router")
    bus.close()

    sz = bus.dlq_size()
    entries = bus.dlq_peek(10)
    _assert(sz >= 3,           f"dlq_size={sz} (expected ≥ 3)")
    _assert(len(entries) == sz, f"peek returned {len(entries)}, size={sz}")


def test_dlq_clear():
    """dlq_clear() empties the queue."""
    print("test_dlq_clear")
    bus = fresh_bus()

    def bad_handler(data, ch, mid):
        raise RuntimeError("fill dlq")

    bus.subscribe("xgboost", bad_handler, channels=["inflation"])
    bus.start_consuming("xgboost")
    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: bus.dlq_size() >= 1)

    bus.stop_consuming("xgboost")
    bus.dlq_clear()
    bus.close()

    _assert(bus.dlq_size() == 0, f"DLQ size after clear={bus.dlq_size()}")


def test_dlq_retry_success():
    """dlq_retry() re-dispatches entries to a now-working handler."""
    print("test_dlq_retry_success")
    bus      = fresh_bus()
    retried  = []
    fail_now = True

    def handler(data, ch, mid):
        if fail_now:
            raise RuntimeError("intentional first failure")
        retried.append(data)

    bus.subscribe("xgboost", handler, channels=["inflation"])
    bus.start_consuming("xgboost")
    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: bus.dlq_size() >= 1)
    bus.stop_consuming("xgboost")

    # Now fix the handler
    fail_now = False
    n = bus.dlq_retry()
    bus.close()

    _assert(n >= 1,           f"dlq_retry returned {n} (expected ≥ 1)")
    _assert(bus.dlq_size() == 0, f"DLQ empty after successful retry (size={bus.dlq_size()})")
    _assert(len(retried) >= 1, f"handler called on retry ({len(retried)} times)")


def test_dlq_retry_requeues_on_failure():
    """If the retry handler also fails, the entry is re-queued with retry_count+1."""
    print("test_dlq_retry_requeues_on_failure")
    bus = fresh_bus()

    def always_fails(data, ch, mid):
        raise RuntimeError("always bad")

    bus.subscribe("xgboost", always_fails, channels=["inflation"])
    bus.start_consuming("xgboost")
    bus.publish(make_signal("CPI_RELEASE"))
    wait_for(lambda: bus.dlq_size() >= 1)
    bus.stop_consuming("xgboost")

    before = bus.dlq_size()
    bus.dlq_retry(max_retries=3)
    after = bus.dlq_size()
    bus.close()

    # Failed retry re-queues — DLQ non-empty, retry_count incremented
    _assert(after >= 1, f"entry re-queued after failed retry (size={after})")
    entries = bus.dlq_peek(1)
    if entries:
        _assert(entries[0]["retry_count"] >= 1,
                f"retry_count={entries[0].get('retry_count')}")


def test_dlq_discards_after_max_retries():
    """After max_retries exhausted, entry is discarded (not re-queued)."""
    print("test_dlq_discards_after_max_retries")
    from services.trigger_bus import _DLQEntry, DeadLetterQueue

    dlq = DeadLetterQueue()
    # Pre-seed an entry that is already at the retry limit
    entry = _DLQEntry(
        payload     = {"trigger_type": "CPI_RELEASE"},
        channel     = "inflation",
        model_id    = "xgboost",
        error_class = "RuntimeError",
        error_msg   = "old failure",
        failed_at   = datetime.now(timezone.utc).isoformat(),
        retry_count = 3,
    )
    dlq.push(entry)

    bus = fresh_bus()
    bus._dlq = dlq

    def handler(data, ch, mid):
        raise RuntimeError("still failing")

    bus.subscribe("xgboost", handler, channels=["inflation"])
    n = bus.dlq_retry(max_retries=3)
    bus.close()

    _assert(n == 0,                "no successful retries")
    _assert(bus.dlq_size() == 0,   "discarded after max_retries — DLQ empty")


def test_serialization_roundtrip():
    """_serialize → deserialize preserves all scalar fields."""
    print("test_serialization_roundtrip")
    from services.trigger_bus import _serialize, deserialize, route_signal

    sig      = make_signal("CPI_RELEASE", deviation=2.5)
    channels = route_signal(sig)
    payload  = _serialize(sig, channels, channels[0])
    data     = deserialize(payload)

    _assert(data["trigger_type"]    == sig.trigger_type,     f"trigger_type={data.get('trigger_type')!r}")
    _assert(data["severity"]        == sig.severity.name,    f"severity={data.get('severity')!r}")
    _assert(data["magnitude_score"] == sig.magnitude_score,  f"magnitude={data.get('magnitude_score')}")
    _assert(data["channel"]         == channels[0],          f"channel={data.get('channel')!r}")
    _assert(data["schema"]          == "TriggerSignal/v1",   f"schema={data.get('schema')!r}")
    _assert("published_at" in data,                          "published_at present")


def test_reconstruct_signal():
    """reconstruct_signal() from bus payload returns a TriggerSignal."""
    print("test_reconstruct_signal")
    from services.trigger_bus import _serialize, deserialize, route_signal, reconstruct_signal
    from services.trigger_classifier import TriggerSignal

    sig      = make_signal("FOMC_RATE_DECISION", deviation=3.0)
    channels = route_signal(sig)
    payload  = _serialize(sig, channels, channels[0])
    data     = deserialize(payload)
    rebuilt  = reconstruct_signal(data)

    _assert(rebuilt is not None,                                   "reconstruct_signal returned non-None")
    _assert(isinstance(rebuilt, TriggerSignal),                    f"type={type(rebuilt).__name__}")
    _assert(rebuilt.trigger_type == sig.trigger_type,              f"trigger_type matches")
    _assert(rebuilt.magnitude_score == sig.magnitude_score,        f"magnitude matches")
    _assert(rebuilt.severity        == sig.severity,               f"severity matches")


def test_health_structure():
    """health() returns a dict with all expected keys."""
    print("test_health_structure")
    bus = fresh_bus()
    bus.subscribe("macro_router", lambda d, c, m: None, channels=["monetary_policy"])
    h   = bus.health()
    bus.close()

    required = {"backend", "subscriptions", "n_subscribers", "consumer_threads",
                "queue_depths", "dlq_size", "all_channels"}
    for key in required:
        _assert(key in h, f"health() has key {key!r}")

    _assert(h["backend"]       == "in_process",  f"backend={h['backend']!r}")
    _assert(h["n_subscribers"] == 1,             f"n_subscribers={h['n_subscribers']}")
    _assert("macro_router" in h["subscriptions"], "macro_router in subscriptions")


def test_consumer_thread_lifecycle():
    """start_consuming() spawns a live thread; stop_consuming() joins it."""
    print("test_consumer_thread_lifecycle")
    import threading
    bus = fresh_bus()
    bus.subscribe("xgboost", lambda d, c, m: None, channels=["inflation"])
    t   = bus.start_consuming("xgboost")

    _assert(isinstance(t, threading.Thread), f"returns Thread, got {type(t).__name__}")
    _assert(t.is_alive(),                    "thread is alive after start_consuming")

    bus.stop_consuming("xgboost")
    ok = wait_for(lambda: not t.is_alive(), timeout=2.0)
    bus.close()

    _assert(ok, "thread stopped within 2 s")


def test_start_consuming_without_subscribe_raises():
    """start_consuming() raises RuntimeError when no subscription exists."""
    print("test_start_consuming_without_subscribe_raises")
    bus = fresh_bus()
    try:
        bus.start_consuming("unregistered_model")
        _fail("expected RuntimeError — not raised")
    except RuntimeError:
        _ok("RuntimeError raised for unregistered model")
    finally:
        bus.close()


def test_subscription_registry_basics():
    """SubscriptionRegistry register / unregister / subscribers_for."""
    print("test_subscription_registry_basics")
    from services.trigger_bus import SubscriptionRegistry

    reg = SubscriptionRegistry()
    handler = lambda d, c, m: None

    reg.register("model_a", ["inflation", "monetary_policy"], handler)
    reg.register("model_b", ["inflation"], handler)
    reg.register("model_c", ["employment"], handler)

    inflation_subs = reg.subscribers_for("inflation")
    ids = {s.model_id for s in inflation_subs}
    _assert("model_a" in ids, f"model_a in inflation subscribers: {ids}")
    _assert("model_b" in ids, f"model_b in inflation subscribers: {ids}")
    _assert("model_c" not in ids, f"model_c NOT in inflation subscribers: {ids}")

    reg.unregister("model_a")
    after = {s.model_id for s in reg.subscribers_for("inflation")}
    _assert("model_a" not in after, f"model_a removed from registry: {after}")


def test_default_subscriptions_cover_model_targets():
    """Every model_target in VALID_MODEL_TARGETS has a DEFAULT_SUBSCRIPTIONS entry."""
    print("test_default_subscriptions_cover_model_targets")
    from services.trigger_bus import DEFAULT_SUBSCRIPTIONS
    from services.trigger_config import VALID_MODEL_TARGETS

    for target in VALID_MODEL_TARGETS - {"all"}:
        _assert(
            target in DEFAULT_SUBSCRIPTIONS,
            f"{target!r} in DEFAULT_SUBSCRIPTIONS",
        )


def test_all_channels_contains_every_mapped_channel():
    """ALL_CHANNELS must contain every channel value in CHANNEL_MAP."""
    print("test_all_channels_contains_every_mapped_channel")
    from services.trigger_bus import CHANNEL_MAP, ALL_CHANNELS

    for tt, ch in CHANNEL_MAP.items():
        _assert(ch in ALL_CHANNELS, f"CHANNEL_MAP[{tt!r}]={ch!r} in ALL_CHANNELS")


def test_channel_map_covers_all_registry_trigger_types():
    """Every trigger_type in the registry has a CHANNEL_MAP entry."""
    print("test_channel_map_covers_all_registry_trigger_types")
    from services.trigger_bus import CHANNEL_MAP
    from services.trigger_config import load_trigger_registry

    reg = load_trigger_registry()
    for cfg in reg.all():
        _assert(
            cfg.trigger_type in CHANNEL_MAP,
            f"{cfg.trigger_type!r} in CHANNEL_MAP",
        )


def test_geopolitical_shock_fan_out():
    """GEOPOLITICAL_SHOCK fans out to both 'geopolitical' and 'energy_supply'."""
    print("test_geopolitical_shock_fan_out")
    from services.trigger_bus import route_signal

    sig      = make_signal("GEOPOLITICAL_SHOCK")
    channels = route_signal(sig)
    _assert("geopolitical" in channels,  f"geopolitical in {channels}")
    _assert("energy_supply" in channels, f"energy_supply in {channels}")


def test_recession_flag_fan_out():
    """RECESSION_FLAG fans out to both 'macro_regime' and 'inflation'."""
    print("test_recession_flag_fan_out")
    from services.trigger_bus import route_signal

    sig      = make_signal("RECESSION_FLAG")
    channels = route_signal(sig)
    _assert("macro_regime" in channels, f"macro_regime in {channels}")
    _assert("inflation"    in channels, f"inflation in {channels}")


def test_publish_returns_channel_list():
    """publish() returns the list of channels the signal was sent to."""
    print("test_publish_returns_channel_list")
    bus = fresh_bus()
    sig = make_signal("CPI_RELEASE")
    channels = bus.publish(sig)
    bus.close()

    _assert(isinstance(channels, list), f"publish() returns list, got {type(channels).__name__}")
    _assert("inflation" in channels,    f"inflation in {channels}")


def test_is_repeat_signal_still_routed():
    """is_repeat=True signals are published normally — bus does not suppress them."""
    print("test_is_repeat_signal_still_routed")
    from services.trigger_classifier import TriggerClassifier
    from services.trigger_config import load_trigger_registry
    from services.macro_ingestion import MacroEvent

    clf = TriggerClassifier(load_trigger_registry())
    ev  = MacroEvent(
        trigger_id="t1", source="FRED", event_type="CPI",
        expected_value=None, actual_value=313.0,
        release_timestamp=datetime.now(timezone.utc).isoformat(),
        deviation_score=2.0, impact="high",
    )
    clf.classify(ev)               # first → is_repeat=False
    sig2 = clf.classify(ev)        # second → is_repeat=True
    assert sig2 is not None

    bus      = fresh_bus()
    received = []
    bus.subscribe("xgboost", lambda d, c, m: received.append(d), channels=["inflation"])
    bus.start_consuming("xgboost")

    channels = bus.publish(sig2)
    wait_for(lambda: len(received) >= 1)

    bus.stop_consuming("xgboost")
    bus.close()

    _assert(len(channels) >= 1,            "is_repeat signal published to at least one channel")
    _assert(len(received) >= 1,            "is_repeat signal received by subscriber")
    if received:
        _assert(received[0].get("is_repeat"), "payload has is_repeat=True")


def test_concurrent_publishes():
    """Concurrent publishers do not corrupt the subscriber's message stream."""
    print("test_concurrent_publishes")
    bus      = fresh_bus()
    received = []
    lock     = threading.Lock()

    bus.subscribe("xgboost", lambda d, c, m: (lock.acquire(), received.append(d), lock.release()),
                  channels=["inflation"])
    bus.start_consuming("xgboost")

    n_threads = 5
    n_per     = 4

    def publisher():
        for _ in range(n_per):
            bus.publish(make_signal("CPI_RELEASE"))

    threads = [threading.Thread(target=publisher) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wait_for(lambda: len(received) >= n_threads * n_per, timeout=3.0)
    bus.stop_consuming("xgboost")
    bus.close()

    expected = n_threads * n_per
    _assert(len(received) == expected,
            f"received {len(received)}/{expected} concurrent messages")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_channel_routing_primary,
        test_multi_channel_fan_out,
        test_publish_delivers_to_subscriber,
        test_non_subscriber_does_not_receive,
        test_multiple_subscribers_same_channel,
        test_publish_raw_delivers,
        test_publish_raw_invalid_channel,
        test_unsubscribe_stops_delivery,
        test_failed_handler_goes_to_dlq,
        test_dlq_size_and_peek,
        test_dlq_clear,
        test_dlq_retry_success,
        test_dlq_retry_requeues_on_failure,
        test_dlq_discards_after_max_retries,
        test_serialization_roundtrip,
        test_reconstruct_signal,
        test_health_structure,
        test_consumer_thread_lifecycle,
        test_start_consuming_without_subscribe_raises,
        test_subscription_registry_basics,
        test_default_subscriptions_cover_model_targets,
        test_all_channels_contains_every_mapped_channel,
        test_channel_map_covers_all_registry_trigger_types,
        test_geopolitical_shock_fan_out,
        test_recession_flag_fan_out,
        test_publish_returns_channel_list,
        test_is_repeat_signal_still_routed,
        test_concurrent_publishes,
    ]

    print()
    print("=" * 64)
    print("TRIGGER BUS — TEST SUITE")
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
