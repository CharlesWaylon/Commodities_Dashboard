"""
Trigger Bus — Redis-backed Pub/Sub layer for TriggerSignal dispatch
=====================================================================
Producers publish TriggerSignal objects to typed channels.
Consumers (models) subscribe to only the channels relevant to them.

Channel taxonomy
----------------
  monetary_policy   FOMC decisions, DXY shocks, Fed speeches
  inflation         CPI, PPI releases
  employment        NFP, unemployment claims
  energy_supply     OPEC decisions, EIA crude & gas storage
  agriculture       USDA WASDE crop reports
  geopolitical      Sanctions, conflict, supply-route disruptions
  macro_regime      Recession flag (USREC), structural regime shifts
  weather           ENSO / climate shock
  structural        Energy transition, battery-metals signal

Transport
---------
  Redis (preferred)
      Each channel maps to a Redis pub/sub topic.  Messages persist in a
      per-subscriber Redis List (LPUSH / BRPOP) so consumers that are briefly
      offline don't lose signals.  DLQ is a dedicated Redis List.
      Requires: REDIS_URL env var (default redis://localhost:6379).
      Install: pip install redis>=5.0

  In-process fallback (automatic when Redis is unavailable)
      Identical pub/sub semantics implemented with threading + queue.Queue.
      DLQ is a thread-safe collections.deque.  No configuration needed.
      Activated automatically when redis-py is missing or the server is down.

Public API
----------
  Bus.publish(signal)                            → list[str]  channels hit
  Bus.publish_raw(channel, data)                 → None
  Bus.subscribe(model_id, handler, channels)     → None
  Bus.unsubscribe(model_id)                      → None
  Bus.start_consuming(model_id)                  → threading.Thread
  Bus.stop_consuming(model_id)                   → None
  Bus.dlq_size()                                 → int
  Bus.dlq_peek(n)                                → list[dict]
  Bus.dlq_retry(max_retries)                     → int  (dispatches retried)
  Bus.dlq_clear()                                → None
  Bus.health()                                   → dict
  Bus.close()                                    → None

  BUS                                            — module-level singleton
  publish_signal(signal)                         — shortcut to BUS.publish
  subscribe_model(model_id, handler, channels)   — shortcut to BUS.subscribe

Subscription registry
---------------------
Default channel interests for every model in the trigger registry.
Models can override or add channels at registration time via subscribe().

Dead-letter queue (DLQ)
-----------------------
When a subscribed handler raises an exception, the message payload plus
error context is pushed to the DLQ instead of being silently dropped.
dlq_retry() re-dispatches up to `max_retries` times before discarding.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import queue
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── Lazy redis import ──────────────────────────────────────────────────────────
try:
    import redis as _redis_mod
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    _redis_mod = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════════
# Channel definitions
# ═══════════════════════════════════════════════════════════════════════════════

# Maps trigger_type → primary channel name.
# A trigger_type in _MULTI_CHANNEL also broadcasts to secondary channels.
CHANNEL_MAP: Dict[str, str] = {
    "FOMC_RATE_DECISION":       "monetary_policy",
    "FED_CHAIR_SPEECH":         "monetary_policy",
    "DOLLAR_SHOCK":             "monetary_policy",
    "CPI_RELEASE":              "inflation",
    "PPI_RELEASE":              "inflation",
    "NONFARM_PAYROLLS":         "employment",
    "OPEC_PRODUCTION_DECISION": "energy_supply",
    "EIA_CRUDE_INVENTORY":      "energy_supply",
    "EIA_GAS_STORAGE":          "energy_supply",
    "USDA_WASDE_REPORT":        "agriculture",
    "GEOPOLITICAL_SHOCK":       "geopolitical",
    "RECESSION_FLAG":           "macro_regime",
    "WEATHER_SHOCK":            "weather",
    "ENERGY_TRANSITION_SIGNAL": "structural",
}

# Trigger types that additionally broadcast to a second channel.
_MULTI_CHANNEL: Dict[str, str] = {
    "GEOPOLITICAL_SHOCK": "energy_supply",   # geo shocks almost always hit energy
    "RECESSION_FLAG":     "inflation",       # recession = deflationary signal
    "DOLLAR_SHOCK":       "inflation",       # DXY move imports inflation
    "CPI_RELEASE":        "monetary_policy", # CPI directly informs Fed path
}

ALL_CHANNELS: Set[str] = {
    "monetary_policy", "inflation", "employment",
    "energy_supply", "agriculture", "geopolitical",
    "macro_regime", "weather", "structural",
}

_REDIS_KEY_PREFIX = "trigger_bus"
_DLQ_KEY          = f"{_REDIS_KEY_PREFIX}:dlq"
_MAX_DLQ          = 500      # in-process DLQ capacity (Redis is unlimited)
_MAX_RETRIES      = 3
_CONSUMER_TIMEOUT = 0.1      # seconds between queue.get() polls


# ── Default channel subscriptions ─────────────────────────────────────────────
# Each model_id declares which channels it wants to receive.
# Overrideable at subscribe() time.

DEFAULT_SUBSCRIPTIONS: Dict[str, List[str]] = {
    "macro_router":   ["monetary_policy", "inflation", "employment",
                       "macro_regime", "energy_supply"],
    "xgboost":        ["monetary_policy", "inflation", "employment",
                       "energy_supply", "agriculture", "geopolitical", "weather"],
    "random_forest":  ["monetary_policy", "inflation",
                       "energy_supply", "agriculture", "weather"],
    "lstm":           ["monetary_policy", "inflation", "energy_supply", "geopolitical"],
    "quantum":        ["monetary_policy", "inflation"],
    "meta_predictor": list(ALL_CHANNELS),     # receives everything
}


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize(signal, channels: List[str], channel: str) -> bytes:
    """
    Serialize a TriggerSignal to a JSON-encoded bytes payload.

    ``config`` is intentionally excluded — consumers look it up from the
    registry using ``trigger_type`` to avoid embedding a large nested object.
    """
    from services.trigger_classifier import SeverityTier  # local to avoid circular import

    payload = {
        "schema":                "TriggerSignal/v1",
        "signal_id":             signal.signal_id,
        "trigger_type":          signal.trigger_type,
        "family_name":           signal.family_name,
        "source_event_id":       signal.source_event_id,
        "magnitude_score":       signal.magnitude_score,
        "severity":              signal.severity.name,       # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
        "direction":             signal.direction,
        "release_timestamp":     signal.release_timestamp,
        "classified_at":         signal.classified_at,
        "decay_expires_at":      signal.decay_expires_at,
        "affected_commodities":  list(signal.affected_commodities),
        "affected_clusters":     list(signal.affected_clusters),
        "model_targets":         list(signal.model_targets),
        "cascade_priority":      signal.cascade_priority,
        "classification_method": signal.classification_method,
        "raw_deviation_score":   signal.raw_deviation_score,
        "event_type":            signal.event_type,
        "actual_value":          signal.actual_value,
        "expected_value":        signal.expected_value,
        "is_repeat":             signal.is_repeat,
        # Bus metadata
        "channel":               channel,
        "all_channels":          channels,
        "published_at":          datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def deserialize(payload: bytes) -> dict:
    """
    Deserialize a bus payload back to a plain dict.

    Consumers can either use the dict directly or reconstruct a TriggerSignal
    via ``reconstruct_signal(data)`` if they need the typed object.
    """
    return json.loads(payload)


def reconstruct_signal(data: dict):
    """
    Reconstruct a TriggerSignal from a deserialized bus payload dict.
    Returns None if the schema version or trigger_type is unrecognised.
    """
    try:
        from services.trigger_classifier import TriggerSignal, SeverityTier
        from services.trigger_config import REGISTRY

        cfg = REGISTRY.get(data["trigger_type"])
        if cfg is None:
            return None

        return TriggerSignal(
            signal_id             = data["signal_id"],
            trigger_type          = data["trigger_type"],
            family_name           = data["family_name"],
            source_event_id       = data["source_event_id"],
            magnitude_score       = data["magnitude_score"],
            severity              = SeverityTier[data["severity"]],
            direction             = data["direction"],
            release_timestamp     = data["release_timestamp"],
            classified_at         = data["classified_at"],
            decay_expires_at      = data["decay_expires_at"],
            affected_commodities  = tuple(data["affected_commodities"]),
            affected_clusters     = tuple(data["affected_clusters"]),
            model_targets         = tuple(data["model_targets"]),
            cascade_priority      = data["cascade_priority"],
            classification_method = data["classification_method"],
            raw_deviation_score   = data["raw_deviation_score"],
            event_type            = data["event_type"],
            actual_value          = data.get("actual_value"),
            expected_value        = data.get("expected_value"),
            is_repeat             = data.get("is_repeat", False),
            config                = cfg,
        )
    except Exception as exc:
        log.warning("reconstruct_signal failed: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Channel routing
# ═══════════════════════════════════════════════════════════════════════════════

def route_signal(signal) -> List[str]:
    """
    Return the ordered list of channel names a TriggerSignal should be
    published to.  Primary channel always comes first.
    """
    primary = CHANNEL_MAP.get(signal.trigger_type)
    if primary is None:
        log.warning("route_signal: no channel mapping for %s — skipping", signal.trigger_type)
        return []

    channels = [primary]
    secondary = _MULTI_CHANNEL.get(signal.trigger_type)
    if secondary and secondary != primary:
        channels.append(secondary)
    return channels


# ═══════════════════════════════════════════════════════════════════════════════
# Dead-letter queue
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _DLQEntry:
    payload:     dict
    channel:     str
    model_id:    str
    error_class: str
    error_msg:   str
    failed_at:   str
    retry_count: int = 0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


class DeadLetterQueue:
    """
    Thread-safe dead-letter queue with push, peek, retry, and clear.

    The in-process implementation uses a bounded deque; the Redis-backed
    implementation delegates to a Redis List.  Both expose the same API
    so TriggerBus uses this class regardless of transport.
    """

    def __init__(
        self,
        redis_client=None,
        maxlen: int = _MAX_DLQ,
    ) -> None:
        self._redis  = redis_client
        self._local: collections.deque = collections.deque(maxlen=maxlen)
        self._lock   = threading.Lock()

    def push(self, entry: _DLQEntry) -> None:
        """Append a failed dispatch to the DLQ."""
        if self._redis is not None:
            try:
                self._redis.rpush(_DLQ_KEY, json.dumps(entry.to_dict()))
                return
            except Exception as exc:
                log.warning("DLQ Redis push failed, falling back to local: %s", exc)
        with self._lock:
            self._local.append(entry)

    def size(self) -> int:
        if self._redis is not None:
            try:
                return int(self._redis.llen(_DLQ_KEY))
            except Exception:
                pass
        with self._lock:
            return len(self._local)

    def peek(self, n: int = 10) -> List[dict]:
        """Return the most recent n entries without removing them."""
        if self._redis is not None:
            try:
                items = self._redis.lrange(_DLQ_KEY, -n, -1)
                return [json.loads(b) for b in items]
            except Exception:
                pass
        with self._lock:
            entries = list(self._local)[-n:]
        return [e.to_dict() if isinstance(e, _DLQEntry) else e for e in entries]

    def pop_all(self) -> List[_DLQEntry]:
        """Drain the entire DLQ and return entries (for retry)."""
        entries: List[_DLQEntry] = []
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.lrange(_DLQ_KEY, 0, -1)
                pipe.delete(_DLQ_KEY)
                results, _ = pipe.execute()
                for raw in results:
                    d = json.loads(raw)
                    entries.append(_DLQEntry(**d))
                return entries
            except Exception as exc:
                log.warning("DLQ Redis pop_all failed: %s", exc)
        with self._lock:
            entries = list(self._local)
            self._local.clear()
        return entries

    def clear(self) -> None:
        if self._redis is not None:
            try:
                self._redis.delete(_DLQ_KEY)
                return
            except Exception:
                pass
        with self._lock:
            self._local.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Subscription registry
# ═══════════════════════════════════════════════════════════════════════════════

Handler = Callable[[dict, str, str], None]   # (payload_dict, channel, model_id) → None


@dataclass
class _Subscription:
    model_id: str
    channels: List[str]
    handler:  Handler


class SubscriptionRegistry:
    """
    Thread-safe registry of model subscriptions.

    Each model declares its channel interests at registration time.
    The registry answers: "which models care about channel X?" and
    "what handler should I call for model Y?".
    """

    def __init__(self) -> None:
        self._subs: Dict[str, _Subscription] = {}
        self._lock  = threading.RLock()

    def register(
        self,
        model_id: str,
        channels: List[str],
        handler: Handler,
    ) -> None:
        """Register or replace a model's subscription."""
        unknown = set(channels) - ALL_CHANNELS
        if unknown:
            log.warning(
                "SubscriptionRegistry: model %r subscribed to unknown channels %s",
                model_id, unknown,
            )
        with self._lock:
            self._subs[model_id] = _Subscription(
                model_id = model_id,
                channels = [c for c in channels if c in ALL_CHANNELS],
                handler  = handler,
            )
        log.debug("Registered %s → %s", model_id, channels)

    def unregister(self, model_id: str) -> None:
        with self._lock:
            self._subs.pop(model_id, None)

    def get(self, model_id: str) -> Optional[_Subscription]:
        with self._lock:
            return self._subs.get(model_id)

    def subscribers_for(self, channel: str) -> List[_Subscription]:
        """Return all subscriptions that include this channel."""
        with self._lock:
            return [s for s in self._subs.values() if channel in s.channels]

    def all_model_ids(self) -> List[str]:
        with self._lock:
            return list(self._subs.keys())

    def channels_for(self, model_id: str) -> List[str]:
        with self._lock:
            s = self._subs.get(model_id)
            return list(s.channels) if s else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._subs)


# ═══════════════════════════════════════════════════════════════════════════════
# Transport layer
# ═══════════════════════════════════════════════════════════════════════════════

class _Transport(ABC):
    """Abstract transport: publish bytes to a named channel, subscribe queues."""

    @abstractmethod
    def publish(self, channel: str, payload: bytes) -> None: ...

    @abstractmethod
    def subscribe(self, channel: str, q: "queue.Queue[bytes]") -> None: ...

    @abstractmethod
    def unsubscribe(self, channel: str, q: "queue.Queue[bytes]") -> None: ...

    @property
    @abstractmethod
    def backend(self) -> str: ...

    def close(self) -> None:
        pass


class _InProcessTransport(_Transport):
    """
    In-process fan-out transport using one queue.Queue per subscriber per channel.

    publish() puts the payload into every subscriber queue for that channel.
    This is synchronous but fast — the put() is non-blocking (put_nowait).
    If a queue is full, the message is dropped with a warning (back-pressure).
    """

    def __init__(self) -> None:
        # channel → list of subscriber queues
        self._ch_queues: Dict[str, List["queue.Queue[bytes]"]] = (
            collections.defaultdict(list)
        )
        self._lock = threading.Lock()

    def publish(self, channel: str, payload: bytes) -> None:
        with self._lock:
            targets = list(self._ch_queues.get(channel, []))
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                log.warning("In-process bus: queue full on channel %r — message dropped", channel)

    def subscribe(self, channel: str, q: "queue.Queue[bytes]") -> None:
        with self._lock:
            if q not in self._ch_queues[channel]:
                self._ch_queues[channel].append(q)

    def unsubscribe(self, channel: str, q: "queue.Queue[bytes]") -> None:
        with self._lock:
            qs = self._ch_queues.get(channel, [])
            if q in qs:
                qs.remove(q)

    def queue_depth(self, channel: str) -> int:
        """Total items across all subscriber queues for a channel."""
        with self._lock:
            return sum(q.qsize() for q in self._ch_queues.get(channel, []))

    @property
    def backend(self) -> str:
        return "in_process"


class _RedisTransport(_Transport):
    """
    Redis-backed transport using pub/sub for real-time delivery and a
    per-subscriber Redis List for reliable buffering.

    Publish:
      1. redis.PUBLISH channel payload       (real-time wake-up)
      2. redis.LPUSH  <model>:<channel> payload  (persistent buffer per subscriber)

    Subscribe:
      A background thread runs pubsub.listen(); on each message it calls
      put_nowait() into the subscriber's in-process queue so the consumer
      loop (TriggerBus.start_consuming) is shared with the in-process path.

    The Redis List acts as a replay buffer: if a consumer was offline, it
    can BRPOP from its personal list on next start.
    """

    def __init__(self, client) -> None:
        self._client   = client
        self._pubsub   = client.pubsub(ignore_subscribe_messages=True)
        # channel → list of (queue, model_id) pairs
        self._ch_subs:  Dict[str, List[Tuple["queue.Queue[bytes]", str]]] = (
            collections.defaultdict(list)
        )
        self._lock      = threading.Lock()
        self._listener  = threading.Thread(
            target=self._listen_loop, daemon=True, name="bus-redis-listener"
        )
        self._listener.start()

    def publish(self, channel: str, payload: bytes) -> None:
        self._client.publish(channel, payload)
        # Push to each subscriber's personal buffering list
        with self._lock:
            for _, model_id in self._ch_subs.get(channel, []):
                list_key = f"{_REDIS_KEY_PREFIX}:{model_id}:{channel}"
                self._client.lpush(list_key, payload)
                self._client.ltrim(list_key, 0, 999)   # keep at most 1000 items

    def subscribe(self, channel: str, q: "queue.Queue[bytes]", model_id: str = "") -> None:
        with self._lock:
            self._ch_subs[channel].append((q, model_id))
        self._pubsub.subscribe(channel)

    def unsubscribe(self, channel: str, q: "queue.Queue[bytes]") -> None:
        with self._lock:
            pairs = self._ch_subs.get(channel, [])
            self._ch_subs[channel] = [(qq, m) for qq, m in pairs if qq is not q]
        # Only unsubscribe from pubsub if no more listeners on this channel
        with self._lock:
            if not self._ch_subs.get(channel):
                self._pubsub.unsubscribe(channel)

    def _listen_loop(self) -> None:
        """Push incoming pub/sub messages into registered in-process queues."""
        try:
            for msg in self._pubsub.listen():
                if msg["type"] != "message":
                    continue
                channel  = msg["channel"].decode() if isinstance(msg["channel"], bytes) else msg["channel"]
                payload  = msg["data"]
                if isinstance(payload, str):
                    payload = payload.encode()
                with self._lock:
                    targets = list(self._ch_subs.get(channel, []))
                for q, _ in targets:
                    try:
                        q.put_nowait(payload)
                    except queue.Full:
                        log.warning("Redis bus: queue full on channel %r", channel)
        except Exception as exc:
            log.error("Redis bus listener exited: %s", exc)

    def replay_from_redis(self, model_id: str, channel: str, q: "queue.Queue[bytes]") -> int:
        """Drain the Redis buffer list for (model_id, channel) into the in-process queue."""
        list_key = f"{_REDIS_KEY_PREFIX}:{model_id}:{channel}"
        try:
            items = self._client.lrange(list_key, 0, -1)
            self._client.delete(list_key)
            for payload in reversed(items):   # oldest first
                q.put_nowait(payload)
            return len(items)
        except Exception as exc:
            log.warning("replay_from_redis %s/%s failed: %s", model_id, channel, exc)
            return 0

    @property
    def backend(self) -> str:
        return "redis"

    def close(self) -> None:
        try:
            self._pubsub.unsubscribe()
            self._pubsub.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Trigger Bus
# ═══════════════════════════════════════════════════════════════════════════════

class TriggerBus:
    """
    Pub/Sub Trigger Bus for TriggerSignal dispatch.

    Parameters
    ----------
    redis_url : str, optional
        Redis connection URL (default: REDIS_URL env var or redis://localhost:6379).
        Pass ``None`` or an empty string to force the in-process transport.
    registry : SubscriptionRegistry, optional
        Subscription registry.  A fresh one is created if omitted.
    max_queue_size : int
        Per-subscriber in-process queue capacity (default 1000).
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        registry:  Optional[SubscriptionRegistry] = None,
        max_queue_size: int = 1000,
    ) -> None:
        self._sub_registry   = registry or SubscriptionRegistry()
        self._max_queue      = max_queue_size
        self._transport, self._redis_client = self._connect(redis_url)
        self._dlq            = DeadLetterQueue(self._redis_client)
        # model_id → {channel → queue.Queue}
        self._queues:  Dict[str, Dict[str, "queue.Queue[bytes]"]] = {}
        # model_id → threading.Event (stop signal)
        self._stop_flags: Dict[str, threading.Event] = {}
        # model_id → consumer thread
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        log.info("TriggerBus started — transport=%s", self._transport.backend)

    # ── Transport setup ────────────────────────────────────────────────────────

    @staticmethod
    def _connect(
        redis_url: Optional[str],
    ) -> Tuple["_Transport", Optional[object]]:
        """Try Redis; fall back to in-process silently."""
        if not _REDIS_AVAILABLE:
            log.info("redis-py not installed — using in-process transport")
            return _InProcessTransport(), None

        url = (
            redis_url
            or os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        if not url:
            return _InProcessTransport(), None

        try:
            client = _redis_mod.from_url(url, decode_responses=False, socket_timeout=2)
            client.ping()
            log.info("TriggerBus: connected to Redis at %s", url)
            return _RedisTransport(client), client
        except Exception as exc:
            log.info("TriggerBus: Redis unavailable (%s) — using in-process transport", exc)
            return _InProcessTransport(), None

    # ── Producer API ───────────────────────────────────────────────────────────

    def publish(self, signal) -> List[str]:
        """
        Publish a TriggerSignal to its typed channel(s).

        Returns the list of channel names the signal was sent to.
        An empty list means the trigger_type has no channel mapping (not an error).
        """
        channels = route_signal(signal)
        if not channels:
            return []

        for ch in channels:
            payload = _serialize(signal, channels, ch)
            try:
                self._transport.publish(ch, payload)
                log.debug("Published %s → %s", signal.trigger_type, ch)
            except Exception as exc:
                log.error("Bus publish failed on channel %r: %s", ch, exc)
                self._dlq.push(_DLQEntry(
                    payload     = {"trigger_type": signal.trigger_type,
                                   "signal_id": signal.signal_id},
                    channel     = ch,
                    model_id    = "_publisher",
                    error_class = type(exc).__name__,
                    error_msg   = str(exc),
                    failed_at   = datetime.now(timezone.utc).isoformat(),
                    retry_count = 0,
                ))

        return channels

    def publish_raw(self, channel: str, data: dict) -> None:
        """
        Publish an arbitrary dict to a channel (e.g. for test injection or
        replaying a DLQ entry without a full TriggerSignal object).
        """
        if channel not in ALL_CHANNELS:
            raise ValueError(
                f"publish_raw: unknown channel {channel!r}. "
                f"Valid channels: {sorted(ALL_CHANNELS)}"
            )
        payload = json.dumps(data, separators=(",", ":")).encode()
        self._transport.publish(channel, payload)

    # ── Consumer API ──────────────────────────────────────────────────────────

    def subscribe(
        self,
        model_id:  str,
        handler:   Handler,
        channels:  Optional[List[str]] = None,
    ) -> None:
        """
        Register a model as a subscriber.

        Parameters
        ----------
        model_id : str
            Unique model identifier (matches model_targets in TriggerConfig).
        handler : callable
            Called as ``handler(payload_dict, channel, model_id)`` for each message.
            Must be thread-safe (called from the consumer thread).
        channels : list[str], optional
            Channels to subscribe to.  Defaults to DEFAULT_SUBSCRIPTIONS[model_id]
            if present, else all channels.
        """
        if channels is None:
            channels = DEFAULT_SUBSCRIPTIONS.get(model_id, list(ALL_CHANNELS))

        self._sub_registry.register(model_id, channels, handler)

        with self._lock:
            self._queues.setdefault(model_id, {})
            for ch in channels:
                if ch not in self._queues[model_id]:
                    q: "queue.Queue[bytes]" = queue.Queue(maxsize=self._max_queue)
                    self._queues[model_id][ch] = q
                    if isinstance(self._transport, _RedisTransport):
                        self._transport.subscribe(ch, q, model_id)
                        # Drain any messages buffered while offline
                        self._transport.replay_from_redis(model_id, ch, q)
                    else:
                        self._transport.subscribe(ch, q)

        log.info("Subscribed %s to channels %s", model_id, channels)

    def unsubscribe(self, model_id: str) -> None:
        """Remove a model's subscription and stop its consumer thread."""
        self.stop_consuming(model_id)
        sub = self._sub_registry.get(model_id)
        if sub:
            with self._lock:
                for ch, q in self._queues.get(model_id, {}).items():
                    self._transport.unsubscribe(ch, q)
                self._queues.pop(model_id, None)
        self._sub_registry.unregister(model_id)
        log.info("Unsubscribed %s", model_id)

    def start_consuming(self, model_id: str) -> threading.Thread:
        """
        Spawn a daemon consumer thread for model_id.

        The thread runs until stop_consuming() is called or the process exits.
        Raises RuntimeError if model_id has no subscription yet.
        """
        sub = self._sub_registry.get(model_id)
        if sub is None:
            raise RuntimeError(
                f"start_consuming: {model_id!r} has no subscription. "
                "Call subscribe() first."
            )

        if model_id in self._threads and self._threads[model_id].is_alive():
            log.debug("start_consuming: %s already running", model_id)
            return self._threads[model_id]

        stop = threading.Event()
        self._stop_flags[model_id] = stop

        queues_snapshot = dict(self._queues.get(model_id, {}))
        handler = sub.handler

        def _loop() -> None:
            log.info("Consumer started: %s", model_id)
            while not stop.is_set():
                for ch, q in queues_snapshot.items():
                    try:
                        payload = q.get(timeout=_CONSUMER_TIMEOUT)
                    except queue.Empty:
                        continue
                    try:
                        data = json.loads(payload)
                        handler(data, ch, model_id)
                        q.task_done()
                    except Exception as exc:
                        log.warning(
                            "Handler error in %s on channel %r: %s",
                            model_id, ch, exc,
                        )
                        self._dlq.push(_DLQEntry(
                            payload     = json.loads(payload) if payload else {},
                            channel     = ch,
                            model_id    = model_id,
                            error_class = type(exc).__name__,
                            error_msg   = str(exc),
                            failed_at   = datetime.now(timezone.utc).isoformat(),
                            retry_count = 0,
                        ))
                        q.task_done()
            log.info("Consumer stopped: %s", model_id)

        t = threading.Thread(target=_loop, daemon=True, name=f"bus-{model_id}")
        t.start()
        with self._lock:
            self._threads[model_id] = t
        return t

    def stop_consuming(self, model_id: str) -> None:
        """Signal the consumer thread for model_id to exit and wait for it."""
        flag = self._stop_flags.get(model_id)
        if flag:
            flag.set()
        t = self._threads.get(model_id)
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._stop_flags.pop(model_id, None)
        self._threads.pop(model_id, None)

    # ── Dead-letter queue API ─────────────────────────────────────────────────

    def dlq_size(self) -> int:
        return self._dlq.size()

    def dlq_peek(self, n: int = 10) -> List[dict]:
        return self._dlq.peek(n)

    def dlq_retry(self, max_retries: int = _MAX_RETRIES) -> int:
        """
        Re-dispatch all DLQ entries.  Entries that fail again are re-queued
        with retry_count incremented; entries that exceed max_retries are
        discarded with a warning.

        Returns the number of entries successfully re-dispatched.
        """
        entries  = self._dlq.pop_all()
        retried  = 0

        for entry in entries:
            if entry.retry_count >= max_retries:
                log.warning(
                    "DLQ: discarding %s/%s after %d retries",
                    entry.model_id, entry.channel, entry.retry_count,
                )
                continue

            sub = self._sub_registry.get(entry.model_id)
            if sub is None:
                log.warning("DLQ: no subscriber for %r — discarding", entry.model_id)
                continue

            try:
                sub.handler(entry.payload, entry.channel, entry.model_id)
                retried += 1
            except Exception as exc:
                entry.retry_count += 1
                entry.failed_at   = datetime.now(timezone.utc).isoformat()
                entry.error_class = type(exc).__name__
                entry.error_msg   = str(exc)
                self._dlq.push(entry)
                log.warning(
                    "DLQ retry #%d failed for %s/%s: %s",
                    entry.retry_count, entry.model_id, entry.channel, exc,
                )

        return retried

    def dlq_clear(self) -> None:
        self._dlq.clear()

    # ── Admin ──────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return a diagnostic snapshot of bus state."""
        subs     = self._sub_registry.all_model_ids()
        per_sub  = {m: self._sub_registry.channels_for(m) for m in subs}
        q_depths: Dict[str, int] = {}
        with self._lock:
            for model_id, ch_map in self._queues.items():
                for ch, q in ch_map.items():
                    q_depths[f"{model_id}:{ch}"] = q.qsize()

        return {
            "backend":           self._transport.backend,
            "subscriptions":     per_sub,
            "n_subscribers":     len(subs),
            "consumer_threads":  {
                m: t.is_alive()
                for m, t in self._threads.items()
            },
            "queue_depths":      q_depths,
            "dlq_size":          self._dlq.size(),
            "all_channels":      sorted(ALL_CHANNELS),
        }

    def close(self) -> None:
        """Stop all consumer threads and close the transport."""
        for model_id in list(self._threads.keys()):
            self.stop_consuming(model_id)
        self._transport.close()
        log.info("TriggerBus closed")


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton + convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════

BUS: TriggerBus = TriggerBus()


def publish_signal(signal) -> List[str]:
    """Publish a TriggerSignal using the module-level singleton BUS."""
    return BUS.publish(signal)


def subscribe_model(
    model_id: str,
    handler:  Handler,
    channels: Optional[List[str]] = None,
) -> None:
    """Subscribe a model using the module-level singleton BUS."""
    BUS.subscribe(model_id, handler, channels)
