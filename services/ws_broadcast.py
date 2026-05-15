"""
WebSocket Broadcast Server  (Mission 7)
=========================================
Subscribes to TriggerSignal events and pushes them in real time to all
connected WebSocket clients (the browser ribbon component).

Architecture
------------
  TriggerBus ──► broadcast(signal)
                       │
              asyncio loop (daemon thread)
                       │
              ┌────────┴─────────┐
              │  browser tab 1   │  ← WS client
              │  browser tab 2   │  ← WS client
              └──────────────────┘

Replay
------
The last ``MAX_REPLAY`` serialized payloads are buffered.  Freshly connected
clients receive the full replay immediately so the ribbon is never blank.

Public API
----------
  BROADCASTER              — module-level singleton
  broadcast_signal(signal) — convenience wrapper

  TriggerBroadcaster.start()          — idempotent; spawns daemon thread
  TriggerBroadcaster.broadcast(sig)   — thread-safe, call from anywhere
  TriggerBroadcaster.broadcast_raw(s) — send a pre-serialized JSON string
  TriggerBroadcaster.client_count     — number of connected WS clients
  TriggerBroadcaster.is_running       — True once background thread is live
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from typing import Optional, Set

log = logging.getLogger(__name__)

MAX_REPLAY   = 10
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765

try:
    import websockets
    from websockets.server import WebSocketServerProtocol as _WS
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    _WS = object  # type: ignore[assignment,misc]
    log.warning(
        "websockets package not installed — TriggerBroadcaster is a no-op. "
        "Run: pip install 'websockets>=12.0'"
    )


# ── Serialisation ──────────────────────────────────────────────────────────────

def serialize_signal(signal) -> str:
    """Serialise a TriggerSignal to a compact JSON string for wire transport."""
    return json.dumps({
        "signal_id":                signal.signal_id,
        "trigger_type":             signal.trigger_type,
        "display_name":             signal.config.display_name,
        "severity":                 signal.severity.label(),
        "magnitude_score":          round(signal.magnitude_score, 4),
        "direction":                signal.direction,
        "affected_clusters":        list(signal.affected_clusters),
        "affected_commodity_count": len(signal.affected_commodities),
        "cascade_priority":         signal.cascade_priority,
        "classified_at":            signal.classified_at,
        "decay_expires_at":         signal.decay_expires_at,
        "is_repeat":                signal.is_repeat,
        "classification_method":    signal.classification_method,
        "source":                   signal.config.source,
        "short_description":        signal.config.description[:100] if signal.config.description else "",
    })


# ── Broadcaster ────────────────────────────────────────────────────────────────

class TriggerBroadcaster:
    """
    WebSocket broadcast server for TriggerSignal events.

    All public methods are thread-safe.  The asyncio event loop lives in a
    dedicated daemon thread separate from Streamlit's main thread.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host    = host
        self._port    = port
        self._clients: Set[_WS] = set()
        self._replay:  deque[str] = deque(maxlen=MAX_REPLAY)
        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._thread:  Optional[threading.Thread] = None
        self._ready    = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background WebSocket server thread (idempotent)."""
        if not _WS_AVAILABLE:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="WS-Broadcast"
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the event loop (best-effort; thread is a daemon so it exits anyway)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread = None

    # ── Broadcast ─────────────────────────────────────────────────────────────

    def broadcast(self, signal) -> None:
        """Serialise a TriggerSignal and push it to all connected clients."""
        if not _WS_AVAILABLE or self._loop is None:
            return
        payload = serialize_signal(signal)
        self._enqueue(payload)

    def broadcast_raw(self, payload: str) -> None:
        """Push a pre-serialised JSON string (useful for test/demo events)."""
        if not _WS_AVAILABLE or self._loop is None:
            return
        self._enqueue(payload)

    def _enqueue(self, payload: str) -> None:
        self._replay.append(payload)
        asyncio.run_coroutine_threadsafe(self._send_all(payload), self._loop)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ws_url(self) -> str:
        return f"ws://{self._host}:{self._port}"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except RuntimeError:
            pass  # loop stopped via stop()

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handler, self._host, self._port,
            ping_interval=20, ping_timeout=10,
        ):
            log.info("WS broadcast server → ws://%s:%d", self._host, self._port)
            self._ready.set()
            await asyncio.Future()  # run until loop is stopped

    async def _handler(self, ws: _WS) -> None:
        self._clients.add(ws)
        log.debug("WS client connected  (total=%d)", len(self._clients))
        try:
            # Replay buffered events to the newly connected client
            for payload in list(self._replay):
                try:
                    await ws.send(payload)
                except Exception:
                    break
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            log.debug("WS client disconnected (total=%d)", len(self._clients))

    async def _send_all(self, payload: str) -> None:
        dead: Set[_WS] = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead


# ── Module-level singleton ─────────────────────────────────────────────────────

BROADCASTER = TriggerBroadcaster()


def broadcast_signal(signal) -> None:
    """Convenience wrapper — broadcasts via the module-level BROADCASTER."""
    BROADCASTER.broadcast(signal)
