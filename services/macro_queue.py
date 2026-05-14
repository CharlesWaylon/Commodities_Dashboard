"""
Thread-safe in-memory queue for MacroEvent objects.

Optionally persists every event to a JSONL file so events survive restarts
and can be replayed for backtesting or debugging.
"""

import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional


class MacroQueue:
    """
    Thread-safe wrapper around queue.Queue.

    put()   — enqueue a MacroEvent (also writes to JSONL if persist_path set)
    get()   — dequeue one event (raises queue.Empty on timeout)
    stream()— generator that yields events indefinitely, blocking between arrivals
    replay_from_file() — read all persisted events from the JSONL sink
    """

    def __init__(self, maxsize: int = 10_000, persist_path: Optional[str] = None):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._persist_path = Path(persist_path) if persist_path else None
        self._write_lock = threading.Lock()

        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, event) -> None:
        self._q.put(event)
        if self._persist_path:
            with self._write_lock:
                with open(self._persist_path, "a") as fh:
                    fh.write(json.dumps(asdict(event)) + "\n")

    def get(self, timeout: float = 1.0):
        return self._q.get(timeout=timeout)

    def stream(self, block_timeout: float = 0.2) -> Iterator:
        """Yield events indefinitely, blocking between arrivals."""
        while True:
            try:
                yield self._q.get(timeout=block_timeout)
            except queue.Empty:
                continue

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def replay_from_file(self) -> list:
        """Load all persisted events from the JSONL file. Returns list of dicts."""
        if not self._persist_path or not self._persist_path.exists():
            return []
        events = []
        with open(self._persist_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events
