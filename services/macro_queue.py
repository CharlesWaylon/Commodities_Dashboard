"""
Thread-safe in-memory queue for MacroEvent objects.

Optionally persists every event to a JSONL file so events survive restarts
and can be replayed for backtesting or debugging.

Durability contract for the persistence sink:
  * Each put() writes exactly one JSON object followed by '\n'.
  * The line is flushed to the kernel and fsync()'d to disk before put()
    returns, so SIGKILL (`kill -9`) of the daemon cannot lose an event
    that the pollers already logged.
  * When the file grows past MAX_BYTES it is rotated: the current file is
    renamed to '<name>.1' (replacing any prior rotation) and a fresh
    empty file is opened. Rotation is atomic on POSIX (os.replace).
"""

import json
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional


# 50 MB cap before we roll the JSONL sink to .1
MAX_BYTES = 50 * 1024 * 1024


class MacroQueue:
    """
    Thread-safe wrapper around queue.Queue with a durable JSONL sink.

    put()   — enqueue a MacroEvent (also writes+fsyncs one line to JSONL if persist_path set)
    get()   — dequeue one event (raises queue.Empty on timeout)
    stream()— generator that yields events indefinitely, blocking between arrivals
    replay_from_file() — read all persisted events from the JSONL sink
    """

    def __init__(self, maxsize: int = 10_000, persist_path: Optional[str] = None):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._persist_path = Path(persist_path) if persist_path else None
        self._write_lock = threading.Lock()

        if self._persist_path:
            # Directory-create guard: missing logs/ must not crash first run.
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, event) -> None:
        self._q.put(event)
        if self._persist_path:
            line = json.dumps(asdict(event)) + "\n"
            with self._write_lock:
                self._rotate_if_needed(extra_bytes=len(line.encode("utf-8")))
                # Append-mode + explicit flush + fsync guarantees the line is
                # on disk before put() returns. kill -9 after this point
                # cannot lose the event.
                with open(self._persist_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())

    def _rotate_if_needed(self, extra_bytes: int = 0) -> None:
        """Roll persist_path → persist_path.1 once the file would exceed MAX_BYTES."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            size = self._persist_path.stat().st_size
        except OSError:
            return
        if size + extra_bytes <= MAX_BYTES:
            return
        rotated = self._persist_path.with_suffix(self._persist_path.suffix + ".1")
        # os.replace is atomic on POSIX and overwrites any existing .1 file.
        os.replace(self._persist_path, rotated)

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
