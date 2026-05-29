"""
Standalone macro ingestion daemon.

Runs MacroIngestionService outside Streamlit so the feed survives dashboard
restarts and a fresh `streamlit run app.py` does not double-start pollers.

Usage:
    python -m pipeline.run_macro_feed

Stop with SIGINT (Ctrl+C) or SIGTERM (e.g. from launchd / systemd).
"""

import argparse
import errno
import fcntl
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing the service so MacroIngestionService.from_env()
# sees FRED_API_KEY / ALPHA_VANTAGE_KEY when launched outside Streamlit.
load_dotenv()

from services.macro_ingestion import MacroIngestionService

# Single-instance lock. A second daemon would run its own rate limiter against
# the same Alpha Vantage key and collectively blow the 25/day cap, so we refuse
# to start if another instance already holds this lock. The flock is released
# automatically when the holding process exits — including SIGKILL — so there
# is never a stale lock to clean up.
_LOCK_PATH = Path(os.getenv("MACRO_FEED_LOCK_PATH", "logs/macro_feed.lock"))


def _acquire_single_instance_lock(log: logging.Logger):
    """
    Try to take an exclusive, non-blocking lock. Returns the open file object
    (keep a reference for the process lifetime) on success, or None if another
    instance already holds it.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            fh.close()
            return None
        raise
    # Record our PID for humans inspecting the lock file. The lock itself is
    # what matters; the PID is just a convenience.
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def main() -> int:
    # Send INFO logs to stdout; only WARNING+ goes to stderr. Under launchd
    # this routes routine activity to macro_feed.log and real problems to
    # macro_feed.error.log instead of dumping everything into the error file.
    fmt = logging.Formatter("%(asctime)s %(name)-24s %(levelname)s %(message)s")

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(logging.INFO)
    stdout_h.addFilter(lambda r: r.levelno < logging.WARNING)
    stdout_h.setFormatter(fmt)

    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setLevel(logging.WARNING)
    stderr_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(stdout_h)
    root.addHandler(stderr_h)

    log = logging.getLogger("macro_feed")

    parser = argparse.ArgumentParser(description="Macro Data Ingestion Daemon")
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=7,
        help="On startup, seed each poller's last-seen cache from the most "
             "recent trigger_events row per family within the last N days. "
             "Prevents restart-induced duplicate floods. Set to 0 to disable.",
    )
    args = parser.parse_args()

    # Refuse to start a second daemon — prevents overlapping API calls that
    # would exhaust the shared Alpha Vantage daily cap. Hold `lock_fh` for the
    # whole process lifetime so the flock stays held.
    lock_fh = _acquire_single_instance_lock(log)
    if lock_fh is None:
        log.error(
            "Another macro feed daemon already holds %s — refusing to start a "
            "second instance (would double-spend the Alpha Vantage daily cap). "
            "Stop the other process first, or check `pgrep -fl run_macro_feed`.",
            _LOCK_PATH,
        )
        return 1

    service = MacroIngestionService.from_env()
    if args.backfill_days > 0:
        service.backfill(days=args.backfill_days)
    service.start()

    if not service.is_running():
        log.error(
            "No pollers started. Check FRED_API_KEY and ALPHA_VANTAGE_KEY env vars."
        )
        return 1

    log.info("MacroIngestionService running. Streaming events (SIGINT/SIGTERM to stop).")

    def _shutdown(sig, _frame):
        log.info("Signal %s received — stopping macro feed.", sig)
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for event in service.queue.stream():
        log.info(
            "[%s] %-20s %-30s actual=%r expected=%r dev=%+.2fσ [%s]",
            event.release_timestamp[:16],
            event.source,
            event.event_type,
            event.actual_value,
            event.expected_value,
            event.deviation_score,
            event.impact,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
