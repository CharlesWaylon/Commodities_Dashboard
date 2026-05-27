"""
Standalone macro ingestion daemon.

Runs MacroIngestionService outside Streamlit so the feed survives dashboard
restarts and a fresh `streamlit run app.py` does not double-start pollers.

Usage:
    python -m pipeline.run_macro_feed

Stop with SIGINT (Ctrl+C) or SIGTERM (e.g. from launchd / systemd).
"""

import argparse
import logging
import signal
import sys

from dotenv import load_dotenv

# Load .env before importing the service so MacroIngestionService.from_env()
# sees FRED_API_KEY / ALPHA_VANTAGE_KEY when launched outside Streamlit.
load_dotenv()

from services.macro_ingestion import MacroIngestionService


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
