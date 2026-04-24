"""
Ingestion scheduler — runs the pipeline automatically on a timer.

HOW THIS WORKS:
  APScheduler (Advanced Python Scheduler) lets you run Python functions
  on a recurring schedule without needing cron or any external tool.
  You start this script once, and it runs forever in the background,
  waking up at the configured interval to pull fresh data.

  Two job types are configured:
    1. DAILY job  — runs once per day at 18:00 UTC (after US markets close)
                    pulls the full day's final OHLCV data
    2. STARTUP job — runs once immediately when you start the scheduler,
                    so you don't wait until 18:00 for the first data pull

HOW TO RUN:
  In a terminal (separate from the Streamlit dashboard):
    python -m pipeline.scheduler

  It will print a log line every time a job fires.
  Press Ctrl+C to stop.

WHEN TO USE A REAL CRON INSTEAD:
  APScheduler is great for development.  In production (e.g. a server that
  must survive reboots), use a system cron job:
    # Edit crontab:  crontab -e
    # Run ingestion every day at 18:05 UTC:
    5 18 * * 1-5 cd /path/to/Commodities_Dashboard && python -m pipeline.ingest
"""

import logging
import signal
import sys
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from pipeline.ingest import run_ingestion
from database.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _startup_ingest():
    """Runs once at scheduler start — incremental pull to catch up."""
    log.info("[Scheduler] Running startup ingestion...")
    run_ingestion(backfill=False)


def _daily_ingest():
    """Runs on the configured daily schedule."""
    log.info("[Scheduler] Running daily ingestion...")
    run_ingestion(backfill=False)


def start_scheduler(
    daily_hour: int = 18,
    daily_minute: int = 5,
    run_on_start: bool = True,
):
    """
    Start the background scheduler.

    Args:
        daily_hour:    UTC hour to run the daily job (default 18 = 6pm UTC / ~2pm ET)
        daily_minute:  minute offset (default 5 = a few minutes after market close)
        run_on_start:  if True, also run one ingestion immediately on startup
    """
    init_db()
    scheduler = BackgroundScheduler(timezone="UTC")

    # Daily job: fires at HH:MM UTC, Monday–Friday
    scheduler.add_job(
        _daily_ingest,
        trigger=CronTrigger(day_of_week="mon-fri", hour=daily_hour, minute=daily_minute),
        id="daily_ingest",
        name="Daily commodity price ingestion",
        replace_existing=True,
    )

    scheduler.start()
    log.info(f"[Scheduler] Started. Daily ingestion at {daily_hour:02d}:{daily_minute:02d} UTC (Mon–Fri).")

    if run_on_start:
        # Run immediately in a separate thread so the scheduler loop isn't blocked
        import threading
        t = threading.Thread(target=_startup_ingest, daemon=True)
        t.start()

    # Graceful shutdown on Ctrl+C or SIGTERM
    def _shutdown(sig, frame):
        log.info("[Scheduler] Shutting down...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("[Scheduler] Running. Press Ctrl+C to stop.")
    while True:
        time.sleep(60)   # Main thread just keeps the process alive


if __name__ == "__main__":
    start_scheduler()
