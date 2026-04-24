"""
One-time migration script: SQLite → PostgreSQL.

HOW THIS WORKS:
  1. Opens two SQLAlchemy engines — one for SQLite (source), one for PostgreSQL (destination)
  2. Creates all tables in PostgreSQL using the same ORM models
  3. Reads every commodity from SQLite and inserts it into PostgreSQL
  4. Reads price_history in batches of 1,000 rows (to avoid loading 52,000 rows into RAM at once)
  5. Inserts each batch into PostgreSQL
  6. Prints a summary

Run once from the project root:
  python -m pipeline.migrate_to_postgres

Prerequisites:
  - Postgres.app is running
  - `createdb commodities` has been run
  - `pip install psycopg2-binary` has been run
  - DATABASE_URL=postgresql://localhost/commodities is set in .env
"""

import os
import sys
import logging
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).parent.parent
SQLITE_URL    = f"sqlite:///{PROJECT_ROOT / 'data' / 'commodities.db'}"
POSTGRES_URL  = os.getenv("DATABASE_URL", "")

BATCH_SIZE = 1000   # rows per INSERT batch — keeps memory usage low


def run_migration():
    if not POSTGRES_URL or not POSTGRES_URL.startswith("postgresql"):
        log.error("DATABASE_URL is not set to a PostgreSQL URL.")
        log.error("Add DATABASE_URL=postgresql://localhost/commodities to your .env file.")
        sys.exit(1)

    # ── Connect to both databases ──────────────────────────────────────────────
    log.info(f"Source:      {SQLITE_URL}")
    log.info(f"Destination: {POSTGRES_URL}")

    sqlite_engine   = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    postgres_engine = create_engine(POSTGRES_URL)

    # Verify PostgreSQL is reachable before doing any work
    try:
        with postgres_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("PostgreSQL connection verified.")
    except Exception as e:
        log.error(f"Cannot connect to PostgreSQL: {e}")
        log.error("Make sure Postgres.app is running and 'createdb commodities' was run.")
        sys.exit(1)

    # ── Create tables in PostgreSQL ────────────────────────────────────────────
    # Uses the same ORM models — generates the correct PostgreSQL DDL automatically.
    from database.models import Base, Commodity, PriceHistory
    Base.metadata.create_all(postgres_engine)
    log.info("PostgreSQL tables created (or already exist).")

    # ── Session factories ──────────────────────────────────────────────────────
    SQLiteSession   = sessionmaker(bind=sqlite_engine)
    PostgresSession = sessionmaker(bind=postgres_engine)

    sqlite_db   = SQLiteSession()
    postgres_db = PostgresSession()

    try:
        # ── Migrate commodities table ──────────────────────────────────────────
        log.info("Migrating commodities reference table...")
        commodities = sqlite_db.query(Commodity).all()

        for c in commodities:
            existing = postgres_db.query(Commodity).filter_by(ticker=c.ticker).first()
            if existing:
                log.info(f"  Already exists: {c.name} — skipping")
                continue
            postgres_db.add(Commodity(
                id     = c.id,      # Preserve IDs so FK references stay valid
                name   = c.name,
                ticker = c.ticker,
                sector = c.sector,
                unit   = c.unit,
            ))

        postgres_db.commit()
        pg_count = postgres_db.query(Commodity).count()
        log.info(f"  Commodities in PostgreSQL: {pg_count}")

        # ── Migrate price_history in batches ───────────────────────────────────
        total_sqlite = sqlite_db.query(PriceHistory).count()
        log.info(f"Migrating {total_sqlite:,} price rows in batches of {BATCH_SIZE}...")

        offset    = 0
        inserted  = 0
        skipped   = 0

        while True:
            batch = (
                sqlite_db.query(PriceHistory)
                .order_by(PriceHistory.id)
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )
            if not batch:
                break

            for row in batch:
                existing = postgres_db.query(PriceHistory).filter_by(
                    commodity_id = row.commodity_id,
                    date         = row.date,
                    interval     = row.interval,
                ).first()

                if existing:
                    skipped += 1
                    continue

                postgres_db.add(PriceHistory(
                    commodity_id = row.commodity_id,
                    date         = row.date,
                    open         = row.open,
                    high         = row.high,
                    low          = row.low,
                    close        = row.close,
                    volume       = row.volume,
                    interval     = row.interval,
                    ingested_at  = row.ingested_at,
                ))
                inserted += 1

            postgres_db.commit()
            offset += BATCH_SIZE
            log.info(f"  Progress: {min(offset, total_sqlite):,} / {total_sqlite:,} rows processed")

        log.info("=" * 60)
        log.info(f"Migration complete.")
        log.info(f"  Rows inserted: {inserted:,}")
        log.info(f"  Rows skipped:  {skipped:,}")

        # ── Sync sequences ─────────────────────────────────────────────────────
        # PostgreSQL uses sequences to auto-generate IDs. After bulk-inserting rows
        # with explicit IDs, the sequence is still at 1. Next INSERT would fail with
        # a duplicate key error. This resets the sequence to max(id) + 1.
        log.info("Syncing PostgreSQL ID sequences...")
        with postgres_engine.connect() as conn:
            conn.execute(text(
                "SELECT setval('commodities_id_seq', (SELECT MAX(id) FROM commodities))"
            ))
            conn.execute(text(
                "SELECT setval('price_history_id_seq', (SELECT MAX(id) FROM price_history))"
            ))
            conn.commit()
        log.info("Sequences synced.")
        log.info("=" * 60)
        log.info("Your PostgreSQL database is ready.")
        log.info("Update .env:  DATABASE_URL=postgresql://localhost/commodities")

    except Exception as e:
        postgres_db.rollback()
        log.error(f"Migration failed: {e}")
        raise
    finally:
        sqlite_db.close()
        postgres_db.close()


if __name__ == "__main__":
    run_migration()
