"""
Database connection manager.

HOW THIS WORKS:
  This module is the single source of truth for "how do we connect to the DB?"
  Everything else imports `get_engine()` or `get_session()` from here — no
  other file ever constructs a connection string.

THE ENGINE vs THE SESSION:
  These are SQLAlchemy's two core concepts, and they're easy to confuse:

  Engine  — the low-level connection pool to the database file/server.
             Think of it as the physical cable between Python and the DB.
             You create it once and reuse it for the whole program lifetime.

  Session — a unit of work. You open a session, do reads/writes inside it,
             then commit (save) or rollback (undo). Think of it like a
             shopping cart: you can add/remove items freely, and only when
             you call checkout() (commit) do changes actually land in the DB.

DATABASE URL FORMAT:
  SQLite  →  sqlite:///path/to/file.db     (3 slashes = relative path)
             sqlite:////absolute/path.db   (4 slashes = absolute path)
  PostgreSQL → postgresql://user:pass@host:5432/dbname

  To switch from SQLite to PostgreSQL, change one line in .env:
    DATABASE_URL=postgresql://postgres:yourpassword@localhost/commodities
  Nothing else in the codebase needs to change.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

# ── Resolve database URL ───────────────────────────────────────────────────────
# Default: SQLite file sitting inside the project directory.
# Override by setting DATABASE_URL in your .env file.
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_SQLITE = f"sqlite:///{_PROJECT_ROOT / 'data' / 'commodities.db'}"

DATABASE_URL = os.getenv("DATABASE_URL", _DEFAULT_SQLITE)

# ── Engine ─────────────────────────────────────────────────────────────────────
# `check_same_thread=False` is a SQLite-specific flag that allows the engine to
# be shared across threads (Streamlit runs in a multi-threaded context).
# It's ignored for PostgreSQL.
def get_engine():
    connect_args = {}
    if DATABASE_URL.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Make sure the data/ directory exists before SQLite tries to create the file
        db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=False,   # Set True to print every SQL statement (useful for debugging)
    )


# ── Session factory ────────────────────────────────────────────────────────────
# `sessionmaker` creates a reusable factory. Calling SessionLocal() gives you
# a new Session bound to the engine — like opening a new "transaction window".
def get_session_factory(engine=None):
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Context manager helper ─────────────────────────────────────────────────────
# Usage:
#   with get_db() as db:
#       db.add(some_object)
#       db.commit()
#
# The `with` block guarantees the session is closed even if an exception occurs.
from contextlib import contextmanager

@contextmanager
def get_db():
    engine  = get_engine()
    Session = get_session_factory(engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Table creation ─────────────────────────────────────────────────────────────
def init_db():
    """
    Create all tables defined in models.py if they don't already exist.
    Safe to call multiple times — SQLAlchemy uses CREATE TABLE IF NOT EXISTS.
    """
    from database.models import Base
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def db_info() -> dict:
    """Return basic stats about the current database for display purposes."""
    from database.models import Commodity, PriceHistory
    from sqlalchemy import func
    with get_db() as db:
        n_commodities = db.query(Commodity).count()
        n_price_rows  = db.query(PriceHistory).count()
        oldest_date = db.query(func.min(PriceHistory.date)).scalar()
        newest_date = db.query(func.max(PriceHistory.date)).scalar()
    return {
        "url":         DATABASE_URL,
        "commodities": n_commodities,
        "price_rows":  n_price_rows,
        "oldest_date": oldest_date,
        "newest_date": newest_date,
    }


def staleness_info() -> dict:
    """
    Return data freshness information for the price_history table.

    days_stale is computed as calendar days between today and the newest stored
    price date. Values to interpret:
      0–1  → current (today's or yesterday's close is stored)
      2–3  → borderline (weekend or US holiday gap — scheduler may still be ok)
      4+   → stale — the ingestion pipeline has likely missed at least one run

    Note on equity/ETF proxies: these trade on NYSE hours, which differ from
    NYMEX futures (e.g., Juneteenth, MLK Day). On those days futures data
    updates but proxy data does not. A staleness of 1–2 days for proxies only
    is expected on those dates; check per_type for detail.
    """
    from database.models import Commodity, PriceHistory
    from sqlalchemy import func
    from datetime import date

    today = date.today()

    with get_db() as db:
        newest_overall = db.query(func.max(PriceHistory.date)).scalar()

        # Per instrument_type newest date — reveals if only some types are stale
        per_type_rows = (
            db.query(Commodity.instrument_type, func.max(PriceHistory.date))
            .join(PriceHistory, Commodity.id == PriceHistory.commodity_id)
            .group_by(Commodity.instrument_type)
            .all()
        )

    if newest_overall is None:
        return {"newest_date": None, "days_stale": None, "per_type": {}}

    days_stale = (today - newest_overall).days
    per_type   = {
        (t or "unknown"): (today - d).days
        for t, d in per_type_rows
        if d is not None
    }

    return {
        "newest_date": newest_overall,
        "days_stale":  days_stale,
        "per_type":    per_type,
    }
