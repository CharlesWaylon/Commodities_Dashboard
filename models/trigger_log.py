"""
Trigger event log — persist every detected TriggerEvent to SQLite.

Purpose
-------
1. Phase 5 backtest harness will read this log to ask: "Did routed signals
   actually outperform during trigger windows?"
2. Audit trail: when did the dashboard claim 'Fed tightening' was active,
   and at what strength?
3. Researcher debugging: did my custom trigger fire last week as expected?

Schema (data/commodities.db / table `trigger_events`)
-----------------------------------------------------
  id                    INTEGER PRIMARY KEY
  detected_at           TEXT       ISO 8601, when detection ran
  trigger_date          TEXT       YYYY-MM-DD, derived; dedup key
  family                TEXT       trigger family name
  strength              REAL       [0, 1]
  rationale             TEXT       human-readable reason
  affected_commodities  TEXT       JSON array of display names
  metadata              TEXT       JSON object (free-form)
  inserted_at           TEXT       UTC ISO 8601, write time
  UNIQUE(family, trigger_date)

Dedup policy
------------
INSERT OR REPLACE keyed on (family, trigger_date). Re-firing the same
trigger on the same day overwrites the prior row with the latest values.
This keeps the log to one row per family per day — the granularity the
backtest harness needs.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from models.triggers import TriggerEvent

# Default DB path — same SQLite file as the rest of the dashboard
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "commodities.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trigger_events (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at          TEXT NOT NULL,
    trigger_date         TEXT NOT NULL,
    family               TEXT NOT NULL,
    strength             REAL NOT NULL,
    rationale            TEXT,
    affected_commodities TEXT,
    metadata             TEXT,
    inserted_at          TEXT NOT NULL,
    UNIQUE(family, trigger_date)
);
CREATE INDEX IF NOT EXISTS idx_trigger_events_family_date
    ON trigger_events(family, trigger_date);
"""


def _connect(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Open a connection, ensure parent dir + schema exist."""
    p = Path(db_path) if db_path else DEFAULT_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(_SCHEMA_SQL)
    return conn


def log_trigger_events(
    events: List[TriggerEvent],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    Persist a list of trigger events. Re-firing a (family, trigger_date) pair
    overwrites the prior entry with the latest values.

    Returns
    -------
    int
        Number of rows written (= len(events) for non-empty inputs).
    """
    if not events:
        return 0

    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for ev in events:
        # Derive trigger_date from the ISO timestamp (first 10 chars: YYYY-MM-DD)
        try:
            trigger_date = ev.detected_at[:10]
            # Sanity check: parseable
            datetime.fromisoformat(trigger_date)
        except Exception:
            trigger_date = datetime.now(timezone.utc).date().isoformat()

        rows.append((
            ev.detected_at,
            trigger_date,
            ev.family,
            float(ev.strength),
            ev.rationale or "",
            json.dumps(list(ev.affected_commodities or [])),
            json.dumps(dict(ev.metadata or {})),
            now_iso,
        ))

    sql = """
    INSERT OR REPLACE INTO trigger_events
        (detected_at, trigger_date, family, strength, rationale,
         affected_commodities, metadata, inserted_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """

    with _connect(db_path) as conn:
        conn.executemany(sql, rows)
        conn.commit()

    return len(rows)


def recent_trigger_events(
    days: int = 30,
    family: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Read recent trigger events from the log.

    Parameters
    ----------
    days : int
        Look-back window in days from today.
    family : str, optional
        Restrict to a single trigger family.

    Returns
    -------
    pd.DataFrame
        Columns include: detected_at, trigger_date, family, strength,
        rationale, affected_commodities (parsed list), metadata (parsed
        dict), inserted_at. Empty DataFrame if no rows match.
    """
    cutoff = (
        pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)
    ).date().isoformat()

    sql = "SELECT * FROM trigger_events WHERE trigger_date >= ?"
    params: List = [cutoff]
    if family is not None:
        sql += " AND family = ?"
        params.append(family)
    sql += " ORDER BY trigger_date DESC, family ASC"

    with _connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if not df.empty:
        df["affected_commodities"] = df["affected_commodities"].apply(
            lambda s: json.loads(s) if s else []
        )
        df["metadata"] = df["metadata"].apply(
            lambda s: json.loads(s) if s else {}
        )
    return df
