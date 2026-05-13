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
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy import text

from database.db import get_engine
from models.triggers import TriggerEvent


def log_trigger_events(
    events: List[TriggerEvent],
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

        rows.append({
            "detected_at":          ev.detected_at,
            "trigger_date":         trigger_date,
            "family":               ev.family,
            "strength":             float(ev.strength),
            "rationale":            ev.rationale or "",
            "affected_commodities": json.dumps(list(ev.affected_commodities or [])),
            "metadata":             json.dumps(dict(ev.metadata or {})),
            "inserted_at":          now_iso,
        })

    sql = text("""
    INSERT INTO trigger_events
        (detected_at, trigger_date, family, strength, rationale,
         affected_commodities, metadata, inserted_at)
    VALUES (:detected_at, :trigger_date, :family, :strength, :rationale,
            :affected_commodities, :metadata, :inserted_at)
    ON CONFLICT (family, trigger_date) DO UPDATE SET
        detected_at          = EXCLUDED.detected_at,
        strength             = EXCLUDED.strength,
        rationale            = EXCLUDED.rationale,
        affected_commodities = EXCLUDED.affected_commodities,
        metadata             = EXCLUDED.metadata,
        inserted_at          = EXCLUDED.inserted_at
    """)

    with get_engine().connect() as conn:
        conn.execute(sql, rows)
        conn.commit()

    return len(rows)


def recent_trigger_events(
    days: int = 30,
    family: Optional[str] = None,
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

    params: dict = {"cutoff": cutoff}
    sql = "SELECT * FROM trigger_events WHERE trigger_date >= :cutoff"
    if family is not None:
        sql += " AND family = :family"
        params["family"] = family
    sql += " ORDER BY trigger_date DESC, family ASC"

    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    if not df.empty:
        df["affected_commodities"] = df["affected_commodities"].apply(
            lambda s: json.loads(s) if s else []
        )
        df["metadata"] = df["metadata"].apply(
            lambda s: json.loads(s) if s else {}
        )
    return df
