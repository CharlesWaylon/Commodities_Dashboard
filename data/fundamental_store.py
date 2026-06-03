"""
Point-in-time fundamental store — the anti-look-ahead access layer for
release-dated data (COT / EIA / USDA / FRED).

The whole point: a signal computing as-of date ``t`` must see a fundamental datum
ONLY if it had been published by ``t``. So every read goes through ``get_asof(t)``,
which filters on ``release_date <= t`` (NOT reference_date). When a series is
revised, each vintage is a separate row (same reference_date, later release_date);
``get_asof`` returns the latest vintage available as of ``t`` — exactly what a PM
would have seen on that date.

Writes are idempotent upserts keyed on
(source, series_id, reference_date, release_date), so re-running an ingest never
duplicates rows.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS


def write_observations(df: pd.DataFrame, default_unit: Optional[str] = None) -> int:
    """
    Idempotently upsert observation rows into ``fundamental_observations``.

    Parameters
    ----------
    df : DataFrame with at least OBSERVATION_COLUMNS
        (source, series_id, reference_date, release_date, value). Optional extra
        columns ``unit``, ``instrument``, ``meta_json`` are persisted when present.

    Returns
    -------
    int — number of rows inserted or updated.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from database.db import get_engine, init_db
    from database.models import FundamentalObservation

    if df is None or df.empty:
        return 0
    missing = [c for c in OBSERVATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"write_observations: missing required columns {missing}")

    init_db()  # idempotent CREATE TABLE IF NOT EXISTS
    now = datetime.now(timezone.utc).isoformat()
    engine = get_engine()

    records = []
    for _, r in df.iterrows():
        records.append(
            {
                "source": str(r["source"]),
                "series_id": str(r["series_id"]),
                "reference_date": pd.Timestamp(r["reference_date"]).date(),
                "release_date": pd.Timestamp(r["release_date"]).date(),
                "value": float(r["value"]),
                "unit": (r["unit"] if "unit" in df.columns and pd.notna(r.get("unit")) else default_unit),
                "instrument": (r["instrument"] if "instrument" in df.columns and pd.notna(r.get("instrument")) else None),
                "meta_json": (r["meta_json"] if "meta_json" in df.columns and pd.notna(r.get("meta_json")) else None),
                "inserted_at": now,
            }
        )

    table = FundamentalObservation.__table__
    is_pg = engine.dialect.name == "postgresql"
    n = 0
    with engine.begin() as conn:
        for rec in records:
            if is_pg:
                stmt = pg_insert(table).values(**rec)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_fundobs_source_series_ref_release",
                    set_={"value": rec["value"], "unit": rec["unit"],
                          "instrument": rec["instrument"], "meta_json": rec["meta_json"],
                          "inserted_at": rec["inserted_at"]},
                )
                conn.execute(stmt)
            else:
                # SQLite / other: delete-then-insert on the unique key (test path).
                conn.execute(
                    table.delete().where(
                        (table.c.source == rec["source"])
                        & (table.c.series_id == rec["series_id"])
                        & (table.c.reference_date == rec["reference_date"])
                        & (table.c.release_date == rec["release_date"])
                    )
                )
                conn.execute(table.insert().values(**rec))
            n += 1
    return n


def get_asof(
    asof: date,
    series_ids: Optional[Iterable[str]] = None,
    source: Optional[str] = None,
    instrument: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the point-in-time view of fundamentals as known on ``asof``.

    Only rows with ``release_date <= asof`` are returned. For each
    (series_id, reference_date) the LATEST available vintage (max release_date
    <= asof) is kept — i.e. exactly the data a decision-maker had on ``asof``.

    Returns a DataFrame [source, series_id, reference_date, release_date, value,
    unit, instrument], sorted by (series_id, reference_date).
    """
    from sqlalchemy import select

    from database.db import get_engine
    from database.models import FundamentalObservation as F

    asof = pd.Timestamp(asof).date()
    cols = [F.source, F.series_id, F.reference_date, F.release_date, F.value, F.unit, F.instrument]
    stmt = select(*cols).where(F.release_date <= asof)
    if series_ids is not None:
        stmt = stmt.where(F.series_id.in_(list(series_ids)))
    if source is not None:
        stmt = stmt.where(F.source == source)
    if instrument is not None:
        stmt = stmt.where(F.instrument == instrument)

    engine = get_engine()
    with engine.connect() as conn:
        rows = pd.DataFrame(conn.execute(stmt).fetchall(),
                            columns=["source", "series_id", "reference_date",
                                     "release_date", "value", "unit", "instrument"])
    if rows.empty:
        return rows

    rows["reference_date"] = pd.to_datetime(rows["reference_date"])
    rows["release_date"] = pd.to_datetime(rows["release_date"])
    # Latest vintage per (series_id, reference_date): max release_date <= asof.
    rows = (
        rows.sort_values("release_date")
        .groupby(["series_id", "reference_date"], as_index=False)
        .tail(1)
        .sort_values(["series_id", "reference_date"])
        .reset_index(drop=True)
    )
    return rows


def latest_series(asof: date, series_id: str) -> pd.Series:
    """Convenience: one series as a reference_date-indexed pd.Series, PIT as-of ``asof``."""
    df = get_asof(asof, series_ids=[series_id])
    if df.empty:
        return pd.Series(dtype=float, name=series_id)
    s = df.set_index("reference_date")["value"].sort_index()
    s.name = series_id
    return s
