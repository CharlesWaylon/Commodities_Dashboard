"""
End-to-end smoke test for the macro feed.

Covers the full path:
    FREDPoller.poll()  →  MacroEvent  →  MacroIngestionService._maybe_write_db()
                       →  trigger_events row in (in-memory SQLite) DB.

The point is to catch wiring regressions: if any of the three stages stops
emitting / classifying / writing, this test fails fast (target: < 5 s).

Run:
    pytest services/test_macro_feed_e2e.py -v
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_session_factory(monkeypatch):
    """
    Build an in-memory SQLite DB with the project's schema and patch
    `database.db.get_db` so anything calling it gets a session bound to this DB.
    Yields the SessionLocal factory so the test can run its own assertions
    against the same engine.
    """
    from database.models import Base

    engine        = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal  = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    @contextmanager
    def fake_get_db():
        s = SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # _maybe_write_db does `from database.db import get_db` inside the function,
    # so patching the module-level attr is what gets resolved at call time.
    import database.db as ddb
    monkeypatch.setattr(ddb, "get_db", fake_get_db)

    return SessionLocal


@pytest.fixture
def fake_fred_requests(monkeypatch):
    """
    Monkeypatch services.macro_ingestion.requests.get so a FRED call for the
    CPIAUCSL series returns one fresh observation; every other series returns
    an empty observation list (skipped by the poller).
    """
    today    = datetime.now(timezone.utc).date()
    fresh_d  = today.isoformat()

    class FakeResponse:
        def __init__(self, payload: dict, status: int = 200):
            self._payload = payload
            self.status_code = status
            self.text = json.dumps(payload)
        def raise_for_status(self): pass
        def json(self): return self._payload

    def fake_get(url, params=None, timeout=None, **kw):
        params = params or {}
        sid = params.get("series_id", "")
        if sid == "CPIAUCSL":
            return FakeResponse({
                "observations": [
                    # Stable baseline so std is small and the latest reads as a
                    # clear surprise — keeps the test deterministic.
                    {"date": (today - timedelta(days=120)).isoformat(), "value": "310.0"},
                    {"date": (today - timedelta(days=90)).isoformat(),  "value": "310.5"},
                    {"date": (today - timedelta(days=60)).isoformat(),  "value": "311.0"},
                    {"date": (today - timedelta(days=30)).isoformat(),  "value": "311.2"},
                    {"date": (today - timedelta(days=15)).isoformat(),  "value": "311.5"},
                    {"date": fresh_d,                                    "value": "315.0"},
                ],
            })
        # All other FRED series: empty → poller skips them.
        return FakeResponse({"observations": []})

    import services.macro_ingestion as mi
    monkeypatch.setattr(mi.requests, "get", fake_get)
    return fresh_d


# ── The test ──────────────────────────────────────────────────────────────────

def test_fred_poll_writes_one_cpi_row(sqlite_session_factory, fake_fred_requests):
    from database.models import TriggerEvent
    from services.macro_ingestion import MacroIngestionService

    svc = MacroIngestionService(fred_api_key="x", av_api_key="")

    # Pre-seed the rolling baseline. The poller only appends the *latest*
    # observation per call, so on a cold start its history is empty and
    # deviation_score is 0. Production gets around this by polling many times;
    # the test inlines the warm-up so a single poll() yields a real z-score.
    svc._fred._history["CPIAUCSL"] = [310.0, 310.5, 311.0, 311.2, 311.5]

    # (2) Run the poller — should emit exactly one MacroEvent (CPI).
    events = svc._fred.poll()
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events}"
    evt = events[0]
    assert evt.event_type == "CPI"
    assert evt.impact == "high"
    assert abs(evt.deviation_score) >= 0.5, (
        f"deviation_score={evt.deviation_score} — too small to clear the gate; "
        "the fixture's baseline/surprise math drifted"
    )

    # (3) Drive the DB-write path with that event.
    svc._maybe_write_db(evt)

    # (4) Exactly one row, family=CPI, strength in (0, 1].
    SessionLocal = sqlite_session_factory
    with SessionLocal() as s:
        rows = s.query(TriggerEvent).filter_by(family="CPI").all()
    assert len(rows) == 1, f"expected 1 trigger_events row, got {len(rows)}"
    row = rows[0]
    assert row.family == "CPI"
    assert 0.0 < row.strength <= 1.0, f"strength={row.strength} out of (0, 1]"
    assert row.trigger_date == fake_fred_requests       # matches today's date
    # Sanity: metadata round-trip preserves source + deviation.
    meta = json.loads(row.trigger_metadata)
    assert meta["source"] == "FRED"
    assert abs(meta["deviation_score"] - evt.deviation_score) < 1e-9


def test_repeat_poll_does_not_duplicate_row(sqlite_session_factory, fake_fred_requests):
    """Same-day re-fire should upsert on (family, trigger_date), not duplicate."""
    from database.models import TriggerEvent
    from services.macro_ingestion import MacroIngestionService

    svc = MacroIngestionService(fred_api_key="x", av_api_key="")
    svc._fred._history["CPIAUCSL"] = [310.0, 310.5, 311.0, 311.2, 311.5]
    evt = svc._fred.poll()[0]
    svc._maybe_write_db(evt)
    svc._maybe_write_db(evt)   # second call: should update, not insert

    SessionLocal = sqlite_session_factory
    with SessionLocal() as s:
        rows = s.query(TriggerEvent).filter_by(family="CPI").all()
    assert len(rows) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
