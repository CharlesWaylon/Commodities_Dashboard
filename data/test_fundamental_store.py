"""
Point-in-time correctness of the fundamental store: get_asof must hide rows that
were not yet released, and must return the latest available vintage of revised
data — exactly what a decision-maker saw on that date.

Uses a dedicated throwaway source name so it never touches real ingested data,
and cleans up after itself.
"""

from datetime import date

import pandas as pd
import pytest

from data import fundamental_store as store
from data.adapters.base import OBSERVATION_COLUMNS

_TEST_SOURCE = "_pytest_fundobs"


@pytest.fixture
def clean_test_rows():
    yield
    # teardown: remove anything this test wrote
    from database.db import get_engine
    from database.models import FundamentalObservation as F

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(F.__table__.delete().where(F.__table__.c.source == _TEST_SOURCE))


def _obs(series_id, ref, rel, value):
    return {
        "source": _TEST_SOURCE,
        "series_id": series_id,
        "reference_date": pd.Timestamp(ref).date(),
        "release_date": pd.Timestamp(rel).date(),
        "value": float(value),
    }


def test_get_asof_hides_unreleased_rows(clean_test_rows):
    df = pd.DataFrame(
        [
            _obs("X", "2026-01-06", "2026-01-09", 100.0),   # released Jan 9
            _obs("X", "2026-01-13", "2026-01-16", 110.0),   # released Jan 16
        ],
        columns=OBSERVATION_COLUMNS,
    )
    store.write_observations(df)

    # As of Jan 12: only the Jan-9 release is visible.
    asof12 = store.get_asof(date(2026, 1, 12), series_ids=["X"], source=_TEST_SOURCE)
    assert list(asof12["value"]) == [100.0]

    # As of Jan 20: both releases visible.
    asof20 = store.get_asof(date(2026, 1, 20), series_ids=["X"], source=_TEST_SOURCE)
    assert sorted(asof20["value"]) == [100.0, 110.0]


def test_get_asof_returns_latest_vintage(clean_test_rows):
    # Same reference_date, two vintages (an initial print + a later revision).
    df = pd.DataFrame(
        [
            _obs("Y", "2026-02-03", "2026-02-05", 50.0),   # first print
            _obs("Y", "2026-02-03", "2026-02-19", 55.0),   # revision two weeks later
        ],
        columns=OBSERVATION_COLUMNS,
    )
    store.write_observations(df)

    # Before the revision: see the first print.
    early = store.get_asof(date(2026, 2, 10), series_ids=["Y"], source=_TEST_SOURCE)
    assert list(early["value"]) == [50.0]

    # After the revision: see the revised value (latest vintage), not duplicated.
    late = store.get_asof(date(2026, 2, 25), series_ids=["Y"], source=_TEST_SOURCE)
    assert list(late["value"]) == [55.0]
    assert len(late) == 1


def test_write_is_idempotent(clean_test_rows):
    df = pd.DataFrame([_obs("Z", "2026-03-02", "2026-03-04", 7.0)], columns=OBSERVATION_COLUMNS)
    store.write_observations(df)
    store.write_observations(df)  # second write must not duplicate
    out = store.get_asof(date(2026, 3, 10), series_ids=["Z"], source=_TEST_SOURCE)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 7.0
