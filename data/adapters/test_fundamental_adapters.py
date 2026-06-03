"""
Network-free tests for the fundamental adapters: they exercise the pure ``_shape``
transforms with synthetic raw payloads, asserting the release-date (publication
lag) math that makes the data point-in-time correct. Live HTTP is never called.
"""

from datetime import date

import pandas as pd

from data.adapters.base import OBSERVATION_COLUMNS
from data.adapters.cftc_adapter import CftcAdapter
from data.adapters.eia_adapter import EiaAdapter
from data.adapters.usda_adapter import UsdaAdapter


def test_cftc_net_position_and_friday_release():
    raw = [
        {
            "cftc_contract_market_code": "067651",
            "report_date_as_yyyy_mm_dd": "2026-01-06T00:00:00.000",  # a Tuesday
            "m_money_positions_long_all": "200000",
            "m_money_positions_short_all": "50000",
        }
    ]
    df = CftcAdapter()._shape(raw)
    assert list(df.columns) == OBSERVATION_COLUMNS
    row = df.iloc[0]
    assert row["value"] == 150000.0  # long - short
    assert row["reference_date"] == date(2026, 1, 6)
    assert row["release_date"] == date(2026, 1, 9)  # +3 days -> Friday
    assert row["source"] == "cftc"


def test_cftc_skips_malformed_rows():
    raw = [
        {"cftc_contract_market_code": "067651"},  # missing date
        {"report_date_as_yyyy_mm_dd": "2026-01-06"},  # missing code
    ]
    assert CftcAdapter()._shape(raw).empty


def test_eia_petroleum_lag_default_5_days():
    raw = [{"period": "2026-01-09", "value": "420.5"}]  # week-ending Friday
    df = EiaAdapter()._shape("PET.WCESTUS1.W", raw)
    row = df.iloc[0]
    assert row["value"] == 420.5
    assert row["reference_date"] == date(2026, 1, 9)
    assert row["release_date"] == date(2026, 1, 14)  # +5 -> Wednesday


def test_eia_natgas_per_series_lag_6_days():
    adapter = EiaAdapter(per_series_lag_days={"NG.X.W": 6})
    raw = [{"period": "2026-01-09", "value": "3200"}]
    row = adapter._shape("NG.X.W", raw).iloc[0]
    assert row["release_date"] == date(2026, 1, 15)  # +6 -> Thursday


def test_eia_skips_missing_values():
    raw = [{"period": "2026-01-09", "value": "."}, {"period": None, "value": "1"}]
    assert EiaAdapter()._shape("PET.X.W", raw).empty


def test_usda_year_end_reference_and_lag():
    raw = [{"year": "2025", "Value": "1,540,000", "reference_period_desc": "MARKETING YEAR"}]
    row = UsdaAdapter()._shape("CORN_ENDING_STOCKS", raw).iloc[0]
    assert row["value"] == 1540000.0  # comma stripped
    assert row["reference_date"] == date(2025, 12, 31)
    assert row["release_date"] == date(2026, 1, 30)  # +30 days default


def test_usda_skips_suppressed_values():
    raw = [{"year": "2025", "Value": "(D)"}, {"year": None, "Value": "5"}]
    assert UsdaAdapter()._shape("CORN_ENDING_STOCKS", raw).empty


def test_adapters_return_empty_frame_on_no_data():
    for shaped in (
        CftcAdapter()._shape([]),
        EiaAdapter()._shape("X", []),
        UsdaAdapter()._shape("X", []),
    ):
        assert isinstance(shaped, pd.DataFrame)
        assert shaped.empty
        assert list(shaped.columns) == OBSERVATION_COLUMNS
