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


def test_usda_quarterly_reference_from_end_code():
    # The four quarterly position reads must land on distinct first-of-month dates,
    # not collapse onto year-end.
    raw = [
        {"year": "2025", "end_code": "03", "Value": "8,147,437,000", "load_time": "2025-03-31 12:00:00.000"},
        {"year": "2025", "end_code": "06", "Value": "4,642,894,000", "load_time": "2025-06-30 12:00:00.000"},
        {"year": "2025", "end_code": "09", "Value": "1,551,286,000", "load_time": "2025-09-30 12:00:00.000"},
        {"year": "2025", "end_code": "12", "Value": "13,305,825,000", "load_time": "2026-01-12 12:00:00.000"},
    ]
    df = UsdaAdapter()._shape("CORN_GRAIN_STOCKS", raw).sort_values("reference_date")
    assert list(df["reference_date"]) == [date(2025, 3, 1), date(2025, 6, 1),
                                          date(2025, 9, 1), date(2025, 12, 1)]
    # release_date comes from the real load_time, not a fixed lag.
    assert list(df["release_date"])[0] == date(2025, 3, 31)
    assert list(df["release_date"])[-1] == date(2026, 1, 12)
    assert df.iloc[0]["value"] == 8147437000.0  # commas stripped


def test_usda_falls_back_to_year_end_and_lag_without_codes():
    raw = [{"year": "2025", "Value": "1,540,000"}]  # no end_code, no load_time
    row = UsdaAdapter()._shape("X", raw).iloc[0]
    assert row["reference_date"] == date(2025, 12, 31)
    assert row["release_date"] == date(2026, 1, 30)  # +30d default lag


def test_usda_skips_suppressed_values():
    raw = [{"year": "2025", "Value": "(D)"}, {"year": None, "Value": "5"}]
    assert UsdaAdapter()._shape("CORN_GRAIN_STOCKS", raw).empty


def test_adapters_return_empty_frame_on_no_data():
    for shaped in (
        CftcAdapter()._shape([]),
        EiaAdapter()._shape("X", []),
        UsdaAdapter()._shape("X", []),
    ):
        assert isinstance(shaped, pd.DataFrame)
        assert shaped.empty
        assert list(shaped.columns) == OBSERVATION_COLUMNS
