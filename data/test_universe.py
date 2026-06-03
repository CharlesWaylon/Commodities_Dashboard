"""Tests for the canonical instrument registry."""

from config.futures_calendar import FUTURES_CONTRACT_SPECS
from data import universe
from models.config import COMMODITY_SECTORS, MODELING_COMMODITIES


def test_registry_covers_full_modeling_universe():
    # The registry must never silently drop an instrument — it IS the universe.
    assert set(universe.display_names()) == set(MODELING_COMMODITIES.keys())
    assert len(universe.all_instruments()) == len(MODELING_COMMODITIES)


def test_tickers_match_modeling_commodities():
    assert universe.as_mapping() == MODELING_COMMODITIES


def test_futures_have_term_structure_and_months():
    for name, spec in FUTURES_CONTRACT_SPECS.items():
        inst = universe.get(name)
        assert inst.is_futures
        assert inst.has_term_structure
        assert inst.contract_months == tuple(spec.months)
        assert inst.roll_offset_bdays == spec.roll_offset_bdays


def test_etfs_and_crypto_have_no_term_structure():
    btc = universe.get("Bitcoin")
    assert btc.kind == "crypto"
    assert not btc.has_term_structure
    assert btc.timezone == "UTC"
    # A starred ETF proxy
    carbon = universe.get("Carbon Credits*")
    assert carbon.kind == "etf_or_equity"
    assert not carbon.has_term_structure


def test_sector_assignment_matches_config():
    for name in MODELING_COMMODITIES:
        assert universe.get(name).sector == COMMODITY_SECTORS.get(name, "unknown")


def test_timezone_is_set_for_every_instrument():
    for inst in universe.all_instruments():
        assert inst.timezone  # non-empty IANA tz, needed for worldwide as-of labels
        assert inst.currency == "USD"


def test_name_for_ticker_roundtrip():
    assert universe.name_for_ticker("CL=F") == "WTI Crude Oil"
    assert universe.name_for_ticker("NOT_A_TICKER") is None
