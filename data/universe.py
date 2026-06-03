"""
Canonical instrument registry — ONE source of truth for "what instruments exist
and what do we know about each one?"

WHY THIS EXISTS
───────────────
Instrument metadata was scattered: the name→ticker map in
``models.config.MODELING_COMMODITIES``, the sector map in
``models.config.COMMODITY_SECTORS``, and the futures contract cycle in
``config.futures_calendar.FUTURES_CONTRACT_SPECS``. A worldwide-investor product
also needs currency, exchange, and timezone per instrument (for market-hours
awareness and as-of labelling). This module merges all of that into one typed
registry so every layer points HERE.

This is the home the MODEL SCOPE RULE refers to: the full 40-instrument modeling
universe. ``models.config.MODELING_COMMODITIES`` remains the raw name→ticker dict
(this registry is BUILT from it, so they can never drift), but new code should
prefer ``data.universe``.

POINT-IN-TIME / WORLDWIDE-INVESTOR FIELDS
──────────────────────────────────────────
``exchange`` + ``timezone`` let the presentation layer show market-hours-aware,
timezone-correct "last updated" labels. ``has_term_structure`` tells the carry /
curve signals which instruments actually have a futures curve (futures) vs which
are spot-like ETFs/equities/crypto (no real term structure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.futures_calendar import FUTURES_CONTRACT_SPECS
from models.config import COMMODITY_SECTORS, MODELING_COMMODITIES

# Exchange code (Yahoo suffix / our convention) → IANA timezone.
_EXCHANGE_TZ: Dict[str, str] = {
    "NYM": "America/New_York",   # NYMEX
    "CMX": "America/New_York",   # COMEX
    "NYB": "America/New_York",   # ICE / NYBOT softs
    "CBT": "America/Chicago",    # CBOT grains
    "CME": "America/Chicago",    # CME livestock
    "US_EQUITY": "America/New_York",
    "CRYPTO": "UTC",
}

# A handful of ETF/equity proxies whose listing venue we pin explicitly; everything
# else falls back to the generic US_EQUITY / America/New_York.
_PROXY_EXCHANGE: Dict[str, str] = {
    # ticker : exchange code (all map to America/New_York via US_EQUITY tz)
}


@dataclass(frozen=True)
class Instrument:
    """Everything the system knows about one tradable instrument."""

    display_name: str
    ticker: str                       # continuous Yahoo ticker == commodities.ticker (stable join key)
    sector: str                       # energy | metals | agriculture | livestock | digital
    kind: str                         # "futures" | "etf_or_equity" | "crypto"
    has_term_structure: bool          # True only for genuine futures
    currency: str                     # ISO 4217, e.g. "USD"
    exchange: str                     # venue code (see _EXCHANGE_TZ)
    timezone: str                     # IANA tz for market-hours / as-of labelling
    contract_months: Optional[Tuple[int, ...]] = None  # listed delivery months (futures only)
    roll_offset_bdays: Optional[int] = None            # see config.futures_calendar.ContractSpec

    @property
    def is_futures(self) -> bool:
        return self.kind == "futures"


def _classify_kind(name: str, ticker: str) -> str:
    if name in FUTURES_CONTRACT_SPECS:
        return "futures"
    if ticker.endswith("-USD"):
        return "crypto"
    return "etf_or_equity"


def _build_registry() -> Dict[str, Instrument]:
    registry: Dict[str, Instrument] = {}
    for name, ticker in MODELING_COMMODITIES.items():
        kind = _classify_kind(name, ticker)
        sector = COMMODITY_SECTORS.get(name, "unknown")

        if kind == "futures":
            spec = FUTURES_CONTRACT_SPECS[name]
            exchange = spec.exchange
            registry[name] = Instrument(
                display_name=name,
                ticker=ticker,
                sector=sector,
                kind=kind,
                has_term_structure=True,
                currency="USD",
                exchange=exchange,
                timezone=_EXCHANGE_TZ.get(exchange, "America/New_York"),
                contract_months=tuple(spec.months),
                roll_offset_bdays=spec.roll_offset_bdays,
            )
        elif kind == "crypto":
            registry[name] = Instrument(
                display_name=name, ticker=ticker, sector=sector, kind=kind,
                has_term_structure=False, currency="USD",
                exchange="CRYPTO", timezone=_EXCHANGE_TZ["CRYPTO"],
            )
        else:  # etf_or_equity
            exch = _PROXY_EXCHANGE.get(ticker, "US_EQUITY")
            registry[name] = Instrument(
                display_name=name, ticker=ticker, sector=sector, kind=kind,
                has_term_structure=False, currency="USD",
                exchange=exch, timezone=_EXCHANGE_TZ.get(exch, "America/New_York"),
            )
    return registry


# The registry itself — built once at import.
INSTRUMENTS: Dict[str, Instrument] = _build_registry()


# ── Accessors ────────────────────────────────────────────────────────────────
def get(display_name: str) -> Instrument:
    """Look up one instrument by display name. Raises KeyError if unknown."""
    return INSTRUMENTS[display_name]


def all_instruments() -> List[Instrument]:
    return list(INSTRUMENTS.values())


def display_names() -> List[str]:
    return list(INSTRUMENTS.keys())


def tickers() -> List[str]:
    return [i.ticker for i in INSTRUMENTS.values()]


def by_sector(sector: str) -> List[Instrument]:
    return [i for i in INSTRUMENTS.values() if i.sector == sector]


def sectors() -> List[str]:
    seen: List[str] = []
    for i in INSTRUMENTS.values():
        if i.sector not in seen:
            seen.append(i.sector)
    return seen


def futures() -> List[Instrument]:
    """Instruments with a genuine term structure (the carry/curve-eligible set)."""
    return [i for i in INSTRUMENTS.values() if i.has_term_structure]


def name_for_ticker(ticker: str) -> Optional[str]:
    for i in INSTRUMENTS.values():
        if i.ticker == ticker:
            return i.display_name
    return None


def as_mapping() -> Dict[str, str]:
    """Back-compat helper: the {display_name: ticker} dict (== MODELING_COMMODITIES)."""
    return {i.display_name: i.ticker for i in INSTRUMENTS.values()}
