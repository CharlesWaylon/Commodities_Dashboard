"""
Contract-calendar resolver — pure, deterministic, no network / no DB.

Given a commodity's ContractSpec (config/futures_calendar.py) this module answers:

    "On date D, which listed contract is the M1 (front)?  Which is the M2
     (second-nearby)?  What is its dated Yahoo ticker?"

and the inverse used by the ingest:

    "To stitch a constant M1/M2 series across [start, end], which dated
     contracts do I need to download?"

DESIGN
──────
A contract for delivery month M is treated as a *candidate* nearby until its
``roll_date`` (an approximation of expiry — see ``approx_roll_date``).  On any
date D the active candidates are every contract with ``roll_date >= D``, sorted
by (year, month).  The first is M1, the second is M2, and so on.  The day after
a contract's roll_date it drops out and everything shifts up by one — i.e. the
roll happens automatically and M1/M2 stay genuinely adjacent listed contracts.

This is the property the basis feature needs: ``log(M1 / M2)`` is always a
near-term calendar spread, never a fixed far-dated contract masquerading as M2.

Because it is pure Python over dates, every function here is unit-tested with
synthetic inputs in pipeline/test_contract_calendar.py (runs in the sandbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from config.futures_calendar import (
    ContractSpec,
    FUTURES_CONTRACT_SPECS,
    MONTH_CODES,
)


@dataclass(frozen=True)
class Contract:
    """A single dated futures contract resolved from a ContractSpec."""
    commodity: str
    year: int
    month: int
    ticker: str       # dated Yahoo ticker, e.g. "CLQ26.NYM"
    roll_date: date   # last date this contract counts as a nearby candidate

    @property
    def delivery_first(self) -> date:
        return date(self.year, self.month, 1)


# ── Ticker construction ─────────────────────────────────────────────────────────

def contract_ticker(spec: ContractSpec, year: int, month: int) -> str:
    """Build a dated Yahoo ticker, e.g. ('CL','NYM',2026,8) → 'CLQ26.NYM'."""
    return f"{spec.root}{MONTH_CODES[month]}{year % 100:02d}.{spec.exchange}"


# ── Roll-date approximation ─────────────────────────────────────────────────────

def approx_roll_date(spec: ContractSpec, year: int, month: int) -> date:
    """
    Approximate the date a contract stops being a nearby candidate (its roll-off).

    Computed as ``roll_offset_bdays`` business days *before* the first calendar
    day of the delivery month:

      positive offset → roll before the delivery month (energy: front expires
                        ~20th of the prior month)
      negative offset → roll into the delivery month (grains/metals/livestock
                        expire mid-to-late delivery month)

    This is intentionally an approximation (see config/futures_calendar.py). The
    exact day matters little — what matters is that M1/M2 stay adjacent.
    """
    first = pd.Timestamp(date(year, month, 1))
    rolled = first - pd.tseries.offsets.BDay(spec.roll_offset_bdays)
    return rolled.date()


# ── Listed-contract enumeration ─────────────────────────────────────────────────

def listed_contracts(
    spec: ContractSpec,
    commodity: str,
    start: date,
    end: date,
) -> list[Contract]:
    """
    Every listed contract whose roll_date plausibly covers [start, end], plus a
    margin on each side so the M1/M2 lookup at the window edges always has enough
    forward candidates.  Sorted ascending by (year, month).
    """
    contracts: list[Contract] = []
    # Generous year span: one year back (contracts rolling off early in the window)
    # through three years forward (enough M2 depth past the window end).
    for yr in range(start.year - 1, end.year + 3):
        for mo in spec.months:
            contracts.append(
                Contract(
                    commodity=commodity,
                    year=yr,
                    month=mo,
                    ticker=contract_ticker(spec, yr, mo),
                    roll_date=approx_roll_date(spec, yr, mo),
                )
            )
    contracts.sort(key=lambda c: (c.year, c.month))
    return contracts


# ── Nearby resolution ───────────────────────────────────────────────────────────

def nearby_on(
    spec: ContractSpec,
    commodity: str,
    on_date: date,
    depth: int,
    contracts: Optional[list[Contract]] = None,
) -> Optional[Contract]:
    """
    Return the depth-th nearby contract active on ``on_date``.

    depth=1 → M1 (front), depth=2 → M2 (second-nearby), etc.
    Active candidates are those with ``roll_date >= on_date``; the day after a
    contract's roll_date it drops out and the next contract becomes M1.

    Returns None if there are fewer than ``depth`` active candidates (only near
    the far edge of the generated range).
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if contracts is None:
        contracts = listed_contracts(spec, commodity, on_date, on_date)

    active = [c for c in contracts if c.roll_date >= on_date]
    # `contracts` is already sorted; the filter preserves order.
    idx = depth - 1
    return active[idx] if idx < len(active) else None


def nearby_schedule(
    spec: ContractSpec,
    commodity: str,
    dates: Iterable[date],
    depth: int,
) -> dict[date, str]:
    """
    Map each date → the dated ticker of its depth-th nearby contract.

    Builds the contract list once and reuses it for all dates (fast).
    Dates with no resolvable contract are omitted.
    """
    dates = sorted(set(dates))
    if not dates:
        return {}
    contracts = listed_contracts(spec, commodity, dates[0], dates[-1])
    out: dict[date, str] = {}
    for d in dates:
        c = nearby_on(spec, commodity, d, depth, contracts)
        if c is not None:
            out[d] = c.ticker
    return out


def required_tickers(
    spec: ContractSpec,
    commodity: str,
    start: date,
    end: date,
    max_depth: int = 2,
    skip_rolled_before: Optional[date] = None,
) -> set[str]:
    """
    The set of dated tickers the ingest must download to stitch a constant
    M1…M(max_depth) series across [start, end].

    Walks every business day in the window and collects the resolved tickers for
    depths 1..max_depth.

    ``skip_rolled_before`` — if given, contracts whose ``roll_date`` is *before*
    this cutoff are omitted.  Yahoo only serves roughly the last ~12 months of
    history per dated contract; anything that rolled off earlier returns an empty
    404 ("possibly delisted").  Passing a cutoff (e.g. today − ~13 months) skips
    those doomed downloads entirely — fewer wasted requests AND no alarming
    ERROR-level 404 spam from yfinance.  Defaults to None (no filter) so the pure
    calendar semantics are preserved for the unit tests.
    """
    contracts = listed_contracts(spec, commodity, start, end)
    tickers: set[str] = set()
    for ts in pd.bdate_range(start, end):
        d = ts.date()
        for depth in range(1, max_depth + 1):
            c = nearby_on(spec, commodity, d, depth, contracts)
            if c is not None:
                if skip_rolled_before is not None and c.roll_date < skip_rolled_before:
                    continue
                tickers.add(c.ticker)
    return tickers


def spec_for(commodity: str) -> Optional[ContractSpec]:
    """Convenience lookup; None for ETF/proxy names not in the calendar."""
    return FUTURES_CONTRACT_SPECS.get(commodity)


def contract_code(ticker: str) -> str:
    """
    The exchange-less contract code used as the ``price_history.interval`` tag for
    a stored dated contract, e.g. 'CLQ26.NYM' → 'CLQ26'.  Stable across exchanges
    and short enough for the interval column (String(10)).
    """
    return ticker.split(".", 1)[0]


# Regex matching a stored contract-code interval, e.g. 'CLQ26', 'GCZ26', 'KEU26'.
# Used by stitch_m2.py to pick out contract rows from price_history (the '1d',
# '1d_m2', '1d_m1_raw', '1wk' intervals never match).
import re as _re  # noqa: E402

CONTRACT_CODE_RE = _re.compile(r"^[A-Z]{1,3}[FGHJKMNQUVXZ]\d{2}$")
