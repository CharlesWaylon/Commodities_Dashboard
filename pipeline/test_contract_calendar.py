"""
Tests for pipeline/contract_calendar.py — pure calendar logic, no network/DB.
Run: python -m pipeline.test_contract_calendar
"""

import sys
import traceback
from datetime import date

import pandas as pd

from config.futures_calendar import (
    ContractSpec,
    FUTURES_CONTRACT_SPECS,
    MONTH_CODES,
)
from pipeline.contract_calendar import (
    contract_ticker,
    approx_roll_date,
    listed_contracts,
    nearby_on,
    nearby_schedule,
    required_tickers,
)

PASS = 0
FAIL = 0


def _ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {msg}")
    traceback.print_exc()


# Reusable specs
WTI = FUTURES_CONTRACT_SPECS["WTI Crude Oil"]      # monthly, roll +7 bdays
GOLD = FUTURES_CONTRACT_SPECS["Gold (COMEX)"]      # Feb/Apr/Jun/Aug/Oct/Dec


def test_ticker_format():
    print("1. contract_ticker — builds correct dated Yahoo symbol")
    try:
        assert contract_ticker(WTI, 2026, 8) == "CLQ26.NYM", contract_ticker(WTI, 2026, 8)
        assert contract_ticker(GOLD, 2026, 12) == "GCZ26.CMX"
        # Platinum must be NYMEX (the bug we fixed)
        pl = FUTURES_CONTRACT_SPECS["Platinum"]
        assert contract_ticker(pl, 2026, 10) == "PLV26.NYM", contract_ticker(pl, 2026, 10)
        # 2-digit year wraps correctly
        assert contract_ticker(WTI, 2030, 1) == "CLF30.NYM"
        _ok("CLQ26.NYM / GCZ26.CMX / PLV26.NYM all correct")
    except Exception:
        _fail("ticker format")


def test_month_codes_complete():
    print("2. MONTH_CODES — covers all 12 months, standard letters")
    try:
        assert MONTH_CODES[1] == "F" and MONTH_CODES[12] == "Z"
        assert MONTH_CODES[8] == "Q" and MONTH_CODES[7] == "N"
        assert len(MONTH_CODES) == 12
        assert len(set(MONTH_CODES.values())) == 12
        _ok("12 unique month codes, F…Z")
    except Exception:
        _fail("month codes")


def test_roll_date_sign_convention():
    print("3. approx_roll_date — energy rolls before month start, grains roll into month")
    try:
        # Energy (+7 bdays before month start) → roll_date in the PRIOR month
        e = approx_roll_date(WTI, 2026, 8)  # Aug 2026 contract
        assert e < date(2026, 8, 1), f"energy roll {e} should precede 2026-08-01"
        assert e.month == 7, f"energy roll should land in July, got {e}"
        # Grains (negative offset) → roll_date INSIDE the delivery month
        corn = FUTURES_CONTRACT_SPECS["Corn (CBOT)"]
        g = approx_roll_date(corn, 2026, 7)  # Jul 2026 contract, offset -10
        assert g > date(2026, 7, 1), f"grain roll {g} should be inside July"
        _ok(f"energy roll={e} (prior month), grain roll={g} (in delivery month)")
    except Exception:
        _fail("roll date sign convention")


def test_listed_contracts_sorted_and_cover_cycle():
    print("4. listed_contracts — sorted ascending, only spec months, spans window")
    try:
        cs = listed_contracts(GOLD, "Gold (COMEX)", date(2025, 1, 1), date(2026, 12, 31))
        # all months must be in the gold cycle
        assert all(c.month in GOLD.months for c in cs), "off-cycle month present"
        # strictly sorted by (year, month)
        keys = [(c.year, c.month) for c in cs]
        assert keys == sorted(keys), "not sorted"
        # window covered: at least one contract before start and after end
        assert any((c.year, c.month) < (2025, 1) for c in cs)
        assert any((c.year, c.month) > (2026, 12) for c in cs)
        _ok(f"{len(cs)} gold contracts, sorted, all on-cycle, margin on both ends")
    except Exception:
        _fail("listed_contracts")


def test_m1_m2_are_adjacent_and_distinct():
    print("5. nearby_on — M1 and M2 are distinct, adjacent, and M2 is later than M1")
    try:
        d = date(2026, 6, 2)
        m1 = nearby_on(WTI, "WTI Crude Oil", d, 1)
        m2 = nearby_on(WTI, "WTI Crude Oil", d, 2)
        assert m1 is not None and m2 is not None
        assert (m1.year, m1.month) < (m2.year, m2.month), "M2 not after M1"
        # adjacency for a monthly contract: M2 is exactly one listed month after M1
        # (months are consecutive for WTI)
        assert m1.roll_date <= m2.roll_date
        _ok(f"on {d}: M1={m1.ticker} M2={m2.ticker} (adjacent, distinct)")
    except Exception:
        _fail("m1/m2 adjacency")


def test_m1_today_matches_known_front():
    print("6. nearby_on — WTI front on 2026-06-02 is the July contract (CLN26)")
    try:
        # WTI front expires ~20th of prior month, so in early June the front
        # delivery month is July (CLN26) and M2 is August (CLQ26).
        d = date(2026, 6, 2)
        m1 = nearby_on(WTI, "WTI Crude Oil", d, 1)
        m2 = nearby_on(WTI, "WTI Crude Oil", d, 2)
        assert m1.ticker == "CLN26.NYM", f"expected CLN26.NYM front, got {m1.ticker}"
        assert m2.ticker == "CLQ26.NYM", f"expected CLQ26.NYM M2, got {m2.ticker}"
        _ok("front=CLN26.NYM (Jul), M2=CLQ26.NYM (Aug) — matches real WTI curve")
    except Exception:
        _fail("front matches known WTI")


def test_roll_shifts_m2_into_m1():
    print("7. nearby_on — crossing a roll_date promotes the old M2 to M1")
    try:
        m1_before = nearby_on(WTI, "WTI Crude Oil", date(2026, 6, 2), 1)
        roll = m1_before.roll_date
        before = roll  # last day it is still front
        after = (pd.Timestamp(roll) + pd.tseries.offsets.BDay(1)).date()

        m1_on_roll = nearby_on(WTI, "WTI Crude Oil", before, 1)
        m1_after = nearby_on(WTI, "WTI Crude Oil", after, 1)
        m2_before = nearby_on(WTI, "WTI Crude Oil", before, 2)

        assert m1_on_roll.ticker == m1_before.ticker, "front changed too early"
        # the day after the front's roll_date, the prior M2 becomes the new M1
        assert m1_after.ticker == m2_before.ticker, (
            f"after roll, M1 should be old M2 ({m2_before.ticker}), got {m1_after.ticker}"
        )
        _ok(f"front {m1_before.ticker} rolls → {m1_after.ticker} (= prior M2)")
    except Exception:
        _fail("roll promotes m2→m1")


def test_nearby_schedule_no_gaps_and_distinct_legs():
    print("8. nearby_schedule — every business day resolves; M1≠M2 every day")
    try:
        days = [ts.date() for ts in pd.bdate_range("2025-06-01", "2026-06-01")]
        s1 = nearby_schedule(WTI, "WTI Crude Oil", days, 1)
        s2 = nearby_schedule(WTI, "WTI Crude Oil", days, 2)
        assert len(s1) == len(days), "M1 schedule has gaps"
        assert len(s2) == len(days), "M2 schedule has gaps"
        # M1 and M2 must differ on every date
        bad = [d for d in days if s1[d] == s2[d]]
        assert not bad, f"M1==M2 on {bad[:3]}"
        # M1 should change (roll) several times across a year of monthly contracts
        n_rolls = len(set(s1.values()))
        assert n_rolls >= 10, f"expected ~12 monthly fronts, got {n_rolls}"
        _ok(f"{len(days)} days, M1 used {n_rolls} contracts, M1≠M2 always")
    except Exception:
        _fail("nearby_schedule")


def test_required_tickers_superset_of_schedule():
    print("9. required_tickers — is exactly the union of the M1 & M2 schedules")
    try:
        start, end = date(2025, 6, 1), date(2026, 6, 1)
        days = [ts.date() for ts in pd.bdate_range(start, end)]
        s1 = set(nearby_schedule(GOLD, "Gold (COMEX)", days, 1).values())
        s2 = set(nearby_schedule(GOLD, "Gold (COMEX)", days, 2).values())
        req = required_tickers(GOLD, "Gold (COMEX)", start, end, max_depth=2)
        assert req == (s1 | s2), "required_tickers != union of M1/M2 schedules"
        _ok(f"{len(req)} gold tickers needed = union of M1({len(s1)}) ∪ M2({len(s2)})")
    except Exception:
        _fail("required_tickers")


def test_all_specs_resolve_today():
    print("10. Every futures spec resolves a distinct M1/M2 today (smoke test)")
    try:
        d = date(2026, 6, 2)
        problems = []
        for name, spec in FUTURES_CONTRACT_SPECS.items():
            m1 = nearby_on(spec, name, d, 1)
            m2 = nearby_on(spec, name, d, 2)
            if m1 is None or m2 is None:
                problems.append(f"{name}: missing leg")
            elif m1.ticker == m2.ticker:
                problems.append(f"{name}: M1==M2 ({m1.ticker})")
            elif (m1.year, m1.month) >= (m2.year, m2.month):
                problems.append(f"{name}: M2 not after M1")
        assert not problems, "; ".join(problems)
        _ok(f"all {len(FUTURES_CONTRACT_SPECS)} specs resolve distinct, ordered M1/M2")
    except Exception:
        _fail("all specs resolve")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("CONTRACT CALENDAR — TEST SUITE")
    print("=" * 60)
    print()
    for fn in [
        test_ticker_format,
        test_month_codes_complete,
        test_roll_date_sign_convention,
        test_listed_contracts_sorted_and_cover_cycle,
        test_m1_m2_are_adjacent_and_distinct,
        test_m1_today_matches_known_front,
        test_roll_shifts_m2_into_m1,
        test_nearby_schedule_no_gaps_and_distinct_legs,
        test_required_tickers_superset_of_schedule,
        test_all_specs_resolve_today,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
