"""
Tests for pipeline/stitch_m2.py — the PURE stitching core only (no network/DB).
Run: python -m pipeline.test_stitch_m2

The DB shell (_load_contract_series / _replace_series / run_stitch) needs Postgres
and is validated on the Mac.  Here we feed synthetic per-contract series into
stitch_constant_maturity and assert the calendar selects the genuine M1/M2 close
on each date.
"""

import sys
import traceback
from datetime import date

import pandas as pd

from config.futures_calendar import FUTURES_CONTRACT_SPECS
from pipeline.contract_calendar import (
    listed_contracts,
    nearby_on,
    contract_code,
)
from pipeline.stitch_m2 import stitch_constant_maturity

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


WTI = FUTURES_CONTRACT_SPECS["WTI Crude Oil"]
GOLD = FUTURES_CONTRACT_SPECS["Gold (COMEX)"]


def _synthetic_contract_series(spec, commodity, start, end, price_fn):
    """
    Build {contract_code: Series(date->close)} covering [start,end] for every
    listed contract, where each date's close is price_fn(contract, d).  Mimics
    what _load_contract_series would return from price_history.
    """
    contracts = listed_contracts(spec, commodity, start, end)
    bdays = [ts.date() for ts in pd.bdate_range(start, end)]
    out: dict[str, dict] = {}
    for c in contracts:
        code = contract_code(c.ticker)
        series = {}
        for d in bdays:
            # Only "list" a contract while it is still a plausible nearby (its
            # roll_date hasn't passed by more than a year) — Yahoo holds ~1y/contract.
            if c.roll_date >= d and (c.roll_date - d).days <= 400:
                series[d] = price_fn(c, d)
        if series:
            out[code] = pd.Series(series, dtype=float).sort_index()
    return out


def test_m1_m2_legs_match_calendar():
    print("1. stitched M1/M2 close == the calendar-resolved contract's close each day")
    try:
        start, end = date(2025, 6, 1), date(2026, 6, 1)
        # Distinct, identifiable price per contract: encode (year,month) in the value.
        def price_fn(c, d):
            return 1000 * c.year + 10 * c.month  # unique per contract, constant in d
        cs = _synthetic_contract_series(WTI, "WTI Crude Oil", start, end, price_fn)
        stitched = stitch_constant_maturity(WTI, "WTI Crude Oil", cs, max_depth=2)
        m1, m2 = stitched[1], stitched[2]
        assert not m1.empty and not m2.empty, "empty stitch"

        contracts = listed_contracts(WTI, "WTI Crude Oil", start, end)
        bad = 0
        for d in m2.index:
            c1 = nearby_on(WTI, "WTI Crude Oil", d, 1, contracts)
            c2 = nearby_on(WTI, "WTI Crude Oil", d, 2, contracts)
            if price_fn(c1, d) != m1.loc[d] or price_fn(c2, d) != m2.loc[d]:
                bad += 1
        assert bad == 0, f"{bad} dates where stitched leg != calendar contract"
        _ok(f"{len(m2)} days: every M1/M2 close traces to the resolved contract")
    except Exception:
        _fail("legs match calendar")


def test_m1_differs_from_m2_every_day():
    print("2. M1 and M2 are different contracts → different values every day")
    try:
        start, end = date(2025, 6, 1), date(2026, 6, 1)
        def price_fn(c, d):
            return 1000 * c.year + 10 * c.month
        cs = _synthetic_contract_series(GOLD, "Gold (COMEX)", start, end, price_fn)
        stitched = stitch_constant_maturity(GOLD, "Gold (COMEX)", cs, max_depth=2)
        m1, m2 = stitched[1], stitched[2]
        shared = m1.index.intersection(m2.index)
        assert len(shared) > 100, "too little overlap"
        same = [d for d in shared if m1.loc[d] == m2.loc[d]]
        assert not same, f"M1==M2 on {same[:3]}"
        _ok(f"{len(shared)} overlapping days, M1≠M2 on all")
    except Exception:
        _fail("m1 != m2")


def test_basis_sign_tracks_carry():
    print("3. basis = log(M1/M2): backwardation→positive, contango→negative")
    try:
        start, end = date(2025, 6, 1), date(2026, 6, 1)
        contracts = listed_contracts(WTI, "WTI Crude Oil", start, end)

        # Contango: later delivery priced higher → M1 < M2 → basis < 0.
        def contango(c, d):
            # higher value for later (year, month)
            return 50.0 + (c.year - 2025) * 12 + c.month
        cs = _synthetic_contract_series(WTI, "WTI Crude Oil", start, end, contango)
        stitched = stitch_constant_maturity(WTI, "WTI Crude Oil", cs, max_depth=2)
        m1, m2 = stitched[1], stitched[2]
        import numpy as np
        shared = m1.index.intersection(m2.index)
        basis = np.log(m1[shared] / m2[shared])
        assert (basis < 0).mean() > 0.95, "contango should give mostly negative basis"
        _ok(f"contango → basis<0 on {(basis<0).mean()*100:.0f}% of days")
    except Exception:
        _fail("basis sign")


def test_roll_promotes_m2_to_m1_in_series():
    print("4. across a front roll, yesterday's M2 value becomes today's M1 value")
    try:
        start, end = date(2025, 6, 1), date(2026, 6, 1)
        def price_fn(c, d):
            return 1000 * c.year + 10 * c.month
        cs = _synthetic_contract_series(WTI, "WTI Crude Oil", start, end, price_fn)
        stitched = stitch_constant_maturity(WTI, "WTI Crude Oil", cs, max_depth=2)
        m1, m2 = stitched[1], stitched[2]
        contracts = listed_contracts(WTI, "WTI Crude Oil", start, end)

        # Find a date where the front contract changes vs the previous trading day.
        idx = list(m1.index)
        rolls = 0
        for prev, cur in zip(idx, idx[1:]):
            c1_prev = nearby_on(WTI, "WTI Crude Oil", prev, 1, contracts)
            c1_cur  = nearby_on(WTI, "WTI Crude Oil", cur, 1, contracts)
            if c1_prev.ticker != c1_cur.ticker:
                # On a roll, the new M1 must equal the prior day's M2 contract value
                if cur in m2.index and prev in m2.index:
                    assert m1.loc[cur] == m2.loc[prev], (
                        f"roll {prev}->{cur}: new M1 {m1.loc[cur]} != prior M2 {m2.loc[prev]}"
                    )
                    rolls += 1
        assert rolls >= 8, f"expected ~12 monthly rolls, saw {rolls}"
        _ok(f"verified M2→M1 promotion across {rolls} rolls")
    except Exception:
        _fail("roll promotion in series")


def test_empty_input_returns_empty_series():
    print("5. no contracts in → empty (not error) M1/M2 series out")
    try:
        stitched = stitch_constant_maturity(WTI, "WTI Crude Oil", {}, max_depth=2)
        assert stitched[1].empty and stitched[2].empty
        _ok("empty input handled gracefully")
    except Exception:
        _fail("empty input")


def test_all_specs_stitch_without_error():
    print("6. every futures spec stitches a non-trivial M2 from synthetic contracts")
    try:
        start, end = date(2025, 9, 1), date(2026, 6, 1)
        def price_fn(c, d):
            return 1000 * c.year + 10 * c.month
        problems = []
        for name, spec in FUTURES_CONTRACT_SPECS.items():
            cs = _synthetic_contract_series(spec, name, start, end, price_fn)
            stitched = stitch_constant_maturity(spec, name, cs, max_depth=2)
            if stitched[2].empty:
                problems.append(f"{name}: empty M2")
        assert not problems, "; ".join(problems)
        _ok(f"all {len(FUTURES_CONTRACT_SPECS)} specs produced a non-empty M2")
    except Exception:
        _fail("all specs stitch")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("M2 STITCHER — TEST SUITE (pure core)")
    print("=" * 60)
    print()
    for fn in [
        test_m1_m2_legs_match_calendar,
        test_m1_differs_from_m2_every_day,
        test_basis_sign_tracks_carry,
        test_roll_promotes_m2_to_m1_in_series,
        test_empty_input_returns_empty_series,
        test_all_specs_stitch_without_error,
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
