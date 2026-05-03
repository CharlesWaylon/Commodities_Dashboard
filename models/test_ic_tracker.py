"""
Tests for models/ic_tracker.py — Phase 5 Part 2.

Coverage
────────
 1.  compute_ic — perfect positive correlation returns ~1.0
 2.  compute_ic — perfect negative correlation returns ~-1.0
 3.  compute_ic — constant forecasts returns NaN
 4.  compute_ic — fewer than 5 observations returns NaN
 5.  compute_ic — NaN values in inputs are dropped gracefully
 6.  compute_ic_from_records — correct IC per (commodity, tier)
 7.  compute_ic_from_records — empty records → empty dict
 8.  compute_ic_from_records — missing tier in some records handled
 9.  ICResult — signal_strength labels at boundary values
10.  ICResult — badge_color returns hex strings
11.  log_ic_scores — writes rows to SQLite
12.  log_ic_scores — empty dict → 0 rows, no crash
13.  log_ic_scores — idempotent: same (computed_at, commodity, tier) → no duplicate
14.  recent_ic_scores — returns rows within window
15.  recent_ic_scores — commodity/tier filter works
16.  recent_ic_scores — empty DB returns empty DataFrame
17.  ic_summary — returns latest IC per (commodity, tier)
18.  ic_trend — aggregates mean IC per tier over time
"""

import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ic_tracker import (
    IC_THRESHOLD_ACTION,
    COMMODITY_SECTORS,
    KNOWN_SECTORS,
    ICResult,
    compute_ic,
    compute_ic_from_records,
    compute_sector_ic_from_records,
    ic_summary,
    ic_sector_summary,
    ic_commodity_summary,
    ic_trend,
    log_ic_scores,
    recent_ic_scores,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fake_record(commodity: str, tier: str, fc: float, actual: float,
                 date_offset: int = 0):
    """Minimal BacktestRecord-like object (named tuple or simple class)."""
    from dataclasses import dataclass, field
    from datetime import date, timedelta as td

    @dataclass
    class FakeRecord:
        commodity: str
        tier_forecasts: dict
        actual_return: float
        date: object

    return FakeRecord(
        commodity=commodity,
        tier_forecasts={tier: fc},
        actual_return=actual,
        date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=date_offset),
    )


def _ic_results_for_db(n: int = 3, tier: str = "ml",
                        commodity: str = "WTI Crude Oil",
                        days_ago: int = 0) -> dict:
    """Build a small dict of ICResult objects for DB tests."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        (commodity, tier): ICResult(
            commodity=commodity,
            tier=tier,
            ic_value=0.08,
            n_obs=n,
            window_start="2025-01-01",
            window_end="2025-03-31",
            computed_at=ts,
        )
    }


def _check(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")
        raise AssertionError(f"Test failed: {label}. {detail}")


# ── Tests ──────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 60)
    print("IC TRACKER — TEST SUITE  (Phase 5 Part 2)")
    print("=" * 60)

    # ── 1. Perfect positive correlation ───────────────────────────────────────
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    ic = compute_ic(x, x)
    _check("compute_ic — perfect positive ≈ 1.0",
           abs(ic - 1.0) < 1e-9, f"ic={ic}")

    # ── 2. Perfect negative correlation ───────────────────────────────────────
    y = list(reversed(x))
    ic2 = compute_ic(x, y)
    _check("compute_ic — perfect negative ≈ -1.0",
           abs(ic2 + 1.0) < 1e-9, f"ic={ic2}")

    # ── 3. Constant forecasts → NaN ───────────────────────────────────────────
    ic3 = compute_ic([0.0] * 10, [1.0, 2.0, 3.0, 4.0, 5.0,
                                   6.0, 7.0, 8.0, 9.0, 10.0])
    _check("compute_ic — constant input → NaN", np.isnan(ic3), f"ic={ic3}")

    # ── 4. Too few observations → NaN ─────────────────────────────────────────
    ic4 = compute_ic([0.01, 0.02, 0.03], [-0.01, 0.02, 0.04])
    _check("compute_ic — <5 obs → NaN", np.isnan(ic4), f"ic={ic4}")

    # ── 5. NaN values dropped ────────────────────────────────────────────────
    # Two NaNs at different positions; 6 valid pairs remain (above the min-5 guard)
    fc5 = [1.0, float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ac5 = [1.0, 2.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0]
    ic5 = compute_ic(fc5, ac5)
    _check("compute_ic — NaN values dropped gracefully",
           not np.isnan(ic5), f"ic={ic5}")

    # ── 6. compute_ic_from_records — correct IC per (commodity, tier) ─────────
    # Build 10 records where forecast rank = actual rank → IC should be ~1.0
    records6 = [
        _fake_record("WTI", "ml", fc=float(i), actual=float(i), date_offset=i)
        for i in range(10)
    ]
    r6 = compute_ic_from_records(records6)
    _check("compute_ic_from_records — (WTI, ml) key present",
           ("WTI", "ml") in r6, f"keys={list(r6.keys())}")
    _check("compute_ic_from_records — IC ≈ 1.0 for perfect ranks",
           abs(r6[("WTI", "ml")].ic_value - 1.0) < 1e-9,
           f"ic={r6[('WTI', 'ml')].ic_value}")
    _check("compute_ic_from_records — n_obs = 10",
           r6[("WTI", "ml")].n_obs == 10, f"n_obs={r6[('WTI', 'ml')].n_obs}")

    # ── 7. Empty records → empty dict ────────────────────────────────────────
    r7 = compute_ic_from_records([])
    _check("compute_ic_from_records — empty records → {}",
           r7 == {}, f"got={r7}")

    # ── 8. Missing tier in some records handled ───────────────────────────────
    records8 = [
        _fake_record("Gold", "statistical", fc=float(i), actual=float(i), date_offset=i)
        for i in range(8)
    ] + [
        _fake_record("Gold", "ml", fc=float(i) * 0.5, actual=float(i), date_offset=i)
        for i in range(6)
    ]
    r8 = compute_ic_from_records(records8)
    _check("compute_ic_from_records — both tiers present",
           ("Gold", "statistical") in r8 and ("Gold", "ml") in r8,
           f"keys={list(r8.keys())}")

    # ── 9. ICResult signal_strength boundaries ────────────────────────────────
    r_action  = ICResult("X", "ml", IC_THRESHOLD_ACTION + 0.01, 10, "2025-01-01", "2025-03-31")
    r_marg    = ICResult("X", "ml", IC_THRESHOLD_ACTION - 0.01, 10, "2025-01-01", "2025-03-31")
    r_neg     = ICResult("X", "ml", -0.02, 10, "2025-01-01", "2025-03-31")
    _check("ICResult — actionable above threshold",
           r_action.signal_strength == "actionable", r_action.signal_strength)
    _check("ICResult — marginal below threshold",
           r_marg.signal_strength == "marginal", r_marg.signal_strength)
    _check("ICResult — negative below zero",
           r_neg.signal_strength == "negative", r_neg.signal_strength)

    # ── 10. ICResult badge_color returns hex strings ──────────────────────────
    _check("ICResult — badge_color is hex string",
           r_action.badge_color.startswith("#"), r_action.badge_color)

    # ── 11. log_ic_scores — writes rows ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        res = _ic_results_for_db()
        n = log_ic_scores(res, db_path=db)
        _check("log_ic_scores — returns row count", n == 1, f"n={n}")

        # Verify round-trip
        df = recent_ic_scores(days=365, db_path=db)
        _check("log_ic_scores — row readable via recent_ic_scores",
               len(df) == 1, f"rows={len(df)}")
        _check("log_ic_scores — ic_value correct",
               abs(float(df.iloc[0]["ic_value"]) - 0.08) < 1e-9,
               f"ic={df.iloc[0]['ic_value']}")

    # ── 12. Empty dict → 0 rows, no crash ────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        n12 = log_ic_scores({}, db_path=Path(td) / "test.db")
        _check("log_ic_scores — empty dict → 0", n12 == 0, f"n={n12}")

    # ── 13. Idempotent: same key → no duplicate ───────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        db13 = Path(td) / "test.db"
        ts = datetime.now(timezone.utc).isoformat()
        fixed_res = {
            ("WTI Crude Oil", "ml"): ICResult(
                "WTI Crude Oil", "ml", 0.07, 20,
                "2025-01-01", "2025-03-31", computed_at=ts,
            )
        }
        log_ic_scores(fixed_res, db_path=db13)
        log_ic_scores(fixed_res, db_path=db13)  # second write, same key
        df13 = recent_ic_scores(days=365, db_path=db13)
        _check("log_ic_scores — idempotent (no duplicate rows)",
               len(df13) == 1, f"rows={len(df13)}")

    # ── 14. recent_ic_scores — window filter ─────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        db14 = Path(td) / "test.db"
        # Old entry (200 days ago)
        log_ic_scores(_ic_results_for_db(days_ago=200, tier="ml"), db_path=db14)
        # Recent entry (1 day ago)
        log_ic_scores(_ic_results_for_db(days_ago=1, tier="statistical"), db_path=db14)

        df14 = recent_ic_scores(days=90, db_path=db14)
        _check("recent_ic_scores — only recent rows returned",
               len(df14) == 1 and df14.iloc[0]["tier"] == "statistical",
               f"rows={len(df14)} tiers={list(df14['tier']) if not df14.empty else []}")

    # ── 15. Commodity/tier filter ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        db15 = Path(td) / "test.db"
        log_ic_scores(_ic_results_for_db(commodity="WTI Crude Oil", tier="ml"),
                      db_path=db15)
        log_ic_scores(_ic_results_for_db(commodity="Gold (COMEX)", tier="statistical"),
                      db_path=db15)

        df15a = recent_ic_scores(days=365, commodity="Gold (COMEX)", db_path=db15)
        df15b = recent_ic_scores(days=365, tier="ml", db_path=db15)
        _check("recent_ic_scores — commodity filter",
               len(df15a) == 1 and df15a.iloc[0]["commodity"] == "Gold (COMEX)",
               f"rows={len(df15a)}")
        _check("recent_ic_scores — tier filter",
               len(df15b) == 1 and df15b.iloc[0]["tier"] == "ml",
               f"rows={len(df15b)}")

    # ── 16. Empty DB → empty DataFrame ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        df16 = recent_ic_scores(db_path=Path(td) / "empty.db")
        _check("recent_ic_scores — empty DB → empty DataFrame",
               df16.empty, f"rows={len(df16)}")

    # ── 17. ic_summary — latest IC per (commodity, tier) ─────────────────────
    with tempfile.TemporaryDirectory() as td:
        db17 = Path(td) / "test.db"
        # Write two entries for the same (commodity, tier) at different times
        ts_old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        ts_new = datetime.now(timezone.utc).isoformat()
        old_res = {("WTI Crude Oil", "ml"): ICResult(
            "WTI Crude Oil", "ml", 0.03, 20, "2025-01-01", "2025-02-01", ts_old)}
        new_res = {("WTI Crude Oil", "ml"): ICResult(
            "WTI Crude Oil", "ml", 0.09, 25, "2025-02-01", "2025-03-01", ts_new)}
        log_ic_scores(old_res, db_path=db17)
        log_ic_scores(new_res, db_path=db17)

        summary = ic_summary(db_path=db17)
        _check("ic_summary — one row per (commodity, tier)",
               len(summary) == 1, f"rows={len(summary)}")
        _check("ic_summary — shows LATEST ic_value",
               abs(float(summary.iloc[0]["ic_value"]) - 0.09) < 1e-9,
               f"ic={summary.iloc[0]['ic_value']}")
        _check("ic_summary — signal_strength column present",
               "signal_strength" in summary.columns,
               str(summary.columns.tolist()))

    # ── 18. ic_trend — aggregates mean IC per tier ────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        db18 = Path(td) / "test.db"
        ts1 = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        ts2 = datetime.now(timezone.utc).isoformat()

        res_a = {
            ("WTI", "ml"):   ICResult("WTI", "ml",   0.06, 20, "2025-01-01", "2025-03-01", ts1),
            ("Gold", "ml"):  ICResult("Gold", "ml",  0.04, 20, "2025-01-01", "2025-03-01", ts1),
        }
        res_b = {
            ("WTI", "ml"):   ICResult("WTI", "ml",   0.08, 25, "2025-03-01", "2025-04-01", ts2),
        }
        log_ic_scores(res_a, db_path=db18)
        log_ic_scores(res_b, db_path=db18)

        trend = ic_trend(days=365, db_path=db18)
        _check("ic_trend — non-empty DataFrame", not trend.empty, "empty")
        _check("ic_trend — columns include mean_ic + n_commodities",
               {"mean_ic", "n_commodities"}.issubset(trend.columns),
               str(trend.columns.tolist()))
        # At ts1 there are 2 commodities for 'ml'
        row_ts1 = trend[trend["tier"] == "ml"].iloc[0]
        _check("ic_trend — n_commodities=2 for first timestamp",
               int(row_ts1["n_commodities"]) == 2,
               f"n={row_ts1['n_commodities']}")
        expected_mean = (0.06 + 0.04) / 2
        _check("ic_trend — mean_ic averaged correctly",
               abs(float(row_ts1["mean_ic"]) - expected_mean) < 1e-9,
               f"mean={row_ts1['mean_ic']} expected={expected_mean}")

    # ── 19. compute_sector_ic_from_records — energy records pool correctly ───────
    # 10 WTI records + 10 Brent records → Energy sector should have 20 obs
    energy_records = (
        [_fake_record("WTI Crude Oil",  "ml", fc=float(i), actual=float(i),
                      date_offset=i) for i in range(10)]
        + [_fake_record("Brent Crude Oil", "ml", fc=float(i), actual=float(i),
                        date_offset=i + 10) for i in range(10)]
    )
    r19 = compute_sector_ic_from_records(energy_records)
    _check("compute_sector_ic_from_records — Energy sector present",
           ("Energy", "ml") in r19, f"keys={list(r19.keys())}")
    _check("compute_sector_ic_from_records — n_obs pools both commodities",
           r19[("Energy", "ml")].n_obs == 20,
           f"n_obs={r19[('Energy', 'ml')].n_obs}")
    _check("compute_sector_ic_from_records — commodity field = sector name",
           r19[("Energy", "ml")].commodity == "Energy",
           r19[("Energy", "ml")].commodity)

    # ── 20. Unknown commodity is skipped at sector level ─────────────────────
    unknown_records = [
        _fake_record("UnknownCommodityXYZ", "ml", float(i), float(i), i)
        for i in range(10)
    ]
    r20 = compute_sector_ic_from_records(unknown_records)
    _check("compute_sector_ic_from_records — unknown commodity skipped",
           r20 == {}, f"got={r20}")

    # ── 21. Sector IC pools multiple sectors independently ────────────────────
    mixed_records = (
        [_fake_record("WTI Crude Oil",  "statistical", float(i), float(i), i)
         for i in range(8)]
        + [_fake_record("Gold (COMEX)", "statistical", float(i), float(i), i)
           for i in range(8)]
    )
    r21 = compute_sector_ic_from_records(mixed_records)
    _check("compute_sector_ic_from_records — Energy and Metals separate",
           ("Energy", "statistical") in r21 and ("Metals", "statistical") in r21,
           f"keys={list(r21.keys())}")
    # Energy and Metals should NOT be mixed together
    _check("compute_sector_ic_from_records — Energy n_obs = 8 (not 16)",
           r21[("Energy", "statistical")].n_obs == 8,
           f"Energy n_obs={r21[('Energy', 'statistical')].n_obs}")

    # ── 22. ic_sector_summary / ic_commodity_summary split correctly ──────────
    with tempfile.TemporaryDirectory() as td:
        db22 = Path(td) / "test.db"
        ts22 = datetime.now(timezone.utc).isoformat()

        # Log one commodity-level row and one sector-level row
        commodity_res = {
            ("WTI Crude Oil", "ml"): ICResult(
                "WTI Crude Oil", "ml", 0.07, 20,
                "2025-01-01", "2025-03-31", ts22)
        }
        sector_res = {
            ("Energy", "ml"): ICResult(
                "Energy", "ml", 0.09, 40,
                "2025-01-01", "2025-03-31", ts22)
        }
        log_ic_scores({**commodity_res, **sector_res}, db_path=db22)

        sec_df  = ic_sector_summary(db_path=db22)
        comm_df = ic_commodity_summary(db_path=db22)

        _check("ic_sector_summary — returns only sector rows",
               len(sec_df) == 1 and sec_df.iloc[0]["commodity"] == "Energy",
               f"rows={len(sec_df)} commodities={list(sec_df['commodity']) if not sec_df.empty else []}")
        _check("ic_commodity_summary — returns only commodity rows",
               len(comm_df) == 1 and comm_df.iloc[0]["commodity"] == "WTI Crude Oil",
               f"rows={len(comm_df)} commodities={list(comm_df['commodity']) if not comm_df.empty else []}")
        _check("KNOWN_SECTORS contains all expected sectors",
               {"Energy", "Metals", "Agriculture", "Livestock", "Digital Assets"}.issubset(KNOWN_SECTORS),
               f"KNOWN_SECTORS={KNOWN_SECTORS}")

    print()
    print("=" * 60)
    print("ALL 22 ASSERTIONS PASSED")
    print("Phase 5 Part 2 — IC Tracker + sector-level IC ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
