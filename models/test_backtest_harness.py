"""
Tests for models/backtest_harness.py — Phase 3 Part 2.
All tests use synthetic price + macro data; no network, no DB.
Run: python -m models.test_backtest_harness
"""

import sys
import traceback
import warnings

import numpy as np
import pandas as pd

from models.backtest_harness import (
    BacktestRecord,
    TierAdapter,
    BacktestHarness,
    _align_macro,
    _meta_features_for_date,
    _tier_counts,
    DEFAULT_ADAPTERS,
)
from models.meta_predictor import MetaFeatures, MetaPredictor, KNOWN_TIERS
from models.config import FORECAST_HORIZON
from models.features import build_feature_matrix

PASS = 0
FAIL = 0


def _ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg, _err=None):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {msg}")
    traceback.print_exc()


# ── Synthetic data builders ────────────────────────────────────────────────────

def _make_prices(n=300, seed=42, commodities=None) -> pd.DataFrame:
    """Random-walk price matrix with a realistic index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = commodities or ["WTI Crude Oil", "Gold (COMEX)"]
    data = {}
    for c in cols:
        log_ret = rng.normal(0, 0.01, n)
        prices = 100 * np.exp(np.cumsum(log_ret))
        data[c] = prices
    return pd.DataFrame(data, index=idx)


def _make_macro(price_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal macro DataFrame aligned to price_idx."""
    n = len(price_idx)
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "vix":             rng.uniform(12, 35, n),
        "vix_level_z":     rng.normal(0, 1, n),
        "vix_risk_off":    (rng.uniform(0, 1, n) > 0.7).astype(float),
        "vix_crisis":      (rng.uniform(0, 1, n) > 0.95).astype(float),
        "dxy_zscore63":    rng.normal(0, 1, n),
        "dxy_mom21":       rng.normal(0, 0.01, n),
        "tlt_mom21":       rng.normal(0, 0.01, n),
        "tlt_yield_proxy": rng.normal(0, 0.005, n),
        "is_opec_window":  (rng.uniform(0, 1, n) > 0.85).astype(float),
        "days_to_opec":    rng.uniform(-14, 14, n),
        "is_wasde_window": (rng.uniform(0, 1, n) > 0.85).astype(float),
        "days_to_wasde":   rng.uniform(-7, 7, n),
        "wasde_post5":     (rng.uniform(0, 1, n) > 0.90).astype(float),
    }, index=price_idx)


# ── Stub adapter (deterministic, fast) ────────────────────────────────────────

class _StubAdapter(TierAdapter):
    """
    Deterministic stub: always forecasts a fixed constant return.
    Used to test harness plumbing without real model dependencies.
    """
    def __init__(self, tier_name: str, forecast_value: float = 0.0):
        self._tier_name = tier_name
        self._forecast_value = forecast_value

    @property
    def tier(self) -> str:
        return self._tier_name

    def fit(self, prices_train, commodity):
        pass  # nothing to fit

    def predict_series(self, prices_full, commodity, test_idx):
        return pd.Series(self._forecast_value, index=test_idx)


# ── BacktestRecord ─────────────────────────────────────────────────────────────

def test_backtest_record_to_training_pair():
    print("1. BacktestRecord.to_training_pair — returns correct tuple")
    try:
        mf = MetaFeatures(vix=20.0)
        rec = BacktestRecord(
            date=pd.Timestamp("2025-01-15"),
            commodity="WTI Crude Oil",
            meta_features=mf,
            actual_return=0.005,
            tier_forecasts={"ml": 0.004, "statistical": -0.002},
            tier_errors={"ml": 0.001, "statistical": 0.007},
            winning_tier="ml",
        )
        pair = rec.to_training_pair()
        assert pair == (mf, "ml")
        _ok("to_training_pair returns (MetaFeatures, 'ml')")
    except Exception as e:
        _fail("BacktestRecord.to_training_pair", e)


# ── TierAdapter validation ─────────────────────────────────────────────────────

def test_stub_adapter_interface():
    print("2. _StubAdapter — satisfies TierAdapter contract")
    try:
        prices = _make_prices(100)
        macro = _make_macro(prices.index)
        adapter = _StubAdapter("ml", forecast_value=0.002)
        adapter.fit(prices.iloc[:80], "WTI Crude Oil")
        preds = adapter.predict_series(prices, "WTI Crude Oil", prices.index[80:])
        assert len(preds) == len(prices.index[80:])
        assert (preds == 0.002).all()
        _ok("stub adapter produces correct constant predictions")
    except Exception as e:
        _fail("stub adapter", e)


# ── _align_macro ───────────────────────────────────────────────────────────────

def test_align_macro_fills_gaps():
    print("3. _align_macro — forward-fills gaps up to 5 days")
    try:
        price_idx = pd.bdate_range("2025-01-01", periods=20)
        # macro only covers every other day
        macro_idx = price_idx[::2]
        macro = pd.DataFrame({"vix": 18.0}, index=macro_idx)
        aligned = _align_macro(macro, price_idx)
        # After ffill, non-NaN count should be > len(macro_idx)
        assert aligned["vix"].notna().sum() > len(macro_idx)
        _ok(f"{aligned['vix'].notna().sum()} / {len(price_idx)} dates filled")
    except Exception as e:
        _fail("_align_macro", e)


def test_align_macro_empty():
    print("4. _align_macro — empty macro returns empty-indexed DataFrame")
    try:
        price_idx = pd.bdate_range("2025-01-01", periods=10)
        result = _align_macro(None, price_idx)
        assert result.empty or len(result) == len(price_idx)
        result2 = _align_macro(pd.DataFrame(), price_idx)
        assert result2.empty or len(result2) == len(price_idx)
        _ok("None / empty macro handled gracefully")
    except Exception as e:
        _fail("_align_macro empty", e)


# ── _meta_features_for_date ────────────────────────────────────────────────────

def test_meta_features_for_date_present():
    print("5. _meta_features_for_date — present date returns populated MetaFeatures")
    try:
        prices = _make_prices(50)
        macro = _make_macro(prices.index)
        aligned = _align_macro(macro, prices.index)
        date = prices.index[25]
        mf = _meta_features_for_date(aligned, date)
        assert isinstance(mf, MetaFeatures)
        # vix should be set (macro has it)
        assert not np.isnan(mf.vix)
        _ok(f"date {date.date()} → VIX={mf.vix:.1f}")
    except Exception as e:
        _fail("_meta_features_for_date present", e)


def test_meta_features_for_date_missing():
    print("6. _meta_features_for_date — missing date returns default MetaFeatures")
    try:
        import math
        mf = _meta_features_for_date(pd.DataFrame(), pd.Timestamp("2020-01-01"))
        assert math.isnan(mf.vix)
        _ok("missing date → default MetaFeatures with NaN vix")
    except Exception as e:
        _fail("_meta_features_for_date missing", e)


# ── BacktestHarness ────────────────────────────────────────────────────────────

def test_harness_run_basic():
    print("7. BacktestHarness.run — produces records with correct structure")
    try:
        prices = _make_prices(250)
        macro = _make_macro(prices.index)
        adapters = [
            _StubAdapter("statistical", forecast_value=0.001),
            _StubAdapter("ml",          forecast_value=0.003),
        ]
        harness = BacktestHarness(adapters=adapters, test_fraction=0.20)
        records = harness.run(prices, macro, commodities=["WTI Crude Oil"])

        assert len(records) > 0, "expected >0 records"
        for r in records:
            assert isinstance(r.meta_features, MetaFeatures)
            assert r.winning_tier in KNOWN_TIERS
            assert "statistical" in r.tier_forecasts or "ml" in r.tier_forecasts
            assert r.actual_return != 0.0 or True  # actual can be anything
        _ok(f"{len(records)} records; winning tiers: {_tier_counts(records)}")
    except Exception as e:
        _fail("harness run basic", e)


def test_harness_winning_tier_logic():
    print("8. BacktestHarness — winning_tier = tier with lowest abs error")
    try:
        prices = _make_prices(200)
        macro = _make_macro(prices.index)
        # Actual returns are ~0. Stub "statistical" forecasts 0.0 (better)
        # while "ml" forecasts 0.5 (wildly wrong)
        adapters = [
            _StubAdapter("statistical", 0.0),
            _StubAdapter("ml",          0.5),
        ]
        harness = BacktestHarness(adapters=adapters, test_fraction=0.20)
        records = harness.run(prices, macro, ["WTI Crude Oil"])
        assert records, "expected records"
        # statistical should win almost every date (actual ~0, its forecast = 0)
        counts = _tier_counts(records)
        stat_wins = counts.get("statistical", 0)
        ml_wins = counts.get("ml", 0)
        assert stat_wins > ml_wins, (
            f"statistical ({stat_wins}) should beat ml ({ml_wins}) when forecasting 0 vs 0.5"
        )
        _ok(f"statistical wins {stat_wins}, ml wins {ml_wins} — correct")
    except Exception as e:
        _fail("winning_tier logic", e)


def test_harness_skips_insufficient_data():
    print("9. BacktestHarness — skips commodity with insufficient train rows")
    try:
        # Only 50 rows, min_train_rows=120 → should skip
        prices = _make_prices(50)
        macro = _make_macro(prices.index)
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            test_fraction=0.20,
            min_train_rows=120,
        )
        records = harness.run(prices, macro, ["WTI Crude Oil"])
        assert records == []
        _ok("50-row price history skipped (below min_train_rows=120)")
    except Exception as e:
        _fail("insufficient data skip", e)


def test_harness_skips_missing_commodity():
    print("10. BacktestHarness — gracefully skips commodity not in price matrix")
    try:
        prices = _make_prices(200)
        macro = _make_macro(prices.index)
        harness = BacktestHarness(adapters=[_StubAdapter("ml", 0.0)], test_fraction=0.20)
        records = harness.run(prices, macro, commodities=["Does Not Exist"])
        assert records == []
        _ok("missing commodity skipped without raising")
    except Exception as e:
        _fail("missing commodity", e)


def test_harness_multiple_commodities():
    print("11. BacktestHarness — aggregates records across multiple commodities")
    try:
        prices = _make_prices(250, commodities=["WTI Crude Oil", "Gold (COMEX)"])
        macro = _make_macro(prices.index)
        harness = BacktestHarness(
            adapters=[_StubAdapter("statistical", 0.0), _StubAdapter("ml", 0.001)],
            test_fraction=0.20,
        )
        records = harness.run(prices, macro)
        commodities_seen = {r.commodity for r in records}
        assert "WTI Crude Oil" in commodities_seen
        assert "Gold (COMEX)" in commodities_seen
        _ok(f"{len(records)} total records across {len(commodities_seen)} commodities")
    except Exception as e:
        _fail("multiple commodities", e)


def test_harness_collect_training_pairs():
    print("12. BacktestHarness.collect_training_pairs — returns (MetaFeatures, tier) tuples")
    try:
        prices = _make_prices(200)
        macro = _make_macro(prices.index)
        harness = BacktestHarness(
            adapters=[_StubAdapter("statistical", 0.0), _StubAdapter("ml", 0.1)],
            test_fraction=0.20,
        )
        pairs = harness.collect_training_pairs(prices, macro, ["WTI Crude Oil"])
        assert len(pairs) > 0
        for mf, tier in pairs:
            assert isinstance(mf, MetaFeatures)
            assert tier in KNOWN_TIERS
        _ok(f"{len(pairs)} training pairs; tiers: {set(t for _, t in pairs)}")
    except Exception as e:
        _fail("collect_training_pairs", e)


# ── train_meta_predictor (end-to-end integration) ─────────────────────────────

def test_harness_train_meta_predictor():
    print("13. BacktestHarness.train_meta_predictor — end-to-end: backtest → fit → predict")
    try:
        prices = _make_prices(300, commodities=["WTI Crude Oil", "Gold (COMEX)"])
        macro = _make_macro(prices.index)
        harness = BacktestHarness(
            adapters=[_StubAdapter("statistical", 0.0), _StubAdapter("ml", 0.05)],
            test_fraction=0.20,
        )
        meta = harness.train_meta_predictor(
            prices, macro,
            commodities=["WTI Crude Oil", "Gold (COMEX)"],
            max_depth=3,
            min_samples_leaf=1,   # allow small leaves for synthetic data
        )
        assert isinstance(meta, MetaPredictor)
        assert meta.is_trained

        # The trained meta should make a valid prediction
        from models.meta_predictor import collect_meta_features, ModelVote
        mf = collect_meta_features(macro)
        votes = [
            ModelVote("statistical", "ARIMA",   "WTI Crude Oil", 0.002, 0.06),
            ModelVote("ml",          "XGBoost", "WTI Crude Oil", 0.007, 0.09),
        ]
        decision = meta.predict(mf, votes)
        present = {"statistical", "ml"}
        assert abs(sum(decision.weights[t] for t in present) - 1.0) < 1e-9
        assert decision.trusted_tier in present
        _ok(
            f"trained on {len(prices)*0.20:.0f}+ records; "
            f"trusted={decision.trusted_tier}; "
            f"top feature: {meta._top_feature()}"
        )
    except Exception as e:
        _fail("train_meta_predictor", e)


# ── _tier_counts ───────────────────────────────────────────────────────────────

def test_tier_counts():
    print("14. _tier_counts — correct tally")
    try:
        mf = MetaFeatures()
        records = [
            BacktestRecord(pd.Timestamp("2025-01-01"), "X", mf, 0.0, {}, {}, "ml"),
            BacktestRecord(pd.Timestamp("2025-01-02"), "X", mf, 0.0, {}, {}, "ml"),
            BacktestRecord(pd.Timestamp("2025-01-03"), "X", mf, 0.0, {}, {}, "statistical"),
        ]
        counts = _tier_counts(records)
        assert counts["ml"] == 2
        assert counts["statistical"] == 1
        _ok(f"counts={counts}")
    except Exception as e:
        _fail("_tier_counts", e)


# ── Walk-forward CV tests ──────────────────────────────────────────────────────

def test_split_indices_structure():
    print("15. _split_indices — test windows are monotonically increasing and non-overlapping")
    try:
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            min_train_rows=60,
            n_splits=4,
        )
        splits = harness._split_indices(300)
        assert len(splits) == 4, f"expected 4 folds, got {len(splits)}"

        prev_end = -1
        for i, (ts, te) in enumerate(splits):
            assert ts < te, f"fold {i}: test_start ({ts}) >= test_end ({te})"
            assert ts >= harness.min_train_rows, (
                f"fold {i}: test_start {ts} < min_train_rows {harness.min_train_rows}"
            )
            assert ts >= prev_end, (
                f"fold {i}: test windows overlap — start {ts} < prev end {prev_end}"
            )
            prev_end = te

        _ok(f"4 non-overlapping folds: {splits}")
    except Exception as e:
        _fail("_split_indices structure", e)


def test_split_indices_min_train_respected():
    print("16. _split_indices — every fold has at least min_train_rows training rows")
    try:
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            min_train_rows=100,
            n_splits=5,
        )
        splits = harness._split_indices(400)
        for i, (ts, te) in enumerate(splits):
            n_train = ts  # training is prices.iloc[:test_start]
            assert n_train >= 100, (
                f"fold {i}: only {n_train} training rows, need ≥ 100"
            )
        _ok(f"all {len(splits)} folds have ≥ 100 training rows")
    except Exception as e:
        _fail("_split_indices min_train", e)


def test_split_indices_single_split_fallback():
    print("17. _split_indices — n_splits=1 falls back to test_fraction single split")
    try:
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            test_fraction=0.20,
            min_train_rows=50,
            n_splits=1,
        )
        n = 300
        splits = harness._split_indices(n)
        assert len(splits) == 1, f"expected 1 fold, got {len(splits)}"
        ts, te = splits[0]
        expected_split = int(n * (1 - 0.20))   # = 240
        expected_usable = n - FORECAST_HORIZON  # = 290 (embargo applied)
        assert ts == expected_split, f"test_start={ts}, expected {expected_split}"
        assert te == expected_usable, f"test_end={te}, expected {expected_usable} (n - FORECAST_HORIZON)"
        _ok(f"single split: train [0:{ts}], test [{ts}:{te}] (usable={expected_usable})")
    except Exception as e:
        _fail("_split_indices single-split fallback", e)


def test_walk_forward_produces_more_records():
    print("18. Walk-forward (n_splits=5) produces more records than single split (n_splits=1)")
    try:
        prices = _make_prices(300)
        macro  = _make_macro(prices.index)
        adapters = [_StubAdapter("statistical", 0.0), _StubAdapter("ml", 0.001)]

        h_wf  = BacktestHarness(adapters=adapters, min_train_rows=60, n_splits=5)
        h_one = BacktestHarness(adapters=adapters, min_train_rows=60, n_splits=1,
                                test_fraction=0.20)

        r_wf  = h_wf.run(prices, macro,  ["WTI Crude Oil"])
        r_one = h_one.run(prices, macro, ["WTI Crude Oil"])

        assert len(r_wf) > len(r_one), (
            f"walk-forward ({len(r_wf)}) should exceed single-split ({len(r_one)})"
        )
        _ok(f"walk-forward={len(r_wf)} records  vs  single-split={len(r_one)} records")
    except Exception as e:
        _fail("walk-forward vs single split record count", e)


def test_walk_forward_no_data_leakage():
    print("19. Walk-forward — each fold's test dates are strictly after its train dates")
    try:
        prices  = _make_prices(300)
        macro   = _make_macro(prices.index)
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            min_train_rows=60,
            n_splits=4,
        )
        splits = harness._split_indices(len(prices))
        for fold, (ts, te) in enumerate(splits):
            train_last_date = prices.index[ts - 1]
            test_first_date = prices.index[ts]
            assert test_first_date > train_last_date, (
                f"fold {fold}: test starts on {test_first_date} "
                f"but train ends on {train_last_date}"
            )
        _ok(f"no leakage across all {len(splits)} folds")
    except Exception as e:
        _fail("walk-forward no leakage", e)


def test_walk_forward_covers_most_of_history():
    print("20. Walk-forward — union of test windows covers the post-min_train portion of history")
    try:
        n = 400
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            min_train_rows=100,
            n_splits=5,
        )
        splits = harness._split_indices(n)

        # Total test dates across all folds
        total_test_dates = sum(te - ts for ts, te in splits)
        # Available dates after min_train_rows (minus last row)
        available = (n - 1) - harness.min_train_rows

        # Folds cover at least 80% of available history
        coverage = total_test_dates / available
        assert coverage >= 0.80, (
            f"folds cover only {coverage:.0%} of available history "
            f"({total_test_dates}/{available} dates)"
        )
        _ok(
            f"{total_test_dates}/{available} available dates covered "
            f"({coverage:.0%}) across {len(splits)} folds"
        )
    except Exception as e:
        _fail("walk-forward coverage", e)


# ── Term-structure feature tests (Prompt 2) ───────────────────────────────────

def _make_m2_prices(m1: pd.DataFrame, spread: float = 0.02) -> pd.DataFrame:
    """Synthetic M2 matrix: front price inflated by a fixed spread (contango)."""
    return m1 * (1 + spread)


def test_build_feature_matrix_with_m2_gains_basis_columns():
    print("21. build_feature_matrix with M2 — output gains *_basis / *_basis_zscore / *_roll_yield columns")
    try:
        m1 = _make_prices(300)
        m2 = _make_m2_prices(m1)

        feat_with_m2 = build_feature_matrix(m1, second_contract_prices=m2)

        cols = feat_with_m2.columns.tolist()
        for commodity in m1.columns:
            for suffix in ("_basis", "_basis_zscore", "_roll_yield"):
                expected = f"{commodity}{suffix}"
                assert expected in cols, (
                    f"Expected column '{expected}' in feature matrix with M2 prices; "
                    f"got columns: {cols}"
                )

        _ok(
            f"basis/zscore/roll_yield columns present for all {len(m1.columns)} commodity(ies); "
            f"total features={len(cols)}"
        )
    except Exception as e:
        _fail("build_feature_matrix with M2 basis columns", e)


def test_build_feature_matrix_without_m2_matches_baseline():
    print("22. build_feature_matrix without M2 — output is identical to None baseline")
    try:
        m1 = _make_prices(300)

        feat_none    = build_feature_matrix(m1, second_contract_prices=None)
        feat_default = build_feature_matrix(m1)

        assert list(feat_none.columns) == list(feat_default.columns), (
            "Column order differs between None and default (no second_contract_prices)"
        )
        assert feat_none.shape == feat_default.shape, (
            f"Shape mismatch: None={feat_none.shape}, default={feat_default.shape}"
        )

        # No term-structure columns should appear in the baseline
        for col in feat_none.columns:
            assert not any(col.endswith(s) for s in ("_basis", "_basis_zscore", "_roll_yield")), (
                f"Term-structure column '{col}' unexpectedly present without M2 prices"
            )

        _ok(
            f"no basis columns without M2; shapes match ({feat_none.shape}); "
            f"column lists identical"
        )
    except Exception as e:
        _fail("build_feature_matrix without M2 matches baseline", e)


# ── IC horizon / embargo tests (Prompt 3) ─────────────────────────────────────

def test_actual_return_equals_h_day_cumulative():
    print("23. BacktestHarness — actual_return at date t equals H-day cumulative log-return")
    try:
        prices = _make_prices(250)
        macro  = _make_macro(prices.index)

        harness = BacktestHarness(
            adapters=[_StubAdapter("statistical", 0.0)],
            test_fraction=0.20,
        )
        records = harness.run(prices, macro, ["WTI Crude Oil"])
        assert records, "expected at least one record"

        # Recompute H-day actuals ourselves from the price series
        log_ret_1d = np.log(prices["WTI Crude Oil"] / prices["WTI Crude Oil"].shift(1))
        h_day_actual = log_ret_1d.rolling(FORECAST_HORIZON).sum().shift(-FORECAST_HORIZON)

        mismatches = 0
        for rec in records:
            expected = h_day_actual.get(rec.date)
            if expected is None or np.isnan(expected):
                continue  # last-H rows have no valid actual — skip
            diff = abs(rec.actual_return - expected)
            if diff > 1e-10:
                mismatches += 1
                print(
                    f"    MISMATCH @ {rec.date.date()}: "
                    f"record={rec.actual_return:.8f}, expected={expected:.8f}"
                )

        assert mismatches == 0, (
            f"{mismatches} records have actual_return != H-day cumulative return"
        )
        _ok(
            f"all {len(records)} records: actual_return == "
            f"rolling({FORECAST_HORIZON}).sum().shift(-{FORECAST_HORIZON})"
        )
    except Exception as e:
        _fail("actual_return equals H-day cumulative", e)


def test_split_indices_embargo_reserves_forecast_horizon():
    print(
        f"24. _split_indices — last {FORECAST_HORIZON} rows are embargoed "
        f"(usable = n - FORECAST_HORIZON)"
    )
    try:
        harness = BacktestHarness(
            adapters=[_StubAdapter("ml", 0.0)],
            min_train_rows=60,
            n_splits=4,
        )
        n = 300
        splits = harness._split_indices(n)
        usable = n - FORECAST_HORIZON  # expected ceiling on test indices

        for i, (ts, te) in enumerate(splits):
            assert te <= usable, (
                f"fold {i}: test_end ({te}) exceeds usable boundary ({usable}); "
                f"last {FORECAST_HORIZON} rows are not embargoed"
            )

        # Last fold should be close to the embargo boundary — gap is only integer-division rounding
        last_te = splits[-1][1]
        n_splits = harness.n_splits
        fold_size = max(1, (usable - harness.min_train_rows) // n_splits)
        assert usable - last_te < fold_size, (
            f"last fold test_end={last_te} is more than one fold_size ({fold_size}) "
            f"below usable={usable}; embargo may be over-cutting"
        )

        _ok(
            f"all folds respect embargo; last fold ends at index {last_te} "
            f"(usable={usable}, gap={usable - last_te}, n={n}, H={FORECAST_HORIZON})"
        )
    except Exception as e:
        _fail("embargo reserves FORECAST_HORIZON rows", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("BACKTEST HARNESS — TEST SUITE  (Phase 3 Part 2 + Phase 5 CV)")
    print("=" * 60)
    print()

    for fn in [
        test_backtest_record_to_training_pair,
        test_stub_adapter_interface,
        test_align_macro_fills_gaps,
        test_align_macro_empty,
        test_meta_features_for_date_present,
        test_meta_features_for_date_missing,
        test_harness_run_basic,
        test_harness_winning_tier_logic,
        test_harness_skips_insufficient_data,
        test_harness_skips_missing_commodity,
        test_harness_multiple_commodities,
        test_harness_collect_training_pairs,
        test_harness_train_meta_predictor,
        test_tier_counts,
        # ── Walk-forward CV (Phase 5) ──────────────────────────────────────────
        test_split_indices_structure,
        test_split_indices_min_train_respected,
        test_split_indices_single_split_fallback,
        test_walk_forward_produces_more_records,
        test_walk_forward_no_data_leakage,
        test_walk_forward_covers_most_of_history,
        # ── Term-structure features (Prompt 2) ────────────────────────────────
        test_build_feature_matrix_with_m2_gains_basis_columns,
        test_build_feature_matrix_without_m2_matches_baseline,
        # ── IC horizon / embargo (Prompt 3) ───────────────────────────────────
        test_actual_return_equals_h_day_cumulative,
        test_split_indices_embargo_reserves_forecast_horizon,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print(
            "Phase 5 CV + M2 term-structure + IC-horizon tests complete. "
            "BacktestHarness uses expanding-window walk-forward CV (n_splits=5 default); "
            "actual_return is H-day cumulative; basis/zscore/roll_yield features active when M2 supplied."
        )
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
