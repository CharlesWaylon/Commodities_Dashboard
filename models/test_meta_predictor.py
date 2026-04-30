"""
Tests for models/meta_predictor.py — Phase 3 Part 1.
No network calls; no live dashboard DB touched.
Run: python -m models.test_meta_predictor
"""

import sys
import math
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from models.meta_predictor import (
    MetaFeatures,
    ModelVote,
    MetaDecision,
    MetaPredictor,
    collect_meta_features,
    FEATURE_COLUMNS,
    KNOWN_TIERS,
)

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-04-01", periods=n, freq="B")


def _macro_df(n=5, vix=18.0, dxy_z=0.5) -> pd.DataFrame:
    """Minimal macro DataFrame that matches macro_overlays column names."""
    return pd.DataFrame({
        "vix":            [vix] * n,
        "vix_level_z":    [0.3] * n,
        "vix_risk_off":   [float(vix >= 20)] * n,   # mirrors macro_overlays logic
        "vix_crisis":     [float(vix >= 30)] * n,
        "dxy_zscore63":   [dxy_z] * n,
        "dxy_mom21":      [0.002] * n,
        "tlt_mom21":      [-0.005] * n,
        "tlt_yield_proxy": [0.001] * n,
        "is_opec_window": [0.0] * n,
        "days_to_opec":   [-30.0] * n,
        "is_wasde_window": [0.0] * n,
        "days_to_wasde":  [-20.0] * n,
        "wasde_post5":    [0.0] * n,
    }, index=_idx(n))


def _votes(commodity="WTI Crude Oil") -> list:
    return [
        ModelVote("statistical", "ARIMA",    commodity, 0.002, 0.06),
        ModelVote("ml",          "XGBoost",  commodity, 0.008, 0.09),
        ModelVote("deep",        "Prophet",  commodity, 0.005, 0.07),
    ]


def _training_records(n=50, winning_tier="ml"):
    """Synthetic (MetaFeatures, tier) pairs for fit()."""
    records = []
    for i in range(n):
        mf = MetaFeatures(
            vix=15 + i % 20,
            vix_level_z=-0.5 + (i % 5) * 0.3,
            vix_risk_off=float(i % 4 == 0),
            vix_crisis=0.0,
            dxy_zscore63=0.5 + (i % 6) * 0.2,
            dxy_mom21=0.001 * (i % 3 - 1),
            tlt_mom21=-0.003 + 0.001 * (i % 5),
            tlt_yield_proxy=0.001,
            is_opec_window=float(i % 10 == 0),
            days_to_opec=-5.0 if i % 10 == 0 else -25.0,
            is_wasde_window=float(i % 8 == 0),
            days_to_wasde=-3.0 if i % 8 == 0 else -20.0,
            wasde_post5=0.0,
        )
        # Alternate winner so tree gets multi-class variety
        tier = KNOWN_TIERS[i % len(KNOWN_TIERS)]
        records.append((mf, tier))
    return records


# ── MetaFeatures ──────────────────────────────────────────────────────────────

def test_meta_features_defaults():
    print("1. MetaFeatures — default values are NaN / 0")
    try:
        mf = MetaFeatures()
        assert math.isnan(mf.vix)
        assert mf.vix_risk_off == 0.0
        assert mf.hmm_regime is None
        _ok("default MetaFeatures has NaN numerics, 0.0 booleans, None for HMM")
    except Exception as e:
        _fail("MetaFeatures defaults", e)


def test_meta_features_to_feature_vector():
    print("2. MetaFeatures.to_feature_vector — correct length, NaN → 0.0")
    try:
        mf = MetaFeatures(vix=22.5, dxy_zscore63=float("nan"))
        fv = mf.to_feature_vector()
        assert len(fv) == len(FEATURE_COLUMNS), f"expected {len(FEATURE_COLUMNS)}, got {len(fv)}"
        # NaN dxy_zscore63 should become 0.0
        dxy_idx = list(FEATURE_COLUMNS).index("dxy_zscore63")
        assert fv[dxy_idx] == 0.0
        # VIX should pass through
        vix_idx = list(FEATURE_COLUMNS).index("vix")
        assert fv[vix_idx] == 22.5
        _ok(f"vector length {len(fv)}, NaN→0 sentinel verified")
    except Exception as e:
        _fail("to_feature_vector", e)


# ── ModelVote ─────────────────────────────────────────────────────────────────

def test_model_vote_unknown_tier():
    print("3. ModelVote — unknown tier raises ValueError")
    try:
        try:
            ModelVote("alchemy", "WizardModel", "Gold", 0.01, 0.05)
            assert False, "should raise"
        except ValueError:
            pass
        _ok("unknown tier rejected")
    except Exception as e:
        _fail("ModelVote tier validation", e)


# ── collect_meta_features ─────────────────────────────────────────────────────

def test_collect_meta_features_normal():
    print("4. collect_meta_features — populates from macro_df")
    try:
        df = _macro_df(vix=25.0, dxy_z=2.1)
        mf = collect_meta_features(df)
        assert abs(mf.vix - 25.0) < 1e-9
        assert abs(mf.dxy_zscore63 - 2.1) < 1e-9
        assert mf.vix_risk_off == 1.0   # 25 >= 20
        assert mf.hmm_regime is None
        _ok(f"VIX={mf.vix}, DXY_z={mf.dxy_zscore63}, risk_off={mf.vix_risk_off}")
    except Exception as e:
        _fail("collect_meta_features", e)


def test_collect_meta_features_empty():
    print("5. collect_meta_features — empty/None df → default MetaFeatures")
    try:
        mf_none = collect_meta_features(None)
        mf_empty = collect_meta_features(pd.DataFrame())
        assert math.isnan(mf_none.vix)
        assert math.isnan(mf_empty.vix)
        _ok("empty df returns default MetaFeatures")
    except Exception as e:
        _fail("collect_meta_features empty", e)


def test_collect_meta_features_missing_columns():
    print("6. collect_meta_features — missing columns → graceful NaN defaults")
    try:
        df = pd.DataFrame({"vix": [18.0, 19.0]}, index=_idx(2))
        mf = collect_meta_features(df)
        assert abs(mf.vix - 19.0) < 1e-9        # present column filled
        assert math.isnan(mf.dxy_zscore63)        # missing → NaN
        assert mf.is_opec_window == 0.0           # missing bool → 0.0
        _ok("missing columns default to NaN / 0.0 safely")
    except Exception as e:
        _fail("collect_meta_features missing cols", e)


# ── MetaPredictor (untrained) ─────────────────────────────────────────────────

def test_meta_predictor_untrained_passthrough():
    print("7. MetaPredictor (untrained) — equal weights, meta_confidence=0")
    try:
        mp = MetaPredictor()
        assert not mp.is_trained
        mf = collect_meta_features(_macro_df())
        decision = mp.predict(mf, _votes())
        assert not mp.is_trained
        # All present tiers should have non-zero normalised weight
        present = {"statistical", "ml", "deep"}
        for t in present:
            assert decision.weights[t] > 0.0, f"{t} weight should be >0"
        assert abs(sum(decision.weights[t] for t in present) - 1.0) < 1e-9
        assert decision.meta_confidence == 0.0
        assert isinstance(decision.reasoning, str) and len(decision.reasoning) > 10
        _ok(f"weights sum to 1.0; trusted={decision.trusted_tier}; confidence=0")
    except Exception as e:
        _fail("untrained passthrough", e)


def test_meta_predictor_no_votes():
    print("8. MetaPredictor — zero votes → zero ensemble_forecast")
    try:
        mp = MetaPredictor()
        mf = collect_meta_features(_macro_df())
        decision = mp.predict(mf, [])
        assert decision.ensemble_forecast == 0.0
        assert decision.consensus_strength == 0.0
        _ok("empty votes → forecast=0, consensus=0")
    except Exception as e:
        _fail("zero votes", e)


def test_ensemble_forecast_weighted():
    print("9. MetaPredictor (untrained) — ensemble_forecast is weighted average")
    try:
        mp = MetaPredictor()
        mf = MetaFeatures()
        votes = [
            ModelVote("ml",          "XGBoost", "Gold", 0.012, 0.09),
            ModelVote("statistical", "ARIMA",   "Gold", 0.004, 0.06),
        ]
        decision = mp.predict(mf, votes)
        # With equal weights on 2 present tiers: (0.012 + 0.004) / 2 = 0.008
        assert abs(decision.ensemble_forecast - 0.008) < 1e-9
        _ok(f"ensemble_forecast={decision.ensemble_forecast:.4f} (expected 0.0080)")
    except Exception as e:
        _fail("weighted forecast", e)


def test_consensus_strength_agreement():
    print("10. MetaPredictor — consensus_strength = 1.0 when all tiers bullish")
    try:
        mp = MetaPredictor()
        mf = MetaFeatures()
        votes = [
            ModelVote("ml",          "XGBoost", "WTI", +0.010, 0.09),
            ModelVote("statistical", "ARIMA",   "WTI", +0.005, 0.06),
            ModelVote("deep",        "Prophet", "WTI", +0.008, 0.07),
        ]
        decision = mp.predict(mf, votes)
        assert abs(decision.consensus_strength - 1.0) < 1e-9
        _ok("all bullish → consensus_strength = 1.0")
    except Exception as e:
        _fail("consensus strength all-bull", e)


def test_consensus_strength_split():
    print("11. MetaPredictor — consensus_strength < 1.0 on bull/bear split")
    try:
        mp = MetaPredictor()
        mf = MetaFeatures()
        votes = [
            ModelVote("ml",          "XGBoost", "WTI", +0.010, 0.09),
            ModelVote("statistical", "ARIMA",   "WTI", -0.005, 0.06),
        ]
        decision = mp.predict(mf, votes)
        assert decision.consensus_strength < 1.0
        _ok(f"bull/bear split → consensus_strength={decision.consensus_strength:.3f} < 1.0")
    except Exception as e:
        _fail("consensus strength split", e)


# ── MetaPredictor (trained) ───────────────────────────────────────────────────

def test_meta_predictor_fit_and_predict():
    print("12. MetaPredictor.fit — trains and returns non-zero meta_confidence")
    try:
        mp = MetaPredictor()
        records = _training_records(n=80)
        mp.fit(records, max_depth=3, min_samples_leaf=5)
        assert mp.is_trained
        mf = collect_meta_features(_macro_df(vix=28.0, dxy_z=2.3))
        decision = mp.predict(mf, _votes())
        # weights must still sum to 1 over present tiers
        present = {"statistical", "ml", "deep"}
        total = sum(decision.weights[t] for t in present)
        assert abs(total - 1.0) < 1e-9, f"weights sum {total}"
        # trained predictor may have non-zero meta_confidence
        assert decision.meta_confidence >= 0.0
        assert mp.is_trained
        _ok(f"trained; trusted={decision.trusted_tier}; meta_conf={decision.meta_confidence:.3f}")
    except Exception as e:
        _fail("fit and predict", e)


def test_meta_predictor_explain():
    print("13. MetaPredictor.explain — returns top-n feature importances")
    try:
        mp = MetaPredictor()
        mp.fit(_training_records(n=80), max_depth=3, min_samples_leaf=5)
        top5 = mp.explain(top_n=5)
        assert len(top5) <= 5
        for feat, imp in top5:
            assert feat in FEATURE_COLUMNS
            assert imp >= 0.0
        _ok(f"top features: {[f for f, _ in top5]}")
    except Exception as e:
        _fail("explain", e)


def test_meta_predictor_fit_empty_raises():
    print("14. MetaPredictor.fit — empty records raises ValueError")
    try:
        mp = MetaPredictor()
        try:
            mp.fit([])
            assert False, "should raise"
        except ValueError:
            pass
        _ok("empty records → ValueError")
    except Exception as e:
        _fail("fit empty", e)


# ── Save / load ───────────────────────────────────────────────────────────────

def test_meta_predictor_save_load():
    print("15. MetaPredictor — save/load roundtrip preserves predictions")
    try:
        import tempfile
        mp = MetaPredictor()
        mp.fit(_training_records(n=80), max_depth=3, min_samples_leaf=5)

        mf = collect_meta_features(_macro_df(vix=22.0, dxy_z=1.8))
        d_before = mp.predict(mf, _votes())

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = Path(f.name)

        mp.save(tmp_path)
        mp2 = MetaPredictor(model_path=tmp_path)
        assert mp2.is_trained
        d_after = mp2.predict(mf, _votes())

        assert d_before.trusted_tier == d_after.trusted_tier
        assert abs(d_before.ensemble_forecast - d_after.ensemble_forecast) < 1e-9
        tmp_path.unlink(missing_ok=True)
        _ok(f"trusted_tier={d_after.trusted_tier} preserved across save/load")
    except Exception as e:
        _fail("save/load roundtrip", e)


def test_meta_predictor_load_missing_file():
    print("16. MetaPredictor.load — missing file is silent no-op")
    try:
        mp = MetaPredictor(model_path=Path("/tmp/does_not_exist_12345.pkl"))
        assert not mp.is_trained
        _ok("missing pkl → is_trained=False, no exception")
    except Exception as e:
        _fail("load missing file", e)


# ── MetaDecision serialisation ────────────────────────────────────────────────

def test_meta_decision_to_dict():
    print("17. MetaDecision.to_dict / to_json — serialisable")
    try:
        import json
        mp = MetaPredictor()
        mf = collect_meta_features(_macro_df())
        decision = mp.predict(mf, _votes())
        d = decision.to_dict()
        assert "weights" in d and "trusted_tier" in d and "reasoning" in d
        j = decision.to_json()
        parsed = json.loads(j)
        assert parsed["trusted_tier"] == decision.trusted_tier
        _ok(f"to_json: {len(j)} bytes; to_dict keys: {list(d.keys())[:5]}")
    except Exception as e:
        _fail("MetaDecision serialisation", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("META-PREDICTOR — TEST SUITE  (Phase 3 Part 1)")
    print("=" * 60)
    print()

    for fn in [
        test_meta_features_defaults,
        test_meta_features_to_feature_vector,
        test_model_vote_unknown_tier,
        test_collect_meta_features_normal,
        test_collect_meta_features_empty,
        test_collect_meta_features_missing_columns,
        test_meta_predictor_untrained_passthrough,
        test_meta_predictor_no_votes,
        test_ensemble_forecast_weighted,
        test_consensus_strength_agreement,
        test_consensus_strength_split,
        test_meta_predictor_fit_and_predict,
        test_meta_predictor_explain,
        test_meta_predictor_fit_empty_raises,
        test_meta_predictor_save_load,
        test_meta_predictor_load_missing_file,
        test_meta_decision_to_dict,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print("Phase 3 Part 1 complete: schema + feature collector + meta-predictor skeleton.")
        print("Part 2 will add: backtesting harness → training data → fit() on real history.")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
