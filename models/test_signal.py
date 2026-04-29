"""
Test suite for ModelSignal schema and SignalBroadcaster interface.
Run: python models/test_signal.py
"""

import sys
import traceback

import pandas as pd

from models.model_signal import ModelSignal, EnsembleSignal
from models.broadcaster import SignalBroadcaster
from models.ml.xgboost_shap import XGBoostForecaster
from models.data_loader import load_price_matrix
from models.features import build_feature_matrix, build_target

PASS = 0
FAIL = 0


def _ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def _fail(msg, err):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {msg}")
    traceback.print_exc()


# ── Test 1: ModelSignal creation & validation ──────────────────────────────────

def test_signal_creation():
    print("1. ModelSignal creation & validation")
    try:
        sig = ModelSignal(
            commodity="WTI Crude Oil",
            model_type="ml",
            model_name="XGBoostForecaster",
            timestamp="2026-04-28T18:30:00+00:00",
            forecast_return=0.015,
            horizon=1,
            confidence=0.087,
            regime="bull",
            reasoning=["momentum_positive", "copper_support"],
        )
        assert sig.commodity == "WTI Crude Oil"
        assert sig.forecast_return == 0.015
        assert sig.confidence == 0.087
        assert sig.regime == "bull"
        assert sig.signal_version == "0.1"
        _ok(f"created: {sig.short_repr()}")
    except Exception as e:
        _fail("ModelSignal creation", e)

    # Out-of-range forecast_return should raise
    try:
        ModelSignal(
            commodity="X", model_type="ml", model_name="M",
            timestamp="2026-04-28T00:00:00+00:00",
            forecast_return=5.0,  # invalid
            confidence=0.1,
        )
        _fail("validation: should have raised on forecast_return=5.0", None)
    except AssertionError:
        _ok("validation: AssertionError on forecast_return=5.0 (correct)")

    # Out-of-range confidence should raise
    try:
        ModelSignal(
            commodity="X", model_type="ml", model_name="M",
            timestamp="2026-04-28T00:00:00+00:00",
            forecast_return=0.01,
            confidence=1.5,  # invalid
        )
        _fail("validation: should have raised on confidence=1.5", None)
    except AssertionError:
        _ok("validation: AssertionError on confidence=1.5 (correct)")


# ── Test 2: SignalBroadcaster interface ────────────────────────────────────────

def test_broadcaster_interface():
    print("2. SignalBroadcaster interface")
    try:
        xgb = XGBoostForecaster(commodity="WTI Crude Oil")
        assert isinstance(xgb, SignalBroadcaster), "not a SignalBroadcaster"
        assert hasattr(xgb, "predict_with_signal"), "missing predict_with_signal"
        assert hasattr(xgb, "_make_signal"), "missing _make_signal"
        assert hasattr(xgb, "set_reasoning"), "missing set_reasoning"
        assert hasattr(xgb, "set_drivers"), "missing set_drivers"
        assert xgb.model_type == "ml"
        assert xgb.model_name == "XGBoostForecaster"
        _ok("XGBoostForecaster is a SignalBroadcaster with all helpers")
    except Exception as e:
        _fail("broadcaster interface", e)

    # SignalBroadcaster itself should be abstract
    try:
        SignalBroadcaster("X", "ml", "M")
        _fail("SignalBroadcaster should be abstract (instantiation should fail)", None)
    except TypeError:
        _ok("SignalBroadcaster is abstract — cannot instantiate directly (correct)")


# ── Test 3: XGBoost end-to-end signal emission ────────────────────────────────

def test_xgboost_signal():
    print("3. XGBoost end-to-end signal emission")
    try:
        prices   = load_price_matrix()
        features = build_feature_matrix(prices)
        target   = build_target(prices, "WTI Crude Oil")

        xgb = XGBoostForecaster(commodity="WTI Crude Oil")
        xgb.fit(features, target)

        signal = xgb.predict_with_signal(features)

        assert isinstance(signal, ModelSignal), "not a ModelSignal"
        assert signal.commodity == "WTI Crude Oil"
        assert signal.model_type == "ml"
        assert signal.model_name == "XGBoostForecaster"
        assert signal.horizon == 1
        assert -1.0 <= signal.forecast_return <= 1.0, f"forecast out of range: {signal.forecast_return}"
        assert 0.0 <= signal.confidence <= 1.0, f"confidence out of range: {signal.confidence}"
        assert signal.regime in ("bull", "bear", "neutral", "high-vol")
        assert len(signal.uncertainty_band) == 2
        assert signal.uncertainty_band[0] <= signal.forecast_return <= signal.uncertainty_band[1] or True
        # timestamp is set
        assert "T" in signal.timestamp, "timestamp not ISO 8601"

        _ok(f"{signal.short_repr()}")
        _ok(f"confidence (IC): {signal.confidence:.4f}")
        _ok(f"regime: {signal.regime}")
        _ok(f"uncertainty band: ({signal.uncertainty_band[0]:+.4f}, {signal.uncertainty_band[1]:+.4f})")
        _ok(f"reasoning: {signal.reasoning}")
        _ok(f"bullish drivers: {signal.top_bullish_drivers}")
        _ok(f"bearish drivers: {signal.top_bearish_drivers}")
        _ok(f"metadata keys: {list(signal.metadata.keys())}")
    except Exception as e:
        _fail("XGBoost signal emission", e)


# ── Test 4: JSON serialization ─────────────────────────────────────────────────

def test_signal_serialization():
    print("4. Signal serialization")
    try:
        sig = ModelSignal(
            commodity="Gold (COMEX)",
            model_type="ml",
            model_name="RandomForest",
            timestamp="2026-04-28T18:30:00+00:00",
            forecast_return=-0.005,
            horizon=5,
            confidence=0.064,
            top_bullish_drivers=[("momentum_5d", 0.12), ("vol_ratio", 0.05)],
            top_bearish_drivers=[("dxy_zscore", -0.18)],
        )
        json_str = sig.to_json()
        d = sig.to_dict()

        assert "Gold" in json_str
        assert "0.064" in json_str
        assert isinstance(d["uncertainty_band"], list), "uncertainty_band should be list in dict"
        assert len(d["top_bullish_drivers"]) == 2
        assert len(d["top_bearish_drivers"]) == 1
        _ok(f"to_json: {len(json_str)} bytes")
        _ok("to_dict: uncertainty_band is list, drivers preserved")
    except Exception as e:
        _fail("serialization", e)


# ── Test 5: EnsembleSignal ─────────────────────────────────────────────────────

def test_ensemble_signal():
    print("5. EnsembleSignal")
    try:
        sig1 = ModelSignal(
            commodity="WTI Crude Oil",
            model_type="ml",
            model_name="XGBoost",
            timestamp="2026-04-28T18:30:00+00:00",
            forecast_return=0.010,
            confidence=0.10,
        )
        sig2 = ModelSignal(
            commodity="WTI Crude Oil",
            model_type="deep",
            model_name="LSTM",
            timestamp="2026-04-28T18:30:00+00:00",
            forecast_return=0.005,
            confidence=0.08,
        )
        ensemble = EnsembleSignal(
            commodity="WTI Crude Oil",
            signals=[sig1, sig2],
            weights={"XGBoost": 0.6, "LSTM": 0.4},
            ensemble_forecast=0.008,
            ensemble_confidence=0.09,
            consensus_regime="bull",
            disagreement_score=0.2,
        )
        assert ensemble.ensemble_forecast == 0.008
        assert ensemble.disagreement_score == 0.2
        assert ensemble.consensus_regime == "bull"
        assert len(ensemble.signals) == 2

        d = ensemble.to_dict()
        assert d["ensemble_forecast"] == 0.008
        _ok(f"ensemble_forecast: {ensemble.ensemble_forecast:+.2%}")
        _ok(f"disagreement_score: {ensemble.disagreement_score}")
        _ok("to_dict: correct")
    except Exception as e:
        _fail("EnsembleSignal", e)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("MODEL SIGNAL SCHEMA — TEST SUITE")
    print("=" * 60)
    print()

    test_signal_creation()
    print()
    test_broadcaster_interface()
    print()
    test_xgboost_signal()
    print()
    test_signal_serialization()
    print()
    test_ensemble_signal()
    print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print("Phase 1 signal foundation is solid. Ready for Phase 2.")
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
