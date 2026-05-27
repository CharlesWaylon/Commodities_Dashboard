"""
Tests for the Step-2 trigger-blending behavior in cascade_orchestrator.

Coverage
────────
 1. Regression: with no active triggers, the snapshot is byte-identical to the
    pre-Step-2 implementation (the legacy last-value extraction).
 2. Regression: with the feature flag off, snapshot is identical even when
    high-strength triggers are present in the DB.
 3. Amplification: a synthetic fed_tightening @ strength=0.9 amplifies the
    rate-related features (dxy_ret, tlt_ret, tlt_yield_proxy) by ≥10%.
 4. Amplification: weaker trigger (strength < 0.5) does NOT amplify anything.
 5. Family routing: an opec_action trigger does NOT touch dxy_ret/tlt_ret
    (commodity-shock families have no snapshot analogue — propagation happens
    via upstream_shocks in Step 4).
 6. regime_hint: high-strength fed_tightening trigger surfaces as
    self._regime_hint = "rate_shock".
 7. Failure isolation: if get_active_triggers raises, the snapshot is returned
    unchanged (non-fatal degradation).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import cascade_orchestrator as co


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def macro_df():
    """5 business days of synthetic macro returns."""
    idx = pd.date_range("2026-05-12", periods=5, freq="B")
    return pd.DataFrame({
        "dxy_ret":         [0.001, 0.002, 0.003, 0.004, 0.005],
        "vix_ret5d":       [-0.01, 0.02, 0.03, 0.04, 0.05],
        "tlt_ret":         [-0.002, -0.001, 0.000, 0.001, 0.002],
        "tlt_yield_proxy": [0.01, 0.02, 0.015, 0.018, 0.020],
    }, index=idx)


@pytest.fixture
def forecaster():
    return co.CascadeForecaster()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MACRO_TRIGGERS_ENABLED", "true")
    yield


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv("MACRO_TRIGGERS_ENABLED", "false")
    yield


def _patch_triggers(monkeypatch, triggers):
    """Replace features.macro_features.get_active_triggers with a fixed list."""
    import features.macro_features as mf
    monkeypatch.setattr(mf, "get_active_triggers", lambda *a, **kw: list(triggers))


# ── 1. No-trigger regression: byte-identical to legacy ────────────────────────

def test_snapshot_unchanged_when_no_active_triggers(forecaster, macro_df, flag_on, monkeypatch):
    _patch_triggers(monkeypatch, [])
    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    # Legacy behavior: latest non-null value per column, plus derived tlt_yield_proxy.
    assert snap == {
        "dxy_ret":         0.005,
        "vix_ret5d":       0.05,
        "tlt_ret":         0.002,
        "tlt_yield_proxy": 0.020,
    }


# ── 2. Feature-flag-off regression ────────────────────────────────────────────

def test_snapshot_unchanged_when_flag_off(forecaster, macro_df, flag_off, monkeypatch):
    # Even with a strong trigger active, the flag prevents any blending.
    _patch_triggers(monkeypatch, [
        {"family": "fed_tightening", "strength": 0.95, "trigger_date": "2026-05-15"},
    ])
    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    assert snap == {
        "dxy_ret":         0.005,
        "vix_ret5d":       0.05,
        "tlt_ret":         0.002,
        "tlt_yield_proxy": 0.020,
    }


# ── 3. Spec acceptance: fed_tightening amplifies rate features ≥ 10% ──────────

def test_fed_tightening_amplifies_rate_features(forecaster, macro_df, flag_on, monkeypatch):
    _patch_triggers(monkeypatch, [
        {"family": "fed_tightening", "strength": 0.9, "trigger_date": "2026-05-15"},
    ])
    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())

    expected_factor = 1.0 + co._TRIGGER_AMPLIFY_COEFF * 0.9   # 1.45
    assert snap["dxy_ret"]         == pytest.approx(0.005 * expected_factor)
    assert snap["tlt_ret"]         == pytest.approx(0.002 * expected_factor)
    assert snap["tlt_yield_proxy"] == pytest.approx(0.020 * expected_factor)
    # vix is not in fed_tightening's feature map → untouched.
    assert snap["vix_ret5d"]       == pytest.approx(0.05)
    # Spec acceptance bar: ≥10% amplification on rate-related features.
    assert (snap["dxy_ret"] - 0.005) / 0.005 >= 0.10


# ── 4. Sub-threshold strength does nothing ────────────────────────────────────

def test_sub_threshold_trigger_does_not_amplify(forecaster, macro_df, flag_on, monkeypatch):
    _patch_triggers(monkeypatch, [
        {"family": "fed_tightening", "strength": 0.4, "trigger_date": "2026-05-15"},  # < 0.5 floor
    ])
    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    assert snap == {
        "dxy_ret":         0.005,
        "vix_ret5d":       0.05,
        "tlt_ret":         0.002,
        "tlt_yield_proxy": 0.020,
    }


# ── 5. Commodity-shock families have no snapshot analogue ─────────────────────

def test_commodity_shock_family_does_not_touch_macro_snapshot(forecaster, macro_df, flag_on, monkeypatch):
    # OPEC is intentionally absent from TRIGGER_FAMILY_TO_MACRO_FEATURES; its
    # influence propagates via upstream_shocks in Step 4 of the spec.
    _patch_triggers(monkeypatch, [
        {"family": "opec_action", "strength": 0.95, "trigger_date": "2026-05-15"},
    ])
    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    assert snap == {
        "dxy_ret":         0.005,
        "vix_ret5d":       0.05,
        "tlt_ret":         0.002,
        "tlt_yield_proxy": 0.020,
    }
    # …but the trigger was still recorded for downstream use.
    assert forecaster._active_triggers == [
        {"family": "opec_action", "strength": 0.95, "trigger_date": "2026-05-15"},
    ]


# ── 6. regime_hint derivation ─────────────────────────────────────────────────

def test_regime_hint_from_high_strength_fed_tightening(forecaster, macro_df, flag_on, monkeypatch):
    _patch_triggers(monkeypatch, [
        {"family": "fed_tightening", "strength": 0.85, "trigger_date": "2026-05-15"},
    ])
    # _extract_macro_snapshot populates _active_triggers as a side effect.
    forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    assert forecaster._derive_regime_hint(macro_df.index[-1].date()) == "rate_shock"


def test_regime_hint_neutral_when_flag_off(forecaster, macro_df, flag_off, monkeypatch):
    _patch_triggers(monkeypatch, [
        {"family": "fed_tightening", "strength": 0.99, "trigger_date": "2026-05-15"},
    ])
    forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    assert forecaster._derive_regime_hint(macro_df.index[-1].date()) == "neutral"


# ── 7. Failure isolation ──────────────────────────────────────────────────────

def test_get_active_triggers_failure_is_nonfatal(forecaster, macro_df, flag_on, monkeypatch):
    import features.macro_features as mf
    def boom(*a, **kw):
        raise RuntimeError("DB unavailable")
    monkeypatch.setattr(mf, "get_active_triggers", boom)

    snap = forecaster._extract_macro_snapshot(macro_df, forecast_date=macro_df.index[-1].date())
    # Snapshot falls back to legacy values; no exception bubbles up.
    assert snap == {
        "dxy_ret":         0.005,
        "vix_ret5d":       0.05,
        "tlt_ret":         0.002,
        "tlt_yield_proxy": 0.020,
    }
