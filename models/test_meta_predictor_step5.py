"""
Step-5 tests for models/meta_predictor.py — surprise features + regime_hint
one-hots flow correctly into MetaFeatures.

Coverage
────────
 1. FEATURE_COLUMNS contains the seven new column names in order.
 2. MetaFeatures defaults populate the new fields to 0.0.
 3. collect_meta_features populates surprise z-scores from
    build_macro_surprise_features when that helper returns values.
 4. collect_meta_features populates t10y2y_change_5d from get_macro_state_at.
 5. regime_hint = "rate_shock" → one-hot fires on the rate_shock column only.
 6. regime_hint = "neutral" → all three one-hot columns are 0.
 7. to_feature_vector matches the FEATURE_COLUMNS order and length.
 8. load() with a feature_columns mismatch falls back to untrained state
    rather than raising (graceful degradation during Step-5 rollout).
"""

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import meta_predictor as mp


# ── 1. Schema ────────────────────────────────────────────────────────────────

def test_feature_columns_include_step5_fields():
    expected_new = (
        "cpi_surprise_z",
        "unrate_surprise_z",
        "fedfunds_surprise_z",
        "t10y2y_change_5d",
        "regime_hint_onehot_rate_shock",
        "regime_hint_onehot_growth_shock",
        "regime_hint_onehot_commodity_shock",
    )
    for col in expected_new:
        assert col in mp.FEATURE_COLUMNS, f"FEATURE_COLUMNS missing {col!r}"


# ── 2. MetaFeatures defaults ──────────────────────────────────────────────────

def test_meta_features_defaults_for_step5_fields():
    f = mp.MetaFeatures()
    assert f.cpi_surprise_z      == 0.0
    assert f.unrate_surprise_z   == 0.0
    assert f.fedfunds_surprise_z == 0.0
    assert f.t10y2y_change_5d    == 0.0
    assert f.regime_hint_onehot_rate_shock      == 0.0
    assert f.regime_hint_onehot_growth_shock    == 0.0
    assert f.regime_hint_onehot_commodity_shock == 0.0


# ── 3-6. collect_meta_features uses macro_features helpers ────────────────────

@pytest.fixture
def macro_df_single_row():
    return pd.DataFrame(
        {"vix": [16.0]},
        index=pd.DatetimeIndex(["2026-05-26"]),
    )


def _patch_macro_features(monkeypatch, surprise: dict, state: dict):
    import features.macro_features as feat
    monkeypatch.setattr(feat, "build_macro_surprise_features", lambda *a, **kw: surprise)
    monkeypatch.setattr(feat, "get_macro_state_at",            lambda *a, **kw: state)


def test_collect_populates_surprise_features(monkeypatch, macro_df_single_row):
    _patch_macro_features(monkeypatch,
        surprise={
            "cpi_surprise_z":      1.4,
            "unrate_surprise_z":   -0.3,
            "fedfunds_surprise_z": 0.8,
            "t10y2y_surprise_z":   0.0,
            "wti_surprise_z":      0.0,
        },
        state={
            "t10y2y_change_5d": 0.05,
            "regime_hint":      "neutral",
        },
    )
    mf = mp.collect_meta_features(macro_df_single_row)
    assert mf.cpi_surprise_z      == pytest.approx(1.4)
    assert mf.unrate_surprise_z   == pytest.approx(-0.3)
    assert mf.fedfunds_surprise_z == pytest.approx(0.8)
    assert mf.t10y2y_change_5d    == pytest.approx(0.05)


def test_collect_populates_rate_shock_onehot(monkeypatch, macro_df_single_row):
    _patch_macro_features(monkeypatch,
        surprise={},
        state={"t10y2y_change_5d": 0.0, "regime_hint": "rate_shock"},
    )
    mf = mp.collect_meta_features(macro_df_single_row)
    assert mf.regime_hint_onehot_rate_shock      == 1.0
    assert mf.regime_hint_onehot_growth_shock    == 0.0
    assert mf.regime_hint_onehot_commodity_shock == 0.0


def test_collect_populates_commodity_shock_onehot(monkeypatch, macro_df_single_row):
    _patch_macro_features(monkeypatch,
        surprise={},
        state={"t10y2y_change_5d": 0.0, "regime_hint": "commodity_shock"},
    )
    mf = mp.collect_meta_features(macro_df_single_row)
    assert mf.regime_hint_onehot_rate_shock      == 0.0
    assert mf.regime_hint_onehot_growth_shock    == 0.0
    assert mf.regime_hint_onehot_commodity_shock == 1.0


def test_collect_neutral_regime_leaves_all_onehots_zero(monkeypatch, macro_df_single_row):
    _patch_macro_features(monkeypatch,
        surprise={},
        state={"t10y2y_change_5d": 0.0, "regime_hint": "neutral"},
    )
    mf = mp.collect_meta_features(macro_df_single_row)
    assert mf.regime_hint_onehot_rate_shock      == 0.0
    assert mf.regime_hint_onehot_growth_shock    == 0.0
    assert mf.regime_hint_onehot_commodity_shock == 0.0


def test_collect_swallows_macro_features_errors(monkeypatch, macro_df_single_row):
    """If FRED is unreachable / DB down, collect_meta_features still returns defaults."""
    import features.macro_features as feat
    def boom(*a, **kw):
        raise RuntimeError("FRED unreachable")
    monkeypatch.setattr(feat, "build_macro_surprise_features", boom)
    monkeypatch.setattr(feat, "get_macro_state_at",            boom)

    mf = mp.collect_meta_features(macro_df_single_row)
    assert mf.cpi_surprise_z == 0.0
    assert mf.regime_hint_onehot_rate_shock == 0.0


# ── 7. to_feature_vector preserves order and length ───────────────────────────

def test_feature_vector_length_matches_columns(monkeypatch, macro_df_single_row):
    _patch_macro_features(monkeypatch,
        surprise={"cpi_surprise_z": 2.0, "unrate_surprise_z": 0.0,
                  "fedfunds_surprise_z": 0.0, "t10y2y_surprise_z": 0.0, "wti_surprise_z": 0.0},
        state={"t10y2y_change_5d": 0.1, "regime_hint": "rate_shock"},
    )
    mf = mp.collect_meta_features(macro_df_single_row)
    fv = mf.to_feature_vector()
    assert len(fv) == len(mp.FEATURE_COLUMNS)
    # cpi_surprise_z column index → fv value matches
    i_cpi = list(mp.FEATURE_COLUMNS).index("cpi_surprise_z")
    assert fv[i_cpi] == pytest.approx(2.0)
    # rate_shock one-hot fires
    i_rate = list(mp.FEATURE_COLUMNS).index("regime_hint_onehot_rate_shock")
    assert fv[i_rate] == 1.0


# ── 8. load() compatibility check ─────────────────────────────────────────────

def test_load_rejects_mismatched_feature_columns(tmp_path):
    """Old pkl (fewer columns) should be silently ignored, predictor stays untrained."""
    fake_pkl = tmp_path / "old_meta.pkl"
    state = {
        "tree":                None,        # any value — load() should bail before using it
        "classes":              ["ml"],
        "feature_importances":  {},
        "training_records":     100,
        "feature_columns":      ["vix"],    # deliberately shorter than current FEATURE_COLUMNS
    }
    with open(fake_pkl, "wb") as fh:
        pickle.dump(state, fh)

    pred = mp.MetaPredictor()
    pred.load(fake_pkl)
    assert not pred.is_trained, "predictor should refuse to load mismatched pkl"
    assert pred._tree is None
