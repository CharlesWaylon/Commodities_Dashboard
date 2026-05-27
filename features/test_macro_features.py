"""
Tests for features/macro_features.py — the shared macro feature surface.

Coverage
────────
 1. Public API exists and has the expected signatures
 2. get_macro_state_at returns the locked schema keys
 3. regime_hint = "neutral" on a quiet date (no high-strength triggers)
 4. regime_hint = "rate_shock" when a CPI release fires at strength ≥ 0.8
 5. regime_hint = "commodity_shock" when an OPEC trigger fires at strength ≥ 0.8
 6. Sub-threshold triggers (strength < 0.8) do NOT override regime
 7. Highest-strength trigger wins when multiple are active
 8. get_active_triggers sorts by strength descending
 9. build_macro_surprise_features returns the documented z-score keys
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import macro_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear lru_caches between tests so monkeypatched data is honoured."""
    macro_features.clear_caches()
    yield
    macro_features.clear_caches()


@pytest.fixture
def fake_history(monkeypatch):
    """Replace the FRED+TLT fetch with a deterministic 3-year synthetic frame."""
    idx = pd.date_range("2023-01-01", "2026-05-20", freq="B")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "dxy":       100 + rng.normal(0, 0.5, len(idx)).cumsum(),
        "vix":       18 + rng.normal(0, 0.3, len(idx)).cumsum().clip(min=-10),
        "tlt":       95 + rng.normal(0, 0.4, len(idx)).cumsum(),
        "tlt_yield": 4.0 + rng.normal(0, 0.05, len(idx)).cumsum().clip(min=-3),
        "cpi":       300 + np.linspace(0, 30, len(idx)),
        "unrate":    4.0 + rng.normal(0, 0.05, len(idx)).cumsum(),
        "fedfunds":  5.0 + np.linspace(0, 0.5, len(idx)),
        "t10y2y":    rng.normal(0, 0.05, len(idx)).cumsum(),
        "wti":       75 + rng.normal(0, 1.0, len(idx)).cumsum(),
    }, index=idx)
    monkeypatch.setattr(macro_features, "_load_macro_history", lambda *a, **kw: df)
    return df


def _patch_triggers(monkeypatch, triggers_for_date: dict[str, list[dict]]):
    """Install a fake get_active_triggers that returns canned rows per date."""
    def fake(date, lookback_days=5):
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows = list(triggers_for_date.get(key, []))
        rows.sort(key=lambda d: d["strength"], reverse=True)
        return rows
    monkeypatch.setattr(macro_features, "get_active_triggers", fake)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_public_api_present():
    for name in ("get_macro_state_at", "get_active_triggers", "build_macro_surprise_features"):
        assert callable(getattr(macro_features, name)), f"missing public function: {name}"


def test_macro_state_schema(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {})
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    expected = {
        "dxy_ret_5d", "vix_ret_5d", "tlt_ret_5d", "tlt_yield_proxy",
        "cpi_zscore", "unrate_zscore", "fedfunds_zscore",
        "t10y2y_level", "t10y2y_change_5d", "wti_ret_5d", "regime_hint",
    }
    assert set(state.keys()) == expected
    assert isinstance(state["regime_hint"], str)


def test_quiet_date_yields_neutral_regime(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {})
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    assert state["regime_hint"] == "neutral"


def test_cpi_surprise_maps_to_rate_shock(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {
        "2026-04-15": [{"family": "cpi_release", "strength": 0.92,
                        "trigger_date": "2026-04-14", "deviation_score": 0.92,
                        "source": "trigger_events"}],
    })
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    assert state["regime_hint"] == "rate_shock"


def test_opec_active_maps_to_commodity_shock(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {
        "2026-04-15": [{"family": "opec_action", "strength": 0.85,
                        "trigger_date": "2026-04-13", "deviation_score": 0.85,
                        "source": "trigger_events"}],
    })
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    assert state["regime_hint"] == "commodity_shock"


def test_sub_threshold_trigger_does_not_override(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {
        "2026-04-15": [{"family": "opec_action", "strength": 0.75,    # below 0.8
                        "trigger_date": "2026-04-13", "deviation_score": 0.75,
                        "source": "trigger_events"}],
    })
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    assert state["regime_hint"] == "neutral"


def test_highest_strength_trigger_wins(fake_history, monkeypatch):
    _patch_triggers(monkeypatch, {
        "2026-04-15": [
            {"family": "cpi_release", "strength": 0.81,
             "trigger_date": "2026-04-14", "deviation_score": 0.81, "source": "trigger_events"},
            {"family": "opec_action", "strength": 0.95,
             "trigger_date": "2026-04-13", "deviation_score": 0.95, "source": "trigger_events"},
        ],
    })
    state = macro_features.get_macro_state_at(pd.Timestamp("2026-04-15"))
    assert state["regime_hint"] == "commodity_shock"


def test_get_active_triggers_sorted_desc(monkeypatch):
    """Smoke test the real get_active_triggers — DB unavailability returns []."""
    # If the DB is reachable, the result is a list sorted by strength desc.
    # If not, we get an empty list. Either way: callable, returns list.
    result = macro_features.get_active_triggers(pd.Timestamp.now())
    assert isinstance(result, list)
    if len(result) >= 2:
        strengths = [r["strength"] for r in result]
        assert strengths == sorted(strengths, reverse=True)


def test_surprise_features_schema(fake_history):
    feats = macro_features.build_macro_surprise_features(pd.Timestamp("2026-04-15"))
    assert set(feats.keys()) == {
        "cpi_surprise_z", "unrate_surprise_z", "fedfunds_surprise_z",
        "t10y2y_surprise_z", "wti_surprise_z",
    }
    # Synthetic CPI is a monotonic ramp — z-score of the latest point should be positive.
    assert feats["cpi_surprise_z"] > 0
