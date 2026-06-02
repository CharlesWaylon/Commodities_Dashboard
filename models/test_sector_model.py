"""
Tests for Step-4 dynamic upstream damping in models/sector_model.py.

The damping logic is unit-testable without fitting an XGBoost model — it lives
entirely inside ``_compute_upstream_adjustment``. These tests instantiate a
SectorSpecificModel, leave the XGB model untrained, and exercise the upstream
helper directly.

Coverage
────────
 1. Regression: no triggers → damping equals static UPSTREAM_DAMPING (0.25).
 2. Spec acceptance: opec_action @ strength 0.7 boosts the WTI→Copper path,
    Copper's upstream adjustment moves further than baseline.
 3. Per-path boost: only paths whose sector matches the trigger family are
    boosted; other upstream paths keep the static damping.
 4. Cap: damping never exceeds UPSTREAM_DAMPING + TRIGGER_BOOST_COEFF (0.50).
 5. Feature-flag off → no boost is applied (regression).
 6. Multiple triggers in the same family → top strength is used (not sum).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import sector_model as sm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def metals_model():
    """A SectorSpecificModel for a Metals commodity (Copper). No fit() needed."""
    return sm.SectorSpecificModel(sector="metals", commodity="Copper")


@pytest.fixture
def prices():
    idx = pd.date_range("2026-05-01", periods=10, freq="B")
    return pd.DataFrame(
        {"WTI Crude Oil": np.linspace(80, 82, 10),
         "Copper":        np.linspace(4.0, 4.05, 10)},
        index=idx,
    )


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    """Force-clear the route/corr caches so each test starts clean."""
    sm._routes_cache = None
    sm._corr_cache   = None
    # Force the upstream-correlation lookup to fall through to the default
    # cross-sector value (0.10) so behavior is deterministic.
    monkeypatch.setattr(sm, "_get_corr", lambda: pd.DataFrame())
    yield


# ── 1. Regression: no triggers → static damping ───────────────────────────────

def test_no_triggers_uses_static_damping(metals_model, prices):
    total, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=None,
    )
    prior = sm._economic_prior("energy", "metals")
    expected = sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * sm.UPSTREAM_DAMPING * prior
    assert detail["WTI Crude Oil"] == pytest.approx(expected, rel=1e-6)


def test_empty_trigger_list_uses_static_damping(metals_model, prices):
    total, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=[],
    )
    prior = sm._economic_prior("energy", "metals")
    expected = sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * sm.UPSTREAM_DAMPING * prior
    assert detail["WTI Crude Oil"] == pytest.approx(expected, rel=1e-6)


# ── 2. Spec acceptance: OPEC @ 0.7 boosts energy-sourced upstream path ────────

def test_opec_trigger_boosts_energy_upstream_path(metals_model, prices):
    triggers = [{"family": "opec_action", "strength": 0.7, "trigger_date": "2026-05-15"}]
    _, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=triggers,
    )
    prior = sm._economic_prior("energy", "metals")
    boosted_damping = sm.UPSTREAM_DAMPING + sm.TRIGGER_BOOST_COEFF * 0.7
    expected = round(sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * boosted_damping * prior, 6)
    assert detail["WTI Crude Oil"] == pytest.approx(expected, abs=1e-6)

    # Sanity: the boosted contribution is strictly larger than the baseline.
    baseline = sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * sm.UPSTREAM_DAMPING * prior
    assert detail["WTI Crude Oil"] > baseline


# ── 3. Per-path: non-matching upstream paths keep static damping ─────────────

def test_only_matching_sector_is_boosted(metals_model, prices):
    """
    A weather_shock trigger maps to 'agriculture'. Energy upstream paths should
    NOT be boosted (spec: per-path, not uniform).
    """
    triggers = [{"family": "weather_shock", "strength": 0.9, "trigger_date": "2026-05-15"}]
    _, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=triggers,
    )
    prior = sm._economic_prior("energy", "metals")
    expected_static = sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * sm.UPSTREAM_DAMPING * prior
    assert detail["WTI Crude Oil"] == pytest.approx(expected_static, rel=1e-6)


def test_mixed_upstream_paths_boosted_independently(prices):
    """
    Livestock receives upstream shocks from both Agriculture (matching family)
    and Energy (matching family). A weather_shock should boost the Ag path but
    not the Energy path.
    """
    model = sm.SectorSpecificModel(sector="livestock", commodity="Live Cattle")
    # Synthetic upstream shocks from two different sectors.
    triggers = [{"family": "weather_shock", "strength": 0.8, "trigger_date": "2026-05-15"}]
    _, detail = model._compute_upstream_adjustment(
        upstream_shocks={"Corn (CBOT)": 0.01, "WTI Crude Oil": 0.01},
        prices=prices,
        active_triggers=triggers,
    )
    ag_prior     = sm._economic_prior("agriculture", "livestock")
    energy_prior = sm._economic_prior("energy", "livestock")
    boosted = round(sm._DEFAULT_CROSS_SECTOR_CORR * 0.01 * (sm.UPSTREAM_DAMPING + sm.TRIGGER_BOOST_COEFF * 0.8) * ag_prior, 6)
    static  = round(sm._DEFAULT_CROSS_SECTOR_CORR * 0.01 * sm.UPSTREAM_DAMPING * energy_prior, 6)
    assert detail["Corn (CBOT)"]          == pytest.approx(boosted, abs=1e-6)
    assert detail["WTI Crude Oil"] == pytest.approx(static,  abs=1e-6)


# ── 4. Cap at 0.50 (max strength = 1.0) ──────────────────────────────────────

def test_damping_capped_at_max_strength(metals_model, prices):
    triggers = [{"family": "opec_action", "strength": 1.0, "trigger_date": "2026-05-15"}]
    _, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=triggers,
    )
    prior = sm._economic_prior("energy", "metals")
    expected = sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * (sm.UPSTREAM_DAMPING + sm.TRIGGER_BOOST_COEFF * 1.0) * prior
    assert detail["WTI Crude Oil"] == pytest.approx(expected, rel=1e-6)
    # Effective damping is exactly 0.50 — the spec's upper bound (prior removed).
    effective_damping = detail["WTI Crude Oil"] / (sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * prior)
    assert effective_damping == pytest.approx(0.50, rel=1e-6)


# ── 5. Feature flag controls _fetch_triggers (caller-supplied path unaffected) ─

def test_fetch_triggers_returns_empty_when_flag_off(metals_model, prices, monkeypatch):
    monkeypatch.setenv("MACRO_TRIGGERS_ENABLED", "false")
    out = metals_model._fetch_triggers(prices)
    assert out == []


def test_fetch_triggers_swallows_db_errors(metals_model, prices, monkeypatch):
    """If DB query raises, _fetch_triggers returns [] rather than propagating."""
    monkeypatch.setenv("MACRO_TRIGGERS_ENABLED", "true")
    import features.macro_features as mf
    def boom(*a, **kw):
        raise RuntimeError("DB unavailable")
    monkeypatch.setattr(mf, "get_active_triggers", boom)
    assert metals_model._fetch_triggers(prices) == []


# ── 6. Multiple triggers in the same family → max strength wins ──────────────

def test_multiple_triggers_use_max_strength(metals_model, prices):
    triggers = [
        {"family": "opec_action",         "strength": 0.5, "trigger_date": "2026-05-13"},
        {"family": "eia_crude_inventory", "strength": 0.9, "trigger_date": "2026-05-14"},
        {"family": "weather_shock",       "strength": 0.6, "trigger_date": "2026-05-15"},
    ]
    _, detail = metals_model._compute_upstream_adjustment(
        upstream_shocks={"WTI Crude Oil": 0.02},
        prices=prices,
        active_triggers=triggers,
    )
    # Both opec_action and eia_crude_inventory map to "energy"; max strength
    # among the two is 0.9. weather_shock maps to "agriculture" — irrelevant
    # for this path.
    prior = sm._economic_prior("energy", "metals")
    expected = round(sm._DEFAULT_CROSS_SECTOR_CORR * 0.02 * (sm.UPSTREAM_DAMPING + sm.TRIGGER_BOOST_COEFF * 0.9) * prior, 6)
    assert detail["WTI Crude Oil"] == pytest.approx(expected, abs=1e-6)
