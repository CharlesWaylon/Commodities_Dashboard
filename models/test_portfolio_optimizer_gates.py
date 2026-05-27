"""
Step-6 tests for models/portfolio_optimizer.apply_trigger_risk_gates().

The gate function is pure — no QAOA needed. Tests exercise each gate rule and
the data-driven config.

Coverage
────────
 1. No triggers → weights pass through unchanged.
 2. weather_shock @ 0.7 caps Agriculture at 1.5 × equal-weight.
 3. opec_action @ 0.7 caps Energy at 1.5 × equal-weight.
 4. fed_tightening @ 0.7 flattens 20% toward equal weight (post-blend variance
    is strictly lower than pre-blend variance).
 5. Any trigger ≥ 0.9 with previous_weights → 30% blend with yesterday.
 6. Sub-threshold trigger → no gate runs.
 7. Output weights always sum to ~1.0.
 8. applied_gates log captures which rules fired.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import portfolio_optimizer as po


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def asset_sectors():
    """Synthetic sector lookup so tests don't depend on the real registry."""
    return {
        "ENERGY_A":      "energy",
        "ENERGY_B":      "energy",
        "AG_A":          "agriculture",
        "AG_B":          "agriculture",
        "METALS_A":      "metals",
    }


@pytest.fixture
def equal_weights():
    return {
        "ENERGY_A": 0.20,
        "ENERGY_B": 0.20,
        "AG_A":     0.20,
        "AG_B":     0.20,
        "METALS_A": 0.20,
    }


@pytest.fixture
def concentrated_weights():
    """Energy and Ag are over-weighted; metals under."""
    return {
        "ENERGY_A": 0.40,
        "ENERGY_B": 0.30,    # energy total = 0.70
        "AG_A":     0.20,
        "AG_B":     0.05,    # ag total = 0.25
        "METALS_A": 0.05,
    }


# ── 1. No triggers → passthrough ──────────────────────────────────────────────

def test_no_triggers_passthrough(equal_weights, asset_sectors):
    out, log = po.apply_trigger_risk_gates(
        equal_weights, active_triggers=[], asset_sectors=asset_sectors,
    )
    assert out == equal_weights
    assert log == []


# ── 2. weather_shock caps agriculture ─────────────────────────────────────────

def test_weather_shock_caps_agriculture(concentrated_weights, asset_sectors):
    # ag total before cap = 0.25
    # k = 5 selected assets, equal weight = 0.20, cap = 1.5 * 0.20 = 0.30
    # 0.25 is already < 0.30 — push ag higher and re-test.
    # Sum = 1.0 with ag heavy.
    weights = {
        "AG_A":     0.30,
        "AG_B":     0.25,    # ag total 0.55
        "ENERGY_A": 0.15,
        "ENERGY_B": 0.20,
        "METALS_A": 0.10,
    }
    triggers = [{"family": "weather_shock", "strength": 0.85, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        weights, active_triggers=triggers, asset_sectors=asset_sectors,
    )
    ag_total = out["AG_A"] + out["AG_B"]
    # The cap is 1.5 / k = 0.30 of the renormalised dict. After renormalisation
    # the sector total will be ≤ 0.30 + tiny float slop.
    assert ag_total <= 0.30 + 1e-9
    assert any("sector_cap[agriculture]" in s for s in log)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


# ── 3. opec_action caps energy ────────────────────────────────────────────────

def test_opec_action_caps_energy(concentrated_weights, asset_sectors):
    triggers = [{"family": "opec_action", "strength": 0.8, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        concentrated_weights, active_triggers=triggers, asset_sectors=asset_sectors,
    )
    energy_total = out["ENERGY_A"] + out["ENERGY_B"]
    # k = 5, equal = 0.20, cap = 0.30. Energy was 0.70 → must be capped.
    assert energy_total <= 0.30 + 1e-9
    assert any("sector_cap[energy]" in s for s in log)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


# ── 4. fed_tightening flattens toward equal weight ────────────────────────────

def test_fed_tightening_flattens(concentrated_weights, asset_sectors):
    triggers = [{"family": "fed_tightening", "strength": 0.75, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        concentrated_weights, active_triggers=triggers, asset_sectors=asset_sectors,
    )
    # Post-blend variance should be strictly lower than pre-blend variance.
    var_before = np.var(list(concentrated_weights.values()))
    var_after  = np.var(list(out.values()))
    assert var_after < var_before
    assert any("flatten_toward_equal" in s for s in log)


# ── 5. Turnover damper with previous weights ──────────────────────────────────

def test_turnover_damper_blends_with_previous(asset_sectors):
    today_w = {
        "ENERGY_A": 0.50, "ENERGY_B": 0.30, "AG_A": 0.10, "AG_B": 0.05, "METALS_A": 0.05,
    }
    yesterday_w = {
        "ENERGY_A": 0.20, "ENERGY_B": 0.20, "AG_A": 0.20, "AG_B": 0.20, "METALS_A": 0.20,
    }
    # Use a non-rate-shock-mapped family so only the __any_strong__ damper fires.
    triggers = [{"family": "geopolitical_shock", "strength": 0.95, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        today_w, active_triggers=triggers,
        previous_weights=yesterday_w, asset_sectors=asset_sectors,
    )
    # 70% today + 30% yesterday for ENERGY_A: 0.70*0.50 + 0.30*0.20 = 0.41
    assert out["ENERGY_A"] == pytest.approx(0.41, abs=1e-6)
    assert any("turnover_damper" in s for s in log)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)


def test_turnover_damper_skipped_without_previous(asset_sectors):
    weights = {
        "ENERGY_A": 0.50, "ENERGY_B": 0.30, "AG_A": 0.10, "AG_B": 0.05, "METALS_A": 0.05,
    }
    triggers = [{"family": "geopolitical_shock", "strength": 0.95, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        weights, active_triggers=triggers, previous_weights=None, asset_sectors=asset_sectors,
    )
    # No previous_weights → damper noop, no gate fires for this family.
    assert out == weights
    assert log == []


# ── 6. Sub-threshold trigger → no gate ────────────────────────────────────────

def test_sub_threshold_no_gate(concentrated_weights, asset_sectors):
    triggers = [{"family": "opec_action", "strength": 0.5, "trigger_date": "2026-05-15"}]
    out, log = po.apply_trigger_risk_gates(
        concentrated_weights, active_triggers=triggers, asset_sectors=asset_sectors,
    )
    assert out == concentrated_weights
    assert log == []


# ── 7. Multiple gates compose; weights sum to ~1.0 ────────────────────────────

def test_multiple_gates_compose(concentrated_weights, asset_sectors):
    yesterday_w = {a: 0.20 for a in concentrated_weights}
    triggers = [
        {"family": "opec_action",        "strength": 0.95, "trigger_date": "2026-05-13"},
        {"family": "weather_shock",      "strength": 0.85, "trigger_date": "2026-05-14"},
        {"family": "fed_tightening",     "strength": 0.95, "trigger_date": "2026-05-15"},
    ]
    out, log = po.apply_trigger_risk_gates(
        concentrated_weights, active_triggers=triggers,
        previous_weights=yesterday_w, asset_sectors=asset_sectors,
    )
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    # 3 per-family gates + 1 turnover damper = up to 4 entries.
    assert len(log) >= 3
    # Energy and ag both capped + flattened: variance should be much lower.
    var_before = np.var(list(concentrated_weights.values()))
    var_after  = np.var(list(out.values()))
    assert var_after < var_before


# ── 8. config-driven: load the real TRIGGER_RISK_GATES at runtime ─────────────

def test_config_gates_load_from_module():
    """Smoke test: TRIGGER_RISK_GATES is the source of truth, not local defaults."""
    from models.config import TRIGGER_RISK_GATES
    assert "fed_tightening" in TRIGGER_RISK_GATES
    assert "weather_shock" in TRIGGER_RISK_GATES
    assert "opec_action" in TRIGGER_RISK_GATES
    assert "__any_strong__" in TRIGGER_RISK_GATES
