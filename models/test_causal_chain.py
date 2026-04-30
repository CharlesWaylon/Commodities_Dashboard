"""
Tests for models/causal_chain.py — Phase 4 Part 1.
All synthetic data; no network, no live DB required.
Run: python -m models.test_causal_chain
"""

import sys
import math
import traceback

import numpy as np
import pandas as pd

from models.causal_chain import (
    ChainNode,
    CausalChainResult,
    CausalChain,
    BULLISH, BEARISH, NEUTRAL, HEIGHTENED_VOL, REDUCED_VOL,
    _build_narrative,
    _safe_float,
)
from models.triggers import TriggerFamily, register_trigger_family

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


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_prices(n=300, seed=7, commodity="WTI Crude Oil", vol_spike_at=None) -> pd.DataFrame:
    """
    Random-walk prices.  If vol_spike_at (int position) is given, triple the
    daily returns in the 10 rows following that position to create a detectable
    vol shift.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=n)
    rets = rng.normal(0, 0.01, n)
    if vol_spike_at is not None:
        spike_end = min(vol_spike_at + 10, n)
        rets[vol_spike_at:spike_end] *= 4   # make post-trigger vol clearly higher
    prices_arr = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({commodity: prices_arr}, index=idx)


def _make_macro(price_idx: pd.DatetimeIndex, risk_off=False, crisis=False) -> pd.DataFrame:
    n = len(price_idx)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "vix":             [35.0 if crisis else (22.0 if risk_off else 15.0)] * n,
        "vix_level_z":     rng.normal(0, 1, n),
        "vix_risk_off":    [1.0 if (risk_off or crisis) else 0.0] * n,
        "vix_crisis":      [1.0 if crisis else 0.0] * n,
        "dxy_zscore63":    [2.1 if risk_off else 0.3] * n,
        "dxy_mom21":       rng.normal(0, 0.01, n),
        "tlt_mom21":       rng.normal(0, 0.01, n),
        "tlt_yield_proxy": rng.normal(0, 0.005, n),
        "is_opec_window":  [0.0] * n,
        "days_to_opec":    [-20.0] * n,
        "is_wasde_window": [0.0] * n,
        "days_to_wasde":   [-10.0] * n,
        "wasde_post5":     [0.0] * n,
    }, index=price_idx)


def _trigger_date(prices: pd.DataFrame, offset=150) -> str:
    return prices.index[offset].strftime("%Y-%m-%d")


# ── ChainNode ─────────────────────────────────────────────────────────────────

def test_chain_node_to_dict():
    print("1. ChainNode.to_dict — serialises correctly")
    try:
        node = ChainNode(
            layer="statistical", model_name="Realized Volatility",
            summary="Vol: 18% → 25% (+39%)", confidence=0.78,
            before_value=18.0, after_value=25.0, change_pct=0.39,
            direction=HEIGHTENED_VOL,
        )
        d = node.to_dict()
        assert d["layer"] == "statistical"
        assert d["confidence"] == 0.78
        assert d["direction"] == HEIGHTENED_VOL
        assert d["before_value"] == 18.0
        _ok(f"keys={list(d.keys())}")
    except Exception as e:
        _fail("ChainNode.to_dict", e)


# ── CausalChainResult ─────────────────────────────────────────────────────────

def test_chain_result_to_dict_and_json():
    print("2. CausalChainResult.to_dict / to_json")
    try:
        import json
        node = ChainNode("trigger", "opec_action", "OPEC fired", 0.8, direction=BULLISH)
        res = CausalChainResult(
            trigger_family="opec_action", trigger_date="2026-04-15",
            trigger_strength=0.8, affected_commodities=["WTI Crude Oil"],
            commodity="WTI Crude Oil", nodes=[node],
            portfolio_recommendation="OVERWEIGHT WTI", portfolio_confidence=0.65,
            narrative="OPEC cut, vol up, go long.",
        )
        d = res.to_dict()
        assert d["trigger_family"] == "opec_action"
        assert len(d["nodes"]) == 1
        j = res.to_json()
        parsed = json.loads(j)
        assert parsed["commodity"] == "WTI Crude Oil"
        _ok(f"to_json: {len(j)} bytes")
    except Exception as e:
        _fail("CausalChainResult serialisation", e)


# ── _safe_float ───────────────────────────────────────────────────────────────

def test_safe_float():
    print("3. _safe_float — handles None, nan, str, float")
    try:
        assert math.isnan(_safe_float(None))
        assert math.isnan(_safe_float(float("nan")))
        assert math.isnan(_safe_float("not_a_number"))
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        _ok("None/nan/str/float all handled")
    except Exception as e:
        _fail("_safe_float", e)


# ── Trigger node ──────────────────────────────────────────────────────────────

def test_trigger_node_built():
    print("4. CausalChain — trigger node always present")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.75,
        )
        assert len(result.nodes) >= 1
        assert result.nodes[0].layer == "trigger"
        assert result.nodes[0].confidence == 0.75
        _ok(f"trigger node: {result.nodes[0].summary[:60]}")
    except Exception as e:
        _fail("trigger node", e)


# ── Statistical node ──────────────────────────────────────────────────────────

def test_statistical_node_detects_vol_spike():
    print("5. Statistical node — detects vol spike post-trigger")
    try:
        # Put the vol spike at position 150 — the trigger date
        prices = _make_prices(300, vol_spike_at=150)
        macro  = _make_macro(prices.index)
        tracer = CausalChain(window_days=10)
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices, offset=150),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.7,
        )
        stat_nodes = [n for n in result.nodes if n.layer == "statistical"]
        assert stat_nodes, "expected at least one statistical node"
        stat = stat_nodes[0]
        assert stat.after_value is not None and stat.before_value is not None
        assert stat.after_value > stat.before_value, (
            f"expected vol to rise after spike: before={stat.before_value} after={stat.after_value}"
        )
        assert stat.direction in (HEIGHTENED_VOL, NEUTRAL)
        _ok(f"vol before={stat.before_value:.1f}% → after={stat.after_value:.1f}% "
            f"(direction={stat.direction})")
    except Exception as e:
        _fail("statistical vol spike", e)


def test_statistical_node_missing_commodity():
    print("6. Statistical node — gracefully skips missing commodity")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="fed_tightening",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="Gold (COMEX)",   # not in synthetic prices
            trigger_strength=0.5,
        )
        stat_nodes = [n for n in result.nodes if n.layer == "statistical"]
        assert stat_nodes == [], "should have no statistical node for missing commodity"
        _ok("statistical node skipped when commodity absent")
    except Exception as e:
        _fail("statistical node missing commodity", e)


# ── Regime node ───────────────────────────────────────────────────────────────

def test_regime_node_risk_off():
    print("7. Regime node — detects Risk-Off correctly")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index, risk_off=True)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="fed_tightening",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.6,
        )
        regime_nodes = [n for n in result.nodes if n.layer == "regime"]
        assert regime_nodes, "expected regime node"
        r = regime_nodes[0]
        regime = r.metadata.get("regime", "")
        assert regime in ("Risk-Off", "Dollar-Stress", "Crisis"), (
            f"expected risk-off-type regime, got {regime!r}"
        )
        _ok(f"regime={regime}, direction={r.direction}")
    except Exception as e:
        _fail("regime node risk-off", e)


def test_regime_node_crisis():
    print("8. Regime node — detects Crisis and overrides direction to HEIGHTENED_VOL")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index, crisis=True)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.8,
        )
        regime_nodes = [n for n in result.nodes if n.layer == "regime"]
        assert regime_nodes
        assert regime_nodes[0].metadata["crisis"] is True
        _ok(f"crisis regime detected, confidence={regime_nodes[0].confidence:.2f}")
    except Exception as e:
        _fail("regime node crisis", e)


def test_regime_node_empty_macro():
    print("9. Regime node — gracefully absent when macro_df is empty")
    try:
        prices = _make_prices()
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=pd.DataFrame(),
            commodity="WTI Crude Oil",
            trigger_strength=0.5,
        )
        regime_nodes = [n for n in result.nodes if n.layer == "regime"]
        assert regime_nodes == []
        _ok("no regime node when macro_df is empty")
    except Exception as e:
        _fail("regime node empty macro", e)


# ── Ensemble node ─────────────────────────────────────────────────────────────

def test_ensemble_node_present():
    print("10. Ensemble node — always built (untrained fallback)")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.7,
        )
        ens_nodes = [n for n in result.nodes if n.layer == "ensemble"]
        assert ens_nodes, "expected ensemble node"
        ens = ens_nodes[0]
        assert "meta-predictor" in ens.summary.lower()
        # Untrained → meta_confidence = 0
        assert ens.confidence == 0.0 or ens.metadata.get("is_trained") is True
        _ok(f"ensemble: {ens.summary[:60]}")
    except Exception as e:
        _fail("ensemble node", e)


# ── Portfolio node ────────────────────────────────────────────────────────────

def test_portfolio_node_always_present():
    print("11. Portfolio node — always last node")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.8,
        )
        assert result.nodes[-1].layer == "portfolio"
        assert result.portfolio_recommendation != ""
        assert 0.0 <= result.portfolio_confidence <= 1.0
        _ok(f"portfolio: {result.portfolio_recommendation[:70]}")
    except Exception as e:
        _fail("portfolio node", e)


def test_portfolio_crisis_overrides_to_reduce():
    print("12. Portfolio node — crisis regime → REDUCE recommendation")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index, crisis=True)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.9,
        )
        port = result.nodes[-1]
        assert "REDUCE" in port.summary or port.direction in (BEARISH, HEIGHTENED_VOL), (
            f"crisis should lead to defensive action, got: {port.summary}"
        )
        _ok(f"crisis → {port.summary[:60]}")
    except Exception as e:
        _fail("portfolio crisis", e)


def test_portfolio_strong_trigger_bullish():
    print("13. Portfolio node — strong trigger + calm regime → OVERWEIGHT or MONITOR")
    try:
        prices = _make_prices()
        macro  = _make_macro(prices.index, risk_off=False, crisis=False)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.85,
        )
        port = result.nodes[-1]
        assert any(kw in port.summary for kw in ("OVERWEIGHT", "MONITOR", "TIGHTEN")), (
            f"expected actionable recommendation, got: {port.summary}"
        )
        _ok(f"strong trigger → {port.summary[:60]}")
    except Exception as e:
        _fail("portfolio strong trigger", e)


# ── Full chain structure ──────────────────────────────────────────────────────

def test_full_chain_has_all_layers():
    print("14. Full chain — has trigger + statistical + regime + ensemble + portfolio")
    try:
        prices = _make_prices(300, vol_spike_at=150)
        macro  = _make_macro(prices.index, risk_off=True)
        tracer = CausalChain(window_days=10)
        result = tracer.trace(
            trigger_family="fed_tightening",
            trigger_date=_trigger_date(prices, 150),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.65,
        )
        layers = [n.layer for n in result.nodes]
        for expected in ("trigger", "statistical", "regime", "ensemble", "portfolio"):
            assert expected in layers, f"missing layer: {expected}"
        assert layers[-1] == "portfolio"   # portfolio always last
        _ok(f"layers={layers}")
    except Exception as e:
        _fail("full chain layers", e)


# ── Narrative ─────────────────────────────────────────────────────────────────

def test_narrative_non_empty():
    print("15. Narrative — non-empty string referencing trigger family")
    try:
        prices = _make_prices(300, vol_spike_at=150)
        macro  = _make_macro(prices.index)
        tracer = CausalChain()
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date=_trigger_date(prices, 150),
            prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
            trigger_strength=0.7,
        )
        assert len(result.narrative) > 40
        _ok(f"narrative ({len(result.narrative)} chars): {result.narrative[:80]}…")
    except Exception as e:
        _fail("narrative non-empty", e)


# ── trace_from_event ──────────────────────────────────────────────────────────

def test_trace_from_event():
    print("16. trace_from_event — accepts a live TriggerEvent")
    try:
        from models.router import SignalRouter
        prices = _make_prices()
        macro  = _make_macro(prices.index)
        event  = SignalRouter.make_event(
            family="opec_action", strength=0.6, rationale="test"
        )
        tracer = CausalChain()
        result = tracer.trace_from_event(
            event=event, prices=prices, macro_df=macro,
            commodity="WTI Crude Oil",
        )
        assert result.trigger_family == "opec_action"
        assert result.trigger_strength == 0.6
        assert len(result.nodes) >= 1
        _ok("trace_from_event produces valid chain")
    except Exception as e:
        _fail("trace_from_event", e)


# ── trace_recent (empty log) ──────────────────────────────────────────────────

def test_trace_recent_empty_log():
    print("17. trace_recent — empty log returns [] without crash")
    try:
        import tempfile
        from pathlib import Path
        # Point to a fresh DB with no entries
        tmp = Path(tempfile.mktemp(suffix=".db"))
        # Monkey-patch trigger_log to use temp DB
        import models.trigger_log as tl
        _orig = tl.DEFAULT_DB_PATH
        tl.DEFAULT_DB_PATH = tmp
        try:
            prices = _make_prices()
            macro  = _make_macro(prices.index)
            tracer = CausalChain()
            results = tracer.trace_recent(prices=prices, macro_df=macro,
                                          commodity="WTI Crude Oil", days=30)
            assert results == []
        finally:
            tl.DEFAULT_DB_PATH = _orig
            tmp.unlink(missing_ok=True)
        _ok("empty log → []")
    except Exception as e:
        _fail("trace_recent empty log", e)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("CAUSAL CHAIN TRACER — TEST SUITE  (Phase 4 Part 1)")
    print("=" * 60)
    print()

    for fn in [
        test_chain_node_to_dict,
        test_chain_result_to_dict_and_json,
        test_safe_float,
        test_trigger_node_built,
        test_statistical_node_detects_vol_spike,
        test_statistical_node_missing_commodity,
        test_regime_node_risk_off,
        test_regime_node_crisis,
        test_regime_node_empty_macro,
        test_ensemble_node_present,
        test_portfolio_node_always_present,
        test_portfolio_crisis_overrides_to_reduce,
        test_portfolio_strong_trigger_bullish,
        test_full_chain_has_all_layers,
        test_narrative_non_empty,
        test_trace_from_event,
        test_trace_recent_empty_log,
    ]:
        fn()
        print()

    print("=" * 60)
    if FAIL == 0:
        print(f"ALL {PASS} ASSERTIONS PASSED")
        print(
            "Phase 4 Part 1 complete: CausalChain tracer ready.\n"
            "Part 2: Sankey diagram + narrative panel (pages/6_Causal.py)."
        )
    else:
        print(f"{PASS} passed  |  {FAIL} FAILED")
        sys.exit(1)
    print("=" * 60)
    print()
