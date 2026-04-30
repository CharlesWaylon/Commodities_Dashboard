"""
Causal Chain Tracer — Phase 4 of the Model Integration Roadmap.

Purpose
───────
When a macro trigger fires (e.g. OPEC+ meeting window, DXY shock), this module
traces how that event rippled through the model ecosystem:

    Trigger event
        │
        ▼  (how did volatility shift around that date?)
    Statistical layer  — realized vol before vs. after
        │
        ▼  (did the macro regime change?)
    Macro regime layer — VIX / DXY state on trigger date
        │
        ▼  (what did the ensemble arbitrator decide?)
    Ensemble layer     — MetaPredictor decision on that date
        │
        ▼  (what should a portfolio do?)
    Portfolio layer    — position-sizing recommendation

Each step is a ChainNode.  The full trace is a CausalChainResult.
Results serialize to dicts/JSON for the dashboard Sankey and narrative panel.

Design choices
──────────────
• Realized vol (rolling std) instead of GARCH — instant, no re-fitting.
• Macro-overlay columns (vix_risk_off, dxy_zscore63) instead of live HMM fit —
  the HMM is expensive; the macro flags are already computed and cached.
• MetaPredictor loaded from disk (data/meta_predictor.pkl) if available;
  equal-weight fallback if not yet trained.
• The portfolio layer is rule-based, not ML — transparent and fast.
• Every layer is wrapped in try/except; a broken layer skips its node and the
  chain still completes. Resilience > completeness.

Extension points
────────────────
• Add a "Deep" node when ProphetDecomposer exposes a trend-slope API.
• Add a "Quantum" node when the kernel SVM is fast enough for live calls.
• Swap the VIX regime proxy for a live HMM call once HMM results are cached
  in the macro DataFrame by features/hmm_regime.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.triggers import TriggerFamily, TriggerEvent, get_trigger_family


# ── Direction enum-like constants ─────────────────────────────────────────────

BULLISH         = "bullish"
BEARISH         = "bearish"
NEUTRAL         = "neutral"
HEIGHTENED_VOL  = "heightened_vol"
REDUCED_VOL     = "reduced_vol"

# How many trading days before / after the trigger to compare
DEFAULT_WINDOW = 10   # 2 calendar weeks


# ── ChainNode ──────────────────────────────────────────────────────────────────

@dataclass
class ChainNode:
    """
    One step in the causal chain.

    Attributes
    ──────────
    layer       : which model tier / system layer produced this node
    model_name  : specific model or data source
    summary     : one-line human-readable description of what changed
    confidence  : [0, 1] — how strong / reliable this node's signal is
    before_value: numeric state before the trigger (e.g. realized vol 18%)
    after_value : numeric state after the trigger (e.g. realized vol 25%)
    change_pct  : (after - before) / |before| — signed percentage change
    direction   : BULLISH | BEARISH | NEUTRAL | HEIGHTENED_VOL | REDUCED_VOL
    metadata    : free-form dict for Sankey value, tooltips, etc.
    """
    layer: str                          # "trigger" | "statistical" | "regime"
                                        # | "ensemble" | "portfolio"
    model_name: str
    summary: str
    confidence: float
    before_value: Optional[float] = None
    after_value: Optional[float] = None
    change_pct: Optional[float] = None
    direction: str = NEUTRAL
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer":        self.layer,
            "model_name":   self.model_name,
            "summary":      self.summary,
            "confidence":   round(self.confidence, 4),
            "before_value": self.before_value,
            "after_value":  self.after_value,
            "change_pct":   self.change_pct,
            "direction":    self.direction,
            "metadata":     self.metadata,
        }


# ── CausalChainResult ─────────────────────────────────────────────────────────

@dataclass
class CausalChainResult:
    """
    Full causal trace for a single (trigger_family, trigger_date, commodity).

    The nodes list is ordered:
        [0] trigger
        [1] statistical   (if computed)
        [2] regime        (if computed)
        [3] ensemble      (if computed)
        [4] portfolio     (always present — fallback rule-based)
    """
    trigger_family: str
    trigger_date: str          # YYYY-MM-DD
    trigger_strength: float    # [0, 1]
    affected_commodities: List[str]
    commodity: str             # which commodity was traced
    nodes: List[ChainNode] = field(default_factory=list)
    portfolio_recommendation: str = ""
    portfolio_confidence: float = 0.0
    narrative: str = ""
    traced_at: str = ""

    def __post_init__(self):
        if not self.traced_at:
            self.traced_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "trigger_family":           self.trigger_family,
            "trigger_date":             self.trigger_date,
            "trigger_strength":         round(self.trigger_strength, 4),
            "affected_commodities":     self.affected_commodities,
            "commodity":                self.commodity,
            "nodes":                    [n.to_dict() for n in self.nodes],
            "portfolio_recommendation": self.portfolio_recommendation,
            "portfolio_confidence":     round(self.portfolio_confidence, 4),
            "narrative":                self.narrative,
            "traced_at":                self.traced_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── CausalChain tracer ────────────────────────────────────────────────────────

class CausalChain:
    """
    Traces how macro trigger events flow through the model ecosystem.

    Parameters
    ----------
    window_days : int
        Number of trading days before and after a trigger to compare for
        vol-shift / regime-shift detection.
    meta_model_path : Path, optional
        Path to the persisted MetaPredictor pkl.  Falls back to the default
        data/meta_predictor.pkl location.

    Usage
    ─────
        from models.causal_chain import CausalChain
        tracer = CausalChain()

        # Trace one specific trigger event
        result = tracer.trace(
            trigger_family="opec_action",
            trigger_date="2026-04-15",
            prices=prices_df,
            macro_df=macro_df,
            commodity="WTI Crude Oil",
        )
        print(result.narrative)

        # Trace all recent trigger events from the SQLite log
        results = tracer.trace_recent(
            days=60, prices=prices_df, macro_df=macro_df,
            commodity="WTI Crude Oil",
        )
    """

    def __init__(
        self,
        window_days: int = DEFAULT_WINDOW,
        meta_model_path: Optional[Path] = None,
    ):
        self.window_days = window_days
        self._meta_model_path = meta_model_path

    # ── Public API ─────────────────────────────────────────────────────────────

    def trace(
        self,
        trigger_family: str,
        trigger_date: str,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity: str,
        trigger_strength: Optional[float] = None,
        trigger_rationale: str = "",
    ) -> CausalChainResult:
        """
        Build a full causal chain for one trigger event.

        Parameters
        ----------
        trigger_family  : TriggerFamily name (e.g. "opec_action")
        trigger_date    : YYYY-MM-DD string
        prices          : full price matrix (n_days × n_commodities)
        macro_df        : macro overlay DataFrame from load_macro()
        commodity       : which commodity to trace (must be in prices.columns)
        trigger_strength: override strength [0,1]; looked up from TriggerFamily
                          if not provided
        trigger_rationale: optional human text appended to the trigger node

        Returns
        -------
        CausalChainResult
        """
        # Resolve trigger date and strength
        try:
            t_date = pd.Timestamp(trigger_date)
        except Exception:
            t_date = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

        fam = get_trigger_family(trigger_family)
        affected = list(fam.affected_commodities) if fam else []
        if trigger_strength is None:
            trigger_strength = 0.5   # neutral default if not supplied

        result = CausalChainResult(
            trigger_family=trigger_family,
            trigger_date=t_date.strftime("%Y-%m-%d"),
            trigger_strength=trigger_strength,
            affected_commodities=affected,
            commodity=commodity,
        )

        # ── Layer 1: Trigger node ─────────────────────────────────────────────
        result.nodes.append(
            self._build_trigger_node(trigger_family, t_date, trigger_strength,
                                     affected, trigger_rationale)
        )

        # ── Layer 2: Statistical — realized vol shift ─────────────────────────
        try:
            stat_node = self._build_statistical_node(prices, commodity, t_date)
            if stat_node:
                result.nodes.append(stat_node)
        except Exception:
            pass

        # ── Layer 3: Macro regime — VIX / DXY state ───────────────────────────
        try:
            regime_node = self._build_regime_node(macro_df, t_date)
            if regime_node:
                result.nodes.append(regime_node)
        except Exception:
            pass

        # ── Layer 4: Ensemble — MetaPredictor decision ────────────────────────
        try:
            ensemble_node = self._build_ensemble_node(macro_df, t_date)
            if ensemble_node:
                result.nodes.append(ensemble_node)
        except Exception:
            pass

        # ── Layer 5: Portfolio recommendation ─────────────────────────────────
        portfolio_node, recommendation, port_confidence = self._build_portfolio_node(
            result.nodes, trigger_family, trigger_strength, affected, commodity
        )
        result.nodes.append(portfolio_node)
        result.portfolio_recommendation = recommendation
        result.portfolio_confidence = port_confidence

        # ── Narrative ─────────────────────────────────────────────────────────
        result.narrative = _build_narrative(result)

        return result

    def trace_recent(
        self,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity: str,
        days: int = 60,
        family_filter: Optional[str] = None,
    ) -> List[CausalChainResult]:
        """
        Read recent trigger events from the SQLite log and trace each one.

        Parameters
        ----------
        days          : how far back in the log to look
        family_filter : restrict to a single trigger family

        Returns
        -------
        list of CausalChainResult, most recent first.
        Empty list if log is empty or inaccessible.
        """
        try:
            from models.trigger_log import recent_trigger_events
            log_df = recent_trigger_events(days=days, family=family_filter)
        except Exception:
            return []

        if log_df.empty:
            return []

        results: List[CausalChainResult] = []
        for _, row in log_df.iterrows():
            try:
                res = self.trace(
                    trigger_family=row["family"],
                    trigger_date=row["trigger_date"],
                    prices=prices,
                    macro_df=macro_df,
                    commodity=commodity,
                    trigger_strength=float(row["strength"]),
                    trigger_rationale=str(row.get("rationale", "")),
                )
                results.append(res)
            except Exception:
                continue

        return results

    def trace_from_event(
        self,
        event: TriggerEvent,
        prices: pd.DataFrame,
        macro_df: pd.DataFrame,
        commodity: str,
    ) -> CausalChainResult:
        """
        Convenience wrapper: trace directly from a live TriggerEvent object.
        Useful for surfacing a chain immediately after detect_all() fires.
        """
        return self.trace(
            trigger_family=event.family,
            trigger_date=event.detected_at[:10],
            prices=prices,
            macro_df=macro_df,
            commodity=commodity,
            trigger_strength=event.strength,
            trigger_rationale=event.rationale,
        )

    # ── Layer builders ─────────────────────────────────────────────────────────

    def _build_trigger_node(
        self,
        family: str,
        t_date: pd.Timestamp,
        strength: float,
        affected: List[str],
        rationale: str,
    ) -> ChainNode:
        fam = get_trigger_family(family)
        desc = fam.description if fam else family.replace("_", " ").title()
        tier = "Strong" if strength >= 0.7 else ("Moderate" if strength >= 0.3 else "Weak")
        short_affected = [c.split(" (")[0] for c in affected[:3]]
        summary = (
            f"{desc} — {tier} signal ({strength:.0%}). "
            f"Affects: {', '.join(short_affected)}"
        )
        if rationale:
            summary += f". {rationale}"
        return ChainNode(
            layer="trigger",
            model_name=family,
            summary=summary,
            confidence=strength,
            before_value=None,
            after_value=strength,
            direction=BULLISH if strength >= 0.5 else NEUTRAL,
            metadata={"family": family, "trigger_date": t_date.strftime("%Y-%m-%d"),
                      "affected_count": len(affected)},
        )

    def _build_statistical_node(
        self,
        prices: pd.DataFrame,
        commodity: str,
        t_date: pd.Timestamp,
    ) -> Optional[ChainNode]:
        """Compute realized vol (21-day rolling std of log-returns) before vs after."""
        if commodity not in prices.columns:
            return None

        log_ret = np.log(prices[commodity] / prices[commodity].shift(1)).dropna()

        # Find the closest available date in the index
        idx = prices.index
        date_positions = idx.searchsorted(t_date)
        if date_positions >= len(idx):
            date_positions = len(idx) - 1
        t_pos = date_positions

        # Before window: `window_days` trading days ending at trigger date
        pre_start  = max(0, t_pos - self.window_days)
        pre_slice  = log_ret.iloc[pre_start:t_pos]

        # After window: `window_days` trading days starting after trigger date
        post_end   = min(len(log_ret), t_pos + self.window_days)
        post_slice = log_ret.iloc[t_pos:post_end]

        if len(pre_slice) < 3 or len(post_slice) < 3:
            return None

        vol_before = float(pre_slice.std() * np.sqrt(252) * 100)   # annualised %
        vol_after  = float(post_slice.std() * np.sqrt(252) * 100)

        if vol_before < 1e-9:
            return None

        change = (vol_after - vol_before) / vol_before
        direction = HEIGHTENED_VOL if change > 0.05 else (REDUCED_VOL if change < -0.05 else NEUTRAL)

        summary = (
            f"Realized vol ({commodity.split('(')[0].strip()}): "
            f"{vol_before:.1f}% → {vol_after:.1f}% "
            f"({'↑' if change > 0 else '↓'}{abs(change):.0%})"
        )

        # Confidence scales with how decisive the shift was
        confidence = min(1.0, abs(change) * 2)

        return ChainNode(
            layer="statistical",
            model_name="Realized Volatility",
            summary=summary,
            confidence=confidence,
            before_value=round(vol_before, 2),
            after_value=round(vol_after, 2),
            change_pct=round(change, 4),
            direction=direction,
            metadata={"window_days": self.window_days, "commodity": commodity,
                      "annualised": True, "unit": "%"},
        )

    def _build_regime_node(
        self,
        macro_df: pd.DataFrame,
        t_date: pd.Timestamp,
    ) -> Optional[ChainNode]:
        """
        Derive the macro regime state from VIX / DXY overlay columns.
        Uses the macro_df flags that are already computed — no HMM re-fit.
        """
        if macro_df is None or macro_df.empty:
            return None

        required = {"vix", "dxy_zscore63"}
        if not required.issubset(macro_df.columns):
            return None

        # Closest row at/before trigger date
        aligned = macro_df.reindex(macro_df.index.union([t_date])).ffill()
        if t_date not in aligned.index:
            return None
        row = aligned.loc[t_date]

        vix      = _safe_float(row.get("vix"))
        dxy_z    = _safe_float(row.get("dxy_zscore63"))
        risk_off = bool(_safe_float(row.get("vix_risk_off", 0.0)) >= 0.5)
        crisis   = bool(_safe_float(row.get("vix_crisis",   0.0)) >= 0.5)

        if math.isnan(vix) and math.isnan(dxy_z):
            return None

        # Determine regime label
        if crisis:
            regime = "Crisis"
            direction = HEIGHTENED_VOL
            confidence = 0.9
        elif risk_off:
            regime = "Risk-Off"
            direction = BEARISH
            confidence = 0.7
        elif not math.isnan(dxy_z) and dxy_z > 1.5:
            regime = "Dollar-Stress"
            direction = BEARISH
            confidence = min(1.0, abs(dxy_z) / 3.0)
        elif not math.isnan(dxy_z) and dxy_z < -1.0:
            regime = "Risk-On / Weak Dollar"
            direction = BULLISH
            confidence = min(1.0, abs(dxy_z) / 3.0)
        else:
            regime = "Neutral"
            direction = NEUTRAL
            confidence = 0.4

        vix_str = f"VIX {vix:.1f}" if not math.isnan(vix) else ""
        dxy_str = f"DXY z={dxy_z:+.2f}" if not math.isnan(dxy_z) else ""
        parts = [p for p in [vix_str, dxy_str] if p]
        summary = f"Macro regime on trigger date: **{regime}** ({'; '.join(parts)})"

        return ChainNode(
            layer="regime",
            model_name="Macro Overlay (VIX/DXY)",
            summary=summary,
            confidence=confidence,
            before_value=None,
            after_value=vix if not math.isnan(vix) else None,
            direction=direction,
            metadata={"regime": regime, "vix": vix, "dxy_zscore63": dxy_z,
                      "risk_off": risk_off, "crisis": crisis},
        )

    def _build_ensemble_node(
        self,
        macro_df: pd.DataFrame,
        t_date: pd.Timestamp,
    ) -> Optional[ChainNode]:
        """
        Load the MetaPredictor from disk (if trained) and produce a MetaDecision
        snapshot for the trigger date's market state.
        Falls back gracefully to equal-weight summary if untrained.
        """
        try:
            from models.meta_predictor import MetaPredictor, collect_meta_features
        except ImportError:
            return None

        # Slice macro_df to the trigger date for feature extraction
        if macro_df is None or macro_df.empty:
            slice_df = pd.DataFrame()
        else:
            aligned = macro_df.reindex(macro_df.index.union([t_date])).ffill()
            slice_df = aligned.loc[:t_date].tail(1) if t_date in aligned.index else pd.DataFrame()

        meta_features = collect_meta_features(slice_df) if not slice_df.empty else None
        if meta_features is None:
            from models.meta_predictor import MetaFeatures
            meta_features = MetaFeatures()

        # Load trained predictor (silent fallback to untrained)
        model_path = self._meta_model_path or (
            Path(__file__).resolve().parent.parent / "data" / "meta_predictor.pkl"
        )
        mp = MetaPredictor()
        mp.load(model_path)

        # Predict with empty votes (weights-only view; real votes need live model fits)
        decision = mp.predict(meta_features, votes=[])

        trusted = decision.trusted_tier
        top_w   = decision.weights.get(trusted, 0.0)
        trained_tag = "trained" if mp.is_trained else "untrained — equal weights"

        summary = (
            f"Meta-predictor ({trained_tag}): "
            f"trusting **{trusted}** tier ({top_w:.0%} weight)"
        )

        return ChainNode(
            layer="ensemble",
            model_name="MetaPredictor",
            summary=summary,
            confidence=decision.meta_confidence if mp.is_trained else 0.0,
            direction=BULLISH if decision.ensemble_forecast >= 0 else BEARISH,
            metadata={
                "trusted_tier":      trusted,
                "weights":           decision.weights,
                "meta_confidence":   decision.meta_confidence,
                "is_trained":        mp.is_trained,
                "reasoning":         decision.reasoning,
            },
        )

    def _build_portfolio_node(
        self,
        nodes: List[ChainNode],
        trigger_family: str,
        trigger_strength: float,
        affected: List[str],
        commodity: str,
    ) -> Tuple[ChainNode, str, float]:
        """
        Rule-based portfolio recommendation that synthesizes upstream nodes.

        Rules (applied in priority order):
        1. If crisis regime + high trigger strength → reduce position sizing
        2. If vol spike > 40% post-trigger → widen stops, reduce leverage
        3. If regime is Risk-Off → underweight trigger-affected commodities
        4. If trigger is strong (≥0.7) + bullish ensemble → +5% tilt
        5. If trigger is moderate (0.3–0.7) + neutral → hold / monitor
        6. Default → no change recommended
        """
        # Extract signals from upstream nodes
        regime_node   = next((n for n in nodes if n.layer == "regime"), None)
        stat_node     = next((n for n in nodes if n.layer == "statistical"), None)
        ensemble_node = next((n for n in nodes if n.layer == "ensemble"), None)

        regime     = regime_node.metadata.get("regime", "Neutral") if regime_node else "Neutral"
        crisis     = regime_node.metadata.get("crisis", False)      if regime_node else False
        risk_off   = regime_node.metadata.get("risk_off", False)    if regime_node else False
        vol_change = stat_node.change_pct                           if stat_node else 0.0
        ens_dir    = ensemble_node.direction                        if ensemble_node else NEUTRAL

        short = [c.split(" (")[0] for c in affected[:2]]
        asset_str = " / ".join(short) if short else commodity.split(" (")[0]

        # Decision tree
        if crisis:
            action = (
                f"REDUCE {asset_str} exposure — Crisis regime (VIX crisis) "
                f"warrants defensive positioning."
            )
            confidence = 0.80
            direction  = BEARISH

        elif vol_change is not None and vol_change > 0.40:
            action = (
                f"TIGHTEN stops on {asset_str} — realized vol surged "
                f"{vol_change:.0%} post-trigger. Consider reducing leverage."
            )
            confidence = 0.70
            direction  = HEIGHTENED_VOL

        elif risk_off and trigger_strength >= 0.5:
            action = (
                f"UNDERWEIGHT {asset_str} — Risk-Off regime + active "
                f"{trigger_family.replace('_', ' ').title()} trigger "
                f"suggests caution."
            )
            confidence = 0.60
            direction  = BEARISH

        elif trigger_strength >= 0.7 and ens_dir == BULLISH:
            action = (
                f"OVERWEIGHT {asset_str} — Strong trigger ({trigger_strength:.0%}) "
                f"+ bullish ensemble consensus. Consider +5% tilt in "
                f"{trigger_family.replace('_', ' ').title()} sector."
            )
            confidence = 0.65
            direction  = BULLISH

        elif trigger_strength >= 0.3:
            action = (
                f"MONITOR {asset_str} — Moderate "
                f"{trigger_family.replace('_', ' ').title()} signal. "
                f"Hold current positioning; watch for regime escalation."
            )
            confidence = 0.45
            direction  = NEUTRAL

        else:
            action = (
                f"NO ACTION — {trigger_family.replace('_', ' ').title()} signal "
                f"is weak ({trigger_strength:.0%}). No portfolio adjustment warranted."
            )
            confidence = 0.30
            direction  = NEUTRAL

        node = ChainNode(
            layer="portfolio",
            model_name="Rule-Based Allocator",
            summary=action,
            confidence=confidence,
            direction=direction,
            metadata={
                "regime": regime, "crisis": crisis, "risk_off": risk_off,
                "vol_change": vol_change, "ensemble_direction": ens_dir,
                "trigger_strength": trigger_strength,
            },
        )
        return node, action, confidence


# ── Narrative builder ─────────────────────────────────────────────────────────

def _build_narrative(result: CausalChainResult) -> str:
    """
    Compose a 2-3 sentence explanation that walks through the chain in plain
    English.  Designed to be readable by a non-technical audience.
    """
    parts: List[str] = []

    # Sentence 1: trigger
    fam = get_trigger_family(result.trigger_family)
    fam_name = fam.description if fam else result.trigger_family.replace("_", " ").title()
    tier = "strong" if result.trigger_strength >= 0.7 else (
           "moderate" if result.trigger_strength >= 0.3 else "weak")
    parts.append(
        f"On {result.trigger_date}, a {tier} **{fam_name}** signal fired "
        f"(strength {result.trigger_strength:.0%})."
    )

    # Sentence 2: statistical + regime combined
    stat   = next((n for n in result.nodes if n.layer == "statistical"), None)
    regime = next((n for n in result.nodes if n.layer == "regime"), None)
    mid: List[str] = []

    if stat and stat.before_value and stat.after_value:
        chg = stat.change_pct or 0.0
        mid.append(
            f"Realized volatility in {result.commodity.split('(')[0].strip()} "
            f"{'rose' if chg > 0 else 'fell'} "
            f"from {stat.before_value:.1f}% to {stat.after_value:.1f}% "
            f"({'+' if chg >= 0 else ''}{chg:.0%})"
        )

    if regime:
        r_label = regime.metadata.get("regime", "Neutral")
        mid.append(f"the macro regime was **{r_label}**")

    if mid:
        parts.append(". ".join(mid).capitalize() + ".")

    # Sentence 3: ensemble + portfolio
    ens   = next((n for n in result.nodes if n.layer == "ensemble"), None)
    port  = next((n for n in result.nodes if n.layer == "portfolio"), None)

    ens_str  = ""
    if ens:
        trusted = ens.metadata.get("trusted_tier", "")
        if ens.metadata.get("is_trained"):
            ens_str = f"The meta-predictor trusted the **{trusted}** tier. "
        else:
            ens_str = "The meta-predictor has not yet been trained (equal weights). "

    port_str = f"Portfolio recommendation: {result.portfolio_recommendation}" if port else ""
    if ens_str or port_str:
        parts.append((ens_str + port_str).strip())

    return " ".join(parts)


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe_float(val, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return default
