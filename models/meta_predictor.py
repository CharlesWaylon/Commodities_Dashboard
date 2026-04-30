"""
Meta-Predictor — Phase 3 of the Model Integration Roadmap.

Purpose
───────
Each model tier (statistical, ml, deep, quantum) forecasts the same commodity
independently.  When they agree, confidence is high.  When they disagree, the
naive response is "average them" — but that destroys information.

The meta-predictor learns *which tier to trust under which market conditions*.
It is a thin, interpretable layer that sits above the individual model signals
and outputs a set of per-tier weights for the ensemble plus a human-readable
explanation of why those weights were chosen today.

Architecture
────────────

    macro_df (latest row)
          │
          ▼
    collect_meta_features()  →  MetaFeatures (market-state snapshot)
          │
          ▼
    MetaPredictor.predict(meta_features, votes)
          │
    ┌─────┴──────────────────────────────────┐
    │  If trained: DecisionTreeClassifier    │
    │  (max_depth=5, Gini, interpretable)    │
    │  Predicts P(tier) per market state     │
    │  If not trained: equal weights         │
    └─────┬──────────────────────────────────┘
          │
          ▼
    MetaDecision (weights, trusted_tier, reasoning, consensus_strength)

Training data (Phase 3 Part 2)
──────────────────────────────
For every historical date the back-tester will produce rows of
(MetaFeatures, winning_tier) — i.e., which tier had the lowest forecast
error on that date for each commodity.  MetaPredictor.fit() consumes that.
Until training data exists, predict() falls back to equal weights so the
dashboard remains live.

Extension points
────────────────
• HMM regime — when `features/hmm_regime.py` lands, add `hmm_regime` and
  `days_since_regime_flip` to MetaFeatures and `FEATURE_COLUMNS`.  The
  predictor re-trains automatically on the next fit() call.

• New tiers — add the tier name to `KNOWN_TIERS` and ensure back-tested
  ModelVotes carry that tier label.

• ENSO / climate features — add them to MetaFeatures and FEATURE_COLUMNS;
  relevant for agriculture-focused meta-models.

Usage (live dashboard)
──────────────────────
    from features.macro_overlays import load_macro
    from models.meta_predictor import collect_meta_features, MetaPredictor
    from models.model_signal import ModelSignal

    macro = load_macro(period="2y")
    mf = collect_meta_features(macro)

    # votes come from whichever model tiers ran this refresh
    votes = [
        ModelVote(tier="statistical", model_name="ARIMA", commodity="WTI Crude Oil",
                  forecast_return=0.004, confidence=0.06),
        ModelVote(tier="ml",          model_name="XGBoost", commodity="WTI Crude Oil",
                  forecast_return=0.009, confidence=0.09),
    ]

    meta = MetaPredictor()          # load from disk or use default equal weights
    decision = meta.predict(mf, votes)

    print(decision.trusted_tier)           # "ml"
    print(decision.reasoning)             # "DXY z-score elevated (2.4); …"
    print(decision.weights)               # {"statistical": 0.35, "ml": 0.50, …}
    print(decision.ensemble_forecast)     # weighted blend of vote forecasts
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_TIERS: Tuple[str, ...] = ("statistical", "ml", "deep", "quantum")

# Ordered list of feature columns fed to the DecisionTree.
# Matches MetaFeatures field names exactly — order matters for sklearn.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "vix",
    "vix_level_z",
    "vix_risk_off",
    "vix_crisis",
    "dxy_zscore63",
    "dxy_mom21",
    "tlt_mom21",
    "tlt_yield_proxy",
    "is_opec_window",
    "days_to_opec",
    "is_wasde_window",
    "days_to_wasde",
    "wasde_post5",
    # ── Extension slots ───────────────────────────────────────────────────────
    # Uncomment as features become available:
    # "hmm_regime_bull",         # 1.0 if HMM state == "bull", else 0.0
    # "hmm_regime_bear",         # 1.0 if HMM state == "bear", else 0.0
    # "days_since_regime_flip",  # integer; cap at 30 for tree stability
    # "enso_phase",              # −1 = La Niña, 0 = neutral, +1 = El Niño
)

# Default equal weights when the meta-predictor hasn't been trained yet.
_EQUAL_WEIGHTS: Dict[str, float] = {t: 1.0 / len(KNOWN_TIERS) for t in KNOWN_TIERS}

# Path to the persisted meta-predictor model inside the project's data dir.
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "meta_predictor.pkl"
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MetaFeatures:
    """
    A snapshot of the market state used as input to the meta-predictor.
    All fields match macro_overlays column names so `collect_meta_features()`
    can fill them with a single `.iloc[-1]` lookup.

    Numeric fields default to NaN so missing data is handled gracefully;
    boolean fields default to False.
    """
    # ── Volatility / risk appetite ─────────────────────────────────────────
    vix: float = float("nan")
    vix_level_z: float = float("nan")    # z-score vs trailing 252d
    vix_risk_off: float = 0.0            # 1.0 if VIX >= 20
    vix_crisis: float = 0.0             # 1.0 if VIX >= 30

    # ── Dollar / rates ─────────────────────────────────────────────────────
    dxy_zscore63: float = float("nan")   # DXY z-score (63-day window)
    dxy_mom21: float = float("nan")      # 21-day log-return of DXY
    tlt_mom21: float = float("nan")      # 21-day log-return of TLT (bonds)
    tlt_yield_proxy: float = float("nan") # −1 × tlt daily log-ret (rate dir)

    # ── Calendar / event proximity ─────────────────────────────────────────
    is_opec_window: float = 0.0          # 1.0 inside OPEC± window
    days_to_opec: float = float("nan")   # signed; negative = before meeting
    is_wasde_window: float = 0.0         # 1.0 inside WASDE window
    days_to_wasde: float = float("nan")  # signed; negative = before release
    wasde_post5: float = 0.0             # 1.0 for 5d after WASDE release

    # ── Extension slots (None until underlying features land) ─────────────
    hmm_regime: Optional[str] = None            # "bull" | "neutral" | "bear"
    days_since_regime_flip: Optional[int] = None

    def to_feature_vector(self) -> List[float]:
        """
        Return a list of floats in the order defined by FEATURE_COLUMNS.
        NaN values are preserved — the DecisionTree imputes them via
        surrogates automatically; for inference we replace them with 0.0.
        """
        row = []
        for col in FEATURE_COLUMNS:
            val = getattr(self, col, float("nan"))
            if val is None or (isinstance(val, float) and math.isnan(val)):
                row.append(0.0)   # safe sentinel for tree inference
            else:
                row.append(float(val))
        return row

    def to_dict(self) -> dict:
        return {
            "vix": self.vix,
            "vix_level_z": self.vix_level_z,
            "vix_risk_off": self.vix_risk_off,
            "vix_crisis": self.vix_crisis,
            "dxy_zscore63": self.dxy_zscore63,
            "dxy_mom21": self.dxy_mom21,
            "tlt_mom21": self.tlt_mom21,
            "tlt_yield_proxy": self.tlt_yield_proxy,
            "is_opec_window": self.is_opec_window,
            "days_to_opec": self.days_to_opec,
            "is_wasde_window": self.is_wasde_window,
            "days_to_wasde": self.days_to_wasde,
            "wasde_post5": self.wasde_post5,
            "hmm_regime": self.hmm_regime,
            "days_since_regime_flip": self.days_since_regime_flip,
        }


@dataclass
class ModelVote:
    """
    One model tier's forecast + confidence for a commodity at a point in time.
    The meta-predictor uses a list of these to compute a weighted blend.
    """
    tier: str            # "statistical" | "ml" | "deep" | "quantum"
    model_name: str      # e.g. "ARIMA", "XGBoostForecaster", "Prophet"
    commodity: str       # canonical display name
    forecast_return: float  # e.g. +0.009 = +0.9%
    confidence: float       # IC or equivalent proxy, [0, 1]
    horizon: int = 1        # forecast horizon in trading days

    def __post_init__(self):
        if self.tier not in KNOWN_TIERS:
            raise ValueError(
                f"Unknown tier {self.tier!r}. "
                f"Add it to KNOWN_TIERS if this is intentional."
            )


@dataclass
class MetaDecision:
    """
    The meta-predictor's output for a single inference run.

    weights
        Per-tier blend weights.  Sum to 1.0.  The highest-weight tier is
        `trusted_tier`.

    ensemble_forecast
        Dot product of weights × per-tier forecast returns (for tiers that
        have a vote).  Ready to surface in the dashboard as the consensus
        forecast.

    consensus_strength
        Measures agreement across voting tiers.  1.0 = all tiers agree on
        direction; 0.0 = equal split of bull/bear forecasts.

    meta_confidence
        How confident the meta-predictor itself is in its weight assignment.
        Before training this is 0.0 (equal-weight fallback); after training
        it reflects the tree's class probability gap.

    reasoning
        Human-readable string explaining what drove the weight assignment.
        Surfaced directly in the dashboard UI.
    """
    weights: Dict[str, float]
    trusted_tier: str
    ensemble_forecast: float
    consensus_strength: float      # [0, 1]
    meta_confidence: float         # [0, 1]
    reasoning: str
    votes_used: List[ModelVote] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "trusted_tier": self.trusted_tier,
            "ensemble_forecast": self.ensemble_forecast,
            "consensus_strength": self.consensus_strength,
            "meta_confidence": self.meta_confidence,
            "reasoning": self.reasoning,
            "votes_used": [
                {
                    "tier": v.tier,
                    "model_name": v.model_name,
                    "forecast_return": v.forecast_return,
                    "confidence": v.confidence,
                }
                for v in self.votes_used
            ],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Feature collector ─────────────────────────────────────────────────────────

def collect_meta_features(macro_df: pd.DataFrame) -> MetaFeatures:
    """
    Extract the latest market state from the macro overlay DataFrame returned
    by `features.macro_overlays.load_macro()`.

    Parameters
    ----------
    macro_df : pd.DataFrame
        Output of load_macro(); must have a DatetimeIndex.  At minimum the
        columns listed in FEATURE_COLUMNS should be present, but the function
        gracefully handles missing columns by leaving MetaFeatures at their
        default NaN / 0.0 values.

    Returns
    -------
    MetaFeatures
        Snapshot of the last available row in macro_df.
    """
    if macro_df is None or macro_df.empty:
        return MetaFeatures()

    last = macro_df.iloc[-1]

    def _get(col: str, default=float("nan")):
        if col in macro_df.columns:
            val = last[col]
            return default if pd.isna(val) else val
        return default

    return MetaFeatures(
        vix=_get("vix"),
        vix_level_z=_get("vix_level_z"),
        vix_risk_off=float(_get("vix_risk_off", 0.0)),
        vix_crisis=float(_get("vix_crisis", 0.0)),
        dxy_zscore63=_get("dxy_zscore63"),
        dxy_mom21=_get("dxy_mom21"),
        tlt_mom21=_get("tlt_mom21"),
        tlt_yield_proxy=_get("tlt_yield_proxy"),
        is_opec_window=float(_get("is_opec_window", 0.0)),
        days_to_opec=_get("days_to_opec"),
        is_wasde_window=float(_get("is_wasde_window", 0.0)),
        days_to_wasde=_get("days_to_wasde"),
        wasde_post5=float(_get("wasde_post5", 0.0)),
        # HMM regime not yet available — extension point
        hmm_regime=None,
        days_since_regime_flip=None,
    )


# ── Meta-Predictor ────────────────────────────────────────────────────────────

class MetaPredictor:
    """
    Thin, interpretable arbitrator that outputs ensemble weights.

    Untrained state
    ───────────────
    Returns equal weights and `meta_confidence=0.0`.  The dashboard stays
    fully live before historical training data exists.

    Training (Phase 3 Part 2)
    ─────────────────────────
    Call `fit(records)` where `records` is a list of
        (MetaFeatures, winning_tier: str)
    tuples produced by the backtesting harness.  A shallow DecisionTreeClassifier
    (max_depth=5) is fitted and optionally persisted to disk via `save()`.

    Live inference
    ──────────────
    `predict(features, votes)` returns a MetaDecision.  The weights come from
    the tree's `predict_proba` if trained, otherwise equal weights.  The
    ensemble_forecast is a weighted average of vote forecasts — but only
    votes from *present* tiers contribute; absent tiers don't dilute the
    blend.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self._tree = None          # sklearn DecisionTreeClassifier | None
        self._classes: List[str] = []
        self._feature_importances: Dict[str, float] = {}
        self._training_records: int = 0

        if model_path is not None:
            self.load(model_path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._tree is not None

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Ordered dict of feature → Gini importance (0 if untrained)."""
        return dict(self._feature_importances)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        records: List[Tuple["MetaFeatures", str]],
        max_depth: int = 5,
        min_samples_leaf: int = 10,
    ) -> "MetaPredictor":
        """
        Train the meta-predictor from historical (features, winning_tier) pairs.

        Parameters
        ----------
        records : list of (MetaFeatures, str)
            Each tuple is (market_state_on_date, tier_that_had_lowest_error).
        max_depth : int
            Cap on tree depth.  Keep shallow for interpretability.
        min_samples_leaf : int
            Prevents overfitting on small backtests.

        Returns
        -------
        self  (for chaining)
        """
        if not records:
            raise ValueError("records is empty — nothing to train on")

        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for MetaPredictor.fit(). "
                "Install with: pip install scikit-learn"
            ) from exc

        X = np.array([mf.to_feature_vector() for mf, _ in records], dtype=float)
        y = [tier for _, tier in records]

        unknown = set(y) - set(KNOWN_TIERS)
        if unknown:
            raise ValueError(
                f"Unknown tiers in training labels: {unknown}. "
                f"Add them to KNOWN_TIERS first."
            )

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        clf.fit(X, y)
        self._tree = clf
        self._classes = list(clf.classes_)
        self._training_records = len(records)

        # Store feature importances for the explain() surface
        self._feature_importances = {
            col: float(imp)
            for col, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
        }
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features: MetaFeatures,
        votes: List[ModelVote],
    ) -> MetaDecision:
        """
        Arbitrate between model tiers and return blended weights + forecast.

        Parameters
        ----------
        features : MetaFeatures
            Latest market state (from collect_meta_features).
        votes : list[ModelVote]
            Forecasts from whichever tiers are available this refresh.
            Missing tiers simply don't contribute to the ensemble.

        Returns
        -------
        MetaDecision
        """
        present_tiers = {v.tier for v in votes}

        # ── 1. Derive per-tier weights ────────────────────────────────────────
        if not self.is_trained or not votes:
            weights = {t: 1.0 / len(KNOWN_TIERS) for t in KNOWN_TIERS}
            meta_confidence = 0.0
            weight_source = "equal (untrained)"
        else:
            fv = np.array([features.to_feature_vector()])
            proba = self._tree.predict_proba(fv)[0]
            raw = {cls: float(p) for cls, p in zip(self._classes, proba)}
            # Fill any tier not in tree classes with 0
            weights = {t: raw.get(t, 0.0) for t in KNOWN_TIERS}
            # Meta-confidence = gap between top-2 class probabilities
            sorted_p = sorted(raw.values(), reverse=True)
            meta_confidence = (sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else sorted_p[0]
            weight_source = "tree"

        # ── 2. Re-normalise weights to present tiers only ─────────────────────
        present_weights = {t: weights[t] for t in present_tiers if t in weights}
        total = sum(present_weights.values()) or 1.0
        norm_weights = {t: w / total for t, w in present_weights.items()}

        # Fill absent tiers with 0 for the output dict
        final_weights = {t: norm_weights.get(t, 0.0) for t in KNOWN_TIERS}

        trusted_tier = max(final_weights, key=lambda t: final_weights[t])

        # ── 3. Weighted ensemble forecast ─────────────────────────────────────
        ensemble_forecast = sum(
            norm_weights.get(v.tier, 0.0) * v.forecast_return
            for v in votes
        )

        # ── 4. Consensus strength ─────────────────────────────────────────────
        # 1.0 = all votes same direction, 0.0 = equal bull/bear split
        if votes:
            bull_w = sum(norm_weights.get(v.tier, 0.0) for v in votes if v.forecast_return >= 0)
            bear_w = 1.0 - bull_w
            consensus_strength = abs(bull_w - bear_w)
        else:
            consensus_strength = 0.0

        # ── 5. Human-readable reasoning ──────────────────────────────────────
        reasoning = _build_reasoning(features, final_weights, trusted_tier, weight_source)

        return MetaDecision(
            weights=final_weights,
            trusted_tier=trusted_tier,
            ensemble_forecast=ensemble_forecast,
            consensus_strength=consensus_strength,
            meta_confidence=meta_confidence,
            reasoning=reasoning,
            votes_used=list(votes),
            metadata={
                "weight_source": weight_source,
                "n_training_records": self._training_records,
                "top_feature": self._top_feature(),
            },
        )

    # ── Explain ───────────────────────────────────────────────────────────────

    def explain(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top-n most important features driving the tree's decisions.
        Returns an empty list when untrained.
        """
        if not self.is_trained:
            return []
        return sorted(
            self._feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the trained tree to disk. Returns the path written."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained MetaPredictor.")
        p = Path(path) if path else _DEFAULT_MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(
                {
                    "tree": self._tree,
                    "classes": self._classes,
                    "feature_importances": self._feature_importances,
                    "training_records": self._training_records,
                    "feature_columns": list(FEATURE_COLUMNS),
                },
                fh,
            )
        return p

    def load(self, path: Optional[Path] = None) -> "MetaPredictor":
        """Load a persisted tree from disk. Silent no-op if file doesn't exist."""
        p = Path(path) if path else _DEFAULT_MODEL_PATH
        if not p.exists():
            return self
        with open(p, "rb") as fh:
            state = pickle.load(fh)
        self._tree = state["tree"]
        self._classes = state["classes"]
        self._feature_importances = state.get("feature_importances", {})
        self._training_records = state.get("training_records", 0)
        return self

    # ── Private helpers ───────────────────────────────────────────────────────

    def _top_feature(self) -> Optional[str]:
        if not self._feature_importances:
            return None
        return max(self._feature_importances, key=lambda k: self._feature_importances[k])


# ── Reasoning builder ─────────────────────────────────────────────────────────

def _build_reasoning(
    f: MetaFeatures,
    weights: Dict[str, float],
    trusted: str,
    source: str,
) -> str:
    """
    Compose a short, human-readable explanation for today's weight assignment.
    The logic here mirrors what a senior analyst would say looking at the
    same numbers — it's deterministic and rule-based so it's always legible.
    """
    parts: List[str] = []

    # Market regime characterisation
    if not math.isnan(f.vix) and f.vix > 0:
        if f.vix_crisis:
            parts.append(f"VIX {f.vix:.0f} (crisis — >30)")
        elif f.vix_risk_off:
            parts.append(f"VIX {f.vix:.0f} (risk-off — >20)")
        else:
            parts.append(f"VIX {f.vix:.0f} (calm)")

    if not math.isnan(f.dxy_zscore63):
        sign = "elevated" if f.dxy_zscore63 > 0 else "depressed"
        parts.append(f"DXY z={f.dxy_zscore63:+.2f} ({sign})")

    if not math.isnan(f.tlt_mom21):
        dir_ = "rising" if f.tlt_mom21 < 0 else "falling"
        parts.append(f"yields {dir_} (TLT mom={f.tlt_mom21:+.3f})")

    # Event proximity
    if f.is_opec_window:
        d = f.days_to_opec
        if not math.isnan(d):
            loc = f"{abs(int(d))}d {'before' if d < 0 else 'after'} OPEC"
            parts.append(f"OPEC window ({loc})")

    if f.is_wasde_window:
        d = f.days_to_wasde
        if not math.isnan(d):
            loc = f"{abs(int(d))}d {'before' if d < 0 else 'after'} WASDE"
            parts.append(f"WASDE window ({loc})")
    elif f.wasde_post5:
        parts.append("WASDE digestion window (post-5d)")

    # HMM regime — when available
    if f.hmm_regime:
        parts.append(f"HMM regime: {f.hmm_regime}")
        if f.days_since_regime_flip is not None:
            parts.append(f"({f.days_since_regime_flip}d since flip)")

    # Weight source tag
    if source == "equal (untrained)":
        suffix = "→ equal weights (meta-predictor not yet trained)"
    else:
        w_pct = int(weights.get(trusted, 0.0) * 100)
        suffix = f"→ trusting {trusted} tier ({w_pct}% weight)"

    context = "; ".join(parts) if parts else "no distinctive macro conditions"
    return f"{context}. {suffix}."
