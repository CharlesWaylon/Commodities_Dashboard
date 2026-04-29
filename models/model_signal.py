"""
Signal schema — standardized output from all model tiers.

Every model (statistical, ML, deep, quantum) emits ModelSignal objects
instead of raw arrays. This enables:
  1. Confidence routing downstream
  2. Causal chain tracing (which features drove this forecast?)
  3. Meta-predictor learning (which model wins under these conditions?)
  4. Ensemble arbitration (when models disagree)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
import json


@dataclass
class ModelSignal:
    """
    Standardized output from a predictive model.

    Carries not just the forecast but the context needed for downstream
    integration: confidence, reasoning, uncertainty, and metadata for
    causal tracing.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    commodity: str                           # e.g., "WTI Crude Oil"
    model_type: str                          # e.g., "xgboost", "lstm", "arima"
    model_name: str                          # e.g., "XGBoostForecaster v1.2"
    timestamp: str                           # ISO 8601, e.g. "2026-04-28T18:30:00+00:00"

    # ── Forecast ───────────────────────────────────────────────────────────
    forecast_return: float                   # Next-day log return (or N-day forward)
    forecast_price: Optional[float] = None  # Absolute price level (if ARIMA)
    horizon: int = 1                         # Days ahead (1, 5, 20, etc.)

    # ── Confidence ─────────────────────────────────────────────────────────
    confidence: float = 0.0                  # IC, R², accuracy metric (0–1 scale)
    confidence_metric: str = "ic"            # What confidence measures ("ic", "r2", "mae")
    uncertainty_band: Tuple[float, float] = (0.0, 0.0)  # (lower, upper) bounds

    # ── Market Context ─────────────────────────────────────────────────────
    regime: str = "neutral"                  # "bull", "bear", "neutral", "high-vol"
    regime_confidence: float = 0.5           # How confident in the regime label (0–1)

    # ── Reasoning ──────────────────────────────────────────────────────────
    reasoning: List[str] = field(default_factory=list)
    # e.g., ["momentum_5d_positive", "copper_zscore_support", "vix_low"]

    top_bullish_drivers: List[Tuple[str, float]] = field(default_factory=list)
    # e.g., [("momentum_5d", +0.35), ("copper_zscore", +0.12)]

    top_bearish_drivers: List[Tuple[str, float]] = field(default_factory=list)
    # e.g., [("dxy_zscore", -0.18), ("vix_spike", -0.08)]

    # ── Dependencies ───────────────────────────────────────────────────────
    depends_on: List[str] = field(default_factory=list)
    # e.g., ["HMM_regime_detector", "GARCH_vol_forecast"]

    # ── Versioning & Metadata ──────────────────────────────────────────────
    signal_version: str = "0.1"
    model_version: str = "1.0"
    training_period: str = "1000d"
    retraining_frequency: str = "weekly"

    # ── Optional: extended metadata ────────────────────────────────────────
    metadata: Dict = field(default_factory=dict)
    # Extensible dict for model-specific info
    # e.g., {"xgb_n_estimators": 500, "shap_base_value": -0.003}

    def __post_init__(self):
        assert -1.0 <= self.forecast_return <= 1.0, (
            f"forecast_return {self.forecast_return} out of plausible range [-1, 1]"
        )
        assert 0.0 <= self.confidence <= 1.0, (
            f"confidence {self.confidence} must be in [0, 1]"
        )
        assert self.horizon >= 1, "horizon must be >= 1 day"
        assert len(self.reasoning) <= 10, "Keep reasoning to <= 10 items"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["uncertainty_band"] = list(self.uncertainty_band)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def short_repr(self) -> str:
        arrow = "UP" if self.forecast_return > 0 else "DN"
        return (
            f"[{arrow}] {self.commodity}: {self.forecast_return:+.2%} "
            f"(IC={self.confidence:.3f}, regime={self.regime})"
        )


@dataclass
class EnsembleSignal:
    """
    Weighted combination of multiple ModelSignal objects.
    Output of the meta-predictor or a manual ensemble.
    """

    commodity: str
    signals: List[ModelSignal]
    weights: Dict[str, float]           # model_name → weight
    ensemble_forecast: float            # Weighted average forecast
    ensemble_confidence: float          # Weighted confidence
    consensus_regime: str               # Most common regime label
    disagreement_score: float           # 0 = all agree, 1 = fully split

    def to_dict(self) -> Dict:
        return {
            "commodity": self.commodity,
            "ensemble_forecast": self.ensemble_forecast,
            "ensemble_confidence": self.ensemble_confidence,
            "consensus_regime": self.consensus_regime,
            "disagreement_score": self.disagreement_score,
            "weights": self.weights,
        }
