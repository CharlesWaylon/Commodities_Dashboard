"""
SignalBroadcaster — base class for all predictive models.

Every model (ARIMA, XGBoost, LSTM, QAOA, etc.) inherits from this and
implements predict_with_signal(). Guarantees:
  1. Consistent API across all model tiers
  2. Automatic timestamp / versioning / metadata attachment
  3. Easy logging to a signal store
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict

import pandas as pd

from models.model_signal import ModelSignal


class SignalBroadcaster(ABC):
    """
    Abstract base class for all models that emit ModelSignal objects.

    Concrete models (XGBoostForecaster, ARIMAForecaster, etc.) inherit
    from this and implement predict_with_signal().
    """

    def __init__(self, commodity: str, model_type: str, model_name: str):
        """
        Parameters
        ----------
        commodity : str
            Display name, e.g. "WTI Crude Oil"
        model_type : str
            Tier label, e.g. "statistical", "ml", "deep", "quantum"
        model_name : str
            Specific class, e.g. "XGBoostForecaster", "ARIMAForecaster"
        """
        self.commodity = commodity
        self.model_type = model_type
        self.model_name = model_name

        # Subclasses populate these before calling _make_signal()
        self._confidence: float = 0.0
        self._reasoning: List[str] = []
        self._top_bullish: List[Tuple[str, float]] = []
        self._top_bearish: List[Tuple[str, float]] = []
        self._regime: str = "neutral"
        self._regime_confidence: float = 0.5
        self._depends_on: List[str] = []
        self._metadata: Dict = {}

    @abstractmethod
    def predict_with_signal(
        self,
        features: pd.DataFrame,
        horizon: int = 1,
    ) -> ModelSignal:
        """
        Run inference and return a standardized ModelSignal.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix with DatetimeIndex.
        horizon : int
            Days ahead to forecast (default 1).

        Returns
        -------
        ModelSignal
        """

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_signal(
        self,
        forecast_return: float,
        confidence: float,
        horizon: int = 1,
        forecast_price: Optional[float] = None,
        uncertainty_band: Tuple[float, float] = (0.0, 0.0),
        regime: Optional[str] = None,
        regime_confidence: Optional[float] = None,
    ) -> ModelSignal:
        """
        Construct a ModelSignal with standard metadata auto-filled.
        Call this from predict_with_signal() in each subclass.
        """
        return ModelSignal(
            commodity=self.commodity,
            model_type=self.model_type,
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            forecast_return=forecast_return,
            forecast_price=forecast_price,
            horizon=horizon,
            confidence=confidence,
            uncertainty_band=uncertainty_band,
            regime=regime if regime is not None else self._regime,
            regime_confidence=(
                regime_confidence
                if regime_confidence is not None
                else self._regime_confidence
            ),
            reasoning=self._reasoning[:5],
            top_bullish_drivers=self._top_bullish[:3],
            top_bearish_drivers=self._top_bearish[:3],
            depends_on=list(self._depends_on),
            metadata=dict(self._metadata),
        )

    def set_reasoning(self, reasons: List[str]) -> "SignalBroadcaster":
        """Set the reasoning list (top features/events that drove the forecast)."""
        self._reasoning = reasons
        return self

    def set_drivers(
        self,
        bullish: List[Tuple[str, float]],
        bearish: List[Tuple[str, float]],
    ) -> "SignalBroadcaster":
        """Set top bullish and bearish feature drivers with their contributions."""
        self._top_bullish = sorted(bullish, key=lambda x: x[1], reverse=True)
        self._top_bearish = sorted(bearish, key=lambda x: x[1])
        return self

    def set_regime(self, regime: str, confidence: float = 0.5) -> "SignalBroadcaster":
        """Set the market regime context."""
        self._regime = regime
        self._regime_confidence = confidence
        return self

    def set_dependencies(self, depends_on: List[str]) -> "SignalBroadcaster":
        """Declare upstream signals this model depends on."""
        self._depends_on = depends_on
        return self

    def set_metadata(self, **kwargs) -> "SignalBroadcaster":
        """Attach model-specific metadata (hyperparams, training stats, etc.)."""
        self._metadata.update(kwargs)
        return self
