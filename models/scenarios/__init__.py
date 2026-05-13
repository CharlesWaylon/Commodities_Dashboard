"""
Scenario-band layer: bear / average / bull forecast envelopes built on top
of the project's existing forecasters.

Phase 1 covers the statistical models (ARIMA, VAR). Subsequent phases will
add conformal ML, MC-dropout DL, and quantum providers behind the same
``ScenarioBand`` contract.
"""

from models.scenarios.band import ScenarioBand, returns_to_price_paths, collapse_to_tail
from models.scenarios.analogs import find_analogs, HistoricalAnalog
from models.scenarios.aggregator import ScenarioAggregator
from models.scenarios.providers import (
    arima_band,
    var_band,
    elastic_net_band,
    random_forest_band,
    xgboost_band,
    lstm_band,
    tft_band,
    prophet_band,
    quantum_kernel_band,
)
from models.scenarios.ripple import CausalRipple
from models.scenarios.narrative import build_narrative
from models.scenarios.calibration import empirical_coverage
from models.scenarios.conformal import ConformalCalibration, calibrate_from_model

__all__ = [
    "ScenarioBand",
    "ScenarioAggregator",
    "arima_band",
    "var_band",
    "elastic_net_band",
    "random_forest_band",
    "xgboost_band",
    "lstm_band",
    "tft_band",
    "prophet_band",
    "quantum_kernel_band",
    "CausalRipple",
    "build_narrative",
    "collapse_to_tail",
    "find_analogs",
    "HistoricalAnalog",
    "empirical_coverage",
    "returns_to_price_paths",
    "ConformalCalibration",
    "calibrate_from_model",
]
