"""
signals/ — the research/signal layer.

Every economically-grounded edge implements the ``Signal`` interface and is
resolvable by name through the registry. This layer must never import streamlit,
pages, app, or the portfolio layer (enforced by ``.importlinter``).
"""

from signals.base import (  # noqa: F401
    DEFAULT_HORIZONS,
    CONFIDENCE_FIELD,
    FORECAST_FIELD,
    Signal,
    get_signal,
    list_signals,
    register_signal,
)

__all__ = [
    "Signal",
    "register_signal",
    "get_signal",
    "list_signals",
    "DEFAULT_HORIZONS",
    "FORECAST_FIELD",
    "CONFIDENCE_FIELD",
]
