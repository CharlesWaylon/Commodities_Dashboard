"""
Signal interface — the single contract every economically-grounded edge implements.

WHY THIS EXISTS
───────────────
The dashboard's spine is the *gate*, not any one model (see CLAUDE.md DASHBOARD
EVOLUTION RULE + the North Star). For the gate to score every edge the same way,
every edge must look the same to it. That uniform shape is the ``Signal`` class
below.

A Signal is a pure, point-in-time *forecast producer*:

    signal.compute(asof, panel) -> DataFrame
        index   = instrument (display name)
        columns = MultiIndex (horizon, field), field ∈ {"forecast", "confidence"}

  * ``forecast``   — an expected forward return OR a cross-sectional score whose
                     SIGN/RANK predicts the forward return at that horizon. The
                     harness only ever uses it cross-sectionally (Spearman IC and
                     long-short books), so an unscaled relative score is fine.
  * ``confidence`` — a non-negative number in [0, 1]; higher = more conviction.
                     Used later by the ensemble/risk layers, ignored by the IC gate.

POINT-IN-TIME CONTRACT (anti-look-ahead)
────────────────────────────────────────
``compute(asof, panel)`` MUST use only rows of ``panel`` with index ``<= asof``.
Appending future rows to ``panel`` must NOT change the output for a given
``asof``. This is enforced by ``evaluation.point_in_time.assert_point_in_time``
and the property test in ``evaluation/test_point_in_time.py``. A signal that
needs fitting fits *inside* compute on the ``<= asof`` slice — there is no
separate ``fit`` step, which keeps the look-ahead surface to exactly one method.

ECONOMIC RATIONALE IS MANDATORY
────────────────────────────────
``economic_rationale`` must be a non-empty sentence explaining the fundamental
reason the edge exists (inventory→price, positioning→reversal, carry premium,
…). Data-mined patterns without a fundamental reason are exactly the
``days_to_opec`` failure mode this rebuild exists to kill. The non-empty check is
enforced in ``signals/test_base.py``.

This module imports NOTHING from streamlit / pages / app / portfolio — enforced
by the import-linter contracts in ``.importlinter``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Tuple, Type

import pandas as pd

# Default forecast horizons (trading days) every signal is scored at unless it
# overrides. 5 ≈ one week, 10 ≈ two weeks (the legacy FORECAST_HORIZON), 21 ≈ a
# month. The harness scores each horizon SEPARATELY so we discover which horizon
# a given edge actually works at.
DEFAULT_HORIZONS: Tuple[int, ...] = (5, 10, 21)

# Column-level field names produced by every signal.
FORECAST_FIELD = "forecast"
CONFIDENCE_FIELD = "confidence"


class Signal(ABC):
    """Base class for every signal producer. See module docstring for the contract."""

    #: Stable machine name, e.g. "momentum_xs". Used by the registry + CLI + scorecard.
    name: str = ""

    #: Mandatory non-empty fundamental justification (enforced in tests).
    economic_rationale: str = ""

    #: Horizons (trading days) this signal emits a forecast for.
    horizons: Tuple[int, ...] = DEFAULT_HORIZONS

    @abstractmethod
    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a point-in-time forecast as of ``asof``.

        Parameters
        ----------
        asof : datetime.date or pandas.Timestamp
            The decision date. Only ``panel`` rows with index <= asof may be used.
        panel : pd.DataFrame
            Wide price panel: DatetimeIndex rows, one column per instrument
            (display name), values = close prices. (The canonical supply from
            ``models.data_loader.load_price_matrix_from_db``.)

        Returns
        -------
        pd.DataFrame
            index = instrument, columns = MultiIndex.from_product(
                [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]]).
            Instruments with insufficient history are dropped (not NaN-filled),
            so the harness scores only the cross-section that actually has a view.
        """
        raise NotImplementedError

    # ── helpers shared by concrete signals ────────────────────────────────────
    def _empty_frame(self, instruments) -> pd.DataFrame:
        """An all-NaN, correctly-shaped frame (used when no instrument qualifies)."""
        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]],
            names=["horizon", "field"],
        )
        return pd.DataFrame(index=pd.Index(instruments, name="instrument"), columns=cols, dtype=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Signal {self.name!r} horizons={self.horizons}>"


# ── Registry ───────────────────────────────────────────────────────────────────
# A signal becomes resolvable by name (CLI, harness, ensemble) by being registered
# here. Registration is via the @register_signal decorator on the class.
_REGISTRY: Dict[str, Type[Signal]] = {}


def register_signal(cls: Type[Signal]) -> Type[Signal]:
    """Class decorator: register ``cls`` under its ``name`` attribute."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute `name` before registration.")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Signal name {cls.name!r} already registered by {_REGISTRY[cls.name].__name__}.")
    _REGISTRY[cls.name] = cls
    return cls


def get_signal(name: str) -> Signal:
    """Instantiate a registered signal by name. Raises KeyError with the known names."""
    _ensure_signals_imported()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"Unknown signal {name!r}. Registered: {known}")
    return _REGISTRY[name]()


def list_signals() -> Tuple[str, ...]:
    """Return the names of all registered signals."""
    _ensure_signals_imported()
    return tuple(sorted(_REGISTRY))


_IMPORTED = False


def _ensure_signals_imported() -> None:
    """
    Import the concrete-signal modules so their @register_signal decorators run.

    Kept lazy (rather than importing at module top) to avoid an import cycle:
    concrete signals import Signal from this module.
    """
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    from importlib import import_module

    for mod in (
        "signals.momentum",
        "signals.trend",
        "signals.carry",
        "signals.seasonality",
        "signals.cot",
        "signals.inventory",
        "signals.macro",
        "signals.reversal",
        "signals.lowvol",
        "signals.value",
        "signals.ensemble",
    ):
        try:
            import_module(mod)
        except Exception:  # pragma: no cover - a broken signal module shouldn't kill the registry
            pass
