"""Contract tests for the Signal interface + registry.

Self-contained on purpose: the signal layer must not import the evaluation layer
(enforced by .importlinter), so this test builds its own synthetic panel rather
than reusing evaluation.point_in_time.make_synthetic_panel.
"""

import numpy as np
import pandas as pd

from signals.base import (
    CONFIDENCE_FIELD,
    FORECAST_FIELD,
    Signal,
    get_signal,
    list_signals,
)


def _panel(n_days: int = 400, n_inst: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days, name="Date")
    rets = rng.normal(0.0002, 0.012, size=(n_days, n_inst))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"INST_{i}" for i in range(n_inst)])


def test_every_registered_signal_has_nonempty_rationale():
    names = list_signals()
    assert names, "no signals registered"
    for name in names:
        sig = get_signal(name)
        assert isinstance(sig, Signal)
        assert sig.economic_rationale and len(sig.economic_rationale.strip()) > 20, (
            f"{name} must declare a substantive economic_rationale"
        )
        assert sig.name == name


def test_momentum_xs_is_registered():
    assert "momentum_xs" in list_signals()


def test_get_unknown_signal_raises():
    try:
        get_signal("nope_not_a_signal")
    except KeyError as e:
        assert "Registered" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for unknown signal")


def test_compute_output_shape():
    sig = get_signal("momentum_xs")
    panel = _panel(n_days=400)
    out = sig.compute(panel.index[-1], panel)
    assert out.index.name == "instrument"
    for h in sig.horizons:
        assert (h, FORECAST_FIELD) in out.columns
        assert (h, CONFIDENCE_FIELD) in out.columns
