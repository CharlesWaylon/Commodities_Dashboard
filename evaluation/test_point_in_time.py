"""
Look-ahead property test: for EVERY registered signal, computing as-of date t
must not change when future rows are appended. This is the single discipline that
keeps the backtest honest.
"""

import pandas as pd
import pytest

from evaluation.point_in_time import assert_point_in_time, make_synthetic_panel
from signals.base import get_signal, list_signals


@pytest.mark.parametrize("name", list(list_signals()))
@pytest.mark.parametrize("seed", [0, 7])
def test_signal_is_point_in_time(name, seed):
    panel = make_synthetic_panel(n_days=600, seed=seed)
    signal = get_signal(name)
    # asof in the interior so there are genuine future rows to (illegally) peek at.
    asof = panel.index[400]
    assert_point_in_time(signal, panel, asof, n_future_rows=len(panel) - 401)


def test_helper_catches_a_leaky_signal():
    """A deliberately leaky signal must trip the assertion (proves the test bites)."""
    from signals.base import Signal, FORECAST_FIELD, CONFIDENCE_FIELD

    class Leaky(Signal):
        name = "_leaky_test"
        economic_rationale = "deliberately peeks at the future to prove the PIT test bites"
        horizons = (5,)

        def compute(self, asof, panel):
            # BUG ON PURPOSE: uses the LAST row of the full panel, not the asof row.
            last = panel.iloc[-1]
            cols = pd.MultiIndex.from_product([self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]],
                                              names=["horizon", "field"])
            out = pd.DataFrame(index=panel.columns, columns=cols, dtype=float)
            out.index.name = "instrument"
            out[(5, FORECAST_FIELD)] = last.values
            out[(5, CONFIDENCE_FIELD)] = 0.5
            return out

    panel = make_synthetic_panel(n_days=300)
    with pytest.raises(AssertionError):
        assert_point_in_time(Leaky(), panel, panel.index[200], n_future_rows=99)
