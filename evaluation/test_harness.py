"""
Harness plumbing tests. These do NOT hit the database or network — they run the
gate on a synthetic panel to prove the mechanics (scoring, folds, costs, verdict,
de-overlap) work and that a planted edge is detected.
"""

import numpy as np
import pandas as pd

from evaluation.harness import HarnessConfig, run_signal
from evaluation.point_in_time import make_synthetic_panel
from signals.base import FORECAST_FIELD, CONFIDENCE_FIELD, Signal, get_signal


def test_random_signal_is_rejected():
    """A pure-noise signal must fail the gate (no real edge)."""
    panel = make_synthetic_panel(n_days=700, seed=3)
    sig = get_signal("momentum_xs")
    card = run_signal(sig, panel, horizons=(5, 10), config=HarnessConfig(min_history=260))
    # On a random walk, momentum has no edge; verdict should be reject.
    assert card.overall_verdict() == "reject"
    for h in card.horizons:
        assert h.n_obs > 0


def test_planted_edge_is_detected():
    """A signal with a strong (but realistically noisy) view of the future must
    PROMOTE — proves the scoring + verdict path detects a genuine edge."""

    class NoisyOracle(Signal):
        name = "_oracle_test"
        economic_rationale = "test fixture: a noisy peek at the future to validate scoring + verdict math"
        horizons = (5,)

        def __init__(self, future_score: pd.DataFrame, noise: float = 0.5, seed: int = 1):
            self._fs = future_score
            self._noise = noise
            self._rng = np.random.default_rng(seed)

        def compute(self, asof, panel):
            asof = pd.Timestamp(asof)
            cols = pd.MultiIndex.from_product([self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]],
                                              names=["horizon", "field"])
            out = pd.DataFrame(index=panel.columns, columns=cols, dtype=float)
            out.index.name = "instrument"
            if asof not in self._fs.index:
                return out
            row = self._fs.loc[asof].to_numpy(dtype=float)
            noisy = row + self._noise * np.nanstd(row) * self._rng.standard_normal(len(row))
            out[(5, FORECAST_FIELD)] = noisy
            out[(5, CONFIDENCE_FIELD)] = 1.0
            return out

    panel = make_synthetic_panel(n_days=700, seed=5)
    log_ret = np.log(panel).diff()
    fwd5 = log_ret.rolling(5).sum().shift(-5)  # the true future the oracle "knows" (noisily)

    card = run_signal(NoisyOracle(fwd5), panel, horizons=(5,),
                      config=HarnessConfig(min_history=260))
    h = card.horizons[0]
    assert h.ic_mean > 0.3, f"noisy oracle should still have a strong positive IC, got {h.ic_mean}"
    assert np.isfinite(h.ic_ir) and h.ic_ir > 0
    assert h.verdict == "promote", f"reasons: {h.reasons}"


def test_cost_model_reduces_pnl():
    from evaluation.costs import TransactionCostModel

    cm_cheap = TransactionCostModel(cost_bps=0.0)
    cm_dear = TransactionCostModel(cost_bps=100.0)
    prev = pd.Series({"A": 0.0, "B": 0.0})
    new = pd.Series({"A": 0.5, "B": -0.5})
    assert cm_dear.cost(prev, new) > cm_cheap.cost(prev, new)
    assert np.isclose(cm_cheap.cost(prev, new), 0.0)
