"""
Tests for the cascade-augmented allocator (portfolio/cascade_allocator.py).

Covers:
  • Falls back to the signal-forecast MV selection when cascade data is empty
    (the design that lets it run through historical backtests without crashing).
  • Substitutes cascade ranks for the signal on the overlap when cascade IS
    available (the legacy "cascade μ" behaviour).
  • Output respects the standard selection / vol-target contracts.
"""

import numpy as np
import pandas as pd
from unittest import mock

from portfolio.allocators import MeanVarianceSelectAllocator
from portfolio.cascade_allocator import CascadeAugmentedAllocator
from portfolio.risk import RiskModel


def _risk_model(n=8, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.01, 0.03, n)
    c = 0.25 * np.ones((n, n)); np.fill_diagonal(c, 1.0)
    cov_arr = np.outer(vols, vols) * c
    cols = [f"A{i}" for i in range(n)]
    cov = pd.DataFrame(cov_arr, index=cols, columns=cols)
    vol = pd.Series(np.sqrt(np.diag(cov_arr)), index=cols)
    return RiskModel(cov=cov, vol=vol, shrinkage=0.0,
                     asof=pd.Timestamp("2026-06-05"), instruments=tuple(cols))


def test_falls_back_to_signal_when_cascade_empty():
    """No cascade data → must produce the SAME book as MeanVarianceSelectAllocator."""
    rm = _risk_model(seed=1)
    forecasts = pd.Series([3.0, 2.0, 1.5, 0.5, 0.0, -1.0, -2.0, -3.0], index=list(rm.vol.index))

    with mock.patch("portfolio.cascade_allocator.load_cascade_view",
                    return_value=pd.Series(dtype=float)):
        casc = CascadeAugmentedAllocator(k=3, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    mv = MeanVarianceSelectAllocator(k=3, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    # SAME selection AND same equal-weight vol-target → series equal.
    assert set(casc.index) == set(mv.index)
    assert np.allclose(casc.sort_index().values, mv.sort_index().values, atol=1e-9)


def test_cascade_substitution_changes_the_selection():
    """When cascade reverses the signal's ranking on the overlap, the selection
    must move toward cascade's preferred names."""
    rm = _risk_model(n=6, seed=2)
    cols = list(rm.vol.index)
    # Signal prefers A0..A2 strongly
    forecasts = pd.Series([3.0, 2.5, 2.0, -1.0, -2.0, -3.0], index=cols)
    # Cascade flips the verdict on the same instruments
    cascade = pd.Series([-3.0, -2.5, -2.0, 1.0, 2.0, 3.0], index=cols)

    with mock.patch("portfolio.cascade_allocator.load_cascade_view", return_value=cascade):
        casc = CascadeAugmentedAllocator(k=2, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    mv = MeanVarianceSelectAllocator(k=2, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    # MV picks from the signal's top; cascade picks from the OPPOSITE side.
    assert set(mv.index) <= {"A0", "A1", "A2"}
    assert set(casc.index) <= {"A3", "A4", "A5"}
    assert set(casc.index).isdisjoint(set(mv.index))


def test_partial_cascade_overlap_promotes_overlap_winner_and_preserves_rest():
    """Cascade covers a subset; its winner is promoted into the selection, and
    the non-cascade names keep their signal-relative ordering."""
    rm = _risk_model(n=8, seed=3)
    cols = list(rm.vol.index)
    forecasts = pd.Series(np.linspace(3, -3, 8), index=cols)
    # cascade only covers A6, A7 (the signal's worst). A6 is cascade's winner.
    cascade = pd.Series([5.0, 4.0], index=["A6", "A7"])

    with mock.patch("portfolio.cascade_allocator.load_cascade_view", return_value=cascade):
        casc = CascadeAugmentedAllocator(k=4, n_universe=8, target_vol=0.10).allocate(forecasts, rm)
    # A6 — cascade's top covered name — must be promoted into the book.
    assert "A6" in casc.index, f"A6 must be promoted by cascade, got {list(casc.index)}"
    # A5 — signal's worst, NOT covered by cascade — must NOT be in the book
    # (cascade leaves its signal verdict untouched, so it stays at the bottom).
    assert "A5" not in casc.index


def test_no_asof_falls_back_safely():
    """If risk_model has no asof, allocator must still produce a valid book via fallback."""
    rm = _risk_model(seed=4)
    rm.asof = None
    forecasts = pd.Series(np.linspace(3, -3, 8), index=list(rm.vol.index))
    casc = CascadeAugmentedAllocator(k=3, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    assert not casc.empty and len(casc) == 3


def test_dollar_neutrality_not_required_for_selection_book():
    """Selection allocators are long-only equal-weight; positive gross expected."""
    rm = _risk_model(seed=5)
    forecasts = pd.Series(np.linspace(3, -3, 8), index=list(rm.vol.index))
    with mock.patch("portfolio.cascade_allocator.load_cascade_view",
                    return_value=pd.Series(dtype=float)):
        w = CascadeAugmentedAllocator(k=3, n_universe=6, target_vol=0.10).allocate(forecasts, rm)
    assert (w > 0).all() and w.sum() > 0
