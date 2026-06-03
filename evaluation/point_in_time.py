"""
Point-in-time (anti-look-ahead) correctness — enforced as a TEST, not a hope.

The single discipline that separates a real backtest from a fantasy: computing a
signal's feature as-of date ``t`` must use ONLY data with timestamp ``<= t``.
``assert_point_in_time`` proves it the only way that's airtight — by showing that
*appending future rows changes nothing*. If a signal secretly peeks ahead, the
two computations diverge and the assertion fails.

(When paid fundamental feeds arrive — EIA/WASDE/COT in Phase 1/2 — the same helper
generalises: the panel must be keyed off RELEASE date, not reference date, and
this test catches any signal that keys off the wrong one.)
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from signals.base import Signal


def assert_point_in_time(
    signal: Signal,
    panel: pd.DataFrame,
    asof: pd.Timestamp,
    n_future_rows: int = 30,
    rtol: float = 1e-9,
) -> None:
    """
    Assert ``signal.compute(asof, ...)`` ignores rows after ``asof``.

    Strategy: compute on the panel truncated at ``asof`` and again on the FULL
    panel (which contains ``n_future_rows`` rows after ``asof``). The forecasts
    must be identical. Any difference is a look-ahead leak.

    Raises
    ------
    AssertionError
        If the two computations differ, with the offending cells reported.
    """
    asof = pd.Timestamp(asof)

    truncated = panel.loc[:asof]
    full = panel  # contains future rows beyond asof

    out_truncated = signal.compute(asof, truncated)
    out_full = signal.compute(asof, full)

    # Align on the intersection of instruments both produced a view for.
    common = out_truncated.index.intersection(out_full.index)
    a = out_truncated.loc[common].sort_index()
    b = out_full.loc[common].sort_index()

    if not a.columns.equals(b.columns):
        raise AssertionError(
            f"[{signal.name}] column shape differs between truncated and full panel:\n"
            f"  truncated: {list(a.columns)}\n  full:      {list(b.columns)}"
        )

    diff = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    mask = ~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=rtol, equal_nan=True)
    if mask.any():
        n_bad = int(mask.sum())
        max_abs = float(np.nanmax(np.abs(diff[mask]))) if n_bad else 0.0
        raise AssertionError(
            f"[{signal.name}] LOOK-AHEAD LEAK at asof={asof.date()}: "
            f"{n_bad} forecast cells changed when {n_future_rows} future rows were "
            f"appended (max |Δ|={max_abs:.3e}). The signal is reading data after asof."
        )


def make_synthetic_panel(
    n_days: int = 600,
    instruments: Optional[list] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """A deterministic geometric-random-walk price panel for PIT property tests."""
    if instruments is None:
        instruments = [f"INST_{i}" for i in range(8)]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days, name="Date")
    rets = rng.normal(0.0002, 0.012, size=(n_days, len(instruments)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=instruments)
