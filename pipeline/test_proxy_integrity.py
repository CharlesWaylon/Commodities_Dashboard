"""
Unit tests for pipeline.proxy_integrity — the cross-asset ratio backstop that
catches scale errors which land INSIDE a sanity band (SIVR-style). Pure detector
tests: no DB, no network.
"""

import numpy as np
import pandas as pd

from pipeline.proxy_integrity import (
    detect_ratio_breaks,
    _nearest_power_of_10,
    RATIO_BREAK_FACTOR,
)


def _series(values, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_nearest_power_of_10():
    assert _nearest_power_of_10(9.5) == 10.0
    assert _nearest_power_of_10(0.11) == 0.1
    assert _nearest_power_of_10(95.0) == 100.0
    assert _nearest_power_of_10(0.0) is None
    assert _nearest_power_of_10(-3.0) is None


def test_no_break_on_normal_tracking():
    # SIVR ≈ silver × 0.96 with ±6% tracking noise — must NOT flag.
    rng = np.random.default_rng(0)
    n = 120
    silver = _series(60 + np.cumsum(rng.normal(0, 0.4, n)))
    sivr = silver * (0.96 + rng.normal(0, 0.02, n))
    breaks = detect_ratio_breaks(sivr, silver, recent_cutoff=silver.index[-5])
    assert breaks == []


def test_detects_div10_corruption_sivr_style():
    # Real ~$70 SIVR values rescaled ÷10 to ~$7 — lands inside the band, but the
    # ratio collapses 10× → must flag with a ×10 suggested fix.
    n = 120
    silver = _series(np.full(n, 75.0))
    sivr = silver * 0.96
    sivr.iloc[-3:] = sivr.iloc[-3:] / 10.0     # corrupt the last 3 days
    breaks = detect_ratio_breaks(sivr, silver, recent_cutoff=silver.index[-5])
    assert len(breaks) == 3
    for b in breaks:
        assert b["suggested_factor"] == 10.0
        assert b["deviation_x"] > 5.0


def test_detects_times10_corruption():
    n = 120
    gold = _series(np.full(n, 5000.0))
    sgol = gold * 0.0096
    sgol.iloc[-1] = sgol.iloc[-1] * 10.0       # ×10 the last day
    breaks = detect_ratio_breaks(sgol, gold, recent_cutoff=gold.index[-5])
    assert len(breaks) == 1
    assert breaks[0]["suggested_factor"] == 0.1   # multiply by 0.1 to restore


def test_only_recent_breaks_reported():
    # An old break (outside the recent window) is not re-flagged.
    n = 120
    silver = _series(np.full(n, 75.0))
    sivr = silver * 0.96
    sivr.iloc[10] = sivr.iloc[10] / 10.0       # break long ago
    breaks = detect_ratio_breaks(sivr, silver, recent_cutoff=silver.index[-5])
    assert breaks == []


def test_insufficient_history_returns_empty():
    silver = _series(np.full(10, 75.0))
    sivr = silver * 0.96
    sivr.iloc[-1] /= 10.0
    assert detect_ratio_breaks(sivr, silver, recent_cutoff=silver.index[-5]) == []
