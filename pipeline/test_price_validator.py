"""
Unit tests for pipeline.price_validator — focused on the per-row scale
normalisation that replaced the batch-median rescaler (regression: 2026-06-09,
ZR=F mixed-unit fetch corrupted already-correct rows to ~0.124). No DB, no
network: everything operates on synthetic in-memory DataFrames.
"""

import numpy as np
import pandas as pd

from pipeline.price_validator import (
    validate_price_series,
    _scale_factor_for_value,
    SANITY_BANDS,
)


def _raw_df(closes, start="2026-05-01"):
    """Minimal yfinance-shaped OHLCV frame with a DatetimeIndex."""
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c * 1.001, "Low": c * 0.999, "Close": c, "Volume": 100},
        index=idx,
    )


# ── _scale_factor_for_value primitive ───────────────────────────────────────────

def test_scale_factor_in_band_returns_none():
    assert _scale_factor_for_value(12.4, 5.0, 50.0) is None


def test_scale_factor_cents_to_dollars():
    # ~1250 cents/cwt → divide by 100 → 12.5 in band
    assert _scale_factor_for_value(1253.0, 5.0, 50.0) == 100.0


def test_scale_factor_perlb_to_percwt():
    # ~0.124 per-lb → divide by 0.01 (×100) → 12.4 in band
    assert _scale_factor_for_value(0.124, 5.0, 50.0) == 0.01


def test_scale_factor_nonpositive_and_nan():
    assert _scale_factor_for_value(0.0, 5.0, 50.0) is None
    assert _scale_factor_for_value(-5.0, 5.0, 50.0) is None
    assert _scale_factor_for_value(np.nan, 5.0, 50.0) is None


def test_scale_factor_no_power_of_ten_fits():
    # 100000 is beyond every listed factor's reach (max ÷1000 → 100) → no fit.
    assert _scale_factor_for_value(100000.0, 5.0, 50.0) is None


# ── The regression: mixed-unit batch must not corrupt correct rows ───────────────

def test_mixed_unit_batch_does_not_corrupt_correct_rows():
    lo, hi = SANITY_BANDS["ZR=F"]
    # Two rows already in USD/cwt, two rows in cents/cwt — the exact pattern that
    # used to make the median-based rescaler divide the correct rows by 100.
    raw = _raw_df([12.40, 1253.0, 12.55, 1266.0])
    out = validate_price_series("ZR=F", "Rough Rice", raw)

    closes = out.clean_df["Close"].sort_index().tolist()
    # Every surviving close must be inside the band — the POST-CONDITION.
    assert all(lo <= c <= hi for c in closes), closes
    # The originally-correct rows are untouched (not crushed to 0.124).
    assert any(abs(c - 12.40) < 1e-6 for c in closes)
    assert any(abs(c - 12.55) < 1e-6 for c in closes)
    # The cents rows were snapped to dollars.
    assert any(abs(c - 12.53) < 1e-6 for c in closes)
    assert any(abs(c - 12.66) < 1e-6 for c in closes)
    # No row left the band, so nothing should be excluded.
    assert len(out.clean_df) == 4
    # OHLC moved with Close (spot-check the cents row's Open).
    snapped = out.clean_df.sort_index().iloc[1]
    assert 5.0 <= snapped["Open"] <= 50.0


def test_uniform_cents_batch_rescaled_uniformly():
    raw = _raw_df([1240.0, 1255.0, 1248.0])
    out = validate_price_series("ZR=F", "Rough Rice", raw)
    assert out.scale_factor_applied == 100.0
    assert all(5.0 <= c <= 50.0 for c in out.clean_df["Close"])


def test_all_in_band_is_noop():
    raw = _raw_df([12.1, 12.4, 12.6, 12.3])
    out = validate_price_series("ZR=F", "Rough Rice", raw)
    assert out.scale_factor_applied is None
    assert not out.anomalies
    assert out.clean_df["Close"].tolist() == [12.1, 12.4, 12.6, 12.3]


def test_uncorrectable_tick_interpolated_or_excluded():
    lo, hi = SANITY_BANDS["ZR=F"]
    # 1e6 has no power-of-ten factor into [5,50]; surrounded by valid neighbours
    # it interpolates back into band rather than being kept out of band.
    raw = _raw_df([12.4, 1_000_000.0, 12.5, 12.6])
    out = validate_price_series("ZR=F", "Rough Rice", raw)
    assert all(lo <= c <= hi for c in out.clean_df["Close"])
