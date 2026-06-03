"""
Tests for the portfolio-wide data-quality gate. Synthetic panels are constructed
so each check fires (or stays silent) deterministically — no DB, no network.
"""

from datetime import date

import numpy as np
import pandas as pd

from data.validation import QualityConfig, run_quality_report


def _clean_panel(n=120, tickers=("A", "B"), seed=0):
    """A clean business-day panel of gently drifting prices."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_clean_panel_passes():
    rep = run_quality_report(_clean_panel(), asof=None)
    assert rep.ok
    assert rep.of_severity("error") == []


def test_empty_panel_is_error():
    rep = run_quality_report(pd.DataFrame())
    assert not rep.ok
    assert rep.of_kind("coverage")


def test_staleness_flagged_against_later_asof():
    panel = _clean_panel(n=60)
    # asof a month after the last bar -> stale beyond 2x threshold -> error.
    rep = run_quality_report(panel, asof=date(2026, 5, 1))
    stale = rep.of_kind("staleness")
    assert stale and any(i.severity == "error" for i in stale)
    assert not rep.ok


def test_outlier_spike_flagged():
    panel = _clean_panel(n=120)
    # Inject a +60% one-day jump well past the hard 40% threshold.
    panel.iloc[80, panel.columns.get_loc("A")] *= 1.6
    rep = run_quality_report(panel, asof=panel.index[-1].date())
    outs = rep.of_kind("outlier")
    assert any(i.instrument == "A" for i in outs)


def test_calendar_gap_flagged():
    panel = _clean_panel(n=60)
    # Drop a two-week block from the middle of one column to open an interior hole.
    panel.iloc[20:30, panel.columns.get_loc("B")] = np.nan
    rep = run_quality_report(panel, asof=panel.index[-1].date())
    assert any(i.instrument == "B" for i in rep.of_kind("calendar_gap"))


def test_coverage_missing_expected_ticker_is_error():
    panel = _clean_panel(tickers=("A",))
    rep = run_quality_report(panel, expected=["A", "MISSING"],
                             asof=panel.index[-1].date())
    cov = rep.of_kind("coverage")
    assert any(i.instrument == "MISSING" and i.severity == "error" for i in cov)
    assert not rep.ok


def test_report_to_frame_shape():
    panel = _clean_panel()
    rep = run_quality_report(panel)
    df = rep.to_frame()
    assert list(df.columns) == ["kind", "severity", "instrument", "detail"]
    assert rep.n_instruments == panel.shape[1]
