"""
Tests for Step-7 HistoricalTriggerReplay in models/scenarios/ripple.py.

The class talks to the DB, so tests monkeypatch the event-date lookup to keep
them hermetic.

Coverage
────────
 1. Returns a ScenarioBand with the right shape (mean/bear/bull lengths
    match horizon).
 2. bear ≤ mean ≤ bull at every step (ScenarioBand invariant).
 3. With zero matching events → returns None (no synthetic fallback here).
 4. Empty prices → constructor raises.
 5. Unknown commodity → returns None.
 6. diagnostics carry family / strength / event count.
 7. Importable from the package: `from models.scenarios import HistoricalTriggerReplay`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.scenarios.ripple import HistoricalTriggerReplay
from models.scenarios.band import ScenarioBand


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def prices():
    """3 years of synthetic daily prices for two commodities."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-01", "2026-05-20", freq="B")
    return pd.DataFrame(
        {
            "WTI Crude Oil":  80 * np.exp(rng.normal(0, 0.018, len(idx)).cumsum()),
            "Brent Crude Oil": 82 * np.exp(rng.normal(0, 0.018, len(idx)).cumsum()),
        },
        index=idx,
    )


def _patch_event_dates(monkeypatch, dates):
    """Replace the DB-backed event-date lookup with a fixed list."""
    monkeypatch.setattr(
        HistoricalTriggerReplay, "_fetch_event_dates",
        lambda self, family, min_strength: [pd.Timestamp(d) for d in dates],
    )


# ── 1-2. Shape + invariant ────────────────────────────────────────────────────

def test_replay_returns_band_with_correct_shape(prices, monkeypatch):
    # 10 historical events spaced across the available history.
    _patch_event_dates(monkeypatch, [
        "2024-03-15", "2024-05-20", "2024-08-01", "2024-10-10", "2024-12-12",
        "2025-02-14", "2025-04-18", "2025-07-22", "2025-09-25", "2025-11-30",
    ])
    replay = HistoricalTriggerReplay(prices)
    band = replay.build_band(
        commodity="WTI Crude Oil", family="opec_action",
        min_strength=0.8, horizon=10,
    )
    assert isinstance(band, ScenarioBand)
    assert len(band.mean) == 10
    assert len(band.bear) == 10
    assert len(band.bull) == 10
    # Invariant: bear ≤ mean ≤ bull at every step (enforced by ScenarioBand).
    assert np.all(band.bear <= band.mean + 1e-9)
    assert np.all(band.mean <= band.bull + 1e-9)


# ── 3. Zero matching events → None ────────────────────────────────────────────

def test_replay_returns_none_for_no_events(prices, monkeypatch):
    _patch_event_dates(monkeypatch, [])
    replay = HistoricalTriggerReplay(prices)
    band = replay.build_band(
        commodity="WTI Crude Oil", family="opec_action",
        min_strength=0.99, horizon=5,
    )
    assert band is None


# ── 4. Empty prices → error ──────────────────────────────────────────────────

def test_constructor_rejects_empty_prices():
    with pytest.raises(ValueError):
        HistoricalTriggerReplay(pd.DataFrame())


# ── 5. Unknown commodity → None ───────────────────────────────────────────────

def test_unknown_commodity_returns_none(prices, monkeypatch):
    _patch_event_dates(monkeypatch, ["2024-05-20"])
    replay = HistoricalTriggerReplay(prices)
    band = replay.build_band(commodity="Bitcoin", family="opec_action",
                              min_strength=0.8, horizon=5)
    assert band is None


# ── 6. Diagnostics surface the event count ────────────────────────────────────

def test_diagnostics_carry_event_info(prices, monkeypatch):
    _patch_event_dates(monkeypatch, ["2024-05-20", "2025-04-18", "2025-09-25"])
    replay = HistoricalTriggerReplay(prices)
    band = replay.build_band(
        commodity="WTI Crude Oil", family="opec_action",
        min_strength=0.85, horizon=5,
    )
    diag = band.diagnostics
    assert diag["family"]       == "opec_action"
    assert diag["min_strength"] == 0.85
    assert diag["n_events"]     == 3
    assert diag["horizon"]      == 5


# ── 7. Package-level import ───────────────────────────────────────────────────

def test_class_importable_from_package():
    from models.scenarios import HistoricalTriggerReplay as Re
    assert Re is HistoricalTriggerReplay
