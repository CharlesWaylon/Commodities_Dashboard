"""
Inventory surprise — the energy theory-of-storage signal (EIA weekly stocks).

ECONOMIC RATIONALE
──────────────────
The theory of storage (Working 1949; Fama-French 1987; Gorton-Hayashi-Rouwenhorst
2013 "The Fundamentals of Commodity Futures Returns") says the state of inventory
is THE fundamental driver of the commodity risk premium: when stocks are scarce
relative to normal, the convenience yield is high, the curve is backwardated, and
expected spot returns are HIGH; when stocks are abundant the curve is contangoed
and returns are low. Empirically, low-inventory commodities earn large positive
excess returns and high-inventory ones earn near-zero.

The tradeable quantity is inventory relative to its SEASONAL norm — raw stock
levels are dominated by seasonality (crude builds in spring, gasoline draws over
summer driving season, natural gas injects Apr-Oct and withdraws Nov-Mar), so the
"surprise" that matters is how far current stocks sit above/below where they
normally are for this week of the year. A surplus vs. the seasonal norm is bearish
(glut), a deficit is bullish (scarcity). Forecast = -z(seasonal deviation).

Scored on the EIA-covered energy sub-universe (crude→WTI, gasoline→RBOB,
distillate→Heating Oil, nat-gas→Natural Gas) — 4 instruments, so the harness must
be run with ``--min-cross-section 4``.

POINT-IN-TIME
─────────────
EIA petroleum prints Wednesday, nat-gas Thursday (release-date lag baked into the
store at ingest). ``compute(asof, panel)`` reads stocks via the store's raw rows
filtered to ``release_date <= asof`` (latest vintage per report week), so a
decision on date ``t`` only sees reports public by ``t``. Raw rows load ONCE and
the PIT filter replays in memory per date. The seasonal norm and residual scale
use only history strictly before the current report week — no look-ahead.
``panel`` supplies only the instrument set (no price look-ahead).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from signals.base import CONFIDENCE_FIELD, FORECAST_FIELD, Signal, register_signal


@register_signal
class InventorySurprise(Signal):
    """EIA weekly-stocks deviation from the trailing seasonal norm (theory of storage)."""

    name = "inventory_surprise"
    economic_rationale = (
        "Theory of storage (Working 1949; Gorton-Hayashi-Rouwenhorst 2013): "
        "inventory scarcity vs. the seasonal norm drives the commodity risk "
        "premium — low stocks => high convenience yield => backwardation => high "
        "expected returns. Uses EIA weekly stocks deseasonalised by week-of-year; "
        "forecast = -z(stocks vs seasonal norm), so a surplus is bearish and a "
        "deficit bullish. Energy sub-universe (crude/gasoline/distillate/nat-gas)."
    )

    def __init__(self, seasonal_window: int = 260, min_weeks: int = 156, min_same_week: int = 3):
        # ~5y trailing window to estimate the seasonal shape; require ~3y of weekly
        # history (so each week-of-year is seen ~3×) before an instrument scores.
        self.seasonal_window = int(seasonal_window)
        self.min_weeks = int(min_weeks)
        self.min_same_week = int(min_same_week)
        self._raw: Optional[pd.DataFrame] = None

    def _eia(self) -> pd.DataFrame:
        """Load raw EIA rows once (every vintage), cached on the instance."""
        if self._raw is None:
            from data import fundamental_store as store

            raw = store.load_raw(source="eia")
            self._raw = raw[raw["instrument"].notna()].copy() if not raw.empty else raw
        return self._raw

    def compute(self, asof: date, panel: pd.DataFrame) -> pd.DataFrame:
        raw = self._eia()
        if raw is None or raw.empty:
            return self._empty_frame(panel.columns)

        asof_ts = pd.Timestamp(asof)
        # ── point-in-time: only reports released by asof, latest vintage each ──
        pit = raw[raw["release_date"] <= asof_ts]
        if pit.empty:
            return self._empty_frame(panel.columns)
        pit = (
            pit.sort_values("release_date")
            .groupby(["instrument", "reference_date"], as_index=False)
            .tail(1)
        )

        scores = {}
        for inst, g in pit.groupby("instrument"):
            s = g.sort_values("reference_date").set_index("reference_date")["value"]
            if len(s) < self.min_weeks:
                continue
            trailing = s.iloc[-self.seasonal_window :]
            cur_level = trailing.iloc[-1]

            # Seasonal norm from history STRICTLY BEFORE the current week (no
            # contamination of the residual scale by the point being scored).
            hist = trailing.iloc[:-1]
            woy_hist = hist.index.isocalendar().week.to_numpy()
            cur_week = int(trailing.index.isocalendar().week.iloc[-1])

            seas = pd.Series(hist.to_numpy(), index=woy_hist).groupby(level=0).mean()
            if cur_week not in seas.index:
                continue
            same_week_n = int((woy_hist == cur_week).sum())
            if same_week_n < self.min_same_week:
                continue

            # Residual scale = std of deseasonalised history.
            resid_hist = hist.to_numpy() - seas.reindex(woy_hist).to_numpy()
            scale = np.nanstd(resid_hist, ddof=1)
            if not np.isfinite(scale) or scale == 0:
                continue

            z = (cur_level - seas.loc[cur_week]) / scale
            scores[inst] = -float(z)  # surplus vs seasonal norm => bearish

        score = pd.Series(scores, dtype=float).dropna()
        if score.empty:
            return self._empty_frame(panel.columns)
        score.index.name = "instrument"

        confidence = score.abs().clip(upper=3.0) / 3.0

        cols = pd.MultiIndex.from_product(
            [self.horizons, [FORECAST_FIELD, CONFIDENCE_FIELD]], names=["horizon", "field"]
        )
        out = pd.DataFrame(index=score.index, columns=cols, dtype=float)
        out.index.name = "instrument"
        for h in self.horizons:
            out[(h, FORECAST_FIELD)] = score
            out[(h, CONFIDENCE_FIELD)] = confidence
        return out
