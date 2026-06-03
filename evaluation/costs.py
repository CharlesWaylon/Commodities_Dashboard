"""
Transaction-cost & turnover model — used IDENTICALLY by the gate (harness) and,
in Phase 5, by the live optimizer, so a "net-of-cost" verdict and the live book
agree. This is the difference between a paper edge and a real one.

The model is intentionally simple and conservative: a per-side cost in basis
points applied to traded notional (|Δweight|). Spread + impact are folded into the
single ``cost_bps`` knob, with the option to override per instrument later (ETFs
and front-month futures are cheap; thin contracts are not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class TransactionCostModel:
    """
    Linear cost on traded notional.

    cost_bps : float
        Round-trip-agnostic *per-side* cost in basis points (1 bp = 0.01%).
        Default 10 bps is a deliberately conservative blended estimate for liquid
        commodity futures + ETFs (spread + slippage). Raise it to stress-test.
    per_instrument_bps : dict
        Optional overrides {instrument: bps}.
    """

    cost_bps: float = 10.0
    per_instrument_bps: Dict[str, float] = field(default_factory=dict)

    def _bps(self, instrument: str) -> float:
        return float(self.per_instrument_bps.get(instrument, self.cost_bps))

    def turnover(self, prev_w: pd.Series, new_w: pd.Series) -> float:
        """Total traded notional = Σ |Δweight| across the union of holdings."""
        prev_w, new_w = prev_w.align(new_w, fill_value=0.0)
        return float((new_w - prev_w).abs().sum())

    def cost(self, prev_w: pd.Series, new_w: pd.Series) -> float:
        """Cost (in return units) of moving from ``prev_w`` to ``new_w``."""
        prev_w, new_w = prev_w.align(new_w, fill_value=0.0)
        delta = (new_w - prev_w).abs()
        bps = delta.index.to_series().map(self._bps).fillna(self.cost_bps)
        return float((delta * bps / 1e4).sum())
