"""
ScenarioAggregator — consensus bear/avg/bull across heterogeneous providers.

Each provider supplies a ScenarioBand. The aggregator returns a single
ScenarioBand whose mean/bear/bull are weighted averages of the inputs at
each forecast step. Different model families can be weighted differently
(e.g. trust statistical short-term, deep long-term).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from models.scenarios.band import ScenarioBand


class ScenarioAggregator:
    """
    Combine multiple ScenarioBand objects into a consensus band.

    Parameters
    ----------
    weights : Mapping[str, float], optional
        Map from ``source_model`` (or its prefix) to weight. Unknown sources
        get weight 1.0. Weights are renormalised per call.
    """

    DEFAULT_WEIGHTS = {
        "ARIMA": 1.0,
        "VAR": 1.0,
        "Conformal": 1.0,
        "MCDropout": 1.0,
        "Quantum": 1.0,
    }

    def __init__(self, weights: Mapping[str, float] | None = None):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def _resolve_weight(self, source_model: str) -> float:
        if source_model in self.weights:
            return self.weights[source_model]
        for prefix, w in self.weights.items():
            if source_model.startswith(prefix):
                return w
        return 1.0

    def aggregate(self, bands: Iterable[ScenarioBand]) -> ScenarioBand:
        bands = [b for b in bands if b is not None]
        if not bands:
            raise ValueError("ScenarioAggregator.aggregate: no bands provided.")

        # Validate consistency
        commodity = bands[0].commodity
        horizon = bands[0].horizon
        bear_q = bands[0].bear_q
        bull_q = bands[0].bull_q
        for b in bands[1:]:
            if b.commodity != commodity:
                raise ValueError("All bands must share a commodity.")
            if b.horizon != horizon:
                raise ValueError("All bands must share a horizon.")

        weights = np.array([self._resolve_weight(b.source_model) for b in bands])
        total = weights.sum()
        if total <= 0:
            weights = np.ones_like(weights)
            total = weights.sum()
        weights = weights / total

        mean_stack = np.vstack([b.mean for b in bands])
        bear_stack = np.vstack([b.bear for b in bands])
        bull_stack = np.vstack([b.bull for b in bands])

        agg_mean = (weights[:, None] * mean_stack).sum(axis=0)
        agg_bear = (weights[:, None] * bear_stack).sum(axis=0)
        agg_bull = (weights[:, None] * bull_stack).sum(axis=0)

        return ScenarioBand(
            commodity=commodity,
            source_model="Consensus",
            asof=max(b.asof for b in bands),
            horizon=horizon,
            bear_q=bear_q,
            bull_q=bull_q,
            mean=agg_mean,
            bear=agg_bear,
            bull=agg_bull,
            diagnostics={
                "n_providers": len(bands),
                "providers": [b.source_model for b in bands],
                "weights": dict(zip([b.source_model for b in bands], weights.round(4).tolist())),
            },
        )

    @staticmethod
    def attribution_frame(bands: Iterable[ScenarioBand]) -> pd.DataFrame:
        """
        Long-format frame for plotting per-provider bear/mean/bull contributions.
        Columns: step, source_model, bear, mean, bull.
        """
        rows = []
        for b in bands:
            for step in range(b.horizon):
                rows.append({
                    "step": step + 1,
                    "source_model": b.source_model,
                    "bear": b.bear[step],
                    "mean": b.mean[step],
                    "bull": b.bull[step],
                })
        return pd.DataFrame(rows)
