"""
portfolio/ — the risk/portfolio layer (seam established in Phase 0; built out in
Phase 5 of the restructure).

Responsibility: consume blended multi-horizon signal forecasts + a risk model and
emit risk-aware target weights, net of costs. It MAY import the signal layer; it
must never import streamlit, pages, or app (enforced by ``.importlinter``).

The concrete risk model, forecast→position mapping, and allocators
(mean-variance / risk-parity / QAOA) land in Phase 5. The ``Allocator`` ABC below
fixes the seam now so pages and Phase-5 work code against a stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Allocator(ABC):
    """Maps an expected-return / score vector to target portfolio weights."""

    @abstractmethod
    def allocate(self, forecasts: pd.Series, risk_model: object) -> pd.Series:
        """
        Parameters
        ----------
        forecasts : pd.Series
            index = instrument, values = blended expected return / score for one horizon.
        risk_model : object
            A covariance / volatility provider (defined in Phase 5 ``portfolio/risk.py``).

        Returns
        -------
        pd.Series
            index = instrument, values = target weight (sums consistent with the
            allocator's leverage/neutrality convention).
        """
        raise NotImplementedError


__all__ = ["Allocator"]
