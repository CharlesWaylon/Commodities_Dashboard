"""
VendorCurveAdapter — stub for a future PAID term-structure / curve vendor.

This is the concrete realisation of the "free now, paid later" hinge: when a paid
forward-curve feed is bought, its implementation lands here and callers that
already depend on the ``PriceAdapter`` ABC pick it up with a one-line wiring
change. Until then it raises a clear NotImplementedError so nothing silently
depends on data we don't have.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from data.adapters.base import PriceAdapter


class VendorCurveAdapter(PriceAdapter):
    source_name = "vendor_curve_stub"

    def get_prices(
        self,
        tickers: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "VendorCurveAdapter is a stub. Wire a paid forward-curve vendor here "
            "when one is procured; callers depending on the PriceAdapter ABC need "
            "no change."
        )
