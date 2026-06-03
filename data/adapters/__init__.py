"""
data.adapters — concrete source adapters behind the PriceAdapter /
FundamentalAdapter ABCs.

Implemented today (all free):
  YFinanceAdapter  — prices (wraps the existing models.data_loader fetchers)
  FredAdapter      — FRED macro/fundamental series (wraps the existing helper)

Stubs / Phase-1 chunk-2:
  VendorCurveAdapter — paid term-structure vendor (raises NotImplementedError)
  CftcCotAdapter / EiaAdapter / UsdaAdapter — live in services/*_ingest.py
"""

from data.adapters.base import (  # noqa: F401
    OBSERVATION_COLUMNS,
    FundamentalAdapter,
    PriceAdapter,
)
from data.adapters.yfinance_adapter import YFinanceAdapter  # noqa: F401
from data.adapters.fred_adapter import FredAdapter  # noqa: F401
from data.adapters.vendor_curve import VendorCurveAdapter  # noqa: F401

__all__ = [
    "PriceAdapter",
    "FundamentalAdapter",
    "OBSERVATION_COLUMNS",
    "YFinanceAdapter",
    "FredAdapter",
    "VendorCurveAdapter",
]
