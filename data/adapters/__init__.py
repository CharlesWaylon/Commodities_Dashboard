"""
data.adapters — concrete source adapters behind the PriceAdapter /
FundamentalAdapter ABCs.

Implemented today (all free):
  YFinanceAdapter  — prices (wraps the existing models.data_loader fetchers)
  FredAdapter      — FRED macro/fundamental series (wraps the existing helper)
  CftcAdapter      — CFTC COT managed-money net positioning (Socrata, no key)
  EiaAdapter       — EIA weekly petroleum / natural-gas stocks (EIA_API_KEY)
  UsdaAdapter      — USDA NASS QuickStats ag fundamentals (USDA_QUICKSTATS_KEY)

The release-dated ingestors that drive these on a schedule live in
services/{cot,eia,usda}_ingest.py.

Stubs / paid-later:
  VendorCurveAdapter — paid term-structure vendor (raises NotImplementedError)
"""

from data.adapters.base import (  # noqa: F401
    OBSERVATION_COLUMNS,
    FundamentalAdapter,
    PriceAdapter,
)
from data.adapters.yfinance_adapter import YFinanceAdapter  # noqa: F401
from data.adapters.fred_adapter import FredAdapter  # noqa: F401
from data.adapters.cftc_adapter import CftcAdapter  # noqa: F401
from data.adapters.eia_adapter import EiaAdapter  # noqa: F401
from data.adapters.usda_adapter import UsdaAdapter  # noqa: F401
from data.adapters.vendor_curve import VendorCurveAdapter  # noqa: F401

__all__ = [
    "PriceAdapter",
    "FundamentalAdapter",
    "OBSERVATION_COLUMNS",
    "YFinanceAdapter",
    "FredAdapter",
    "CftcAdapter",
    "EiaAdapter",
    "UsdaAdapter",
    "VendorCurveAdapter",
]
