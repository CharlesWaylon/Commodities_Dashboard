"""
data/ — the data layer (interior, lowest seam of the architecture).

Responsibility: a clean, point-in-time, vendor-agnostic supply of prices and
fundamentals. Everything above (signals → portfolio → presentation) reads from
here through typed accessors and the source-adapter ABCs; nothing here imports
upward (signals/portfolio/evaluation) or sideways into presentation
(streamlit/pages/app) — enforced by ``.importlinter``.

Submodules
----------
- ``data.universe``          — the canonical instrument registry (one source of
                               truth the MODEL SCOPE RULE points to).
- ``data.adapters``          — PriceAdapter / FundamentalAdapter ABCs + concrete
                               adapters (the free-now / paid-later hinge).
- ``data.fundamental_store`` — point-in-time store for release-dated fundamentals
                               (COT / EIA / USDA), accessed via ``get_asof(date)``.
- ``data.config``            — data-layer feature flags.

NOTE: this directory also physically holds legacy data files (commodities.db,
pkls). Making it an importable package does not change those files.
"""
