"""
Forward-looking feature engineering from external data sources.

These signals differentiate this system from a generic quant library —
they capture information that price-history features cannot.

  energy_transition — uranium spread proxy, battery metals index, ETS stress
  macro_overlays    — DXY/VIX/TLT + WASDE and OPEC meeting calendar dummies
  climate_weather   — PDSI (Corn Belt drought), HDD/CDD deviation, MEI (ENSO)
  sentiment         — FinBERT per-commodity score + EIA inventory surprise
  assembler         — combines all sources into one aligned DatetimeIndex DataFrame

All modules are importable without API keys. Missing credentials return NaN
columns with a logged warning — the rest of the feature matrix is unaffected.
"""
