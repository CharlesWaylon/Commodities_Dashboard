"""
Central configuration for predictive models.

Everything that controls what we model, how features are built,
and how models are trained lives here. Change one value and it
propagates through the whole models/ package.
"""

# ── Target commodities ─────────────────────────────────────────────────────────
# A curated subset of the full ticker registry — chosen to represent each
# sector and to have clean, liquid futures prices (no proxies).
# Keys are display names matching COMMODITY_TICKERS in services/price_data.py.

MODELING_COMMODITIES = {
    # Energy — futures
    "WTI Crude Oil":           "CL=F",
    "Brent Crude Oil":         "BZ=F",
    "Natural Gas":             "NG=F",
    "Gasoline (RBOB)":         "RB=F",
    "Heating Oil":             "HO=F",
    # Energy — ETFs / equities
    "Carbon Credits*":         "KRBN",
    "LNG / Intl Gas*":         "LNG",
    "Metallurgical Coal*":     "HCC",
    "Thermal Coal*":           "BTU",
    "Uranium*":                "URA",
    # Metals — futures
    "Gold (COMEX)":            "GC=F",
    "Silver (COMEX)":          "SI=F",
    "Copper (COMEX)":          "HG=F",
    "Platinum":                "PL=F",
    "Palladium":               "PA=F",
    "Aluminum (COMEX)":        "ALI=F",
    "HRC Steel":               "HRC=F",
    # Metals — ETFs / equities
    "Gold (Physical/London)*": "SGOL",
    "Silver (Physical)*":      "SIVR",
    "Iron Ore / Steel*":       "SLX",
    "Lithium*":                "LIT",
    "Rare Earths*":            "REMX",
    "Zinc & Cobalt*":          "GLNCY",
    # Agriculture — futures
    "Corn (CBOT)":             "ZC=F",
    "Wheat (CBOT SRW)":        "ZW=F",
    "Wheat (KC HRW)":          "KE=F",
    "Soybeans (CBOT)":         "ZS=F",
    "Soybean Meal":            "ZM=F",
    "Soybean Oil":             "ZL=F",
    "Coffee":                  "KC=F",
    "Cocoa":                   "CC=F",
    "Sugar":                   "SB=F",
    "Cotton":                  "CT=F",
    "Orange Juice (FCOJ-A)":   "OJ=F",
    "Oats (CBOT)":             "ZO=F",
    "Rough Rice (CBOT)":       "ZR=F",
    # Agriculture — ETF
    "Lumber*":                 "WOOD",
    # Livestock
    "Live Cattle":             "LE=F",
    "Feeder Cattle":           "GF=F",
    "Lean Hogs":               "HE=F",
    # Digital Assets
    "Bitcoin":                 "BTC-USD",
}

# Primary regression target — the commodity whose forward return we forecast
DEFAULT_TARGET = "WTI Crude Oil"

# ── Data retrieval ─────────────────────────────────────────────────────────────
HISTORY_PERIOD   = "5y"     # how far back to pull from yfinance
HISTORY_INTERVAL = "1d"     # daily bars

# ── Forecast horizon ──────────────────────────────────────────────────────────
# H-day cumulative log-return is the target for all supervised models.
# At H=10 (2 trading weeks) the signal-to-noise ratio is meaningfully above
# the 1-day random-walk floor and momentum/carry become detectable.
# Overlapping H-day windows create autocorrelated targets; models that split
# train from val must embargo the H rows at the boundary (see XGBoostForecaster
# and BacktestHarness._run_commodity) so early-stopping and winning-tier labels
# are not contaminated.
FORECAST_HORIZON = 10  # trading days

# ── Feature engineering ────────────────────────────────────────────────────────
# RETURN_LAGS includes 10d so the feature matrix carries a lagged return that
# matches the forecast horizon (important autocorrelation anchor for XGBoost).
RETURN_LAGS          = [1, 2, 5, 10]    # lagged log-return features (in days)
ROLLING_VOL_WINDOW   = 21               # rolling std dev window (trading month)
ROLLING_MOM_WINDOW   = 10              # short momentum lookback (two trading weeks)
ROLLING_MOM_WINDOW_LONG = 21           # medium momentum lookback (~calendar month)
ZSCORE_WINDOW        = 63              # rolling z-score window (~quarter)
CORRELATION_WINDOW   = 21              # rolling pairwise correlation window

# ── Train / test split ────────────────────────────────────────────────────────
TEST_FRACTION  = 0.20    # hold out last 20 % of dates as test set
RANDOM_SEED    = 42

# ── Quantum layer ──────────────────────────────────────────────────────────────
# Number of qubits = number of input features fed to the quantum circuit.
# Keep this small (≤ 6) for reasonable simulation speed on a laptop.
# At N qubits the state vector has 2^N amplitudes:
#   3 qubits →  8-dim Hilbert space
#   4 qubits → 16-dim Hilbert space
#   5 qubits → 32-dim Hilbert space
N_QUBITS          = 4      # we'll use 4 features per sample (see features.py)
QUANTUM_RIDGE_REG = 1e-5   # regularisation for kernel ridge regression solver

# ── Classical baseline ────────────────────────────────────────────────────────
ROLLING_MEAN_BASELINE_WINDOW = 5    # n-day rolling mean used as naive forecast

# ── Sector groupings ───────────────────────────────────────────────────────────
# Maps every display name in MODELING_COMMODITIES to one of five sectors.
# Used by sector_tuner.py (Optuna), xgboost_shap.py, and random_forest.py
# to look up sector-specific hyperparameters instead of one global default.
COMMODITY_SECTORS: dict[str, str] = {
    # Energy
    "WTI Crude Oil":           "energy",
    "Brent Crude Oil":         "energy",
    "Natural Gas":             "energy",
    "Gasoline (RBOB)":         "energy",
    "Heating Oil":             "energy",
    "Carbon Credits*":         "energy",
    "LNG / Intl Gas*":         "energy",
    "Metallurgical Coal*":     "energy",
    "Thermal Coal*":           "energy",
    "Uranium*":                "energy",
    # Metals
    "Gold (COMEX)":            "metals",
    "Silver (COMEX)":          "metals",
    "Copper (COMEX)":          "metals",
    "Platinum":                "metals",
    "Palladium":               "metals",
    "Aluminum (COMEX)":        "metals",
    "HRC Steel":               "metals",
    "Gold (Physical/London)*": "metals",
    "Silver (Physical)*":      "metals",
    "Iron Ore / Steel*":       "metals",
    "Lithium*":                "metals",
    "Rare Earths*":            "metals",
    "Zinc & Cobalt*":          "metals",
    # Agriculture
    "Corn (CBOT)":             "agriculture",
    "Wheat (CBOT SRW)":        "agriculture",
    "Wheat (KC HRW)":          "agriculture",
    "Soybeans (CBOT)":         "agriculture",
    "Soybean Meal":            "agriculture",
    "Soybean Oil":             "agriculture",
    "Coffee":                  "agriculture",
    "Cocoa":                   "agriculture",
    "Sugar":                   "agriculture",
    "Cotton":                  "agriculture",
    "Orange Juice (FCOJ-A)":   "agriculture",
    "Oats (CBOT)":             "agriculture",
    "Rough Rice (CBOT)":       "agriculture",
    "Lumber*":                 "agriculture",
    # Livestock
    "Live Cattle":             "livestock",
    "Feeder Cattle":           "livestock",
    "Lean Hogs":               "livestock",
    # Digital
    "Bitcoin":                 "digital",
}

# ── Cross-sector economic transmission priors ─────────────────────────────────
# Upstream shock propagation in sector_model._compute_upstream_adjustment used to
# rely ONLY on in-sample pairwise price correlation (corr × forecast × damping).
# That measures co-movement, not causation: Metals co-move with grains via shared
# macro drivers (USD, risk-off, China demand), which made Metals→Agriculture look
# ~3× stronger than Energy→Agriculture — the reverse of established economics.
#
# These priors are economic transmission coefficients (0–1 relative strength)
# grounded in input-cost-share data, blended with the measured correlation by
# UPSTREAM_PRIOR_STRENGTH (alpha). They are data-driven so they can be tuned
# without touching model code. Verified against external sources 2026-06-01 —
# see MODEL_VERIFICATION_LOG.md for the full justification and citations.
#
# Rationale (src → tgt):
#   energy→agriculture  0.90 — fertilizer is 35–36% of corn/wheat operating cost;
#                              natural gas is 70–80% of ammonia (N-fertilizer)
#                              cost (Haber-Bosch); fuel/irrigation. Elasticity of
#                              composite fertilizer to NG ≈ 0.86, to crude ≈ 1.0.
#   energy→metals       0.75 — mining/smelting electricity + ore-processing fuel.
#   energy→digital      0.85 — electricity is Bitcoin mining's dominant variable cost.
#   energy→livestock    0.55 — facility heating/cooling + cold-chain transport fuel.
#   metals→agriculture  0.15 — farm-equipment steel / irrigation copper: diffuse,
#                              discretionary, multi-year capital cost (weak, lagged).
#   metals→livestock    0.20 — barn/equipment construction material (capital).
#   metals→digital      0.55 — ASIC hardware metals + data-centre cooling infra.
#   agriculture→livestock 0.90 — feed-grain is the primary rearing cost (~60–70%).
#   agriculture→digital 0.25 — food-CPI → rate expectations → BTC (indirect signal).
SECTOR_TRANSMISSION_PRIORS: dict[str, dict[str, float]] = {
    "energy": {
        "metals":      0.75,
        "agriculture": 0.90,
        "livestock":   0.55,
        "digital":     0.85,
    },
    "metals": {
        "agriculture": 0.15,
        "livestock":   0.20,
        "digital":     0.55,
    },
    "agriculture": {
        "livestock":   0.90,
        "digital":     0.25,
    },
}

# Fallback coefficient for any cross-sector edge not explicitly listed above.
DEFAULT_TRANSMISSION_PRIOR: float = 0.50

# Blend weight (alpha) between neutral 1.0 and the economic prior:
#   effective = (1 - alpha) + alpha * prior
# alpha = 0.0  → legacy behavior (pure measured correlation; restores old output).
# alpha = 1.0  → full economic prior (recommended for presentation-grade output).
# Tune downward if backtest IC degrades after enabling.
UPSTREAM_PRIOR_STRENGTH: float = 1.0

# ── Sector hyperparameter seeds ────────────────────────────────────────────────
# Domain-informed starting points used when no tuned JSON is present.
# Rationale:
#   energy      — geopolitical shocks + mean-reversion → moderate depth/lr
#   metals      — macro/DXY trend-following → deeper trees, slower learning
#   agriculture — seasonal/WASDE cycles → deeper, more trees, more regularisation
#   livestock   — smooth trends, limited sample → shallow, strong regularisation
#   digital     — extreme vol, non-stationary → fast learning, shallow trees
SECTOR_XGB_DEFAULTS: dict[str, dict] = {
    # n_estimators are ceilings — early stopping typically fires at 60-120 trees.
    # max_depth capped at 4: at depth-4 each tree has ≤16 leaves; 200 trees →
    # ≤3,200 total leaves against ~4,300 fit rows (leaves ≪ rows).
    "energy": dict(
        n_estimators=200, learning_rate=0.04, max_depth=4,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=15, reg_alpha=0.10, reg_lambda=2.0,
    ),
    "metals": dict(
        n_estimators=200, learning_rate=0.02, max_depth=4,
        subsample=0.75, colsample_bytree=0.75,
        min_child_weight=15, reg_alpha=0.05, reg_lambda=2.5,
    ),
    "agriculture": dict(
        n_estimators=250, learning_rate=0.02, max_depth=4,
        subsample=0.70, colsample_bytree=0.70,
        min_child_weight=20, reg_alpha=0.15, reg_lambda=3.0,
    ),
    "livestock": dict(
        n_estimators=150, learning_rate=0.03, max_depth=3,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=25, reg_alpha=0.20, reg_lambda=3.0,
    ),
    "digital": dict(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=15, reg_alpha=0.10, reg_lambda=2.0,
    ),
}

SECTOR_RF_DEFAULTS: dict[str, dict] = {
    "energy":      dict(n_estimators=300, max_depth=6,  min_samples_leaf=10, max_features="sqrt"),
    "metals":      dict(n_estimators=400, max_depth=7,  min_samples_leaf=12, max_features="sqrt"),
    "agriculture": dict(n_estimators=500, max_depth=8,  min_samples_leaf=15, max_features="sqrt"),
    "livestock":   dict(n_estimators=200, max_depth=4,  min_samples_leaf=20, max_features="sqrt"),
    "digital":     dict(n_estimators=300, max_depth=4,  min_samples_leaf=8,  max_features="sqrt"),
}


# ── Portfolio risk gates (Step 6 of macro_features spec) ──────────────────────
# Post-QAOA weight adjustments applied when a trigger of the specified family
# fires at or above ``min_strength``. Edit these to tune behavior without
# touching the optimizer code.
#
# action types:
#   "sector_cap"            — cap the sector's total weight at
#                             cap_multiplier × (1/k) (1/k = equal-weight share).
#   "flatten_toward_equal"  — blend `blend` fraction of weight toward equal
#                             weight to force diversification.
#   "turnover_damper"       — blend `damp` fraction of yesterday's portfolio
#                             into today's weights. Skipped if previous_weights
#                             is unavailable.
#
# The special key "__any_strong__" matches any trigger whose strength is at
# or above its own ``min_strength``, regardless of family.
TRIGGER_RISK_GATES: dict = {
    "fed_tightening": {
        "min_strength": 0.7,
        "action":       "flatten_toward_equal",
        "params":       {"blend": 0.20},
        "description":  "Force diversification under rate-shock (blend 20% toward equal weight).",
    },
    "weather_shock": {
        "min_strength": 0.7,
        "action":       "sector_cap",
        "params":       {"sector": "agriculture", "cap_multiplier": 1.5},
        "description":  "Cap Agriculture allocation at 1.5× equal weight.",
    },
    "opec_action": {
        "min_strength": 0.7,
        "action":       "sector_cap",
        "params":       {"sector": "energy", "cap_multiplier": 1.5},
        "description":  "Cap Energy allocation at 1.5× equal weight.",
    },
    "__any_strong__": {
        "min_strength": 0.9,
        "action":       "turnover_damper",
        "params":       {"damp": 0.30},
        "description":  "Damp 30% toward yesterday's portfolio for any trigger at strength ≥ 0.9.",
    },
}

