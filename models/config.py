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

# Primary regression target — the commodity whose *next-day return* we forecast
DEFAULT_TARGET = "WTI Crude Oil"

# ── Data retrieval ─────────────────────────────────────────────────────────────
HISTORY_PERIOD   = "5y"     # how far back to pull from yfinance
HISTORY_INTERVAL = "1d"     # daily bars

# ── Feature engineering ────────────────────────────────────────────────────────
RETURN_LAGS          = [1, 2, 5]         # lagged log-return features (in days)
ROLLING_VOL_WINDOW   = 21                # rolling std dev window (trading month)
ROLLING_MOM_WINDOW   = 10               # momentum lookback (two trading weeks)
ZSCORE_WINDOW        = 63               # rolling z-score window (~quarter)
CORRELATION_WINDOW   = 21               # rolling pairwise correlation window

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

# ── Sector hyperparameter seeds ────────────────────────────────────────────────
# Domain-informed starting points used when no tuned JSON is present.
# Rationale:
#   energy      — geopolitical shocks + mean-reversion → moderate depth/lr
#   metals      — macro/DXY trend-following → deeper trees, slower learning
#   agriculture — seasonal/WASDE cycles → deeper, more trees, more regularisation
#   livestock   — smooth trends, limited sample → shallow, strong regularisation
#   digital     — extreme vol, non-stationary → fast learning, shallow trees
SECTOR_XGB_DEFAULTS: dict[str, dict] = {
    "energy": dict(
        n_estimators=600, learning_rate=0.04, max_depth=4,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=8,  reg_alpha=0.10, reg_lambda=1.0,
    ),
    "metals": dict(
        n_estimators=600, learning_rate=0.02, max_depth=5,
        subsample=0.75, colsample_bytree=0.75,
        min_child_weight=10, reg_alpha=0.05, reg_lambda=1.5,
    ),
    "agriculture": dict(
        n_estimators=700, learning_rate=0.02, max_depth=5,
        subsample=0.70, colsample_bytree=0.70,
        min_child_weight=12, reg_alpha=0.15, reg_lambda=2.0,
    ),
    "livestock": dict(
        n_estimators=400, learning_rate=0.03, max_depth=3,
        subsample=0.80, colsample_bytree=0.80,
        min_child_weight=15, reg_alpha=0.20, reg_lambda=2.0,
    ),
    "digital": dict(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=5,  reg_alpha=0.10, reg_lambda=0.5,
    ),
}

SECTOR_RF_DEFAULTS: dict[str, dict] = {
    "energy":      dict(n_estimators=300, max_depth=6,  min_samples_leaf=10, max_features="sqrt"),
    "metals":      dict(n_estimators=400, max_depth=7,  min_samples_leaf=12, max_features="sqrt"),
    "agriculture": dict(n_estimators=500, max_depth=8,  min_samples_leaf=15, max_features="sqrt"),
    "livestock":   dict(n_estimators=200, max_depth=4,  min_samples_leaf=20, max_features="sqrt"),
    "digital":     dict(n_estimators=300, max_depth=4,  min_samples_leaf=8,  max_features="sqrt"),
}
