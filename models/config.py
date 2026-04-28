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
