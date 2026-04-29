"""
Commodity price data service using Yahoo Finance (yfinance).

Two types of instruments are tracked:
  1. Futures contracts  — direct commodity price (e.g. CL=F = WTI Crude futures)
  2. ETF/equity proxies — used when no free futures ticker exists; these track the
                          price of a related company or fund, NOT the raw commodity.
                          All proxies are flagged in COMMODITY_IS_PROXY and shown
                          with an asterisk (*) throughout the dashboard.

Sectors:
  Energy · Metals · Agriculture · Livestock · Digital Assets
"""

import yfinance as yf
import pandas as pd
import streamlit as st

# ── Ticker registry ────────────────────────────────────────────────────────────
# Format: display name → Yahoo Finance symbol
COMMODITY_TICKERS = {

    # ── ENERGY ─────────────────────────────────────────────────────────────────
    # Crude oil — two global benchmarks
    "WTI Crude Oil":            "CL=F",    # West Texas Intermediate (NYMEX) — US benchmark
    "Brent Crude Oil":          "BZ=F",    # North Sea Brent (ICE) — global benchmark; prices ~2/3 of world oil
    # Natural gas — US vs global
    "Natural Gas (Henry Hub)":  "NG=F",    # Henry Hub, Louisiana (NYMEX) — US benchmark
    "LNG / Intl Gas*":          "LNG",     # Cheniere Energy (proxy) — largest US LNG exporter; tracks global LNG demand *
    # Oil products
    "Gasoline (RBOB)":          "RB=F",    # Reformulated blendstock (NYMEX) — drives US pump prices
    "Heating Oil (No.2)":       "HO=F",    # No.2 fuel oil (NYMEX) — diesel proxy; northeast US seasonal
    # Coal — two distinct markets
    "Thermal Coal*":            "BTU",     # Peabody Energy (proxy) — world's largest thermal coal miner *
    "Metallurgical Coal*":      "HCC",     # Warrior Met Coal (proxy) — pure-play US met coal (for steelmaking) *
    # Nuclear / alternatives
    "Uranium*":                 "URA",     # Global X Uranium ETF (proxy) — tracks uranium miners *
    # Carbon
    "Carbon Credits*":          "KRBN",    # KraneShares Global Carbon ETF (proxy) — California + EU + RGGI allowances *

    # ── METALS — Precious ──────────────────────────────────────────────────────
    "Gold (COMEX)":             "GC=F",    # Gold futures (NYMEX/COMEX) — NY settlement
    "Gold (Physical/London)*":  "SGOL",    # Aberdeen Physical Gold ETF (proxy) — London LBMA vault-backed *
    "Silver (COMEX)":           "SI=F",    # Silver futures (NYMEX/COMEX)
    "Silver (Physical)*":       "SIVR",    # Aberdeen Physical Silver ETF (proxy) — LBMA vault-backed *
    "Platinum":                 "PL=F",    # Platinum futures (NYMEX)
    "Palladium":                "PA=F",    # Palladium futures (NYMEX) — ~85% from Russia + South Africa

    # ── METALS — Base / Industrial ─────────────────────────────────────────────
    "Copper (COMEX)":           "HG=F",    # Copper futures (NYMEX/COMEX) — "Dr. Copper", economic indicator
    "Aluminum (COMEX)":         "ALI=F",   # Aluminum futures (COMEX) — EV/aerospace/packaging input
    "HRC Steel":                "HRC=F",   # Hot-Rolled Coil steel futures (CME) — construction/manufacturing
    "Iron Ore / Steel*":        "SLX",     # VanEck Steel ETF (proxy) — steel producers are primary iron ore consumers *
    "Zinc & Cobalt*":           "GLNCY",   # Glencore ADR (proxy) — world's #1 zinc and cobalt producer *

    # ── METALS — Strategic / Battery ──────────────────────────────────────────
    "Lithium*":                 "LIT",     # Global X Lithium & Battery Tech ETF (proxy) — EV battery supply chain *
    "Rare Earths*":             "REMX",    # VanEck Rare Earth/Strategic Metals ETF (proxy) — neodymium, dysprosium etc. *

    # ── AGRICULTURE — Grains & Oilseeds ───────────────────────────────────────
    "Corn (CBOT)":              "ZC=F",    # CBOT corn futures — largest US crop; ethanol + feed
    "Wheat (CBOT SRW)":         "ZW=F",    # CBOT soft red winter wheat — primary Chicago benchmark
    "Wheat (KC HRW)":           "KE=F",    # KCBT hard red winter wheat — higher protein; bread wheat
    "Soybeans (CBOT)":          "ZS=F",    # CBOT soybean futures — China is world's largest buyer
    "Soybean Oil":              "ZL=F",    # CBOT soybean oil futures — cooking oil + biodiesel feedstock
    "Soybean Meal":             "ZM=F",    # CBOT soybean meal futures — primary global livestock protein feed
    "Oats (CBOT)":              "ZO=F",    # CBOT oats futures — smaller grain; animal feed + food
    "Rough Rice (CBOT)":        "ZR=F",    # CBOT rough rice futures — staple for ~50% of world population

    # ── AGRICULTURE — Softs ───────────────────────────────────────────────────
    "Coffee (Arabica)":         "KC=F",    # ICE Arabica coffee — higher-quality; Brazil + Colombia supply
    "Sugar (Raw #11)":          "SB=F",    # ICE raw sugar No.11 — global benchmark; Brazil swing producer
    "Cotton (No.2)":            "CT=F",    # ICE cotton No.2 futures — US, India, China dominant
    "Cocoa (ICE)":              "CC=F",    # ICE cocoa futures — 70% supply from West Africa
    "Orange Juice (FCOJ-A)":    "OJ=F",    # ICE frozen concentrated OJ — Florida weather sensitive
    "Lumber*":                  "WOOD",    # iShares Global Timber & Forestry ETF (proxy) — housing/construction input *

    # ── LIVESTOCK ─────────────────────────────────────────────────────────────
    "Live Cattle":              "LE=F",    # CME live cattle — finished cattle ready for slaughter
    "Feeder Cattle":            "GF=F",    # CME feeder cattle — ~500lb calves; leading indicator for live cattle
    "Lean Hogs":                "HE=F",    # CME lean hogs — pork export demand sensitive

    # ── DIGITAL ASSETS ────────────────────────────────────────────────────────
    "Bitcoin":                  "BTC-USD", # Bitcoin spot price (USD) — digital store-of-value / inflation hedge
}

# ── Sector mapping ─────────────────────────────────────────────────────────────
COMMODITY_SECTORS = {
    "WTI Crude Oil":            "Energy",
    "Brent Crude Oil":          "Energy",
    "Natural Gas (Henry Hub)":  "Energy",
    "LNG / Intl Gas*":          "Energy",
    "Gasoline (RBOB)":          "Energy",
    "Heating Oil (No.2)":       "Energy",
    "Thermal Coal*":            "Energy",
    "Metallurgical Coal*":      "Energy",
    "Uranium*":                 "Energy",
    "Carbon Credits*":          "Energy",
    "Gold (COMEX)":             "Metals",
    "Gold (Physical/London)*":  "Metals",
    "Silver (COMEX)":           "Metals",
    "Silver (Physical)*":       "Metals",
    "Platinum":                 "Metals",
    "Palladium":                "Metals",
    "Copper (COMEX)":           "Metals",
    "Aluminum (COMEX)":         "Metals",
    "HRC Steel":                "Metals",
    "Iron Ore / Steel*":        "Metals",
    "Zinc & Cobalt*":           "Metals",
    "Lithium*":                 "Metals",
    "Rare Earths*":             "Metals",
    "Corn (CBOT)":              "Agriculture",
    "Wheat (CBOT SRW)":         "Agriculture",
    "Wheat (KC HRW)":           "Agriculture",
    "Soybeans (CBOT)":          "Agriculture",
    "Soybean Oil":              "Agriculture",
    "Soybean Meal":             "Agriculture",
    "Oats (CBOT)":              "Agriculture",
    "Rough Rice (CBOT)":        "Agriculture",
    "Coffee (Arabica)":         "Agriculture",
    "Sugar (Raw #11)":          "Agriculture",
    "Cotton (No.2)":            "Agriculture",
    "Cocoa (ICE)":              "Agriculture",
    "Orange Juice (FCOJ-A)":    "Agriculture",
    "Lumber*":                  "Agriculture",
    "Live Cattle":              "Livestock",
    "Feeder Cattle":            "Livestock",
    "Lean Hogs":                "Livestock",
    "Bitcoin":                  "Digital Assets",
}

# ── Unit mapping ───────────────────────────────────────────────────────────────
COMMODITY_UNITS = {
    "WTI Crude Oil":            "USD/bbl",
    "Brent Crude Oil":          "USD/bbl",
    "Natural Gas (Henry Hub)":  "USD/MMBtu",
    "LNG / Intl Gas*":          "USD (ETF)",
    "Gasoline (RBOB)":          "USD/gal",
    "Heating Oil (No.2)":       "USD/gal",
    "Thermal Coal*":            "USD (equity)",
    "Metallurgical Coal*":      "USD (equity)",
    "Uranium*":                 "USD (ETF)",
    "Carbon Credits*":          "USD (ETF)",
    "Gold (COMEX)":             "USD/oz",
    "Gold (Physical/London)*":  "USD (ETF)",
    "Silver (COMEX)":           "USD/oz",
    "Silver (Physical)*":       "USD (ETF)",
    "Platinum":                 "USD/oz",
    "Palladium":                "USD/oz",
    "Copper (COMEX)":           "USD/lb",
    "Aluminum (COMEX)":         "USD/ton",
    "HRC Steel":                "USD/ton",
    "Iron Ore / Steel*":        "USD (ETF)",
    "Zinc & Cobalt*":           "USD (equity)",
    "Lithium*":                 "USD (ETF)",
    "Rare Earths*":             "USD (ETF)",
    "Corn (CBOT)":              "USc/bu",
    "Wheat (CBOT SRW)":         "USc/bu",
    "Wheat (KC HRW)":           "USc/bu",
    "Soybeans (CBOT)":          "USc/bu",
    "Soybean Oil":              "USc/lb",
    "Soybean Meal":             "USD/ton",
    "Oats (CBOT)":              "USc/bu",
    "Rough Rice (CBOT)":        "USD/cwt",
    "Coffee (Arabica)":         "USc/lb",
    "Sugar (Raw #11)":          "USc/lb",
    "Cotton (No.2)":            "USc/lb",
    "Cocoa (ICE)":              "USD/MT",
    "Orange Juice (FCOJ-A)":    "USc/lb",
    "Lumber*":                  "USD (ETF)",
    "Live Cattle":              "USc/lb",
    "Feeder Cattle":            "USc/lb",
    "Lean Hogs":                "USc/lb",
    "Bitcoin":                  "USD",
}

# ── Proxy flag & notes ─────────────────────────────────────────────────────────
# True = this is an ETF/equity proxy, not a direct commodity futures price.
# The dashboard shows an asterisk (*) and the note below for these instruments.
COMMODITY_IS_PROXY = {k: k.endswith("*") for k in COMMODITY_TICKERS}

COMMODITY_PROXY_NOTES = {
    "LNG / Intl Gas*":         "* No free global LNG futures available. Tracks Cheniere Energy (LNG) — largest US LNG exporter. Price reflects LNG market sentiment, not spot TTF/JKM rates.",
    "Thermal Coal*":           "* No free thermal coal futures available on Yahoo Finance. Tracks Peabody Energy (BTU) — world's largest thermal coal miner by volume.",
    "Metallurgical Coal*":     "* No free met coal futures available. Tracks Warrior Met Coal (HCC) — pure-play US metallurgical coal producer for steelmaking.",
    "Uranium*":                "* No free uranium spot futures available. Tracks Global X Uranium ETF (URA) — holds uranium miners including Cameco, Kazatomprom.",
    "Carbon Credits*":         "* No free carbon futures available on Yahoo Finance. Tracks KraneShares Global Carbon ETF (KRBN) — holds California (CCA), EU (EUA), and RGGI carbon allowance futures.",
    "Gold (Physical/London)*": "* Tracks Aberdeen Physical Gold Shares ETF (SGOL) — gold held in Zurich/London LBMA vaults. Reflects London fix price vs NY COMEX futures.",
    "Silver (Physical)*":      "* Tracks Aberdeen Physical Silver Shares ETF (SIVR) — silver held in London vaults. Complement to COMEX silver futures.",
    "Iron Ore / Steel*":       "* No free iron ore futures available on Yahoo Finance (SGX-traded). Tracks VanEck Steel ETF (SLX) — steel producers are the primary consumers of iron ore.",
    "Zinc & Cobalt*":          "* No free zinc or cobalt futures available (LME-traded). Tracks Glencore ADR (GLNCY) — world's largest zinc producer and largest cobalt supplier.",
    "Lithium*":                "* No free lithium futures available. Tracks Global X Lithium & Battery Tech ETF (LIT) — lithium miners and battery material producers.",
    "Rare Earths*":            "* No free rare earth futures available. Tracks VanEck Rare Earth/Strategic Metals ETF (REMX) — holds rare earth and critical mineral miners globally.",
    "Lumber*":                 "* Lumber futures (LBS=F) were delisted from Yahoo Finance. Tracks iShares Global Timber & Forestry ETF (WOOD) — timber/lumber producers.",
}

# Commodities with no viable proxy (documented for transparency)
UNTRACEABLE_COMMODITIES = {
    "Robusta Coffee":       "ICE Robusta futures (RC=F) not published on Yahoo Finance. No single-commodity ETF available. Contrast with Arabica (KC=F) which is tracked.",
    "Minneapolis Spring Wheat": "MGEX hard red spring wheat futures (MWE=F) are delisted on Yahoo Finance. The Teucrium Wheat ETF (WEAT) blends all three US benchmarks as an alternative.",
    "Nickel":               "LME nickel futures not on Yahoo Finance. iPath ETNs (JJN) are delisted. No US-traded single-commodity proxy currently available.",
    "Palm Oil":             "Bursa Malaysia Derivatives (FCPO) not on Yahoo Finance. No US-listed ETF tracks palm oil alone. Significant gap — ~35% of global vegetable oil.",
    "Natural Gas (TTF/European)": "Dutch TTF natural gas not on Yahoo Finance. BOIL (ProShares 2x Nat Gas) is US-only and leveraged — not a clean proxy.",
    "Tin":                  "LME tin futures not on Yahoo Finance. iPath Tin ETN (JJT) is delisted. No current proxy available.",
    "Lead":                 "LME lead futures not on Yahoo Finance. iPath Lead ETN (LD) is delisted. No current proxy available.",
}


# ── Fetch functions ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_current_prices() -> pd.DataFrame:
    """
    DB-first: returns current prices from PostgreSQL when data is fresh (≤3 days stale).
    Falls back to yfinance when the DB is stale or unreachable.
    Returns DataFrame: Name, Sector, Price, Change, Pct_Change, Unit, Ticker, Is_Proxy.
    """
    try:
        from services.data_contract import fetch_current_prices_db, data_freshness
        freshness = data_freshness()
        if not freshness["recommend_live"]:
            db_df = fetch_current_prices_db()
            if not db_df.empty:
                return db_df
    except Exception:
        pass

    tickers = list(COMMODITY_TICKERS.values())

    try:
        data  = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
        close = data["Close"]

        rows = []
        failed = []
        for name, ticker in COMMODITY_TICKERS.items():
            try:
                series = close[ticker].dropna()
                if len(series) >= 2:
                    price  = series.iloc[-1]
                    prev   = series.iloc[-2]
                    change = price - prev
                    pct    = (change / prev) * 100
                elif len(series) == 1:
                    price  = series.iloc[-1]
                    change = 0.0
                    pct    = 0.0
                else:
                    failed.append(name)
                    continue

                rows.append({
                    "Name":       name,
                    "Sector":     COMMODITY_SECTORS.get(name, "Other"),
                    "Price":      round(float(price), 4),
                    "Change":     round(float(change), 4),
                    "Pct_Change": round(float(pct), 2),
                    "Unit":       COMMODITY_UNITS.get(name, "USD"),
                    "Ticker":     ticker,
                    "Is_Proxy":   COMMODITY_IS_PROXY.get(name, False),
                })
            except Exception:
                failed.append(name)
                continue

        if failed:
            st.warning(
                f"⚠️ Could not retrieve prices for {len(failed)} instrument(s): "
                + ", ".join(failed[:5])
                + ("…" if len(failed) > 5 else "")
                + ". Data shown may be incomplete."
            )

        return pd.DataFrame(rows)

    except Exception as e:
        st.warning(f"Could not fetch live prices: {e}. Showing sample data.")
        return _mock_prices()


@st.cache_data(ttl=300)
def fetch_historical(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV history for a single ticker.
    period:   1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
    interval: 1m 5m 15m 30m 1h 1d 1wk 1mo
    """
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.warning(f"Could not fetch history for {ticker}: {e}")
        return pd.DataFrame()


def _mock_prices() -> pd.DataFrame:
    """Fallback mock data when the API is unavailable."""
    mock = [
        ("WTI Crude Oil",          "Energy",        78.45,  -0.32, -0.41, "USD/bbl",     "CL=F",    False),
        ("Brent Crude Oil",        "Energy",        82.10,  -0.28, -0.34, "USD/bbl",     "BZ=F",    False),
        ("Natural Gas (Henry Hub)","Energy",         2.41,  +0.05, +2.12, "USD/MMBtu",   "NG=F",    False),
        ("Gold (COMEX)",           "Metals",      2345.60, +12.40, +0.53, "USD/oz",      "GC=F",    False),
        ("Silver (COMEX)",         "Metals",        27.85,  +0.42, +1.53, "USD/oz",      "SI=F",    False),
        ("Copper (COMEX)",         "Metals",         4.32,  -0.03, -0.69, "USD/lb",      "HG=F",    False),
        ("Corn (CBOT)",            "Agriculture",  448.25,  -2.50, -0.56, "USc/bu",      "ZC=F",    False),
        ("Wheat (CBOT SRW)",       "Agriculture",  562.00,  +8.75, +1.58, "USc/bu",      "ZW=F",    False),
        ("Soybeans (CBOT)",        "Agriculture", 1148.50,  -5.25, -0.46, "USc/bu",      "ZS=F",    False),
        ("Coffee (Arabica)",       "Agriculture",  238.90,  +3.10, +1.31, "USc/lb",      "KC=F",    False),
        ("Live Cattle",            "Livestock",    182.35,  +0.85, +0.47, "USc/lb",      "LE=F",    False),
        ("Bitcoin",                "Digital Assets",45000,  +500,  +1.12, "USD",         "BTC-USD", False),
    ]
    cols = ["Name", "Sector", "Price", "Change", "Pct_Change", "Unit", "Ticker", "Is_Proxy"]
    return pd.DataFrame(mock, columns=cols)
