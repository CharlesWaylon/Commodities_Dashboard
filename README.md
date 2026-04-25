# 📈 Commodities Dashboard

An interactive, real-time market dashboard for the **Future of Commodities Club**.  
Built with Python, Streamlit, and Plotly — live data, no paid APIs required.

---

## What It Does

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Market snapshot — price cards grouped by sector, top movers, quick table |
| 💰 **Pricing** | Full sortable/filterable table + performance treemap heatmap |
| 📊 **Charts** | Interactive historical charts (line, candlestick, area) with SMAs & volume |
| 📰 **News** | Live commodity news aggregated from free RSS feeds (Reuters, CNBC, FT, etc.) |
| 🤖 **Models** | 5-tab predictive dashboard: statistical, ML, deep learning, quantum, and market signals |
| 🗄️ **Database** | Inspect stored data, trigger manual ingestion, view coverage per commodity |

---

## Prerequisites

- Python 3.10 or higher
- `pip` (comes with Python)
- An internet connection (for live data)
- [Postgres.app](https://postgresapp.com/) (for the local database — free, Mac)

**No API keys required.** All data comes from Yahoo Finance (free) and public RSS feeds (free).

---

## Setup

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd Commodities_Dashboard
```

### 2. Create a virtual environment (recommended)

A virtual environment keeps this project's dependencies isolated from other Python projects.

```bash
# Create the environment
python -m venv venv

# Activate it (Mac/Linux):
source venv/bin/activate

# Activate it (Windows):
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

**Core dashboard**
- `streamlit` — web framework that turns Python scripts into interactive dashboards
- `plotly` — interactive charting library
- `yfinance` — fetches market data from Yahoo Finance
- `feedparser` — parses RSS news feeds
- `pandas` / `numpy` — data manipulation
- `python-dotenv` — loads `.env` config files
- `apscheduler` — runs the ingestion pipeline on a schedule
- `sqlalchemy` — ORM that talks to SQLite or PostgreSQL
- `psycopg2-binary` — PostgreSQL driver

**Tier 1 — Statistical models**
- `statsmodels` — ARIMA/SARIMA and VAR/VECM
- `arch` — GARCH and GJR-GARCH volatility models

**Tier 2 — ML models**
- `scikit-learn` — Random Forest, ElasticNet, StandardScaler
- `hmmlearn` — Gaussian Hidden Markov Model
- `xgboost` — gradient-boosted trees
- `shap` — SHAP TreeExplainer for model explainability

**Tier 3 — Deep learning (optional, heavier)**
- `torch` — PyTorch for BiLSTM
- `prophet` — Facebook/Meta Prophet decomposition
- `pytorch-forecasting` + `lightning` — Temporal Fusion Transformer

**Tier 4 — Quantum (optional, experimental)**
- `pennylane` — quantum circuit simulation
- `pennylane-qiskit` — IBM Quantum hardware backend

**Feature engineering**
- `transformers` — FinBERT NLP sentiment (optional, GPU recommended)

### 4. Set up PostgreSQL

The pipeline stores price history in a local PostgreSQL database.

```bash
# 1. Install Postgres.app from https://postgresapp.com/ and start it

# 2. Create the database
createdb commodities

# 3. Create a .env file in the project root
echo "DATABASE_URL=postgresql://localhost/commodities" > .env
```

### 5. Run the ingestion pipeline (first time)

Populate the database with 5 years of price history for all 41 instruments:

```bash
python -m pipeline.ingest --backfill
```

This takes ~30 seconds and writes ~52,000 rows into PostgreSQL.

### 6. Run the roll adjustment

Clean the price series by removing futures contract-roll artifacts:

```bash
python -m pipeline.roll_adjust
```

This populates `adjusted_close` and `adjustment_factor` for every row. Always run this after a backfill.

### 7. Build the aligned calendar

Time-align all 41 instruments to a single canonical US trading calendar:

```bash
python -m pipeline.align_calendar
```

This creates the `aligned_prices` table — every instrument on exactly the same 1,258-day date index, with gaps forward-filled. Run this after `roll_adjust`. Required for any cross-asset analysis (correlation matrices, model training).

### 8. Start the automatic daily scheduler (optional)

Leave this running in a terminal to keep the database updated:

```bash
python -m pipeline.scheduler
```

It runs an update immediately on start, then every weekday at 18:05 UTC automatically.

### 9. Run the dashboard

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).
Open that in your browser — the dashboard loads automatically.

---

## Project Structure

```
Commodities_Dashboard/
│
├── app.py                    ← Main entry point / Overview page
│
├── pages/                    ← Streamlit auto-discovers these as sidebar pages
│   ├── 1_Pricing.py          ← Full price table + treemap
│   ├── 2_Charts.py           ← Historical charts + comparison
│   ├── 3_News.py             ← RSS news aggregator
│   ├── 4_Models.py           ← 5-tab predictive model dashboard (see Models section below)
│   └── 5_Database.py         ← DB inspector + manual ingestion trigger
│
├── models/                   ← Predictive model library
│   ├── features.py           ← Core feature matrix builder (48 features)
│   ├── config.py             ← Shared constants (commodities list, train/test split)
│   ├── statistical/          ← Tier 1: classical econometric models
│   │   ├── arima.py          ← ARIMA/SARIMA price forecasting (AIC grid search)
│   │   ├── garch.py          ← GARCH/GJR-GARCH volatility forecasting
│   │   ├── kalman.py         ← Kalman Filter dynamic hedge ratio
│   │   └── var_vecm.py       ← VAR/VECM Granger causality + impulse response
│   ├── ml/                   ← Tier 2: machine learning models
│   │   ├── hmm_regime.py     ← Gaussian HMM 4-state regime detector
│   │   ├── random_forest.py  ← Random Forest with rolling re-fit
│   │   ├── xgboost_shap.py   ← XGBoost + SHAP waterfall explainability
│   │   └── elastic_net.py    ← Elastic Net / Lasso sparse factor model
│   ├── deep/                 ← Tier 3: deep learning (requires torch/prophet)
│   │   ├── lstm.py           ← Bidirectional LSTM multi-commodity forecaster
│   │   ├── prophet_decomp.py ← Meta-Prophet trend + seasonality decomposition
│   │   └── tft.py            ← Temporal Fusion Transformer (multi-horizon quantile)
│   └── quantum/              ← Tier 4: quantum models (requires pennylane)
│       ├── kernel_benchmark.py ← Quantum kernel SVM vs classical baseline
│       ├── qaoa_portfolio.py   ← QAOA cardinality-constrained portfolio optimiser
│       └── qnn_hybrid.py       ← QNN hybrid layer vs classical network comparison
│
├── features/                 ← Forward-looking external signal library
│   ├── assembler.py          ← Single entry point: builds the augmented feature matrix
│   ├── energy_transition.py  ← Uranium spread, battery metals PC1, ETS stress
│   ├── macro_overlays.py     ← DXY/VIX/TLT + WASDE/OPEC+ calendar dummies
│   ├── climate_weather.py    ← ENSO MEI v2, PDSI Corn Belt, HDD/CDD deviations
│   └── sentiment.py          ← FinBERT NLP sentiment + EIA inventory surprise
│
├── database/                 ← Database layer
│   ├── models.py             ← SQLAlchemy ORM table definitions (schema)
│   └── db.py                 ← Engine, session factory, init_db(), db_info()
│
├── pipeline/                 ← Data pipeline scripts
│   ├── ingest.py             ← Fetch yfinance → write to DB (seed + upsert)
│   ├── roll_adjust.py        ← Proportional backward roll adjustment for futures
│   ├── align_calendar.py     ← Unify all instruments onto one trading calendar
│   ├── audit.py              ← Data quality audit (gaps, spikes, coverage)
│   ├── migrate_to_postgres.py← One-time SQLite → PostgreSQL migration
│   └── scheduler.py          ← APScheduler: runs ingest daily at 18:05 UTC
│
├── data/
│   └── commodities.db        ← SQLite fallback (if no DATABASE_URL set)
│
├── reports/                  ← Audit outputs (auto-created)
│   └── audit_YYYYMMDD.csv    ← Machine-readable audit results by date
│
├── services/                 ← Live data layer: fetching & caching
│   ├── price_data.py         ← Yahoo Finance via yfinance
│   └── news_data.py          ← RSS feed aggregation via feedparser
│
├── utils/
│   └── formatting.py         ← Shared display helpers (colors, emoji, number formats)
│
├── .streamlit/
│   └── config.toml           ← Theme config (dark mode, colors, fonts)
│
├── .env                      ← DATABASE_URL (not committed to git)
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## How the Data Flows

```
Yahoo Finance (free)
       │
       ▼
  yfinance library
       │
       ├──► pipeline/ingest.py ──► price_history  (raw OHLCV, one row per instrument per day)
       │                                │
       │                                ▼
       │                         pipeline/roll_adjust.py
       │                                │  adjusts futures for contract-roll artifacts
       │                                ▼
       │                         price_history.adjusted_close  ← use for single-instrument analysis
       │                                │
       │                                ▼
       │                         pipeline/align_calendar.py
       │                                │  reindexes all 41 instruments to one shared calendar
       │                                ▼
       │                         aligned_prices  ← use for cross-asset analysis & model training
       │
       └──► services/price_data.py  ←── @st.cache_data(ttl=300)  [5 min cache]
                   │
                   ▼
             Streamlit pages  →  Plotly charts / DataFrames / Metric cards


Public RSS Feeds (Reuters, CNBC, FT, etc.)
       │
       ▼
  feedparser library
       │
       ▼
services/news_data.py   ←── @st.cache_data(ttl=600)  [10 min cache]
       │
       ▼
  pages/3_News.py  →  Article cards with links
```

---

## Database Summary

The PostgreSQL database (`commodities`) is populated by the ingestion pipeline and contains:

| Stat | Value |
|------|-------|
| **Total commodities tracked** | 41 (28 direct futures + 13 ETF/equity proxies) |
| **price_history rows** | 52,116 (raw OHLCV + roll-adjusted prices) |
| **aligned_prices rows** | 51,577 (all instruments on a single 1,258-day calendar) |
| **Date range** | 2021-04-23 → present (updated daily on weekdays) |
| **Database engine** | PostgreSQL (via Postgres.app) |
| **Fallback** | SQLite (`data/commodities.db`) if no `DATABASE_URL` is set |
| **Update frequency** | Daily at 18:05 UTC via scheduler, or on-demand |

`price_history` stores: `date`, `open`, `high`, `low`, `close`, `volume`, `interval`, `ingested_at`, `adjusted_close`, `adjustment_factor`.  
`aligned_prices` stores: `date`, `adjusted_close`, `is_filled` (True = forward-filled, no real observation that day).

---

## Database Schema

Three tables in a layered design — each layer is cleaner and more analysis-ready than the one below it:

```
commodities                      price_history                    aligned_prices
───────────────────              ─────────────────────────        ──────────────────────
id  (primary key)   ◄─────────── commodity_id  (FK)    ◄──────── commodity_id  (FK)
name                      1:N    date                      1:1    date  (canonical only)
ticker                           open / high / low / close        adjusted_close  (ffilled)
sector                           volume                           is_filled
unit                             interval  (1d or 1wk)
                                 ingested_at
                                 adjusted_close
                                 adjustment_factor
```

| Table | When to use it |
|-------|---------------|
| `commodities` | Looking up a ticker, sector, or unit for an instrument |
| `price_history` | Single-instrument charts, auditing raw data, rolling statistics |
| `aligned_prices` | Correlation matrices, model training, any calculation involving two or more instruments simultaneously |

`adjusted_close` is the primary price to use for any return calculation, chart, or model.  
`close` holds the raw price exactly as Yahoo Finance returned it.  
`adjustment_factor = 1.0` means no adjustment was applied (ETF proxies, Bitcoin, most recent contract).  
`is_filled = True` in `aligned_prices` means that day had no real observation — the prior day's price was carried forward.

---

## Data Quality Pipeline

Before the database was used for analysis, three quality passes were run.

### Pass 1 — Audit (`pipeline/audit.py`)

Scans every instrument across six dimensions and saves results to `reports/audit_YYYYMMDD.csv`:

| Check | Method | Threshold |
|-------|--------|-----------|
| Coverage % | Rows present vs global trading calendar | Flag < 98% |
| Date gaps | Calendar days between consecutive records | Flag > 4 days |
| Zero volume | Days with zero or null volume | Count only |
| Stale prices | Consecutive days with the exact same close | Flag runs ≥ 3 |
| Price spikes | Z-score on log returns: `\|(lr − μ) / σ\|` | Flag > 4σ |
| Price = 0 | Physically impossible entries | Count only |

Z-scores are used instead of a fixed percentage threshold so that high-volatility instruments (Natural Gas) don't swamp low-volatility ones (Gold) — every instrument is measured against its own historical range.

```bash
python -m pipeline.audit
```

### Pass 2 — Roll adjustment (`pipeline/roll_adjust.py`)

Commodity futures expire every 1–3 months. When Yahoo Finance rolls from an expiring contract to the next front-month contract, it creates an **artificial price jump** in the raw series that looks like a real market move but isn't. This corrupts every downstream return calculation.

**Example — WTI Crude Oil, March 2022:**
```
2022-03-08  raw close  =  $119.40  (March contract — last day)
2022-03-09  raw close  =  $106.02  (April contract — first day)
Raw daily return  =  −11.2%  ← this never happened; oil didn't crash overnight
Adjusted return   =     0.0%  ← roll artifact removed
```

The fix is **proportional backward adjustment**:
1. Compute log returns and z-score them per instrument
2. Any day where `|z| > 5.0σ` is flagged as a roll artifact (6.5σ for Natural Gas, which has legitimately large real moves)
3. Working backward from the most recent date, multiply all prior prices by `close_new / close_old` at each roll point — making the series continuous at that date while preserving all genuine intra-contract moves

**Results across the full database:**

| Category | Count |
|----------|-------|
| Futures instruments adjusted | 28 |
| ETF/crypto pass-through (no adjustment needed) | 13 |
| Roll events detected and removed | 95 |
| Rows with `adjusted_close` populated | 52,116 |

Two corrupt Rough Rice rows were also identified (prices of ~$17 and ~$11 on days when surrounding prices were ~$1,800 — Yahoo Finance had reported in the wrong unit) and deleted before adjustment.

```bash
python -m pipeline.roll_adjust
```

### Pass 3 — Calendar alignment (`pipeline/align_calendar.py`)

Even with clean, roll-adjusted prices, the 41 instruments are still on **different time axes**:

- Bitcoin trades 7 days/week → ~1,827 rows over 5 years
- Futures trade ~252 US exchange days/year → ~1,260 rows
- ETF proxies follow US equity hours → close on a few days futures are open

This means you cannot directly compare row 500 of Bitcoin with row 500 of WTI — they represent different calendar dates. Building a correlation matrix or multi-instrument model from misaligned series produces statistically invalid results.

The fix is a **canonical US trading calendar** derived empirically from the futures data itself: any date where at least 50% of the 28 direct futures contracts have a price record is declared a valid trading day. This naturally excludes weekends, US public holidays, and exchange closures without requiring any external calendar library.

Every instrument is then **reindexed** to this 1,258-day calendar. Days with no observation are **forward-filled** — the last known price is carried forward, which is the standard financial convention (a market that didn't trade today is still worth what it was worth yesterday).

**Results:**

| Metric | Value |
|--------|-------|
| Canonical trading days | 1,258 |
| Aligned rows written | 51,577 |
| Max forward-fill rate (any instrument) | 0.2% |
| Instruments with zero fills | 18 of 41 |

The forward-fill rate of 0.2% (at most 2 synthetic rows per instrument) confirms the underlying data is high quality. ETF proxies account for the 2-row fills — they close on 2 days per year when futures markets remain open.

```bash
python -m pipeline.align_calendar
```

---

## Commodities in the Database

> **Key:** Direct futures/spot prices have no marker. ETF or equity proxies are marked with `*` and a note explaining what they track and why no free direct price exists.

### ⛽ Energy (10 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| WTI Crude Oil | `CL=F` | USD/bbl | 4 | NYMEX futures | West Texas Intermediate — the US benchmark crude. Most-cited oil price in North America. |
| Brent Crude Oil | `BZ=F` | USD/bbl | 5 | ICE futures | North Sea Brent — the global benchmark. Prices ~2/3 of world oil. Typically $2–5 above WTI. |
| Natural Gas (Henry Hub) | `NG=F` | USD/MMBtu | 2 | NYMEX futures | Henry Hub, Louisiana — US benchmark. Highly seasonal; spikes on winter heating demand. |
| LNG / Intl Gas `*` | `LNG` | USD (equity) | — | NYSE stock | **Proxy:** Cheniere Energy — largest US LNG exporter. Reflects global LNG market but not TTF/JKM spot rates directly. |
| Gasoline (RBOB) | `RB=F` | USD/gal | 4 | NYMEX futures | Reformulated blendstock — the wholesale gasoline contract. Drives US pump prices. |
| Heating Oil (No.2) | `HO=F` | USD/gal | 4 | NYMEX futures | No.2 fuel oil — also a diesel price proxy. Strongly seasonal (northeast US winter). |
| Thermal Coal `*` | `BTU` | USD (equity) | — | NYSE stock | **Proxy:** Peabody Energy — world's largest thermal coal miner. No free coal futures on Yahoo Finance. |
| Metallurgical Coal `*` | `HCC` | USD (equity) | — | NYSE stock | **Proxy:** Warrior Met Coal — pure-play US met coal for steelmaking. Distinct market from thermal coal. |
| Uranium `*` | `URA` | USD (ETF) | — | NYSE ETF | **Proxy:** Global X Uranium ETF — holds miners incl. Cameco, Kazatomprom. No free uranium spot futures. |
| Carbon Credits `*` | `KRBN` | USD (ETF) | — | NYSE ETF | **Proxy:** KraneShares Global Carbon ETF — tracks California (CCA), EU (EUA), and RGGI carbon allowances. |

### ⚙️ Metals — Precious (6 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Gold (COMEX) | `GC=F` | USD/oz | 4 | COMEX futures | NY futures settlement. Classic safe-haven; moves inversely to USD and real rates. |
| Gold (Physical/London) `*` | `SGOL` | USD (ETF) | — | NYSE ETF | **Proxy:** Aberdeen Physical Gold ETF — LBMA vault-backed (London/Zurich). Reflects London fix vs NY COMEX. |
| Silver (COMEX) | `SI=F` | USD/oz | 2 | COMEX futures | NY futures. Precious + industrial (solar panels, electronics). More volatile than gold. |
| Silver (Physical) `*` | `SIVR` | USD (ETF) | — | NYSE ETF | **Proxy:** Aberdeen Physical Silver ETF — LBMA vault-backed. London market complement to COMEX. |
| Platinum | `PL=F` | USD/oz | 3 | NYMEX futures | Rarer than gold; mainly catalytic converters and fuel cells. Supply concentrated in South Africa. |
| Palladium | `PA=F` | USD/oz | 4 | NYMEX futures | ~85% from Russia + South Africa. Almost exclusively used in gasoline catalytic converters. |

### ⚙️ Metals — Base & Industrial (5 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Copper (COMEX) | `HG=F` | USD/lb | 3 | COMEX futures | "Dr. Copper" — leading economic indicator. Critical for EVs, construction, infrastructure. |
| Aluminum (COMEX) | `ALI=F` | USD/ton | 0 | COMEX futures | Key input for EVs, aerospace, packaging. Most energy-intensive common metal to produce. |
| HRC Steel | `HRC=F` | USD/ton | 14 | CME futures | Hot-rolled coil steel — monthly contracts with large carry. 14 rolls in 5 years is normal for this market. |
| Iron Ore / Steel `*` | `SLX` | USD (ETF) | — | NYSE ETF | **Proxy:** VanEck Steel ETF — holds steel producers who are the primary iron ore consumers. |
| Zinc & Cobalt `*` | `GLNCY` | USD (equity) | — | OTC ADR | **Proxy:** Glencore ADR — world's #1 zinc producer and #1 cobalt supplier. |

### ⚙️ Metals — Strategic / Battery (2 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Lithium `*` | `LIT` | USD (ETF) | — | NYSE ETF | **Proxy:** Global X Lithium & Battery Tech ETF — lithium miners and battery material producers. Critical for EV transition. |
| Rare Earths `*` | `REMX` | USD (ETF) | — | NYSE ETF | **Proxy:** VanEck Rare Earth/Strategic Metals ETF — neodymium, dysprosium, and other critical minerals for clean energy/defense. |

### 🌾 Agriculture — Grains & Oilseeds (8 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Corn (CBOT) | `ZC=F` | USc/bu | 3 | CBOT futures | Largest US crop. Used for ethanol, animal feed, and food. Major weather-driven volatility. |
| Wheat (CBOT SRW) | `ZW=F` | USc/bu | 3 | CBOT futures | Soft red winter wheat — Chicago benchmark. Highly geopolitical (Ukraine/Russia supply risk). |
| Wheat (KC HRW) | `KE=F` | USc/bu | 0 | KCBT futures | Hard red winter wheat — higher protein, used for bread flour. Kansas/Oklahoma growing region. |
| Soybeans (CBOT) | `ZS=F` | USc/bu | 3 | CBOT futures | World's primary protein/oil oilseed. China is largest buyer; central to US-China trade tension. |
| Soybean Oil | `ZL=F` | USc/lb | 0 | CBOT futures | Crushed from soybeans; used in cooking oil and biodiesel. Part of the "soy complex" with ZS=F + ZM=F. |
| Soybean Meal | `ZM=F` | USD/ton | 5 | CBOT futures | Protein byproduct of crushing soybeans. Primary global livestock protein feed. |
| Oats (CBOT) | `ZO=F` | USc/bu | 3 | CBOT futures | Smaller grain market; important for animal feed, oat milk, and oatmeal food products. |
| Rough Rice (CBOT) | `ZR=F` | USD/cwt | 3 | CBOT futures | Staple food for ~50% of the world's population. Supply concentrated in Asia. Two corrupt Yahoo Finance rows (impossible prices) were removed. |

### 🌾 Agriculture — Softs (6 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Coffee (Arabica) | `KC=F` | USc/lb | 0 | ICE futures | Higher-quality Arabica (Brazil, Colombia, Ethiopia). Frost + drought sensitive. |
| Sugar (Raw #11) | `SB=F` | USc/lb | 0 | ICE futures | Raw cane sugar — global benchmark. Brazil can swing production between sugar and ethanol. |
| Cotton (No.2) | `CT=F` | USc/lb | 1 | ICE futures | Primary textile fiber. US, India, and China dominate. Sensitive to drought and trade policy. |
| Cocoa (ICE) | `CC=F` | USD/MT | 3 | ICE futures | ~70% supply from West Africa (Ivory Coast, Ghana). Subject to crop disease and political risk. |
| Orange Juice (FCOJ-A) | `OJ=F` | USc/lb | 0 | ICE futures | Frozen concentrated OJ. Niche but volatile — highly sensitive to Florida freeze and hurricane risk. |
| Lumber `*` | `WOOD` | USD (ETF) | — | NYSE ETF | **Proxy:** iShares Global Timber & Forestry ETF. LBS=F futures were delisted from Yahoo Finance. |

### 🐄 Livestock (3 commodities)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Live Cattle | `LE=F` | USc/lb | 6 | CME futures | Finished cattle ready for slaughter. Key input for US beef prices. |
| Feeder Cattle | `GF=F` | USc/lb | 4 | CME futures | ~500lb calves entering feedlots. Leading indicator for live cattle prices 4–6 months ahead. |
| Lean Hogs | `HE=F` | USc/lb | 12 | CME futures | Market-weight hogs. Highly seasonal; sensitive to pork export demand (esp. China). |

### ₿ Digital Assets (1 commodity)

| Commodity | Ticker | Unit | Rolls Removed | Exchange / Type | What it represents |
|-----------|--------|------|---------------|-----------------|-------------------|
| Bitcoin | `BTC-USD` | USD | — | Spot (Yahoo) | Bitcoin spot price in USD. Tracked as a digital store-of-value alongside traditional commodities. 7-day market (hence more rows than futures). |

---

## Commodities NOT Trackable (No Working Free Proxy)

These gaps exist in the dashboard. No free Yahoo Finance ticker or viable US-listed ETF is currently available.

| Commodity | Why it matters | Why we can't track it | Paid alternative |
|-----------|---------------|----------------------|-----------------|
| Robusta Coffee | London-traded (LIFFE). Lower quality than Arabica but ~40% of world supply. Key for instant coffee. | ICE Robusta (RC=F) not published on Yahoo; no single-commodity US ETF. | ICE Data Services |
| Minneapolis Spring Wheat | Highest-protein US wheat (MGEX). Key for premium bread and pasta flour. | MWE=F delisted on Yahoo Finance. Teucrium WEAT ETF blends all 3 US benchmarks as a partial proxy. | CME DataMine |
| Nickel | LME-traded. Critical for stainless steel and EV battery cathodes. Major LME scandal (2022 short squeeze). | iPath Nickel ETN (JJN) delisted. No current US-listed single-commodity nickel instrument. | LME data (paid) |
| Palm Oil | Bursa Malaysia Derivatives (FCPO). ~35% of global vegetable oil. Largest food oil by volume. | No US-listed ETF. PALM ETF delisted. Significant coverage gap for food commodities analysis. | Bursa Malaysia (paid) |
| European Natural Gas (TTF) | Dutch Title Transfer Facility — the European gas benchmark. Hugely important post-Ukraine energy crisis. | BOIL is a 2× leveraged US nat gas ETF — not a clean TTF proxy. TTE (TotalEnergies) is too broad. | ICE/Platts (paid) |
| Tin | LME-traded. Critical for electronics soldering; concentrated supply in Indonesia/Myanmar. | iPath Tin ETN (JJT) delisted. No current US proxy. | LME data (paid) |
| Lead | LME-traded. Used in lead-acid batteries (still dominant in vehicles). | iPath Lead ETN (LD) delisted. No current US proxy. | LME data (paid) |

---

## How to Add a Commodity

Edit [`services/price_data.py`](services/price_data.py) — add one entry to each dictionary:

```python
# 1. Ticker (append * to name if it's a proxy, not a direct futures price)
COMMODITY_TICKERS["My Commodity"]   = "TICKER=F"

# 2. Sector
COMMODITY_SECTORS["My Commodity"]   = "Energy"   # Energy / Metals / Agriculture / Livestock / Digital Assets

# 3. Unit
COMMODITY_UNITS["My Commodity"]     = "USD/unit"

# 4. Only if it's a proxy — add a note explaining what it tracks
COMMODITY_PROXY_NOTES["My Commodity*"] = "* Free futures not available. Tracks XYZ ETF which holds..."
```

Then run the full pipeline to load, clean, and align the new instrument:

```bash
python -m pipeline.ingest --backfill
python -m pipeline.roll_adjust
python -m pipeline.align_calendar
```

The dashboard picks it up automatically — no other changes needed.

---

## Pipeline Commands Reference

Run these in order on first setup. After that, only `ingest` + `roll_adjust` + `align_calendar` need to be re-run as new data arrives.

| Command | What it does | Run when |
|---------|-------------|----------|
| `python -m pipeline.ingest --backfill` | Pull 5-year history for all 41 instruments | First setup, or adding a new instrument |
| `python -m pipeline.ingest` | Incremental update (last 5 days only) | Daily (handled automatically by scheduler) |
| `python -m pipeline.roll_adjust` | Detect roll artifacts and write `adjusted_close` | After any backfill |
| `python -m pipeline.align_calendar` | Rebuild `aligned_prices` on the canonical calendar | After `roll_adjust` |
| `python -m pipeline.audit` | Data quality audit — gaps, spikes, coverage | On demand; saves CSV to `reports/` |
| `python -m pipeline.scheduler` | Start daily auto-ingestion (18:05 UTC weekdays) | Keep running in background |
| `python -m pipeline.migrate_to_postgres` | One-time migration from SQLite to PostgreSQL | One time only |

---

## Customisation

### Change the theme
Edit `.streamlit/config.toml`:
```toml
primaryColor = "#F5A623"              # Accent color (currently orange/gold)
backgroundColor = "#0E1117"           # Main background
secondaryBackgroundColor = "#1C2333"  # Sidebar / card background
```

### Add news sources
Edit `services/news_data.py` → `RSS_FEEDS` dict. Any publication that publishes an RSS feed works.

---

## Predictive Models

The Models page (`pages/4_Models.py`) runs live model inference on real market data across five tabs. All models load from the `models/` and `features/` packages and use `@st.cache_data` / `@st.cache_resource` to avoid re-running on every interaction.

**Primary accuracy metric throughout: Spearman IC (Information Coefficient).** IC > 0.05 is considered actionable for a daily commodity signal.

---

### Tab 1 — Statistical Models

#### ARIMA / SARIMA
**What it does:** Forecasts the price level 1–30 days ahead. The model is described by three parameters — AR (how many past prices to use), I (how many times to difference the series to make it stationary), and MA (how many past forecast errors to correct for). The dashboard grid-searches ~64 combinations and selects the order with the lowest AIC score. For seasonal commodities (Natural Gas, agricultural grains), a SARIMA seasonal extension adds a 5-period seasonal cycle.

**Dashboard output:** Price chart with 95% confidence interval bands, winning model order (e.g., ARIMA(3,1,3)), AIC score, Day-1 and Day-N forecast prices.

**Key insight:** A tight confidence interval signals the model is confident. A wide CI signals high uncertainty — both are informative. The ARIMA forecast is best interpreted as a mean-reversion baseline, not an absolute price call.

---

#### GARCH / GJR-GARCH
**What it does:** Forecasts *volatility*, not price direction. GARCH models the empirical fact that volatile days cluster together — a large move today predicts elevated volatility tomorrow. GJR-GARCH adds an asymmetry term (γ) for the leverage effect: bad news raises volatility more than equally-sized good news. This is a well-documented feature of energy markets.

Model selection is automatic: energy commodities (WTI, Natural Gas) use GJR-GARCH; metals and grains use GARCH. Both use Student-t innovations to handle fat-tailed commodity return distributions.

**Dashboard output:** Overlay chart of GARCH conditional volatility vs 21-day realized volatility (annualized %), plus a 21-day forward forecast. Metrics include current conditional vol, forward vol, and the leverage coefficient γ.

**Key insight:** When GARCH forecast vol rises above realized vol, it's a forward warning — options premiums are likely to expand before the realized move arrives. A negative γ (e.g., WTI at −0.077) confirms the asymmetric response is statistically significant.

---

#### Kalman Filter Dynamic Hedge Ratio
**What it does:** Estimates the time-varying β relationship between two commodity prices in real time using a state-space model. β drifts each day as a random walk; the Kalman filter optimally balances prior estimates against new observations. The spread z-score measures how far the current spread is from its recent norm in standard-deviation units.

Four built-in pairs: WTI ↔ Natural Gas, Gold ↔ Silver, Corn ↔ Soybeans, Wheat ↔ Corn.

**Dashboard output:** Two-panel chart — time-varying β (top) and spread z-score with ±2σ threshold lines (bottom). Metrics include current β, current z-score, and a plain-English signal (Long spread / Neutral / Short spread).

**Key insight:** When z-score exceeds +2.0, the spread is stretched and historically mean-reverts. This is the primary signal for commodity pair trades. A β that drifts significantly over time indicates a structural relationship shift — for example, the Gold/Silver ratio expanding during risk-off regimes.

---

#### VAR Granger Causality
**What it does:** A Vector Autoregression model fits all commodities in a group (Energy, Metals, or Grains) simultaneously and asks: does knowing Commodity A's past values help predict Commodity B's future values, above and beyond what B's own history already tells you? This is the Granger causality test. A low p-value means yes, A Granger-causes B.

**Dashboard output:** A p-value heatmap (green = strong causality, red = none) plus the optimal VAR lag order selected by BIC.

**Key insight:** WTI not Granger-causing Natural Gas (p ≈ 0.985) is economically correct — US Henry Hub gas is priced by domestic storage and weather, not global crude oil. Confirming this in real data is the model working as intended. Finding an unexpected causal link (e.g., Copper → Gold) can indicate a macro regime where industrial demand is driving safe-haven flows.

---

### Tab 2 — ML Models

#### HMM Market Regime Detector
**What it does:** An unsupervised Gaussian Hidden Markov Model trained on two daily observations — log-return and 21-day rolling volatility. It infers which of four hidden market states the commodity was most likely in on each day. States are labeled by sorting on mean return and injecting a High-Vol label at the highest-variance middle state: **Bear** (low return), **High-Vol** (middle return, highest variance), **Neutral** (middle return, lower variance), **Bull** (high return).

**Dashboard output:** Price chart with each day colour-coded by regime, a state probability heatmap for the last 126 trading days (shows gradual regime transitions before they become obvious), regime breakdown metrics, and today's current regime.

**Key insight:** The probability heatmap is more useful than the binary regime label. Watching the Bull probability creep from 5% to 30% over several weeks is an early-warning signal before the model formally flips to Bull. Very few Bull days (e.g., WTI had only 6 out of 756 in the last 3 years) reveal the persistent bearish macro backdrop.

---

#### XGBoost + SHAP Explainability
**What it does:** A gradient-boosted tree ensemble that predicts next-day log-return using 48 engineered features (momentum at multiple horizons, mean-reversion z-scores, rolling volatility ratios, cross-commodity return spreads). Trained on an 85%/15% walk-forward split with early stopping.

SHAP (SHapley Additive exPlanations) solves the black-box problem: for every prediction, each feature receives a signed contribution value derived from game-theoretic Shapley values — what did this feature add, relative to the average prediction across all data?

**Dashboard output:** Two side-by-side charts. Left: global SHAP importance (average absolute SHAP across all dates — what matters most overall). Right: a waterfall chart for today's prediction (which specific features pushed today's forecast up or down). Metrics: IC, top bullish driver, top bearish driver for today.

**Key insight:** The global importance chart answers "what drives WTI returns in general." The waterfall answers "why is the model bullish/bearish *today specifically*." Cross-commodity drivers appearing in the waterfall (e.g., Gold z-score pushing WTI) reveal macro linkages — flight-to-safety Gold spikes often lead energy by a day during risk-off events.

---

#### Random Forest
**What it does:** An ensemble of 300 decision trees, each trained on a bootstrap sample and a random feature subset (√N features per split). The final prediction is the mean across all trees. Gini impurity feature importance measures how much a feature reduces impurity across *all* splits in *all* trees — a structural signal of explanatory power across the full training window.

Rolling re-fit: the model re-trains on a 252-day rolling window every 21 days, so importance rankings adapt as the macro regime shifts.

**Dashboard output:** Horizontal bar chart of top 20 features by Gini importance, IC score, and top feature share.

**Key insight:** Gini importance differs from SHAP in a useful way — it's a global structural measure (what matters across all market conditions) rather than a prediction-attribution measure (what matters for this specific observation). When momentum features dominate the Gini chart, it means the commodity is in a trend-following regime. When volatility-ratio features dominate, it suggests a mean-reversion regime.

---

#### Elastic Net / Lasso Factor Model
**What it does:** A linear model with combined L1 (Lasso) and L2 (Ridge) regularisation. L1 forces most coefficients exactly to zero, automatically selecting the minimal set of factors that explain the target. L2 prevents collinear features from blowing up. The regularisation strength and L1/L2 mix are tuned via 5-fold time-series cross-validation.

**Dashboard output:** Bar chart of non-zero coefficients (green = positive/bullish, red = negative/bearish), IC, number of active features, sparsity percentage, regularisation strength, and a plain-English factor narrative ("WTI return is driven by: momentum_5d (+), DXY_zscore (−)").

**Key insight:** Extreme sparsity (e.g., 97.9% of features zeroed out, leaving 1 active) tells you the signal is very concentrated right now. A single dominant factor is sometimes more reliable than a complex multi-factor model — and easier to monitor. The factor narrative makes the model legible to a non-technical audience.

---

### Tab 3 — Market Signals

#### DXY / VIX / TLT Macro Overlay
Three instruments that proxy for three distinct macro regimes, fetched free via Yahoo Finance:

- **DXY (^DXY):** US Dollar Index. Most commodity futures are USD-priced, so a strong dollar depresses commodity demand from non-USD buyers. Gold and oil are particularly sensitive. The dashboard shows the 63-day z-score (not raw level) so you see dollar *strength* relative to recent history.
- **VIX (^VIX):** CBOE implied equity volatility ("fear gauge"). Above 20 = risk-off; above 30 = crisis. In risk-off regimes, commodity correlations converge and momentum signals weaken — this is when most systematic strategies fail. The bar chart colours each day green (risk-on), purple (risk-off), or red (crisis).
- **TLT:** iShares 20+ Year Treasury ETF. TLT price moves inversely to long-term US interest rates. Rising rates = falling TLT = commodity headwind (higher inventory discount rates, stronger USD carry, competition from fixed income). The 21-day momentum shows whether rates are trending up or down.

**Dashboard output:** Three-panel chart (one per instrument), plus five live metrics: VIX level and regime label, DXY z-score, TLT 21d momentum, days to next WASDE report, days to next OPEC+ meeting.

---

#### WASDE & OPEC+ Calendar Dummies
Two recurring events that cause predictable volatility spikes in specific commodity groups:

- **WASDE (USDA World Agricultural Supply and Demand Estimates):** Published monthly, typically the 2nd Tuesday at 12:00 noon ET. Agricultural futures (Corn, Wheat, Soybeans) experience elevated volatility in the 1–7 days before and 5 days after release as traders position around surprise estimates. Release dates are hardcoded through 2026, with an algorithmic 2nd-Tuesday fallback for future dates.
- **OPEC+ meetings:** Typically twice yearly (June and November/December), with extraordinary meetings when production policy changes urgently. Energy futures (WTI, Brent, Heating Oil, Gasoline) trade elevated vol in the 2 weeks before a scheduled meeting. 24 historical and future meeting dates are hardcoded through 2026.

The dashboard shows signed days-to-event (negative = days before, positive = days after) and flags whether the current date is inside the event window.

---

#### ENSO Climate Signal (MEI v2)
**What it does:** The Multivariate ENSO Index v2 from NOAA PSL (no API key required). El Niño / La Niña disrupts global precipitation and atmospheric circulation, with lagged effects on specific commodity groups:

- **El Niño (MEI > +0.5):** Drier SE Asia → bearish palm oil, cocoa, coffee (3–6 month lead). Wetter US Gulf Coast → bullish US soy. Drier Australia → bearish Australian wheat.
- **La Niña (MEI < −0.5):** Wetter SE Asia. Drier South America → bearish Brazil soy (2–4 month lead). Cooler US winters → higher natural gas demand.
- **Neutral:** No directional bias from ENSO.

**Dashboard output:** Bar chart of MEI v2 (last 2 years) coloured by phase, live MEI reading, 3-month lag signal, 6-month lag signal, and a plain-English interpretation of the current phase and which commodities to watch.

---

#### Energy Transition Signals
Three proprietary signals built from free market data:

- **Uranium Spread Proxy:** URA (Global X Uranium ETF) price vs its 90-day simple moving average, z-scored over 63 days. A positive z-score indicates uranium miners are outperforming trend, signalling increased nuclear energy interest.
- **Battery Metals PC1:** First principal component of LIT (lithium), GLNCY (cobalt/zinc), and REMX (rare earths) returns — a single index capturing the shared "battery metals demand" factor across EV-critical materials.
- **ETS Policy Stress:** KRBN (KraneShares Global Carbon ETF) volatility-weighted return, measuring how aggressively carbon markets are pricing in regulatory tightening.

---

### Tab 4 — Deep Learning (on-demand)

These models are computationally expensive and run only when you click the button. They require optional heavy dependencies (`torch`, `prophet`, `pytorch-forecasting`).

#### BiLSTM Multi-Commodity Forecaster
A 2-layer bidirectional Long Short-Term Memory network with a shared encoder and per-commodity output heads. It processes 63-day sequences in both forward and backward directions, capturing asymmetric temporal patterns (e.g., recoveries look different from drawdowns). Trains on all commodities jointly — multi-task learning — which prevents overfitting on individual sparse series. Uses Adam optimiser with ReduceLROnPlateau and early stopping (patience = 15 epochs).

#### Meta-Prophet Decomposition
Facebook/Meta's additive decomposition model separates each price series into trend + yearly seasonality + weekly seasonality + holiday effects. Ten major macro events (Covid crash, Ukraine invasion, Fed rate hike cycle, etc.) are injected as change-point priors. The dashboard shows a 90-day price forecast, detected changepoints matched to known macro events, seasonal strength scores, and a trend regime classification (accelerating, decelerating, or reversing).

#### Temporal Fusion Transformer (TFT)
Produces multi-horizon quantile forecasts [q10, q50, q90] for 1-day, 5-day, and 20-day horizons simultaneously via pytorch-forecasting. The Variable Selection Network (VSN) inside TFT learns which features to weight at which time steps. Attention weight outputs show the model's temporal "memory" — how far back it is looking for each prediction horizon. Best suited for longer-horizon planning where interval forecasts (not just point estimates) are required.

---

### Tab 5 — Quantum Models (experimental, on-demand)

These are research-grade models. They require `pennylane >= 0.38` and run as classical simulations unless an IBM Quantum backend is configured via `pennylane-qiskit`.

#### Quantum Kernel SVM Benchmark
Encodes commodity return features into quantum circuits using three circuit families — RY/RZ (basic rotation), ZZFeatureMap (entangling map from Qiskit), and IQP (Instantaneous Quantum Polynomial) — at 4, 6, and 8 qubits. Computes a quantum kernel matrix and trains an SVM classifier. Benchmarks Spearman IC vs a classical RBF-kernel SVM baseline. Includes depolarizing noise simulation at five noise levels to model realistic IBM quantum hardware degradation.

#### QAOA Portfolio Optimiser
Formulates asset selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem — each qubit represents one commodity asset, and the Quantum Approximate Optimization Algorithm searches for the binary allocation (include/exclude) that maximises expected return minus λ×variance, subject to an exact cardinality constraint (pick exactly K assets). The cost landscape chart shows how solution quality varies across the QAOA circuit parameter space (γ, β), letting you inspect whether the optimiser converged.

#### QNN Hybrid Layer Comparison
A hybrid quantum-classical neural network where classical linear layers feed into a PennyLane `qml.qnn.TorchLayer` (trained via parameter-shift gradients — the quantum-native backpropagation method), then back to classical output layers. The `comparison_study()` function trains both a QNN and an equivalent fully classical network side-by-side on the same data split and reports IC, training loss curves, and wall-clock runtime. The goal is empirical benchmarking — does the quantum layer provide any advantage on financial time-series data?

---

### Feature Engineering (`features/` package)

All models in Tabs 1–2 use a 48-column feature matrix built from price history. The `features/` package adds external signals on top of this base:

| Signal group | Columns | Source | API key needed |
|---|---|---|---|
| Price momentum | `*_mom5`, `*_mom21`, `*_zscore`, `*_vol21` | yfinance | No |
| Cross-commodity | Return spreads, correlation ratios | yfinance | No |
| DXY / VIX / TLT | `dxy_zscore63`, `vix`, `vix_risk_off`, `tlt_mom21` | yfinance | No |
| Calendar dummies | `days_to_wasde`, `is_wasde_window`, `days_to_opec` | Hardcoded dates | No |
| ENSO | `mei`, `mei_lag3m`, `mei_lag6m`, `enso_phase` | NOAA PSL flat file | No |
| Energy transition | `uranium_spread_zscore`, `battery_demand_index`, `ets_stress_zscore` | yfinance | No |
| PDSI / HDD / CDD | `pdsi_cornbelt`, `hdd_deviation_pct`, `cdd_deviation_pct` | NOAA CDO API | Yes — free at ncdc.noaa.gov/cdo-web/token |
| FinBERT sentiment | `sentiment_wti_crude_oil`, etc. | HuggingFace model | No (GPU recommended) |
| EIA inventory | `eia_crude_surprise`, `eia_crude_surprise_z` | EIA API v2 | Yes — free at eia.gov/opendata |

The `augment_model_features()` function in `features/assembler.py` filters to columns with ≥ 50% data coverage, preventing sparse external signals from contaminating the ML training matrix when API keys are absent.

**Why `adjusted_close` matters for models:** Training on the raw `close` series means every model sees fictional −11% crashes at futures roll dates. These non-events corrupt every feature derived from the series (momentum, z-scores, rolling volatility). `adjusted_close` gives models a return series where every data point reflects a real price move.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in your activated venv |
| `could not connect to server` | Make sure Postgres.app is running |
| `database "commodities" does not exist` | Run `createdb commodities` in your terminal |
| Prices show "mock data" warning | Yahoo Finance may be throttled — wait a few minutes and refresh |
| News shows mock articles | Some RSS feeds block automated requests — this is normal; others will still load |
| Dashboard is slow on first load | First load fetches all API data; subsequent interactions use the cache |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## Contributing

1. Create a branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Test locally: `streamlit run app.py`
4. Open a pull request with a description of what you added

---

## License

MIT — free to use, modify, and distribute.
