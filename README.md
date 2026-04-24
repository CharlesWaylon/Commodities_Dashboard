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
| 🤖 **Models** | Statistical analysis (volatility, returns, correlation) + Phase 3 model roadmap |
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
- `streamlit` — web framework that turns Python scripts into interactive dashboards
- `plotly` — interactive charting library
- `yfinance` — fetches market data from Yahoo Finance
- `feedparser` — parses RSS news feeds
- `pandas` / `numpy` — data manipulation
- `python-dotenv` — loads `.env` config files
- `apscheduler` — runs the ingestion pipeline on a schedule
- `sqlalchemy` — ORM that talks to SQLite or PostgreSQL
- `psycopg2-binary` — PostgreSQL driver

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
│   ├── 4_Models.py           ← Stats analysis + model roadmap
│   └── 5_Database.py         ← DB inspector + manual ingestion trigger
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

## Phase 3 — Predictive Models (Roadmap)

Phase 2 (the data pipeline) is complete. Phase 3 will use the clean, roll-adjusted price history to build predictive models:

```
PostgreSQL (adjusted_close series — clean, roll-adjusted)
       │
       ├── Feature engineering (rolling stats, spreads, seasonality flags)
       │
       ├── ARIMA / SARIMA    — classical time-series forecasting
       ├── Prophet           — Facebook's trend + seasonality decomposition
       ├── XGBoost           — gradient boosted trees on engineered features
       └── LSTM              — deep learning sequence model
              │
              ▼
       Forecast outputs → Models page in this dashboard
```

**Why `adjusted_close` matters for models:** Training a model on the raw `close` series means the model sees fictional −11% one-day crashes at every futures roll date. Those non-events pollute every feature (rolling volatility, momentum, z-scores) calculated from the series. `adjusted_close` gives models a return series where every data point reflects a real price move.

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
