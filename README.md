# Accendio — Commodities Intelligence Dashboard

*An interactive market-intelligence platform I've been building to trace how a single shock — a dollar move, a vol spike, a regime flip — reverberates across the entire commodity universe.*

Built with Python, Streamlit, and Plotly. Live data, no paid APIs.

---

## Why I'm Building This

I started this project with one stubborn question: **when something happens in one corner of the market, can I actually see the wave move through everything else?** A stronger dollar isn't just a number on a screen — it leans on energy, pushes gold around, changes what a Brazilian soy exporter earns, eventually shows up in feeder cattle prices through the cost of corn. Those linkages exist. They're real. But most dashboards show 41 disconnected price tiles and call it a day.

What I wanted was a *chain*. A system where a macro stimulus enters at one end, propagates through sectors with quantified confidence at each hop, and lands on a directional recommendation at the other end — with every link in that chain inspectable, every assumption traceable back to a model I can defend.

That's the thesis. Below is how close I've gotten.

---

## The Two-Layer Architecture (and Why It Works)

After a lot of iteration I landed on splitting the system into two analytical layers. The distinction matters because they answer different questions.

### Layer 1 — External Input: the Macro-Market Cascade
Three macro instruments set the weather before any commodity-specific model fires: **DXY** (dollar strength), **VIX** (risk sentiment), **TLT** (rate expectations). Each carries a transmission path into Energy, Metals, Grains, and Livestock that I learn empirically (`models/macro_router.py`) and re-validate against textbook economic theory every week. The cascade visualisation shows which links are firing right now and how strongly.

### Layer 2 — Internal Response: the Causal QS Engine
Once the macro environment is established, the QS Engine takes a specific event — a vol spike, a momentum break, a regime shift — and walks it through the model stack: **trigger → GARCH volatility estimate → HMM regime → MetaPredictor ensemble → directional position recommendation**. Every transition is auditable. If the recommendation surprises me, I can drop into any link and ask *why*.

**Read them together.** The Cascade tells me *why conditions are what they are*. The QS Engine tells me *what to do about it*.

---

## How Far It's Come

Here's an honest map of where the chain stands today.

| Stage of the chain | Status | What's in the dashboard |
|---|---|---|
| **Clean data foundation** | ✅ Solid | 41 instruments, 5-year history, roll-adjusted, calendar-aligned, autonomous daily refresh |
| **Single-instrument forecasting** | ✅ Solid | ARIMA, GARCH, BiLSTM, Prophet, TFT — point + interval forecasts with IC tracking |
| **Cross-sectional regime detection** | ✅ Solid | HMM 4-state regimes per commodity + macro overlay (DXY/VIX/TLT z-scores) |
| **Macro → sector transmission** | ✅ Working | Empirical OLS routes (4 macros × 5 sectors × 5 regimes = 100 regressions), domain-validated 7/9 |
| **Cross-asset propagation** | ✅ Working | Kalman pairs, VAR Granger causality, cross-commodity feature spillovers in XGBoost/RF |
| **Event-triggered cascade** | ✅ Working | Causal QS Engine traces a trigger end-to-end; Macro-Market Cascade visualises live flow |
| **Scenario envelopes** | ✅ New | Bear / average / bull fan charts from an ensemble of all 9 model families |
| **Live event ribbon + alerts** | ✅ New | WebSocket ribbon for incoming trigger events; severity-tiered alert engine |
| **Cyclic feedback edges** | 🟡 Backlogged | Agriculture → Energy (biofuels) and Livestock → Agriculture (feed) edges still pending |
| **Capital allocation** | ✅ Working | QAOA quantum-inspired portfolio optimiser with consistency flags |

So the skeleton is wired end-to-end. A stimulus can enter at the macro layer, flow through sector routes, fire commodity-specific models, generate scenario bands, raise alerts, and inform a portfolio recommendation — all in one session, all on live data. There are still gaps (the cyclic edges in particular), but the chain exists.

---

## What Each Page Does

| # | Page | What it answers |
|---|------|-----------------|
| Home | **Command Centre** | What's moving today and how are sectors correlated right now? |
| 1 | **Pricing** | Full sortable price table + treemap across 41 instruments |
| 2 | **Charts** | Interactive OHLCV with SMAs + multi-commodity overlays |
| 3 | **News** | Live RSS aggregation (Reuters, CNBC, FT, etc.) — no paid feed needed |
| 4 | **Models** | 5-tab predictive lab: statistical, ML, deep learning, quantum, market signals |
| 5 | **Database** | Inspect storage, trigger manual ingestion, audit per-commodity coverage |
| 6 | **Causal QS Engine** | Trace a single trigger event end-to-end → position recommendation |
| 7 | **Macro-Market Cascade** | Watch DXY/VIX/TLT flow through sectors to individual commodities |
| 8 | **Portfolio** | QAOA allocation + correlation heatmap + consistency flags |
| 9 | **Scenarios** | Bear / mean / bull fan charts ensembled across all 9 model families |
| 10 | **Event Ribbon** | Live WebSocket ribbon of incoming trigger signals + lifecycle state |
| 11 | **Macro Exposure** | Heatmap of which triggers are currently elevating which commodities |
| 12 | **Alerts** | Severity-tiered alert engine drained from the trigger bus, with rules |

---

## How I Actually Use It

A typical session for me looks like this:

1. **Open the Command Centre.** Scan the heatmap and the sector correlation panel. If something's lit up unusually, I already have a hypothesis before I dig.
2. **Page 7 — Macro-Market Cascade.** Check the three macro z-scores (DXY, VIX, TLT). If any is outside ±1σ, that's the dominant driver today. The cascade shows which sectors are getting hit hardest.
3. **Page 11 — Macro Exposure.** Cross-reference: are any triggers currently active, and which commodities are they elevating? This is the "live exposure" view.
4. **Page 4 — Models, Tab 2 (ML).** For whatever commodity I'm interested in, look at the SHAP waterfall for *today specifically*. The global importance chart tells me what matters in general; the waterfall tells me what's driving the prediction *right now*. Pair that with the HMM regime probability heatmap to see if the regime is shifting underneath me.
5. **Page 6 — Causal QS Engine.** If a trigger has fired, trace it through the stack and read the recommendation. If the recommendation surprises me, I drill into the GARCH / HMM / MetaPredictor links.
6. **Page 9 — Scenarios.** Pull the fan chart for a horizon I care about. The width of the cone tells me how much disagreement there is across model families — narrow cone = consensus, wide cone = the models disagree and I should be humble.
7. **Page 8 — Portfolio.** If I'm thinking about allocation, QAOA gives me a binary include/exclude set under a cardinality constraint, and the consistency flags warn when the optimiser is picking assets the directional models disagree on.

---

## Simple, Actionable Insights I Can Pull Consistently

Not every output is alpha, but several signals have proven repeatedly useful:

- **DXY z-score > +1 with a falling VIX:** dollar strength is the dominant force, gold is at risk, energy headwind is real. The macro router scores this cleanly across the relevant sectors.
- **HMM Bull-state probability rising from <10% to >25% over 2–3 weeks:** early warning of a regime flip *before* the binary label changes. The probability heatmap is the most consistently useful chart on Page 4 Tab 2.
- **Kalman spread z-score > +2 on a known pair (WTI–NG, Gold–Silver, Corn–Soy, Wheat–Corn):** historically mean-reverts. The bread-and-butter pair trade setup.
- **GARCH forecast vol rising above 21-day realised vol:** options premiums are about to expand. Even when I don't trade options, this is a heads-up that the realised move is coming.
- **Elastic Net showing >95% sparsity with one dominant factor:** the signal is concentrated. A single-factor regime is often easier to monitor and more reliable than a multi-factor one.
- **VAR p-value heatmap shifting:** when a commodity that historically didn't Granger-cause another suddenly does, the macro regime has changed. (Copper → Gold lighting up = industrial demand is driving safe-haven flows.)
- **Scenario fan-chart width spiking:** model disagreement is high. Size positions down. This single visual has saved me from over-confidence more than once.
- **WASDE / OPEC+ event windows:** the calendar dummies flag when ag or energy vol is about to be elevated. Even a club member who only checks the dashboard once a week can pull this signal off Page 4 Tab 3.

---

## Prerequisites

- Python 3.10+
- `pip`
- An internet connection (for live data)
- [Postgres.app](https://postgresapp.com/) (free, Mac)

**No API keys required.** All data comes from Yahoo Finance (free) and public RSS feeds (free).

---

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd Commodities_Dashboard
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate          # mac/linux
# venv\Scripts\activate          # windows
```

### 3. Install

```bash
pip install -r requirements.txt
```

**Core dashboard:** streamlit, plotly, yfinance, feedparser, pandas/numpy, python-dotenv, apscheduler, sqlalchemy, psycopg2-binary
**Statistical:** statsmodels, arch
**ML:** scikit-learn, hmmlearn, xgboost, shap
**Deep (optional):** torch, prophet, pytorch-forecasting, lightning
**Quantum (optional):** pennylane, pennylane-qiskit
**Feature engineering:** transformers (FinBERT sentiment, GPU recommended)

### 4. Postgres

```bash
# install Postgres.app and start it
createdb commodities
echo "DATABASE_URL=postgresql://localhost/commodities" > .env
```

### 5. Backfill, roll-adjust, align

```bash
python -m pipeline.ingest --backfill
python -m pipeline.roll_adjust
python -m pipeline.align_calendar
```

### 6. Register the autonomous agents

```bash
launchctl load ~/Library/LaunchAgents/com.accendio.commodities.ingest.plist
launchctl load ~/Library/LaunchAgents/com.accendio.commodities.ingest-close.plist
launchctl load ~/Library/LaunchAgents/com.accendio.commodities.retrain.plist
```

### 7. Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## Autonomous Daily Schedule

The whole pipeline runs without me touching it. Three `launchd` agents:

| Time (ET) | Agent | What it does |
|---|---|---|
| **14:05** Mon–Fri | `com.accendio.commodities.ingest` | Midday incremental ingestion (last 5 days upsert) |
| **16:20** Mon–Fri | `com.accendio.commodities.ingest-close` | Post-close EOD capture |
| **16:45** Mon–Fri | `com.accendio.commodities.retrain` | Retrain all models, log IC scores, rebuild `macro_routes.pkl` |

Both ingestion runs are idempotent. The 14:05 run handles the case where my machine isn't on at close; the 16:20 run captures the true EOD that the 16:45 retrain trains on.

```bash
launchctl list | grep accendio                          # status
launchctl start com.accendio.commodities.ingest         # trigger manually
launchctl stop com.accendio.commodities.retrain         # stop a running job
tail -50 logs/retrain.log                                # last retrain output
tail -50 logs/ingest.log                                 # last ingest output
```

---

## Data Foundation

### Database — PostgreSQL

| Stat | Value |
|---|---|
| Instruments tracked | 41 (28 direct futures + 13 ETF/equity proxies) |
| `price_history` rows | ~52,000 (raw OHLCV + roll-adjusted) |
| `aligned_prices` rows | ~51,500 (one canonical 1,258-day calendar) |
| Date range | 2021-04-23 → present |
| Update frequency | Daily, Mon–Fri, via launchd |

**Tables:** `commodities`, `price_history`, `aligned_prices`, `correlation_snapshots`, `forecast_log`, `ingestion_log`, `price_validation_log`, `trigger_events`, `ic_log`, `model_training_log`, `threshold_config`, `cascade_validation_log`, `cascade_validation_summary`, `cascade_forecasts`, `causal_monitoring_log`.

### The three-pass cleaning pipeline

1. **Audit** (`pipeline/audit.py`) — coverage %, date gaps, zero-volume days, stale runs, z-scored spikes, impossible prices.
2. **Roll adjustment** (`pipeline/roll_adjust.py`) — proportional backward adjustment of futures roll artifacts using a per-instrument z-score threshold. 95 roll events detected and removed across the 28 futures instruments.
3. **Calendar alignment** (`pipeline/align_calendar.py`) — reindex all 41 instruments onto a canonical US-trading calendar derived empirically from the futures themselves. Forward-fill rate maxes out at 0.2% per instrument.

`adjusted_close` is what every model trains on. Raw `close` is preserved for audit only.

---

## Project Structure

```
Commodities_Dashboard/
├── app.py                       # Entry point / Command Centre
├── pages/                       # Streamlit auto-discovers
│   ├── 1_Pricing.py
│   ├── 2_Charts.py
│   ├── 3_News.py
│   ├── 4_Models.py              # 5-tab predictive lab
│   ├── 5_Database.py
│   ├── 6_Causal_QS_Engine.py    # internal response layer
│   ├── 7_Macro_Market_Cascade.py# external input layer
│   ├── 8_Portfolio.py           # QAOA optimiser
│   ├── 9_Scenarios.py           # bear/mean/bull fan charts
│   ├── 10_Event_Ribbon.py       # live WS trigger feed
│   ├── 11_Macro_Exposure.py     # exposure heatmap + commodity cards
│   └── 12_Alerts.py             # alert engine + rules
│
├── models/
│   ├── config.py                # MODELING_COMMODITIES (full 40-instrument universe)
│   ├── features.py              # 48-feature matrix builder
│   ├── statistical/             # ARIMA, GARCH, Kalman, VAR/VECM
│   ├── ml/                      # HMM, RF, XGBoost+SHAP, ElasticNet
│   ├── deep/                    # BiLSTM, Prophet, TFT
│   ├── quantum/                 # Quantum kernel, QAOA, QNN hybrid
│   ├── scenarios/               # Scenario aggregator + provider bands
│   ├── causal_chain.py          # QS Engine spine
│   ├── cascade_orchestrator.py  # Macro-Market Cascade engine
│   ├── macro_router.py          # Empirical OLS macro→sector routes
│   ├── meta_predictor.py        # Ensemble layer
│   ├── triggers.py              # Trigger classifier
│   ├── threshold_tuner.py
│   ├── ic_tracker.py
│   └── daily_retrain.py
│
├── features/                    # External-signal library
│   ├── assembler.py
│   ├── macro_overlays.py        # DXY/VIX/TLT + WASDE/OPEC+
│   ├── climate_weather.py       # ENSO MEI v2, PDSI, HDD/CDD
│   ├── energy_transition.py     # Uranium spread, battery PC1, ETS stress
│   └── sentiment.py             # FinBERT + EIA surprise
│
├── services/                    # Live data, trigger lifecycle, WS broadcast, alerts
├── components/                  # Event ribbon, macro heatmap, commodity cards, notifications
├── database/                    # SQLAlchemy ORM + engine
├── pipeline/                    # ingest, roll_adjust, align_calendar, audit
├── logs/                        # launchd agent stdout/stderr
├── reports/                     # Audit CSVs
├── utils/                       # theme, formatting helpers
└── .streamlit/config.toml
```

---

## The Model Library at a Glance

All models live under `models/` and load through `pages/4_Models.py`. Spearman IC > 0.05 is my actionable bar for a daily commodity signal.

**Tier 1 — Statistical:** ARIMA/SARIMA (AIC grid-searched), GARCH / GJR-GARCH (Student-t innovations), Kalman dynamic hedge ratio (4 built-in pairs), VAR Granger causality.

**Tier 2 — ML:** HMM 4-state regime detector, Random Forest (rolling re-fit), XGBoost + SHAP waterfall, Elastic Net sparse factor model.

**Tier 3 — Deep (on-demand):** BiLSTM multi-commodity forecaster, Meta-Prophet decomposition with macro changepoint priors, Temporal Fusion Transformer (multi-horizon quantile forecasts).

**Tier 4 — Quantum (experimental):** Quantum kernel SVM benchmark, QAOA portfolio optimiser, QNN hybrid layer comparison.

**Market signals:** Macro router (4 macros × 5 sectors × 5 regimes), DXY/VIX/TLT overlay, WASDE & OPEC+ calendar dummies, ENSO MEI v2 with 3- and 6-month lag, energy transition signals (uranium spread, battery PC1, ETS stress).

**Universe rule (mandatory):** every model uses `MODELING_COMMODITIES` (full 40-instrument universe). `CORE_TICKERS` is reserved for the lightweight 11-commodity macro-trigger sidebar.

---

## Adding a Commodity

Edit `services/price_data.py`:

```python
COMMODITY_TICKERS["My Commodity"] = "TICKER=F"
COMMODITY_SECTORS["My Commodity"] = "Energy"
COMMODITY_UNITS["My Commodity"]   = "USD/unit"
COMMODITY_PROXY_NOTES["My Commodity*"] = "* Tracks XYZ ETF which holds..."  # only if proxy
```

Then:

```bash
python -m pipeline.ingest --backfill
python -m pipeline.roll_adjust
python -m pipeline.align_calendar
```

Dashboard picks it up automatically.

---

## Pipeline Commands Reference

| Command | Run when |
|---|---|
| `python -m pipeline.ingest --backfill` | First setup or new instrument |
| `python -m pipeline.ingest` | On demand (launchd handles daily) |
| `python -m pipeline.roll_adjust` | After any backfill |
| `python -m pipeline.align_calendar` | After roll adjust |
| `python -m pipeline.audit` | On demand → `reports/audit_YYYYMMDD.csv` |
| `python -m models.daily_retrain` | On demand (launchd handles 16:45 ET) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` in your venv |
| `could not connect to server` | Make sure Postgres.app is running |
| `database "commodities" does not exist` | `createdb commodities` |
| Prices show "mock data" | yfinance throttled — wait and refresh |
| Slow first load | Cache warming; subsequent loads use `@st.cache_data` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| IC trend chart empty | Need 2+ days of retraining; check `logs/retrain.log` |
| Data stale | `launchctl list | grep accendio` — confirm agents are firing |

---

## What's Next

Highest priority for the chain right now:

- **Cyclic edges.** Agriculture → Energy (biofuels: corn/soy → ethanol/biodiesel demand) and Livestock → Agriculture (feed demand on corn/soybean meal). These are real economic feedback loops the cascade doesn't yet capture.
- **Macro-router refit on post-2022 data.** Two of the nine domain-validation checks still fail because the rolling window includes the Ukraine-invasion energy crisis. Target Q3 2026 for a clean refit.
- **Persistent alert routing.** Right now alerts surface inside the dashboard session. The natural next step is a push channel (email, webhook) so triggers can find me when I'm not staring at the screen.

---

## License

MIT — free to use, modify, and distribute.
