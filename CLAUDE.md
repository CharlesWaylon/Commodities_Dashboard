# Commodities Dashboard – Project Notes for Claude

## ⚠️ STANDING REMINDER (show this every session)
At the start of any task in this project, remind the user:

> "Heads up: your ingestion is now fully autonomous via **macOS launchd** — it runs daily at 2:05 PM Eastern (Mon–Fri) without any terminal or Cowork. If data ever looks stale, check `logs/ingest.log` and `logs/ingest_error.log` in the project root."

---

## ⚠️ MODEL SCOPE RULE — MANDATORY FOR ALL NEW MODELS
Every new model, selectbox, or data-loading path added to this project **must** use the full instrument universe. Do **not** hardcode `CORE_TICKERS` (the 11-commodity subset) for any model.

### Required patterns
- **Commodity list for selectboxes:** `from models.config import MODELING_COMMODITIES` — use `list(MODELING_COMMODITIES.keys())`.
- **Price data for models:** use `prices_full = load_prices_en()` (already a global in `pages/4_Models.py`, cached for 4 hours via `@st.cache_data`). Never pass the 11-commodity `prices` variable to a model.
- **Returns:** derive from `prices_full`: `returns_full = np.log(prices_full / prices_full.shift(1)).dropna()`.
- **`CORE_TICKERS`** is only used by `load_prices()` for the lightweight 11-commodity price ticker used in the macro trigger sidebar and certain UI display paths. It must not be used as the universe for any predictive model.

### Why
`CORE_TICKERS` was the old 11-commodity subset. `MODELING_COMMODITIES` is the full 40-instrument universe (futures + ETFs/index funds) defined in `models/config.py`. Every model in the dashboard was audited on 2026-05-12 and updated to use `MODELING_COMMODITIES`; do not regress.

---

## Database
- **Backend:** PostgreSQL — `postgresql://charlesmerkel@localhost/commodities`
- **SQLite is no longer used.** The old `data/commodities.db` file still exists but is not read by any code.
- **All 9 previously SQLite-only modules** were migrated to SQLAlchemy + Postgres on 2026-05-11.
- **Row count as of 2026-05-11:** 52,419 rows in `price_history`; 51,865 rows in `aligned_prices`
- **Tables:** `commodities`, `price_history`, `aligned_prices`, `correlation_snapshots`, `forecast_log`, `ingestion_log`, `price_validation_log`, `trigger_events`, `ic_log`, `model_training_log`, `threshold_config`, `cascade_validation_log`, `cascade_validation_summary`, `cascade_forecasts`, `causal_monitoring_log`

## Data Ingestion Pipeline
- `pipeline/ingest.py` — fetches OHLCV data from Yahoo Finance via `yfinance`; safe to run repeatedly (idempotent upserts). Full pipeline: fetch → roll_adjust → align_calendar → correlation_snapshot.
- **Autonomous scheduler:** macOS launchd agent at `~/Library/LaunchAgents/com.accendio.commodities.ingest.plist`
  - Runs daily at **14:05 Eastern (18:05 UTC)** Mon–Fri
  - Starts automatically on login; survives reboots
  - Logs: `logs/ingest.log` (stdout) and `logs/ingest_error.log` (stderr)
- **Manual catch-up command:**
  ```bash
  cd ~/Desktop/Future_of_Commodities/Commodities_Dashboard
  python -m pipeline.ingest
  ```
- **Manage launchd agent:**
  ```bash
  launchctl list | grep accendio          # check status (PID or last exit code)
  launchctl stop com.accendio.commodities.ingest   # stop a running job
  launchctl start com.accendio.commodities.ingest  # trigger manually
  launchctl unload ~/Library/LaunchAgents/com.accendio.commodities.ingest.plist  # disable
  launchctl load ~/Library/LaunchAgents/com.accendio.commodities.ingest.plist    # re-enable
  ```
- **Network note:** yfinance is blocked in the Cowork sandbox. Ingestion must run on the user's Mac directly (launchd handles this).
