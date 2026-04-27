# Commodities Dashboard – Project Notes for Claude

## ⚠️ STANDING REMINDER (show this every session)
At the start of any task in this project, remind the user:

> "Heads up: your long-term goal is to make the data ingestion scheduler **autonomous within the dashboard scripts themselves** (e.g., APScheduler embedded in the Streamlit app, or a macOS `launchd` daemon), so it doesn't rely on Cowork or any token usage. The current Cowork scheduled task is a temporary stopgap — ask me to set up `launchd` whenever you're ready."

---

## Database
- **Location:** `data/commodities.db` (SQLite)
- **Schema migrated:** 2026-04-27 — added `instrument_type` column to `commodities`, and created `aligned_prices` and `ingestion_log` tables to match current ORM models
- **Row count as of migration:** 52,118 rows in `price_history`
- **Last successful ingest from dashboard terminal:** ~April 23, 2026

## Data Ingestion Pipeline
- `pipeline/ingest.py` — fetches OHLCV data from Yahoo Finance via `yfinance`; safe to run repeatedly (idempotent upserts)
- `pipeline/scheduler.py` — APScheduler wrapper; runs ingest daily at 18:05 UTC Mon–Fri; **must be kept alive in a terminal** (not autonomous)
- **Manual catch-up command:**
  ```bash
  cd ~/Desktop/Future_of_Commodities/Commodities_Dashboard
  python -m pipeline.ingest
  ```
- **Network note:** yfinance is blocked in the Cowork sandbox (403 proxy errors). Ingestion must run on the user's Mac directly.

## Autonomous Scheduler Options (priority order)
1. **macOS launchd** — best; survives reboots; runs silently. Ask Claude to generate the `.plist` file.
2. **cron** — simpler: `crontab -e` → `5 18 * * 1-5 cd ~/Desktop/Future_of_Commodities/Commodities_Dashboard && python -m pipeline.ingest`
3. **APScheduler embedded in Streamlit** — run `pipeline/scheduler.py` as a background thread started by `app.py` on launch
4. **Cowork scheduled task** — current stopgap; runs daily at 18:05 local time; will remind user to run manually if network fails
