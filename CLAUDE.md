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

## ⚠️ MODEL VERIFICATION RULE — MANDATORY FOR ALL MODELS
**Every model, weight, prior, or parameter that is trained or implemented must be
verified against outside resources before it ships.** This applies to new models
and to changes to existing ones.

### Required workflow
1. **Check against external sources.** Before trusting a learned weight,
   coefficient, edge, or prior, confirm it is economically/physically sensible
   using outside resources (literature, industry data, domain references — use
   web search when available).
2. **Communicate the result to Charles — positive OR negative.** Always report
   whether the verification confirmed, refuted, or was inconclusive about the
   model's behaviour. Do not silently fix and move on; do not hide negative
   findings.
3. **Record it in the repository.** Append an entry to
   [`MODEL_VERIFICATION_LOG.md`](MODEL_VERIFICATION_LOG.md) (repo root) with:
   what was verified, the sources, the verdict, and what changed in code.

### Why
In-sample **correlation measures co-movement, not causation.** On 2026-06-01 the
cascade topology edge weights were found to be economically inverted
(Metals→Agriculture ranked near Energy→Agriculture) because they were derived
from correlation alone. Outside sources showed energy→agriculture transmission
dominates (natural gas = 70–80% of fertiliser cost). See the log for the full
finding and remediation.

### Economic-prior methodology (the fix pattern)
When a learned cross-sector weight contradicts fundamentals, blend an economic
prior in multiplicatively rather than overriding the data:
- Priors live in `models/config.py::SECTOR_TRANSMISSION_PRIORS` (data-driven;
  tunable without touching code), with `DEFAULT_TRANSMISSION_PRIOR` and
  `UPSTREAM_PRIOR_STRENGTH` (α).
- `models/sector_model.py::_economic_prior` computes
  `prior = (1 − α) + α × economic_prior`; the upstream contribution becomes
  `corr × upstream_forecast × damping × prior`.
- `α = 0` reproduces legacy correlation-only behaviour (backward compatible);
  `α = 1` applies the full prior.
- Priors take effect during cascade **fitting**, so existing `cascade_forecasts`
  rows reflect the blend only after the next `models/daily_retrain.py` run.

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
- **Macro feed daemon (separate from ingest):** `pipeline/run_macro_feed.py` is supervised by `ops/com.accendio.macrofeed.plist` (`KeepAlive=true`, restarts on crash). Bootstrap once with `launchctl bootstrap gui/$(id -u) ops/com.accendio.macrofeed.plist` (run from repo root); check with `launchctl list | grep macrofeed`; logs at `logs/macro_feed.log` and `logs/macro_feed.error.log`.

---

## Macro trigger integration (shipped 2026-05-27)

### Where features live
- **`features/macro_features.py`** is the single source of truth for "what was the macro state on date X?" Every downstream model reads from here rather than re-deriving its own snapshot.
- **Public API:** `get_macro_state_at(date)`, `get_active_triggers(date, lookback_days)`, `build_macro_surprise_features(date)`, `family_to_regime(family)`, `regime_hint_from_triggers(triggers)`.

### Feature flag
- `MACRO_TRIGGERS_ENABLED` (env var, default `true`) gates every trigger-aware code path.
- Rollback for the whole stack: `export MACRO_TRIGGERS_ENABLED=false`. All affected modules (`cascade_orchestrator`, `macro_router`, `sector_model`, `meta_predictor`, `portfolio_optimizer`) check this independently and fall back to pre-trigger behavior.

### Adding a new trigger family
1. **Register the family** in `config/trigger_registry.json` (the existing pattern) and write its detector in `features/trigger_detectors.py` or a `services/` ingestor.
2. **Map it to a regime** by adding an entry to `_FAMILY_TO_REGIME` in `features/macro_features.py`. The regime must be one of `rate_shock`, `growth_shock`, `commodity_shock`. If the family doesn't fit cleanly, the `family_to_regime` prefix-substring fallback handles `fed_*`, `cpi_*`, `ppi_*`, `opec_*`, `weather_*`, `eia_*`, `usda_*`, `energy_*`, `geo*`, `unemployment_*`, `gdp_*`, `nonfarm_*`, `recession_*` automatically.
3. **(Optional) macro-snapshot amplification** — if this family should boost specific snapshot features in cascade fits, add it to `TRIGGER_FAMILY_TO_MACRO_FEATURES` at the top of `models/cascade_orchestrator.py`.
4. **(Optional) upstream-sector boost** — if this family should intensify a specific upstream sector path in `sector_model.predict_with_context`, add it to `TRIGGER_FAMILY_TO_UPSTREAM_SECTOR` at the top of `models/sector_model.py`.
5. **(Optional) portfolio risk gate** — if this family should reshape allocations, add an entry to `TRIGGER_RISK_GATES` in `models/config.py` (data-driven; non-engineers can tune without touching code).

### Per-step audit (where to inspect that triggers are flowing)
- **Step 1 surface:** `features/macro_features.py` + `features/test_macro_features.py`.
- **Step 2 cascade snapshot blending:** `models/cascade_orchestrator.py` — see `_extract_macro_snapshot`, `_apply_trigger_amplification`. Result carries `regime_hint` + `active_triggers`. Per-row macro_detail JSON includes `regime_hint` + `n_active_triggers`.
- **Step 3 regime override:** `models/macro_router.py` — `get_current_regime` is overridden by triggers when `MACRO_TRIGGERS_ENABLED`. Backtest classifier `_classify_regimes` writes shock-regime labels into the historical regime series; `fit()` learns separate β coefficients for `rate_shock` / `growth_shock` / `commodity_shock` with linear shrinkage toward neutral β when `n_obs < SHOCK_REGIME_SHRINKAGE_N` (=30).
- **Step 4 dynamic upstream damping:** `models/sector_model.py` — `_compute_upstream_adjustment` reads `active_triggers` (caller-supplied by cascade, otherwise auto-fetched) and boosts the damping on upstream paths whose sector matches `TRIGGER_FAMILY_TO_UPSTREAM_SECTOR`.
- **Step 5 meta-predictor features:** `models/meta_predictor.py` — `FEATURE_COLUMNS` now includes 7 trigger-derived columns (`cpi_surprise_z`, `unrate_surprise_z`, `fedfunds_surprise_z`, `t10y2y_change_5d`, and three `regime_hint_onehot_*` indicators). After this change, the next `daily_retrain` rewrites `data/meta_predictor.pkl`; old pkls are detected by feature-count mismatch in `load()` and the predictor falls back to equal-weights instead of crashing.
- **Step 6 portfolio risk gates:** `models/portfolio_optimizer.py::apply_trigger_risk_gates` — pure function applied post-QAOA. Rules live in `models/config.py::TRIGGER_RISK_GATES`. `CascadePortfolioResult.weights` returns `gated_weights` when any gate fires (transparent to existing callers).
- **Step 7 historical replay scenarios:** `models/scenarios/ripple.py::HistoricalTriggerReplay` — surfaceable on `pages/9_Scenarios.py` as a new scenario type.
