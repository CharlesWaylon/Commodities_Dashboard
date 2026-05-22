# Engineering History — Accendio

> A chronological technical log of how this dashboard was built. Audience: other engineers reading the codebase. **This is not the README.** The README explains *what the dashboard does today*; this file explains *how we got here, what assumptions are baked in, and where the load-bearing flaws still live.* Commit hashes are included so you can `git show <hash>` to inspect any milestone.

---

## TL;DR — the arc

The project went through six distinct phases in roughly four weeks:

1. **Skeleton (Apr 24–26)** — Streamlit shell, SQLite-backed price ingestion, classical+quantum baseline scaffolding.
2. **Model library (Apr 26–28)** — full 5-tier model suite materialised: statistical, ML, deep, quantum, plus the `features/` external-signal package.
3. **Live ecosystem (Apr 28–30)** — triggers, broadcaster, causal chain, meta-predictor, backtest harness. The first attempt at making models *talk to each other*.
4. **Signal quality (May 2–9)** — threshold tuner, IC tracker, cross-asset features, macro router. The chain went from "wired" to "calibrated."
5. **Two-layer split (May 10–11)** — the causal chain was bisected into the **Causal QS Engine** (internal response) and **Macro-Market Cascade** (external input). Price-validator alert system landed alongside.
6. **Polish & expansion (May 13–15)** — Optuna-tuned models on the full `MODELING_COMMODITIES` universe, scenario fan charts, FRED+Alpha Vantage ingestion, trigger bus + WebSocket broadcaster, alert engine, exposure heatmap.

The hard lessons: **alignment matters more than model choice** (Phase 4 only worked because Phase 1 cleaned the data); **interconnection is the product** (Phase 3 was where the dashboard became defensible); **bisecting concerns saved the architecture** (the May 10 split unblocked everything after).

---

## Phase 1 — Skeleton (2026-04-24 → 2026-04-26)

### `3cf8807` — initial commit
The starting shape:
- `app.py` Streamlit entry + 5 pages (Pricing, Charts, News, Models, Database).
- SQLAlchemy ORM in `database/models.py`; SQLite at `data/commodities.db`.
- `models/classical/baseline.py`, `models/quantum/{embedding,hybrid,kernel}.py`, `models/run_experiment.py` — classical-vs-quantum experiment harness was always part of the thesis.
- `models/features.py` with the 48-column feature matrix (momentum + mean-reversion + cross-commodity spreads). This matrix has barely changed since; every later model still consumes it.

**Assumption baked in here:** the 48-feature matrix is the canonical input to every tabular model. Adding a new feature means editing one file. Removing one risks breaking downstream IC tracking. The matrix is intentionally not versioned — if you change it, you invalidate historical IC logs.

### `cd0e090`, `ebbfc42` — model library scaffold
`ebbfc42` is enormous (+6,616 lines): it scaffolded the entire `models/statistical/`, `models/ml/`, `models/deep/`, and `features/` package trees. ARIMA, GARCH, Kalman, VAR/VECM, HMM, RF, XGBoost+SHAP, ElasticNet, BiLSTM, Prophet, TFT, plus the four feature groups (`macro_overlays`, `climate_weather`, `energy_transition`, `sentiment`). The README jumped from ~350 to ~600 lines to document it.

**Flaw introduced:** at this point every model independently loaded data and built its own feature matrix subset. Caching was inconsistent. This haunted us through Phase 3 — the meta-predictor needed a single source of truth, which only landed in Phase 4.

### `be9db97` — DB plumbing hardening
`pipeline/align_calendar.py` started enforcing the canonical US-trading calendar derived empirically from the futures themselves (≥50% of 28 direct futures present on a date). Forward-fill convention locked in here. `is_filled` flag added so downstream models can exclude synthetic rows if they want to (most don't — fwd-fill rate <0.2% per instrument so the cost is negligible).

---

## Phase 2 — Model library hardening (2026-04-27 → 2026-04-28)

### `130a04b` — VAR Granger overhaul
+479 lines in `pages/4_Models.py`, +74 in `var_vecm.py`. The VAR was rewritten to handle all commodity groups (Energy, Metals, Grains) with BIC-selected lag order and a p-value heatmap rendered in Plotly. Kalman pairs (`kalman.py` +221 lines) got proper state-space initialisation with `init_state_var` instead of the prior-arbitrary `1e3` default.

**Assumption to remember:** VAR Granger is *bivariate causality conditional on the group's joint history.* Don't read "WTI Granger-causes Copper, p=0.04" as a cross-sector finding — it's conditional on the Energy group's VAR. Cross-sector Granger requires a different specification we have not implemented.

### `9f49b1c` — config rationalisation
`models/config.py` got the first version of `MODELING_COMMODITIES` (the full 40-instrument universe) alongside `CORE_TICKERS` (the 11-commodity sidebar subset). **This split was the source of the May 12 audit incident — see Phase 6.**

### `02c7a96` — UI rewrite
`utils/theme.py` (+359) introduced the dark-mode design system: `apply_theme()`, `render_topbar()`, `PLOTLY_LAYOUT`, and the colour tokens (`SIGNAL`, `ASCEND`, `DESCEND`, `AMBER`, `VOID`, `DEPTH`, `ABYSS`, `ICE`, `ICE_MID`, `BORDER`) that every later page consumes. If you're styling a new component, **import from `utils.theme`** — do not redefine colours inline.

### `08ee962` — trigger primitives + data contract
First appearance of `features/trigger_detectors.py`, `models/broadcaster.py`, `models/router.py`, `models/triggers.py`, `models/model_signal.py`, and crucially `services/data_contract.py` (+298 lines).

`services/data_contract.py` is load-bearing: it standardises the dataclass shapes that flow between layers (`TriggerSignal`, `RouteDecision`, `ModelSignal`). Every subsequent component depends on it. **If you change a field, you break the wire format** between detectors, router, broadcaster, and downstream subscribers.

`Model_Integration_Roadmap.md` and `PHASE_1_Implementation_Guide.md` landed in this commit too — internal planning docs that the user has been working from. They're not authoritative now; treat the code as truth.

---

## Phase 3 — First live ecosystem (2026-04-28 → 2026-04-30)

### `41d548a` — the big one (+6,893 lines)
This is the single largest commit in the project's history. It introduced:

- `models/causal_chain.py` (+758) — first end-to-end orchestration: trigger → vol estimate → regime → ensemble → recommendation.
- `models/meta_predictor.py` (+643) — ensemble layer that consumes the 48-feature matrix and per-tier model predictions.
- `models/backtest_harness.py` (+657) — walk-forward backtester producing IC, hit-rate, drawdown.
- `models/daily_retrain.py` (+519) — first version of the retrain orchestrator (later rewritten in Phase 6 to support Optuna).
- `models/ic_tracker.py` (+435) — persistent IC logging per (commodity, model, horizon).
- `models/trigger_log.py` (+168) — append-only trigger event log to Postgres.
- `pages/6_Causal.py` (+535) — the first causal-chain UI.

Critically, **test coverage landed at the same time**: `test_backtest_harness.py`, `test_causal_chain.py`, `test_daily_retrain.py`, `test_ic_tracker.py`, `test_meta_predictor.py`, `test_trigger_log.py` (combined ~2,300 lines). The project went from "scripted notebooks" to "tested module library" in one commit.

**Architectural decision baked in here:** the causal chain is a *stateless function* — given a trigger payload, it produces a recommendation. State (IC scores, trigger history, regime labels) lives in Postgres, not in process memory. This is what made the later split into internal/external layers tractable.

**Flaw introduced:** `meta_predictor.py` originally hardcoded which tier-1/tier-2 models contributed to the ensemble. Adding a model meant editing the meta-predictor. This was partially fixed in `0254990` (Phase 6) but the coupling still exists — search for the `MODEL_TIERS` constant.

---

## Phase 4 — Signal quality (2026-05-02 → 2026-05-09)

### `0560f12` — threshold tuner
`models/threshold_tuner.py` (+645 lines, with 328 lines of tests) — sweeps decision thresholds per (commodity, signal) to maximise expected payoff under a configurable cost model. Outputs go to the `threshold_config` table. **This module is the difference between "the model has IC" and "the model produces tradable signals."**

### `00ce3af` — trigger detector refit + IC tracker hardening
`features/trigger_detectors.py` (+222) got real statistical thresholds (rolling z-score on returns, GARCH-residual breaks, regime-flip probability spikes) instead of the prior fixed cut-offs. `models/ic_tracker.py` (+172) added per-(commodity, model, horizon, regime) IC, not just per-(commodity, model). The regime conditioning matters — a model with IC=0.04 overall but IC=0.12 in Bear regimes is *different intelligence* than a uniformly-good model.

### `cb14fed` — cross-asset learning
`models/cross_asset.py` (+793) and `models/ml/sector_tuner.py` (+383) — sector-level models that train on the concatenated feature matrices of every commodity in a sector and predict at the sector level. The hypothesis: sector-level signal-to-noise should be higher than individual commodities because idiosyncratic noise averages out. The empirical IC supports this for Energy and Grains; less so for Metals.

**Assumption to verify:** sector models assume within-sector commodities share enough macro exposure that a single feature matrix can represent the sector. For Energy this is fine. For "Metals" (precious + base + battery) the assumption is shakier — Gold and Lithium do not share the same macro drivers. Watch the sector IC scores; if Metals stays below 0.03 long-term, the sector grouping in `models/config.py` needs splitting.

### `3b76925` — macro router + sector model + QAOA
+3,904 lines in one commit. The headliner is `models/macro_router.py` (+944): four macro variables (DXY, VIX 5-day, TLT, TLT-yield proxy) × five sectors × five regime bins = 100 OLS regressions, stored in `models/macro_routes.pkl`. The pickle is regenerated weekly by `daily_retrain.py`.

**Domain validation:** the router runs nine textbook-economic checks after each fit (DXY↑→Gold↓, VIX↑→Bitcoin↓, etc.). Two currently fail (DXY→Energy and Rates→Energy), both with R²<0.02 — i.e. the failures are inside the noise. The CLAUDE.md notes flag this as expected sample-window artefact (the 2022 Ukraine-invasion energy crisis lives in the rolling window). The system reports failures rather than masking them.

`models/cascade_orchestrator.py` (+597) is the engine that consumes the macro routes and produces a sector→commodity propagation graph. It runs in `pages/7_Macro_Market_Cascade.py`.

`models/quantum/qaoa_portfolio.py` (+475) landed here too — QUBO formulation with cardinality constraint, classical simulation via PennyLane. Note: it is **not** running on quantum hardware; the IBM Quantum backend is supported but not wired into the dashboard's default code path.

**Flaw to be aware of:** `macro_router.py` uses OLS on overlapping multi-day returns (e.g. VIX 5-day return). The standard errors are wrong (serially correlated residuals). The point estimates are still consistent, but don't trust the p-values inside the router. The dashboard never surfaces those p-values, so this is contained — but if you add a significance test, fix the standard errors first.

---

## Phase 5 — Causal-chain V1 → two-layer split (2026-05-09 → 2026-05-11)

### `dcce364`, `22e4b5f` — causal chain V1
`models/cascade_validator.py` (+664) and `models/backtest_harness.py` enhancements. `pages/6_Causal_Chains.py` (+970) — the prototype causal-chain UI. `pages/7_Portfolio.py` (+414) — QAOA visualisation.

**Architectural problem that emerged:** the original Page 6 tried to show *both* macro inputs *and* the internal model stack on one page. It became unmaintainable: too many concerns, too many state vars, too many cache invalidations. The user described the V1 as "maybe it's good?? ..." which, reading the code, is generous.

### `9a7ed59` — the split
This is the architecturally most important commit in the project:

```
pages/6_Causal.py         → pages/6_Causal_QS_Engine.py        (internal response)
pages/7_Causal_Chains.py  → pages/7_Macro_Market_Cascade.py    (external input)
pages/7_Portfolio.py      → pages/8_Portfolio.py
```

The causal chain was bisected:
- **Page 6 (QS Engine)** answers: given an event has fired, what does the model stack recommend? Pure internal response.
- **Page 7 (Cascade)** answers: what does the macro environment look like and how is it propagating into sectors? Pure external input.

The two layers communicate only through the trigger bus (Phase 6). Everything downstream — the alert engine, the macro exposure heatmap, the scenarios page — is only tractable because of this split.

**Read [README.md](README.md) for the user-facing framing of this split; here, the takeaway is the architectural decision: bisect on "input vs response," not on "macro vs micro."**

### `8934adb` — price validator + alert reporter
`pipeline/price_validator.py` (+579) and `pipeline/alert_reporter.py` (+418). This was triggered by the Rough Rice incident: Yahoo Finance briefly reported ZR=F at ~$17 on days when surrounding prices were ~$1,800 (unit mix-up at the data provider). Those rows were poisoning the feature matrix.

The fix: every ingestion run now z-scores incoming prices against a 252-day window and writes any |z|>5 row to `price_validation_log` with a severity tier. The roll-adjust pass already had this protection for the adjustment step itself; this extends it to *raw ingestion*. **Downstream models still see the raw row** — the validator is a flag, not a filter. If you build a model that should ignore validator-flagged rows, you need to join on `price_validation_log` yourself.

---

## Phase 6 — Polish & expansion (2026-05-13 → 2026-05-15)

### `0254990` — Optuna + MODELING_COMMODITIES audit
This is the commit that fixed a class of bugs introduced silently during Phase 4. **Every new model since Phase 2 had been hardcoding `CORE_TICKERS` (11 commodities) instead of `MODELING_COMMODITIES` (40 commodities).** Selectboxes showed the right tickers but the underlying training data was a 4× smaller subset.

The audit:
- Every model file now imports `from models.config import MODELING_COMMODITIES`.
- `pages/4_Models.py` defines `prices_full = load_prices_en()` as a 4-hour cached global; every model in the page consumes it.
- The standing rule was added to `CLAUDE.md` to prevent regression.

Optuna integration landed alongside: `models/threshold_tuner.py`, `models/ml/sector_tuner.py`, and `models/cascade_validator.py` got Optuna `TPESampler` hyperparameter search with persistent study storage. Tuning runs are checkpointed; you can resume a tuning study across retraining cycles.

**Flaw still present after this commit:** `CORE_TICKERS` is still referenced by `services/price_data.py` for the sidebar macro-trigger reader. If you delete `CORE_TICKERS`, that reader breaks. The rule in CLAUDE.md is "don't use `CORE_TICKERS` for new models," not "delete it."

### `02544e6` — scenario bands
Entire new module: `models/scenarios/` (+~1,800 lines) and `pages/9_Scenarios.py` (+990).

The scenarios architecture:
- `band.py` — converts return distributions to price-path bands.
- `aggregator.py` — `ScenarioAggregator` consumes per-model bands (ARIMA, VAR, ElasticNet, RF, XGBoost, LSTM, TFT, Prophet, quantum kernel) and produces a consensus envelope.
- `calibration.py` — empirical-coverage audit (nominal vs realised quantile coverage). **This is the most important file in the module.** A scenario fan chart that says "90% interval" but has 50% empirical coverage is worse than no chart at all.
- `conformal.py` — conformal-prediction wrapper for distribution-free intervals.
- `mc_dropout.py` — MC-Dropout for the deep models.
- `analogs.py` — historical-analog retrieval (`find_analogs` returns similar past windows).
- `narrative.py` — auto-generates plain-English summaries of the band.
- `ripple.py` — `CausalRipple` class that propagates a scenario through the causal chain.
- `providers.py` (+787) — per-model `*_band()` providers.

**Open issue (Phase 6 flaw):** calibration is computed but not enforced. The aggregator currently equal-weights all model bands. A natural next step is inverse-calibration-error weighting, which would penalise models whose nominal intervals overstate coverage. Hooks exist in `aggregator.py` but the weighting is not wired up.

### `c1ef81d` — cascade narrative + Page 6/7 expansion
`utils/macro_narrative.py` (+334) and `utils/narrative.py` (+137) — narrative generators that turn the cascade and QS engine outputs into prose. `pages/6_Causal_QS_Engine.py` (+776) and `pages/7_Macro_Market_Cascade.py` (+674) got their final visual shape here.

**Architectural note:** narratives are generated *from the same structured outputs* the charts consume. They are not separate model calls. If you change the narrative template, you don't need to refit anything; if you change the underlying data shape, both the chart and narrative must be updated together.

### `320f92b` — FRED + Alpha Vantage + trigger bus
+6,566 lines. Three independent things in one commit:

1. **FRED ingestion** (`features/fred_price_reference.py` +298) — pulls reference series (e.g. industrial production, oil inventories) for cross-validation against yfinance prices.
2. **Alpha Vantage news sentiment** (`features/av_news_sentiment.py` +496) — supplements FinBERT with Alpha Vantage's pre-scored news sentiment. Both feed `features/sentiment.py`.
3. **Trigger bus** (`services/trigger_bus.py` +947, `trigger_classifier.py` +500, `trigger_config.py` +438, `cascade_handlers.py` +568, `macro_ingestion.py` +679) — the production trigger pipeline. Triggers fire from detectors, are classified by severity, routed through `cascade_handlers`, and broadcast over WebSocket.

**Test coverage:** `test_trigger_bus.py` (+757), `test_trigger_classifier.py` (+524), `test_cascade_handlers.py` (+474). Good coverage. Run `pytest services/test_trigger_bus.py` after any change to the bus — there are race conditions that only show up under threaded load.

**Critical assumption:** the trigger bus runs in a background thread inside the Streamlit process. If Streamlit restarts (which it does on every code edit), in-flight triggers are lost. There is no durable queue. For a production deployment this would need a real broker (Redis Streams, NATS, etc.); for our single-user dashboard it's acceptable.

### `2dc0844` — components, event ribbon, exposure heatmap, alert engine
The final big commit (+8,918 lines). New page set:

- `pages/10_Event_Ribbon.py` (+375) + `components/event_ribbon.py` (+482) + `services/ws_broadcast.py` (+214) — WebSocket-driven horizontal ribbon of live trigger events.
- `pages/11_Macro_Exposure.py` (+247) + `components/macro_heatmap.py` (+536) + `components/commodity_cards.py` (+443) + `services/trigger_lifecycle.py` (+488) — trigger lifecycle manager; tracks which commodities are currently "elevated" by an active trigger.
- `pages/12_Alerts.py` (+520) + `components/notification_panel.py` (+434) + `services/alert_engine.py` (+550) — severity-tiered alert engine, rule management, session history.

`pages/4_Models.py` was massively refactored in this commit (+4,940 / −2,320). Most of that is UI tightening, but it also collapsed a lot of duplicated cache logic into shared helpers — if you're editing Page 4, **read the top of the file first** because the helper hierarchy is not obvious.

Integration tests landed too: `services/test_integration.py` (+1,451). This is the end-to-end test you should run before any change to the trigger bus, lifecycle, or alert engine: it boots the bus, fires synthetic triggers, and asserts they propagate through to alerts.

---

## Cross-cutting assumptions you should know

These are decisions that span the codebase. Violating them silently is how regressions happen.

1. **Adjusted close, always.** Every model trains on `aligned_prices.adjusted_close`. Never `close`. The roll-adjust pass removes ~95 fake price jumps; raw `close` would feed those into every momentum and z-score feature.
2. **`MODELING_COMMODITIES` for models, `CORE_TICKERS` for the sidebar.** See Phase 6 commit `0254990`. CLAUDE.md documents this as a mandatory rule.
3. **Feature matrix is canonical.** `models/features.py` builds the 48-column matrix. If you add a column, every tabular model picks it up automatically; if you change a column's semantics, every historical IC log becomes apples-to-oranges.
4. **State lives in Postgres, not memory.** The trigger bus, IC tracker, threshold config, alert engine all persist to Postgres. Streamlit restarts must not lose audit-able state.
5. **Cache TTLs are deliberate.** `@st.cache_data(ttl=300)` for prices (5 min), `ttl=600` for news, `ttl=14400` for the full price matrix in Page 4 (4 hours). If you change one, profile the page — some interactions trigger re-cascade and will get expensive.
6. **Narratives are derived, not generated independently.** Anything in `utils/narrative.py` or `utils/macro_narrative.py` is a deterministic template over structured data. No LLM calls in the runtime path.

---

## Known flaws and where they live

| Flaw | Where | Severity | Why we accept it |
|---|---|---|---|
| Macro-router OLS uses overlapping returns; standard errors are biased | `models/macro_router.py` | Low | Point estimates are fine; p-values are never surfaced |
| `meta_predictor.MODEL_TIERS` is a hardcoded constant | `models/meta_predictor.py` | Medium | Refactor pending; for now, adding a model to the ensemble needs a manual edit |
| Sector model groups Precious + Base + Battery into "Metals" | `models/config.py` | Medium | Sector IC for Metals stays low; split is on the backlog |
| Trigger bus is in-process; restarts lose in-flight triggers | `services/trigger_bus.py` | Medium | Single-user dashboard; no real consumers downstream |
| Scenario aggregator equal-weights models regardless of calibration | `models/scenarios/aggregator.py` | Medium | Calibration is measured but not used as a weight |
| Validator flags bad rows but does not filter them | `pipeline/price_validator.py` | Low | Intentional — keeps the raw log auditable. Models can join `price_validation_log` if they want filtering |
| `CORE_TICKERS` still exists; risk of accidental regression | `models/config.py`, `services/price_data.py` | Low | Documented in CLAUDE.md; audited 2026-05-12 |
| Domain-validation checks 4 and 7 fail in `macro_router` | `models/macro_router.py` | Low | Documented; R²<0.02 on both; expected to clear once 2022 falls out of rolling window in Q3 2026 |
| Cyclic edges absent (Ag→Energy biofuels, Livestock→Ag feed) | `models/cascade_orchestrator.py` | Medium | Backlogged; needs upstream pipeline restructure |
| Quantum models run as classical simulation by default | `models/quantum/*` | By design | IBM Quantum backend supported via `pennylane-qiskit` but not wired into the default UI path |

---

## How to extend safely

**Adding a model.**
1. Drop the file under the appropriate tier (`models/statistical/`, `models/ml/`, `models/deep/`, `models/quantum/`).
2. Consume `prices_full` / `returns_full` from `pages/4_Models.py` — do not re-load.
3. Use the 48-column feature matrix from `models/features.py`.
4. Add an IC tracker call so the daily retrain logs the model's IC.
5. If it should feed the meta-predictor, edit `MODEL_TIERS` in `models/meta_predictor.py`.
6. If it should contribute a scenario band, add a `*_band()` provider in `models/scenarios/providers.py`.

**Adding a trigger.**
1. Detector goes in `features/trigger_detectors.py`.
2. Register the trigger type in `config/trigger_registry.json`.
3. The classifier (`services/trigger_classifier.py`) will route it to severity tiers.
4. Add a handler in `services/cascade_handlers.py` if it should propagate through the cascade.
5. Add an alert rule in `services/alert_engine.py:DEFAULT_RULES` if it should notify.

**Adding a page.**
- Read `utils/theme.py` first. Use `apply_theme()`, `render_topbar()`, `PLOTLY_LAYOUT`, and the colour tokens.
- File naming: `pages/N_Page_Name.py` — N controls sidebar order.
- For any chart that depends on macro state, consume `services/trigger_lifecycle.LIFECYCLE` rather than recomputing.

---

## Reading the repo as a newcomer

Suggested reading order:

1. `README.md` — what it does.
2. This file — how it got built.
3. `CLAUDE.md` — the standing rules.
4. `models/config.py` — the universe definition.
5. `models/features.py` — the canonical feature matrix.
6. `models/causal_chain.py` + `models/cascade_orchestrator.py` — the two engine spines.
7. `services/data_contract.py` + `services/trigger_bus.py` — the wire format and the pipe.
8. `pages/4_Models.py` — the model UI patterns; every other page inherits from these conventions.

Run the integration test before your first real change:

```bash
pytest services/test_integration.py
```

If that passes, your environment is wired correctly.
