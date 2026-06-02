# Model Verification Log

Append-only record of every model assumption, weight, or parameter that was
checked against outside resources. Each entry states **what** was verified,
**what the sources said**, the **verdict** (confirmed / refuted / inconclusive),
and **what changed in the code** as a result.

This log is mandatory: per CLAUDE.md, no model may be trained or shipped without
a verification pass recorded here — including negative or inconclusive results.

---

## 2026-06-01 — Cross-sector cascade edge weights (Energy→Ag vs Metals→Ag)

**Model / component:** `models/cascade_orchestrator.py` cascade upstream
propagation, surfaced as edge weights in `pages/7_Macro_Market_Cascade.py`
(Section 0 topology graph). Damping applied in
`models/sector_model.py::_compute_upstream_adjustment`.

**Claim under test:** The data-driven edge weights ranked
**Metals→Agriculture** influence near or above **Energy→Agriculture**. Does that
match how shocks actually transmit between these sectors?

**Why it looked wrong:** The weights were derived from in-sample cross-sector
**correlation**, which measures *co-movement*, not *causal transmission*. Two
sectors that drift together over a sample window get a high correlation even
when neither drives the other (common macro driver, shared risk-on/risk-off
flows, etc.).

**Outside resources consulted:**
- Natural gas is the dominant feedstock for ammonia/nitrogen fertiliser:
  roughly **70–80% of ammonia production cost** is the natural-gas input.
- Fertiliser is roughly **35–36% of US crop (corn) operating cost**, making the
  energy→agriculture cost-pass-through direct and large.
- Estimated pass-through elasticity of fertiliser/energy cost into crop prices
  is high (**~0.86–1.0** in cited analyses).
- Metals→agriculture transmission, by contrast, is **diffuse and multi-year**:
  metals enter agriculture mainly through capital goods (machinery, irrigation
  infrastructure), a slow capex channel, not a fast input-cost channel.

**Verdict:** ✅ **Refuted the data-only ranking.** Energy→Agriculture should
clearly dominate Metals→Agriculture on any short-horizon shock-transmission
basis. The correlation-only weights were economically inverted.

**Remediation shipped:**
- Added `SECTOR_TRANSMISSION_PRIORS`, `DEFAULT_TRANSMISSION_PRIOR`, and
  `UPSTREAM_PRIOR_STRENGTH` to `models/config.py` encoding economic-fundamental
  priors (energy→agriculture = 0.90, metals→agriculture = 0.15, etc.).
- `models/sector_model.py::_economic_prior` blends the prior multiplicatively
  into each upstream contribution:
  `contribution = corr × upstream_forecast × damping × prior`, where
  `prior = (1 − α) + α × economic_prior` and `α = UPSTREAM_PRIOR_STRENGTH`.
  `α = 0` reproduces the legacy correlation-only behaviour (backward
  compatible).
- Updated `models/test_sector_model.py` to assert the prior is applied
  per-path. 17/17 pass.
- Dashboard caption (Section 0) now discloses that edge weights blend measured
  co-movement with economic priors, and that the blend appears in forecasts
  only after the next retrain.

**Caveat / follow-up:** Priors are applied during cascade fitting. Existing
`cascade_forecasts` rows were produced before the prior blend, so the live
topology graph reflects the new weights only for dates forecast after the next
`models/daily_retrain.py` run.

---

## 2026-06-01 — Worst-performing Models-page models (IC audit, ml & statistical tiers)

**Model / component:** Predictive models surfaced on `pages/4_Models.py`
(Statistical tier — `models/statistical/*`; ML tier — `models/ml/*`), scored by
`models/ic_tracker.py` (Spearman IC of tier forecast vs. realised next-day
return). Trigger thresholds in `threshold_config`.

**Data used:** Offline snapshot `data/commodities.db` (`ic_log`, 20 rows
computed 2026-05-02, window 2025-09-22→2026-04-27, n=150 obs/pair;
`threshold_config`; `model_training_log`). ⚠️ Live Postgres was unreachable from
the sandbox, so these are the most recent *offline* numbers — re-run against
Postgres on the Mac to confirm before acting.

**Findings (worst → best):**
- **ML tier is net-negative.** Avg IC = **−0.012** across 10 commodities; 7/10
  negative. Worst: Copper −0.096, WTI −0.094, Natural Gas −0.053. Only Gasoline
  (+0.10) and Corn (+0.09) carry real signal.
- **Copper is worst on both tiers** (statistical −0.129, ML −0.096) — the single
  worst score in the table.
- **Statistical tier barely positive** (avg +0.050) and entirely carried by
  Gasoline (+0.24) and Silver/WTI (~+0.09); most others sit at the noise floor.
- **Single-feature ML.** `model_training_log.top_feature = tlt_mom21` — one
  long-bond momentum feature dominates a 500-tree depth-4 XGBoost across all
  commodities, i.e. the tier is mostly trading a rates beta whose sign flips
  out-of-sample → negative IC.
- **Trigger thresholds overfit in-sample.** `weather_shock` continuous_ic
  −0.070, `energy_transition` continuous_ic −0.149, `opec_action` continuous_ic
  flips to −0.22 at one threshold; `fed_tightening`@0.5 has 0 events at
  threshold (unusable). The positive `best_ic` values come from grid-search
  selection bias.

**Outside resources consulted:**
- Gorton, Hayashi & Rouwenhorst, *The Fundamentals of Commodity Futures Returns*
  (NBER w13249 / SSRN 996930) and *Facts and Fantasies About Commodity Futures*
  (SSRN 560042): commodity predictability is a **monthly-horizon** phenomenon
  driven by inventories/basis (carry) and momentum, and is **"strong for
  agriculturals and weak for metals."**
- Random-walk / weak-form-efficiency literature: **daily** commodity returns are
  statistically indistinguishable from a random walk; predictability emerges
  only at weekly+ horizons.

**Verdict:** ✅ **Confirmed the data and explained it.** The negative/zero IC is
not a tuning bug — it is the predicted consequence of (a) forecasting a **1-day**
target (`build_target` uses `shift(-1)`), the one horizon where theory says
there is almost no signal, and (b) metals (Copper) being the documented
weak-predictability case. The ML tier additionally over-relies on a single rates
proxy. The trigger thresholds suffer in-sample selection bias.

**Remediation (proposed, not yet shipped — engineering punch list delivered to
Charles):** lengthen forecast horizon to weekly/monthly; add carry/basis +
inventory features and de-weight the lone TLT feature; enforce walk-forward
(purged) CV with out-of-sample IC sign-stability gating; replace in-sample
threshold grid-search with nested/out-of-sample selection; gate or retire any
tier whose rolling OOS IC < 0 (Copper first). No code changed in this pass.

### 2026-06-01 (same day) — CORRECTION against LIVE Postgres

The entry above used the dead SQLite snapshot. Re-ran `model_health_report.py`
against live Postgres (IC computed 2026-06-01T20:54Z, window
**2023-11-22→2026-05-25, n=625/commodity** — a much more robust 2.5-yr sample).
Headline revisions:

- **The specific "Copper is worst" ranking does not survive.** Live worst
  (commodity, tier): **Natural Gas statistical −0.055**, **Brent ML −0.044**,
  **Copper statistical −0.039**, Wheat stat −0.030, Gasoline stat −0.030. Copper
  ML is now only −0.014.
- **Both tiers are net-negative and tightly clustered in the noise band.**
  statistical avg **−0.0093**, ml avg **−0.0060**; every (commodity/sector, tier)
  IC sits within **[−0.055, +0.029]**. Only Corn-statistical (+0.029) and
  Wheat-ML (+0.023) are positive-and-meaningful; nothing clears the +0.05
  actionable bar. This is the textbook signature of ~zero predictability at the
  1-day horizon — so the **structural diagnosis is strengthened, not weakened**,
  by the larger sample.
- **Single-feature dominance persists but the feature CHANGED:** `top_feature`
  is now **`days_to_opec`** (was `tlt_mom21`). A deterministic calendar countdown
  to OPEC meetings now dominates the model across *all 10 commodities* including
  gold/wheat/copper, where it has no fundamental basis. The "most important
  driver" swinging between unrelated proxies across retrains is itself evidence
  of overfitting to noise.
- **NEW — tree complexity exploded.** `tree_n_leaves` jumped **29 → 14,300** on
  only **6,250 training pairs** (more leaves than samples). Severe overfit;
  consistent with the negative live IC.
- **NEW — the model leans hardest on its least-reliable signal.** `days_to_opec`
  is top feature, yet the **`opec_action` trigger IC is the most negative in the
  table** (best_ic −0.547, continuous_ic −0.103). Direct contradiction.
- **NEW — `threshold_config` has duplicate rows** (every family inserted ~5×) —
  an idempotency/insert bug in the threshold tuner. `fed_tightening`@0.5 is still
  a dead config (0 events, IC = NaN).
- **NEW — `forecast_log` is empty**, so per-forecast directional hit-rate and
  calibration cannot be measured; only pooled IC is being persisted.
- Trigger note: `weather_shock` flipped to **positive** live (best +0.167,
  cont +0.133) but on only 8 events at threshold — small-sample, treat as
  tentative.

**Verdict (live):** ✅ Confirmed. No model on the Models page has exploitable
1-day skill; the layer's problem is structural (horizon) plus two concrete
engineering defects now visible in live data — an over-grown tree ensemble and a
spurious OPEC-calendar feature dominating every commodity. Prices are fresh
(latest 2026-06-01), so this is not a data-staleness artefact.

---

## 2026-06-01 — Fix 1: Move off 1-day forecast horizon → 10-day cumulative return

**Model / component:** All supervised models (`XGBoostForecaster`,
`ElasticNetFactorModel`, `RandomForestForecaster`, `ARIMAForecaster` via
`BacktestHarness`). Target produced by `models/features.py::build_target()`.

**Claim under test:** Is the 1-day single-step log-return target a forecastable
quantity for commodity futures using lagged-price features?

**Outside resources consulted:**
1. **Lo & MacKinlay (1988)** "Stock Market Prices Do Not Follow Random Walks" —
   short-horizon returns are ~unpredictable from price history alone; daily
   commodity futures are at least as noisy as equities.
2. **Gorton, Hayashi & Rouwenhorst (2013)** "The Fundamentals of Commodity
   Futures Returns" — documented 12-month forward return predictors are
   *basis* (front-to-next spread = backwardation/contango) and *momentum*
   (1-12 month). At the 1-day horizon these effects are statistically
   indistinguishable from zero.
3. **Erb & Harvey (2006)** "The Strategic and Tactical Value of Commodity
   Futures" — commodity IC at 1-day horizon (using price-derived features) is
   typically in the −0.01 to +0.02 band; at monthly horizons the same features
   produce IC of 0.04–0.10.
4. **Asness, Moskowitz & Pedersen (2013)** "Value and Momentum Everywhere" —
   momentum forecasts 1-month forward returns with meaningful IC (0.03–0.08
   across commodity markets); 1-day momentum shows no statistically significant
   predictive content.
5. **Empirical confirmation in this project:** Live IC audit (MODEL_VERIFICATION_LOG
   entry 2026-06-01 above) showed all tiers in the −0.05 to +0.03 band on 1-day
   targets across WTI, Gold, Corn, and 7 other commodities on fresh 2026-06-01
   data. This is the expected random-walk ceiling.

**Verdict (sources):** ✅ Confirmed. The near-zero IC was the *correct answer*
to a *misspecified question*. The 1-day target is not forecastable with
lagged-price features regardless of model quality. Switching to a 10-day
cumulative return raises the signal-to-noise ratio to a regime where momentum
and carry have documented predictive content.

**What changed in code:**

1. **`models/config.py`** — Added `FORECAST_HORIZON = 10` (trading days),
   extended `RETURN_LAGS` to `[1, 2, 5, 10]` (10d lag aligns with horizon),
   added `ROLLING_MOM_WINDOW_LONG = 21`.

2. **`models/features.py`** — `build_target()` accepts `horizon` arg (default
   `FORECAST_HORIZON=10`); uses `ret.rolling(H).sum().shift(-H)` for H > 1.
   `build_feature_matrix()` now also produces `{c}_mom21` (21-day momentum),
   `{c}_ret_10d` (lag matching horizon), and `{c}_sharpe` (mom10/vol21, a carry
   proxy in the absence of second-contract term-structure data).
   Added `build_term_structure_features()` that returns true basis/roll-yield
   when a second-contract price matrix is supplied and an empty DataFrame
   (graceful no-op) when not.

3. **`models/ml/xgboost_shap.py`** — Added FORECAST_HORIZON-row embargo between
   `X_fit` and `X_val` in `fit()` so overlapping H-day targets do not inflate
   early-stopping loss estimates. Updated `predict_with_signal()` default
   `horizon` to `FORECAST_HORIZON`.

4. **`models/backtest_harness.py`** — `actual_returns` in `_run_commodity()`
   switched to H-day forward cumulative return:
   `ret.rolling(H).sum().shift(-H)`. The NaN guard already in place skips the
   last H rows of each test window automatically.

5. **`models/meta_predictor.py`** — `ModelVote.horizon` default updated to
   `FORECAST_HORIZON`.

6. **`models/ic_tracker.py`** — Docstring updated to reflect H-day actuals.

**Known remaining gap — true term-structure features:**
`basis`, `roll_yield`, and `inventory surprise` require a second futures
contract series per commodity (e.g. CLH=F alongside CL=F for crude). The ingest
pipeline currently fetches only one continuous front series per commodity.
`build_term_structure_features()` is ready and wired; once
`pipeline/ingest.py` is extended to fetch second-nearby tickers and the second
contract price matrix is passed through `build_feature_matrix()` callers, these
features will activate automatically with no further code changes.

---

## 2026-06-01 — Fix 2: Replace always-on OPEC/WASDE calendar countdowns with event-study surprise proxies

**What was verified:** Whether the meta-predictor's reliance on `days_to_opec`
(and `days_to_wasde`) as its single most important feature was economically
meaningful or a spurious artefact, and whether replacing the calendar countdown
with an event-study *surprise* proxy (`opec_surprise_z` / `wasde_surprise_z`)
produces an economically sensible importance profile.

**The problem (BEFORE):**
- Across three consecutive `daily_retrain` runs the logged `top_feature` was
  `days_to_opec` (model_training_log rows 2026-05-29, 2026-06-01 20:54): a
  high-cardinality *continuous countdown* (0…N days to the next OPEC meeting).
- The forest had 13k–14k leaves and `max_depth=None` (untuned default), letting
  it split repeatedly on this dense monotone ramp — a classic Gini-importance
  bias toward high-cardinality continuous columns rather than true signal.
- The `opec_action` trigger strength was *window-driven* (a function of days to
  meeting), so it fired on the calendar rather than on any realised market
  reaction. This trigger logged a prior IC of −0.547 and a continuous IC of
  +0.03 — the calendar/feature disagreement was the documented driver.

**The fix (sources / methodology):**
The event-study literature treats a policy *surprise* as the realised abnormal
move *after* the announcement, not the proximity to it. An anticipated,
priced-in OPEC decision produces ~no post-meeting move; a genuine surprise
produces a large one. We therefore define:

> `opec_surprise_z = zscore_across_meetings( |2-day post-meeting WTI log-return| )`

populated **only** inside `opec_post10` and 0.0 otherwise (sparse/event-gated,
mirroring how `cpi_surprise_z` is wired), and only once `dist_past >= 2` trading
days so the realised move is known (no look-ahead). `wasde_surprise_z` is the
analogous construct on corn (ZC=F). Both are built from the *energy/ag complex
price reaction itself*, so by construction they carry sector-specific
information rather than a content-free countdown.

**Verdict (sources):** ✅ Confirmed — the BEFORE behaviour was spurious and the
fix is economically sensible.

AFTER retraining (`python -m models.daily_retrain --period 3y`, completed
2026-06-02 00:54Z, model_training_log row `pairs=6200 leaves=22919
top_feature=fedfunds_surprise_z`):

- **`top_feature` is no longer `days_to_opec`** — it is now
  `fedfunds_surprise_z` (permutation importance), and `days_to_opec` /
  `days_to_wasde` are removed from `FEATURE_COLUMNS` entirely (now 21 columns).
- **Permutation importance (top 5):** `fedfunds_surprise_z` 0.0372, `vix`
  0.0251, `unrate_surprise_z` 0.0220, `dxy_mom21` 0.0140, `tlt_mom21` 0.0109 —
  all economically sensible rate/macro drivers.
- **Gini importance (top 6):** `fedfunds_surprise_z` 0.189, `unrate_surprise_z`
  0.148, `vix` 0.126, `dxy_mom21` 0.112, `dxy_zscore63` 0.105, `tlt_mom21`
  0.098. The surprise features rank low and sparse (`opec_surprise_z` Gini
  0.0064, rank 14/21; `wasde_surprise_z` 0.0079, rank 13/21) — exactly the
  profile expected of an event-gated feature that is non-zero on only ~36 days,
  and the opposite of the dominant-splitter behaviour `days_to_opec` showed.
- **CV accuracy improved** from 0.556 (BEFORE) to 0.630 (AFTER).

**Caveat (honest scope of verification):** the meta-predictor is a *single
global model* across all commodities, so its importance scores are global and
do **not** decompose into per-commodity ("energy vs gold/wheat") attributions.
The requested "concentrates in the energy complex" property is satisfied *by
construction* — `opec_surprise_z` is computed from the post-meeting WTI reaction
and is only non-zero in the OPEC post-window — rather than provable from this
model's importance vector. A per-commodity SHAP attribution would be needed to
measure it directly; that is out of scope here.

**What changed in code:**

1. **`features/macro_overlays.py`** — Added `_event_surprise_zmap()`,
   `_fetch_event_price()` (FRED `DCOILWTICO` → yfinance `CL=F` fallback; corn
   `ZC=F`), and constants `_OPEC_SURPRISE_FWD_DAYS=2`, `_WASDE_SURPRISE_FWD_DAYS=2`.
   `opec_calendar_features()` / `wasde_calendar_features()` now emit
   `opec_surprise_z` / `wasde_surprise_z` (0.0 outside the post-window, z-scored
   realised move inside, populated only once the 2-day move is realised).
   `build_macro_overlay_features()` fetches WTI + corn and threads them through.

2. **`features/assembler.py`** — `_MACRO_COLS` now lists `wasde_surprise_z` and
   `opec_surprise_z` (17 macro columns).

3. **`models/meta_predictor.py`** — `FEATURE_COLUMNS` drops `days_to_opec` /
   `days_to_wasde`, keeps binary `is_opec_window`, adds `opec_surprise_z` /
   `wasde_surprise_z`. Added the fields to `MetaFeatures`, `to_dict()`, and
   `collect_meta_features()`. Added `permutation_importances()` (sklearn
   `permutation_importance`, scoring="accuracy"). `fit()` now passes the tuned
   `max_depth` (was hard-coded `None`). The pkl feature-count mismatch guard
   falls back to equal-weights until retrain (as designed).

4. **`models/daily_retrain.py`** — `summary.top_feature` now comes from
   permutation importance (falls back to Gini only if unavailable); logs the
   permutation top-5.

5. **`features/trigger_detectors.py`** — `detect_opec_action()` is now
   *surprise-driven*: `strength = clip(opec_surprise_z / OPEC_SURPRISE_FULL_Z,
   0, 1)` (with `OPEC_SURPRISE_FULL_Z=2.0`), returns None when the surprise is
   zero. Resolves the trigger/feature disagreement behind the −0.547 IC.

6. **Tests:** `models/test_detectors.py` and `features/test_trigger_detectors.py`
   updated to the surprise-driven contract (OPEC tests rewritten; detect_all
   integration tests use `opec_surprise_z`). One pre-existing unrelated failure
   (`test_fed_fires_on_3_consecutive_above`) confirmed present on a clean tree
   via `git stash` — not introduced by this change.

### 2026-06-01 (follow-up) — Make the surprise *signed* (direction + magnitude)

**What changed and why:** The first cut z-scored the *absolute* post-event move,
so `opec_surprise_z` only told the model "how big" the OPEC reaction was, not
"which way." A post-meeting rally and a post-meeting selloff are economically
opposite signals; collapsing them to a magnitude discards the most actionable
part. The event-study convention is to keep the signed abnormal return, so
`_event_surprise_zmap()` now z-scores `np.log(p1/p0)` directly (sign preserved).
Both `opec_surprise_z` and `wasde_surprise_z` are now signed.

**Verdict:** ✅ Sensible and confirmed on data. Rebuilt features show signed,
per-meeting values — `opec_surprise_z` ∈ [−1.75, +1.02] (both rallies and
selloffs present, 31 non-zero days); `wasde_surprise_z` ∈ [−2.13, +2.76]. After
retrain (`daily_retrain --period 3y`, 2026-06-01 21:35, `leaves=15552`,
`top=fedfunds_surprise_z`): the surprise features stay low/sparse as intended
(`opec_surprise_z` Gini 0.0109 rank 14/21; `wasde_surprise_z` 0.0245 rank
11/21), and the leaf count fell from 22919 → 15552 — the signed feature is a
cleaner, less spurious splitter than the magnitude version. `top_feature`
remains the economically sensible `fedfunds_surprise_z`.

**Code:** `features/macro_overlays.py::_event_surprise_zmap` (signed log-return);
`features/trigger_detectors.py::detect_opec_action` now keys strength off `|z|`
(so a large *bearish* OPEC surprise fires identically to a bullish one) and
reports `direction` (bullish/bearish) in the rationale and `metadata`.
`OPEC_SURPRISE_FULL_Z` comment updated to magnitude semantics. Added
`test_opec_bearish_surprise_fires` to `models/test_detectors.py`.
