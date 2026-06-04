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

---

## 2026-06-02 — M2 term structure: contract-month cycles + roll calendar

**Context.** The first M2 ingest stored a single fixed dated contract per
commodity (e.g. `CLQ26.NYM`). Diagnosis on 2026-06-02 found two problems: (1) the
DB held only ~4 rows for 12 of 26 futures (an *incremental* run + yfinance
rate-limiting, not a Yahoo coverage issue — a direct probe returned 252 rows for
24 of 26 tickers); (2) a fixed dated contract is **not** a constant-maturity M2
over history — `CLQ26` (Aug 2026) is the genuine 2nd-nearby today but was ~M13 a
year ago, so `basis = log(front/CLQ26)` mislabels long-dated calendar spreads as
near-term carry. Also the front leg used roll-adjusted `adjusted_close` while the
M2 leg used raw `close` — mixing adjusted and raw distorts the spread.

**Decision.** Build a proper contract calendar that resolves the genuine M1/M2
listed contracts on each date and stitches a constant-2nd-nearby series from raw
adjacent legs. New files: `config/futures_calendar.py`,
`pipeline/contract_calendar.py`, `pipeline/test_contract_calendar.py`.

**What was verified (this entry).** The per-commodity listed-month cycles, which
drive every resolved ticker. Sources: CME Group and ICE contract specifications
via web search, 2026-06-02.

- **Web-confirmed directly:** COMEX Gold (Feb/Apr/Jun/Aug/Oct/Dec), Silver &
  Copper (Mar/May/Jul/Sep/Dec); **NYMEX Platinum (Jan/Apr/Jul/Oct) & Palladium
  (Mar/Jun/Sep/Dec)** — and crucially that Pt/Pd trade on **NYMEX (`.NYM`)**, not
  COMEX; the prior `.CMX` symbols 404'd. ICE Coffee & Cocoa (Mar/May/Jul/Sep/Dec),
  Cotton (Mar/May/Jul/Oct/Dec), Orange Juice (Jan/Mar/May/Jul/Sep/Nov); CME Lean
  Hogs (Feb/Apr/May/Jun/Jul/Aug/Oct/Dec); CBOT Soybean Meal
  (Jan/Mar/May/Jul/Aug/Sep/Oct/Dec).
- **Standard exchange cycles (textbook references; match every web-confirmed
  sibling on the same exchange):** NYMEX energy (all 12 months); CBOT Corn / SRW
  Wheat / KC Wheat / Oats (Mar/May/Jul/Sep/Dec), Soybeans (Jan/Mar/May/Jul/Aug/
  Sep/Nov), Soybean Oil (Jan/Mar/May/Jul/Aug/Sep/Oct/Dec), Rough Rice (Jan/Mar/
  May/Jul/Sep/Nov); CME Live Cattle (Feb/Apr/Jun/Aug/Oct/Dec), Feeder Cattle
  (Jan/Mar/Apr/May/Aug/Sep/Oct/Nov).

**Verdict.** ✅ Confirmed / corrected. The Pt/Pd exchange fix is verified against
source. Documented simplifications: (a) Pt/Pd list serial + quarterly months — we
use only the liquid *quarterly* cycle; (b) `roll_offset_bdays` is a per-group
approximation of expiry timing — exact roll day is not calibrated to volume/OI
(tunable in `config/futures_calendar.py`). These are acceptable because the basis
feature only needs M1/M2 to be genuinely *adjacent* listed contracts; the rolling
z-score absorbs the small roll-boundary discontinuity.

**Independent check (code).** `pipeline/test_contract_calendar.py` (10/10 passing)
confirms the resolver reproduces the real WTI curve: on 2026-06-02 it returns
front = `CLN26.NYM` (Jul), M2 = `CLQ26.NYM` (Aug), and correctly promotes M2→M1
at the roll boundary. All 26 specs resolve distinct, ordered M1/M2.

**Still pending (not yet verified empirically).** The stitched basis values and
their IC contribution — requires `pipeline/ingest_contracts.py` +
`pipeline/stitch_m2.py` to run on the Mac (network + Postgres), then the
prompt-4 retrain and prompt-5 IC comparison. No IC numbers are claimed yet.

---

## 2026-06-02 — Stitched M2 carry features: empirical run + coverage-gate finding

**Context.** First end-to-end Mac run of the constant-maturity stack
(`ingest_contracts` → `stitch_m2` → `daily_retrain --period 3y` → health report).

**What was verified (empirically, on live Postgres).**
- **Stitch correctness:** for the commodities that ingested (WTI/Brent/Gasoline),
  the stitcher produced genuine M1/M2 legs (e.g. WTI M1=7, M2=28; Brent M1=12,
  M2=47) from the real adjacent listed contracts. ✅
- **ML pipeline health restored:** after the coverage gate (`MIN_BASIS_COVERAGE`,
  see below), the retrain went from FAILED (ML tiers wiped → MetaPredictor
  degenerate-model failure) to SUCCESS with both tiers populated
  (ml: 2491 | statistical: 3659; tree leaves=7681; top feature
  `fedfunds_surprise_z`). ✅

**Verdict on the carry signal itself: ⚠️ INCONCLUSIVE — features are dormant.**
The stitched M2/M1-raw series are very shallow today (M2 ≈ 47 rows, M1-raw ≈ 14
rows vs a 749-row / 3y training window) because Yahoo only serves currently-listed
+ recently-expired dated contracts (older ones 404 as "delisted"). That is ~3–6%
coverage — far below the 0.50 coverage gate — so **the basis / basis_zscore /
roll_yield columns are intentionally dropped from the ML feature matrix** and do
NOT yet contribute. The post-run IC (ml avg ≈ −0.026, statistical ≈ −0.010) is
therefore the *base* feature set, not carry; no IC improvement is expected or
claimed at this depth. The Gorton–Hayashi–Rouwenhorst carry premium is
well-documented externally, but it cannot be realised here until the stitched
series accrues forward depth (months of daily observations). This is by design
(graceful degradation), not a defect.

**Code changes shipped this session (bug fixes surfaced by the run).**
1. **Coverage gate** — `models/config.py::MIN_BASIS_COVERAGE = 0.50`;
   `models/features.py::build_feature_matrix` drops term-structure columns whose
   non-NaN coverage over the training window is below the gate. Prevents shallow
   basis columns from making `feat.join(target).dropna()` wipe every row (the
   zero-sample regression). Regression test: `test_backtest_harness.py` #26.
2. **Ticker-based commodity resolution** — `config/futures_calendar.py` gains
   `ContractSpec.yf_ticker`; `pipeline/stitch_m2.py::resolve_commodity_id` joins
   on the stable unique `commodities.ticker` (the `name` column is seeded
   inconsistently). Fixes 12 "commodity not seeded" skips. Used by both
   `stitch_m2` and `ingest_contracts`.
3. **OutOfMemory / max_locks_per_transaction crash** — `ingest_contracts` now
   commits per contract. `ingest_commodity` wraps each row insert in a SAVEPOINT;
   PostgreSQL holds those subtransaction XID locks until the top-level commit, so
   backfilling 26×25 contracts in one transaction exhausted shared memory
   (crashed mid-Brent, starving the other 23 commodities). Per-contract commit
   caps held locks at ~one contract and preserves partial progress.
4. **Legacy fixed-contract M2 disabled** — `pipeline/ingest.py` no longer calls
   `_run_m2_ingestion` (it wrote economically-invalid fixed-contract rows to
   `1d_m2`, which would clobber the stitched series on every daily run).
5. **causal_monitoring_log tuple-key JSON crash** — `daily_retrain._jsonify_keys`
   stringifies `(sector, tier)` tuple keys before `json.dumps` so the row persists
   instead of being silently dropped.
6. **covariance_snapshots table** — added ORM model `CovarianceSnapshot` so
   `init_db()` creates the table the raw-SQL upsert in `cross_asset.py` targets.

**Still pending.** Re-run `ingest_contracts` (now that the OOM crash is fixed) so
all 26 futures populate their near-dated contracts and forward accrual begins for
the whole universe; the carry IC contribution remains to be measured once the
stitched depth crosses the coverage gate.

---

## 2026-06-03 — Phase 0: the out-of-sample evaluation gate + layer seams

**What was built.** The foundation of the dashboard restructure — the *gate* that
every future signal must pass before going live (North Star principle #1). This is
infrastructure, not a model promotion, but the first real signal was run through
it and its verdict is reported here per the rule.

New code (branch `feat/phase0-eval-gate`, behind no user-facing flag — it is
headless research infrastructure):
- `signals/base.py` — the single `Signal` interface every edge implements
  (`compute(asof, panel)`, mandatory non-empty `economic_rationale`, multi-horizon
  `{5,10,21}`) + a name registry.
- `signals/momentum.py` — `momentum_xs`, the first real signal: 12-1
  cross-sectional momentum (long winners / short losers), vol-scaled, z-scored.
- `evaluation/harness.py` — the gate: per-date cross-sectional Spearman IC,
  IC information ratio + t-stat on a **de-overlapped** subsample (dates spaced ≥ H
  apart, so overlapping H-day targets don't inflate significance), directional hit
  rate, **net-of-cost** long-short PnL (`evaluation/costs.py`, 10 bps/side),
  walk-forward fold sign-stability, and a PROMOTE/REJECT contract. Writes the
  `signal_scorecard` table and a human-readable diff vs the prior run.
- `evaluation/point_in_time.py` + `evaluation/test_point_in_time.py` — the
  anti-look-ahead property test: for every registered signal, `compute(asof=t)`
  must not change when future rows are appended. A deliberately-leaky fixture
  confirms the test bites.
- `signal_scorecard` table (`database/models.py::SignalScorecardRow`) — the
  append-only experiment ledger backing this log.
- `.importlinter` — layer-boundary contracts (signals must not import
  streamlit/pages/app/portfolio/evaluation; portfolio/evaluation must not import
  presentation). `lint-imports` → **3 kept, 0 broken**.

**Verification of `momentum_xs` against external sources.** Cross-sectional
commodity momentum is one of the most robustly documented factor premia
(Erb & Harvey 2006; Miffre & Rallis 2007; Asness, Moskowitz & Pedersen 2013 "Value
and Momentum Everywhere"). So the *economic rationale is confirmed* — this is not a
data-mined pattern. The question the gate answers is whether THIS construction, on
THIS 40-instrument universe, after costs, clears the bar.

**Verdict: ⚠️ REJECT (gate working as designed; result is honest, not a failure).**
On ~5y of daily data (1,580 daily IC obs):

| H  | OOS IC | IC IR | t-stat | hit | net LS Sharpe | fold sign-stability |
|----|--------|-------|--------|-----|---------------|---------------------|
| 5  | 0.032  | 0.136 | 2.42   | 51.0% | 0.38 | 5/5 positive |
| 10 | 0.038  | 0.160 | 2.01   | 51.3% | 0.43 | 5/5 positive |
| 21 | 0.040  | 0.191 | 1.67   | 51.5% | 0.47 | 4/5 positive |

The IC is **positive, sign-stable across every walk-forward fold, and the IC
t-stat exceeds 2 at H=5/10** — i.e. there is a real, weak edge, exactly as theory
predicts (and it strengthens with horizon, as momentum should). It is rejected
only because the IC **information ratio (0.14–0.19) is below the 0.30 promotion
bar**. This is the intended behaviour: a single weak signal is not promotable
alone. The North Star is breadth (Edge = IC × √breadth) — momentum_xs is a
*survivor to be combined*, not a standalone live model. It will re-enter the gate
as one input to the honest ensemble in Phase 4. No code was changed in response to
the verdict; the signal is recorded as a validated-but-not-yet-promotable edge.

**Reproduce:** `python -m evaluation.harness --signal momentum_xs --horizons 5,10,21`
(persists to `signal_scorecard`; add `--no-db` to skip). Tests:
`pytest signals/ evaluation/`. Boundaries: `lint-imports`.

**Pending (Phase 0 follow-ups, not blocking).** Wire `lint-imports` + `pytest` into
CI once a CI provider is chosen (deferred per Charles's call this session — local
runner only for now). The presentation layer is not yet a single package, so it is
enforced as a forbidden *target*, not a root package; the page-taxonomy move lands
in Phase 6.

---

## 2026-06-03 — Phase 1 fundamental ingestors: publication-lag approximations (DATA)

**What was verified.** The release-timing assumptions baked into the three new
free fundamental adapters (CFTC COT, EIA, USDA), since these drive point-in-time
correctness — a wrong release_date silently re-introduces look-ahead.

**Sources checked.**
- *CFTC COT.* CFTC publishes the Commitments-of-Traders report each **Friday at
  15:30 ET** for positions held the preceding **Tuesday** (cftc.gov COT release
  schedule). → 3-calendar-day lag. **Confirmed.** Caveat: federal-holiday weeks
  slip the release by ~1 day; the public Socrata dataset carries no publish
  timestamp, so the fixed +3d is a documented approximation.
- *EIA Weekly Petroleum Status Report.* Week ending Friday, released the following
  **Wednesday ~10:30 ET** (eia.gov release calendar). → ~5-day lag. **Confirmed.**
  *Weekly Natural Gas Storage Report* releases **Thursday** → 6-day lag, applied as
  a per-series override. **Confirmed.**
- *USDA WASDE / NASS QuickStats.* Reports release on scheduled dates well after the
  reference period; QuickStats does not expose a clean per-row first-publish date.
  A conservative **+30-day** lag is used pending wiring of the published WASDE
  release calendar. **Inconclusive / conservative approximation** (errs toward
  *later* visibility, which is the safe direction for anti-look-ahead).

**Verdict.** ✅ Confirmed for COT and EIA (lags match official schedules);
⚠️ conservative-approximation for USDA and for FRED (FredAdapter already documents
its ALFRED-vintage gap). All lags err toward *delaying* visibility, so they cannot
manufacture look-ahead — the failure mode is being slightly too cautious, never
too optimistic.

**What changed in code.** Added `data/adapters/{cftc,eia,usda}_adapter.py`
(release-dated `_shape` transforms, network best-effort), the
`services/{cot,eia,usda}_ingest.py` runners (idempotent, flag-gated by
`FUNDAMENTAL_FEEDS_ENABLED`, launchd-schedulable), and `data/validation.py` (the
portfolio-wide quality gate: staleness / outlier / calendar-gap / coverage) plus a
flagged Data-Health console on `pages/5_Database.py` (`DATA_HEALTH_ENABLED`).

**Follow-up (explicit task, not blocking).** Replace the fixed-lag approximations
with exact release calendars: FRED→ALFRED realtime vintages; USDA→published WASDE
calendar; COT→holiday-aware Friday. Tracked here so no signal silently assumes
more precision than we have.

**Reproduce / test.** `pytest data/` (adapter shaping + validation; network-free).
Boundaries: `lint-imports`.

---

## 2026-06-03 — Phase 1 live smoke test: CFTC/EIA validated, USDA corrected (DATA)

**What was verified.** A bounded live ingest of each free fundamental feed into
the real Postgres store, checking values, release-date math, the PIT invariant,
and idempotency.

**CFTC COT — ✅ confirmed.** WTI managed-money net (code 067651), 126 weekly rows
2024→present. Net long ~73k–98k contracts (specs structurally net-long crude —
correct magnitude). Release = Tuesday ref + 3d = Friday ✓. As-of a past date
hides later releases; re-run stayed at 126 rows (idempotent) ✓.

**EIA weekly stocks — ✅ confirmed (values), ⚠️ minor flag bug.** Crude ending
stocks 433.7M bbl, nat-gas working storage 2,483 Bcf — both match published
levels. Release lags correct (crude +5d, nat-gas +6d). **Bug:** the EIA v2
``/seriesid/`` endpoint ignores the ``start`` param, so the runner pulls full
history (1982→present) regardless of ``--start``. Harmless (idempotent, and full
history is desirable for backfill) but ``--start`` is misleading — follow-up to
move to the ``/v2/{route}/data`` endpoint with proper faceting.

**USDA QuickStats — ⚠️ refuted then corrected.** First run was wrong in two ways:
(1) every observation was stamped to ``Dec-31`` of its year, so the four quarterly
Grain Stocks reads (Mar1/Jun1/Sep1/Dec1) collapsed onto one date — 99 API records
deduped to 27, silently dropping 3 of every 4 quarters; (2) ``release_date`` was a
fabricated ``Dec-31 + 30d``. **Fix:** ``reference_date`` now anchors to the first
of the position month from ``end_code``; ``release_date`` now uses QuickStats'
real ``load_time`` publish timestamp (revision reloads only push visibility later
— safe for anti-look-ahead). Series renamed ``*_ENDING_STOCKS`` →
``*_GRAIN_STOCKS`` since these are quarterly stocks, not WASDE carryout (the Sep1
read ≈ marketing-year carryout). Stale rows purged and re-ingested: 99 rows, all
four quarters preserved; Corn Sep1 2025 = 1.55B bu (trough) vs Dec1 = 13.3B
(post-harvest) — textbook seasonality. **Verdict: corrected and confirmed.**

**What changed in code.** Rewrote ``data/adapters/usda_adapter.py`` date logic
(``_reference_date`` via end_code, new ``_release_date`` via load_time); renamed
``services/usda_ingest.py`` DEFAULT_QUERIES keys; added quarterly + load_time
adapter tests. Also added ``data.config.load_env()`` so the runners load ``.env``
under launchd (keys were otherwise invisible).

**Follow-ups (not blocking).** EIA ``--start`` faceting; USDA load_time is a load
(not strictly first-print) timestamp — fine and conservative, but a published
Grain Stocks release calendar would be exact.

**Reproduce.** ``FUNDAMENTAL_FEEDS_ENABLED=true python -m services.{cot,eia,usda}_ingest``
(needs EIA_API_KEY / USDA_QUICKSTATS_KEY in .env). Tests: ``pytest data/``.

---

## 2026-06-03 — Phase 2 wave-1 signals through the gate (trend_ts, carry_proxy, seasonality)

**What was verified.** Three new price-only Signal producers, each run through the
walk-forward / purged-embargoed / cost-adjusted IC gate
(`python -m evaluation.harness --signal NAME --horizons 5,10,21`). Promotion bar:
IC mean > 0, IC IR ≥ 0.30, fold-sign-frac ≥ 0.60, net LS Sharpe ≥ 0.

**trend_ts (time-series momentum, 12-1 vol-scaled) — ⚠️ inconclusive (right sign,
under the bar).** IC is positive and rising with horizon (H5 0.032, H10 0.037,
H21 0.040) with positive t-stats (2.40 / 1.98 / 1.67) and positive net LS returns
— exactly the direction Moskowitz-Ooi-Pedersen (2012) predict, and the horizon
profile (premium strengthening out to a month) matches the literature. But IC IR
tops out at 0.19 < 0.30, so it is too noisy on this 40-instrument universe/sample
to promote. **Verdict: economically confirmed, statistically not yet promotable —
held out of the ensemble.** Candidate for promotion once breadth grows or it is
combined with the cross-sectional book (the two trend forms are near-orthogonal).

**carry_proxy (mom10/vol21 risk-adjusted short momentum) — ❌ refuted as a proxy.**
The placeholder hypothesis (short-horizon front strength stands in for
backwardation) is wrong in this universe: IC is *negative* at every horizon
(−0.029 / −0.045 / −0.064), t-stats significantly negative, net LS Sharpe ≈ −1.1
to −1.4 with turnover > 1.0/period. Short-horizon momentum here *reverses*, and
costs make the short-leg book actively lose. This is a useful negative: it
confirms the proxy must NOT be trusted as carry — the signal stays gated as
inconclusive-by-construction exactly as its docstring warns, and the real edge
waits on the stitched-M2 term-structure basis (Erb-Harvey 2006,
Gorton-Rouwenhorst 2006, Koijen 2018). **Verdict: refuted as a carry stand-in; do
not promote; prioritise the true basis series.**

**seasonality (forward-window expected return from calendar-month means) — ⚠️
inconclusive (no cross-sectional edge).** IC ≈ 0 and weakly negative
(−0.006 / −0.009 / −0.011), t-stats |·| < 0.5 — indistinguishable from noise.
The physical rationale is sound (Sorensen 2002; energy seasonality), but a single
pooled monthly-mean read does not produce a *cross-sectionally* rankable edge net
of cost on this universe — likely because seasonal effects are
instrument-specific (nat-gas winter, gasoline summer, grains harvest) and cancel
when ranked against each other on the same calendar month. **Verdict: no promotable
cross-sectional edge as constructed; revisit as a per-instrument timing overlay
rather than a cross-sectional ranker.**

**What changed in code.** Added `signals/trend.py` (TimeSeriesMomentum / trend_ts),
`signals/carry.py` (CarryProxy / carry_proxy), `signals/seasonality.py`
(Seasonality); registered all three in `signals/base._ensure_signals_imported()`.
No promotions — all three correctly REJECTED by the gate and held out of the
ensemble. Scorecard rows persisted to `signal_scorecard`. The contract test and
the look-ahead property test both parametrize over `list_signals()`, so the three
new signals are covered automatically (`pytest signals/ evaluation/` green;
`lint-imports` 4/4 contracts kept).

**Takeaway.** The gate did its job on the first real wave: of three economically-
motivated candidates, zero cleared the bar, and each rejection is interpretable
(trend = real but noisy, carry-proxy = wrong-signed placeholder, seasonality =
not cross-sectional). This is the intended behaviour — the spine rejects honestly
rather than shipping near-random forecasts.

---

## 2026-06-04 — Phase 2 data-layer expansion (COT breadth, EIA labels, FRED feed)

**Context.** Wave-2's fundamental signals (COT positioning-reversal, macro-surprise,
real term-structure carry) were blocked not on signal code but on **data breadth**:
the point-in-time fundamental store held only 1 COT series (WTI), 4 EIA energy
series with no instrument labels, and zero FRED rows — far too narrow for a
cross-sectional gate (`min_cross_section = 5`). This entry records widening the
data layer so those signals become gate-scoreable next.

**What was verified — CFTC contract-market codes (against the live source).**
Every code in `services/cot_ingest.py::DEFAULT_SERIES` was checked on 2026-06-04
against the live CFTC Disaggregated Futures-Only catalog (Socrata resource
`72hh-3qpy`, fields `contract_market_name` / `market_and_exchange_names`, grouped
over 2025+ reports). **Finding (refuted prior map): the previous grain codes were
wrong.** The old map had `001602 → "Corn"`, but CFTC `001602` is **WHEAT-SRW**;
`002602` is CORN; `005602` is SOYBEANS; `007601` is SOYBEAN OIL. The old map also
used instrument *names* ("Gold", "Corn") that do not match the price-panel display
names ("Gold (COMEX)", "Corn (CBOT)"), so even the correct codes would not have
joined. Only WTI (`067651`) had ever actually been ingested, so the bad codes
never produced visibly wrong data — but they would have on the next run. **Verdict:
prior grain mapping refuted and corrected; 27 liquid futures now carry
source-verified codes mapped to exact `data.universe` display names.**

**What was verified — FRED publication lags (against release calendars).** FRED's
REST helper keys observations on the *reference* date, not the first-print date
(true vintages need ALFRED — a still-open upgrade, flagged in
`data/adapters/fred_adapter.py`). We approximate `release_date = reference_date +
lag_bdays`, rounded UP so we never lead the real print. **Caught a 1-day leak in
testing:** at a 31-bday lag, April-2024 CPI (ref `2024-04-01`) resolved to a
`2024-05-14` release, one day *before* the actual BLS print of `2024-05-15`.
Corrected to 34 bdays (CPI/PPI), 35 (INDPRO), 27 (employment); April-2024 CPI now
resolves to `2024-05-17` — 2 days *after* the real print (safe), and is correctly
invisible as-of `2024-05-14`. Verdict: monthly-macro PIT timing confirmed
non-leaking after the fix; daily market series carry a 1-bday lag.

**What changed in code.**
- `services/cot_ingest.py` — `DEFAULT_SERIES` rebuilt: 1 → **27** source-verified
  CFTC codes (energy/metals/grains/softs/livestock), values are exact universe
  display names; the ingest now persists the `instrument` column so cross-sectional
  COT signals join straight onto price-panel columns. ETF/index proxies and crypto
  (no managed-money line in this report) are intentionally omitted.
- `services/eia_ingest.py` — added `SERIES_TO_INSTRUMENT` (crude→WTI Crude Oil,
  gasoline→Gasoline (RBOB), distillate→Heating Oil, nat-gas→Natural Gas) and
  attaches `instrument` on write. Existing EIA rows backfilled in place via SQL
  (label-only update; values/dates untouched).
- `services/fred_ingest.py` — **new** release-dated macro feed: 12 series
  (CPIAUCSL, PPIACO, T10YIE, UNRATE, PAYEMS, INDPRO, DGS10, DGS2, T10Y2Y, DFF,
  VIXCLS, DTWEXBGS) with per-series publication lags.

**DB after expansion** (`fundamental_observations`): cftc 22,162 rows / 27 series
/ 27 instruments (was 1/1/0); eia 7,314 rows / 4 series / 4 instruments (was
4 series/0 instruments); fred 31,644 rows / 12 series (new); usda unchanged. All
27 COT instruments verified to join the live price panel (0 unmatched). Tests:
`pytest data/ services/` → 264 passed; `lint-imports` 4/4 contracts kept.

**Open items (flagged, not silently assumed).** (1) FRED/CFTC release dates remain
calendar approximations — ALFRED (FRED) and exact COT publish timestamps are the
vintage-truth upgrade. (2) EIA refetch needs `EIA_API_KEY` in the launchd env
(present in `.env`); this session backfilled labels by SQL because the key was not
loaded in the interactive shell. (3) No signal shipped here — this is data-layer
groundwork; the COT-reversal / inventory-surprise / macro-surprise signals are the
next wave and must each clear the gate on their own.

---

## 2026-06-04 — COT positioning signals: reversal REFUTED, risk-premium CONFIRMED (sub-threshold)

**What was verified.** The first fundamental (non-price) signal through the gate,
built on the newly-widened CFTC COT feed (27 instruments). Two pre-registered,
opposite-signed economic hypotheses were run on the SAME positioning z-score
(current net managed-money position vs its own ~3y trailing norm), so the gate
adjudicates which theory holds rather than us choosing by hand:
- `cot_reversal` (SIGN -1): COT-extreme contrarian view — crowded longs unwind.
- `cot_risk_premium` (SIGN +1): hedging-pressure risk premium — speculators are
  paid to absorb hedgers' risk (Cootner 1960; De Roon-Nijman-Veld 2000;
  Basu-Miffre 2013).

**Result (walk-forward, purged/embargoed, cost-adjusted IC; 27-instrument
cross-section).**
- `cot_reversal` — ❌ **refuted.** IC negative at every horizon (H5/H10/H21 =
  −0.025 / −0.037 / −0.032; t = −1.84 / −1.99 / −1.20), net LS Sharpe −0.31 to
  −0.50. The contrarian COT-extreme story does NOT hold in this universe.
- `cot_risk_premium` — ⚠️ **confirmed-but-sub-threshold.** Exact mirror: IC
  +0.025 / +0.037 / +0.032 (t = +1.84 / +1.99 / +1.20), hit-rate >50%, net LS
  Sharpe +0.09 / +0.30 / +0.33 — right-signed and cost-positive, but IC IR
  0.10–0.16 < 0.30 bar, so not promotable standalone. Held out; ensemble candidate
  (same class as trend_ts), strongest at the 10-day horizon.

**Verdict.** Positioning behaves as a **hedging-pressure risk premium, not a
reversal**, in our 27-instrument futures universe — go WITH stretched specs, not
against them. This matches the academic literature (Basu-Miffre 2013) over the
practitioner "COT-index contrarian" folklore. Sources: Cootner (1960),
De Roon-Nijman-Veld (2000), Basu-Miffre (2013); contra Sanders-Irwin-Merrin
(2009) on crowded-spec limits.

**Honesty caveat (in-sample sign selection).** Choosing the +z direction is
legitimate model selection between two ex-ante theories — NOT parameter torture —
but the sign was still confirmed on the same sample used to reject its mirror.
Because `cot_risk_premium` is sub-threshold it ships **held-out, not promoted**, so
no in-sample-selected edge trades live; promotion would require it to clear the
gate as part of the ensemble on a later/independent window.

**What changed in code.** Added `data/fundamental_store.load_raw()` (bulk
all-vintage loader so a signal evaluated at thousands of dates replays the PIT
filter in memory instead of one DB round-trip per date). Added `signals/cot.py`
with a shared `_CotPositioning` base and two registered signals (`cot_reversal`,
`cot_risk_premium`); registered the module in `signals/base`. The look-ahead
property test and contract test cover both automatically via `list_signals()`
(`pytest signals/ evaluation/` green; `lint-imports` 4/4). Scorecard rows persisted
to `signal_scorecard`.

---

## 2026-06-04 — Inventory-surprise signal: no cross-sectional edge as built (REJECTED)

**What was verified.** `inventory_surprise` — EIA weekly stocks deseasonalised by
week-of-year, forecast = -z(level vs trailing seasonal norm), i.e. surplus bearish
/ deficit bullish (theory of storage: Working 1949; Gorton-Hayashi-Rouwenhorst
2013). Scored on the EIA-covered energy sub-universe (crude→WTI, gasoline→RBOB,
distillate→Heating Oil, nat-gas→Natural Gas) with `--min-cross-section 4`.

**Result (walk-forward, purged/embargoed, cost-adjusted; 4-instrument
cross-section).** ❌ REJECTED at all horizons. IC ≈ 0 at H5 (+0.008, t=0.22),
mildly negative at H10/H21 (−0.033 / −0.087; t = −0.69 / −1.25); net LS Sharpe
negative throughout (−0.42 / −0.29 / −0.32). The near-zero short-horizon IC and
weak t-stats indicate NOISE, not a clean wrong sign — flipping would not rescue
it (H5 stays ~0), so unlike cot_reversal no risk-premium variant is warranted.

**Verdict — inconclusive / not cross-sectionally rankable as built; theory of
storage NOT refuted.** Two structural reasons, both economic rather than coding:
(1) **Degenerate cross-section** — the 4 EIA instruments are one tightly
co-moving energy complex (dominated by the common crude factor), so ranking them
against each other each week is close to meaningless; the theory-of-storage edge
in the literature is cross-sectional across DOZENS of commodities or time-series
per instrument, neither of which a 4-name energy book tests. (2) **News
absorption** — the weekly EIA print moves prices on release day; by the next daily
decision date the surprise is largely in the price, leaving little for a 5–21 day
cross-sectional book. This mirrors the `seasonality` finding: economically sound,
not a cross-sectional ranker in this form.

**Revisit path (not done now).** (a) a per-instrument TIME-SERIES overlay (long
each energy contract when its OWN inventory is tight) evaluated with a time-series
harness; (b) re-test cross-sectionally once inventory breadth spans many more
commodities (LME metals stocks, USDA/CONAB ags), so the cross-section is genuinely
diverse. Until then it ships rejected and is held out of the ensemble.

**What changed in code.** Added `signals/inventory.py` (`InventorySurprise`),
registered in `signals/base`. Added a general `--min-cross-section` flag to
`evaluation/harness.py` (default unchanged at 5) so sub-universe signals are
scoreable without weakening the default gate. Tests `pytest signals/ evaluation/`
green (22 passed); `lint-imports` 4/4. Scorecard rows persisted.

---

## 2026-06-04 — Macro-surprise (per-instrument betas): betas CONFIRMED, alpha REJECTED

**What was verified — the betas (MODEL VERIFICATION RULE).** Before gating, the
learned trailing factor betas were checked against economic priors at 2026-05-29
(252-day window, factors T10YIE/DGS10/DTWEXBGS/VIXCLS):
- Gold: USD −0.0050, 10y rates −0.0014, inflation +0.0009 — textbook (gold rises
  as the dollar/real rates fall; mild inflation hedge).
- Silver: USD −0.0123 (even more dollar-sensitive than gold). ✓
- Copper: USD −0.0067, VIX −0.0027 (industrial metal; risk-off hurts it). ✓
- WTI: inflation +0.0107 (strong — oil is an inflation driver). ✓
- Nat gas: near-zero macro betas (correctly identified as weather-driven). ✓
**Verdict: the factor-beta structure is economically sound** — the model captures
real macro transmission, not noise. (Two small off-signs — crude's mildly positive
USD/VIX beta — are explainable by the recent oil-as-geopolitical-risk regime.)

**Result of the alpha signal (walk-forward, purged/embargoed, cost-adjusted; full
40-instrument cross-section).** ❌ REJECTED at all horizons. IC negative at H5/H10
(−0.022 / −0.023; t = −1.31 / −0.99) and only weakly positive at H21 (+0.027,
t=0.75, not significant). Net LS Sharpe −0.79 / −0.58 / +0.15, with very high
turnover (0.83 / 1.11 / 1.30 per period).

**Verdict — macro betas are real, but "beta × recent move" is not standalone
alpha.** Two interpretable failures: (1) the construction uses CONTEMPORANEOUS
betas (r_t on f_t) applied to a recent factor move that the cross-section has
largely already repriced — hence the negative short-horizon IC (a reversal/
already-in-the-price effect dominates the lagged-diffusion hypothesis). (2) The
recent-move "surprise" is noisy day-to-day, churning the book (turnover >1.0/
period) so even the weakly right-signed H21 edge is eaten by costs. The economic
diffusion hypothesis is at best faintly present at 21 days and not significant.

**Product insight + revisit path (not done now).** The verified beta structure is
genuinely valuable, but its home is the **risk/covariance layer** (portfolio
construction, factor-hedging, regime conditioning) — NOT as a cross-sectional
alpha. As an alpha, the principled re-tests are: (a) LAGGED betas (regress r_t on
f_{t-k}) to test prediction rather than contemporaneous co-movement; (b) heavy
smoothing of the factor surprise to cut turnover; (c) feed the betas to the risk
layer and let macro shape sizing/hedging instead of direction. No passing variant
was manufactured — flipping the sign does not yield a clean win (only H5/H10 would
turn positive, still sub-0.30 IC IR and still turnover-killed).

**What changed in code.** Added `signals/macro.py` (`MacroSurprise`): vectorised
multivariate OLS betas for the whole cross-section at once (one K×K solve per date),
PIT factor panel from the FRED store (daily unrevised series stamped by
release_date). Registered in `signals/base`. Tests `pytest signals/ evaluation/`
green; `lint-imports` 4/4. Scorecard rows persisted.

---

## 2026-06-04 — ensemble_v1: composite lifts significance, still sub-threshold (REJECTED)

**What was built.** `ensemble_v1` — a deliberately parameter-free equal-weight
composite of the right-signed, gate-confirmed-but-sub-threshold edges. For each
horizon it standardises each component's cross-sectional forecast, averages across
the components covering each instrument, and re-standardises. No weights are fitted
(fitting them on the same history the gate scores would be the in-sample
optimisation this rebuild exists to avoid).

**Finding — two of the three candidates were the SAME signal.** Measured mean
cross-sectional rank-correlation of the component forecasts (H10, 41 sampled
dates): `momentum_xs` vs `trend_ts` = **+1.000**; each vs `cot_risk_premium` =
+0.59. The perfect correlation is correct-by-construction: both momentum signals
are the identical 12-1 vol-scaled trailing return, differing only in that
`momentum_xs` cross-sectionally demeans — and demeaning does not change RANKS, so
under the gate's rank-IC + dollar-neutral book they are indistinguishable. (Their
standalone scorecards were already near-identical.) The time-series vs
cross-sectional distinction only matters for net-directional exposure, which the
dollar-neutral gate does not reward. `trend_ts` was therefore dropped from the
composite to avoid silently double-weighting momentum; the distinct edges are
`momentum_xs` + `cot_risk_premium`.

**Result (walk-forward, purged/embargoed, cost-adjusted; momentum_xs +
cot_risk_premium, equal weight).** Still REJECTED, but the best composite so far:
IC 0.029 / 0.039 / 0.041, IC IR 0.131 / 0.182 / 0.208, t-stat 2.20 / 2.29 / 1.81,
net LS Sharpe 0.33 / 0.46 / 0.46, turnover 0.20-0.42. Versus the best standalone
(momentum IC IR 0.191 at H21), the ensemble nudges H21 IR to 0.208 and — more
importantly — pushes the short-horizon t-stats to significance (>2). But IC IR
remains < 0.30 at every horizon.

**Verdict — combining two positively-correlated edges is not enough breadth.**
With effective breadth ≈ 2 and component correlation 0.59, the IR lift is modest
(0.19 → 0.21). Clearing 0.30 requires genuinely ORTHOGONAL components, not more
momentum. The clear path: build the price-only edges deferred when we chose to
expand the data layer first — short-term reversal (≈ anti-momentum at short
horizon; empirically motivated by the carry_proxy negative-IC finding) and
low-volatility / betting-against-vol (Frazzini-Pedersen 2014) — both near-orthogonal
to momentum, then re-form the ensemble. Weight-optimisation (IC/risk-weighting) is
deferred until it can be justified on an independent window.

**What changed in code.** Added `signals/ensemble.py` (`EnsembleComposite`,
`ensemble_v1`), registered in `signals/base`; restricts output to the panel
universe (a component such as COT may score instruments beyond the given panel).
Tests `pytest signals/ evaluation/` green (26 passed incl. the look-ahead property
test over the ensemble); `lint-imports` 4/4. Scorecard rows persisted.

---

## 2026-06-04 — Orthogonal wave: reversal_st (strong), low_vol (null), ensemble_v1 rebuilt

**What was built.** Two price-only edges intended to add breadth orthogonal to
momentum: `reversal_st` (1-month short-term reversal, -z of the trailing 21-day
vol-scaled return; Jegadeesh 1990, Lehmann 1990) and `low_vol` (betting-against-vol,
-z of trailing realised volatility; Frazzini-Pedersen 2014).

**Standalone gate results (walk-forward, purged/embargoed, cost-adjusted; full
universe).**
- `reversal_st` — ⚠️ right-signed, sub-threshold, STRONG at H10. IC 0.017 / 0.049 /
  0.034; IC IR 0.075 / 0.208 / 0.144; t-stat 1.33 / 2.63 / 1.26; net LS Sharpe
  0.17 / 0.53 / 0.26. The H10 IC (0.049, t=2.63) is the best single-signal IC in
  the project so far. High turnover (0.70-1.43) as expected for a reversal. REJECTED
  standalone but a clear ensemble candidate. Note this confirms the carry_proxy
  finding (short-horizon momentum reverses) with a sign-correct factor.
- `low_vol` — ❌ NULL. IC ≈ 0 at every horizon (0.007 / 0.001 / 0.009; |t| < 0.5)
  and net LS Sharpe NEGATIVE (-0.50 / -0.39 / -0.27). The low-risk anomaly does not
  rank this commodity cross-section: it is an equity-centric effect, and here a vol
  ranking is largely a static low-vol-ETF vs high-vol (BTC/nat-gas) tilt that lost
  over the sample. Not an ensemble candidate (orthogonal, but orthogonal NOISE).
  Verdict: low-vol anomaly not present cross-sectionally in this universe.

**Orthogonality check (mean cross-sectional rank-corr, H10).** momentum/cot +0.59,
momentum/reversal **-0.10**, cot/reversal **-0.28**, low_vol vs all ≈ 0. reversal_st
is genuinely orthogonal (indeed anti-correlated) — exactly the breadth the ensemble
needed; low_vol is orthogonal but null.

**ensemble_v1 rebuilt (momentum_xs + cot_risk_premium + reversal_st, equal
weight).** Best composite to date, still REJECTED: IC 0.034 / 0.055 / 0.050; IC IR
0.159 / **0.253** / 0.241; t-stat 2.82 / **3.19** / 2.10; net LS Sharpe 0.41 /
**0.71** / 0.53; turnover 0.50 / 0.71 / 1.01. Versus the 2-component composite, H10
IC IR rose 0.182 → 0.253 and net LS Sharpe 0.46 → 0.71 (Δ+0.25) — the
anti-correlated reversal added real diversification. t-stats are now strongly
significant (>3 at H10).

**Verdict — methodology validated, bar not yet cleared.** IC IR has tracked
0.19 → 0.21 → 0.25 (H10) as genuinely orthogonal edges are added; the gate's
0.30 IC IR bar remains unmet but is now close, with a t-stat > 3 and a
cost-adjusted LS Sharpe of 0.71 at the 10-day horizon. The bar was NOT moved.
Path to promotion: one or two more orthogonal edges (e.g. a true term-structure
carry/basis once the stitched-M2 series exists; a commodity value / long-run
mean-reversion factor), then re-form. Weight-optimisation remains deferred until it
can be justified out-of-sample.

**What changed in code.** Added `signals/reversal.py` (`reversal_st`) and
`signals/lowvol.py` (`low_vol`), registered in `signals/base`; added `reversal_st`
to `ensemble_v1.COMPONENTS` (low_vol excluded as a null). Tests `pytest signals/
evaluation/` green (27 in the PIT/contract set); `lint-imports` 4/4. Scorecards
persisted.

---

## 2026-06-04 — Value factor: wrong-signed on a single-regime sample (REJECTED, not added)

**What was built.** `value` — commodity long-horizon mean reversion (Asness-
Moskowitz-Pedersen 2013): forecast = +z(reference log-price − current log-price),
reference = mean log-price ~1.4-2.75y back. Cheap-vs-own-history → long. Intended
as the orthogonal value/momentum complement to push the ensemble over the bar.

**History constraint.** The aligned panel has only ~5y of COMMON history (start
gated by the youngest instruments) and is calendar-day aligned, so a textbook 5y
reference is infeasible; the reference is ~1.4-2.75y back — the deepest the data
supports.

**Result (walk-forward, purged/embargoed, cost-adjusted; full universe).** ❌
WRONG-SIGNED. IC NEGATIVE at all horizons (−0.034 / −0.030 / −0.015; t = −1.91 /
−1.17 / −0.38), net LS Sharpe negative throughout. Over this sample, cheap-vs-
multi-year UNDERperformed: multi-year trend persisted rather than reverted.

**Why no sign-flip / not added to the ensemble.** Value is strongly NEGATIVELY
correlated with momentum (measured H10 rank-corr −0.654 — the canonical value/
momentum relationship). So a sign-flipped "value" (+IC) would be ~+0.65 correlated
with momentum_xs — i.e. just long-horizon MOMENTUM, redundant with a signal we
already have, NOT a new orthogonal edge. Flipping would add no breadth, so the
ensemble is unchanged (momentum_xs + cot_risk_premium + reversal_st).

**Verdict — inconclusive on the merits; AMP value NOT refuted.** The ~5-year panel
is a single, strongly-trending commodity regime (2021-2026), and value is well
known to underperform in trending regimes (cf. equity value's 2010s drawdown).
A multi-decade, multi-regime factor cannot be fairly judged on 5 years of one
regime. Honest read: value is wrong-signed IN THIS SAMPLE, which is itself
consistent with a trend-dominated regime, not evidence the factor is dead. Revisit
once the panel's common history deepens to span multiple regimes (and ideally with
the canonical ~5y reference).

**What changed in code.** Added `signals/value.py` (`value`), registered in
`signals/base`. NOT added to `ensemble_v1`. Tests `pytest signals/ evaluation/`
green; `lint-imports` 4/4. Scorecard rows persisted.

---

## 2026-06-04 — Phase 2 conclusion: research ensemble shipped behind a flag (NOT promoted)

**Decision.** Stop adding signals for now and accept the composite as the honest
Phase-2 research output. Wire it into the dashboard as a research-grade, NOT-promoted
surface behind a feature flag, and move on. (No goalposts moved; the 0.30 IC IR bar
stands and remains unmet.)

**Phase-2 signal scorecard (all gate verdicts, full universe unless noted).**
| signal | best horizon | IC IR | verdict | note |
|---|---|---|---|---|
| momentum_xs | H21 | 0.191 | reject | right-signed, sub-threshold |
| trend_ts | H21 | 0.191 | reject | == momentum_xs under ranking (corr +1.000) |
| carry_proxy | — | <0 | reject | wrong-signed proxy; needs real basis |
| seasonality | — | ~0 | reject | not cross-sectional as built |
| cot_reversal | — | <0 | reject | reversal refuted |
| cot_risk_premium | H10 | 0.158 | reject | right-signed (hedging pressure) — KEEP |
| inventory_surprise | — | ~0 | reject | degenerate 4-name energy cross-section |
| macro_surprise | — | <0/~0 | reject | betas real → belong in risk layer |
| reversal_st | H10 | 0.208 | reject | right-signed, strong — KEEP |
| low_vol | — | ~0 | reject | low-risk anomaly absent here |
| value | — | <0 | reject | wrong-signed on single-regime 5y sample |
| **ensemble_v1** | **H10** | **0.253** | **reject** | **best honest result** |

**ensemble_v1** = equal-weight(momentum_xs, cot_risk_premium, reversal_st), the
three right-signed, mutually-distinct edges. Best at H10: IC 0.055, IC IR 0.253,
t-stat 3.19, cost-adjusted net LS Sharpe 0.71 — significant and economically
meaningful, but below the 0.30 IC IR promotion bar. The IR climbed 0.19 → 0.21 →
0.25 as orthogonal breadth was added, validating the Edge = IC × √breadth approach;
it simply has not crossed the bar yet.

**What shipped (Dashboard Evolution Rule compliant).**
- `evaluation/reporting.py` — read-only, presentation-agnostic helpers (latest
  scorecard; current ensemble cross-sectional tilts), defensive/empty-on-failure.
- `pages/13_Signal_Lab.py` — new ADDITIVE page gated by `SIGNAL_RESEARCH_ENABLED`
  (default OFF). Loud "RESEARCH-GRADE — NOT PROMOTED" banner; renders the gate
  scorecard and the current dollar-neutral ensemble tilts. Thin: all computation
  stays in the signal/eval layers. No existing page or model was modified or
  replaced.
- `utils/theme.py` — sidebar lists "Signal Lab ⚗️" only when the flag is on.

**Rollback.** `unset SIGNAL_RESEARCH_ENABLED` (or set false) removes the surface
entirely — no redeploy. Old paths untouched.

**Open path to promotion (future).** The two most promising orthogonal edges remain
data-blocked or regime-blocked: a true term-structure carry/basis (needs the
stitched-M2 deferred-contract series — a data-layer build) and a multi-regime value
test (needs the aligned panel's common history to deepen past one commodity-bull
regime). macro betas should be deployed in the risk/covariance layer rather than as
alpha. Component weighting remains equal-weight until an out-of-sample (nested
walk-forward) scheme can justify otherwise.

---

## 2026-06-04 — Multi-regime value: FIRST GATE PASS (PROMOTE on deep panel) + regime findings

**Context.** The 5-year aligned panel rejected `value` (wrong-signed). Hypothesis:
that sample is a single trend-dominated commodity-bull regime, which is exactly
when value underperforms — not evidence the factor is dead. To test fairly we
backfilled ~24y of deep history for the core futures and re-gated.

**What was built.** `services/deep_history_ingest.py` backfills 25 core genuine
futures (~21 trading-years, 2001→2026) from Yahoo into `price_history` under a
DISTINCT `interval='1d_deep'` (production `'1d'` untouched). `load_long_history_core_panel()`
reads it; `evaluation/harness.py` gains `--panel aligned|long_core` (default
aligned). Multi-regime runs use `--no-db` (the ledger has no panel column yet), so
the machine scorecard stays aligned-panel-only; results recorded here.

**Result on long_core (21y, 25 instruments, walk-forward / purged / cost-adjusted).**
- `value` — ✅ **PROMOTE** (project's first). H5 IC 0.036 (IR 0.131, t 3.86);
  H10 IC 0.059 (IR 0.219, t 4.55, LS Sharpe 1.26); **H21 IC 0.098 (IR 0.348,
  t 4.99, net LS Sharpe 1.18) — clears the 0.30 bar.** Strengthens with horizon,
  exactly as a slow mean-reversion factor should.
- `momentum_xs` — ❌ now NEGATIVE IC (−0.004 / −0.010 / −0.023) on the true 21y
  sample (note: on long_core's trading-day index, 252 rows = a true 12-1 window).
  Cross-sectional commodity momentum is weak/regime-dependent over the full cycle —
  its positive 5y reading was regime-specific.
- `reversal_st` — ❌ right-signed and robust across regimes (IC 0.021 / 0.037 /
  0.044; IR up to 0.173; t 2.5-3.0), consistent with the 5y result.

**External verification (MODEL VERIFICATION RULE) — CONFIRMED.** The multi-regime
value result matches the literature: Asness-Moskowitz-Pedersen (2013, "Value and
Momentum Everywhere") document a significant commodity value premium and a strong
NEGATIVE value/momentum correlation. We measure value vs momentum rank-corr −0.65
and value t≈5 at the monthly horizon — consistent in sign, magnitude and the
value/momentum diversification. Verdict: confirmed; value is a genuine,
economically-grounded, gate-clearing edge over a proper multi-regime sample.

**Regime insight.** value and momentum are two sides of one coin: value was in
drawdown over 2021-26 (trend regime) precisely when momentum worked, and vice
versa over the full cycle. This is the textbook case FOR combining them — neither is
reliable alone, but their −0.65 correlation makes the pair powerful across regimes.

**Caveats (do not over-claim a live promotion).**
1. **Panel.** value PROMOTES on the research long_core panel (25 core futures, 21y)
   but is REJECTED on the production aligned panel (41 instruments, 5y) — because
   the production sample is value's drawdown regime. It must NOT be naively added to
   the current production ensemble, where it is presently adverse.
2. **Series construction.** `1d_deep` is RAW Yahoo continuous front-month, NOT
   roll-adjusted (unlike the production pipeline). Multi-year value uses price
   ratios, which carry some roll noise; the result should be re-confirmed on a
   cleanly roll-adjusted (or spot/index) deep series before any live deployment.
3. **Promotion is at H21 only**, on a deep research universe; treat as validated-in-
   research, not yet wired live.

**Path forward.** (a) Re-gate the ENSEMBLE on long_core (value + reversal + cot
post-2010) to see whether the value/momentum/positioning blend clears the bar
across a full cycle. (b) Add a `panel` column to `signal_scorecard` so multi-regime
runs are first-class in the machine ledger. (c) Roll-adjust the deep series and
re-confirm. (d) For production, treat value as a regime-diversifier to be combined,
not a standalone live signal yet.

**What changed in code.** `services/deep_history_ingest.py` (new),
`models/data_loader.py::load_long_history_core_panel` (new),
`evaluation/harness.py` `--panel` flag. Tests `pytest signals/ evaluation/` green
(32); `lint-imports` 4/4. 157,944 `1d_deep` rows ingested; production `1d`
unaffected.
