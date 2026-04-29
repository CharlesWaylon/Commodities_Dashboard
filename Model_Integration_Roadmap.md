# Quantum Speculation Engine (QS Engine) — Model Integration Roadmap

## Executive Summary

Your QS Engine currently operates as **five parallel model tiers** (statistical, ML, deep learning, quantum, market signals) that each produce predictions independently. The next phase is to synthesize these into an **integrated decision ecosystem** where:

1. **Earlier models feed forward** — Statistical models provide regime anchors; ML adds pattern recognition; Deep Learning captures long-range dependencies
2. **Signal routing by confidence** — Each model broadcasts its confidence level; downstream models weight their inputs accordingly  
3. **Causal chains surface actionable insights** — When a macro signal (ENSO, DXY stress) triggers regime shifts, the system traces this through all commodity groups systematically
4. **Meta-predictor arbitrates conflicts** — When models disagree, the meta-layer learns which model to trust under which market conditions

---

## Current Architecture Analysis

### What's Working Well ✅

**Data Foundation**
- 41 commodities tracked daily (5 years history)
- 48+ engineered features (price momentum, volatility, cross-commodity spreads)
- 15 external signals (macro, climate, sentiment, energy transition)
- Clean, roll-adjusted futures prices + calendar-aligned data

**Model Coverage**
- Tier 1 (Statistical): ARIMA (trend), GARCH (volatility), Kalman (pairs), VAR (causality) — all backward-looking signal extraction
- Tier 2 (ML): HMM (regime), XGBoost+SHAP (daily forecasting with interpretability), Random Forest (structural features), ElasticNet (sparse factors)
- Tier 3 (Deep): LSTM (sequential), Prophet (trend+seasonality decomposition), TFT (multi-horizon quantile)
- Tier 4 (Quantum): Kernel benchmarks, QAOA portfolio optimization, hybrid QNN
- Plus: Market signals layer (DXY, VIX, TLT, WASDE, OPEC, ENSO, energy transition spreads)

**Key Metric**
- Spearman IC (Information Coefficient) as the universal accuracy standard — IC > 0.05 is actionable

---

### The Integration Gap 🔗

**Models run independently**
```
Price data → Statistical models → IC score → Dashboard card ❌ No feedback
         → ML models       → IC score → Dashboard card ❌ Isolated
         → Deep models     → IC score → Dashboard card ❌ No cross-talk
         → Quantum models  → IC score → Dashboard card ❌ Parallel
         → Market signals  → Metrics  → Dashboard card ❌ Siloed
```

**No information flow between tiers**
- A strong regime signal from HMM doesn't inform ARIMA's confidence bands
- When ENSO shifts, nobody routes this through energy commodities specifically
- XGBoost's top bullish driver (e.g., "Copper z-score") isn't fed back to a Copper-specific ensemble
- Disagreement between statistical and ML models isn't analyzed — it's just shown side-by-side

**Missing: the causal chain**
- Macro event (e.g., OPEC+ production cut announcement) should cascade through: DXY stress → energy regime shift → downstream commodity correlations → portfolio rebalancing signal
- Currently: each model sees the data independently and produces its own forecast with no traceability

---

## Three-Layer Integration Architecture

### Layer 1: Signal Pipeline (Feeds Confidence Forward)

**Goal:** Each model outputs not just a point prediction but a **confidence tuple** that downstream models consume.

```python
# Example output from a single model
class ModelSignal:
    commodity: str
    forecast_return: float              # next-day return
    confidence: float                   # IC or equivalent
    horizon: int                        # 1, 5, or 20 days
    regime: str                         # 'bullish', 'neutral', 'bearish'
    reasoning: List[str]                # which features / events drove this
    uncertainty_band: Tuple[float, float]  # confidence interval
    depends_on: List[str]               # which other signals it needs
    model_type: str                     # 'statistical', 'ml', 'deep', etc.
```

**Key Insight:** Statistical models (ARIMA, GARCH) are **regime detectors**. ML models are **pattern matchers**. Deep models are **trend followers**. Quantum models are **optimization solvers**.

Organize by **primary use case**:
- **Regime detection** → HMM (unsupervised 4-state) + VAR Granger causality (which sectors drive each other)
- **Next-day directional signal** → XGBoost (highest IC on financial series) as the anchor forecast
- **Volatility envelope** → GARCH (forecasted vol band) + LSTM (scenario generators)
- **Longer-horizon trend** → Prophet decomposition (trend direction) + TFT (multi-horizon quantiles)
- **Portfolio optimization** → QAOA (cardinality-constrained allocation) seeded by consensus bullish/bearish list

---

### Layer 2: The Cascade System (Routes Signals Sector-by-Sector)

**Goal:** When a **macro trigger fires**, systematically propagate it through all affected commodity groups.

#### The Four Macro Trigger Families

**Trigger 1: Energy Policy (OPEC+, US reserve releases)**
- Signal enters via: calendar dummy + sentiment scan
- Affected commodities: WTI, Brent, Natural Gas, Gasoline, Heating Oil, Carbon Credits
- Cascade:
  1. Statistical models boost GARCH vol forecast for 2 weeks
  2. ML models shift regime probabilities (bear → high-vol)
  3. Deep models update trend slope (Prophet changepoint)
  4. If high IC consensus forms: feed to portfolio optimizer

**Trigger 2: Macro Monetary (Fed rate hike / currency shock)**
- Signal enters via: DXY z-score spike + TLT momentum flip
- Affected commodities: Gold (inverse USD), Copper (industrial sensitivity), All precious metals
- Cascade:
  1. Kalman hedge ratios shift (Gold ↔ Silver beta changes)
  2. ARIMA confidence bands widen (higher regime uncertainty)
  3. Elastic Net reweights: which factors matter under tightening?
  4. If persistent (>3 weeks): trigger multi-tier retrain

**Trigger 3: Climate / Supply Shock (drought, freeze, geopolitical)**
- Signal enters via: PDSI / HDD deviation + sentiment spikes
- Affected commodities: Wheat, Corn, Soybeans, Coffee, Orange Juice, Cotton
- Cascade:
  1. HMM regime flip to "High-Vol" (official signal)
  2. GARCH asymmetry (γ) increases (bad news > good news in uncertainty)
  3. LSTM attention weights shift to short-horizon features
  4. Elastic Net sparse factors: what's the minimal set that drives this shock?

**Trigger 4: Energy Transition (renewable capacity online, battery demand spike)**
- Signal enters via: Battery Metals PC1 + Uranium spread + Carbon vol
- Affected commodities: Lithium, Rare Earths, Uranium, Coal (thermal + met), Natural Gas, Gold
- Cascade:
  1. Regime detector: is this a structural level-shift or temporary?
  2. VAR Granger: does Lithium now Granger-cause more sectors?
  3. Elastic Net: sparse factor model — is it just one factor or multi-dimensional?
  4. Prophet: inject a changepoint (this shift may be permanent)

#### Implementation: A Signal Router

```python
class SignalRouter:
    """Routes macro triggers through affected commodity groups."""
    
    def __init__(self):
        self.triggers = {
            'opec_meeting': ['WTI', 'Brent', 'NG', 'Gasoline', 'HO', 'KRBN'],
            'fed_tightening': ['Gold', 'Silver', 'Copper', 'USD-sensitive'],
            'weather_shock': ['Corn', 'Wheat', 'Soybeans', 'Coffee', 'OJ', 'Cotton'],
            'energy_transition': ['Lithium', 'Rare Earths', 'Uranium', 'Coal', 'NG', 'Gold'],
        }
    
    def route(self, trigger: str, signal_strength: float) -> Dict[str, float]:
        """
        Given a trigger and its strength (0-1), return a reweighting
        for each commodity's confidence multiplier.
        
        Example:
          'opec_meeting' at 0.8 strength → WTI conf *= 1.3, NG conf *= 1.1, etc.
        """
        multipliers = {c: 1.0 for c in ALL_COMMODITIES}
        
        if signal_strength > 0.5:
            for affected in self.triggers[trigger]:
                multipliers[affected] *= (1.0 + 0.5 * signal_strength)
        
        return multipliers
```

---

### Layer 3: The Meta-Predictor (Arbitrating Model Disagreement)

**Goal:** Build a **meta-model** that learns when to trust which tier.

When XGBoost (ML) forecasts +0.5% but LSTM (Deep) forecasts −0.3%, the meta-predictor learns:
- "During Risk-Off regimes (VIX > 25), trust LSTM over XGBoost 70% of the time"
- "When VAR says sector is in causal shock, trust statistical vol models over ML"
- "On WASDE days (first 3 days post-event), HMM regime signal is worth +0.15 IC boost"

#### Meta-Model Architecture

```python
class MetaPredictor:
    """
    Meta-learner that predicts which model's forecast will be most accurate
    given the current market state.
    
    Inputs:
      - Point forecasts from all 5 tiers
      - Each model's stated confidence (IC)
      - Market regime (HMM state)
      - Macro trigger flags (OPEC, WASDE, Fed meeting, etc.)
      - VIX level, DXY stress, TLT momentum
      - Days since last regime transition
      - Sector-specific volume/sentiment scores
    
    Output:
      - Weighted ensemble forecast
      - Recommendation: which model to display / trust most
      - Uncertainty estimate (weighted IC)
    """
    
    def predict(self, context: MarketContext, model_outputs: Dict) -> EnsembleSignal:
        # Decision tree learned via backtesting:
        # IF vix > 25 AND regime == 'bear':
        #     weight[statistical] *= 1.2   (vol models reliable in chaos)
        #     weight[ml] *= 0.8            (pattern matching breaks down)
        # ELIF regime == 'bull' AND trend_momentum > +2:
        #     weight[deep] *= 1.3          (trend followers shine)
        # ... etc
        
        return ensemble_forecast(weights, model_outputs)
```

---

## Actionable Implementation Roadmap

### Phase 1: Foundation (Week 1–2) — Signal Standardization

**Objective:** Every model emits a standardized signal.

**Tasks:**
1. **Create `SignalBroadcaster` base class**
   - All models inherit from it
   - Enforces output schema: `commodity`, `forecast`, `confidence`, `horizon`, `reasoning`
   - Timestamp and metadata auto-attached

2. **Refactor each model to emit signals, not just DataFrames**
   ```python
   # Before (current):
   xgb_result = xgb.predict(features)  # returns array of forecasts
   
   # After (standardized):
   signal = xgb_forecaster.predict_with_signal(features)
   # → ModelSignal(commodity='WTI', forecast=0.012, confidence=0.087, 
   #               reasoning=['momentum_5d', 'copper_zscore'], ...)
   ```

3. **Add confidence estimates to models that lack them**
   - LSTM: use dropout as uncertainty (already do this implicitly)
   - Prophet: use built-in prediction intervals
   - TFT: already outputs quantiles — convert to IC proxy
   - QAOA: use solution landscape flatness as confidence (if landscape flat = multiple optima = low confidence)

4. **Version the signal schema** 
   - Today's schema is v0.1
   - When you add new fields, this stays backward-compatible

**Deliverable:** All models in `4_Models.py` emit `ModelSignal` objects instead of raw arrays.

---

### Phase 2: Cascade & Routing (Week 2–3) — Macro Integration

**Objective:** Macro triggers automatically flow through affected commodity groups.

**Tasks:**
1. **Build the `SignalRouter`** (from above)
   - Hardcode the 4 trigger families and their affected commodities
   - Attach to every daily run

2. **Implement trigger detection**
   - OPEC: calendar-based (hardcoded dates)
   - WASDE: calendar-based (2nd Tuesday monthly)
   - Fed tightening: DXY z-score > 2.0 for 3+ consecutive days
   - Drought / freeze: PDSI drop > −1.5σ OR HDD deviation spike
   - Energy transition: Battery PC1 crosses recent MA + large volatility

3. **Route signals at inference time**
   ```python
   # In pages/4_Models.py, on each commodity forecast:
   for trigger_type, affected_commodities in router.triggers.items():
       if trigger_detected(trigger_type):
           trigger_strength = measure_trigger_strength()
           multipliers = router.route(trigger_type, trigger_strength)
           
           if commodity in affected_commodities:
               # Boost confidence for XGBoost, LSTM, etc.
               for model in all_models:
                   model.confidence_multiplier = multipliers[commodity]
   ```

4. **Track trigger events in a log**
   - Log every detected trigger with strength, date, affected commodities
   - Use this for post-hoc validation (did IC improve for routed sectors?)

**Deliverable:** When you run the Models page on an OPEC meeting date, energy sector models automatically show boosted confidence; calendar dummies alone don't drive it.

---

### Phase 3: Meta-Predictor (Week 3–4) — Model Arbitration

**Objective:** The dashboard learns which model to trust under which conditions.

**Tasks:**
1. **Gather training data**
   - For every day (last 2 years), collect:
     - Actual next-day return per commodity
     - Forecast from each tier (statistical, ML, deep, quantum)
     - Each model's confidence (IC)
     - Market state (HMM regime, VIX, DXY, TLT, WASDE/OPEC window, ENSO phase)
   - This is a backtesting dataset, not a live prediction task

2. **Train a small decision-tree meta-model**
   ```python
   X_train = [
       [vix, dxy_zscore, hdd_deviation, tlt_momentum, days_since_regime_flip, ...],
       ...
   ]
   
   y_train = [
       # For each date, which model had lowest forecast error?
       model_with_lowest_error,
       ...
   ]
   
   meta_dt = DecisionTreeClassifier(max_depth=5)  # shallow, interpretable
   meta_dt.fit(X_train, y_train)
   ```

3. **Use meta-model to reweight ensemble**
   - Predict: "probability this date should weight Statistical = 40%, ML = 35%, Deep = 25%"
   - Apply these weights to each model's forecast

4. **Log meta-predictor decisions**
   - For every inference, show which model was "trusted most" and why
   - Adds transparency to the dashboard ("Today we trust the LSTM trend model because VIX is low and we're in a Bull regime")

**Deliverable:** A new dashboard card showing "Model Consensus Strength" — if all 5 tiers agree, IC = 0.15; if they disagree, show which model is winning and why.

---

### Phase 4: Chain of Insights (Week 4–5) — Causal Visualization

**Objective:** Surface the **chain of causality** from macro event to actionable portfolio decision.

**Tasks:**
1. **Build a causal chain tracer**
   ```python
   class CausalChain:
       """Traces how a macro event flows through the model ecosystem."""
       
       def trace(self, trigger: str, start_date: str) -> Dict:
           # E.g., trigger = 'opec_cut', start_date = '2026-04-28'
           
           return {
               'trigger': 'OPEC+ cuts 1.3M bpd',
               'affected_sectors': ['Energy'],
               'statistical_signal': {
                   'model': 'GARCH',
                   'old_forecast_vol': 0.023,
                   'new_forecast_vol': 0.035,
                   'change_pct': '+52%',
                   'confidence': 0.12,
               },
               'ml_signal': {
                   'model': 'HMM Regime Detector',
                   'old_regime': 'Neutral',
                   'new_regime': 'High-Vol',
                   'probability': 0.78,
               },
               'deep_signal': {
                   'model': 'Prophet',
                   'old_trend_slope': +0.002,
                   'new_trend_slope': +0.005,
                   'detected_changepoint': '2026-04-29',
               },
               'portfolio_action': {
                   'recommendation': 'Increase WTI, Brent position sizing',
                   'confidence': 0.65,  # meta-model consensus
                   'rebalance_size': '5% of energy allocation',
               },
           }
   ```

2. **Visualize the chain**
   - Sankey diagram: Macro event → Statistical shift → Regime flip → Portfolio action
   - Shows at each step: confidence, which commodities affected, who disagreed

3. **Add explanatory narrative**
   - "On 2026-04-28, OPEC+ announced a 1.3M bpd cut. Our statistical models (GARCH) upgraded WTI vol forecast by 52%. The regime detector flipped to High-Vol (78% confidence). Trend models expect continuation upside. Consensus portfolio recommendation: increase WTI/Brent sizing by 5%."

**Deliverable:** A new "Causal Events" page that shows recent macro events, their cascade through models, and recommended portfolio actions.

---

### Phase 5: Ecosystem Optimization (Week 5–6) — Continuous Improvement

**Objective:** The system learns from its own feedback loop.

**Tasks:**
1. **Build a daily backtest harness**
   - Every day (or weekly), measure:
     - Which sector had highest IC this week? (Statistical? ML? Deep?)
     - Did routed triggers improve IC for affected commodities?
     - Meta-predictor accuracy: did the trusted model outperform?
     - Did causal chains convert to profitable trades?

2. **Auto-tune trigger thresholds**
   - If "Fed tightening" trigger shows weak correlation with actual regime flips, lower its threshold
   - If "WASDE window" shows strong correlation with volatility spikes, raise its multiplier

3. **Retrain meta-predictor weekly**
   - Feed new out-of-sample data
   - Update decision boundaries
   - Log any major regime shift in meta-predictor's behavior

4. **Compute sector-specific ICs**
   - Before ecosystem: "XGBoost IC = 0.087 across all 41 commodities"
   - After ecosystem: "XGBoost IC for energy when routed = 0.12; for ag = 0.06"
   - Use this to inform which sectors to focus tuning on

**Deliverable:** Weekly "Model Health Report" showing IC trends by tier, by sector, by trigger type.

---

## Expected Outcomes by Phase

| Phase | Outcome | IC Improvement Target |
|-------|---------|----------------------|
| **1: Foundation** | Standardized signals, no yet integrated | +0% (baseline) |
| **2: Cascade** | Macro triggers route through sectors | +2–5% (triggers were already known; organization should help) |
| **3: Meta** | Models learn which to trust when | +5–10% (arbitration reduces bad bets in disagreement) |
| **4: Causal** | Insights surface automatically | +0% (explainability, not forecasting; but reduces analyst workload) |
| **5: Optimization** | System learns from feedback loop | +5–15% (tuning + sector specialization) |

**Realistic total by Q3 2026:** +12–30% IC improvement, assuming disciplined execution and no external market regime changes.

---

## The Quantum Angle

Your quantum models (kernel SVM, QAOA, QNN hybrid) should occupy a **specialized role**, not compete generically:

1. **Quantum Kernel SVM** → Use as a **signal diversity check**. If quantum kernel finds a signal classical ML missed (e.g., highly non-linear interaction), route it through the ensemble for potential +IC boost.

2. **QAOA Portfolio Optimizer** → Feed consensus bullish/bearish list from Phases 1–3, let QAOA solve the exact cardinality-constrained allocation (best 10 assets, exact weight allocation). This is the **portfolio decision output**.

3. **Hybrid QNN** → Research layer. Compare actual vs classical baseline on hold-out quarterly data. If it beats both classical NN and all Tier 2 models, promote it to Phase 2 routing.

---

## Next Immediate Steps (This Week)

Pick **one** and execute:

1. **Signal standardization (Phase 1, Day 1)**
   - Create `models/signal_base.py` with `ModelSignal` class
   - Refactor `XGBoostForecaster.predict()` to emit `ModelSignal` instead of raw array
   - Update `pages/4_Models.py` XGBoost tab to consume the new signal schema
   - Test: does the tab still render correctly?

2. **Trigger detection (Phase 2, Day 1)**
   - In `features/macro_overlays.py`, add a `detect_triggers()` function
   - Hardcode OPEC/WASDE dates for next 12 months
   - Add DXY stress detection (3-day consecutive z-score > 2.0)
   - Test: does the trigger log populate correctly?

3. **Meta-predictor skeleton (Phase 3, Day 1)**
   - Create `models/meta_predictor.py`
   - Write the decision-tree structure (don't train yet)
   - Define the input feature set (VIX, DXY, HMM regime, etc.)
   - Test: can it consume forecasts from all 5 tiers?

---

## Strategic Alignment with Accendio's Vision

> "Our goal is to have all the models working together in tandem like an eco-system where our specialized research on causality between macro-trends, economic events, and commodities ripple like a chain reaction through the dashboard."

**This roadmap delivers exactly that:**
- ✅ Models work "in tandem" (Phase 2: cascade routing)
- ✅ Causality flows "like a chain reaction" (Phase 4: causal visualization)
- ✅ Macro events "ripple through" the dashboard (Phase 2: trigger propagation)
- ✅ System is "ecosystem-like" (Phase 3: meta-arbitration teaches models to cooperate)

By Q3 2026, users won't see five isolated model cards. They'll see a **living, breathing system** that explains itself: "Here's the macro event. Here's how it cascaded through our model ecosystem. Here's the portfolio action we recommend."

That's your differentiation vs. generic commodity dashboards. **Launch Phase 1–3 by end of Q2, you've got a defensible moat.**

---

## Open Questions for You

1. **Priority**: Do you want to pursue **quantum** as a signal-diversity layer (QAOA optimizer), or keep it as research?
2. **Portfolio integration**: Should the meta-predictor output feed directly to a position-sizing module, or stay advisory?
3. **Governance**: Who (human or algorithm) makes the final trade decision on routed signals?
4. **Frequency**: Run the full cascade daily, or only on macro events?

