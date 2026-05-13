"""
Scenario Bands — bear / average / bull forecast envelopes.

Phase 1 surfaces consensus bands from the project's statistical models
(ARIMA + VAR) for any commodity in MODELING_COMMODITIES. The fan chart
shows a shaded cone whose width represents forecast uncertainty growing
over the horizon. A calibration panel audits empirical coverage against
the nominal quantile spread — the audit the design notes flagged as the
most important step.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.theme import apply_theme, render_topbar, PLOTLY_LAYOUT
from models.config import MODELING_COMMODITIES
from models.scenarios import (
    ScenarioAggregator,
    arima_band,
    var_band,
    elastic_net_band,
    random_forest_band,
    xgboost_band,
    lstm_band,
    tft_band,
    prophet_band,
    quantum_kernel_band,
    empirical_coverage,
    CausalRipple,
    build_narrative,
    collapse_to_tail,
    find_analogs,
)
from models.scenarios.band import returns_to_price_paths, attach_business_dates
from models.scenarios.providers import find_var_group


st.set_page_config(
    page_title="Accendio | Scenarios",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()
render_topbar()

# ── Theme constants (match pages/4_Models.py) ─────────────────────────────────
BG, PLOT_BG, GRID = "#060912", "#0C1228", "#1A2A5E"
AMBER, BLUE, GREEN, RED, PURPLE = "#F59E0B", "#4FC3F7", "#66BB6A", "#EF5350", "#AB47BC"


# ── Data loaders (mirrors pages/4_Models.py) ──────────────────────────────────
@st.cache_data(ttl=14400, show_spinner=False)
def load_prices_full():
    from models.data_loader import load_price_matrix_from_db
    return load_price_matrix_from_db()


# ── Cached band builders ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def build_arima_band(returns_dict, commodity, horizon, bear_q, bull_q):
    """returns_dict is a serialisable {date: float} so the cache key is hashable."""
    ser = pd.Series(returns_dict).sort_index()
    ser.index = pd.to_datetime(ser.index)
    return arima_band(ser, commodity=commodity, horizon=horizon,
                      bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_var_band(prices_records, commodity, group, horizon, bear_q, bull_q):
    df = pd.DataFrame(prices_records)
    df.index = pd.to_datetime(df.pop("__date"))
    df = df.sort_index()
    return var_band(df, commodity=commodity, group=group, horizon=horizon,
                    bear_q=bear_q, bull_q=bull_q)


def _prices_records(prices: pd.DataFrame):
    return prices.reset_index().rename(
        columns={prices.index.name or "index": "__date"}
    ).to_dict("records")


def _records_to_df(records):
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df.pop("__date"))
    return df.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def build_elastic_net_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return elastic_net_band(df, commodity=commodity, horizon=horizon,
                            bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_random_forest_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return random_forest_band(df, commodity=commodity, horizon=horizon,
                              bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_xgboost_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return xgboost_band(df, commodity=commodity, horizon=horizon,
                        bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_lstm_band(prices_records, commodity, horizon, bear_q, bull_q, n_mc):
    df = _records_to_df(prices_records)
    return lstm_band(df, commodity=commodity, horizon=horizon,
                     bear_q=bear_q, bull_q=bull_q, n_mc_samples=n_mc)


@st.cache_data(ttl=3600, show_spinner=False)
def build_tft_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return tft_band(df, commodity=commodity, horizon=horizon,
                    bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_prophet_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return prophet_band(df, commodity=commodity, horizon=horizon,
                        bear_q=bear_q, bull_q=bull_q)


@st.cache_data(ttl=3600, show_spinner=False)
def build_quantum_band(prices_records, commodity, horizon, bear_q, bull_q):
    df = _records_to_df(prices_records)
    return quantum_kernel_band(df, commodity=commodity, horizon=horizon,
                               bear_q=bear_q, bull_q=bull_q)


# ── UI ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.divider()
    st.page_link("app.py",                           label="Home")
    st.page_link("pages/1_Pricing.py",               label="Pricing")
    st.page_link("pages/2_Charts.py",                label="Charts")
    st.page_link("pages/3_News.py",                  label="News")
    st.page_link("pages/4_Models.py",                label="Models")
    st.page_link("pages/5_Database.py",              label="Database")
    st.divider()
    st.page_link("pages/6_Causal_QS_Engine.py",      label="Causal QS Engine")
    st.page_link("pages/7_Macro_Market_Cascade.py",  label="Macro-Market Cascade")
    st.page_link("pages/8_Portfolio.py",             label="Portfolio")
    st.page_link("pages/9_Scenarios.py",             label="Scenarios")
    st.divider()

st.title("Scenario Bands")
st.caption(
    "Bear / average / bull envelopes derived from your existing models — what could "
    "this commodity do over the next N days, and how confident is the model in that range?"
)

# Auto-generated investor-facing summary, filled in once the consensus is built.
narrative_slot = st.empty()

with st.expander("📖 Glossary — what do these terms mean?"):
    st.markdown(
        """
| Term | Plain English |
|---|---|
| **Spot** | The latest observed close — what the commodity is worth right now. |
| **Horizon** | How many trading days ahead we're forecasting. 5 ≈ one week. |
| **Bear / Bull (Pn)** | The pessimistic/optimistic end of a likely-range scenario. **P10** = "only a 10 % chance things end up worse than this." Higher number = more extreme tail. |
| **Consensus** | The weighted-average forecast across all enabled providers. Each provider is one model (ARIMA, XGBoost, LSTM, …). |
| **Log return** | The natural-log of (tomorrow's price ÷ today's price). Used internally because log returns add cleanly across days. ~1 % log return ≈ ~1 % price change. |
| **Fan chart** | The shaded "cone" showing the bear–bull range over the forecast horizon. The cone widens because uncertainty compounds with time. |
| **Conformal calibration** | A statistical technique where the band width is set by the model's actual past errors rather than a theoretical formula — gives more honest coverage. |
| **MC Dropout** | "Monte-Carlo Dropout." Run the neural network 100 × with random neurons turned off; the spread of predictions is the model's uncertainty about itself. |
| **Quantile loss** | A training objective that teaches a model to predict P10 / P50 / P90 directly (rather than a single point). Used by the TFT provider. |
| **IRF / impulse response** | "If commodity A moves by 1 unit today, how does commodity B respond today, tomorrow, day after?" Computed from a Vector Autoregression fit to a complex (e.g. all 5 energy benchmarks). |
| **Coupling** | The IRF rescaled so 100 % means "moves in perfect lockstep with the source." 0 % means uncorrelated. Negative means moves opposite. |
| **IC (Information Coefficient)** | Spearman rank correlation between the model's predictions and what actually happened on the holdout. IC > 5 % is considered actionable for daily commodity signals. |
        """
    )

with st.spinner("Loading full commodity matrix…"):
    prices_full = load_prices_full()

if prices_full is None or prices_full.empty:
    st.error("Price matrix is empty. Run `python -m pipeline.ingest` and reload.")
    st.stop()

returns_full = np.log(prices_full / prices_full.shift(1)).dropna()

# Controls
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    commodity = st.selectbox(
        "Commodity",
        list(MODELING_COMMODITIES.keys()),
        index=list(MODELING_COMMODITIES.keys()).index("WTI Crude Oil")
            if "WTI Crude Oil" in MODELING_COMMODITIES else 0,
        help="The commodity whose price scenario you want to project.",
    )
with c2:
    horizon = st.slider(
        "Horizon (trading days)", 1, 20, 10,
        help="How many trading days forward to forecast. 5 ≈ one week, "
             "10 ≈ two weeks, 20 ≈ one month. The fan widens as horizon grows "
             "because uncertainty compounds.",
    )
with c3:
    bear_q = st.slider(
        "Bear quantile (P-low)", 0.01, 0.49, 0.10, step=0.01,
        help="The pessimistic percentile of the forecast distribution. "
             "**P10** means \"price stays above this level 90% of the time\" — "
             "i.e. there's a 10% chance things are worse. "
             "P5 = stricter downside (1-in-20 case); P10 = standard; P25 = mild.",
    )
with c4:
    bull_q = st.slider(
        "Bull quantile (P-high)", 0.51, 0.99, 0.90, step=0.01,
        help="The optimistic percentile of the forecast distribution. "
             "**P90** means \"90% of paths stay below this level\". "
             "Need not be symmetric to bear — pair P5/P75 if you weight "
             "downside more heavily than upside.",
    )

# Quick read-out so users see the asymmetric coverage they've selected
_coverage = bull_q - bear_q
_asymmetry = (bear_q + bull_q) - 1.0
st.caption(
    f"Selected coverage: **{_coverage*100:.0f}%** between P{int(bear_q*100)} and "
    f"P{int(bull_q*100)}"
    + ("." if abs(_asymmetry) < 1e-6 else
       f"  · asymmetric (median shifted {_asymmetry*100:+.0f} pp from symmetric)")
)

st.markdown("##### Providers")
st.caption(
    "Each provider is one model that contributes a bear/avg/bull view. The consensus "
    "(below) is a weighted average. Statistical (fast) ➔ ML (10–30s) ➔ Deep (minutes — "
    "LSTM/TFT train from scratch)."
)
pa1, pa2, pa3, pa4, pa5 = st.columns(5)
with pa1:
    use_arima = st.checkbox(
        "ARIMA", True, key="use_arima",
        help="Classical time-series model. Uses only this commodity's own return history. "
             "Bands come from the model's analytical Gaussian prediction interval — "
             "cheap, well-understood, but tends to be narrow on real markets.",
    )
with pa2:
    use_var = st.checkbox(
        "VAR", True, key="use_var",
        help="**Vector Autoregression** — joint model across the whole complex (e.g. "
             "WTI + Brent + NatGas + Gasoline + Heating Oil). Captures cross-commodity "
             "lead/lag dynamics. Only available for commodities in a predefined complex.",
    )
with pa3:
    use_en = st.checkbox(
        "Conformal[ElasticNet]", False, key="use_en",
        help="Sparse linear factor model using all 200+ features. **Conformal** means "
             "the band width is calibrated against the model's actual historical errors "
             "(out-of-sample residuals) rather than a theoretical formula.",
    )
with pa4:
    use_rf = st.checkbox(
        "Conformal[RandomForest]", False, key="use_rf",
        help="Random forest ensemble — captures non-linear interactions. Conformal "
             "calibration on the holdout makes the band reflect realised forecast error.",
    )
with pa5:
    use_xgb = st.checkbox(
        "Conformal[XGBoost]", False, key="use_xgb",
        help="Gradient-boosted trees — currently the strongest non-deep workhorse. "
             "Conformal calibration as above. Requires `xgboost` installed.",
    )

pb1, pb2, pb3, pb4, pb5 = st.columns(5)
with pb1:
    use_lstm = st.checkbox(
        "MCDropout[LSTM]", False, key="use_lstm",
        help="Bidirectional LSTM trained on 30-day input sequences. **MC Dropout** = "
             "keep dropout active at inference and run many forward passes, generating "
             "a distribution of predictions (model uncertainty). Band width is "
             "conformal-calibrated on holdout. Slow: trains from scratch (~1–3 min).",
    )
with pb2:
    use_tft = st.checkbox(
        "TFT[quantile]", False, key="use_tft",
        help="**Temporal Fusion Transformer** — a transformer architecture for time "
             "series. Trained natively with quantile loss to output P10/P50/P90 directly "
             "(no resampling). Slow: trains from scratch (~2–5 min).",
    )
with pb3:
    use_prophet = st.checkbox(
        "Prophet", False, key="use_prophet",
        help="Decomposition model (trend + seasonality + holidays) fit on log-prices. "
             "Useful when seasonality matters (NatGas, ags). Bands derive from "
             "Prophet's MCMC-style uncertainty intervals. Fast: ~5–20s.",
    )
with pb4:
    use_quantum = st.checkbox(
        "Quantum[KernelRidge]", False, key="use_quantum",
        help="Quantum-kernel ridge regression: classical features are encoded into "
             "a 4-qubit quantum state, and similarities are measured in Hilbert space. "
             "Currently a research-prototype band — interpretable but does not "
             "outperform classical kernels on this data. Slow: ~1–2 min on simulator.",
    )
with pb5:
    n_mc = st.number_input(
        "MC samples (LSTM)", min_value=20, max_value=500, value=100, step=20,
        help="Number of stochastic forward passes for MC-Dropout. 100 is standard; "
             "higher gives a tighter empirical distribution at proportional cost.",
    )

if commodity not in returns_full.columns:
    st.warning(f"No price history for {commodity} in the modeling matrix.")
    st.stop()

ret_series = returns_full[commodity].dropna()
if len(ret_series) < 250:
    st.warning(
        f"Only {len(ret_series)} return observations for {commodity} — "
        "need ≥250 to fit ARIMA reliably."
    )
    st.stop()

# ── Build provider bands ──────────────────────────────────────────────────────
bands = []
provider_errors = {}

if use_arima:
    with st.spinner("Fitting ARIMA…"):
        try:
            ar_band = build_arima_band(
                ret_series.to_dict(), commodity, horizon, bear_q, bull_q
            )
            bands.append(ar_band)
        except Exception as e:
            provider_errors["ARIMA"] = str(e)

prices_records = _prices_records(prices_full)
var_group = find_var_group(commodity)
if use_var and var_group:
    with st.spinner(f"Fitting VAR[{var_group}]…"):
        try:
            v_band = build_var_band(
                prices_records, commodity, var_group, horizon, bear_q, bull_q
            )
            if v_band is not None:
                bands.append(v_band)
        except Exception as e:
            provider_errors[f"VAR[{var_group}]"] = str(e)

if use_en:
    with st.spinner("Fitting Elastic Net + conformal calibration…"):
        try:
            b = build_elastic_net_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["Conformal[ElasticNet]"] = "Returned None — see logs."
        except Exception as e:
            provider_errors["Conformal[ElasticNet]"] = str(e)

if use_rf:
    with st.spinner("Fitting Random Forest + conformal calibration…"):
        try:
            b = build_random_forest_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["Conformal[RandomForest]"] = "Returned None."
        except Exception as e:
            provider_errors["Conformal[RandomForest]"] = str(e)

if use_xgb:
    with st.spinner("Fitting XGBoost + conformal calibration…"):
        try:
            b = build_xgboost_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["Conformal[XGBoost]"] = "Returned None."
        except Exception as e:
            provider_errors["Conformal[XGBoost]"] = str(e)

if use_lstm:
    with st.spinner("Training BiLSTM + running MC-Dropout passes (1–3 min)…"):
        try:
            b = build_lstm_band(
                prices_records, commodity, horizon, bear_q, bull_q, int(n_mc)
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["MCDropout[LSTM]"] = "Returned None (torch missing or training failed)."
        except Exception as e:
            provider_errors["MCDropout[LSTM]"] = str(e)

if use_tft:
    with st.spinner("Training Temporal Fusion Transformer (2–5 min)…"):
        try:
            b = build_tft_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["TFT[quantile]"] = "Returned None (pytorch_forecasting missing or training failed)."
        except Exception as e:
            provider_errors["TFT[quantile]"] = str(e)

if use_prophet:
    with st.spinner("Fitting Prophet (log-price)…"):
        try:
            b = build_prophet_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["Prophet"] = "Returned None (prophet package missing?)."
        except Exception as e:
            provider_errors["Prophet"] = str(e)

if use_quantum:
    with st.spinner("Running quantum kernel ridge + conformal calibration (1–2 min)…"):
        try:
            b = build_quantum_band(
                prices_records, commodity, horizon, bear_q, bull_q
            )
            if b is not None:
                bands.append(b)
            else:
                provider_errors["Quantum[KernelRidge]"] = "Returned None (pennylane missing or insufficient data)."
        except Exception as e:
            provider_errors["Quantum[KernelRidge]"] = str(e)

if not bands:
    st.error("No providers produced a band. See errors below.")
    st.json(provider_errors)
    st.stop()

# ── IC-weighted defaults ──────────────────────────────────────────────────────
# Providers that carry an `ic` diagnostic (the ML/deep/quantum ones) get a
# default weight proportional to their out-of-sample IC, normalised so the
# best IC maps to ~1.5 and a zero/negative IC maps to ~0.1 (kept above zero
# so a single mis-IC provider can't entirely drop out of the consensus, and
# the user can always re-weight via the sliders). Providers without an IC
# field (ARIMA, VAR) inherit the median informed weight.

def _ic_to_weight(ic_values: list) -> dict:
    if not ic_values:
        return {}
    arr = np.asarray([(v if v is not None else 0.0) for v in ic_values])
    arr_pos = np.clip(arr, 0.0, None)
    max_pos = arr_pos.max() if arr_pos.max() > 0 else 1.0
    scaled = 0.1 + 1.4 * (arr_pos / max_pos)   # range [0.1, 1.5]
    return {i: float(round(s, 2)) for i, s in enumerate(scaled)}

_ics = [(b.diagnostics.get("ic") if b.diagnostics else None) for b in bands]
_informed = [v for v in _ics if v is not None]
_w_map = _ic_to_weight(_informed)
_informed_weights = list(_w_map.values())
_median_informed = float(np.median(_informed_weights)) if _informed_weights else 1.0

_default_weights = []
_informed_idx = 0
for ic in _ics:
    if ic is None:
        # Statistical models / native-quantile providers without an IC field
        # inherit the median weight of providers that do report one.
        _default_weights.append(round(_median_informed, 2))
    else:
        _default_weights.append(_w_map[_informed_idx])
        _informed_idx += 1

st.markdown("##### Provider weights")
st.caption(
    "Defaults are IC-weighted: providers with stronger out-of-sample Spearman IC start "
    "with a larger consensus weight (range 0.1 – 1.5). Providers without an IC field "
    "(ARIMA, VAR, native-quantile TFT/Prophet) inherit the median informed weight. "
    "Override freely."
)
weight_cols = st.columns(len(bands))
weights = {}
for col, b, w0, ic in zip(weight_cols, bands, _default_weights, _ics):
    with col:
        label = b.source_model + (f"  ·  IC {ic:+.3f}" if ic is not None else "")
        weights[b.source_model] = st.slider(
            label, 0.0, 2.0, w0, step=0.1, key=f"w_{b.source_model}"
        )

aggregator = ScenarioAggregator(weights=weights)
consensus = aggregator.aggregate(bands)

# ── Fan chart ─────────────────────────────────────────────────────────────────
last_price = float(prices_full[commodity].iloc[-1])
last_date = prices_full.index[-1]

# Build the ripple engine once if this commodity is in a known complex —
# we use it for both the narrative summary and the lower causal-ripple section.
@st.cache_resource(show_spinner=False)
def _cached_ripple(prices_records, group):
    return CausalRipple(_records_to_df(prices_records), group=group)

ripple_engine_top = None
if var_group is not None:
    try:
        ripple_engine_top = _cached_ripple(prices_records, var_group)
    except Exception:
        ripple_engine_top = None

# Render the auto-narrative into the slot reserved at the top.
narrative_slot.info(
    build_narrative(consensus, last_price, ripple_engine=ripple_engine_top)
)

# Trailing history for context
hist_window = 60
hist = prices_full[commodity].dropna().iloc[-hist_window:]

paths = returns_to_price_paths(last_price, consensus)
paths = attach_business_dates(paths, anchor=last_date)

fig = go.Figure()

# Historical line
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values,
    mode="lines", name=f"{commodity} (observed)",
    line=dict(color=BLUE, width=2),
))

# Bull bound
fig.add_trace(go.Scatter(
    x=paths.index, y=paths["bull"],
    mode="lines", name=f"Bull (P{int(bull_q*100)})",
    line=dict(color=GREEN, width=1, dash="dot"),
))
# Bear bound + shading via fill='tonexty'
fig.add_trace(go.Scatter(
    x=paths.index, y=paths["bear"],
    mode="lines", name=f"Bear (P{int(bear_q*100)})",
    line=dict(color=RED, width=1, dash="dot"),
    fill="tonexty", fillcolor="rgba(245, 158, 11, 0.18)",
))
# Mean
fig.add_trace(go.Scatter(
    x=paths.index, y=paths["mean"],
    mode="lines+markers", name="Consensus mean",
    line=dict(color=AMBER, width=2.5),
    marker=dict(size=5),
))

# ── Historical analog overlay ────────────────────────────────────────────────
show_analogs = st.checkbox(
    "Overlay historical analogs",
    value=False,
    key="show_analogs",
    help="Find the 5 most-similar past 20-day return patterns to today's and "
         "overlay what actually happened in the days that followed. If the "
         "analogs cluster around the consensus mean, that's confirmation. "
         "If they diverge, the model may be missing regime context.",
)
analog_results = []
if show_analogs:
    analog_results = find_analogs(
        ret_series, horizon=horizon, spot=last_price,
        window=20, k=5, min_lookback_days=horizon,
    )
    if analog_results:
        x_idx = paths.index
        for i, a in enumerate(analog_results):
            fig.add_trace(go.Scatter(
                x=x_idx, y=a.forward_price_path,
                mode="lines",
                name=f"Analog {a.match_end_date.date()} (r={a.similarity:+.2f})",
                line=dict(color="rgba(180, 180, 200, 0.55)", width=1, dash="dash"),
                hovertemplate=(
                    f"<b>Analog: {a.match_end_date.date()}</b><br>"
                    f"Similarity r={a.similarity:+.3f}<br>"
                    f"Forward total: {a.forward_total_return*100:+.1f}%<extra></extra>"
                ),
            ))

fig.update_layout(
    **PLOTLY_LAYOUT,
    title=f"{commodity} — {horizon}-day scenario fan",
    xaxis_title="", yaxis_title="Price",
    height=520,
    legend_orientation="h", legend_yanchor="bottom", legend_y=1.02,
    legend_xanchor="right", legend_x=1,
)
st.plotly_chart(fig, use_container_width=True)

# Honest disclosure about how the cone widens for 1-step providers
_sqrt_h_providers = [b.source_model for b in bands
                     if b.source_model.startswith(("Conformal", "MCDropout", "Quantum"))]
if _sqrt_h_providers:
    st.caption(
        "ⓘ ML, deep-MC, and quantum providers are 1-step forecasters; their bands widen as √h "
        "across the horizon, which is a Brownian-motion (random-walk residuals) assumption. "
        "Real volatility scales sub-linearly in calm regimes and super-linearly across "
        "macro/earnings events. Statistical (ARIMA, VAR) and Prophet/TFT bands derive their "
        "h-step uncertainty natively from the model and don't make this assumption."
    )

# ── Numerical summary ─────────────────────────────────────────────────────────
final_step = paths.iloc[-1]
spot = last_price
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"${spot:,.2f}",
          help="Most recent observed close price — the anchor for the forecast.")
m2.metric(
    f"Bear @ {horizon}d (P{int(bear_q*100)})", f"${final_step['bear']:,.2f}",
    f"{(final_step['bear']/spot - 1) * 100:+.2f}%",
    help=f"Pessimistic-case price at end of horizon. Under the consensus, "
         f"there's a ~{int(bear_q*100)}% chance the price ends below this level.",
)
m3.metric(
    "Consensus (end)", f"${final_step['mean']:,.2f}",
    f"{(final_step['mean']/spot - 1) * 100:+.2f}%",
    help="Weighted average of all enabled providers' point forecasts at horizon end. "
         "Not the median — it's literally the consensus mean path.",
)
m4.metric(
    f"Bull @ {horizon}d (P{int(bull_q*100)})", f"${final_step['bull']:,.2f}",
    f"{(final_step['bull']/spot - 1) * 100:+.2f}%",
    help=f"Optimistic-case price at end of horizon. Under the consensus, "
         f"there's a ~{int((1-bull_q)*100)}% chance the price ends above this level.",
)

# ── Provider attribution ──────────────────────────────────────────────────────
with st.expander("Per-provider bands (attribution)"):
    attr = aggregator.attribution_frame(bands)
    st.dataframe(
        attr.pivot_table(index="step", columns="source_model",
                          values=["bear", "mean", "bull"]).round(5),
        use_container_width=True,
    )
    st.caption(
        "Each provider's raw log-return bands at every forecast step. "
        "The consensus row is the weighted sum across providers."
    )

if provider_errors:
    with st.expander("Provider errors"):
        st.json(provider_errors)

# Surface in-band quantile warnings (e.g. TFT's native q10/q50/q90 don't
# adapt to arbitrary slider settings)
_quantile_warnings = {
    b.source_model: b.diagnostics["quantile_warning"]
    for b in bands
    if b.diagnostics and b.diagnostics.get("quantile_warning")
}
if _quantile_warnings:
    with st.expander("⚠ Quantile mismatches", expanded=True):
        for src, msg in _quantile_warnings.items():
            st.warning(f"**{src}**: {msg}")

# ── Causal ripple (Phase 4) ───────────────────────────────────────────────────
st.divider()
st.markdown("### Causal ripple")
st.caption(
    "When the consensus scenario fires on this commodity, how does it propagate to "
    "the rest of the complex? We use the **VAR impulse-response function** "
    "(\"if A moves by 1, how does B respond at lag 0, 1, 2 … ?\") as a per-step "
    "coupling coefficient. Normalised so a value of 1.00 means the peer moves 100 % "
    "in lockstep with the source; 0 means uncorrelated; negative means inversely."
)

if var_group is None:
    st.info(
        f"{commodity} is not in any predefined VAR complex (energy / grains / etc.), "
        "so causal ripple is unavailable here."
    )
else:
    try:
        ripple_engine = ripple_engine_top or _cached_ripple(prices_records, var_group)
        peers = [c for c in ripple_engine.members() if c != commodity]
        target = st.selectbox(
            f"Propagate {commodity} → target ({var_group} complex)",
            peers,
            key="ripple_target",
        )
        derived = ripple_engine.propagate(consensus, target)
        if derived is None:
            st.warning("Could not propagate to that target.")
        else:
            last_price_t = float(prices_full[target].iloc[-1])
            paths_t = returns_to_price_paths(last_price_t, derived)
            paths_t = attach_business_dates(paths_t, anchor=last_date)
            hist_t = prices_full[target].dropna().iloc[-hist_window:]

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=hist_t.index, y=hist_t.values, mode="lines",
                name=f"{target} (observed)",
                line=dict(color=BLUE, width=2),
            ))
            fig_t.add_trace(go.Scatter(
                x=paths_t.index, y=paths_t["bull"], mode="lines",
                name=f"Bull (rippled)", line=dict(color=GREEN, width=1, dash="dot"),
            ))
            fig_t.add_trace(go.Scatter(
                x=paths_t.index, y=paths_t["bear"], mode="lines",
                name=f"Bear (rippled)", line=dict(color=RED, width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(171, 71, 188, 0.18)",
            ))
            fig_t.add_trace(go.Scatter(
                x=paths_t.index, y=paths_t["mean"], mode="lines+markers",
                name="Rippled mean", line=dict(color=PURPLE, width=2.5),
                marker=dict(size=5),
            ))
            fig_t.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Ripple: {commodity} consensus → {target}",
                xaxis_title="", yaxis_title="Price",
                height=440,
                legend_orientation="h", legend_yanchor="bottom", legend_y=1.02,
                legend_xanchor="right", legend_x=1,
            )
            st.plotly_chart(fig_t, use_container_width=True)

            coup = derived.diagnostics["coupling"]
            coup_pct = [f"{c*100:+.0f}%" for c in coup]
            st.caption(
                f"Per-step coupling (h=1..{len(coup)}): "
                + ", ".join(coup_pct)
                + f". Read: at h=1, a unit move in {commodity} translates into a "
                f"{coup_pct[0]} move in {target}; subsequent steps decay quickly because "
                f"the VAR(p={derived.diagnostics['var_lag_order']}) sees most of the "
                "propagation as contemporaneous."
            )

        with st.expander(f"{var_group} complex coupling matrix"):
            mat = ripple_engine.coupling_matrix(horizon=horizon)
            st.dataframe(mat, use_container_width=True)
            st.caption(
                "Rows = responding commodity, columns = shock source. "
                "Values are the IRF impulse summed over the forecast horizon."
            )
    except Exception as e:
        st.warning(f"Causal ripple unavailable: {e}")


# ── Stress test (Phase 5) ─────────────────────────────────────────────────────
st.divider()
st.markdown("### Stress test — *if* the tail plays out, what happens to the complex?")
st.caption(
    "Forces the source commodity onto the bear (or bull) path with certainty, then "
    "propagates that deterministic scenario through the causal-ripple engine to every "
    "peer. Useful for the question \"if WTI really drops to P5, what's my Brent / "
    "Heating Oil exposure?\"."
)
if var_group is None:
    st.info(f"{commodity} is not in a predefined complex — stress test unavailable.")
else:
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        stress_side = st.radio(
            "Stress scenario", ["bear", "bull"], index=0, horizontal=True,
            help="`bear` forces the P-low path on the source; `bull` forces P-high. "
                 "All peers respond via the IRF coupling matrix.",
            key="stress_side",
        )
    try:
        ripple_engine_st = ripple_engine_top or _cached_ripple(prices_records, var_group)
        stressed = collapse_to_tail(consensus, side=stress_side)
        peers = [c for c in ripple_engine_st.members() if c != commodity]
        # Render source + peers as a small grid (3 cols)
        n_total = 1 + len(peers)
        rows = []
        # Source first
        rows.append({
            "asset": f"{commodity} (source)",
            "spot": float(prices_full[commodity].iloc[-1]),
            "stressed_band": stressed,
        })
        for peer in peers:
            d = ripple_engine_st.propagate(stressed, peer)
            if d is None: continue
            rows.append({
                "asset": peer,
                "spot": float(prices_full[peer].iloc[-1]),
                "stressed_band": d,
            })
        # Render as a compact table of end-state prices and % moves
        records = []
        for r in rows:
            b = r["stressed_band"]
            end_price = r["spot"] * float(np.exp(np.sum(b.mean)))
            move_pct = (end_price / r["spot"] - 1.0) * 100.0
            records.append({
                "Asset": r["asset"],
                "Spot": f"${r['spot']:,.2f}",
                f"Stressed @ {horizon}d": f"${end_price:,.2f}",
                "Move": f"{move_pct:+.2f}%",
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

        # Mini stacked-bar of stressed moves
        bar_records = []
        for r in rows[1:]:    # peers only — skip the source bar
            b = r["stressed_band"]
            end_price = r["spot"] * float(np.exp(np.sum(b.mean)))
            move_pct = (end_price / r["spot"] - 1.0) * 100.0
            bar_records.append({"asset": r["asset"], "move_pct": move_pct})
        bar_df = pd.DataFrame(bar_records).sort_values("move_pct")
        # Always include source as reference
        src_move = (rows[0]["spot"] * float(np.exp(np.sum(rows[0]["stressed_band"].mean)))
                    / rows[0]["spot"] - 1.0) * 100.0
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bar_df["asset"], y=bar_df["move_pct"],
            marker_color=[RED if v < 0 else GREEN for v in bar_df["move_pct"]],
            name="Peer (rippled)",
            hovertemplate="%{x}<br>%{y:+.2f}%<extra></extra>",
        ))
        fig_bar.add_hline(
            y=src_move, line_dash="dot", line_color=AMBER,
            annotation_text=f"Source: {commodity} {src_move:+.2f}%",
            annotation_position="top right",
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Peer moves under {stress_side} stress (h={horizon}d, source forced to P{int((bear_q if stress_side=='bear' else bull_q)*100)})",
            yaxis_title="% move from spot",
            height=360,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.warning(f"Stress test unavailable: {e}")


# ── Historical-analog summary (Phase 5) ───────────────────────────────────────
st.divider()
st.markdown("### Historical analogs — *what actually happened* after similar setups?")
st.caption(
    "The fan chart above lets you overlay the 5 historical 20-day windows most "
    "similar to today's. Below is a numeric summary: when each analog occurred, how "
    "similar it was (Pearson r), and what return it delivered in the following "
    f"{horizon} trading days. If most analogs land near the consensus mean, the "
    "model and the past agree. If they cluster on one tail, that's a real-data warning."
)
analogs_summary = find_analogs(
    ret_series, horizon=horizon, spot=last_price,
    window=20, k=10, min_lookback_days=horizon,
)
if not analogs_summary:
    st.info("Insufficient history for analog matching at this horizon.")
else:
    rows = []
    for a in analogs_summary:
        rows.append({
            "Match end date": a.match_end_date.date().isoformat(),
            "Similarity r": round(a.similarity, 3),
            f"Forward {horizon}d return": f"{a.forward_total_return*100:+.2f}%",
            "End price (anchored at spot)": f"${a.forward_price_path[-1]:,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Quick sanity-check vs consensus
    cons_return = float(np.exp(np.sum(consensus.mean)) - 1.0)
    analog_returns = np.array([a.forward_total_return for a in analogs_summary])
    median_a = float(np.median(analog_returns))
    p25, p75 = float(np.quantile(analog_returns, 0.25)), float(np.quantile(analog_returns, 0.75))
    same_side = "✅ same side" if (cons_return * median_a >= 0) else "⚠ opposite side"
    st.caption(
        f"**Consensus says:** {cons_return*100:+.1f}% over {horizon}d. "
        f"**Analogs median:** {median_a*100:+.1f}% (IQR {p25*100:+.1f}% to {p75*100:+.1f}%). "
        f"Direction agreement: {same_side}."
    )


# ── Calibration audit ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### Calibration audit")
st.caption(
    "Rolling 1-step backtest on the most recent window: for each day we refit "
    "ARIMA on prior data, generate a 1-step band, and check whether the next "
    "day's realized return fell inside [bear, bull]. Empirical coverage should "
    "approach the nominal spread; persistent gaps mean the visible fan is "
    "mis-calibrated."
)

run_cal = st.button("Run calibration audit (ARIMA, ~30s for 60 days)")
if run_cal:
    with st.spinner("Refitting ARIMA over rolling window…"):
        def _build(ret_train):
            return arima_band(
                ret_train, commodity=commodity, horizon=1,
                bear_q=bear_q, bull_q=bull_q,
            )
        cal = empirical_coverage(ret_series, _build, window=60, min_train=250)

    if cal["n_eval"] == 0:
        st.warning("Calibration produced no evaluations.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Nominal coverage", f"{cal['expected']*100:.1f}%")
        cc2.metric("Empirical coverage", f"{cal['empirical']*100:.1f}%",
                   f"{(cal['empirical'] - cal['expected'])*100:+.1f} pp")
        cc3.metric("Days evaluated", cal["n_eval"])

        # Realized vs band
        eval_idx = ret_series.index[-cal["n_eval"] - 1:-1]
        cal_df = pd.DataFrame({
            "realized": cal["realized"],
            "bear": cal["bear"],
            "bull": cal["bull"],
            "hit": cal["hits"],
        }, index=eval_idx)

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=cal_df.index, y=cal_df["bull"], mode="lines",
            name="Bull bound", line=dict(color=GREEN, width=1, dash="dot"),
        ))
        fig_c.add_trace(go.Scatter(
            x=cal_df.index, y=cal_df["bear"], mode="lines",
            name="Bear bound", line=dict(color=RED, width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(245, 158, 11, 0.12)",
        ))
        fig_c.add_trace(go.Scatter(
            x=cal_df.index, y=cal_df["realized"], mode="markers",
            name="Realized",
            marker=dict(
                size=7,
                color=[GREEN if h else RED for h in cal_df["hit"]],
                line=dict(width=0.5, color="white"),
            ),
        ))
        fig_c.update_layout(
            **PLOTLY_LAYOUT,
            title="Rolling 1-step bands vs realized log returns",
            yaxis_title="Log return",
            height=380,
            legend_orientation="h", legend_yanchor="bottom", legend_y=1.02,
            legend_xanchor="right", legend_x=1,
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ── Footnote / phase status ───────────────────────────────────────────────────
with st.expander("Phase 1 status & next steps"):
    st.markdown(
        """
**Phase 1 (live)** — Statistical providers:
- `ARIMA` via `statsmodels` closed-form prediction intervals.
- `VAR[<group>]` for commodities in a predefined complex (energy, grains, etc.).
- `ScenarioAggregator` weighted consensus.
- Rolling-origin calibration audit.

**Phase 2 (live)** — ML providers via split-conformal prediction:
- `Conformal[ElasticNet]`, `Conformal[RandomForest]`, `Conformal[XGBoost]`.
- Signed-residual quantiles on each model's holdout, widened by √h across the horizon.

**Phase 3 (live)** — Deep providers:
- `MCDropout[LSTM]`: ~100 forward passes with dropout active at inference; √h widening.
- `TFT[quantile]`: native q10/q50/q90 from the Temporal Fusion Transformer (no resampling).
- `Prophet`: native log-price intervals with `interval_width = bull_q − bear_q`.

**Phase 4 (live)** — Quantum + causal ripple:
- `Quantum[KernelRidge]`: fidelity-kernel ridge regressor + split-conformal calibration on a 4-qubit angle-encoded feature map.
- `CausalRipple` engine: propagates the consensus band to any peer commodity in the same VAR complex via per-horizon IRF coupling coefficients.

**Phase 5 (live)** — Investor UX polish:
- Auto-narrative summary at top of page (direction, magnitudes, peer couplings).
- 📖 Glossary expander covering every term on the page.
- Stress test: force the consensus onto its bear/bull path with certainty and propagate across the complex.
- Historical analog overlay: top-K most-similar past 20-day windows shown both on the fan chart and as a tabular cross-check against the consensus forecast.
**Phase 4** — Quantum measurement distribution; causal-chain ripple of scenarios across correlated commodities.
**Phase 5** — Stress-test toggle, model-attribution heatmap, historical-analog overlay.
        """
    )
