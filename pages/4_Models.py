"""
Models & Analytics — multi-tier predictive model dashboard.

Tabs
----
1  Statistical    ARIMA forecast · GARCH vol · Kalman hedge ratio · VAR Granger
2  ML Models      HMM regime detector · XGBoost+SHAP · Random Forest · ElasticNet
3  Market Signals DXY / VIX / TLT macro · WASDE/OPEC calendar · ENSO · Energy transition
4  Deep Learning  LSTM · Prophet · TFT  (heavy deps — graceful fallback)
5  Quantum        Kernel benchmark · QAOA · QNN  (teaser)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

st.set_page_config(page_title="Models | Commodities", page_icon="🤖", layout="wide")

# ── Dark-theme helpers ─────────────────────────────────────────────────────────
BG       = "#0E1117"
PLOT_BG  = "#1C2333"
GRID     = "#2C3347"
AMBER    = "#F5A623"
BLUE     = "#4FC3F7"
GREEN    = "#66BB6A"
RED      = "#EF5350"
PURPLE   = "#AB47BC"

def dark_layout(fig, height=420, title=""):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
        font_color="#FAFAFA", title_text=title,
        height=height, margin=dict(t=50, l=55, r=20, b=40),
        legend=dict(bgcolor=PLOT_BG, borderwidth=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

REGIME_COLORS = {
    "Bull":     GREEN,
    "Neutral":  BLUE,
    "Bear":     RED,
    "High-Vol": PURPLE,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Commodities Hub")
    st.caption("Future of Commodities Club")
    st.divider()
    st.markdown("""
**Navigation**
- 🏠 [Overview](/)
- 💰 [Pricing](/Pricing)
- 📊 [Charts](/Charts)
- 📰 [News](/News)
- 🤖 **Models** ← you are here
    """)
    st.divider()

st.title("🤖 Analytics & Predictive Models")
st.caption("Live model inference on real market data — Spearman IC is the primary accuracy metric")

# ── Data loading ───────────────────────────────────────────────────────────────
CORE_TICKERS = {
    "WTI Crude Oil":           "CL=F",
    "Brent Crude Oil":         "BZ=F",
    "Natural Gas (Henry Hub)": "NG=F",
    "Gasoline (RBOB)":         "RB=F",
    "Heating Oil":             "HO=F",
    "Gold (COMEX)":            "GC=F",
    "Silver (COMEX)":          "SI=F",
    "Copper (COMEX)":          "HG=F",
    "Corn (CBOT)":             "ZC=F",
    "Wheat (CBOT SRW)":        "ZW=F",
    "Soybeans (CBOT)":         "ZS=F",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(period="3y"):
    import yfinance as yf
    raw = yf.download(
        list(CORE_TICKERS.values()), period=period,
        interval="1d", progress=False, auto_adjust=True,
    )
    prices = raw["Close"].rename(columns={v: k for k, v in CORE_TICKERS.items()})
    return prices.ffill().dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices_db(period="3y"):
    """Load all 41 commodities from SQLite — used exclusively by the VAR section."""
    import sqlite3 as _sql
    _conn = _sql.connect(os.path.join(os.path.dirname(__file__), "..", "data", "commodities.db"))
    _raw = pd.read_sql(
        "SELECT c.name, ph.date, ph.close FROM price_history ph "
        "JOIN commodities c ON c.id = ph.commodity_id WHERE ph.interval = '1d' ORDER BY ph.date",
        _conn,
    )
    _conn.close()
    _df = _raw.pivot(index="date", columns="name", values="close")
    _df.index = pd.to_datetime(_df.index)
    _df = _df.sort_index().ffill(limit=3)
    if period.endswith("y"):
        _cutoff = _df.index.max() - pd.DateOffset(years=int(period[:-1]))
        _df = _df[_df.index >= _cutoff]
    return _df

@st.cache_data(ttl=3600, show_spinner=False)
def load_macro(period="2y", _price_index=None):
    from features.macro_overlays import build_macro_overlay_features
    try:
        return build_macro_overlay_features(period=period, index=_price_index)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_energy_transition(period="2y"):
    from features.energy_transition import build_energy_transition_features
    try:
        return build_energy_transition_features(period=period)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def load_enso():
    from features.climate_weather import fetch_mei
    try:
        return fetch_mei(lag_months=[3, 6])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_features(_prices):
    from models.features import build_feature_matrix, build_target
    feat = build_feature_matrix(_prices)
    return feat

with st.spinner("Loading market data…"):
    prices = load_prices()

returns = np.log(prices / prices.shift(1)).dropna()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Statistical",
    "🌲 ML Models",
    "🌍 Market Signals",
    "🧠 Deep Learning",
    "⚛️ Quantum",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STATISTICAL MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📐 Statistical Models")
    st.caption(
        "Classical econometric models: ARIMA short-term forecasting, "
        "GARCH volatility, Kalman dynamic hedge ratio, and VAR Granger causality."
    )

    col_sel, col_per = st.columns([2, 1])
    with col_sel:
        stat_commodity = st.selectbox(
            "Commodity", list(CORE_TICKERS.keys()), index=0, key="stat_comm"
        )
    with col_per:
        stat_horizon = st.slider("ARIMA forecast horizon (days)", 1, 30, 5, key="stat_hor")

    st.divider()

    # ── ARIMA ──────────────────────────────────────────────────────────────────
    with st.spinner(f"Running ARIMA on {stat_commodity}…"):
        try:
            from models.statistical.arima import ARIMAForecaster
            arima = ARIMAForecaster()
            arima.fit(prices[stat_commodity])
            arima_result = arima.forecast(steps=stat_horizon)

            close = prices[stat_commodity]
            hist_tail = close.tail(60)

            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail,
                name="Historical", line=dict(color=AMBER, width=1.8),
            ))

            # Build forecast index (business days)
            last_date  = close.index[-1]
            fcast_idx  = pd.bdate_range(start=last_date, periods=stat_horizon + 1)[1:]
            fc_vals    = arima_result["forecast"]
            lo95       = arima_result["lower_95"]
            hi95       = arima_result["upper_95"]

            fig_arima.add_trace(go.Scatter(
                x=fcast_idx, y=fc_vals,
                name=f"ARIMA{arima_result['order']} forecast",
                line=dict(color=BLUE, width=2, dash="dot"),
                mode="lines+markers", marker=dict(size=5),
            ))
            fig_arima.add_trace(go.Scatter(
                x=list(fcast_idx) + list(fcast_idx[::-1]),
                y=list(hi95) + list(lo95[::-1]),
                fill="toself",
                fillcolor="rgba(79,195,247,0.12)",
                line=dict(width=0),
                name="95% CI", showlegend=True,
            ))
            dark_layout(fig_arima, height=400,
                        title=f"{stat_commodity} — ARIMA{arima_result['order']} {stat_horizon}d Forecast")
            st.plotly_chart(fig_arima, width='stretch')

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Order", str(arima_result["order"]))
            c2.metric("AIC", f"{arima_result['aic']:.1f}")
            c3.metric(f"Day-1 Forecast", f"${fc_vals[0]:,.2f}")
            c4.metric(
                f"Day-{stat_horizon} Forecast", f"${fc_vals[-1]:,.2f}",
                delta=f"{(fc_vals[-1]/close.iloc[-1]-1)*100:+.2f}% vs today",
            )
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")

    st.divider()

    # ── GARCH ──────────────────────────────────────────────────────────────────
    st.markdown("#### GARCH Volatility Model")
    with st.spinner(f"Fitting GARCH on {stat_commodity}…"):
        try:
            from models.statistical.garch import GARCHForecaster
            garch = GARCHForecaster()
            pct_returns = (np.log(prices[stat_commodity] / prices[stat_commodity].shift(1)) * 100).dropna()
            garch.fit(pct_returns)
            vol_df = garch.forecast_volatility(horizon=21)
            lev    = garch.leverage_effect()
            params = garch.parameter_summary()

            in_sample  = vol_df[vol_df.index <= prices.index[-1]].tail(252)
            out_sample = vol_df[vol_df.index > prices.index[-1]]

            fig_garch = go.Figure()
            fig_garch.add_trace(go.Scatter(
                x=in_sample.index, y=in_sample["conditional_vol_ann"],
                name="GARCH cond. vol (ann. %)", line=dict(color=AMBER, width=1.5),
            ))
            fig_garch.add_trace(go.Scatter(
                x=in_sample.index, y=in_sample["realized_vol_ann"],
                name="Realized vol (21d)", line=dict(color=BLUE, width=1.2, dash="dash"),
            ))
            if not out_sample.empty:
                fig_garch.add_trace(go.Scatter(
                    x=out_sample.index, y=out_sample["conditional_vol_ann"],
                    name="Forecast vol", line=dict(color=GREEN, width=2, dash="dot"),
                    mode="lines+markers", marker=dict(size=5),
                ))
            dark_layout(fig_garch, height=380,
                        title=f"{stat_commodity} — {garch.model} Conditional Volatility (Annualised %)")
            st.plotly_chart(fig_garch, width='stretch')

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Model", garch.model)
            g2.metric("Current Cond. Vol", f"{in_sample['conditional_vol_ann'].iloc[-1]:.1f}%")
            g3.metric("21d Fwd Vol (avg)", f"{out_sample['conditional_vol_ann'].mean():.1f}%")
            g4.metric("Leverage (γ)", f"{lev:.4f}" if lev is not None else "N/A",
                      help="GJR-GARCH asymmetry: negative γ = bad news raises vol more than good news")

            with st.expander("Model Parameters"):
                st.dataframe(params.style.format("{:.6f}"), width='stretch')
        except Exception as e:
            st.warning(f"GARCH failed: {e}")

    st.divider()

    # ── Kalman Hedge Ratio ──────────────────────────────────────────────────────
    st.markdown("#### Kalman Dynamic Hedge Ratio")
    st.caption("Time-varying β between a commodity pair — signals mean-reversion in the spread.")

    from models.statistical.kalman import HEDGE_PAIRS, KalmanHedgeRatio

    pair_options = list(HEDGE_PAIRS.keys()) + ["Custom…"]
    pair_choice  = st.selectbox("Pair", pair_options, key="kalman_pair")

    if pair_choice == "Custom…":
        available_cols = sorted(prices.columns.tolist())
        c1, c2 = st.columns(2)
        y_name     = c1.selectbox("Y leg (long)", available_cols, key="kalman_y")
        x_default  = 1 if len(available_cols) > 1 else 0
        x_name     = c2.selectbox("X leg (hedge)", available_cols, index=x_default, key="kalman_x")
        pair_label = f"{y_name} / {x_name}"
    else:
        y_name, x_name = HEDGE_PAIRS[pair_choice]
        pair_label = pair_choice

    zscore_window = st.slider(
        "Z-score lookback (days)", min_value=21, max_value=126, value=63, step=7,
        key="kalman_zscore_window",
        help="Rolling window for spread mean and std. Shorter = more sensitive; longer = more stable.",
    )

    if y_name == x_name:
        st.info("Select two different instruments.")
    elif y_name not in prices.columns or x_name not in prices.columns:
        st.info("Price data for this pair not fully loaded. Try another pair.")
    else:
        with st.spinner("Running Kalman filter…"):
            try:
                kf = KalmanHedgeRatio(pair=pair_label)
                kf.fit(prices[y_name], prices[x_name])
                beta          = kf.hedge_ratios()
                beta_lo, beta_hi = kf.beta_ci()
                zscore        = kf.spread_zscore(window=zscore_window)
                info          = kf.hedge_info()
                coint_p       = kf.cointegration_pvalue()

                # Cointegration badge
                import math
                if not math.isnan(coint_p):
                    if coint_p < 0.05:
                        st.success(
                            f"Cointegrated — Engle-Granger p = {coint_p:.3f} "
                            f"(< 0.05). Spread mean-reversion is statistically supported."
                        )
                    elif coint_p < 0.10:
                        st.warning(
                            f"Weak cointegration — Engle-Granger p = {coint_p:.3f} "
                            f"(0.05–0.10). Treat signals with caution."
                        )
                    else:
                        st.error(
                            f"Not cointegrated — Engle-Granger p = {coint_p:.3f} "
                            f"(> 0.10). Hedge ratio and spread may be spurious."
                        )

                fig_kf = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.55, 0.45], vertical_spacing=0.06,
                    subplot_titles=["Dynamic Hedge Ratio (β) with 95% CI", "Spread Z-Score"])

                # 95% CI band — upper boundary (invisible line, used as fill target)
                fig_kf.add_trace(go.Scatter(
                    x=beta_hi.index, y=beta_hi,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                    name="β upper CI"), row=1, col=1)
                # Lower boundary with fill back to upper
                fig_kf.add_trace(go.Scatter(
                    x=beta_lo.index, y=beta_lo,
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(245,158,11,0.15)",
                    showlegend=False, hoverinfo="skip",
                    name="β lower CI"), row=1, col=1)
                # β central estimate
                fig_kf.add_trace(go.Scatter(
                    x=beta.index, y=beta,
                    line=dict(color=AMBER, width=1.5), name="β (hedge ratio)"), row=1, col=1)
                fig_kf.add_hline(y=float(beta.mean()), line_dash="dash",
                                 line_color=BLUE, opacity=0.6, row=1, col=1)

                fig_kf.add_trace(go.Scatter(
                    x=zscore.index, y=zscore,
                    line=dict(color=BLUE, width=1.5), name="z-score"), row=2, col=1)
                fig_kf.add_hline(y=2.0,  line_dash="dot", line_color=RED,    opacity=0.6, row=2, col=1)
                fig_kf.add_hline(y=-2.0, line_dash="dot", line_color=GREEN,  opacity=0.6, row=2, col=1)
                fig_kf.add_hline(y=0.0,  line_dash="dash", line_color="#888", opacity=0.4, row=2, col=1)
                dark_layout(fig_kf, height=480, title=f"Kalman Filter — {pair_label}")
                st.plotly_chart(fig_kf, width='stretch')

                k1, k2, k3, k4 = st.columns(4)
                z_now  = float(zscore.dropna().iloc[-1])
                ci_now = float(beta_hi.iloc[-1]) - float(beta_lo.iloc[-1])
                k1.metric("Current β", f"{info['current_beta']:.3f}")
                k2.metric("Spread Z-Score", f"{z_now:.2f}")
                signal = "Short spread" if z_now > 2 else ("Long spread" if z_now < -2 else "Neutral")
                k3.metric("Signal", signal)
                k4.metric("β 95% CI width", f"{ci_now:.3f}",
                          help="Width of the Kalman uncertainty band. Wider = less confident in current β.")
            except Exception as e:
                st.warning(f"Kalman failed: {e}")

    st.divider()

    # ── VAR Granger Causality ──────────────────────────────────────────────────
    st.markdown("#### VAR Granger Causality")
    st.caption("Does commodity A's history help predict commodity B? Low p-value = yes.")
    st.warning(
        "**Methodology note:** Granger causality detects lead-lag *relationships* between price series — "
        "it does not imply economic causation and is **not a directional trading signal**. "
        "Walk-forward backtesting shows VAR 1-step directional accuracy is near-random (28–43%) "
        "for liquid commodity futures. Use these outputs to understand *structural linkages*, "
        "not to forecast tomorrow's price move.",
        icon="⚠️",
    )

    VAR_GROUPS = {
        "Energy (WTI, Brent, NG, RBOB, HO)":              ("energy",            None),
        "Energy Transition (LNG, Coal, Uranium, Carbon)":  ("energy_transition",  None),
        "Precious Metals (Gold, Silver, Pt, Pd)":          ("precious_metals",    None),
        "Industrial Metals (Cu, Al, Steel, Iron, Zn)":     ("industrial_metals",  None),
        "Grains (Corn, Wheat, Soybeans)":                  ("grains",             None),
        "Ag Extended (Soy Oil/Meal, Oats, Rice, KC Wht)":  ("ag_extended",        None),
        "Livestock (Cattle, Hogs)":                        ("livestock",          None),
        "Softs (Coffee, Sugar, Cotton, Cocoa, OJ)":        ("softs",              None),
        "New Commodities (Lithium, Rare Earths, Bitcoin)": ("new_commodities",    None),
    }
    var_group = st.selectbox("Commodity group", list(VAR_GROUPS.keys()), key="var_group")
    g_key, _ = VAR_GROUPS[var_group]

    # Shared color palette — defined here so both IRF and FEVD blocks can access it
    VAR_COLORS = [AMBER, BLUE, GREEN, RED, PURPLE, "#26C6DA", "#FFA726", "#EC407A",
                  "#7E57C2", "#26A69A", "#EF9A9A", "#80CBC4"]

    with st.spinner("Fitting VAR…"):
        try:
            from models.statistical.var_vecm import CommodityVAR
            prices_db = load_prices_db()
            var = CommodityVAR(group=g_key)
            var.fit(prices_db)

            # ── Granger causality heatmap ──────────────────────────────────────
            gc = var.granger_causality(max_lag=5)
            # Shorten labels for readability on the heatmap
            short = {c: c.replace(" (COMEX)", "").replace(" (CBOT)", "")
                       .replace(" (Henry Hub)", "").replace(" (FCOJ-A)", "")
                       .replace("*", "").strip()
                     for c in gc.columns}
            gc_plot = gc.rename(index=short, columns=short)
            hmap_h = max(350, 80 * len(var._cols))
            fig_gc = px.imshow(
                gc_plot.astype(float),
                color_continuous_scale="RdYlGn_r",
                zmin=0, zmax=0.1,
                text_auto=".3f",
                title="Granger Causality p-values — row caused by column (min p across lags 1–5)",
            )
            fig_gc.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font_color="#FAFAFA", height=hmap_h,
                margin=dict(t=55, l=120, r=20, b=100),
                coloraxis_colorbar=dict(title="p-value"),
            )
            fig_gc.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_gc, use_container_width=True)

            v1, v2, v3 = st.columns(3)
            v1.metric("VAR Lag Order", var._lag_order)
            v2.metric("Commodities in system", len(var._cols))
            sig_total = int((gc < 0.05).sum().sum())
            total_pairs = int(gc.count().sum())
            v3.metric("Significant pairs (p<0.05)", f"{sig_total} / {total_pairs}")

            # ── Dynamic economic interpretation ───────────────────────────────
            interp_lines = []

            def _p(row, col):
                try:
                    return gc.loc[row, col]
                except KeyError:
                    return float("nan")

            def _sig_causes(effect, threshold=0.05):
                """Return list of commodities that significantly Granger-cause `effect`."""
                return [c for c in gc.columns if c != effect and _p(effect, c) < threshold]

            def _sig_effects(cause, threshold=0.05):
                """Return list of commodities significantly caused by `cause`."""
                return [r for r in gc.index if r != cause and _p(r, cause) < threshold]

            if g_key == "energy":
                wti_ng = _p("Natural Gas", "WTI Crude Oil")
                ng_wti = _p("WTI Crude Oil", "Natural Gas")
                if wti_ng > 0.05 and ng_wti > 0.05:
                    interp_lines.append(
                        "⚡ **WTI and Natural Gas show no significant lead-lag relationship** — "
                        "reflects their structural decoupling post-2022 as U.S. LNG export capacity "
                        "expanded. Natural Gas now trades on global supply dynamics, not oil signals."
                    )
                if _p("Brent Crude Oil", "WTI Crude Oil") < 0.05:
                    interp_lines.append(
                        "🛢️ **WTI Granger-causes Brent** — U.S. crude sets the short-run benchmark; "
                        "Brent adjusts with a measurable lag."
                    )
                refined = [n for n, col in [("RBOB Gasoline", "Gasoline (RBOB)"), ("Heating Oil", "Heating Oil")]
                           if _p(col, "WTI Crude Oil") < 0.05]
                if refined:
                    interp_lines.append(
                        f"⛽ **WTI Granger-causes {', '.join(refined)}** — crude feedstock costs "
                        "propagate into refined product prices within 1–5 trading days."
                    )

            elif g_key == "energy_transition":
                coal_pair = _p("Metallurgical Coal*", "Thermal Coal*")
                if coal_pair < 0.05:
                    interp_lines.append(
                        "⛏️ **Thermal Coal Granger-causes Met Coal** — thermal price signals "
                        "spill across coal grades as miners and buyers substitute between end uses."
                    )
                if _p("Carbon Credits*", "Thermal Coal*") < 0.05 or _p("Thermal Coal*", "Carbon Credits*") < 0.05:
                    interp_lines.append(
                        "🌿 **Carbon Credits and Thermal Coal are linked** — carbon pricing "
                        "incentivizes fuel-switching, creating a feedback loop between EUA prices "
                        "and coal demand."
                    )
                u_causes = _sig_causes("Uranium*")
                if not u_causes:
                    interp_lines.append(
                        "☢️ **Uranium trades independently** of coal and carbon — driven by "
                        "long-term utility contracting cycles rather than short-run energy prices."
                    )
                if _p("Lumber*", "Carbon Credits*") < 0.05:
                    interp_lines.append(
                        "🌲 **Carbon Credits Granger-cause Lumber** — forest carbon sequestration "
                        "values influence timber harvest decisions."
                    )

            elif g_key == "precious_metals":
                gold_to_silver = _p("Silver", "Gold")
                silver_to_gold = _p("Gold", "Silver")
                if gold_to_silver < 0.05:
                    interp_lines.append(
                        "🥇 **Gold Granger-causes Silver** — institutional gold flows lead; "
                        "silver follows as the high-beta precious metals proxy."
                    )
                if silver_to_gold < 0.05:
                    interp_lines.append(
                        "🥈 **Silver Granger-causes Gold** — industrial demand signals "
                        "in silver can anticipate broader precious metals moves."
                    )
                pd_indep = not _sig_causes("Palladium")
                pt_indep = not _sig_causes("Platinum")
                if pd_indep:
                    interp_lines.append(
                        "🚗 **Palladium trades independently** — dominated by auto catalyst demand "
                        "(ICE vehicles) and Russian supply risk rather than monetary metal flows."
                    )
                phys_gold = _p("Gold (Physical/London)*", "Gold")
                if phys_gold < 0.05:
                    interp_lines.append(
                        "🏦 **Spot Gold Granger-causes the Physical ETF** — futures price "
                        "discovery leads the London AM/PM fix."
                    )

            elif g_key == "industrial_metals":
                cu_effects = _sig_effects("Copper")
                if cu_effects:
                    targets = ", ".join(c.replace("*", "").strip() for c in cu_effects)
                    interp_lines.append(
                        f"🔧 **Copper Granger-causes {targets}** — copper's role as a macro "
                        "bellwether ('Dr. Copper') propagates demand signals across the base metals complex."
                    )
                if _p("HRC Steel", "Iron Ore / Steel*") < 0.05 or _p("Iron Ore / Steel*", "HRC Steel") < 0.05:
                    interp_lines.append(
                        "🏗️ **HRC Steel and Iron Ore are causally linked** — steelmaking input "
                        "costs (iron ore, coking coal) feed through to hot-rolled coil prices."
                    )
                zn_indep = not _sig_causes("Zinc & Cobalt*")
                if zn_indep:
                    interp_lines.append(
                        "🔋 **Zinc & Cobalt trade independently** — battery metal dynamics "
                        "and galvanizing demand are driven by separate supply chains."
                    )

            elif g_key == "grains":
                corn_effects = _sig_effects("Corn")
                if corn_effects:
                    targets_str = " and ".join(f"**{r.split('(')[0].strip()}**" for r in corn_effects)
                    interp_lines.append(
                        f"🌽 **Corn Granger-causes {targets_str}** — shared acreage competition "
                        "and feed-cost dynamics make corn the primary grain complex driver. "
                        "FEVD shows corn accounts for ~20–24% of Wheat and Soybean forecast variance."
                    )
                if _p("Soybeans", "Wheat") < 0.05:
                    interp_lines.append(
                        "🌾 **Wheat Granger-causes Soybeans** — crop rotation and substitution "
                        "signals in global export markets."
                    )
                if _p("Corn", "Soybeans") < 0.05:
                    interp_lines.append(
                        "🫘 **Soybeans Granger-cause Corn** — the soy/corn price ratio is a "
                        "key acreage-switching signal watched by commodity traders."
                    )

            elif g_key == "ag_extended":
                crush = _p("Soybean Meal", "Soybean Oil") < 0.05 or _p("Soybean Oil", "Soybean Meal") < 0.05
                if crush:
                    interp_lines.append(
                        "🫘 **Soybean Oil and Soybean Meal are causally linked** — both are "
                        "crush co-products; the soy crush spread drives processing margins and "
                        "creates feedback between the two."
                    )
                wheat_link = _p("Wheat (KC HRW)", "Oats (CBOT)") < 0.05 or _p("Wheat", "Wheat (KC HRW)") < 0.05
                if wheat_link:
                    interp_lines.append(
                        "🌾 **Inter-wheat price discovery detected** — CBOT SRW and KC HRW "
                        "wheat trade on different protein/quality specs but share export demand signals."
                    )
                rice_indep = not _sig_causes("Rough Rice (CBOT)")
                if rice_indep:
                    interp_lines.append(
                        "🍚 **Rough Rice trades independently** — primarily driven by Asian "
                        "demand and weather in Southeast Asia, decoupled from U.S. grain complex dynamics."
                    )

            elif g_key == "livestock":
                fc_lc = _p("Feeder Cattle", "Live Cattle")
                lc_fc = _p("Live Cattle", "Feeder Cattle")
                if fc_lc < 0.05:
                    interp_lines.append(
                        "🐄 **Live Cattle Granger-causes Feeder Cattle** — finished cattle prices "
                        "set the ceiling for what feedlots will pay for feeder cattle, "
                        "creating a downstream pricing cascade."
                    )
                if lc_fc < 0.05:
                    interp_lines.append(
                        "🐂 **Feeder Cattle Granger-causes Live Cattle** — input cost signals "
                        "from the feeder market propagate forward to finished cattle prices."
                    )
                hogs_indep = not _sig_causes("Lean Hogs")
                if hogs_indep:
                    interp_lines.append(
                        "🐷 **Lean Hogs trade independently of cattle** — pork and beef supply "
                        "chains are largely separate, with hog prices driven by packer margins "
                        "and feed costs independent of the cattle cycle."
                    )

            elif g_key == "softs":
                coffee_sugar = _p("Sugar", "Coffee") < 0.05 or _p("Coffee", "Sugar") < 0.05
                if coffee_sugar:
                    interp_lines.append(
                        "☕ **Coffee and Sugar are causally linked** — both are tropical crops "
                        "heavily exposed to Brazilian weather and Real/USD FX, creating "
                        "correlated supply shocks."
                    )
                cocoa_indep = not _sig_causes("Cocoa")
                if cocoa_indep:
                    interp_lines.append(
                        "🍫 **Cocoa trades independently** — West African supply concentration "
                        "(Ghana/Ivory Coast ~60% of global output) means price discovery "
                        "is driven by regional crop conditions rather than broader soft commodity signals."
                    )
                oj_indep = not _sig_causes("Orange Juice (FCOJ-A)")
                if oj_indep:
                    interp_lines.append(
                        "🍊 **Orange Juice trades independently** — Florida and Brazil citrus "
                        "greening disease and USDA crop estimates dominate price discovery, "
                        "disconnected from other softs."
                    )

            elif g_key == "new_commodities":
                btc_effects = _sig_effects("Bitcoin")
                li_re = _p("Rare Earths*", "Lithium*") < 0.05 or _p("Lithium*", "Rare Earths*") < 0.05
                if li_re:
                    interp_lines.append(
                        "🔋 **Lithium and Rare Earths are causally linked** — both are critical "
                        "battery and EV supply chain inputs; demand signals from one propagate to the other."
                    )
                if btc_effects:
                    targets = ", ".join(c.replace("*", "").strip() for c in btc_effects)
                    interp_lines.append(
                        f"₿ **Bitcoin Granger-causes {targets}** — risk-appetite spillover: "
                        "Bitcoin acts as a high-beta risk asset; moves can lead commodity ETFs "
                        "that share speculative investor bases (Lithium, Rare Earths)."
                    )
                btc_causes = _sig_causes("Bitcoin")
                if not btc_causes and not btc_effects:
                    interp_lines.append(
                        "₿ **Bitcoin trades independently** of Lithium and Rare Earths — "
                        "crypto price discovery is dominated by on-chain flows and macro "
                        "liquidity rather than physical commodity fundamentals."
                    )

            if not interp_lines:
                interp_lines.append(
                    f"No dominant lead-lag relationships detected at p < 0.05 "
                    f"({sig_total}/{total_pairs} pairs significant). "
                    "This may reflect efficient price discovery, a structural break in the "
                    "sample window, or lag dynamics beyond 5 days for this group."
                )

            st.info("\n\n".join(interp_lines))

            st.divider()

            # ── IRF line chart ─────────────────────────────────────────────────
            st.markdown("##### Impulse Response Functions (10-day horizon)")
            st.caption("One-standard-deviation shock to each commodity → response across the system.")
            try:
                irf = var.impulse_response(steps=10)
                shock_names = irf.columns.get_level_values("shock").unique().tolist()
                n_shocks = len(shock_names)
                from plotly.subplots import make_subplots as _make_subplots
                irf_cols = min(n_shocks, 3)
                irf_rows = (n_shocks + irf_cols - 1) // irf_cols
                fig_irf = _make_subplots(
                    rows=irf_rows, cols=irf_cols,
                    subplot_titles=[s.replace("*", "").split("(")[0].strip() for s in shock_names],
                    shared_yaxes=False,
                )
                for s_idx, shock in enumerate(shock_names):
                    r, c = divmod(s_idx, irf_cols)
                    for r_idx, resp in enumerate(irf[shock].columns.tolist()):
                        fig_irf.add_trace(
                            go.Scatter(
                                x=list(range(11)), y=irf[(shock, resp)].values,
                                mode="lines",
                                name=resp.replace("*", "").split("(")[0].strip(),
                                line=dict(color=VAR_COLORS[r_idx % len(VAR_COLORS)], width=2),
                                showlegend=(s_idx == 0),
                            ),
                            row=r + 1, col=c + 1,
                        )
                    fig_irf.add_hline(y=0, line_dash="dash", line_color="#555",
                                      row=r + 1, col=c + 1)
                fig_irf.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
                    font_color="#FAFAFA", height=280 * irf_rows,
                    margin=dict(t=50, l=50, r=20, b=40),
                    legend=dict(bgcolor=PLOT_BG, borderwidth=0),
                )
                fig_irf.update_xaxes(title_text="Days", gridcolor=GRID)
                fig_irf.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
                st.plotly_chart(fig_irf, use_container_width=True)
            except Exception as e:
                st.caption(f"IRF unavailable: {e}")

            st.divider()

            # ── FEVD stacked bar chart ─────────────────────────────────────────
            st.markdown("##### Forecast Error Variance Decomposition (10-day horizon)")
            st.caption("What fraction of each commodity's forecast error originates from shocks in other commodities?")
            try:
                fevd = var.fevd(steps=10)
                def _short(name):
                    return name.replace("*", "").replace(" (COMEX)", "").replace(" (CBOT)", "") \
                               .replace(" (Henry Hub)", "").replace(" (FCOJ-A)", "").strip()
                fevd_plot = fevd.copy()
                fevd_plot.index = [_short(c) for c in fevd_plot.index]
                fevd_plot.columns = [_short(c) for c in fevd_plot.columns]
                fig_fevd = go.Figure()
                for i, shock_col in enumerate(fevd_plot.columns):
                    fig_fevd.add_trace(go.Bar(
                        name=shock_col,
                        x=fevd_plot.index.tolist(),
                        y=(fevd_plot[shock_col] * 100).round(1).tolist(),
                        marker_color=VAR_COLORS[i % len(VAR_COLORS)],
                    ))
                fig_fevd.update_layout(
                    barmode="stack",
                    paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
                    font_color="#FAFAFA", height=360,
                    margin=dict(t=30, l=50, r=20, b=100),
                    legend=dict(bgcolor=PLOT_BG, borderwidth=0, title="Shock source"),
                    xaxis=dict(title="Commodity", gridcolor=GRID, tickangle=-40),
                    yaxis=dict(title="% of forecast error variance", gridcolor=GRID),
                )
                st.plotly_chart(fig_fevd, use_container_width=True)
            except Exception as e:
                st.caption(f"FEVD unavailable: {e}")

        except Exception as e:
            st.warning(f"VAR failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌲 Machine Learning Models")
    st.caption(
        "Regime detection, tree ensembles, SHAP explainability, and sparse factor models. "
        "Primary accuracy metric: Spearman IC (Information Coefficient). IC > 0.05 = actionable."
    )

    ml_commodity = st.selectbox(
        "Target commodity", list(CORE_TICKERS.keys()), index=0, key="ml_comm"
    )

    with st.spinner("Building feature matrix…"):
        feat_matrix = get_features(prices)
        from models.features import build_target
        ml_target = build_target(prices, ml_commodity)

    st.divider()

    # ── HMM Regime Detector ────────────────────────────────────────────────────
    st.markdown("#### HMM Market Regime Detector")
    st.caption("Gaussian HMM on (return, volatility) pairs — 4-state: Bear / Neutral / High-Vol / Bull")

    hmm_commodity = st.selectbox(
        "Commodity for regime detection",
        list(CORE_TICKERS.keys()), index=0, key="hmm_comm"
    )

    with st.spinner(f"Fitting HMM on {hmm_commodity}…"):
        try:
            from models.ml.hmm_regime import HMMRegimeDetector
            hmm = HMMRegimeDetector(n_states=4)
            hmm.fit(returns[hmm_commodity])
            regime_series = hmm.regime_series()
            probs = hmm.state_probabilities()
            counts = regime_series.value_counts()

            # Color-coded price chart
            fig_hmm = go.Figure()
            for regime, color in REGIME_COLORS.items():
                mask = regime_series == regime
                dates = prices.index[prices.index.isin(regime_series[mask].index)]
                if len(dates) == 0:
                    continue
                fig_hmm.add_trace(go.Scatter(
                    x=dates, y=prices[hmm_commodity].reindex(dates),
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.7),
                    name=regime,
                ))
            dark_layout(fig_hmm, height=380,
                        title=f"{hmm_commodity} Price — Colour-coded by HMM Regime")
            st.plotly_chart(fig_hmm, width='stretch')

            # State probabilities heatmap (last 126 days)
            prob_tail = probs.tail(126)
            fig_prob = px.imshow(
                prob_tail.T,
                color_continuous_scale="Viridis",
                aspect="auto",
                labels=dict(x="Date", y="Regime", color="Probability"),
                title="State Probabilities (last 126 trading days)",
            )
            fig_prob.update_layout(
                paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
                font_color="#FAFAFA", height=250,
                margin=dict(t=40, l=80, r=20, b=40),
            )
            st.plotly_chart(fig_prob, width='stretch')

            cols = st.columns(len(counts))
            for col_widget, (regime, cnt) in zip(cols, counts.items()):
                pct = round(100 * cnt / len(regime_series), 1)
                col_widget.metric(regime, f"{pct}%", f"{cnt} days")

            current_regime = regime_series.iloc[-1]
            st.info(f"**Current regime ({regime_series.index[-1].date()}):** {current_regime}")
        except Exception as e:
            st.warning(f"HMM failed: {e}")

    st.divider()

    # ── XGBoost + SHAP ─────────────────────────────────────────────────────────
    st.markdown("#### XGBoost + SHAP Explainability")
    st.caption(
        "Gradient boosted trees with SHAP TreeExplainer. "
        "Waterfall shows which features pushed today's forecast up or down."
    )

    with st.spinner(f"Training XGBoost on {ml_commodity}…"):
        try:
            from models.ml.xgboost_shap import XGBoostForecaster
            xgb = XGBoostForecaster(commodity=ml_commodity)
            xgb.fit(feat_matrix, ml_target)
            xgb_ic = xgb.ic_score(feat_matrix, ml_target)
            shap_imp = xgb.global_shap_importance(feat_matrix)
            waterfall = xgb.waterfall_data(feat_matrix)

            x1, x2 = st.columns(2)
            with x1:
                # Global SHAP importance bar chart
                top_shap = shap_imp.head(15)
                fig_shap = go.Figure(go.Bar(
                    x=top_shap["mean_abs_shap"],
                    y=top_shap["feature"],
                    orientation="h",
                    marker_color=AMBER,
                ))
                dark_layout(fig_shap, height=380, title="Global SHAP Feature Importance (Top 15)")
                fig_shap.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_shap, width='stretch')

            with x2:
                # Waterfall for latest observation
                feats  = waterfall["features"][:12]   # top 12 for readability
                names  = [f[0] for f in feats]
                values = [f[1] for f in feats]
                colors = [GREEN if v > 0 else RED for v in values]

                fig_wf = go.Figure(go.Bar(
                    x=values, y=names, orientation="h",
                    marker_color=colors,
                ))
                dark_layout(fig_wf, height=380,
                            title=f"SHAP Waterfall — {waterfall['date']}")
                fig_wf.update_layout(yaxis=dict(autorange="reversed"))
                fig_wf.add_vline(x=0, line_color="#888", line_width=1)
                st.plotly_chart(fig_wf, width='stretch')

            xc1, xc2, xc3 = st.columns(3)
            xc1.metric("Spearman IC", f"{xgb_ic:.4f}",
                       delta="Actionable" if xgb_ic > 0.05 else "Marginal")
            top_pos = waterfall["top_positive"][0][0] if waterfall["top_positive"] else "—"
            top_neg = waterfall["top_negative"][0][0] if waterfall["top_negative"] else "—"
            xc2.metric("Top bullish driver", top_pos)
            xc3.metric("Top bearish driver", top_neg)
        except Exception as e:
            st.warning(f"XGBoost/SHAP failed: {e}")

    st.divider()

    # ── Random Forest ──────────────────────────────────────────────────────────
    st.markdown("#### Random Forest — Feature Importance")
    with st.spinner(f"Training Random Forest on {ml_commodity}…"):
        try:
            from models.ml.random_forest import CommodityRF
            rf = CommodityRF(commodity=ml_commodity)
            rf.fit(feat_matrix, ml_target)
            rf_ic = rf.ic_score(feat_matrix, ml_target)
            rf_imp = rf.feature_importance(top_n=20)

            fig_rf = go.Figure(go.Bar(
                x=rf_imp["importance"],
                y=rf_imp["feature"],
                orientation="h",
                marker_color=BLUE,
            ))
            dark_layout(fig_rf, height=400, title=f"RF Gini Importance — {ml_commodity} (Top 20)")
            fig_rf.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rf, width='stretch')

            r1, r2, r3 = st.columns(3)
            r1.metric("Spearman IC", f"{rf_ic:.4f}",
                      delta="Actionable" if rf_ic > 0.05 else "Marginal")
            r2.metric("Top feature", rf_imp["feature"].iloc[0])
            r3.metric("Top feature importance", f"{rf_imp['importance_pct'].iloc[0]:.1f}%")
        except Exception as e:
            st.warning(f"Random Forest failed: {e}")

    st.divider()

    # ── Elastic Net ────────────────────────────────────────────────────────────
    st.markdown("#### Elastic Net / Lasso Factor Model")
    st.caption("Sparse linear model: Lasso regularisation selects the minimal driver set.")
    with st.spinner(f"Fitting Elastic Net on {ml_commodity}…"):
        try:
            from models.ml.elastic_net import ElasticNetFactorModel
            en = ElasticNetFactorModel(commodity=ml_commodity)
            en.fit(feat_matrix, ml_target)
            en_ic      = en.ic_score(feat_matrix, ml_target)
            coef_table = en.coefficient_table()
            sparsity   = en.sparsity()
            narrative  = en.factor_narrative(top_n=5)

            nonzero = coef_table[coef_table["coefficient"] != 0].copy()
            nonzero = nonzero.sort_values("coefficient", key=abs, ascending=False).head(20)
            colors  = [GREEN if c > 0 else RED for c in nonzero["coefficient"]]

            fig_en = go.Figure(go.Bar(
                x=nonzero["coefficient"],
                y=nonzero["feature"],
                orientation="h",
                marker_color=colors,
            ))
            dark_layout(fig_en, height=400, title=f"Elastic Net Coefficients — {ml_commodity}")
            fig_en.update_layout(yaxis=dict(autorange="reversed"))
            fig_en.add_vline(x=0, line_color="#888", line_width=1)
            st.plotly_chart(fig_en, width='stretch')

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Spearman IC", f"{en_ic:.4f}")
            e2.metric("Active features", sparsity["n_features_nonzero"])
            e3.metric("Sparsity", f"{sparsity['sparsity_pct']:.1f}%")
            e4.metric("α (regularisation)", f"{sparsity['alpha']:.4f}")

            st.info(f"**Factor narrative:** {narrative}")
        except Exception as e:
            st.warning(f"Elastic Net failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🌍 Forward-Looking Market Signals")
    st.caption(
        "Macro overlays (DXY / VIX / TLT), commodity calendar events (WASDE / OPEC+), "
        "climate signals (ENSO), and energy transition proxies."
    )

    # ── DXY / VIX / TLT ───────────────────────────────────────────────────────
    with st.spinner("Loading macro signals…"):
        macro_df = load_macro(period="2y", _price_index=prices.index)

    if not macro_df.empty:
        st.markdown("#### DXY · VIX · TLT Macro Overlay")

        macro_tail = macro_df.tail(252)

        fig_macro = make_subplots(rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.34, 0.33, 0.33], vertical_spacing=0.04,
            subplot_titles=["DXY Z-Score (63d)", "VIX (Fear Gauge)", "TLT 21d Momentum"])

        fig_macro.add_trace(go.Scatter(
            x=macro_tail.index, y=macro_tail["dxy_zscore63"],
            line=dict(color=AMBER, width=1.5), name="DXY z-score",
        ), row=1, col=1)
        fig_macro.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5, row=1, col=1)

        vix_vals = macro_tail["vix"]
        vix_colors = [RED if v >= 30 else (PURPLE if v >= 20 else GREEN) for v in vix_vals]
        fig_macro.add_trace(go.Bar(
            x=macro_tail.index, y=vix_vals,
            marker_color=vix_colors, name="VIX",
        ), row=2, col=1)
        fig_macro.add_hline(y=20, line_dash="dot", line_color=PURPLE, opacity=0.6, row=2, col=1)
        fig_macro.add_hline(y=30, line_dash="dot", line_color=RED,    opacity=0.6, row=2, col=1)

        fig_macro.add_trace(go.Scatter(
            x=macro_tail.index, y=macro_tail["tlt_mom21"],
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.12)",
            line=dict(color=BLUE, width=1.5), name="TLT 21d mom",
        ), row=3, col=1)
        fig_macro.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5, row=3, col=1)

        dark_layout(fig_macro, height=520, title="Macro Regime Dashboard")
        st.plotly_chart(fig_macro, width='stretch')

        vix_now = float(macro_df["vix"].dropna().iloc[-1])
        dxy_z   = float(macro_df["dxy_zscore63"].dropna().iloc[-1])
        tlt_m   = float(macro_df["tlt_mom21"].dropna().iloc[-1])
        wasde_d = int(macro_df["days_to_wasde"].dropna().iloc[-1])
        opec_d  = int(macro_df["days_to_opec"].dropna().iloc[-1])

        m1, m2, m3, m4, m5 = st.columns(5)
        vix_label = "Crisis" if vix_now >= 30 else ("Risk-Off" if vix_now >= 20 else "Risk-On")
        m1.metric("VIX", f"{vix_now:.1f}", delta=vix_label)
        m2.metric("DXY Z-Score", f"{dxy_z:+.2f}",
                  delta="Dollar strong" if dxy_z > 0.5 else ("Dollar weak" if dxy_z < -0.5 else "Neutral"))
        m3.metric("TLT 21d Mom", f"{tlt_m:+.3f}",
                  delta="Rates falling" if tlt_m > 0 else "Rates rising")
        m4.metric("Days to WASDE", f"{wasde_d:+d}",
                  delta="Window active" if macro_df["is_wasde_window"].iloc[-1] else "Outside window")
        m5.metric("Days to OPEC+", f"{opec_d:+d}",
                  delta="Window active" if macro_df["is_opec_window"].iloc[-1] else "Outside window")
    else:
        st.warning("Macro data unavailable — check internet connection.")

    st.divider()

    # ── ENSO / Climate ─────────────────────────────────────────────────────────
    st.markdown("#### ENSO Climate Signal (MEI v2)")
    st.caption(
        "Multivariate ENSO Index: El Niño (>+0.5) disrupts global precipitation, "
        "lagging Cocoa/Coffee 3-6 months; La Niña (<−0.5) stresses Brazil soy."
    )

    with st.spinner("Loading ENSO data…"):
        mei_df = load_enso()

    if not mei_df.empty:
        mei_tail = mei_df.tail(252 * 2)  # 2 years
        colors_enso = [
            RED if v >= 0.5 else (BLUE if v <= -0.5 else "#888")
            for v in mei_tail["mei"]
        ]
        fig_mei = go.Figure()
        fig_mei.add_trace(go.Bar(
            x=mei_tail.index, y=mei_tail["mei"],
            marker_color=colors_enso, name="MEI v2",
        ))
        fig_mei.add_hline(y=0.5,  line_dash="dot", line_color=RED,  opacity=0.7)
        fig_mei.add_hline(y=-0.5, line_dash="dot", line_color=BLUE, opacity=0.7)
        dark_layout(fig_mei, height=320, title="MEI v2 — El Niño (red) / La Niña (blue)")
        st.plotly_chart(fig_mei, width='stretch')

        enso_now = float(mei_df["mei"].dropna().iloc[-1])
        phase    = "🔴 El Niño" if enso_now >= 0.5 else ("🔵 La Niña" if enso_now <= -0.5 else "⚪ Neutral")
        lag3     = float(mei_df["mei_lag3m"].dropna().iloc[-1]) if "mei_lag3m" in mei_df.columns else np.nan
        lag6     = float(mei_df["mei_lag6m"].dropna().iloc[-1]) if "mei_lag6m" in mei_df.columns else np.nan

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Current MEI", f"{enso_now:+.2f}", delta=phase)
        ec2.metric("3-Month Lag Signal", f"{lag3:+.2f}" if not np.isnan(lag3) else "—")
        ec3.metric("6-Month Lag Signal", f"{lag6:+.2f}" if not np.isnan(lag6) else "—")

        if enso_now <= -0.5:
            st.info("**La Niña active:** Watch Brazil soy stress (2-4 month lead), "
                    "higher US natural gas demand from cooler winters, "
                    "and wetter SE Asia conditions for palm oil/cocoa.")
        elif enso_now >= 0.5:
            st.info("**El Niño active:** Watch drier SE Asia (palm oil/cocoa/coffee bearish), "
                    "wetter US Gulf Coast (US soy bullish), drier Australian wheat.")
    else:
        st.warning("ENSO data unavailable.")

    st.divider()

    # ── Energy Transition ──────────────────────────────────────────────────────
    st.markdown("#### Energy Transition Signals")
    st.caption(
        "Uranium spread proxy (URA vs 90d SMA), battery metals PC1 (LIT/GLNCY/REMX), "
        "and ETS policy stress (KRBN volatility-weighted return)."
    )

    with st.spinner("Loading energy transition data…"):
        et_df = load_energy_transition(period="2y")

    if not et_df.empty:
        et_tail = et_df.tail(252)
        et_cols = [c for c in ["uranium_spread_zscore", "battery_demand_index", "ets_stress_zscore"]
                   if c in et_df.columns]

        if et_cols:
            fig_et = make_subplots(rows=len(et_cols), cols=1, shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=[c.replace("_", " ").title() for c in et_cols])
            palette = [AMBER, BLUE, GREEN]
            for i, col in enumerate(et_cols, 1):
                fig_et.add_trace(go.Scatter(
                    x=et_tail.index, y=et_tail[col],
                    line=dict(color=palette[i-1], width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(palette[i-1][1:3],16)},{int(palette[i-1][3:5],16)},{int(palette[i-1][5:],16)},0.1)",
                    name=col,
                ), row=i, col=1)
                fig_et.add_hline(y=0, line_dash="dash", line_color="#555", opacity=0.6, row=i, col=1)
            dark_layout(fig_et, height=100 + 170 * len(et_cols), title="Energy Transition Signals")
            st.plotly_chart(fig_et, width='stretch')

            et_cols_metrics = st.columns(len(et_cols))
            for col_w, col_name in zip(et_cols_metrics, et_cols):
                val = float(et_tail[col_name].dropna().iloc[-1])
                col_w.metric(col_name.replace("_", " ").title(), f"{val:+.3f}")
        else:
            st.info("Energy transition columns not yet available.")
    else:
        st.warning("Energy transition data unavailable — yfinance fetch may have failed.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧠 Deep Learning Models")
    st.caption("Tier 3 models: BiLSTM, Meta-Prophet decomposition, and Temporal Fusion Transformer.")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.markdown("#### Prophet Trend Decomposition")
        dl_commodity = st.selectbox(
            "Commodity", list(CORE_TICKERS.keys()), index=0, key="dl_comm"
        )
        run_prophet = st.button("▶ Run Prophet (30-60s)", key="run_prophet")

    if run_prophet:
        with st.spinner(f"Running Prophet on {dl_commodity} — this takes ~30s…"):
            try:
                from models.deep.prophet_decomp import ProphetDecomposer
                pd_model = ProphetDecomposer(commodity=dl_commodity)
                pd_model.fit(prices)
                future_df, forecast_df = pd_model.forecast(periods=90)
                ss = pd_model.seasonal_strength()
                cp_summary = pd_model.changepoint_summary()
                tr = pd_model.trend_regime()

                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=forecast_df["ds"], y=np.exp(forecast_df["yhat"]),
                    line=dict(color=AMBER, width=1.8), name="Prophet forecast",
                ))
                fig_pr.add_trace(go.Scatter(
                    x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]),
                    y=list(np.exp(forecast_df["yhat_upper"])) + list(np.exp(forecast_df["yhat_lower"])[::-1]),
                    fill="toself", fillcolor="rgba(245,166,35,0.12)",
                    line=dict(width=0), name="95% CI",
                ))
                close_log = np.log(prices[dl_commodity])
                act = forecast_df[forecast_df["ds"].isin(prices.index)]
                fig_pr.add_trace(go.Scatter(
                    x=prices.index[-180:], y=prices[dl_commodity].iloc[-180:],
                    line=dict(color=BLUE, width=1.2, dash="dash"), name="Actual (last 180d)",
                ))
                dark_layout(fig_pr, height=420,
                            title=f"{dl_commodity} — Prophet 90-Day Forecast")
                st.plotly_chart(fig_pr, width='stretch')

                p1, p2, p3 = st.columns(3)
                p1.metric("Yearly seasonal strength", f"{ss.get('yearly', 0):.3f}")
                p2.metric("Weekly seasonal strength", f"{ss.get('weekly', 0):.3f}")
                p3.metric("Trend regime", tr.get("regime", "Unknown"))

                if cp_summary:
                    st.markdown("**Key Changepoints detected:**")
                    for cp in cp_summary[:5]:
                        st.caption(f"- {cp['date']} | {cp['label']} | Δtrend: {cp['delta']:+.4f}")
            except ImportError as e:
                st.warning(f"Prophet not installed: `pip install prophet`\n\n{e}")
            except Exception as e:
                st.warning(f"Prophet error: {e}")
    else:
        st.info("Click **▶ Run Prophet** to run the decomposition model on live data.")

    st.divider()

    st.markdown("#### BiLSTM Multi-Commodity Forecaster")
    run_lstm = st.button("▶ Run BiLSTM (1-5 min)", key="run_lstm")
    if run_lstm:
        with st.spinner("Training BiLSTM — this may take 1-5 min on CPU…"):
            try:
                from models.deep.lstm import LSTMForecaster
                lstm = LSTMForecaster(seq_len=63)
                lstm.fit(prices, epochs=30)
                forecasts = lstm.forecast(prices, horizon=21)
                st.success("BiLSTM trained successfully!")
                st.dataframe(forecasts.round(5), width='stretch')
            except ImportError as e:
                st.warning(f"PyTorch not installed or GPU unavailable: {e}")
            except Exception as e:
                st.warning(f"LSTM error: {e}")
    else:
        st.info("Click **▶ Run BiLSTM** to train the 2-layer bidirectional LSTM.")

    st.divider()

    st.markdown("#### Temporal Fusion Transformer (TFT)")
    st.caption("Multi-horizon quantile forecasts [1d, 5d, 20d] via pytorch-forecasting.")
    run_tft = st.button("▶ Run TFT (5-15 min)", key="run_tft")
    if run_tft:
        with st.spinner("Training TFT — this may take 5-15 min…"):
            try:
                from models.deep.tft import CommodityTFT
                tft = CommodityTFT(max_encoder_length=63, max_prediction_length=20)
                tft.fit(prices, max_epochs=20)
                fc_df = tft.forecast(prices)
                st.success("TFT trained!")
                st.dataframe(fc_df.round(5), width='stretch')
            except ImportError as e:
                st.warning(f"pytorch-forecasting not installed: {e}")
            except Exception as e:
                st.warning(f"TFT error: {e}")
    else:
        st.info("Click **▶ Run TFT** to train the Temporal Fusion Transformer.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — QUANTUM
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("⚛️ Quantum-Enhanced Models")
    st.caption(
        "Tier 4 (experimental): quantum kernel SVMs, QAOA portfolio optimisation, "
        "and QNN hybrid layers via PennyLane."
    )

    st.info(
        "**Status:** Quantum models require PennyLane ≥ 0.38 (`pip install pennylane`). "
        "Simulations run on CPU — IBM Quantum backend available via pennylane-qiskit."
    )

    # ── Kernel Benchmark ───────────────────────────────────────────────────────
    st.markdown("#### Quantum Kernel SVM Benchmark")
    st.caption(
        "Compare RY/RZ, ZZFeatureMap, and IQP circuits across 4/6/8 qubits. "
        "IC reported against classical RBF-SVM baseline. Depolarizing noise simulation included."
    )
    run_kernel = st.button("▶ Run Kernel Benchmark (5-15 min)", key="run_kernel")
    if run_kernel:
        with st.spinner("Running quantum kernel benchmark…"):
            try:
                from models.quantum.kernel_benchmark import KernelBenchmark
                from models.features import build_feature_matrix, build_target
                feat = build_feature_matrix(prices)
                tgt  = build_target(prices, "WTI Crude Oil")
                clean = pd.concat([feat, tgt], axis=1).dropna()
                X = clean.drop(columns=[tgt.name]).values[-200:, :8]
                y = clean[tgt.name].values[-200:]

                kb = KernelBenchmark()
                results = kb.run(X, y)
                st.dataframe(results.style.format("{:.4f}", subset=["ic_ideal", "runtime_s"]),
                             use_container_width=True)
            except ImportError as e:
                st.warning(f"PennyLane not installed: {e}")
            except Exception as e:
                st.warning(f"Kernel benchmark error: {e}")
    else:
        st.info("Click **▶ Run Kernel Benchmark** to compare quantum circuits.")

    st.divider()

    # ── QAOA Portfolio ─────────────────────────────────────────────────────────
    st.markdown("#### QAOA Portfolio Optimiser")
    st.caption(
        "QUBO→Ising transformation, CommutingEvolution mixer, "
        "cardinality-constrained basket selection across commodities."
    )

    n_select = st.slider("Number of assets to select (K)", 2, 6, 3, key="qaoa_k")
    run_qaoa  = st.button("▶ Run QAOA", key="run_qaoa")
    if run_qaoa:
        with st.spinner("Running QAOA optimiser…"):
            try:
                from models.quantum.qaoa_portfolio import QAOAPortfolioOptimizer
                ret = returns.tail(252).mean() * 252
                cov = returns.tail(252).cov()  * 252
                qaoa = QAOAPortfolioOptimizer(
                    n_assets=len(ret), n_layers=2, n_select=n_select
                )
                qaoa.fit(ret.values, cov.values, asset_names=list(ret.index))
                basket = qaoa.optimal_basket()
                landscape = qaoa.cost_landscape()

                st.success(f"**Optimal basket (K={n_select}):** {basket['assets']}")
                q1, q2, q3 = st.columns(3)
                q1.metric("Expected Return", f"{basket['expected_return']:.2%}")
                q2.metric("Expected Volatility", f"{basket['expected_vol']:.2%}")
                q3.metric("Sharpe Ratio", f"{basket['sharpe']:.2f}")

                fig_ql = px.scatter(landscape, x="gamma", y="cost",
                                    color="beta", title="QAOA Cost Landscape",
                                    color_continuous_scale="Viridis")
                fig_ql.update_layout(paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
                                     font_color="#FAFAFA", height=350,
                                     margin=dict(t=50, l=55, r=20, b=40))
                st.plotly_chart(fig_ql, width='stretch')
            except ImportError as e:
                st.warning(f"PennyLane not installed: {e}")
            except Exception as e:
                st.warning(f"QAOA error: {e}")
    else:
        st.info("Click **▶ Run QAOA** to run the quantum portfolio optimiser.")

    st.divider()

    # ── QNN Hybrid ─────────────────────────────────────────────────────────────
    st.markdown("#### QNN Hybrid vs Classical Comparison")
    run_qnn = st.button("▶ Run QNN Study (10-30 min)", key="run_qnn")
    if run_qnn:
        with st.spinner("Running QNN vs Classical comparison…"):
            try:
                from models.quantum.qnn_hybrid import comparison_study
                from models.features import build_feature_matrix, build_target
                feat = build_feature_matrix(prices)
                tgt  = build_target(prices, "WTI Crude Oil")
                clean = pd.concat([feat, tgt], axis=1).dropna()
                X = clean.drop(columns=[tgt.name]).values[-300:, :4]
                y = clean[tgt.name].values[-300:]
                result = comparison_study(X, y, n_qubits=4)
                st.dataframe(result["summary"], width='stretch')
            except ImportError as e:
                st.warning(f"PennyLane not installed: {e}")
            except Exception as e:
                st.warning(f"QNN error: {e}")
    else:
        st.info("Click **▶ Run QNN Study** to benchmark quantum vs classical neural networks.")
