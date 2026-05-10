"""
QAOA Portfolio Optimizer — page 7 of the Commodities Dashboard.

Three panels:
  1. Correlation heatmap  — loaded from the DB's correlation_snapshots table
  2. Consistency flags    — cross-asset forecast divergence detector
  3. QAOA allocation      — quantum-inspired portfolio selection (button-triggered)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.theme import apply_theme, render_topbar, PLOTLY_LAYOUT

st.set_page_config(
    page_title="Accendio | Portfolio",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()
render_topbar()

# ── Theme constants ────────────────────────────────────────────────────────────
PLOT_BG = "#0C1228"
GRID    = "#1A2A5E"
AMBER   = "#F59E0B"
BLUE    = "#4FC3F7"
GREEN   = "#66BB6A"
RED     = "#EF5350"
TEAL    = "#26C6DA"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.divider()
    st.page_link("app.py",               label="Home")
    st.page_link("pages/1_Pricing.py",   label="Pricing")
    st.page_link("pages/2_Charts.py",    label="Charts")
    st.page_link("pages/3_News.py",      label="News")
    st.page_link("pages/4_Models.py",    label="Models")
    st.page_link("pages/5_Database.py",  label="Database")
    st.page_link("pages/6_Causal.py",    label="Causal Chain")
    st.page_link("pages/7_Portfolio.py", label="Portfolio")
    st.divider()

    st.subheader("QAOA Parameters")
    n_assets = st.slider(
        "Universe size (qubits)", 8, 15, 12, 1,
        help="Top-Sharpe assets passed as qubits to the circuit",
    )
    k_assets = st.slider(
        "Assets to hold (k)", 3, 8, 5, 1,
        help="Budget constraint: exactly k assets in the final portfolio",
    )
    p_layers = st.slider(
        "Circuit depth (p)", 1, 4, 2, 1,
        help="QAOA layers — higher p = better solution, slower runtime",
    )
    lam = st.slider(
        "λ risk–return", 0.5, 5.0, 2.0, 0.5,
        help="Higher → optimizer chases return over variance reduction",
    )
    penalty = st.slider(
        "Budget penalty P", 1.0, 10.0, 5.0, 0.5,
        help="Lagrange multiplier enforcing exactly k assets",
    )
    st.divider()
    st.caption(
        "**Runtime:** QAOA at p=2, n=12 takes ~2–4 min on a laptop simulator. "
        "Results are cached for this session."
    )
    st.divider()
    st.subheader("🌊 Cascade Mode")
    cascade_k = st.slider(
        "Cascade assets to hold", 3, 8, k_assets, 1,
        key="casc_k_slider",
        help="Top-k assets selected by cascade signal for the optimized portfolio",
    )
    backtest_rebal = st.slider(
        "Backtest rebalance (days)", 10, 63, 21, 1,
        key="backtest_rebal_slider",
        help="How often to rebalance in the walk-forward backtest",
    )


# ── Page header ────────────────────────────────────────────────────────────────
st.title("⚛️ QAOA Portfolio Optimizer")
st.caption(
    "Quantum-inspired mean-variance selection · "
    "cross-asset correlation heatmap · "
    "forecast consistency flags"
)
st.divider()


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_corr_matrix():
    from models.cross_asset import load_correlation_matrix
    return load_correlation_matrix()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices():
    from models.data_loader import load_price_matrix_from_db
    return load_price_matrix_from_db()


@st.cache_data(ttl=900, show_spinner=False)
def _load_macro() -> pd.DataFrame:
    """Load macro signals (dxy_zscore63, vix, tlt_mom21) from the DB trigger log."""
    import sqlite3
    import os
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "commodities.db")
    db_path = os.path.normpath(db_path)
    try:
        with sqlite3.connect(db_path) as con:
            # Try trigger_log table first (has macro columns populated by detector)
            df = pd.read_sql_query(
                "SELECT date, dxy_zscore, vix, tlt_mom21 FROM trigger_log ORDER BY date",
                con, parse_dates=["date"]
            )
            df = df.set_index("date").rename(columns={"dxy_zscore": "dxy_zscore63"})
            return df
    except Exception:
        return pd.DataFrame()


def _recent_returns(prices: pd.DataFrame, window: int = 5) -> dict:
    """Mean log-return over the last `window` days — proxy forecasts for consistency check."""
    if len(prices) < window + 1:
        return {}
    sub = prices.iloc[-(window + 1):]
    lr  = np.log(sub / sub.shift(1)).dropna()
    return lr.mean().to_dict()


# ── Section 1: Correlation Heatmap ────────────────────────────────────────────
st.subheader("📊 Cross-Asset Correlation Heatmap")
st.caption("21-day rolling Pearson correlations · most recent snapshot stored in the DB")

with st.spinner("Loading correlation matrix…"):
    corr_mat = _load_corr_matrix()

if corr_mat.empty:
    st.info(
        "No correlation snapshots found yet. "
        "Run `store_correlation_snapshot()` at least once after prices are ingested.",
        icon="📭",
    )
else:
    def _short(name: str) -> str:
        words = name.replace("(", " ").split()
        return words[0]

    labels = [_short(c) for c in corr_mat.columns]
    z = corr_mat.values
    n = len(corr_mat)

    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0,  RED],
            [0.5,  PLOT_BG],
            [1.0,  GREEN],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 9, "color": "#E0E0E0"},
        hoverongaps=False,
        colorbar=dict(
            title="ρ",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0.5", "0", "+0.5", "+1"],
            tickfont=dict(color="#E0E0E0"),
            titlefont=dict(color="#E0E0E0"),
        ),
    ))
    cell_px = max(18, min(40, 700 // n))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        height=max(420, n * cell_px + 80),
        margin=dict(t=20, l=10, r=10, b=10),
        xaxis=dict(tickfont=dict(size=10, color="#C0CDE0"), side="bottom"),
        yaxis=dict(tickfont=dict(size=10, color="#C0CDE0"), autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{n} × {n} matrix · {n} commodities tracked")

st.divider()


# ── Section 2: Consistency Flags ──────────────────────────────────────────────
st.subheader("🚩 Forecast Consistency Flags")
st.caption(
    "5-day rolling log-returns used as proxy forecasts. "
    "Flags correlated pairs (|ρ| ≥ 0.70) with divergent or mismatched signals."
)

with st.spinner("Loading prices and checking cross-asset consistency…"):
    try:
        prices_df   = _load_prices()
        proxy_fcast = _recent_returns(prices_df)
    except Exception as exc:
        prices_df   = pd.DataFrame()
        proxy_fcast = {}
        st.warning(f"Could not load price data: {exc}", icon="⚠️")

if proxy_fcast and not corr_mat.empty:
    from models.cross_asset import check_forecast_consistency
    flags = check_forecast_consistency(proxy_fcast, corr_matrix=corr_mat)

    if not flags:
        st.success(
            "No consistency issues detected across highly-correlated pairs.",
            icon="✅",
        )
    else:
        n_crit = sum(1 for f in flags if f.severity == "critical")
        n_warn = sum(1 for f in flags if f.severity == "warning")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total flags", len(flags))
        c2.metric("Critical",    n_crit)
        c3.metric("Warnings",    n_warn)

        rows = []
        for f in flags:
            rows.append({
                "Severity":    f.severity.upper(),
                "Type":        f.flag_type.replace("_", " ").title(),
                "Asset A":     f.commodity_a,
                "Forecast A":  f"{f.forecast_a:+.2%}",
                "Asset B":     f.commodity_b,
                "Forecast B":  f"{f.forecast_b:+.2%}",
                "Correlation": f"{f.correlation:.2f}",
                "Message":     f.message,
            })
        flags_df = pd.DataFrame(rows)

        def _sev_color(val):
            if val == "CRITICAL":
                return f"color: {RED}; font-weight: bold"
            if val == "WARNING":
                return f"color: {AMBER}"
            return ""

        st.dataframe(
            flags_df.style.applymap(_sev_color, subset=["Severity"]),
            use_container_width=True,
            hide_index=True,
            height=min(38 * len(flags_df) + 44, 420),
        )
else:
    st.info(
        "Consistency check requires both price data and a correlation snapshot. "
        "Run data ingestion and `store_correlation_snapshot()` first.",
        icon="ℹ️",
    )

st.divider()


# ── Section 3: QAOA Allocation ────────────────────────────────────────────────
st.subheader("⚛️ QAOA Allocation Vector")
st.caption(
    "Quantum Approximate Optimisation Algorithm on the PennyLane simulator — "
    "QUBO → Ising → variational COBYLA optimisation."
)

run_col, info_col = st.columns([2, 3])
with run_col:
    run_btn = st.button(
        "⚛️ Run QAOA Optimizer",
        type="primary",
        help=(
            f"Selects {k_assets} assets from the top-{n_assets} Sharpe universe. "
            f"p={p_layers} QAOA layers. Takes ~2–4 min."
        ),
    )
with info_col:
    st.info(
        f"Universe: top-**{n_assets}** Sharpe assets · "
        f"Hold **{k_assets}** · "
        f"p=**{p_layers}** layers · λ=**{lam}** · P=**{penalty}**",
        icon="⚛️",
    )

if run_btn:
    st.session_state["qaoa_params"] = dict(
        n_assets=n_assets, k=k_assets, p=p_layers,
        lam=lam, penalty=penalty,
    )
    st.session_state.pop("qaoa_result", None)

if st.session_state.get("qaoa_params") and "qaoa_result" not in st.session_state:
    params = st.session_state["qaoa_params"]
    with st.spinner(
        f"Running QAOA (n={params['n_assets']}, k={params['k']}, "
        f"p={params['p']})… grab a coffee ☕ this takes a few minutes"
    ):
        try:
            from models.quantum.qaoa_portfolio import run_qaoa_portfolio
            result = run_qaoa_portfolio(
                n_assets  = params["n_assets"],
                k         = params["k"],
                p         = params["p"],
                lam       = params["lam"],
                penalty   = params["penalty"],
                prices    = prices_df if not prices_df.empty else None,
            )
            st.session_state["qaoa_result"] = result
        except Exception as exc:
            st.error(f"QAOA optimizer failed: {exc}", icon="❌")
            st.session_state.pop("qaoa_params", None)

if "qaoa_result" in st.session_state:
    res = st.session_state["qaoa_result"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected assets",  len(res.selected_assets))
    m2.metric("Expected return",  f"{res.expected_return:.1%}")
    m3.metric("Portfolio vol",    f"{res.portfolio_vol:.1%}")
    m4.metric("Sharpe (rf = 0)",  f"{res.sharpe:.2f}")

    st.divider()

    left_col, right_col = st.columns([2, 1])

    # ── Allocation bar chart ──────────────────────────────────────────────────
    with left_col:
        st.markdown("#### Allocation vector")

        alloc_df = (
            pd.DataFrame.from_dict(res.weights, orient="index", columns=["weight"])
            .reset_index()
            .rename(columns={"index": "asset"})
            .sort_values("weight", ascending=False)
        )
        alloc_df["label"] = alloc_df["asset"].apply(
            lambda n: n.split("(")[0].strip()
        )

        fig_bar = go.Figure(go.Bar(
            x=alloc_df["label"],
            y=alloc_df["weight"],
            marker_color=TEAL,
            marker_line_color=GRID,
            marker_line_width=1,
            text=[f"{w:.1%}" for w in alloc_df["weight"]],
            textposition="outside",
            textfont=dict(color="#E0E0E0", size=11),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            margin=dict(t=20, l=10, r=10, b=60),
            xaxis=dict(tickangle=-30, tickfont=dict(size=11, color="#C0CDE0")),
            yaxis=dict(
                tickformat=".0%",
                tickfont=dict(color="#C0CDE0"),
                gridcolor=GRID,
                range=[0, max(alloc_df["weight"]) * 1.3],
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── Weights table ─────────────────────────────────────────────────────────
    with right_col:
        st.markdown("#### Weights")
        wt = alloc_df[["asset", "weight"]].copy()
        wt["weight"] = wt["weight"].apply(lambda w: f"{w:.2%}")
        wt.columns = ["Asset", "Weight"]
        st.dataframe(wt, use_container_width=True, hide_index=True)

    st.divider()

    # ── QAOA convergence ──────────────────────────────────────────────────────
    if res.cost_history:
        with st.expander("📉 QAOA convergence (cost history)", expanded=False):
            hist = res.cost_history
            fig_conv = go.Figure(go.Scatter(
                x=list(range(len(hist))),
                y=hist,
                mode="lines",
                line=dict(color=AMBER, width=2),
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.08)",
            ))
            fig_conv.update_layout(
                **PLOTLY_LAYOUT,
                height=260,
                margin=dict(t=10, l=10, r=10, b=40),
                xaxis=dict(title="COBYLA iteration", tickfont=dict(color="#C0CDE0")),
                yaxis=dict(
                    title="⟨H_C⟩",
                    tickfont=dict(color="#C0CDE0"),
                    gridcolor=GRID,
                ),
                showlegend=False,
            )
            st.plotly_chart(
                fig_conv, use_container_width=True, config={"displayModeBar": False}
            )
            st.caption(
                f"{len(hist)} COBYLA iterations · "
                f"final cost {hist[-1]:.4f} · "
                f"optimal params: {', '.join(f'{v:.3f}' for v in res.optimal_params)}"
            )

else:
    st.info(
        "Click **⚛️ Run QAOA Optimizer** above to compute an allocation. "
        "The simulator runs ~2–4 minutes at p=2 with 12 assets.",
        icon="⚛️",
    )

st.divider()


# ── Section 4: Cascade-Informed QAOA ─────────────────────────────────────────
st.subheader("🌊 Cascade-Informed QAOA Allocation")
st.caption(
    "Replaces the historical expected-return vector (μ) with macro-adjusted cascade "
    "forecasts, blended by per-commodity confidence. Covariance structure is unchanged. "
    "QAOA then finds the cardinality-constrained allocation that maximises "
    "cascade-informed Sharpe."
)

casc_run_col, casc_info_col = st.columns([2, 3])
with casc_run_col:
    casc_run_btn = st.button(
        "🌊 Run Cascade QAOA",
        type="primary",
        key="casc_run_btn",
        help=(
            f"Computes cascade forecasts for all commodities using today's macro state, "
            f"then runs QAOA (p={p_layers}) with cascade μ. ~2–4 min."
        ),
    )

macro_df = _load_macro()
latest_macro: dict = {}
if not macro_df.empty:
    latest_macro = macro_df.iloc[-1].to_dict()
    dxy_z_disp = latest_macro.get("dxy_zscore63", 0.0) or 0.0
    vix_disp   = latest_macro.get("vix", 18.0) or 18.0
    tlt_disp   = latest_macro.get("tlt_mom21", 0.0) or 0.0
    with casc_info_col:
        st.info(
            f"Macro state: DXY z={dxy_z_disp:+.2f} · VIX {vix_disp:.1f} · "
            f"TLT mom {tlt_disp:+.1f}% · "
            f"Hold **{cascade_k}** assets · p=**{p_layers}** layers",
            icon="🌊",
        )
else:
    with casc_info_col:
        st.warning(
            "No macro data found — cascade will use momentum-only signals.",
            icon="⚠️",
        )

if casc_run_btn:
    st.session_state["casc_params"] = dict(
        k=cascade_k, p=p_layers, n_assets=n_assets,
        lam=lam, penalty=penalty,
    )
    st.session_state.pop("casc_result", None)

if st.session_state.get("casc_params") and "casc_result" not in st.session_state:
    cp = st.session_state["casc_params"]
    with st.spinner(
        f"Computing cascade forecasts + running QAOA "
        f"(n={cp['n_assets']}, k={cp['k']}, p={cp['p']})… ☕"
    ):
        try:
            from models.portfolio_optimizer import (
                run_cascade_portfolio,
                compute_cascade_forecast,
                COMM_EFFECTS,
            )
            _pf = prices_df if not prices_df.empty else None
            if _pf is None or _pf.empty:
                st.error("No price data available for cascade computation.", icon="❌")
                st.session_state.pop("casc_params", None)
            else:
                # Compute cascade forecast for each commodity in the universe
                _cascade_fcast: dict = {}
                _confidences:   dict = {}
                _channels:      dict = {}

                for _comm in COMM_EFFECTS:
                    try:
                        _, _, _fp, _, _conf, _ch = compute_cascade_forecast(
                            _comm, latest_macro, _pf
                        )
                        _cascade_fcast[_comm] = _fp
                        _confidences[_comm]   = _conf
                        _channels[_comm]      = _ch
                    except Exception:
                        pass

                _casc_res = run_cascade_portfolio(
                    cascade_forecasts = _cascade_fcast,
                    confidences       = _confidences,
                    prices            = _pf,
                    macro_row         = latest_macro,
                    channel_contribs  = _channels,
                    n_assets          = cp["n_assets"],
                    k                 = cp["k"],
                    p                 = cp["p"],
                    lam               = cp["lam"],
                    penalty           = cp["penalty"],
                    run_baseline      = False,
                    n_shots           = 512,
                )
                st.session_state["casc_result"] = _casc_res
        except Exception as _exc:
            st.error(f"Cascade QAOA failed: {_exc}", icon="❌")
            st.session_state.pop("casc_params", None)

if "casc_result" in st.session_state:
    _cr = st.session_state["casc_result"]

    # ── Narrative ─────────────────────────────────────────────────────────────
    if _cr.causal_narrative:
        st.info(_cr.causal_narrative, icon="🌊")

    # ── Metrics ───────────────────────────────────────────────────────────────
    _cm1, _cm2, _cm3, _cm4 = st.columns(4)
    _cm1.metric("Selected assets",  len(_cr.selected_assets))
    _cm2.metric("Expected return",  f"{_cr.expected_return:.1%}",
                help="Cascade-informed annualised expected return")
    _cm3.metric("Portfolio vol",    f"{_cr.portfolio_vol:.1%}")
    _cm4.metric("Sharpe (rf = 0)", f"{_cr.sharpe:.2f}")

    st.divider()

    _cl, _cr_col = st.columns([2, 1])

    with _cl:
        st.markdown("#### Cascade Allocation")
        _casc_alloc = (
            pd.DataFrame.from_dict(_cr.weights, orient="index", columns=["weight"])
            .reset_index()
            .rename(columns={"index": "asset"})
            .sort_values("weight", ascending=False)
        )
        _casc_alloc["label"] = _casc_alloc["asset"].apply(lambda n: n.split("(")[0].strip())
        _casc_alloc["cascade_pct"] = _casc_alloc["asset"].apply(
            lambda a: _cr.cascade_forecasts.get(a, 0.0)
        )
        _casc_alloc["conf"] = _casc_alloc["asset"].apply(
            lambda a: _cr.confidences.get(a, 0.5)
        )

        _bar_colors = [GREEN if w >= 0 else RED for w in _casc_alloc["cascade_pct"]]
        _fig_casc = go.Figure(go.Bar(
            x=_casc_alloc["label"],
            y=_casc_alloc["weight"],
            marker_color=_bar_colors,
            marker_line_color=GRID,
            marker_line_width=1,
            text=[
                f"{w:.0%}<br>{p:+.1f}%"
                for w, p in zip(_casc_alloc["weight"], _casc_alloc["cascade_pct"])
            ],
            textposition="outside",
            textfont=dict(color="#E0E0E0", size=10),
        ))
        _fig_casc.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            margin=dict(t=20, l=10, r=10, b=70),
            xaxis=dict(tickangle=-30, tickfont=dict(size=11, color="#C0CDE0")),
            yaxis=dict(
                tickformat=".0%",
                tickfont=dict(color="#C0CDE0"),
                gridcolor=GRID,
                range=[0, max(_casc_alloc["weight"]) * 1.4],
            ),
            showlegend=False,
        )
        st.plotly_chart(_fig_casc, use_container_width=True, config={"displayModeBar": False})

    with _cr_col:
        st.markdown("#### Cascade Forecasts")
        _fc_rows = []
        for _a in _cr.selected_assets:
            _fc_rows.append({
                "Asset":      _a.split("(")[0].strip(),
                "Weight":     f"{_cr.weights.get(_a, 0):.0%}",
                "Cascade %":  f"{_cr.cascade_forecasts.get(_a, 0):+.2f}%",
                "Confidence": f"{_cr.confidences.get(_a, 0):.0%}",
            })
        st.dataframe(pd.DataFrame(_fc_rows), use_container_width=True, hide_index=True)

    # ── Channel breakdown ─────────────────────────────────────────────────────
    with st.expander("📡 Channel contributions per selected asset"):
        _ch_rows = []
        for _a in _cr.selected_assets:
            _ch = _cr.channel_contributions.get(_a, {})
            _ch_rows.append({
                "Asset":       _a.split("(")[0].strip(),
                "Real Yields": f"{_ch.get('Real Yields', 0):+.2f}pp",
                "USD":         f"{_ch.get('USD', 0):+.2f}pp",
                "Risk-Off":    f"{_ch.get('Risk-Off', 0):+.2f}pp",
                "Total Macro": f"{sum(_ch.values()):+.2f}pp",
            })
        st.dataframe(pd.DataFrame(_ch_rows), use_container_width=True, hide_index=True)

    # ── QAOA convergence ──────────────────────────────────────────────────────
    if _cr.qaoa_result.cost_history:
        with st.expander("📉 QAOA convergence (cascade run)", expanded=False):
            _hist = _cr.qaoa_result.cost_history
            _fig_cc = go.Figure(go.Scatter(
                x=list(range(len(_hist))), y=_hist,
                mode="lines", line=dict(color=TEAL, width=2),
                fill="tozeroy", fillcolor="rgba(38,198,218,0.08)",
            ))
            _fig_cc.update_layout(
                **PLOTLY_LAYOUT, height=220,
                margin=dict(t=10, l=10, r=10, b=40),
                xaxis=dict(title="COBYLA iteration", tickfont=dict(color="#C0CDE0")),
                yaxis=dict(title="⟨H_C⟩", tickfont=dict(color="#C0CDE0"), gridcolor=GRID),
                showlegend=False,
            )
            st.plotly_chart(_fig_cc, use_container_width=True, config={"displayModeBar": False})

else:
    st.info(
        "Click **🌊 Run Cascade QAOA** to compute a macro-adjusted portfolio allocation.",
        icon="🌊",
    )

st.divider()


# ── Section 5: Backtest — Cascade vs Baseline ─────────────────────────────────
st.subheader("📊 Backtest: Cascade vs Momentum vs Equal-Weight")
st.caption(
    "Walk-forward simulation: every rebalance period, each strategy selects the "
    f"top-{cascade_k} commodities by its signal (cascade / momentum / none), holds "
    "equal-weight, and earns the actual forward return. No QAOA in the backtest "
    "— this tests signal quality independently of the solver."
)

_bt_col1, _bt_col2 = st.columns([2, 3])
with _bt_col1:
    _bt_btn = st.button(
        "📊 Run Backtest",
        key="backtest_btn",
        help=f"Walk-forward over the full price history, rebalancing every {backtest_rebal} days.",
    )
with _bt_col2:
    st.info(
        f"Rebalance every **{backtest_rebal}** days · "
        f"Hold top-**{cascade_k}** · "
        f"Lookback **252** days · "
        f"Benchmark: equal-weight all",
        icon="📊",
    )

if _bt_btn:
    st.session_state.pop("bt_result", None)
    st.session_state["bt_params"] = dict(k=cascade_k, rebal=backtest_rebal)

if st.session_state.get("bt_params") and "bt_result" not in st.session_state:
    _btp = st.session_state["bt_params"]
    with st.spinner("Running walk-forward backtest (greedy selection, no QAOA)…"):
        try:
            from models.portfolio_optimizer import backtest_cascade_vs_baseline
            _bt_macro = macro_df if not macro_df.empty else pd.DataFrame()
            _bt_prices = prices_df if not prices_df.empty else pd.DataFrame()
            if _bt_prices.empty:
                st.error("No price data for backtest.", icon="❌")
                st.session_state.pop("bt_params", None)
            else:
                _bt_res = backtest_cascade_vs_baseline(
                    prices         = _bt_prices,
                    macro_df       = _bt_macro,
                    k              = _btp["k"],
                    rebalance_days = _btp["rebal"],
                )
                st.session_state["bt_result"] = _bt_res
        except Exception as _exc:
            st.error(f"Backtest failed: {_exc}", icon="❌")
            st.session_state.pop("bt_params", None)

if "bt_result" in st.session_state:
    _bt = st.session_state["bt_result"]

    if _bt.n_periods == 0:
        st.warning(
            "Not enough data to run the backtest — need at least 126 days of history "
            "across the commodity universe.",
            icon="⚠️",
        )
    else:
        # ── Summary metrics table ─────────────────────────────────────────────
        st.markdown(f"**{_bt.n_periods} rebalance periods** · "
                    f"rebalance every {_bt.rebalance_days}d · "
                    f"hold {_bt.k} assets")
        _summ_df = _bt.summary_df()
        st.dataframe(_summ_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Cumulative return chart ───────────────────────────────────────────
        _colors_bt = {
            "Equal Weight":   AMBER,
            "Momentum Top-k": BLUE,
            "Cascade Top-k":  GREEN,
        }
        _fig_bt = go.Figure()
        for _s, _cum in _bt.cumulative_returns.items():
            _fig_bt.add_trace(go.Scatter(
                x=_cum.index, y=_cum.values,
                mode="lines",
                name=_s,
                line=dict(color=_colors_bt.get(_s, "#A0AEC0"), width=2),
                fill="none",
            ))
        _fig_bt.add_hline(y=1.0, line_dash="dot", line_color="#718096",
                          annotation_text="Starting value", annotation_position="right")
        _fig_bt.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            margin=dict(t=20, l=10, r=20, b=40),
            xaxis=dict(title="Date", tickfont=dict(color="#C0CDE0")),
            yaxis=dict(
                title="Cumulative return (log-scale)",
                type="log",
                tickfont=dict(color="#C0CDE0"),
                gridcolor=GRID,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(color="#E0E0E0"),
            ),
        )
        st.plotly_chart(_fig_bt, use_container_width=True, config={"displayModeBar": False})

        # ── Lift vs equal-weight ──────────────────────────────────────────────
        ew_ann = _bt.annual_returns.get("Equal Weight", 0.0)
        casc_ann = _bt.annual_returns.get("Cascade Top-k", 0.0)
        mom_ann  = _bt.annual_returns.get("Momentum Top-k", 0.0)
        lift_vs_ew   = casc_ann - ew_ann
        lift_vs_mom  = casc_ann - mom_ann

        _lc1, _lc2, _lc3 = st.columns(3)
        _lc1.metric(
            "Cascade lift vs Equal-Weight",
            f"{lift_vs_ew:+.1%}",
            help="Annualised return difference",
        )
        _lc2.metric(
            "Cascade lift vs Momentum",
            f"{lift_vs_mom:+.1%}",
            help="Annualised return difference",
        )
        _lc3.metric(
            "Cascade Sharpe",
            f"{_bt.sharpe_ratios.get('Cascade Top-k', 0):.2f}",
            delta=f"vs {_bt.sharpe_ratios.get('Momentum Top-k', 0):.2f} momentum",
        )

        # ── Period-return distribution ────────────────────────────────────────
        with st.expander("📦 Period return distribution"):
            _fig_dist = go.Figure()
            for _s, _ser in _bt.strategy_returns.items():
                _fig_dist.add_trace(go.Histogram(
                    x=_ser.values * 100,
                    name=_s,
                    opacity=0.65,
                    nbinsx=30,
                    marker_color=_colors_bt.get(_s, "#A0AEC0"),
                ))
            _fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                barmode="overlay",
                height=280,
                margin=dict(t=20, l=10, r=10, b=40),
                xaxis=dict(title="Period log-return (%)", tickfont=dict(color="#C0CDE0")),
                yaxis=dict(title="Count", tickfont=dict(color="#C0CDE0"), gridcolor=GRID),
                legend=dict(font=dict(color="#E0E0E0")),
            )
            st.plotly_chart(_fig_dist, use_container_width=True,
                            config={"displayModeBar": False})

else:
    st.info(
        "Click **📊 Run Backtest** to compare cascade-informed allocation against "
        "momentum and equal-weight strategies over the full price history.",
        icon="📊",
    )
