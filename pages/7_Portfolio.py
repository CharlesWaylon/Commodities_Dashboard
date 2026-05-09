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
