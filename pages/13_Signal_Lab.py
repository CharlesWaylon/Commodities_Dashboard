"""
Signal Lab — RESEARCH-GRADE signal diagnostics (NOT promoted).

A flagged, additive surface (Dashboard Evolution Rule) that exposes the Phase-2
signal-research output: the equal-weight ensemble of right-signed-but-sub-threshold
edges and the gate scorecards behind it. NOTHING here has passed the out-of-sample
gate — the page is explicitly labelled research-grade so the institutional audience
is never misled into treating it as a live, promoted signal.

Gated by SIGNAL_RESEARCH_ENABLED (default off). Thin by design: all computation
lives in the signal layer (signals/) and evaluation/reporting.py.
"""

import os

import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav, PLOTLY_LAYOUT  # noqa: F401


def _enabled() -> bool:
    return os.getenv("SIGNAL_RESEARCH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


st.set_page_config(page_title="Accendio | Signal Lab", page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Signal Lab")

if not _enabled():
    st.info(
        "**Signal Lab is off.** This is a research-grade surface gated behind the "
        "`SIGNAL_RESEARCH_ENABLED` feature flag. Set `SIGNAL_RESEARCH_ENABLED=true` "
        "to enable it."
    )
    st.stop()

# Loud, unmissable research banner — these signals have NOT passed the gate.
st.warning(
    "⚗️ **RESEARCH-GRADE — NOT PROMOTED.** Every signal below was REJECTED by the "
    "out-of-sample evaluation gate (walk-forward, purged/embargoed, cost-adjusted). "
    "They are shown for research transparency only and must not be traded as live "
    "signals. The composite is the best honest Phase-2 result, not a promoted model."
)

from evaluation import reporting as rep  # noqa: E402  (import after flag/stop guard)

asof = rep.asof_date()
st.caption(f"As of {asof.date() if asof is not None else 'n/a'} · gate bar = IC IR ≥ 0.30 · none cleared it")

# ── Gate scorecard ────────────────────────────────────────────────────────────
st.subheader("Gate scorecard — ensemble & components")
sc = rep.latest_scorecard()
if sc.empty:
    st.info("No scorecard rows found. Run the gate: `python -m evaluation.harness --signal ensemble_v1`.")
else:
    show = sc.rename(columns={
        "signal_name": "signal", "horizon": "H", "ic_mean": "IC", "ic_ir": "IC_IR",
        "ic_tstat": "t-stat", "hit_rate": "hit", "ls_sharpe_net": "LS_Sharpe",
        "avg_turnover": "turnover",
    })[["signal", "H", "IC", "IC_IR", "t-stat", "hit", "LS_Sharpe", "turnover", "verdict"]]
    st.dataframe(
        show.style.format({
            "IC": "{:+.4f}", "IC_IR": "{:.3f}", "t-stat": "{:+.2f}", "hit": "{:.3f}",
            "LS_Sharpe": "{:+.3f}", "turnover": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Composite `ensemble_v1` = equal-weight(momentum_xs, cot_risk_premium, "
        "reversal_st). Best: H10 IC IR 0.25, t≈3.2, cost-adjusted LS Sharpe 0.71 — "
        "still below the 0.30 promotion bar."
    )

# ── Current ensemble tilts ────────────────────────────────────────────────────
st.subheader("Current ensemble cross-section (10-day horizon)")
tilts = rep.ensemble_tilts(horizon=10, top_n=8)
if tilts.empty:
    st.info("No ensemble snapshot available.")
else:
    c1, c2 = st.columns(2)
    longs = tilts[tilts["side"] == "LONG"]
    shorts = tilts[tilts["side"] == "SHORT"].iloc[::-1]
    with c1:
        st.markdown("**Top longs**")
        st.dataframe(longs[["instrument", "score"]].style.format({"score": "{:+.2f}"}),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Top shorts**")
        st.dataframe(shorts[["instrument", "score"]].style.format({"score": "{:+.2f}"}),
                     use_container_width=True, hide_index=True)
    st.caption(
        "Dollar-neutral cross-sectional z-scores from the research composite. "
        "Illustrative tilts, not trade recommendations."
    )
