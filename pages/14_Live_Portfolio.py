"""
Live Portfolio — production target weights from the bake-off winner.

The first production-grade Layer-3 surface: runs the allocator bake-off on the
chosen signal+panel, picks the winner via the "ships only where it wins" policy
(``portfolio.compete.production_allocator``), and renders today's target weights
from that winner with the supporting risk decomposition.

Distinct from the Signal Lab (which is explicitly research-grade) in tone and
defaults — this page is the deployment surface, gated by
``PRODUCTION_PORTFOLIO_ENABLED`` (default off). Thin by design: all the work lives
in ``portfolio.*`` and ``evaluation.reporting.production_targets``.
"""

import os

import pandas as pd
import streamlit as st

from utils.theme import (  # noqa: F401
    apply_theme, render_topbar, render_sidebar_nav, PLOTLY_LAYOUT, SIGNAL, ASCEND, DESCEND, AMBER,
)


def _enabled() -> bool:
    return os.getenv("PRODUCTION_PORTFOLIO_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


st.set_page_config(page_title="Accendio | Live Portfolio",
                   page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Live Portfolio")
st.caption("Bake-off winner → today's risk-managed, vol-targeted target weights.")

if not _enabled():
    st.info(
        "**Live Portfolio is off.** Gated by `PRODUCTION_PORTFOLIO_ENABLED` "
        "(default off). Set the env var to `true` to enable."
    )
    st.stop()

# ── controls ──────────────────────────────────────────────────────────────────
_SIGNALS = ["value", "ensemble_v2", "reversal_st", "cot_risk_premium", "momentum_xs"]
c0, c1, c2, c3 = st.columns(4)
with c0:
    sig_name = st.selectbox("Signal", _SIGNALS, index=0)
with c1:
    panel_label = st.selectbox("Panel", ["long_core (~21y)", "aligned (~5y)"], index=0)
with c2:
    target_vol = st.slider("Portfolio vol target", 0.05, 0.25, 0.10, 0.01)
with c3:
    risk_method = st.selectbox("Risk model", ["lw_cc", "factor", "sample"], index=0)

panel_source = "long_core" if panel_label.startswith("long_core") else "aligned"


@st.cache_data(show_spinner=False)
def _targets(signal_name, panel_source, target_vol, risk_method):
    from evaluation import reporting as rep
    return rep.production_targets(
        signal_name=signal_name, panel_source=panel_source,
        target_vol=target_vol, risk_method=risk_method,
    )


with st.spinner("Running bake-off and computing target weights…"):
    out = _targets(sig_name, panel_source, float(target_vol), risk_method)

if not out:
    st.error("Could not produce a production allocation — check that the panel and the signal are available.")
    st.stop()

# ── verdict + winner banner ───────────────────────────────────────────────────
st.markdown(f"**As of {out['asof'].date()}** · signal `{out['signal']}` · panel `{out['panel']}` · risk `{risk_method}`")
banner_msg = f"Bake-off winner: **`{out['winner']}`** — {out['verdict']}"
st.info(banner_msg)

# ── bake-off table ────────────────────────────────────────────────────────────
st.subheader("Allocator bake-off")
tbl = out["bakeoff_table"].copy()
if not tbl.empty:
    show = tbl.rename(columns={
        "net_sharpe": "Net Sharpe", "ann_return": "Ann. ret", "ann_vol": "Ann. vol",
        "max_drawdown": "Max DD", "avg_turnover": "Turnover/reb", "n_rebalances": "N rebal",
    })
    st.dataframe(
        show.style.format({
            "Net Sharpe": "{:+.3f}", "Ann. ret": "{:+.1%}", "Ann. vol": "{:.1%}",
            "Max DD": "{:.1%}", "Turnover/reb": "{:.2f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
st.caption(
    "Same walk-forward, net-of-cost backtest engine for every allocator (classical "
    "mean-variance, risk-parity, QAOA, cascade). The 'ships only where it wins' "
    "policy means QAOA is gated off unless it beats the best classical baseline."
)

# ── today's target weights ────────────────────────────────────────────────────
st.subheader(f"Today's target weights — {out['winner']}")
w = out["allocation"]
if w is None or w.empty:
    st.warning("Winner produced no current allocation (insufficient data for asof).")
else:
    longs = w[w > 0].sort_values(ascending=False)
    shorts = w[w < 0].sort_values()
    m = st.columns(4)
    m[0].metric("Names", f"{int((w != 0).sum())}")
    m[1].metric("Gross", f"{w.abs().sum():.2f}")
    m[2].metric("Net", f"{w.sum():+.2f}")
    m[3].metric("Vol target", f"{target_vol:.0%}")

    c_long, c_short = st.columns(2)
    with c_long:
        st.markdown(f"**Longs ({len(longs)})**")
        if not longs.empty:
            st.dataframe(longs.rename("weight").to_frame().style.format({"weight": "{:+.3f}"}),
                         use_container_width=True)
    with c_short:
        st.markdown(f"**Shorts ({len(shorts)})**")
        if not shorts.empty:
            st.dataframe(shorts.rename("weight").to_frame().style.format({"weight": "{:+.3f}"}),
                         use_container_width=True)
        elif "_mv" in out["winner"] or "cascade" in out["winner"] or "qaoa" in out["winner"]:
            st.caption("Selection allocators (MV-select, cascade, QAOA) are long-only by construction.")

# ── cascade visibility ────────────────────────────────────────────────────────
st.subheader("Cascade data coverage (informational)")
if out["cascade_view_asof"] is not None and out["cascade_view_n"] > 0:
    st.success(
        f"`cascade` has live forecasts for **{out['cascade_view_n']}** instruments "
        f"as of {out['cascade_view_asof'].date()}. The cascade allocator uses these "
        "to substitute the signal forecast on overlap; non-cascade names keep the signal view. "
        "Note: cascade rarely wins the historical bake-off because its data tail is short — "
        "it's primarily a live-data allocator."
    )
else:
    st.caption("No cascade forecasts available at the current asof — cascade is falling back to the signal view.")
