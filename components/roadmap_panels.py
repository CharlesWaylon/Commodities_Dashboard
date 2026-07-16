"""Shared alpha-surface panel bodies (spec §G).

Mechanical extraction from pages/13_Signal_Lab.py and
pages/15_Research_Library.py so the Roadmap page can render the same content
without duplicating logic. The original pages keep their boilerplate + flag
gates and now call these renderers; they are NOT retired (Dashboard Evolution
Rule — retirement needs its own proof + commit).
"""
import datetime

import streamlit as st

from utils.theme import PLOTLY_LAYOUT, ASCEND, DESCEND


def render_signal_lab_panel() -> None:
    """Signal Lab body — research-grade signal diagnostics (NOT promoted)."""
    # Loud, unmissable research banner — these signals have NOT passed the gate.
    st.warning(
        "⚗️ **RESEARCH-GRADE — NOT PROMOTED.** Every signal below was REJECTED by the "
        "out-of-sample evaluation gate (walk-forward, purged/embargoed, cost-adjusted). "
        "They are shown for research transparency only and must not be traded as live "
        "signals. The composite is the best honest Phase-2 result, not a promoted model."
    )

    from evaluation import reporting as rep

    asof = rep.asof_date()
    st.caption(f"As of {asof.date() if asof is not None else 'n/a'} · gate bar = IC IR ≥ 0.30 · none cleared it")

    # ── Gate scorecard ─────────────────────────────────────────────────────────
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

    # ── Current ensemble tilts ─────────────────────────────────────────────────
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

    # ── Portfolio backtest (Layer 3) ───────────────────────────────────────────
    st.subheader("Portfolio backtest — risk-managed, net of cost")
    st.caption(
        "Walk-forward: point-in-time Ledoit-Wolf risk model → vol-targeted, "
        "concentration-capped sleeve allocator → transaction costs on turnover. "
        "Equity is net of cost; every metric is realized, not paper."
    )

    import plotly.graph_objects as go

    _BT_SIGNALS = ["value", "ensemble_v2", "reversal_st", "cot_risk_premium", "momentum_xs"]
    _c0, _c1, _c2, _c3 = st.columns(4)
    with _c0:
        bt_signal = st.selectbox("Signal", _BT_SIGNALS, index=0)
    with _c1:
        bt_panel_label = st.selectbox("Panel", ["long_core (~21y)", "aligned (~5y)"], index=0)
    with _c2:
        bt_cost = st.slider("Cost (bps/side)", 0, 50, 10, 5)
    with _c3:
        bt_reb = st.selectbox("Rebalance (days)", [5, 10, 21, 63], index=2)

    @st.cache_data(show_spinner=False)
    def _run_bt(signal_name: str, panel_source: str, cost_bps: int, rebalance_days: int):
        from models.data_loader import load_long_history_core_panel, load_price_matrix_from_db
        from portfolio.allocators import AllocatorConfig
        from portfolio.backtest import BacktestConfig, run_backtest
        from signals.base import get_signal

        panel = load_long_history_core_panel() if panel_source == "long_core" else load_price_matrix_from_db()
        if panel is None or panel.empty:
            return None
        cfg = BacktestConfig(
            rebalance_days=int(rebalance_days), cost_bps=float(cost_bps),
            allocator=AllocatorConfig(target_vol=0.10),
        )
        res = run_backtest(get_signal(signal_name), panel, cfg)
        eq = res.equity
        return {
            "equity": eq, "drawdown": eq / eq.cummax() - 1.0,
            "sharpe": res.sharpe, "ann_return": res.ann_return, "ann_vol": res.ann_vol,
            "max_drawdown": res.max_drawdown, "avg_turnover": res.avg_turnover,
            "avg_gross": res.avg_gross, "cagr": res.cagr, "n_rebalances": res.n_rebalances,
        }

    _panel_source = "long_core" if bt_panel_label.startswith("long_core") else "aligned"
    with st.spinner("Running walk-forward backtest…"):
        bt = _run_bt(bt_signal, _panel_source, int(bt_cost), int(bt_reb))

    if not bt or len(bt["equity"]) == 0:
        st.info("No backtest available for this selection (try the long_core panel).")
    else:
        m = st.columns(6)
        m[0].metric("Net Sharpe", f"{bt['sharpe']:+.2f}")
        m[1].metric("Ann. return", f"{bt['ann_return']:+.1%}")
        m[2].metric("Ann. vol", f"{bt['ann_vol']:.1%}")
        m[3].metric("Max drawdown", f"{bt['max_drawdown']:.1%}")
        m[4].metric("Turnover/reb", f"{bt['avg_turnover']:.2f}")
        m[5].metric("Avg gross", f"{bt['avg_gross']:.2f}")

        eq = bt["equity"]
        up = bt["sharpe"] is not None and bt["sharpe"] >= 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines",
                                 line=dict(color=ASCEND if up else DESCEND, width=2), name="equity"))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(height=300, margin=dict(t=30, b=10, l=10, r=10),
                          title="Net-of-cost equity (growth of $1, log scale)")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

        dd = bt["drawdown"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
                                  line=dict(color=DESCEND, width=1), name="drawdown"))
        fig2.update_layout(**PLOTLY_LAYOUT)
        fig2.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10), title="Drawdown (underwater)")
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"`{bt_signal}` · {bt_panel_label} · {bt_reb}-day rebalance · {bt_cost} bps/side · "
            f"{bt['n_rebalances']} rebalances · target vol 10%. RESEARCH-GRADE — not a live, "
            "promoted strategy. On long_core, `value` is the standout (net Sharpe ≈ 0.8, "
            "cost-robust); `momentum_xs` loses over the full cycle."
        )


def render_research_library_panel() -> None:
    """Research Library body — distilled, downloadable knowledge compilation."""
    from knowledge import digest

    st.caption(
        "A distilled, downloadable compilation of the project's knowledge documents — "
        "built so you can research how this model is stress-tested, how it has evolved, "
        "and where its actionable edges and limits actually are."
    )

    # ── Controls ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 3])
    with c1:
        mode = st.radio(
            "Reading level",
            ["Research depth", "Plain-English"],
            horizontal=True,
            help=(
                "**Research depth** keeps the technical nuance: assumptions, known flaws, "
                "IC / Sharpe / verdict numbers, and gate thresholds — only code, setup, and "
                "commit/file plumbing are stripped.\n\n"
                "**Plain-English** reduces further for a non-technical reader: it also drops "
                "dense numeric tables, algorithm step-lists, and inline-code jargon, keeping "
                "the narrative of what works and how well."
            ),
        )
    level = "research" if mode == "Research depth" else "plain"

    @st.cache_data(show_spinner=False)
    def _build(level: str):
        return (
            digest.build_summary(level),
            digest.compile_html(level),
            digest.compile_markdown(level),
        )

    with st.spinner("Distilling knowledge documents…"):
        summary_md, html_doc, markdown_doc = _build(level)

    today = datetime.date.today().isoformat()
    fname = f"accendio_research_compilation_{level}_{today}.html"

    with c2:
        st.write("")  # vertical nudge to align with the radio
        st.download_button(
            "⬇  Download full compilation (HTML)",
            data=html_doc,
            file_name=fname,
            mime="text/html",
            use_container_width=True,
            help="A self-contained HTML file — opens in any browser and prints cleanly to PDF.",
        )
        st.caption(
            f"{len(digest.available_documents())} source documents · {len(markdown_doc):,} characters · "
            "auto-distilled deterministically (no AI rewriting — reproducible & auditable)."
        )

    st.divider()

    # ── Executive digest (the synthesized 'actionable edges & evaluation' view) ─
    st.markdown(summary_md)

    st.divider()

    # ── Per-document previews ──────────────────────────────────────────────────
    st.subheader("Full compilation — by document")
    st.caption(
        "Each document below is the same distilled content included in the download. "
        "Code blocks, setup/troubleshooting, and commit-hash plumbing have been removed; "
        "structure, evaluation, edges, and honest limitations are kept."
    )

    for spec in digest.available_documents():
        with st.expander(spec.title):
            st.caption(spec.blurb)
            st.markdown(digest.distill_document(spec, level))

    st.divider()
    st.caption(
        "Sources: "
        + " · ".join(f"`{s.path}`" for s in digest.available_documents())
        + ". Daily alert logs and engineering caches are intentionally excluded. "
        "This compilation preserves the project's own candid assessments — including "
        "rejected and inconclusive results — and is for research/education, not investment advice."
    )
