"""
Accendio — Command Centre (home page)

Five-panel live market intelligence dashboard:
  Panel 1  Global Commodity Heatmap
  Panel 2  Top Active Signals
  Panel 3  Sector Performance Timeline (30 trading days)
  Panel 4  Cross-Market Correlations
  Panel 5  Today's Intelligence Brief
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

from services.price_data import fetch_current_prices
from utils.theme import (
    apply_theme, render_topbar, panel_header, PLOTLY_LAYOUT,
    VOID, ABYSS, DEPTH, SIGNAL, ICE, ICE_MID,
    ASCEND, DESCEND, AMBER, BORDER,
)

st.set_page_config(
    page_title="Accendio",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Cached data helpers ───────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _load_sector_timeline():
    """30-day sector daily return heatmap from aligned_prices table."""
    try:
        from database.db import get_engine
        engine = get_engine()
        cutoff = (datetime.now() - timedelta(days=90)).date()

        df = pd.read_sql(
            f"""
            SELECT c.name, c.sector, a.date, a.adjusted_close
            FROM aligned_prices a
            JOIN commodities c ON c.id = a.commodity_id
            WHERE a.date >= '{cutoff}' AND a.is_filled = 0
            ORDER BY c.name, a.date
            """,
            engine,
            parse_dates=["date"],
        )
        if df.empty:
            return None

        df = df.sort_values(["name", "date"])
        df["ret"] = df.groupby("name")["adjusted_close"].pct_change() * 100

        pivot = (
            df.groupby(["sector", "date"])["ret"]
            .mean()
            .reset_index()
            .pivot(index="sector", columns="date", values="ret")
        )
        return pivot.iloc[:, -30:].round(2)
    except Exception:
        return None


@st.cache_data(ttl=600)
def _load_correlations():
    """60-day pairwise return correlations for sector representative instruments."""
    reps = [
        "WTI Crude Oil", "Natural Gas (Henry Hub)",
        "Gold (COMEX)", "Copper (COMEX)",
        "Corn (CBOT)", "Wheat (CBOT SRW)", "Soybeans (CBOT)",
        "Live Cattle",
    ]
    name_map = {
        "WTI Crude Oil": "WTI", "Natural Gas (Henry Hub)": "Nat Gas",
        "Gold (COMEX)": "Gold", "Copper (COMEX)": "Copper",
        "Corn (CBOT)": "Corn", "Wheat (CBOT SRW)": "Wheat",
        "Soybeans (CBOT)": "Soybeans", "Live Cattle": "Cattle",
    }
    try:
        from database.db import get_engine
        engine = get_engine()
        cutoff = (datetime.now() - timedelta(days=100)).date()
        rep_sql = ", ".join(f"'{r}'" for r in reps)

        df = pd.read_sql(
            f"""
            SELECT c.name, a.date, a.adjusted_close
            FROM aligned_prices a
            JOIN commodities c ON c.id = a.commodity_id
            WHERE a.date >= '{cutoff}' AND c.name IN ({rep_sql})
            ORDER BY c.name, a.date
            """,
            engine,
            parse_dates=["date"],
        )
        if df.empty or df["name"].nunique() < 3:
            return None

        pivot   = df.pivot(index="date", columns="name", values="adjusted_close")
        returns = pivot.pct_change().dropna()
        corr    = returns.corr().round(2)
        return corr.rename(index=name_map, columns=name_map)
    except Exception:
        return None


def _build_brief(df: pd.DataFrame) -> str:
    """Build the HTML for Today's Intelligence Brief from live price data."""
    best   = df.loc[df["Pct_Change"].idxmax()]
    worst  = df.loc[df["Pct_Change"].idxmin()]
    sp     = df.groupby("Sector")["Pct_Change"].mean().sort_values(ascending=False)
    top3   = df.sort_values("Pct_Change", key=abs, ascending=False).head(3)
    ts_str = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")

    def _c(v): return ASCEND if v >= 0 else DESCEND

    brief = f"""
<div style="background:{DEPTH};border:0.5px solid {BORDER};border-radius:8px;padding:16px 18px;height:100%">
  <div style="font-size:10px;color:rgba(238,242,255,0.28);letter-spacing:.1em;margin-bottom:10px">{ts_str}</div>
  <p style="font-size:13px;line-height:1.7;color:rgba(238,242,255,0.72);margin-bottom:14px">
    <b style="color:{ICE}">{sp.index[0]}</b> leads today at an avg
    <span style="color:{_c(sp.iloc[0])}">{sp.iloc[0]:+.2f}%</span>,
    while <b style="color:{ICE}">{sp.index[-1]}</b> trails at
    <span style="color:{_c(sp.iloc[-1])}">{sp.iloc[-1]:+.2f}%</span>.
    Top performer: <b style="color:{ICE}">{best["Name"].split("(")[0].strip()}</b>
    <span style="color:{ASCEND}">{best["Pct_Change"]:+.2f}%</span>.
    Biggest decline: <b style="color:{ICE}">{worst["Name"].split("(")[0].strip()}</b>
    <span style="color:{DESCEND}">{worst["Pct_Change"]:+.2f}%</span>.
  </p>
  <div style="font-size:10px;color:rgba(238,242,255,0.3);letter-spacing:.12em;margin-bottom:8px">TOP SIGNALS</div>
  <div style="display:flex;flex-direction:column;gap:6px">
"""
    for _, row in top3.iterrows():
        p    = row["Pct_Change"]
        c    = ASCEND if p > 0 else DESCEND
        flag = "↑" if p > 0 else "↓"
        nm   = row["Name"].split("(")[0].strip()
        brief += f"""
<div style="background:{ABYSS};border:0.5px solid rgba(123,156,255,0.1);
border-left:2px solid {c};border-radius:0 6px 6px 0;
padding:8px 12px;display:flex;justify-content:space-between;align-items:center">
  <div>
    <div style="font-size:12px;color:{ICE};margin-bottom:2px">{flag}&nbsp;{nm}</div>
    <div style="font-size:10px;color:rgba(238,242,255,0.3)">{row["Sector"]} · {row["Unit"]}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:13px;color:{c};font-weight:500">{p:+.2f}%</div>
    <div style="font-size:10px;color:rgba(238,242,255,0.3)">{row["Price"]:,.2f}</div>
  </div>
</div>"""

    brief += "</div></div>"
    return brief


# ── Load live data ────────────────────────────────────────────────────────────
with st.spinner(""):
    df = fetch_current_prices()

render_topbar(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/accendio_logo_dark_630x120.png", use_container_width=True)
    st.caption("Commodity Intelligence. Ignited.")
    st.divider()
    st.markdown("""
**Navigation**
- **Overview** ← you are here
- **Pricing** — full price table
- **Charts** — interactive price history
- **News** — live market news
- **Models** — analytics pipeline
- **Database** — data management
    """)
    st.divider()
    st.caption(f"Last refresh · {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    if st.button("↻  Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if df.empty:
    st.error("Could not load price data. Check your internet connection.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
n_adv = int((df["Pct_Change"] > 0).sum())
n_dec = int((df["Pct_Change"] < 0).sum())
n_tot = len(df)

st.markdown(
    f'<h1 style="margin-bottom:4px">Command Centre</h1>'
    f'<p style="color:rgba(238,242,255,0.28);font-size:12px;letter-spacing:.05em;margin-top:0">'
    f'{n_tot} instruments &nbsp;·&nbsp;'
    f'<span style="color:{ASCEND}">{n_adv} advancing</span>&nbsp;&nbsp;'
    f'<span style="color:{DESCEND}">{n_dec} declining</span>'
    f'</p>',
    unsafe_allow_html=True,
)

# ── Sub-tabs ──────────────────────────────────────────────────────────────────
tab_live, tab_watch, tab_alerts = st.tabs(["LIVE", "WATCHLIST", "ALERTS"])

# ─────────────────────────────────────────────────────────────────────────────
# LIVE TAB — five-panel Command Centre grid
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:

    # ── Row 1: Heatmap (wide) + Signals (narrow) ─────────────────────────────
    col_heat, col_sig = st.columns([3, 2], gap="medium")

    # Panel 1 — Global Commodity Heatmap
    with col_heat:
        panel_header("Global Commodity Heatmap", badge="LIVE")

        df_tm = df.copy()
        df_tm["label"] = df_tm.apply(
            lambda r: f"{r['Name'].split('(')[0].strip()[:18]}<br>{r['Pct_Change']:+.2f}%",
            axis=1,
        )
        df_tm["color_val"] = df_tm["Pct_Change"].clip(-5, 5)

        fig_heat = go.Figure(go.Treemap(
            ids        = df_tm["Name"],
            labels     = df_tm["label"],
            parents    = df_tm["Sector"],
            values     = [1] * len(df_tm),
            customdata = df_tm[["Name", "Price", "Pct_Change", "Unit"]].values,
            marker=dict(
                colors=df_tm["color_val"],
                colorscale=[
                    [0.0,  DESCEND],
                    [0.38, "#2C1F1F"],
                    [0.5,  "#141C38"],
                    [0.62, "#1A2B1A"],
                    [1.0,  ASCEND],
                ],
                cmid=0,
                showscale=False,
                line=dict(width=0.5, color="rgba(9,16,42,0.8)"),
            ),
            textinfo  = "label",
            textfont  = dict(color=ICE, size=11),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Price: %{customdata[1]:,.2f} %{customdata[3]}<br>"
                "Change: %{customdata[2]:+.2f}%<extra></extra>"
            ),
        ))
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # Panel 2 — Top Active Signals
    with col_sig:
        n_active = int((df["Pct_Change"].abs() > 1.0).sum())
        panel_header("Top Active Signals", badge=f"{n_active} active")

        top8 = df.sort_values("Pct_Change", key=abs, ascending=False).head(8)

        for _, row in top8.iterrows():
            p  = row["Pct_Change"]
            c  = ASCEND if p > 0 else DESCEND
            dl = "BULL" if p > 0 else "BEAR"
            if abs(p) > 3:
                sig_lbl = "STRONG MOVE"
            elif abs(p) > 1.5:
                sig_lbl = "MOMENTUM"
            else:
                sig_lbl = "MOVE"
            nm = row["Name"].split("(")[0].strip()
            if len(nm) > 22:
                nm = nm[:20] + "…"

            st.markdown(f"""
<div style="padding:8px 12px;background:{DEPTH};border:0.5px solid rgba(123,156,255,0.1);
border-radius:6px;margin-bottom:5px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="font-size:12px;color:{ICE}">{nm}</span>
    <span style="font-size:12px;color:{c};font-weight:500">{p:+.2f}%</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px">
    <span style="font-size:10px;color:rgba(238,242,255,0.3)">{row["Sector"]} · {sig_lbl}</span>
    <span style="font-size:9px;color:{c};padding:1px 6px;border-radius:10px;
    background:rgba(123,156,255,0.06);border:0.5px solid {c}40">{dl}</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Row 2: Sector Performance Timeline (full width) ───────────────────────
    panel_header("Sector Performance Timeline", badge="30 TRADING DAYS")

    timeline = _load_sector_timeline()
    if timeline is not None and not timeline.empty:
        fig_tl = go.Figure(go.Heatmap(
            z          = timeline.values,
            x          = [str(d)[-5:] for d in timeline.columns],
            y          = timeline.index.tolist(),
            colorscale = [
                [0.0, DESCEND], [0.38, "#2C1F1F"],
                [0.5, "#141C38"], [0.62, "#1A2B1A"],
                [1.0, ASCEND],
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(
                title=dict(text="%", font=dict(color=ICE_MID, size=10)),
                tickfont=dict(color=ICE_MID, size=9),
                thickness=8,
                len=0.75,
            ),
            hovertemplate="%{y}<br>%{x}: %{z:+.2f}%<extra></extra>",
        ))
        fig_tl.update_layout(
            **PLOTLY_LAYOUT,
            height=170,
            margin=dict(l=80, r=60, t=4, b=36),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(tickfont=dict(size=11), showgrid=False),
        )
        st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Avg daily return by sector · derived from aligned price history  ·  "
            "HMM regime labels available in Models → Regimes"
        )
    else:
        st.markdown(
            f'<div style="padding:28px;background:{DEPTH};border:0.5px solid {BORDER};'
            f'border-radius:8px;text-align:center">'
            f'<span style="color:rgba(238,242,255,0.28);font-size:13px">'
            f'Historical data not yet populated — run the ingestion pipeline to enable this panel</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Row 3: Cross-Market Correlations + Intelligence Brief ─────────────────
    col_net, col_brief = st.columns([2, 3], gap="medium")

    # Panel 4 — Cross-Market Correlations
    with col_net:
        panel_header("Cross-Market Correlations", badge="60D")

        corr = _load_correlations()
        if corr is not None and not corr.empty:
            fig_corr = go.Figure(go.Heatmap(
                z          = corr.values,
                x          = corr.columns.tolist(),
                y          = corr.index.tolist(),
                colorscale = "RdBu_r",
                zmid=0, zmin=-1, zmax=1,
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color=ICE_MID, size=9),
                    thickness=8,
                    len=0.75,
                ),
                hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=9, color=ICE),
            ))
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                margin=dict(l=60, r=10, t=4, b=60),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
            st.caption("60-day rolling Pearson correlation · sector representative instruments")
        else:
            st.markdown(
                f'<div style="padding:28px;background:{DEPTH};border:0.5px solid {BORDER};'
                f'border-radius:8px;text-align:center">'
                f'<span style="color:rgba(238,242,255,0.28);font-size:12px">'
                f'Populate DB to enable cross-market analysis</span></div>',
                unsafe_allow_html=True,
            )

    # Panel 5 — Today's Intelligence Brief
    with col_brief:
        panel_header("Today's Intelligence Brief", badge="AUTO")
        st.markdown(_build_brief(df), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST / ALERTS (stubs — wired in later)
# ─────────────────────────────────────────────────────────────────────────────
with tab_watch:
    st.markdown(
        f'<div style="margin-top:3.5rem;text-align:center">'
        f'<p style="color:rgba(238,242,255,0.18);font-size:12px;letter-spacing:.1em">WATCHLIST</p>'
        f'<p style="color:rgba(238,242,255,0.12);font-size:11px">Custom commodity watchlists with threshold alerts — coming soon</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

with tab_alerts:
    st.markdown(
        f'<div style="margin-top:3.5rem;text-align:center">'
        f'<p style="color:rgba(238,242,255,0.18);font-size:12px;letter-spacing:.1em">ALERTS</p>'
        f'<p style="color:rgba(238,242,255,0.12);font-size:11px">Signal and regime-change alert configuration — coming soon</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
