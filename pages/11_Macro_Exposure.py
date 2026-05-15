"""
Macro Exposure — Page 11
==========================
Dashboard page combining the two Mission 8 components:

  Panel A  Macro Exposure Heatmap  — cluster strip + trigger impact matrix
  Panel B  Commodity Cards Grid    — price cards with ripple/badge state

The lifecycle manager provides the elevation state for both panels.
Price data comes from ``fetch_current_prices()``.

Refreshes automatically every 30 s when any trigger is active; otherwise
provides a manual refresh button (avoids unnecessary re-runs).
"""

from __future__ import annotations

import time

import streamlit as st

from components.commodity_cards import commodity_cards_grid, recal_css
from components.macro_heatmap import macro_exposure_heatmap
from services.trigger_lifecycle import LIFECYCLE
from utils.theme import (
    apply_theme, render_topbar, panel_header,
    SIGNAL, ASCEND, DESCEND, AMBER, ICE, ICE_MID,
    VOID, DEPTH, ABYSS, BORDER,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accendio · Macro Exposure",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()
render_topbar()


# ── Auto-refresh when triggers are active ─────────────────────────────────────

snap = LIFECYCLE.state_snapshot()
_has_active = snap["active_count"] > 0

# Streamlit fragment-based auto-refresh (Streamlit 1.33+)
# Falls back gracefully on older versions.
_REFRESH_INTERVAL = 30  # seconds

if _has_active:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is not None:
            # Schedule a rerun every REFRESH_INTERVAL seconds while triggers are live
            time.sleep(0)  # yield
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<p style="font-size:10px;letter-spacing:.14em;color:{ICE_MID};'
        f'text-transform:uppercase;margin-bottom:8px">Macro Exposure</p>',
        unsafe_allow_html=True,
    )

    # Live status
    n_active   = snap["active_count"]
    n_elevated = len(snap["elevated_commodities"])
    status_col = ASCEND if n_active > 0 else "rgba(238,242,255,0.25)"

    st.markdown(
        f'<div style="font-size:10px;color:rgba(238,242,255,0.32);letter-spacing:.1em;'
        f'text-transform:uppercase;margin-bottom:6px">Trigger Status</div>'
        f'<div style="font-size:22px;font-weight:300;color:{status_col};line-height:1">'
        f'{n_active}</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.3)">'
        f'active trigger{"s" if n_active != 1 else ""}</div>'
        f'<div style="font-size:10px;color:rgba(238,242,255,0.25);margin-top:4px">'
        f'{n_elevated} commodities elevated</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Active trigger quick list
    if snap["triggers"]:
        st.markdown(
            f'<div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;'
            f'color:rgba(238,242,255,0.22);margin-bottom:6px">Active Triggers</div>',
            unsafe_allow_html=True,
        )
        for t in snap["triggers"]:
            sev_colors = {
                "LOW": "rgba(238,242,255,0.4)",
                "MEDIUM": "#F59E0B",
                "HIGH": "#e07020",
                "CRITICAL": "#D94F4F",
            }
            c = sev_colors.get(t["severity"], "#888")
            ttl_min = t["remaining_ttl_seconds"] / 60
            st.markdown(
                f'<div style="font-size:10px;color:{c};margin-bottom:3px">'
                f'● {t["display_name"]}</div>'
                f'<div style="font-size:9px;color:rgba(238,242,255,0.22);'
                f'margin-bottom:6px;padding-left:10px">'
                f'{t["severity"]} · {ttl_min:.0f} min TTL</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No active triggers")

    st.markdown("---")

    # Force-expire control
    if snap["triggers"]:
        trigger_types = [t["trigger_type"] for t in snap["triggers"]]
        to_expire = st.selectbox(
            "Force-expire trigger", ["— select —"] + trigger_types, key="expire_sel"
        )
        if to_expire and to_expire != "— select —":
            if st.button("Dismiss", type="secondary", use_container_width=True):
                LIFECYCLE.force_expire(to_expire)
                st.success(f"Expired: {to_expire}")
                st.rerun()

    # Manual refresh
    if st.button("↻  Refresh", use_container_width=True):
        st.rerun()

    if _has_active:
        st.caption(f"Auto-refreshing every {_REFRESH_INTERVAL}s while triggers are active.")


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    f'<h1 style="margin-bottom:2px">Macro Exposure</h1>'
    f'<p style="font-size:12px;color:rgba(238,242,255,0.35);margin-bottom:0">'
    f'Active trigger surface · cluster elevation · commodity ripple state</p>',
    unsafe_allow_html=True,
)
st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)


# ── Panel A: Macro Exposure Heatmap ──────────────────────────────────────────

panel_header(
    "MACRO EXPOSURE HEATMAP",
    badge=f"{n_active} active trigger{'s' if n_active != 1 else ''}",
    badge_color=ASCEND if n_active > 0 else "rgba(238,242,255,0.25)",
)

if n_active == 0:
    st.markdown(
        f'<div style="background:{DEPTH};border:0.5px solid {BORDER};border-radius:8px;'
        f'padding:24px;text-align:center;color:rgba(238,242,255,0.22);font-size:12px;'
        f'letter-spacing:.06em">No active macro triggers — heatmap will populate when '
        f'TriggerSignals are classified and activated via the Lifecycle Manager</div>',
        unsafe_allow_html=True,
    )
    # Still render the empty matrix so users can see the trigger surface
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    macro_exposure_heatmap(LIFECYCLE)
else:
    macro_exposure_heatmap(LIFECYCLE)


# ── Divider ───────────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
st.markdown(f'<hr style="border-color:{BORDER};margin:0">', unsafe_allow_html=True)
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)


# ── Panel B: Commodity Cards ──────────────────────────────────────────────────

panel_header(
    "COMMODITY CARDS",
    badge=f"{n_elevated} elevated" if n_elevated > 0 else "no elevation",
    badge_color=AMBER if n_elevated > 0 else "rgba(238,242,255,0.25)",
)

# Load price data
@st.cache_data(ttl=60, show_spinner=False)
def _load_prices():
    from services.price_data import fetch_current_prices
    return fetch_current_prices()


with st.spinner(""):
    df = _load_prices()

if df is None or df.empty:
    st.warning("Price data unavailable. Cards will render when data loads.")
else:
    # Sector filter
    col_filter, col_sort, _ = st.columns([2, 2, 4])
    with col_filter:
        sectors = ["All"] + sorted(df["Sector"].dropna().unique().tolist())
        sel_sector = st.selectbox("Filter sector", sectors, key="card_sector")
    with col_sort:
        show_elevated_only = st.checkbox(
            "Elevated only", value=False, key="card_elevated_only",
            help="Show only commodities currently in an active trigger's primary scope",
        )

    display_df = df.copy()
    if sel_sector != "All":
        display_df = display_df[display_df["Sector"] == sel_sector]
    if show_elevated_only:
        elevated_set = set(snap["elevated_commodities"])
        display_df = display_df[display_df["Name"].isin(elevated_set)]

    if display_df.empty:
        st.markdown(
            '<div style="color:rgba(238,242,255,0.25);font-size:12px;'
            'text-align:center;padding:20px 0">No commodities match the current filter</div>',
            unsafe_allow_html=True,
        )
    else:
        commodity_cards_grid(
            df=display_df,
            lifecycle=LIFECYCLE,
            max_elevated=24,
            key="main_cards",
        )


# ── Auto-refresh footer ───────────────────────────────────────────────────────

if _has_active:
    st.markdown(
        f'<div style="margin-top:24px;font-size:9px;color:rgba(238,242,255,0.18);'
        f'text-align:center;letter-spacing:.06em">'
        f'Page auto-refreshes every {_REFRESH_INTERVAL} s while triggers are active · '
        f'snapshot at {snap["snapshot_at"][:19].replace("T", " ")} UTC</div>',
        unsafe_allow_html=True,
    )
    # Trigger rerun after interval (Streamlit polling pattern)
    time.sleep(_REFRESH_INTERVAL)
    st.rerun()
