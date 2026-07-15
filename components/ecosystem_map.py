"""Vertical Ecosystem Map (spec §D): water-column bands + macro current column."""
import streamlit as st

from config.ecosystem_registry import PAGES, ZONE_ORDER, MACRO_FEEDS, cached_fact
from utils.theme import ZONES

# Edge labels between falling bands, keyed by (upper_zone, lower_zone).
_FALL_LABELS = {
    ("data", "signals"): "log-returns · aligned calendar ↓",
    ("signals", "risk"): "cascade forecasts · meta-weights ↓",
}
_ZONE_SUBTITLES = {"data": "surface", "signals": "mid-depth", "risk": "floor"}


def build_map_bands():
    """Pure: ([{zone, pages:[keys]}...] in ZONE_ORDER, [macro keys])."""
    bands = [
        {"zone": z, "pages": [k for k, p in PAGES.items() if p["zone"] == z]}
        for z in ZONE_ORDER
    ]
    macro_col = [k for k, p in PAGES.items() if p["zone"] == "macro"]
    return bands, macro_col


def _node_card(key: str) -> None:
    p = PAGES[key]
    z = ZONES[p["zone"]]
    facts = [e for e in p.get("upstream", []) + p.get("downstream", []) if "fact" in e]
    status = cached_fact(facts[0]["fact"]) if facts else ""
    with st.container(border=True):
        st.page_link(p["nav"], label=p["name"])
        if status:
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']};margin-top:-6px'>{status}</div>",
                unsafe_allow_html=True,
            )


def render_ecosystem_map() -> None:
    bands, macro_col = build_map_bands()
    main, side = st.columns([2.6, 1])

    with main:
        prev_zone = None
        for band in bands:
            zone = band["zone"]
            z = ZONES[zone]
            if prev_zone:
                st.markdown(
                    f"<div style='padding:2px 0 2px 30px;font-size:10px;"
                    f"color:rgba(238,242,255,0.35)'>│ {_FALL_LABELS[(prev_zone, zone)]}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']};letter-spacing:.16em;"
                f"margin:6px 0 4px 0'>◈ {z['label']} — {_ZONE_SUBTITLES[zone]}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(max(len(band["pages"]), 1))
            for col, key in zip(cols, band["pages"]):
                with col:
                    _node_card(key)
            prev_zone = zone

    with side:
        z = ZONES["macro"]
        st.markdown(
            f"<div style='font-size:10px;color:{z['accent']};letter-spacing:.16em;"
            f"margin:6px 0 4px 0'>◈ {z['label']} — the current</div>",
            unsafe_allow_html=True,
        )
        for key in macro_col:
            _node_card(key)
        for zone, label in MACRO_FEEDS:
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']}'>← {label} → "
                f"<span style='color:{ZONES[zone]['accent']}'>{ZONES[zone]['label'].split(' ')[0]}</span></div>",
                unsafe_allow_html=True,
            )
