"""Slim per-page flow footer (spec §C): upstream fact · zone breadcrumb · downstream fact."""
import streamlit as st

from config.ecosystem_registry import PAGES, cached_fact
from utils.theme import ZONES, _ecosystem_on


def render_flow_footer(page_key: str) -> None:
    if not _ecosystem_on() or page_key not in PAGES:
        return
    entry = PAGES[page_key]
    zone = entry["zone"]

    crumb = " ▸ ".join(
        f"<b style='color:{ZONES[z]['accent']}'>{ZONES[z]['label'].split(' ')[0]}</b>"
        if z == zone else
        f"<span style='color:rgba(238,242,255,0.3)'>{ZONES[z]['label'].split(' ')[0]}</span>"
        for z in ("data", "signals", "risk", "macro")
    )
    st.markdown(
        f"<hr style='margin:1.2rem 0 0.4rem 0'>"
        f"<div style='text-align:center;font-size:10px;letter-spacing:.08em'>{crumb}</div>",
        unsafe_allow_html=True,
    )
    up = entry.get("upstream", [])
    down = entry.get("downstream", [])
    col_up, col_down = st.columns(2)
    with col_up:
        for e in up[:2]:
            tgt = PAGES[e["page"]]
            fact = f" · {cached_fact(e['fact'])}" if "fact" in e else ""
            st.page_link(tgt["nav"], label=f"← {tgt['name']} · {e['label']}{fact}")
    with col_down:
        for e in down[:2]:
            tgt = PAGES[e["page"]]
            fact = f" · {cached_fact(e['fact'])}" if "fact" in e else ""
            st.page_link(tgt["nav"], label=f"{tgt['name']} · {e['label']}{fact} →")
