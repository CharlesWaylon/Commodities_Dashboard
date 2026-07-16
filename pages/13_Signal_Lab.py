"""
Signal Lab — RESEARCH-GRADE signal diagnostics (NOT promoted).

A flagged, additive surface (Dashboard Evolution Rule) that exposes the Phase-2
signal-research output: the equal-weight ensemble of right-signed-but-sub-threshold
edges and the gate scorecards behind it. NOTHING here has passed the out-of-sample
gate — the page is explicitly labelled research-grade so the institutional audience
is never misled into treating it as a live, promoted signal.

Gated by SIGNAL_RESEARCH_ENABLED (default off). Thin by design: all computation
lives in the signal layer (signals/) and evaluation/reporting.py; the panel body
is shared with the Roadmap page via components/roadmap_panels.py.
"""

import os

import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav


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

from components.roadmap_panels import render_signal_lab_panel  # noqa: E402  (import after flag/stop guard)

render_signal_lab_panel()
