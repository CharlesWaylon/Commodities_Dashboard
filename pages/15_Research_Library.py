"""
Research Library — downloadable, distilled knowledge compilation.

A flagged, additive surface (Dashboard Evolution Rule) that turns the project's
internal knowledge documents (README, Engineering History, Model Verification
Log, Methodology) into a reader-facing research compilation. It strips code/setup
plumbing and foregrounds the structural, evaluation, and actionable-edge content —
the goal is to make the dashboard *explainable* rather than a black box.

Gated by RESEARCH_LIBRARY_ENABLED (default off). Thin by design: every parsing
and distillation decision lives in the knowledge/ layer (knowledge/digest.py),
which is deterministic (no LLM, no DB, no network) and unit-tested headlessly;
the panel body is shared with the Roadmap page via components/roadmap_panels.py.
"""

import os

import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav


def _enabled() -> bool:
    return os.getenv("RESEARCH_LIBRARY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


st.set_page_config(
    page_title="Accendio | Research Library",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Research Library")

if not _enabled():
    st.info(
        "**Research Library is off.** This is an explainability surface gated behind "
        "the `RESEARCH_LIBRARY_ENABLED` feature flag. Set `RESEARCH_LIBRARY_ENABLED=true` "
        "to enable it."
    )
    st.stop()

from components.roadmap_panels import render_research_library_panel  # noqa: E402  (import after flag/stop guard)

render_research_library_panel()
