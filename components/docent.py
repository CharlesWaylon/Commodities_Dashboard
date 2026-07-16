"""Docent Mode (spec §E): Guided/Analyst toggle + per-panel ⓘ popovers."""
import os

import streamlit as st

from config.ecosystem_registry import DOCENT


def _enabled() -> bool:
    # Default ON as of 2026-07-15 (ecosystem UI is the default experience);
    # rollback = DOCENT_ENABLED=false.
    return os.getenv("DOCENT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def docent_mode() -> str:
    # The widget key is authoritative: it is updated at the START of a rerun,
    # so panels rendered before the sidebar toggle still see the fresh value.
    if "docent_guided" in st.session_state:
        return "guided" if st.session_state["docent_guided"] else "analyst"
    return st.session_state.get("docent_mode", "guided")


def docent_toggle() -> None:
    """Sidebar Guided/Analyst switch. Call from render_sidebar_nav."""
    if not _enabled():
        return
    if "docent_guided" not in st.session_state:
        st.session_state["docent_guided"] = True   # default: Guided
    guided = st.toggle(
        "Guided mode",
        key="docent_guided",
        help="Plain-English ⓘ explanations on every panel. Switch off for a clean analyst view.",
    )
    st.session_state["docent_mode"] = "guided" if guided else "analyst"


# Docent popover triggers render as compact chips, not full-width buttons.
_POPOVER_CSS = """<style>
button[data-testid="stPopoverButton"] {
  padding: 0px 9px !important;
  min-height: 22px !important;
  height: 22px !important;
  font-size: 11px !important;
  border-radius: 11px !important;
  margin-top: -6px !important;
}
</style>"""


def docent(panel_id: str) -> None:
    """Render the ⓘ popover for a panel. No-op unless enabled + Guided + content exists."""
    if not _enabled() or docent_mode() != "guided":
        return
    text = DOCENT.get(panel_id)
    if not text:
        return
    st.markdown(_POPOVER_CSS, unsafe_allow_html=True)
    with st.popover("ⓘ"):
        st.markdown(text)
