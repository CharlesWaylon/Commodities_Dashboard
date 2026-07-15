"""Ecosystem — live system map (spec §D). Gated by ECOSYSTEM_UI_ENABLED."""
import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav, _ecosystem_on
from components.ecosystem_map import render_ecosystem_map

st.set_page_config(
    page_title="Accendio | Ecosystem",
    page_icon="assets/accendio_icon_transparent_32.png",
    layout="wide",
)
apply_theme()          # cross-zone page: brand base, no single-zone override
render_topbar()
render_sidebar_nav()

st.title("The Accendio Ecosystem")

if not _ecosystem_on():
    st.info("**Ecosystem map is off.** Set `ECOSYSTEM_UI_ENABLED=true` to enable it.")
    st.stop()

st.caption("Every card is a live surface — click to open it. Statuses refresh every 2 minutes.")

# Node-card labels must never clip: the link's inner span defaults to
# nowrap+ellipsis — let it wrap and tighten the type instead.
st.markdown("""<style>
[data-testid="stMain"] [data-testid="stPageLink-NavLink"] span {
  white-space: normal !important;
  overflow: visible !important;
}
[data-testid="stMain"] [data-testid="stPageLink-NavLink"] p {
  font-size: 12.5px !important;
  line-height: 1.25 !important;
}
</style>""", unsafe_allow_html=True)

render_ecosystem_map()
