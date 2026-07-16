"""Accendio Intelligence Roadmap | Alpha Phase (spec §G). Gated by ROADMAP_ENABLED."""
import json
import os
from pathlib import Path

import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav, panel_header, AMBER
from components.docent import docent

st.set_page_config(page_title="Accendio | Roadmap", page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Accendio Intelligence Roadmap | Alpha Phase")
docent("roadmap_description")

if os.getenv("ROADMAP_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
    st.info("**Roadmap is off.** Set `ROADMAP_ENABLED=true` to enable this alpha surface.")
    st.stop()

from components.roadmap_panels import render_signal_lab_panel, render_research_library_panel  # noqa: E402

left, right = st.columns([1.1, 1])
with left:
    panel_header("Signal Lab", badge="ROADMAP · Alpha", badge_color=AMBER)
    render_signal_lab_panel()
with right:
    panel_header("Research Library", badge="ROADMAP · Alpha", badge_color=AMBER)
    render_research_library_panel()

# ── Development milestones (data-driven) ─────────────────────────────────────
panel_header("Development Milestones", badge="ALPHA ROADMAP", badge_color=AMBER)
_ICON = {"done": "✅", "in_progress": "🟡", "planned": "⬜"}
milestones = json.loads(Path("config/roadmap_milestones.json").read_text())
cols = st.columns(len(milestones))
for col, m in zip(cols, milestones):
    with col:
        st.markdown(f"{_ICON[m['status']]} **{m['label']}**")
        st.caption(m.get("note", ""))


# ── Alpha feedback ───────────────────────────────────────────────────────────
@st.dialog("Provide Alpha Feedback")
def _feedback_dialog():
    msg = st.text_area("What's working? What's missing?", max_chars=2000)
    contact = st.text_input("Contact (optional)")
    if st.button("Submit", type="primary") and msg.strip():
        from database.db import get_db
        from database.models import AlphaFeedback
        with get_db() as session:
            session.add(AlphaFeedback(page="roadmap", message=msg.strip(),
                                      contact=contact.strip() or None))
            session.commit()
        st.success("Thank you — logged.")


if st.button("Provide Alpha Feedback"):
    _feedback_dialog()
