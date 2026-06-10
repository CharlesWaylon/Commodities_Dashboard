"""
Research Library — downloadable, distilled knowledge compilation.

A flagged, additive surface (Dashboard Evolution Rule) that turns the project's
internal knowledge documents (README, Engineering History, Model Verification
Log, Methodology) into a reader-facing research compilation. It strips code/setup
plumbing and foregrounds the structural, evaluation, and actionable-edge content —
the goal is to make the dashboard *explainable* rather than a black box.

Gated by RESEARCH_LIBRARY_ENABLED (default off). Thin by design: every parsing
and distillation decision lives in the knowledge/ layer (knowledge/digest.py),
which is deterministic (no LLM, no DB, no network) and unit-tested headlessly.
"""

import datetime
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

from knowledge import digest  # noqa: E402  (import after flag/stop guard)

st.caption(
    "A distilled, downloadable compilation of the project's knowledge documents — "
    "built so you can research how this model is stress-tested, how it has evolved, "
    "and where its actionable edges and limits actually are."
)

# ── Controls ────────────────────────────────────────────────────────────────
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

# ── Executive digest (the synthesized 'actionable edges & evaluation' view) ──
st.markdown(summary_md)

st.divider()

# ── Per-document previews ───────────────────────────────────────────────────
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
