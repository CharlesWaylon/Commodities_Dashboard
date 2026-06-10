"""
Knowledge layer — research-explainability surface.

Distils the project's internal knowledge documents (README, Engineering History,
Model Verification Log, Methodology) into a reader-facing compilation that strips
code/setup plumbing and foregrounds the structural, evaluation, and actionable-edge
content. Powers the flagged `pages/15_Research_Library.py` subpage.

All computation lives here (layered-architecture rule); the page stays thin. The
distiller is fully deterministic — no LLM calls in the runtime path — so the output
is auditable and reproducible.
"""

from knowledge.digest import (  # noqa: F401
    SOURCE_DOCS,
    DocSpec,
    distill_document,
    build_summary,
    compile_markdown,
    compile_html,
    available_documents,
)
