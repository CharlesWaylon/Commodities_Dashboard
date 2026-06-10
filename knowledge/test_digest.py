"""
Tests for the knowledge digest distiller.

These run headlessly (no streamlit, no DB, no network) — the distiller is pure and
deterministic by design, so the compilation is fully reproducible and auditable.
Run: ``pytest knowledge/test_digest.py``
"""

import pytest

from knowledge import digest
from knowledge.digest import Block


# ── Parsing ─────────────────────────────────────────────────────────────────
def test_parse_separates_block_kinds():
    md = (
        "# Title\n\n"
        "Some prose.\n\n"
        "```bash\nrm -rf /\n```\n\n"
        "- item one\n- item two\n\n"
        "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
        "> a quote\n\n"
        "---\n"
    )
    kinds = [b.kind for b in digest._parse_blocks(md)]
    assert kinds == ["heading", "paragraph", "code", "list", "table", "blockquote", "hr"]


def test_heading_level_and_text():
    (b,) = digest._parse_blocks("### Deep Heading")
    assert b.kind == "heading" and b.level == 3 and b.heading_text == "Deep Heading"


# ── Commit-hash heading stripping ───────────────────────────────────────────
def test_commit_hash_heading_keeps_title_drops_hash():
    assert digest._strip_commit_hash("`be9db97` — DB plumbing hardening") == "DB plumbing hardening"
    assert digest._strip_commit_hash("`3cf8807`, `ebbfc42` — model library scaffold") == "model library scaffold"
    # A normal heading is untouched.
    assert digest._strip_commit_hash("Cross-cutting assumptions") == "Cross-cutting assumptions"


# ── Section dropping ────────────────────────────────────────────────────────
def test_drop_setup_and_troubleshooting_sections():
    md = (
        "## Overview\nKeep me.\n\n"
        "## Setup\n### 1. Clone\n`git clone x`\n\n"
        "## Troubleshooting\n| problem | fix |\n|---|---|\n| x | y |\n\n"
        "## Edges\nKeep me too.\n"
    )
    blocks = digest._filter_blocks(digest._parse_blocks(md), "research")
    headings = [b.heading_text for b in blocks if b.kind == "heading"]
    assert "Overview" in headings and "Edges" in headings
    assert "Setup" not in headings and "Troubleshooting" not in headings


def test_code_blocks_dropped_both_levels():
    md = "Prose.\n\n```python\nx = 1\n```\n"
    for level in ("research", "plain"):
        blocks = digest._filter_blocks(digest._parse_blocks(md), level)
        assert all(b.kind != "code" for b in blocks)


# ── Numeric-table handling differs by level ─────────────────────────────────
def test_dense_numeric_table_dropped_in_plain_kept_in_research():
    md = "| Instrument | Rolls | Factor |\n|---|---|---|\n| Brent | 5 | 0.504 |\n| Gold | 4 | 0.834 |\n"
    research = digest._filter_blocks(digest._parse_blocks(md), "research")
    plain = digest._filter_blocks(digest._parse_blocks(md), "plain")
    assert any(b.kind == "table" for b in research)
    assert all(b.kind != "table" for b in plain)


def test_word_table_kept_in_plain():
    md = "| Stage | Status |\n|---|---|\n| Data foundation | Solid |\n| Forecasting | Working |\n"
    plain = digest._filter_blocks(digest._parse_blocks(md), "plain")
    assert any(b.kind == "table" for b in plain)


def test_plain_strips_inline_code_keeps_snake_case_in_research():
    md = "Trains on `aligned_prices.adjusted_close` always."
    research = digest._render(digest._filter_blocks(digest._parse_blocks(md), "research"))
    plain = digest._render(digest._filter_blocks(digest._parse_blocks(md), "plain"))
    assert "`aligned_prices.adjusted_close`" in research
    assert "`" not in plain and "aligned_prices.adjusted_close" in plain


# ── Gate / verdict classification (unit-level) ──────────────────────────────
def test_gate_unit_accepts_rules_rejects_findings_and_plumbing():
    assert digest._is_gate_unit("Spearman IC > 0.05 is my actionable bar")
    assert digest._is_gate_unit("|z| > 4.0σ on log returns")
    # plumbing (commit size + file path) is not a rule
    assert not digest._is_gate_unit("`models/backtest_harness.py` (+657) — produces IC, hit-rate")


def test_verdict_bucketing_trusts_author_markers():
    assert digest._bucket_verdict("✅ Confirmed on data.", "x") == "confirmed"
    assert digest._bucket_verdict("❌ NOT promotable.", "x") == "refuted"
    assert digest._bucket_verdict("⚠️ INCONCLUSIVE — features dormant.", "x") == "inconclusive"
    assert digest._bucket_verdict("⚠️ REJECT (gate working as designed).", "x") == "refuted"
    # no marker → fall back to keywords
    assert digest._bucket_verdict("QAOA does not beat the baseline.", "x") == "refuted"


# ── Summary & compilation over the real docs ────────────────────────────────
@pytest.mark.parametrize("level", ["research", "plain"])
def test_build_summary_has_core_sections(level):
    s = digest.build_summary(level)
    assert "Actionable Edges & Evaluation" in s
    assert "Verification verdicts" in s
    assert "Known limitations" in s


def test_verification_verdicts_dedupe_one_per_entry():
    v = digest._verification_verdicts("research")
    titles = [t for t, _, _ in v]
    assert len(titles) == len(set(titles)), "each verification entry should appear once"
    assert len(v) >= 10  # the log is substantial


@pytest.mark.parametrize("level", ["research", "plain"])
def test_compile_markdown_and_html(level):
    md = digest.compile_markdown(level)
    assert md.startswith("# Accendio")
    assert "Executive Digest" in md
    html = digest.compile_html(level)
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "<table" in html  # at least one table survived
    assert "research and educational use" in html  # disclaimer present


def test_research_is_richer_than_plain():
    assert len(digest.compile_markdown("research")) > len(digest.compile_markdown("plain"))


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        digest.compile_markdown("legalese")


def test_available_documents_exist():
    docs = digest.available_documents()
    assert docs and all((digest._REPO_ROOT / d.path).exists() for d in docs)
