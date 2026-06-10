"""
Knowledge digest — deterministic distiller for the project's .md knowledge docs.

Turns the raw engineering/verification markdown into a reader-facing research
compilation. Two design goals drive every choice here:

1. **Separate plumbing from substance.** Code blocks, shell commands, commit-hash
   headings, file-path lists, and setup/troubleshooting sections are stripped.
   What survives is the *structural / evaluation / actionable-edge* content: what
   the dashboard does, how it evolved, what was stress-tested, what worked, how
   well, and where the limits are.

2. **Stay auditable.** The distiller is pure and deterministic — no LLM calls, no
   network, no DB. The same docs always produce the same compilation. This mirrors
   the project's "narratives are derived, not generated" cross-cutting assumption.

Two reading levels:
  - ``research`` — keeps technical nuance (assumptions, IC/Sharpe/verdict numbers,
    gate thresholds, known flaws). Strips only code/setup/commit plumbing.
  - ``plain``    — heavier reduction for a non-technical reader: also drops dense
    numeric tables, algorithm step-lists, and inline-code jargon.

Public API: ``SOURCE_DOCS``, ``distill_document``, ``build_summary``,
``compile_markdown``, ``compile_html``, ``available_documents``.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import markdown as _markdown

# Repo root = parent of the knowledge/ package directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Brand colours mirror utils/theme.py. Duplicated (not imported) so this module
# stays importable headlessly without pulling in streamlit for tests/export.
_VOID = "#060912"
_ABYSS = "#09102A"
_DEPTH = "#0C1228"
_SIGNAL = "#7B9CFF"
_ICE = "#EEF2FF"
_ICE_MID = "rgba(238,242,255,0.62)"
_ASCEND = "#3DB87A"
_DESCEND = "#D94F4F"
_AMBER = "#F59E0B"
_BORDER = "rgba(123,156,255,0.16)"

VALID_LEVELS = ("research", "plain")


# ──────────────────────────────────────────────────────────────────────────────
# Document registry — the curated "knowledge set" (per the agreed scope).
# Daily alert logs and pytest-cache READMEs are intentionally excluded.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DocSpec:
    path: str          # relative to repo root
    title: str         # reader-facing section title (replaces the doc's own H1)
    blurb: str         # one-line description shown under the title


SOURCE_DOCS: List[DocSpec] = [
    DocSpec(
        "README.md",
        "Overview — What the Dashboard Does",
        "The product thesis, the two-layer architecture, and the signals it surfaces.",
    ),
    DocSpec(
        "ENGINEERING_HISTORY.md",
        "Evolution — How It Was Built",
        "The phase-by-phase build narrative, the load-bearing decisions, and an honest "
        "known-flaws ledger.",
    ),
    DocSpec(
        "MODEL_VERIFICATION_LOG.md",
        "Stress-Testing — Verification Record",
        "Every model assumption checked against outside sources, with the verdict — "
        "confirmed, refuted, or inconclusive (negatives reported, not hidden).",
    ),
    DocSpec(
        "reports/methodology.md",
        "Foundation — Data Pipeline Methodology",
        "How raw prices are cleaned, roll-adjusted, and aligned before any model sees them.",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Filtering rules
# ──────────────────────────────────────────────────────────────────────────────

# Sections dropped in BOTH reading levels — pure setup/ops/plumbing. Matched
# (case-insensitive substring/prefix) against the heading text.
_DROP_SECTION_PATTERNS = [
    r"^prerequisites\b",
    r"^setup\b",
    r"^install\b",
    r"^run\b",
    r"^clone\b",
    r"^virtual environment\b",
    r"^backfill\b",
    r"register the autonomous",
    r"pipeline commands",
    r"^project structure\b",
    r"adding a commodity",
    r"adding a model",
    r"adding a trigger",
    r"adding a page",
    r"^troubleshooting\b",
    r"how to extend",
    r"reading the repo",
    r"^license\b",
    r"autonomous daily schedule",
    r"^\d+\.\s",  # numbered setup sub-steps, e.g. "1. Clone", "4. Postgres"
]

# Headings that are nothing but a commit hash (optionally several) followed by a
# descriptive title. We keep the title, drop the hash plumbing.
_COMMIT_HEADING = re.compile(
    r"^\s*`?[0-9a-f]{6,40}`?"          # first hash
    r"(?:\s*,\s*`?[0-9a-f]{6,40}`?)*"  # extra comma-separated hashes
    r"\s*[—\-:]+\s*(?P<title>.+?)\s*$"
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^([-*_])\1{2,}\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


@dataclass
class Block:
    kind: str                       # heading|paragraph|list|table|code|blockquote|hr
    lines: List[str]
    level: int = 0                  # heading depth (1-6) for kind == 'heading'
    heading_text: str = ""          # cleaned heading text for kind == 'heading'

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────
def _parse_blocks(md_text: str) -> List[Block]:
    """Line-based markdown block parser. Robust enough for these well-formed docs."""
    lines = md_text.replace("\r\n", "\n").split("\n")
    blocks: List[Block] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            buf = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                i += 1
            blocks.append(Block("code", buf))
            continue

        # Blank line.
        if stripped == "":
            i += 1
            continue

        # Heading.
        m = _HEADING_RE.match(line)
        if m:
            blocks.append(
                Block("heading", [line], level=len(m.group(1)), heading_text=m.group(2).strip())
            )
            i += 1
            continue

        # Horizontal rule.
        if _HR_RE.match(stripped):
            blocks.append(Block("hr", [line]))
            i += 1
            continue

        # Table (header row + separator row of dashes/pipes).
        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]) and "-" in lines[i + 1]:
            buf = [line]
            i += 1
            while i < n and lines[i].strip() != "" and "|" in lines[i]:
                buf.append(lines[i])
                i += 1
            blocks.append(Block("table", buf))
            continue

        # Blockquote.
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i])
                i += 1
            blocks.append(Block("blockquote", buf))
            continue

        # List.
        if _LIST_RE.match(line):
            buf = []
            while i < n and lines[i].strip() != "":
                if _HEADING_RE.match(lines[i]) or lines[i].strip().startswith("```"):
                    break
                buf.append(lines[i])
                i += 1
            blocks.append(Block("list", buf))
            continue

        # Paragraph.
        buf = []
        while i < n and lines[i].strip() != "":
            if _HEADING_RE.match(lines[i]):
                break
            if lines[i].strip().startswith("```"):
                break
            if _LIST_RE.match(lines[i]):
                break
            buf.append(lines[i])
            i += 1
        blocks.append(Block("paragraph", buf))

    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ──────────────────────────────────────────────────────────────────────────────
def _matches_drop_section(heading_text: str) -> bool:
    h = heading_text.strip().lower()
    return any(re.search(p, h) for p in _DROP_SECTION_PATTERNS)


def _strip_commit_hash(heading_text: str) -> str:
    m = _COMMIT_HEADING.match(heading_text.strip())
    if m:
        return m.group("title").strip()
    return heading_text


def _strip_inline_code(text: str) -> str:
    return _INLINE_CODE_RE.sub(r"\1", text)


def _table_numeric_ratio(block: Block) -> float:
    """Fraction of body cells that are predominantly numeric. Header + separator skipped."""
    rows = [ln for ln in block.lines if "|" in ln]
    if len(rows) < 3:
        return 0.0
    body = rows[2:]  # skip header row + separator row
    total = 0
    numeric = 0
    for row in body:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        for c in cells:
            if not c:
                continue
            total += 1
            digits = sum(ch.isdigit() for ch in c)
            letters = sum(ch.isalpha() for ch in c)
            if digits and digits >= letters:
                numeric += 1
    return numeric / total if total else 0.0


def _is_plumbing_block(block: Block) -> bool:
    """A short paragraph/list that is essentially just a file path or code token."""
    if block.kind not in ("paragraph", "list"):
        return False
    raw = block.text.strip()
    had_code = "`" in raw
    bare = _strip_inline_code(raw)
    bare = re.sub(r"[#>*\-\d.()\[\]:|]", "", bare).strip()
    return had_code and len(bare) < 12


def _should_drop_block(block: Block, level: str) -> bool:
    if block.kind == "code":
        return True
    if block.kind == "table":
        return level == "plain" and _table_numeric_ratio(block) > 0.40
    if level == "plain" and _is_plumbing_block(block):
        return True
    return False


def _transform_block(block: Block, level: str) -> Block:
    if level != "plain":
        return block
    if block.kind in ("paragraph", "list", "heading"):
        block = Block(
            block.kind,
            [_strip_inline_code(ln) for ln in block.lines],
            level=block.level,
            heading_text=_strip_inline_code(block.heading_text),
        )
    return block


# ──────────────────────────────────────────────────────────────────────────────
# Document filtering
# ──────────────────────────────────────────────────────────────────────────────
def _filter_blocks(blocks: List[Block], level: str) -> List[Block]:
    out: List[Block] = []
    drop_until_level: Optional[int] = None

    for b in blocks:
        if b.kind == "heading":
            # A heading at-or-above the dropped section's level ends the drop.
            if drop_until_level is not None and b.level <= drop_until_level:
                drop_until_level = None
            if drop_until_level is not None:
                continue
            if _matches_drop_section(b.heading_text):
                drop_until_level = b.level
                continue
            clean = _strip_commit_hash(b.heading_text)
            hashes = "#" * b.level
            out.append(Block("heading", [f"{hashes} {clean}"], level=b.level, heading_text=clean))
            continue

        if drop_until_level is not None:
            continue
        if _should_drop_block(b, level):
            continue
        out.append(_transform_block(b, level))

    return _tidy(out)


def _tidy(blocks: List[Block]) -> List[Block]:
    """Collapse consecutive HRs and drop leading/trailing HRs and empty-heading tails."""
    cleaned: List[Block] = []
    for b in blocks:
        if b.kind == "hr" and cleaned and cleaned[-1].kind == "hr":
            continue
        cleaned.append(b)
    while cleaned and cleaned[0].kind == "hr":
        cleaned.pop(0)
    while cleaned and cleaned[-1].kind in ("hr",):
        cleaned.pop()
    # Drop a trailing heading with no content after it.
    while cleaned and cleaned[-1].kind == "heading":
        cleaned.pop()
    return cleaned


def _render(blocks: List[Block]) -> str:
    return "\n\n".join(b.text for b in blocks).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Caching of parsed/filtered docs (keyed by path + mtime + level)
# ──────────────────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _read_doc(path: str) -> Optional[str]:
    p = (_REPO_ROOT / path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _filtered_doc(spec: DocSpec, level: str) -> Optional[List[Block]]:
    p = (_REPO_ROOT / spec.path)
    if not p.exists():
        return None
    key = (spec.path, p.stat().st_mtime, level)
    if key not in _CACHE:
        raw = p.read_text(encoding="utf-8", errors="replace")
        self_blocks = _parse_blocks(raw)
        # Drop the document's own top-level H1 — replaced by DocSpec.title.
        if self_blocks and self_blocks[0].kind == "heading" and self_blocks[0].level == 1:
            self_blocks = self_blocks[1:]
        _CACHE[key] = _filter_blocks(self_blocks, level)
    return _CACHE[key]


def available_documents() -> List[DocSpec]:
    """The subset of SOURCE_DOCS that actually exist on disk (graceful if one is missing)."""
    return [s for s in SOURCE_DOCS if (_REPO_ROOT / s.path).exists()]


# ──────────────────────────────────────────────────────────────────────────────
# Section extraction (for the executive summary)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_section(blocks: List[Block], heading_pattern: str) -> List[Block]:
    """Return the blocks under the first heading matching `heading_pattern` (regex,
    case-insensitive), up to the next heading of the same or higher level."""
    pat = re.compile(heading_pattern, re.IGNORECASE)
    out: List[Block] = []
    capturing = False
    capture_level = 0
    for b in blocks:
        if b.kind == "heading":
            if capturing and b.level <= capture_level:
                break
            if not capturing and pat.search(b.heading_text):
                capturing = True
                capture_level = b.level
                continue
        if capturing:
            out.append(b)
    return out


def _first_sentence(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    if len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s


def _verification_verdicts(level: str):
    """Pull (entry_title, verdict_sentence, bucket) for each verification-log entry."""
    spec = next((s for s in SOURCE_DOCS if s.path == "MODEL_VERIFICATION_LOG.md"), None)
    if spec is None:
        return []
    blocks = _filtered_doc(spec, level) or []
    results = []
    current_title = None
    seen_titles = set()
    for b in blocks:
        if b.kind == "heading" and b.level == 2:
            current_title = b.heading_text
            continue
        if not current_title or current_title in seen_titles:
            continue
        if not re.search(r"\*\*verdict", b.text, re.IGNORECASE):
            continue
        # The verdict text may sit on the same line as the **Verdict** marker or
        # spill onto following lines — operate on the whole block, sliced from the
        # first "verdict" occurrence, so multi-line verdicts survive.
        txt = _EMPH_RE.sub("", " ".join(b.text.split()))
        idx = txt.lower().find("verdict")
        tail = txt[idx:]
        tail = re.sub(r"^verdict\b[^:.—\-]*[:.—\-]+\s*", "", tail, flags=re.IGNORECASE).strip()
        if not tail:
            continue
        verdict = _first_sentence(tail)
        results.append((current_title, verdict, _bucket_verdict(verdict, current_title)))
        seen_titles.add(current_title)
    return results


_NEG_WORDS = ("reject", "refut", "not promot", "loses", "false positive", "dilute",
              "does not beat", "wrong-signed", "not promotable")
_POS_WORDS = ("confirm", "verified", "economically sound", "corrected", "gate pass", "promote")
_INC_WORDS = ("inconclusive", "not yet", "dormant")


def _bucket_verdict(verdict: str, title: str) -> str:
    """Classify a verdict. The author's own ✅/⚠️/❌ markers are deliberate and
    take priority; keyword heuristics on the verdict text (then the title) only
    decide when no marker is present."""
    v = verdict
    vlow = verdict.lower()
    if "❌" in v:
        return "refuted"
    if "✅" in v:
        return "confirmed"
    if "⚠" in v:
        return "refuted" if any(w in vlow for w in _NEG_WORDS) else "inconclusive"
    if any(w in vlow for w in _NEG_WORDS):
        return "refuted"
    if any(w in vlow for w in _POS_WORDS):
        return "confirmed"
    if any(w in vlow for w in _INC_WORDS):
        return "inconclusive"
    tlow = title.lower()
    if any(w in tlow for w in _NEG_WORDS):
        return "refuted"
    if any(w in tlow for w in _POS_WORDS):
        return "confirmed"
    return "inconclusive"


# A gate/threshold line must pair a *numeric bound* (an inequality with a number,
# or an explicit threshold phrase) with a *signal keyword* — and must not be a
# code/commit plumbing line. This keeps real decision rules ("IC > 0.05 is the
# bar", "|z| > 4.0σ", "target vol 10%") and rejects findings ("IC = −0.012") and
# plumbing ("backtest_harness.py (+657)").
_GATE_NUM = re.compile(r"[<>≥≤]\s*[+\-]?\d")
_GATE_PHRASE = re.compile(
    r"(actionable bar|promotion bar|gate bar|Z_THRESHOLD|target vol|"
    r"threshold\s*[=:]|\d+\s*(σ|bps)\b|\d+%\s*(sparsity|coverage)|"
    r"\d+\s*calendar days|\d+\s*consecutive)",
    re.IGNORECASE,
)
_GATE_KEYWORD = re.compile(
    r"(\bIC\b|\bIR\b|Sharpe|z-?score|\|z\||\bvol\b|sparsity|coverage|σ|\bbps\b|"
    r"drawdown|calendar days|consecutive|threshold|target vol|cardinality)",
    re.IGNORECASE,
)
_GATE_PLUMBING = re.compile(r"\(\+\d|\(\d{3,}|\.py\b|\.pkl\b|\.json\b")
_EMPH_RE = re.compile(r"[*`]+")  # strip bold/code markers; keep snake_case underscores

# A line earns a place in "decision rules" only if it states a *rule* (a bar, a
# threshold, a flag trigger) — not a *measurement/finding* (where a metric landed
# this sample). These two cues separate the two.
_RULE_CUE = re.compile(
    r"(promotion bar|actionable bar|gate bar|\bthreshold\b|Z_THRESHOLD|target vol|"
    r"writes any|\bflag(s|ged|ging)?\b|outside ±|is (my|the)[^.]{0,24}\bbar\b|"
    r"must (be|exceed)|require[ds]?\b)",
    re.IGNORECASE,
)
_FINDING_CUE = re.compile(
    r"(tops out|has tracked|tracked|nudges|remains|≈|best at|best standalone|"
    r"versus\b|net-negative|avg ic|currently|punch list|remediation|proposed|"
    r"scorecard|\bnull\b|sub-threshold|<<|results \()",
    re.IGNORECASE,
)
_ENUM_PREFIX = re.compile(r"^\s*(\d+\.|[-*+•])\s+")


def _clean_snippet(text: str, limit: int = 200) -> str:
    text = _EMPH_RE.sub("", " ".join(text.split())).strip(" :—-")
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return text


def _is_gate_unit(text: str) -> bool:
    if _GATE_PLUMBING.search(text):
        return False
    if not _GATE_KEYWORD.search(text):
        return False
    return bool(_GATE_NUM.search(text) or _GATE_PHRASE.search(text))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _gate_threshold_snippets(level: str, exclude_norms=None, cap: int = 14) -> List[str]:
    """Sweep paragraphs (sentence-level), list items, and table rows for numeric
    decision rules / evaluation gates. Dedupes against `exclude_norms` (the edges
    list) so the same rule isn't surfaced twice."""
    exclude_norms = exclude_norms or set()
    seen = set()
    out: List[str] = []

    def _consider(raw: str, require_cue: bool = True) -> bool:
        if not _is_gate_unit(raw):
            return False
        if require_cue and raw.count("|") >= 3:  # an inline markdown-table fragment
            return False
        if require_cue and (not _RULE_CUE.search(raw) or _FINDING_CUE.search(raw)):
            return False
        snip = _clean_snippet(_ENUM_PREFIX.sub("", raw.strip()))
        if len(snip) < 12:
            return False
        key = _norm(snip)[:60]
        if key in seen:
            return False
        if any(key[:40] and key[:40] in ex for ex in exclude_norms):
            return False
        seen.add(key)
        out.append(snip)
        return True

    for spec in available_documents():
        for b in _filtered_doc(spec, level) or []:
            if len(out) >= cap:
                return out
            if b.kind == "list":
                for ln in b.text.split("\n"):
                    _consider(ln.strip().lstrip("-*+ ").strip())
            elif b.kind == "paragraph":
                for sent in re.split(r"(?<=[.!?])\s+", b.text):
                    _consider(sent)
            elif b.kind == "table":
                # Threshold tables (e.g. the methodology audit table) are rules by
                # construction — let qualifying rows through without the cue gate.
                rows = [ln for ln in b.lines if "|" in ln][2:]  # skip header + separator
                for row in rows:
                    # Preserve escaped pipes (\| inside cells, e.g. |z|) across the split.
                    safe = row.replace("\\|", "\x00")
                    cells = [c.strip().replace("\x00", "|") for c in safe.strip().strip("|").split("|")]
                    hit = next((c for c in cells[1:] if _is_gate_unit(c)), None)
                    if hit and cells[0]:
                        _consider(f"{cells[0]} — {hit}", require_cue=False)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Public: per-document distillation
# ──────────────────────────────────────────────────────────────────────────────
def distill_document(spec: DocSpec, level: str = "research") -> str:
    """Return the distilled markdown body of one document (no DocSpec title header)."""
    _check_level(level)
    blocks = _filtered_doc(spec, level)
    if blocks is None:
        return f"_Document not found: `{spec.path}`._"
    body = _render(blocks)
    return body or "_No reader-facing content remained after distillation._"


# ──────────────────────────────────────────────────────────────────────────────
# Public: executive summary
# ──────────────────────────────────────────────────────────────────────────────
def build_summary(level: str = "research") -> str:
    """Synthesize the 'Actionable Edges & Evaluation' executive digest across all docs."""
    _check_level(level)
    parts: List[str] = []
    parts.append("# Actionable Edges & Evaluation — Executive Digest\n")
    parts.append(
        "> Auto-compiled from the project's knowledge documents. This is the "
        "*what's working, how well, and where the edges and limits are* view — "
        "pulled from the Overview, Engineering History, Verification Record, and "
        "Methodology. Every claim below also appears, in context, in the full "
        "compilation that follows."
    )

    readme = next((s for s in SOURCE_DOCS if s.path == "README.md"), None)
    eng = next((s for s in SOURCE_DOCS if s.path == "ENGINEERING_HISTORY.md"), None)

    # 1) What's working today — README status table.
    if readme:
        rb = _filtered_doc(readme, level) or []
        status = _extract_section(rb, r"how far it'?s come")
        if status:
            parts.append("## What's working today\n")
            intro = [b for b in status if b.kind == "paragraph"][:1]
            tables = [b for b in status if b.kind == "table"]
            for b in intro:
                parts.append(b.text)
            if tables:
                parts.append(tables[0].text)
            else:
                # plain mode may have dropped the numeric-ish status table — keep prose.
                tail = [b for b in status if b.kind == "paragraph"][1:2]
                for b in tail:
                    parts.append(b.text)

    # 2) Actionable edges — README's "Simple, Actionable Insights" list.
    edge_norms = set()
    if readme:
        rb = _filtered_doc(readme, level) or []
        edges = _extract_section(rb, r"actionable insights|actionable signals")
        edge_lists = [b for b in edges if b.kind == "list"]
        if edge_lists:
            parts.append("## Actionable edges you can pull")
            for b in edge_lists:
                parts.append(b.text)
                for ln in b.text.split("\n"):
                    edge_norms.add(_norm(ln)[:60])

    # 3) Verification verdicts — the stress-testing record.
    verdicts = _verification_verdicts(level)
    if verdicts:
        parts.append("## Verification verdicts — the stress-testing record\n")
        parts.append(
            f"The model-verification log records **{len(verdicts)}** documented checks "
            "against outside sources. Negatives are reported, not hidden — a refuted or "
            "rejected signal is a result, not a failure."
        )
        groups = [
            ("✅ Confirmed / promoted", "confirmed"),
            ("⚠️ Inconclusive", "inconclusive"),
            ("❌ Refuted / rejected / not promoted", "refuted"),
        ]
        for label, bucket in groups:
            items = [(t, v) for (t, v, b) in verdicts if b == bucket]
            if not items:
                continue
            parts.append(f"**{label}** ({len(items)})\n")
            lines = []
            for title, verdict in items:
                title_clean = re.sub(r"\*\*", "", title).strip()
                lines.append(f"- **{title_clean}** — {verdict}")
            parts.append("\n".join(lines))

    # 4) Gate thresholds & decision rules.
    gates = _gate_threshold_snippets(level, exclude_norms=edge_norms)
    if gates:
        parts.append("## Gate thresholds & decision rules\n")
        parts.append(
            "The numeric bars the dashboard actually uses to decide what is tradable, "
            "what gets promoted, and when a setup is in play:"
        )
        parts.append("\n".join(f"- {g}" for g in gates))

    # 5) Known limitations — Engineering History flaws + cross-cutting assumptions.
    if eng:
        eb = _filtered_doc(eng, level) or []
        flaws = _extract_section(eb, r"known flaws")
        assumptions = _extract_section(eb, r"cross-cutting assumptions")
        if flaws or assumptions:
            parts.append("## Known limitations — read these before trusting an output\n")
            if flaws:
                ftables = [b for b in flaws if b.kind == "table"]
                fpara = [b for b in flaws if b.kind == "paragraph"][:1]
                for b in fpara:
                    parts.append(b.text)
                if ftables:
                    parts.append(ftables[0].text)
                else:
                    flists = [b for b in flaws if b.kind == "list"]
                    for b in flists:
                        parts.append(b.text)
            if assumptions:
                parts.append("**Cross-cutting assumptions baked into every output:**")
                alists = [b for b in assumptions if b.kind == "list"]
                if alists:
                    for b in alists:
                        parts.append(b.text)
                else:
                    for b in [b for b in assumptions if b.kind == "paragraph"][:3]:
                        parts.append(b.text)

    return "\n\n".join(parts).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Public: full compilation (markdown + HTML)
# ──────────────────────────────────────────────────────────────────────────────
def _disclaimer() -> str:
    return (
        "This compilation is auto-distilled from the project's internal knowledge "
        "documents for research and educational use. It deliberately preserves the "
        "project's own candid assessments — including known flaws, inconclusive "
        "results, and rejected signals. Nothing here is investment advice; "
        "research-grade findings have not necessarily passed the out-of-sample "
        "promotion gate."
    )


def compile_markdown(level: str = "research") -> str:
    """The full reader-facing compilation as a single markdown string."""
    _check_level(level)
    docs = available_documents()
    today = _dt.date.today().isoformat()
    mode_label = "Research depth" if level == "research" else "Plain-English"

    parts: List[str] = []
    parts.append("# Accendio — Research & Methodology Compilation\n")
    parts.append(
        f"*Generated {today} · {mode_label} · distilled from "
        f"{len(docs)} knowledge document{'s' if len(docs) != 1 else ''}.*"
    )
    parts.append(f"> {_disclaimer()}")
    parts.append("---")

    # Executive digest.
    parts.append(build_summary(level))
    parts.append("---")

    # Table of contents.
    parts.append("## Contents")
    toc = []
    for idx, spec in enumerate(docs, start=1):
        anchor = re.sub(r"[^a-z0-9]+", "-", spec.title.lower()).strip("-")
        toc.append(f"{idx}. [{spec.title}](#{anchor}) — {spec.blurb}")
    parts.append("\n".join(toc))
    parts.append("---")

    # Each document.
    for spec in docs:
        parts.append(f"# {spec.title}")
        parts.append(f"*{spec.blurb}*")
        parts.append(distill_document(spec, level))
        parts.append("---")

    if parts and parts[-1] == "---":
        parts.pop()
    return "\n\n".join(parts).strip() + "\n"


_HTML_CSS = f"""
:root {{
  --void:{_VOID}; --abyss:{_ABYSS}; --depth:{_DEPTH}; --signal:{_SIGNAL};
  --ice:{_ICE}; --ice-mid:{_ICE_MID}; --ascend:{_ASCEND}; --descend:{_DESCEND};
  --amber:{_AMBER}; --border:{_BORDER};
}}
* {{ box-sizing: border-box; }}
body {{
  margin:0; background:var(--void); color:var(--ice);
  font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
  font-size:15px; line-height:1.65;
}}
.wrap {{ max-width: 900px; margin:0 auto; padding:56px 28px 96px; }}
.banner {{
  border:0.5px solid var(--border); border-radius:12px;
  background:linear-gradient(180deg, var(--depth), var(--abyss));
  padding:26px 28px; margin-bottom:36px;
}}
.banner h1 {{ margin:0 0 6px; border:0; padding:0; }}
.banner .meta {{ color:var(--ice-mid); font-size:13px; }}
h1, h2, h3, h4 {{ color:var(--ice); font-weight:700; line-height:1.3; }}
h1 {{ font-size:26px; margin:48px 0 14px; padding-bottom:10px;
      border-bottom:0.5px solid var(--border); }}
h2 {{ font-size:20px; margin:34px 0 12px; color:var(--signal); }}
h3 {{ font-size:16px; margin:26px 0 10px; }}
h4 {{ font-size:14px; margin:20px 0 8px; color:var(--ice-mid);
      text-transform:uppercase; letter-spacing:0.06em; }}
p {{ margin:12px 0; }}
a {{ color:var(--signal); text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
ul, ol {{ padding-left:22px; }}
li {{ margin:6px 0; }}
strong {{ color:#fff; }}
code {{
  background:rgba(123,156,255,0.10); border:0.5px solid var(--border);
  border-radius:4px; padding:1px 5px; font-size:0.86em;
  font-family:"SF Mono", Menlo, Consolas, monospace; color:#cfd9ff;
}}
blockquote {{
  margin:16px 0; padding:10px 18px; color:var(--ice-mid);
  border-left:3px solid var(--signal);
  background:rgba(123,156,255,0.05); border-radius:0 8px 8px 0;
}}
hr {{ border:0; border-top:0.5px solid var(--border); margin:30px 0; }}
table {{
  border-collapse:collapse; width:100%; margin:18px 0; font-size:13.5px;
  border:0.5px solid var(--border);
}}
th, td {{
  border:0.5px solid var(--border); padding:8px 11px; text-align:left;
  vertical-align:top;
}}
th {{ background:var(--abyss); color:var(--signal); font-weight:700; }}
tr:nth-child(even) td {{ background:rgba(123,156,255,0.03); }}
.disclaimer {{ color:var(--ice-mid); font-style:italic; }}
@media print {{
  body {{ background:#fff; color:#111; }}
  .wrap {{ max-width:100%; padding:0; }}
  h2 {{ color:#1a3aa0; }}
  a {{ color:#1a3aa0; }}
  th {{ background:#eef; color:#1a3aa0; }}
  .banner {{ background:#f4f6ff; }}
  code {{ background:#f0f0f5; color:#333; border-color:#ddd; }}
  blockquote {{ background:#f6f8ff; color:#333; }}
}}
"""


def compile_html(level: str = "research") -> str:
    """The full compilation as a single self-contained HTML document (print-to-PDF friendly)."""
    _check_level(level)
    md = compile_markdown(level)
    body_html = _markdown.markdown(
        md, extensions=["tables", "fenced_code", "sane_lists", "toc", "attr_list"]
    )
    today = _dt.date.today().isoformat()
    mode_label = "Research depth" if level == "research" else "Plain-English"
    title = "Accendio — Research &amp; Methodology Compilation"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="banner">
    <h1>Accendio — Research &amp; Methodology Compilation</h1>
    <div class="meta">Generated {today} · {mode_label}</div>
    <p class="disclaimer">{_html.escape(_disclaimer())}</p>
  </div>
  {body_html}
</div>
</body>
</html>
"""


def _check_level(level: str) -> None:
    if level not in VALID_LEVELS:
        raise ValueError(f"level must be one of {VALID_LEVELS}, got {level!r}")
