# UI/UX Ecosystem Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the approved ecosystem redesign — Depth Zone theming, ecosystem registry, topbar-integrated flow, vertical Ecosystem Map, Docent Mode, native-Streamlit interactivity, and the Alpha Roadmap page — in seven flag-gated increments.

**Architecture:** Zone-aware theme core in `utils/theme.py`; one registry module (`config/ecosystem_registry.py`) drives the topbar zone dots, flow footers, Ecosystem Map, and Docent popovers. All interactivity is native Streamlit 1.56 (`on_select`, `st.query_params`, `st.fragment`, `st.popover`, `st.dialog`). Computation stays below the page layer.

**Tech Stack:** Python 3.12, Streamlit 1.56, Plotly, SQLAlchemy + Postgres, pytest.

**User decisions (already made):** Scope B + reorg appendix on standby · Depth Zones palette · flow = topbar dots + slim footer (no ribbon strip) · vertical Ecosystem Map · Docent Mode only (no first-run tour) · all four interactivity types · Roadmap page at full mockup scope · Approach 1 (zone-aware theme core, no new deps).

**Spec:** `docs/superpowers/specs/2026-07-15-ui-ux-ecosystem-redesign-design.md`

**Conventions that bind every task** (from `CLAUDE.md`):
- One git branch per task (`feat/zone-theme-core`, `feat/ecosystem-registry`, …), atomic commits, merge via reviewable diff to `main`.
- Never break the unflagged path: with `ECOSYSTEM_UI_ENABLED`/`DOCENT_ENABLED`/`ROADMAP_ENABLED` unset, every page renders exactly as today.
- No model/weight/prior changes. Docent text making economic claims must match `MODEL_VERIFICATION_LOG.md` (the fertiliser 70–80% claim is already verified there, 2026-06-01 entry).
- Run the app for manual verification with: `MACRO_TRIGGERS_ENABLED=true ECOSYSTEM_UI_ENABLED=true DOCENT_ENABLED=true ROADMAP_ENABLED=true NAV_TAXONOMY_V2_ENABLED=true streamlit run app.py`

---

### Task 1: Zone theme core in `utils/theme.py` (native task #10)

**Goal:** `apply_theme(zone=None)` + `zone_plotly_layout(zone)` with four Depth Zone palettes; no-zone output byte-identical to today.

**Files:**
- Modify: `utils/theme.py` (add after the `PLOTLY_LAYOUT` block, ~line 50; change `apply_theme` at ~line 303)
- Test: `utils/test_theme_zones.py` (new)

**Acceptance Criteria:**
- [ ] `ZONES` dict with `data`/`signals`/`risk`/`macro`, each defining label, accent, bg gradient pair, panel, border, glow
- [ ] `theme_css(None)` returns exactly the legacy `_CSS` string
- [ ] `theme_css("signals")` with flag OFF returns exactly `_CSS`; with flag ON appends zone override CSS containing the zone accent
- [ ] `zone_plotly_layout("risk")` deep-copies `PLOTLY_LAYOUT` (mutating the copy leaves the original untouched) and swaps paper/plot backgrounds when flag ON
- [ ] Flag: `ECOSYSTEM_UI_ENABLED` via the existing `_nav_flag_on` helper (default off)

**Verify:** `python -m pytest utils/test_theme_zones.py -v` → all pass; `streamlit run app.py` with no env flags → visually identical to today.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `utils/test_theme_zones.py`:

```python
"""Zone theme core tests — headless, no Streamlit runtime needed."""
import os
import pytest

from utils.theme import ZONES, theme_css, zone_plotly_layout, PLOTLY_LAYOUT, _CSS


ZONE_KEYS = ("data", "signals", "risk", "macro")


def test_zones_complete():
    assert set(ZONES) == set(ZONE_KEYS)
    for z in ZONES.values():
        for field in ("label", "accent", "bg_top", "bg_bot", "panel", "border", "glow"):
            assert field in z and z[field]


def test_theme_css_no_zone_is_legacy(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    assert theme_css(None) == _CSS


def test_theme_css_flag_off_is_legacy(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "false")
    assert theme_css("signals") == _CSS


def test_theme_css_flag_on_appends_zone(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    css = theme_css("signals")
    assert css.startswith(_CSS)
    assert ZONES["signals"]["accent"] in css


def test_zone_plotly_layout_flag_on(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "true")
    layout = zone_plotly_layout("risk")
    assert layout["paper_bgcolor"] == ZONES["risk"]["panel"]
    assert layout["plot_bgcolor"] == ZONES["risk"]["bg_top"]
    # deep copy: mutating must not leak into module default
    layout["xaxis"]["gridcolor"] = "SENTINEL"
    assert PLOTLY_LAYOUT["xaxis"]["gridcolor"] != "SENTINEL"


def test_zone_plotly_layout_flag_off(monkeypatch):
    monkeypatch.setenv("ECOSYSTEM_UI_ENABLED", "false")
    assert zone_plotly_layout("risk")["paper_bgcolor"] == PLOTLY_LAYOUT["paper_bgcolor"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest utils/test_theme_zones.py -v`
Expected: FAIL / ERROR with `ImportError: cannot import name 'ZONES'`

- [ ] **Step 3: Implement in `utils/theme.py`**

Add after the `PLOTLY_LAYOUT` block (before `_CSS`):

```python
import copy
import os

# ── Depth Zones (spec §A) — approved mockup values; minor tuning in visual QA OK
ZONES = {
    "data":    dict(label="MARKETS & DATA",     accent="#5A8CFF", bg_top="#060B1A",
                    bg_bot="#0A1430", panel="#0A1430", border="rgba(90,140,255,0.22)",
                    glow="rgba(90,140,255,0.5)"),
    "signals": dict(label="SIGNALS & RESEARCH", accent="#A78BFF", bg_top="#0A0920",
                    bg_bot="#141238", panel="#12102E", border="rgba(167,139,255,0.25)",
                    glow="rgba(167,139,255,0.5)"),
    "risk":    dict(label="PORTFOLIO & RISK",   accent="#F5A65B", bg_top="#140D08",
                    bg_bot="#2A1A0E", panel="#1E130A", border="rgba(245,166,91,0.25)",
                    glow="rgba(245,166,91,0.5)"),
    "macro":   dict(label="MACRO CONTEXT",      accent="#4EC9A8", bg_top="#061410",
                    bg_bot="#0D2A20", panel="#0B241B", border="rgba(78,201,168,0.25)",
                    glow="rgba(78,201,168,0.5)"),
}


def _ecosystem_on() -> bool:
    return os.getenv("ECOSYSTEM_UI_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def zone_plotly_layout(zone: str | None = None) -> dict:
    """Per-zone PLOTLY_LAYOUT variant. Falls back to the brand default."""
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    if zone in ZONES and _ecosystem_on():
        z = ZONES[zone]
        layout["paper_bgcolor"] = z["panel"]
        layout["plot_bgcolor"] = z["bg_top"]
        layout["legend"]["bordercolor"] = z["border"]
        layout["hoverlabel"]["bordercolor"] = z["border"]
    return layout


def theme_css(zone: str | None = None) -> str:
    """Full CSS for a page. zone=None (or flag off) → legacy CSS, unchanged."""
    if zone not in ZONES or not _ecosystem_on():
        return _CSS
    z = ZONES[zone]
    return _CSS + f"""<style>
/* ── Depth Zone override: {zone} ── (topbar + sidebar deliberately untouched) */
[data-testid="stAppViewContainer"], .stApp {{
  background: linear-gradient(180deg, {z['bg_top']} 0%, {z['bg_bot']} 140%) !important;
}}
[data-testid="stMetric"], details {{
  background: {z['panel']} !important;
  border-color: {z['border']} !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  color: {z['accent']} !important;
  border-bottom-color: {z['accent']} !important;
}}
:root {{ --zone-accent: {z['accent']}; --zone-border: {z['border']}; }}
</style>"""
```

Replace the existing `apply_theme` body:

```python
def apply_theme(zone: str | None = None):
    """Inject the Accendio CSS theme (+ optional Depth Zone override).

    Call once per page after set_page_config. zone=None or flag off → legacy.
    """
    st.markdown(theme_css(zone), unsafe_allow_html=True)
```

(Note `theme_css` must be defined **after** `_CSS` in the file, or forward-reference `_CSS` — simplest: place `ZONES`/`_ecosystem_on`/`zone_plotly_layout` before `_CSS`, and `theme_css` right after `_CSS`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest utils/test_theme_zones.py -v`
Expected: 6 passed

- [ ] **Step 5: Manual no-regression check**

Run: `streamlit run app.py` (no flags). Home page must look identical to before the change. Then rerun the full suite: `python -m pytest utils/ features/ models/ -q` — no new failures vs. baseline.

- [ ] **Step 6: Commit**

```bash
git checkout -b feat/zone-theme-core
git add utils/theme.py utils/test_theme_zones.py
git commit -m "feat(theme): zone-aware Depth Zone palettes behind ECOSYSTEM_UI_ENABLED"
```

---

### Task 2: Ecosystem registry + topbar dots + flow footer, pilot on Cascade (native task #11)

**Goal:** `config/ecosystem_registry.py` (zones, edges, live facts, docent text, glossary) + topbar zone-dot indicator + flow footer component, wired into the pilot page `pages/7_Macro_Market_Cascade.py`.

**Files:**
- Create: `config/ecosystem_registry.py`, `components/flow_footer.py`
- Modify: `utils/theme.py` (`render_topbar` signature + one insert), `pages/7_Macro_Market_Cascade.py` (3 lines)
- Test: `config/test_ecosystem_registry.py` (new)

**Acceptance Criteria:**
- [ ] Registry entry for all 16 existing surfaces + `roadmap` placeholder; every entry has a valid zone and an existing nav path
- [ ] Every upstream/downstream target is itself a registry key
- [ ] Fact callables never raise — DB down ⇒ `"—"` (test monkeypatches the engine to explode)
- [ ] Topbar shows four zone dots with the current zone lit (flag ON + zone passed); flag OFF ⇒ topbar HTML unchanged
- [ ] Cascade page renders footer: `← Pricing · <n> rows   DATA ▸ SIGNALS ▸ RISK ▸ MACRO   Portfolio →`

**Verify:** `python -m pytest config/test_ecosystem_registry.py -v` → all pass; run app with flags on → pilot page shows dots + footer; flags off → identical to today.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `config/test_ecosystem_registry.py`:

```python
"""Registry integrity tests — headless."""
from pathlib import Path

from config.ecosystem_registry import PAGES, GLOSSARY, DOCENT, safe_fact


def test_every_entry_valid():
    for key, p in PAGES.items():
        assert p["zone"] in ("data", "signals", "risk", "macro"), key
        assert Path(p["nav"]).exists(), f"{key}: nav path {p['nav']} missing"
        assert p["name"]


def test_edges_point_to_registry_keys():
    for key, p in PAGES.items():
        for edge in p.get("upstream", []) + p.get("downstream", []):
            assert edge["page"] in PAGES, f"{key} → {edge['page']} not registered"


def test_facts_never_raise(monkeypatch):
    import database.db as db

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "get_engine", boom)
    for key, p in PAGES.items():
        for edge in p.get("upstream", []) + p.get("downstream", []):
            if "fact" in edge:
                out = safe_fact(edge["fact"])
                assert isinstance(out, str)  # "—" or a real value, never an exception


def test_glossary_core_terms():
    for term in ("IC", "QAOA", "regime", "damping"):
        assert term in GLOSSARY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest config/test_ecosystem_registry.py -v`
Expected: ERROR — `ModuleNotFoundError: config.ecosystem_registry`

- [ ] **Step 3: Create `config/ecosystem_registry.py`**

```python
"""Single source of truth for the ecosystem UI (spec §B).

Consumed by: topbar zone dots (utils/theme.py), flow footers
(components/flow_footer.py), the Ecosystem Map (components/ecosystem_map.py),
and Docent Mode (components/docent.py). Edit here, all four surfaces follow.
"""
from __future__ import annotations


# ── Live facts ────────────────────────────────────────────────────────────────
# Every fact is a zero-arg callable returning a short string. safe_fact() wraps
# them non-fatally (pipeline-wrapper pattern): any exception ⇒ "—".

def _count(table: str) -> str:
    from sqlalchemy import text
    from database.db import get_engine
    with get_engine().connect() as conn:
        n = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()  # noqa: S608 — table names are registry-internal literals
    return f"{n:,} rows"


def fact_aligned_rows() -> str:
    return _count("aligned_prices")


def fact_cascade_rows() -> str:
    return _count("cascade_forecasts")


def fact_active_triggers() -> str:
    import pandas as pd
    from features.macro_features import get_active_triggers
    n = len(get_active_triggers(pd.Timestamp.utcnow(), lookback_days=5))
    return f"{n} active triggers"


_FACTS = {
    "aligned_rows": fact_aligned_rows,
    "cascade_rows": fact_cascade_rows,
    "active_triggers": fact_active_triggers,
}


def safe_fact(name: str) -> str:
    """Resolve a fact by name; never raises. Cached by callers via st.cache_data."""
    try:
        return _FACTS[name]()
    except Exception:
        return "—"


def cached_fact(name: str) -> str:
    """st.cache_data(ttl=120) wrapper for page use (import-safe headless)."""
    try:
        import streamlit as st

        @st.cache_data(ttl=120, show_spinner=False)
        def _cf(n: str) -> str:
            return safe_fact(n)

        return _cf(name)
    except Exception:
        return safe_fact(name)


# ── Page registry ─────────────────────────────────────────────────────────────
# upstream/downstream edges: {"page": <registry key>, "label": str, "fact": <_FACTS key, optional>}
PAGES: dict[str, dict] = {
    "home":       dict(zone="data",    name="Command Centre",      nav="app.py",
                       upstream=[], downstream=[dict(page="models", label="log-returns")]),
    "pricing":    dict(zone="data",    name="Pricing",              nav="pages/1_Pricing.py",
                       upstream=[], downstream=[dict(page="models", label="aligned prices", fact="aligned_rows")]),
    "charts":     dict(zone="data",    name="Charts",               nav="pages/2_Charts.py",
                       upstream=[dict(page="pricing", label="price history")], downstream=[]),
    "data_health": dict(zone="data",   name="Data Health",          nav="pages/5_Database.py",
                       upstream=[dict(page="pricing", label="validation log")], downstream=[]),
    "models":     dict(zone="signals", name="Models",               nav="pages/4_Models.py",
                       upstream=[dict(page="pricing", label="aligned prices", fact="aligned_rows")],
                       downstream=[dict(page="portfolio", label="forecasts")]),
    "causal":     dict(zone="signals", name="Causal QS Engine",     nav="pages/6_Causal_QS_Engine.py",
                       upstream=[dict(page="models", label="returns")],
                       downstream=[dict(page="cascade", label="causal edges")]),
    "cascade":    dict(zone="signals", name="Macro-Market Cascade", nav="pages/7_Macro_Market_Cascade.py",
                       upstream=[dict(page="pricing", label="aligned prices", fact="aligned_rows")],
                       downstream=[dict(page="portfolio", label="cascade forecasts", fact="cascade_rows")]),
    "signal_lab": dict(zone="signals", name="Signal Lab",           nav="pages/13_Signal_Lab.py",
                       upstream=[dict(page="models", label="signal scorecards")], downstream=[]),
    "library":    dict(zone="signals", name="Research Library",     nav="pages/15_Research_Library.py",
                       upstream=[], downstream=[]),
    "portfolio":  dict(zone="risk",    name="Portfolio (QAOA)",     nav="pages/8_Portfolio.py",
                       upstream=[dict(page="cascade", label="cascade forecasts", fact="cascade_rows")],
                       downstream=[dict(page="scenarios", label="target book")]),
    "scenarios":  dict(zone="risk",    name="Scenarios",            nav="pages/9_Scenarios.py",
                       upstream=[dict(page="portfolio", label="weights")], downstream=[]),
    "alerts":     dict(zone="risk",    name="Alerts",               nav="pages/12_Alerts.py",
                       upstream=[dict(page="models", label="signals")], downstream=[]),
    "live_portfolio": dict(zone="risk", name="Live Portfolio",      nav="pages/14_Live_Portfolio.py",
                       upstream=[dict(page="portfolio", label="target weights")], downstream=[]),
    "news":       dict(zone="macro",   name="News",                 nav="pages/3_News.py",
                       upstream=[], downstream=[dict(page="cascade", label="headline corpus")]),
    "events":     dict(zone="macro",   name="Event Ribbon",         nav="pages/10_Event_Ribbon.py",
                       upstream=[], downstream=[dict(page="cascade", label="trigger events", fact="active_triggers")]),
    "exposure":   dict(zone="macro",   name="Macro Exposure",       nav="pages/11_Macro_Exposure.py",
                       upstream=[dict(page="pricing", label="returns")], downstream=[]),
}

ZONE_ORDER = ("data", "signals", "risk")   # vertical map bands, top → bottom
MACRO_FEEDS = [                            # macro column labelled feeds (spec §D)
    ("signals", "regime hints"),
    ("risk", "risk gates"),
    ("data", "trigger events"),
]


# ── Glossary ─────────────────────────────────────────────────────────────────
GLOSSARY: dict[str, str] = {
    "IC":      "Information Coefficient — correlation between forecasts and what actually happened; above ~0.03 is meaningful at daily horizons.",
    "QAOA":    "Quantum Approximate Optimization Algorithm — the optimizer used to pick portfolio weights.",
    "regime":  "The market's current 'weather': rate shock, growth shock, or commodity shock.",
    "damping": "How much an upstream sector's move is discounted before it influences a downstream forecast.",
}


# ── Docent content (spec §E) — what is this / how do I read it / why it matters
DOCENT: dict[str, str] = {
    "home_heatmap":     "**What:** every commodity sized by importance and colored by today's move. **Read it:** green = up, red = down; boxes group by sector. **Why:** one glance shows where today's action is concentrated.",
    "home_signals":     "**What:** the instruments moving hardest right now. **Read it:** BULL/BEAR tags show model direction; the number is today's move. **Why:** these are the markets most likely to matter for your book today.",
    "home_corr":        "**What:** how strongly each pair of markets moves together (last 60 days). **Read it:** red cells rise and fall together; blue cells move opposite. **Why:** two big positions in dark-red cells are secretly one position — that's hidden concentration risk.",
    "home_timeline":    "**What:** each sector's cumulative move over the last 30 trading days. **Read it:** diverging lines = sectors decoupling. **Why:** context for whether today's move is a blip or a trend.",
    "cascade_state":    "**What:** the macro backdrop the cascade model sees right now (dollar, volatility, rates). **Read it:** colored chips flag stressed readings. **Why:** the same commodity move means different things in different macro weather.",
    "cascade_flow":     "**What:** how a macro shock travels: macro channel → sector → commodity. **Read it:** thicker ribbons carry more of the shock. **Why:** energy dominates transmission into agriculture — natural gas is 70–80% of nitrogen-fertiliser cost, so energy shocks become food shocks.",
    "cascade_forecast": "**What:** each sector's forecast before and after macro adjustment. **Read it:** the 'final' column is what downstream portfolio logic consumes. **Why:** shows exactly how much the macro layer changed the model's mind.",
    "cascade_triggers": "**What:** the macro events (CPI surprises, OPEC actions, weather) currently steering the model. **Read it:** stronger triggers push forecasts harder. **Why:** this is the audit trail for 'why did the forecast move today?'",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest config/test_ecosystem_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Create `components/flow_footer.py`**

```python
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
```

- [ ] **Step 6: Add zone dots to `render_topbar` in `utils/theme.py`**

Change the signature `def render_topbar(df=None):` → `def render_topbar(df=None, zone=None):` and add this helper above it:

```python
def _zone_dots_html(zone: str | None) -> str:
    """Topbar zone indicator (spec §C). Empty string when flag off / no zone."""
    if zone not in ZONES or not _ecosystem_on():
        return ""
    dots = ""
    for key in ("data", "signals", "risk", "macro"):
        z = ZONES[key]
        if key == zone:
            dots += (f'<span style="width:7px;height:7px;border-radius:50%;background:{z["accent"]};'
                     f'box-shadow:0 0 6px {z["glow"]};display:inline-block"></span>')
        else:
            dots += (f'<span style="width:5px;height:5px;border-radius:50%;background:{z["accent"]};'
                     f'opacity:.35;display:inline-block"></span>')
    word = ZONES[zone]["label"].split(" ")[0]
    return (f'<div style="display:flex;align-items:center;gap:4px;margin-right:18px;flex-shrink:0">{dots}'
            f'<span style="font-size:8px;color:{ZONES[zone]["accent"]};letter-spacing:.1em;'
            f'margin-left:4px">{word}</span></div>')
```

In the topbar HTML f-string, insert `{_zone_dots_html(zone)}` immediately after the ACCENDIO logo `</div>` + divider block (right before the `gpills` flex div).

- [ ] **Step 7: Wire the pilot page**

In `pages/7_Macro_Market_Cascade.py`: change `apply_theme()` → `apply_theme(zone="signals")` and `render_topbar()` → `render_topbar(zone="signals")`. At the very end of the file add:

```python
from components.flow_footer import render_flow_footer
render_flow_footer("cascade")
```

- [ ] **Step 8: Manual check both flag states**

Flags on (env from header) → Cascade page: violet gradient, topbar dots with SIGNALS lit, footer with live row counts and working page links. All flags unset → page identical to today. Also flip `NAV_TAXONOMY_V2_ENABLED=true` into your standard run command from now on (spec §H step 2; the legacy nav stays available by unsetting it).

- [ ] **Step 9: Commit**

```bash
git checkout -b feat/ecosystem-registry
git add config/ecosystem_registry.py config/test_ecosystem_registry.py components/flow_footer.py utils/theme.py pages/7_Macro_Market_Cascade.py
git commit -m "feat(ecosystem): registry + topbar zone dots + flow footer, piloted on Cascade"
```

---

### Task 3: Vertical Ecosystem Map page (native task #12)

**Goal:** `pages/0_Ecosystem.py` — vertical water-column map (Data → Signals → Risk bands, Macro full-height column) with live-status node cards linking to every page.

**Files:**
- Create: `pages/0_Ecosystem.py`, `components/ecosystem_map.py`
- Modify: `utils/theme.py` (`_render_sidebar_nav_v2`: add Ecosystem link)
- Test: `components/test_ecosystem_map.py` (new)

**Implementation note (refinement vs spec §D):** node cards are styled `st.page_link` containers, not Plotly `on_select` — natively clickable, no JS bridge, identical behavior. Record this in the PR description.

**Acceptance Criteria:**
- [ ] Three zone bands in `ZONE_ORDER` top→bottom with labelled falling edges ("log-returns ↓", "cascade forecasts ↓"); Macro column beside them with the three labelled feeds from `MACRO_FEEDS`
- [ ] One card per registry page in its zone band: name + live fact (2-min cache, "—" on failure) + click-through
- [ ] Page gated: flag off ⇒ `st.info` explaining the flag, `st.stop()`
- [ ] `build_map_bands()` (pure) returns bands in order with every non-macro page placed; macro pages in the column

**Verify:** `python -m pytest components/test_ecosystem_map.py -v`; run app → click every card, each lands on its page; stop Postgres → cards show "—", no traceback.

**Steps:**

- [ ] **Step 1: Write the failing test**

Create `components/test_ecosystem_map.py`:

```python
from components.ecosystem_map import build_map_bands
from config.ecosystem_registry import PAGES


def test_bands_cover_all_pages():
    bands, macro_col = build_map_bands()
    assert [b["zone"] for b in bands] == ["data", "signals", "risk"]
    placed = {k for b in bands for k in b["pages"]} | set(macro_col)
    assert placed == set(PAGES)


def test_band_pages_match_zone():
    bands, macro_col = build_map_bands()
    for band in bands:
        for key in band["pages"]:
            assert PAGES[key]["zone"] == band["zone"]
    for key in macro_col:
        assert PAGES[key]["zone"] == "macro"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest components/test_ecosystem_map.py -v`
Expected: ERROR — `ModuleNotFoundError: components.ecosystem_map`

- [ ] **Step 3: Create `components/ecosystem_map.py`**

```python
"""Vertical Ecosystem Map (spec §D): water-column bands + macro current column."""
import streamlit as st

from config.ecosystem_registry import PAGES, ZONE_ORDER, MACRO_FEEDS, cached_fact
from utils.theme import ZONES

# Edge labels between falling bands, keyed by (upper_zone, lower_zone).
_FALL_LABELS = {
    ("data", "signals"): "log-returns · aligned calendar ↓",
    ("signals", "risk"): "cascade forecasts · meta-weights ↓",
}
_ZONE_SUBTITLES = {"data": "surface", "signals": "mid-depth", "risk": "floor"}


def build_map_bands():
    """Pure: ([{zone, pages:[keys]}...] in ZONE_ORDER, [macro keys])."""
    bands = [
        {"zone": z, "pages": [k for k, p in PAGES.items() if p["zone"] == z]}
        for z in ZONE_ORDER
    ]
    macro_col = [k for k, p in PAGES.items() if p["zone"] == "macro"]
    return bands, macro_col


def _node_card(key: str) -> None:
    p = PAGES[key]
    z = ZONES[p["zone"]]
    facts = [e for e in p.get("upstream", []) + p.get("downstream", []) if "fact" in e]
    status = cached_fact(facts[0]["fact"]) if facts else ""
    with st.container(border=True):
        st.page_link(p["nav"], label=p["name"])
        if status:
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']};margin-top:-6px'>{status}</div>",
                unsafe_allow_html=True,
            )


def render_ecosystem_map() -> None:
    bands, macro_col = build_map_bands()
    main, side = st.columns([2.6, 1])

    with main:
        prev_zone = None
        for band in bands:
            zone = band["zone"]
            z = ZONES[zone]
            if prev_zone:
                st.markdown(
                    f"<div style='padding:2px 0 2px 30px;font-size:10px;"
                    f"color:rgba(238,242,255,0.35)'>│ {_FALL_LABELS[(prev_zone, zone)]}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']};letter-spacing:.16em;"
                f"margin:6px 0 4px 0'>◈ {z['label']} — {_ZONE_SUBTITLES[zone]}</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(max(len(band["pages"]), 1))
            for col, key in zip(cols, band["pages"]):
                with col:
                    _node_card(key)
            prev_zone = zone

    with side:
        z = ZONES["macro"]
        st.markdown(
            f"<div style='font-size:10px;color:{z['accent']};letter-spacing:.16em;"
            f"margin:6px 0 4px 0'>◈ {z['label']} — the current</div>",
            unsafe_allow_html=True,
        )
        for key in macro_col:
            _node_card(key)
        for zone, label in MACRO_FEEDS:
            st.markdown(
                f"<div style='font-size:10px;color:{z['accent']}'>← {label} → "
                f"<span style='color:{ZONES[zone]['accent']}'>{ZONES[zone]['label'].split(' ')[0]}</span></div>",
                unsafe_allow_html=True,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest components/test_ecosystem_map.py -v`
Expected: 2 passed

- [ ] **Step 5: Create `pages/0_Ecosystem.py`**

```python
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
render_ecosystem_map()
```

- [ ] **Step 6: Add nav link in `utils/theme.py`**

In `_render_sidebar_nav_v2`, directly under `st.page_link("app.py", label="Home")`:

```python
    if _ecosystem_on():
        st.page_link("pages/0_Ecosystem.py", label="Ecosystem")
```

- [ ] **Step 7: Manual check**

Run app with flags on → Ecosystem in sidebar; three bands + macro column render; click every card; stop Postgres (`brew services stop postgresql` or equivalent) → facts show "—", page still renders; restart Postgres.

- [ ] **Step 8: Commit**

```bash
git checkout -b feat/ecosystem-map
git add pages/0_Ecosystem.py components/ecosystem_map.py components/test_ecosystem_map.py utils/theme.py
git commit -m "feat(ecosystem): vertical Ecosystem Map page with live-status node cards"
```

---

### Task 4: Docent Mode (native task #13)

**Goal:** `components/docent.py` — Guided/Analyst sidebar toggle + `docent(panel_id)` ⓘ popovers from registry content, wired on pilot pages (Home + Cascade).

**Files:**
- Create: `components/docent.py`
- Modify: `app.py` (4 docent calls beside `panel_header` calls), `pages/7_Macro_Market_Cascade.py` (4 docent calls), `utils/theme.py` (`render_sidebar_nav`: call toggle)
- Test: `components/test_docent.py` (new)

**Acceptance Criteria:**
- [ ] `DOCENT_ENABLED` flag (default off); off ⇒ `docent()` and toggle render nothing
- [ ] Toggle in sidebar, default Guided, persists in `st.session_state["docent_mode"]`
- [ ] Analyst mode ⇒ `docent()` renders nothing
- [ ] Coverage test: every `docent("<id>")` call in `app.py` + `pages/*.py` has a matching `DOCENT` key
- [ ] Docent text follows what/read/why format (already written in registry, Task 2)

**Verify:** `python -m pytest components/test_docent.py -v`; run app with `DOCENT_ENABLED=true` → ⓘ popovers on Home and Cascade panels in Guided, gone in Analyst.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `components/test_docent.py`:

```python
import re
from pathlib import Path

from config.ecosystem_registry import DOCENT

_CALL = re.compile(r'docent\(\s*"([a-z0-9_]+)"\s*\)')


def _used_ids():
    ids = set()
    for f in [Path("app.py"), *Path("pages").glob("*.py")]:
        ids |= set(_CALL.findall(f.read_text()))
    return ids


def test_every_docent_call_has_content():
    used = _used_ids()
    assert used, "expected docent() calls on pilot pages"
    missing = used - set(DOCENT)
    assert not missing, f"docent ids without registry content: {missing}"


def test_docent_content_format():
    for pid, text in DOCENT.items():
        for tag in ("**What:**", "**Read it:**", "**Why:**"):
            assert tag in text, f"{pid} missing {tag}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest components/test_docent.py -v`
Expected: FAIL — `_used_ids()` returns empty set (no docent calls exist yet)

- [ ] **Step 3: Create `components/docent.py`**

```python
"""Docent Mode (spec §E): Guided/Analyst toggle + per-panel ⓘ popovers."""
import os

import streamlit as st

from config.ecosystem_registry import DOCENT


def _enabled() -> bool:
    return os.getenv("DOCENT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def docent_mode() -> str:
    return st.session_state.get("docent_mode", "guided")


def docent_toggle() -> None:
    """Sidebar Guided/Analyst switch. Call from render_sidebar_nav."""
    if not _enabled():
        return
    guided = st.toggle(
        "Guided mode",
        value=(docent_mode() == "guided"),
        help="Plain-English ⓘ explanations on every panel. Switch off for a clean analyst view.",
    )
    st.session_state["docent_mode"] = "guided" if guided else "analyst"


def docent(panel_id: str) -> None:
    """Render the ⓘ popover for a panel. No-op unless enabled + Guided + content exists."""
    if not _enabled() or docent_mode() != "guided":
        return
    text = DOCENT.get(panel_id)
    if not text:
        return
    with st.popover("ⓘ"):
        st.markdown(text)
```

- [ ] **Step 4: Wire the toggle into the sidebar**

In `utils/theme.py::render_sidebar_nav`, inside the `with st.sidebar:` block after the nav render:

```python
        from components.docent import docent_toggle
        st.divider()
        docent_toggle()
```

- [ ] **Step 5: Wire pilot pages**

`app.py` — next to each of the four `panel_header(...)` calls for heatmap / signals / correlations / timeline, add on the following line:

```python
from components.docent import docent   # once, with the other imports

docent("home_heatmap")    # after panel_header("Global Commodity Heatmap", ...)
docent("home_signals")    # after panel_header("Top Active Signals", ...)
docent("home_corr")       # after panel_header("Cross-Market Correlations", ...)
docent("home_timeline")   # after panel_header("Sector Performance Timeline", ...)
```

`pages/7_Macro_Market_Cascade.py` — same pattern beside its panel headers:

```python
from components.docent import docent   # once, with the other imports

docent("cascade_state")
docent("cascade_flow")
docent("cascade_forecast")
docent("cascade_triggers")
```

Place each `docent(...)` immediately after the corresponding panel's header markdown so the ⓘ sits beside the title. (Tip: `panel_header` + `docent` can share a `st.columns([20,1])` row if the inline look needs tightening — visual QA call.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest components/test_docent.py config/test_ecosystem_registry.py -v`
Expected: all pass

- [ ] **Step 7: Manual check**

`DOCENT_ENABLED=true` run → sidebar shows Guided toggle (on); ⓘ beside 8 panels; each popover shows What/Read/Why text; toggle off → all ⓘ vanish. `DOCENT_ENABLED` unset → no toggle, no ⓘ.

- [ ] **Step 8: Commit**

```bash
git checkout -b feat/docent-mode
git add components/docent.py components/test_docent.py utils/theme.py app.py pages/7_Macro_Market_Cascade.py
git commit -m "feat(docent): Guided/Analyst toggle + registry-driven panel popovers"
```

---

### Task 5: Cross-filtering + drill-down on Home and Models (native task #14)

**Goal:** Heatmap/correlation clicks filter Home panels; signal rows deep-link into Models with the instrument pre-selected; `?commodity=` URLs shareable.

**Files:**
- Create: `utils/interactions.py`
- Modify: `app.py` (heatmap + correlation `on_select`, signal-row links), `pages/4_Models.py` (query-param pre-select)
- Test: `utils/test_interactions.py` (new)

**Acceptance Criteria:**
- [ ] Clicking a heatmap sector tile filters Top Active Signals + Sector Timeline to that sector; a "clear filter" chip restores all
- [ ] Clicking a correlation cell renders a pair-detail strip (60-day overlaid normalized closes) under the matrix
- [ ] Each Top-Signal row gets a "→ model" button that opens Models with that instrument selected (session-state hint)
- [ ] Opening `/, Models?commodity=<name>` in a fresh tab pre-selects that commodity (query param wins over hint)
- [ ] `selected_labels()` never raises on malformed/empty events

**Verify:** `python -m pytest utils/test_interactions.py -v`; manual click-through of all three paths; paste `http://localhost:8501/Models?commodity=Wheat` in a fresh tab.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `utils/test_interactions.py`:

```python
from types import SimpleNamespace

from utils.interactions import selected_labels, resolve_commodity_hint


def _event(points):
    return SimpleNamespace(selection=SimpleNamespace(points=points))


def test_selected_labels_happy_path():
    ev = _event([{"label": "Energy"}, {"label": "Wheat"}])
    assert selected_labels(ev) == ["Energy", "Wheat"]


def test_selected_labels_malformed():
    assert selected_labels(None) == []
    assert selected_labels(object()) == []
    assert selected_labels(_event([{}])) == []


def test_resolve_commodity_hint_query_param_wins():
    names = ["Crude Oil", "Wheat", "Gold"]
    assert resolve_commodity_hint("Wheat", "Gold", names) == 1
    assert resolve_commodity_hint(None, "Gold", names) == 2
    assert resolve_commodity_hint("Nope", None, names) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest utils/test_interactions.py -v`
Expected: ERROR — `ModuleNotFoundError: utils.interactions`

- [ ] **Step 3: Create `utils/interactions.py`**

```python
"""Pure helpers for chart selection events and deep links (spec §F1–F2)."""
from __future__ import annotations


def selected_labels(event) -> list[str]:
    """Labels from a st.plotly_chart on_select event. Never raises."""
    try:
        return [p["label"] for p in event.selection.points if p.get("label")]
    except Exception:
        return []


def resolve_commodity_hint(query_param: str | None, session_hint: str | None,
                           names: list[str]) -> int:
    """Selectbox index for a deep link: query param wins, then session hint, else 0."""
    for candidate in (query_param, session_hint):
        if candidate in names:
            return names.index(candidate)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest utils/test_interactions.py -v`
Expected: 3 passed

- [ ] **Step 5: Wire Home cross-filtering (`app.py`)**

Heatmap (line ~472): capture the event and derive the sector filter:

```python
heat_event = st.plotly_chart(
    fig_heat, use_container_width=True, config={"displayModeBar": False},
    on_select="rerun", key="home_heatmap",
)
from utils.interactions import selected_labels
_sectors = set(df["Sector"].unique())
picked = [l for l in selected_labels(heat_event) if l in _sectors]
sector_filter = picked[0] if picked else None
if sector_filter:
    if st.button(f"✕ clear {sector_filter} filter", key="clear_sector"):
        sector_filter = None
        st.rerun()
```

Then, in the Top Active Signals block and the Sector Timeline block, filter their source frames first:

```python
df_view = df[df["Sector"] == sector_filter] if sector_filter else df
```

Correlation matrix (line ~598): same pattern —

```python
corr_event = st.plotly_chart(
    fig_corr, use_container_width=True, config={"displayModeBar": False},
    on_select="rerun", key="home_corr",
)
pts = getattr(getattr(corr_event, "selection", None), "points", None) or []
if pts:
    p = pts[0]
    a, b = p.get("x"), p.get("y")
    if a and b and a != b:
        pair = prices[[a, b]].tail(60)
        pair = pair / pair.iloc[0] * 100.0
        import plotly.graph_objects as go
        fig_pair = go.Figure()
        for col_name in pair.columns:
            fig_pair.add_trace(go.Scatter(x=pair.index, y=pair[col_name], name=col_name, mode="lines"))
        fig_pair.update_layout(**PLOTLY_LAYOUT, height=220)
        panel_header(f"Pair detail — {a} vs {b}", badge="60D · click-through")
        st.plotly_chart(fig_pair, use_container_width=True, config={"displayModeBar": False})
```

(`prices` = the frame already used to build the correlation matrix in `app.py`; keep its actual variable name.)

Signal rows (Top Active Signals loop): add a per-row hop:

```python
if st.button("→ model", key=f"sig_model_{ticker}"):
    st.session_state["models_commodity"] = commodity_name  # display name used by MODELING_COMMODITIES
    st.switch_page("pages/4_Models.py")
```

- [ ] **Step 6: Wire Models deep link (`pages/4_Models.py`)**

At the commodity selectbox (the page's main instrument picker, which per the MODEL SCOPE RULE uses `list(MODELING_COMMODITIES.keys())`):

```python
from utils.interactions import resolve_commodity_hint
_names = list(MODELING_COMMODITIES.keys())
_idx = resolve_commodity_hint(
    st.query_params.get("commodity"),
    st.session_state.pop("models_commodity", None),
    _names,
)
selected_commodity = st.selectbox("Commodity", _names, index=_idx, key="models_commodity_select")
st.query_params["commodity"] = selected_commodity   # keeps the URL shareable
```

- [ ] **Step 7: Manual click-through**

(1) Click Energy tile → signals + timeline filter, clear-chip restores. (2) Click a corr cell → pair strip. (3) Click "→ model" on a signal row → Models opens pre-selected. (4) Paste `http://localhost:8501/Models?commodity=Wheat` in a fresh tab → Wheat pre-selected.

- [ ] **Step 8: Commit**

```bash
git checkout -b feat/cross-filter-drilldown
git add utils/interactions.py utils/test_interactions.py app.py pages/4_Models.py
git commit -m "feat(interactivity): home cross-filtering + query-param drill-down into Models"
```

---

### Task 6: What-if sandboxes + live fragments (native task #15)

**Goal:** In-memory what-if panels (cascade prior α/damping on Cascade; risk-gate toggles on Portfolio) + `st.fragment(run_every=120)` status refresh on the Ecosystem Map.

**Files:**
- Create: `models/whatif.py`
- Modify: `pages/7_Macro_Market_Cascade.py` (sandbox expander), `pages/8_Portfolio.py` (gate sandbox expander), `pages/0_Ecosystem.py` (fragment wrap)
- Test: `models/test_whatif.py` (new)

**Acceptance Criteria:**
- [ ] `models/whatif.py` is pure — imports no `database.*` and never writes files (asserted by test)
- [ ] `blended_prior(p, 0) == 1.0` for any p; `blended_prior(p, 1) == p`; contribution monotone in damping
- [ ] Cascade sandbox: α + damping sliders re-tabulate every `SECTOR_TRANSMISSION_PRIORS` edge contribution live, badged "SANDBOX — in-memory only"
- [ ] Portfolio sandbox: multiselect of `TRIGGER_RISK_GATES` families re-runs `apply_trigger_risk_gates` on the currently displayed weights, shows before/after weight table + actions applied
- [ ] Map statuses refresh via fragment (no full-page rerun every 2 min)

**Verify:** `python -m pytest models/test_whatif.py -v`; manual slider/toggle checks; watch map refresh without page flicker.

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `models/test_whatif.py`:

```python
import ast
from pathlib import Path

from models.whatif import blended_prior, upstream_contribution, prior_table


def test_blended_prior_bounds():
    assert blended_prior(0.9, 0.0) == 1.0
    assert blended_prior(0.9, 1.0) == 0.9
    assert abs(blended_prior(0.5, 0.5) - 0.75) < 1e-12


def test_contribution_monotone_in_damping():
    lo = upstream_contribution(0.4, 0.02, 0.5, 0.9, 1.0)
    hi = upstream_contribution(0.4, 0.02, 1.0, 0.9, 1.0)
    assert hi > lo


def test_prior_table_covers_all_edges():
    from models.config import SECTOR_TRANSMISSION_PRIORS
    rows = prior_table(alpha=1.0, damping=1.0)
    n_edges = sum(len(v) for v in SECTOR_TRANSMISSION_PRIORS.values())
    assert len(rows) == n_edges
    assert {"src", "dst", "prior", "effective", "contribution"} <= set(rows[0])


def test_whatif_module_is_pure():
    """Guardrail (spec §F3): the sandbox layer may never touch the DB."""
    tree = ast.parse(Path("models/whatif.py").read_text())
    imported = {
        n.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for n in (node.names if isinstance(node, ast.Import) else [node])
        if (n.name if isinstance(node, ast.Import) else node.module)
    }
    assert not any(m and m.startswith("database") for m in imported)
    assert not any(m and m.startswith("sqlalchemy") for m in imported)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest models/test_whatif.py -v`
Expected: ERROR — `ModuleNotFoundError: models.whatif`

- [ ] **Step 3: Create `models/whatif.py`**

```python
"""In-memory what-if helpers (spec §F3). Pure functions — no DB, no files.

Implements the documented cascade formula (CLAUDE.md, economic-prior
methodology): effective_prior = (1 - alpha) + alpha * prior;
upstream contribution = corr * upstream_forecast * damping * effective_prior.
"""
from __future__ import annotations

from models.config import SECTOR_TRANSMISSION_PRIORS


def blended_prior(economic_prior: float, alpha: float) -> float:
    return (1.0 - alpha) + alpha * economic_prior


def upstream_contribution(corr: float, upstream_forecast: float, damping: float,
                          economic_prior: float, alpha: float) -> float:
    return corr * upstream_forecast * damping * blended_prior(economic_prior, alpha)


def prior_table(alpha: float, damping: float,
                corr: float = 0.4, upstream_forecast: float = 0.01) -> list[dict]:
    """One row per configured transmission edge at the sandbox settings.

    corr/upstream_forecast are illustrative constants so the table isolates
    what alpha and damping change; the page labels them as such.
    """
    rows = []
    for src, targets in SECTOR_TRANSMISSION_PRIORS.items():
        for dst, prior in targets.items():
            rows.append(dict(
                src=src, dst=dst, prior=prior,
                effective=round(blended_prior(prior, alpha), 4),
                contribution=round(
                    upstream_contribution(corr, upstream_forecast, damping, prior, alpha), 6),
            ))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest models/test_whatif.py -v`
Expected: 4 passed

- [ ] **Step 5: Cascade sandbox (`pages/7_Macro_Market_Cascade.py`, end of page before flow footer)**

```python
import pandas as _pd
from models.whatif import prior_table
from models.config import UPSTREAM_PRIOR_STRENGTH

with st.expander("🧪 What-if: transmission priors (SANDBOX — in-memory only)"):
    st.caption(
        "Explore how the economic-prior blend reshapes upstream transmission. "
        "Nothing here writes to the database or model files; live forecasts are untouched."
    )
    c1, c2 = st.columns(2)
    wa = c1.slider("Prior strength α", 0.0, 1.0, float(UPSTREAM_PRIOR_STRENGTH), 0.05,
                   help="0 = pure measured correlation (legacy); 1 = full economic prior.")
    wd = c2.slider("Upstream damping ×", 0.25, 2.0, 1.0, 0.05)
    tbl = _pd.DataFrame(prior_table(alpha=wa, damping=wd))
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption("Illustrative corr=0.40, upstream forecast=+1.0% — constants isolate the α/damping effect.")
```

- [ ] **Step 6: Portfolio gate sandbox (`pages/8_Portfolio.py`, after the weights are displayed)**

The page already holds a post-QAOA weight dict (the object whose `.weights` the page renders — reuse that variable; below it is called `current_weights`).

```python
from models.config import TRIGGER_RISK_GATES
from models.portfolio_optimizer import apply_trigger_risk_gates

with st.expander("🧪 What-if: trigger risk gates (SANDBOX — in-memory only)"):
    st.caption("Simulate gate firings against the currently displayed weights. No DB writes.")
    fams = st.multiselect("Fire these gate families", sorted(TRIGGER_RISK_GATES.keys()))
    if fams and current_weights:
        simulated = [dict(family=f, strength=1.0) for f in fams]
        gated, actions = apply_trigger_risk_gates(dict(current_weights), simulated)
        import pandas as _pd
        cmp_df = _pd.DataFrame({"current": current_weights, "gated": gated}).fillna(0.0)
        cmp_df["Δ"] = (cmp_df["gated"] - cmp_df["current"]).round(4)
        st.dataframe(cmp_df, use_container_width=True)
        for a in actions:
            st.caption(f"gate applied: {a}")
```

(Execution note: confirm the page's actual weight-dict variable name and the trigger-dict field the gate reader expects — check how `apply_trigger_risk_gates` consumes `active_triggers` items at `models/portfolio_optimizer.py:234-300` and mirror the real key names, e.g. `family`/`strength`.)

- [ ] **Step 7: Fragment refresh on the map (`pages/0_Ecosystem.py`)**

```python
@st.fragment(run_every=120)
def _live_map():
    render_ecosystem_map()

_live_map()
```

(Replace the direct `render_ecosystem_map()` call. `cached_fact`'s ttl=120 means each refresh pulls at most one new DB round per fact.)

- [ ] **Step 8: Manual check**

Cascade sandbox: α→0 makes every `effective` = 1.0; α→1 restores priors; damping scales `contribution` linearly. Portfolio sandbox: firing `fed_tightening` flattens weights ~20% toward equal. Map: leave page open 4+ min → statuses update without full-page flicker.

- [ ] **Step 9: Commit**

```bash
git checkout -b feat/whatif-sandboxes
git add models/whatif.py models/test_whatif.py pages/7_Macro_Market_Cascade.py pages/8_Portfolio.py pages/0_Ecosystem.py
git commit -m "feat(whatif): in-memory prior/gate sandboxes + fragment-refreshed map"
```

---

### Task 7: Roadmap page (Alpha Phase) (native task #16)

**Goal:** `pages/16_Roadmap.py` — Signal Lab (Alpha) + Research Library (Alpha) panels (extracted from pages 13/15), JSON-driven milestones, alpha-feedback dialog → new `alpha_feedback` table.

**Files:**
- Create: `pages/16_Roadmap.py`, `components/roadmap_panels.py`, `config/roadmap_milestones.json`
- Modify: `database/models.py` (add `AlphaFeedback`), `pages/13_Signal_Lab.py` + `pages/15_Research_Library.py` (import extracted renderers), `utils/theme.py` (nav link), `config/ecosystem_registry.py` (add `roadmap` entry + `roadmap_description` docent text)
- Test: `components/test_roadmap.py` (new)

**Acceptance Criteria:**
- [ ] `ROADMAP_ENABLED` gate (default off); flag off ⇒ `st.info` + `st.stop()`
- [ ] Pages 13 and 15 render identically after extraction (they now call the shared panel functions; they are NOT deleted — retirement needs its own later proof + commit per spec §G)
- [ ] Milestones render from `config/roadmap_milestones.json` (status: done / in_progress / planned)
- [ ] Feedback dialog inserts a row into `alpha_feedback` (verified by SELECT)
- [ ] Functional description panel is `docent("roadmap_description")` content, not bespoke markup

**Verify:** `python -m pytest components/test_roadmap.py -v`; manual: submit feedback then `psql commodities -c "SELECT page, left(message,40) FROM alpha_feedback ORDER BY id DESC LIMIT 3;"`

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `components/test_roadmap.py`:

```python
import json
from pathlib import Path


def test_milestones_json_shape():
    data = json.loads(Path("config/roadmap_milestones.json").read_text())
    assert isinstance(data, list) and data
    for m in data:
        assert set(m) >= {"label", "status"}
        assert m["status"] in ("done", "in_progress", "planned")


def test_alpha_feedback_model():
    from database.models import AlphaFeedback
    cols = {c.name for c in AlphaFeedback.__table__.columns}
    assert {"id", "created_at", "page", "message", "contact"} <= cols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest components/test_roadmap.py -v`
Expected: FAIL — missing JSON file / missing model

- [ ] **Step 3: Create `config/roadmap_milestones.json`**

```json
[
  {"label": "Model A/B Sandbox",  "status": "done",        "note": "Signal Lab backtesting sandbox"},
  {"label": "Contextual Archive", "status": "in_progress", "note": "Research Library with data snapshots"},
  {"label": "BETA RELEASE",       "status": "planned",     "note": "Feature-flag defaults flip on"}
]
```

- [ ] **Step 4: Add `AlphaFeedback` to `database/models.py`**

Follow the file's existing declarative pattern (match neighbours like `TriggerEvent` for style):

```python
class AlphaFeedback(Base):
    """Alpha-phase user feedback captured from the Roadmap page (spec §G)."""

    __tablename__ = "alpha_feedback"

    id:         Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    page:       Mapped[str] = mapped_column(String(64))
    message:    Mapped[str] = mapped_column(Text)
    contact:    Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
```

(Adjust imports to the module's existing `Mapped`/`mapped_column`/`func` imports.) Create the table once:

```bash
python -c "from database.db import get_engine; from database.models import Base; Base.metadata.create_all(get_engine())"
```

- [ ] **Step 5: Extract shared panels into `components/roadmap_panels.py`**

Mechanical move, no logic changes:

1. Open `pages/13_Signal_Lab.py`. Everything after its flag gate (`st.stop()` block) is the panel body. Cut it into `components/roadmap_panels.py` as `def render_signal_lab_panel() -> None:` — move the page's imports it needs along with it. Page 13 then becomes: config/theme/nav boilerplate + flag gate + `render_signal_lab_panel()`.
2. Same for `pages/15_Research_Library.py` → `def render_research_library_panel() -> None:`.
3. Run the app and diff-eyeball both pages against `main` — byte-identical rendering expected.

- [ ] **Step 6: Create `pages/16_Roadmap.py`**

```python
"""Accendio Intelligence Roadmap | Alpha Phase (spec §G). Gated by ROADMAP_ENABLED."""
import json
import os
from pathlib import Path

import streamlit as st

from utils.theme import apply_theme, render_topbar, render_sidebar_nav, panel_header, AMBER
from components.docent import docent
from components.roadmap_panels import render_signal_lab_panel, render_research_library_panel

st.set_page_config(page_title="Accendio | Roadmap", page_icon="assets/accendio_icon_transparent_32.png", layout="wide")
apply_theme()
render_topbar()
render_sidebar_nav()

st.title("Accendio Intelligence Roadmap | Alpha Phase")
docent("roadmap_description")

if os.getenv("ROADMAP_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
    st.info("**Roadmap is off.** Set `ROADMAP_ENABLED=true` to enable this alpha surface.")
    st.stop()

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
```

(Execution note: confirm `database/db.py` session-helper name — `get_db` per CLAUDE.md's macro feed snippet; mirror how existing modules open write sessions.)

- [ ] **Step 7: Register the page**

`utils/theme.py::_render_sidebar_nav_v2`, in the Signals & Research section:

```python
    if _nav_flag_on("ROADMAP_ENABLED"):
        st.page_link("pages/16_Roadmap.py", label="Roadmap 🚧")
```

`config/ecosystem_registry.py`: add to `PAGES`:

```python
    "roadmap": dict(zone="signals", name="Roadmap (Alpha)", nav="pages/16_Roadmap.py",
                    upstream=[dict(page="signal_lab", label="sandbox results")], downstream=[]),
```

and to `DOCENT`:

```python
    "roadmap_description": "**What:** a transparent view of the platform's development pipeline — the Signal Lab sandbox for A/B backtesting, the Research Library archive of past briefs and signals, and the milestone tracker toward beta. **Read it:** amber badges mark alpha surfaces still under construction. **Why:** you can watch (and steer, via feedback) how the intelligence layer is evolving before it ships.",
```

- [ ] **Step 8: Run all tests + manual check**

Run: `python -m pytest components/ config/ utils/ models/test_whatif.py -v` → all pass.
Manual: flags on → Roadmap in nav; both alpha panels render (compare pages 13/15 side-by-side — identical); milestones row; feedback dialog inserts (verify with the psql SELECT from **Verify**). Flag off → info box only.

- [ ] **Step 9: Commit**

```bash
git checkout -b feat/roadmap-page
git add pages/16_Roadmap.py components/roadmap_panels.py components/test_roadmap.py config/roadmap_milestones.json database/models.py pages/13_Signal_Lab.py pages/15_Research_Library.py utils/theme.py config/ecosystem_registry.py
git commit -m "feat(roadmap): Alpha Phase page — signal lab + library panels, milestones, feedback capture"
```

---

## Post-plan notes

- **Spec §F4 scoping:** the `st.fragment(run_every=120)` refresh applies to the Ecosystem Map only. The topbar is re-rendered on every page interaction/navigation anyway, and its trigger count is already 2-min cached (`_cached_active_trigger_count`), so a topbar fragment would add rerun complexity for no visible freshness gain. The "live pulse" element ships as the glowing current-zone dot (Task 2).
- **Zone adoption beyond the pilot** is deliberately not a task here: after Task 2 proves the pattern, migrating each remaining page is a 2-line change (`apply_theme(zone=...)`, `render_topbar(zone=...)`, plus `render_flow_footer(key)`) done opportunistically, one commit per zone group.
- **Reorg appendix items** (Pricing+Charts merge, News+Events merge, `4_Models.py` split) stay on standby per the spec — each needs its own brainstorm → spec → plan cycle.
- **Flag-flip to defaults** (making the ecosystem UI the default experience) happens only after all seven tasks have soaked — that's the "proven" bar from the Dashboard Evolution Rule, and it's a one-line env change.
