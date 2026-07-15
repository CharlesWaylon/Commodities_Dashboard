# UI/UX Ecosystem Redesign — Design Spec

**Date:** 2026-07-15
**Status:** Approved by Charles (brainstorming session with visual companion)
**Scope decision:** Option B — visual/interaction overhaul of existing pages + new Roadmap page. Reorganization ideas recorded in the appendix, not built now.

## Goals

1. Maximize interactiveness for both non-technical and advanced users.
2. Make the flow of data/information across the four-layer taxonomy visible end-to-end.
3. Deliver an "ecosystem" feel — shifting tone and color palettes across layers — while staying one recognizable Accendio product.
4. Add a tutorial layer (Docent Mode) so all users feel comfortable.

All work follows the Dashboard Evolution Rule in `CLAUDE.md`: flag-gated, additive, branch-per-change, computation pushed below the page layer. No model, weight, or prior changes are included, so no `MODEL_VERIFICATION_LOG.md` entries are triggered by this design — except that docent explanations making economic claims must be checked against that log's existing findings before shipping.

## Approach (chosen from three)

**Approach 1 — Zone-aware theme core, incremental page adoption.** Extend `utils/theme.py` rather than fork pages or build custom React components. All interactivity uses native Streamlit (`st.plotly_chart(on_select=...)`, `st.query_params`, `st.fragment`, `st.popover`, `st.dialog`) — no new dependencies, Python only.

Rejected: parallel v2 pages (double maintenance for little safety over flags); custom React component (Node toolchain risk). A custom component remains a later upgrade path for the Ecosystem Map only — the registry design keeps the map's renderer swappable.

## Section A — Zone Theme System

`apply_theme(zone: str | None = None)` in `utils/theme.py`. Four zones ("Depth Zones" palette strategy — chosen over subtle Accent Drift and heavy Full Biomes):

| Zone | Pages | Palette character | Accent |
|---|---|---|---|
| `data` | Home, Pricing, Charts, Data Health | Deep-water blues | `#5A8CFF` |
| `signals` | Models, Causal QS, Cascade, Signal Lab | Bioluminescent violet | `#A78BFF` |
| `risk` | Portfolio, Scenarios, Alerts, Live Portfolio | Ember amber | `#F5A65B` |
| `macro` | News, Event Ribbon, Macro Exposure | Canopy teal-green | `#4EC9A8` |

Each zone defines: background gradient pair, accent, panel border tint, and a zone-specific `PLOTLY_LAYOUT` variant so charts inherit the palette automatically. Background gradients are the approved mockup values: deep water `#060B1A→#0A1430`, biolume `#0A0920→#141238`, ember `#140D08→#2A1A0E`, canopy `#061410→#0D2A20` (minor tuning during visual QA is fine; character stays fixed).

**Continuity anchors:** topbar and sidebar stay brand-navy on every page. `apply_theme()` with no zone renders today's theme exactly — unmigrated pages are untouched.

## Section B — Ecosystem Registry

`config/ecosystem_registry.py` — single source of truth consumed by the topbar zone indicator, flow footers, Ecosystem Map, and Docent Mode. Per page:

- `zone`, `display_name`, `nav_path`
- `upstream` / `downstream` edges, each with a label and a status callable returning the live fact shown in flow footers and map nodes (e.g. row count, last-retrain time, active-trigger count)
- `docent` — explanations keyed by panel id (see Section E)
- Status callables are cached (`@st.cache_data(ttl=120)`) and wrapped non-fatally: on failure a node/footer shows "—", never an error. Same pattern as the pipeline wrappers.

A `glossary` dict (IC, QAOA, regime, damping, …) lives in the same module, reused across docent popovers.

## Section C — Flow presentation: Topbar-Integrated (chosen over ribbon strips)

Chosen for minimal chrome after the ribbon mockup felt overwhelming:

- **Topbar zone indicator:** four zone dots + current zone name, current dot lit in zone color with a subtle glow, added to the existing `render_topbar`. CSS pulse on the dot when triggers are active.
- **Flow footer:** slim strip at the bottom of each migrated page: `← Pricing · 52,419 rows   DATA ▸ SIGNALS ▸ RISK ▸ MACRO   Portfolio →`. Upstream/downstream are real `st.page_link` hops; facts come from the registry.
- No ribbon strip, no space cost at the top of the content region.

## Section D — Ecosystem Map (vertical)

`pages/0_Ecosystem.py` + `components/ecosystem_map.py`, top of nav under Home.

- **Vertical water-column layout** (chosen over horizontal): Markets & Data at the surface → Signals & Research at mid-depth → Portfolio & Risk on the floor, with labelled falling edges between zones ("log-returns ↓", "cascade forecasts ↓"). Macro Context is a full-height column alongside — "the current" — with labelled feeds into each zone (regime hints → Signals, risk gates → Portfolio, trigger events → Data).
- Every node is a real page card with a live status line from the registry; click → `st.switch_page` (Plotly `on_select="rerun"`).
- Statuses refresh via the 2-minute cache; failures degrade to "—".
- Vertical suits Streamlit's natural scroll and makes the Depth Zones metaphor literal.

## Section E — Docent Mode (tutorial layer; chosen over first-run tour)

`components/docent.py`:

- Sidebar **Guided / Analyst** toggle, `st.session_state`, default Guided.
- Pages call `docent("panel_id")` beside a panel header. Guided: renders a small teal ⓘ opening an `st.popover` with the registry explanation. Analyst: renders nothing.
- Explanation format, fixed: *what is this / how do I read it / why does it matter*, ~3 sentences, plain English. Teal is reserved as the "help" color so guidance never reads as data.
- Glossary terms inside explanations get short parenthetical definitions from the registry glossary.
- No first-run tour (explicitly declined); Docent is the sole tutorial mechanism.

## Section F — Interactivity (all four types requested)

1. **Cross-filtering:** Home heatmap + correlation matrix get `on_select="rerun"`; a sector click filters the signals list, timeline, and intelligence brief; a correlation-cell click opens a pair-detail strip.
2. **Drill-down:** summary elements deep-link via `st.query_params` (e.g. `4_Models.py?commodity=WHEAT` pre-selects the instrument). Views become shareable URLs.
3. **What-if sandboxes:** Signals and Risk pages get a "What-if" expander — sliders for prior strength α, upstream damping, risk-gate toggles — re-running existing model functions **in-memory only**, badged "SANDBOX", never writing to DB or pkl files.
4. **Live-market feel:** topbar and map statuses refresh with `st.fragment(run_every=120)` partial reruns; subtle CSS pulses tied to active triggers.

## Section G — Roadmap page

`pages/16_Roadmap.py`, "Accendio Intelligence Roadmap | Alpha Phase", amber roadmap accent, built to the reference mockups. Full mockup scope (chosen):

- **Signal Lab (Alpha) panel** — wraps existing `13_Signal_Lab.py` logic (model layer stays put): data-source picker, model-type selection (Statistical / Neural Net / Market Signal / Quantum-legacy), hyperparameter sliders, backtest params, Run Backtest, Saved Research Variants. Page 13 survives until this proves out, then is retired in one clean commit.
- **Research Library (Alpha) panel** — searchable card grid of archived narrative briefs + saved historical signals with context snapshots, sourced from `forecast_log`, `trigger_events`, and the narrative engine; evolves `15_Research_Library.py` the same way.
- **Development Milestones tracker** — data-driven from `config/roadmap_milestones.json` (Model A/B Sandbox ✓ → Contextual Archive in progress → BETA RELEASE).
- **Provide Alpha Feedback** — `st.dialog` form writing to a new `alpha_feedback` table (timestamp, page, text, optional contact). The only schema addition in this design.
- The mockup's "Roadmap Functional Description" panel is delivered as this page's docent content — no bespoke mechanism.

## Section H — Feature flags & rollout

New flags: `ECOSYSTEM_UI_ENABLED` (zones, topbar dots, flow footers, map page), `DOCENT_ENABLED`, `ROADMAP_ENABLED`. `NAV_TAXONOMY_V2_ENABLED` flips on at step 2 (zones mirror the four-layer taxonomy; legacy fallback intact). Rollback anywhere = flag flip.

Rollout — seven branches, each independently shippable/revertible:

1. Zone theme core in `theme.py` (invisible; no page opts in)
2. Ecosystem registry + topbar dots + flow footer on pilot page (Macro-Market Cascade); flip `NAV_TAXONOMY_V2_ENABLED`
3. Ecosystem Map page
4. Docent on pilot pages (Home + Macro-Market Cascade) + sidebar toggle
5. Cross-filtering + drill-down on Home and Models
6. What-if sandboxes + live fragments
7. Roadmap page

## Testing

- Registry: unit tests that every page entry has a valid zone, resolvable nav path, and callable statuses (headless, no Streamlit).
- Theme: snapshot test that `apply_theme()` without zone emits today's CSS unchanged.
- Docent: test that every `docent()` panel id used in pages exists in the registry (grep-based check in CI).
- What-if sandboxes: assert no DB writes occur during override runs (transaction spy or read-only session).
- Manual visual pass per rollout step behind the flag before flipping defaults.

## Appendix — Optional reorganization (standby; each would be its own branch + flag later)

Not built now; recorded for a later decision. Nothing above depends on these:

1. **Merge Pricing + Charts → "Markets"** — same zone, heavy overlap.
2. **Merge News + Event Ribbon → "Macro Feed"** — one macro narrative surface.
3. **Split `4_Models.py` (3,606 lines) into per-model-family surfaces** — the layered-architecture rule already wants this; the zone migration is a natural moment to do it.

## Session decisions log

- Scope: B (+ appendix) · Palette: Depth Zones (B) · Flow: Map + Ribbon → refined to Map + Topbar-Integrated (B) after overwhelm feedback · Tutorial: Docent only (B) · Interactivity: all four · Roadmap: full mockup scope · Map: vertical · Approach: 1 (zone-aware theme core).
- Mockups from the session persist in `.superpowers/brainstorm/36579-1784123267/content/` (gitignored).
