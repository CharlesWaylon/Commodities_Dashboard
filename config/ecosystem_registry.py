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
